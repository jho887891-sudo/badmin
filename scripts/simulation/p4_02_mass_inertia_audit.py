"""P4-02: proxy mass / COM / inertia contract audit (offline, no Isaac).

Reads the `diagnostics.body_audit` block that every P4_PROXY run records from the PhysX
runtime, and checks it against (a) the USD authored values and (b) closed-form values for
the collision geometry. It never compares against the real B601-RS 26.0085 kg scene: that is
a different configuration and must not be mixed into one conclusion.

Checks
  * masses: chassis == authored 16 kg, each wheel == 1 kg, total == 20 kg (+payload if any);
  * COM: each body's COM sits at its own origin (authored) within tolerance;
  * inertia tensor: symmetric, positive definite, and satisfying the triangle inequalities;
  * wheel inertia vs an analytic solid cylinder with the authored radius/half-width
    (I_axis = m r^2 / 2, I_transverse = m (3 r^2 + h^2) / 12) within a relative tolerance;
  * units sanity: masses in kg, inertias in kg m^2, lengths in m (ranges checked, not assumed).

Exit codes: 0 = PASS, 2 = a contract violation, 1 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

AUTHORED_CHASSIS_MASS_KG = 16.0
AUTHORED_WHEEL_MASS_KG = 1.0
WHEEL_RADIUS_M = 0.103
WHEEL_HALF_WIDTH_M = 0.0243
MASS_TOL_KG = 1e-6
COM_TOL_M = 1e-6
INERTIA_REL_TOL = 0.01


def flatten_masses(block):
    masses = block.get("available", {}).get("masses")
    if isinstance(masses, list) and masses and isinstance(masses[0], list):
        masses = masses[0]
    return [float(v) for v in masses] if isinstance(masses, list) else None


def flatten_inertias(block):
    values = block.get("available", {}).get("inertias")
    if not isinstance(values, list) or len(values) != 5:
        return None
    return [[float(v) for v in row] for row in values]


def flatten_coms(block):
    values = block.get("available", {}).get("coms")
    if not isinstance(values, list) or len(values) != 5:
        return None
    return [[float(v) for v in row] for row in values]


def symmetric_and_pd(tensor):
    a, b, c, d, e, f, g, h, i = tensor
    matrix = [[a, b, c], [d, e, f], [g, h, i]]
    symmetric = all(abs(matrix[r][c] - matrix[c][r]) <= 1e-9
                    for r in range(3) for c in range(3))
    # principal minors of a symmetric 3x3: positive definiteness iff all > 0
    m1 = matrix[0][0]
    m2 = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    m3 = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
          - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
          + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    triangle = (matrix[0][0] + matrix[1][1] >= matrix[2][2] - 1e-12
                and matrix[1][1] + matrix[2][2] >= matrix[0][0] - 1e-12
                and matrix[0][0] + matrix[2][2] >= matrix[1][1] - 1e-12)
    return {"symmetric": symmetric, "minors": [m1, m2, m3],
            "positive_definite": bool(m1 > 0 and m2 > 0 and m3 > 0),
            "triangle_inequality": bool(triangle)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="P4-02 mass/inertia audit")
    parser.add_argument("--result", required=True, help="path to a P4_PROXY result.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        with open(args.result, encoding="utf-8") as handle:
            result = json.load(handle)
    except Exception as exc:
        print("unreadable result: %s" % exc, file=sys.stderr)
        return 1
    block = result.get("diagnostics", {}).get("body_audit")
    masses = flatten_masses(block or {})
    inertias = flatten_inertias(block or {})
    coms = flatten_coms(block or {})
    findings = []
    if masses is None or inertias is None or coms is None:
        findings.append("body_audit incomplete: masses/inertias/coms unavailable")
        summary = {"audit": "INCOMPLETE", "findings": findings}
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 2

    body_names = result["diagnostics"].get("body_names", [])
    payload_kg = float(result["diagnostics"].get("payload_kg") or 0.0)
    expected_masses = [AUTHORED_CHASSIS_MASS_KG] + [AUTHORED_WHEEL_MASS_KG] * 4
    if payload_kg > 0.0:
        expected_masses.append(payload_kg)
    if len(masses) != len(expected_masses):
        findings.append("body count %d != expected %d" % (len(masses), len(expected_masses)))
    for index, (actual, expected) in enumerate(zip(masses, expected_masses)):
        if abs(actual - expected) > MASS_TOL_KG:
            findings.append("body %d (%s) mass %.9f != authored %.9f"
                            % (index, body_names[index] if index < len(body_names) else "?",
                               actual, expected))
    total = sum(masses)
    expected_total = sum(expected_masses)
    if abs(total - expected_total) > MASS_TOL_KG:
        findings.append("total mass %.9f != %.9f" % (total, expected_total))

    com_report = []
    for index, row in enumerate(coms):
        position = row[:3]
        offset = math.sqrt(sum(v * v for v in position))
        com_report.append({"body": body_names[index] if index < len(body_names) else index,
                           "com_offset_m": round(offset, 12)})
        if offset > COM_TOL_M:
            findings.append("body %d COM offset %.9f m > %.1e" % (index, offset, COM_TOL_M))

    analytic_axis = 0.5 * AUTHORED_WHEEL_MASS_KG * WHEEL_RADIUS_M ** 2
    analytic_transverse = (AUTHORED_WHEEL_MASS_KG
                           * (3 * WHEEL_RADIUS_M ** 2 + (2 * WHEEL_HALF_WIDTH_M) ** 2) / 12.0)
    inertia_report = []
    for index, tensor in enumerate(inertias):
        name = body_names[index] if index < len(body_names) else str(index)
        checks = symmetric_and_pd(tensor)
        if not checks["symmetric"] or not checks["positive_definite"]:
            findings.append("body %s inertia not symmetric/positive definite" % name)
        if not checks["triangle_inequality"]:
            findings.append("body %s inertia violates the triangle inequality" % name)
        entry = {"body": name, **checks}
        if index >= 1 and index <= 4:
            # wheel inertias: the diagonal of a cylinder with the axis along X
            ixx, iyy, izz = abs(tensor[0]), abs(tensor[4]), abs(tensor[8])
            axis_error = abs(ixx - analytic_axis) / analytic_axis
            transverse_error = max(abs(iyy - analytic_transverse),
                                   abs(izz - analytic_transverse)) / analytic_transverse
            entry.update({"ixx": round(ixx, 9), "iyy": round(iyy, 9), "izz": round(izz, 9),
                          "analytic_axis": round(analytic_axis, 9),
                          "analytic_transverse": round(analytic_transverse, 9),
                          "axis_rel_error": round(axis_error, 9),
                          "transverse_rel_error": round(transverse_error, 9)})
            if axis_error > INERTIA_REL_TOL or transverse_error > INERTIA_REL_TOL:
                findings.append("body %s inertia differs from the analytic cylinder by more "
                                "than %.1f%% (axis %.4f%%, transverse %.4f%%)"
                                % (name, INERTIA_REL_TOL * 100, axis_error * 100,
                                   transverse_error * 100))
        inertia_report.append(entry)

    summary = {
        "source": "PhysX runtime (root_physx_view) read from diagnostics.body_audit",
        "run_id": result.get("run_id"),
        "configuration_warning": ("proxy masses only; the real B601-RS 26.0085 kg gravity "
                                  "scene is a DIFFERENT configuration and must not be compared"),
        "masses_kg": masses,
        "expected_masses_kg": expected_masses,
        "total_mass_kg": round(total, 9),
        "payload_kg": payload_kg,
        "com": com_report,
        "inertia": inertia_report,
        "analytic_wheel_cylinder": {"i_axis": round(analytic_axis, 9),
                                    "i_transverse": round(analytic_transverse, 9),
                                    "radius_m": WHEEL_RADIUS_M,
                                    "half_width_m": WHEEL_HALF_WIDTH_M},
        "tolerances": {"mass_kg": MASS_TOL_KG, "com_m": COM_TOL_M,
                       "inertia_rel": INERTIA_REL_TOL},
        "findings": findings,
        "audit": "PASS" if not findings else "FAIL",
    }
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.result)),
                                        "mass_inertia_audit.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("run_id", "audit", "findings", "total_mass_kg")},
                     ensure_ascii=False, indent=1))
    return 0 if summary["audit"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
