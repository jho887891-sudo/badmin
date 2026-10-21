"""Offline G6 comparison: determinism per implementation + cross-implementation table.

Reads only result.json files (no Isaac, no GPU) and applies the thresholds frozen in
management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md.

Usage:
    python p2_wc_g6_compare.py --run-dir /home/dgut/runs/g/G6A_r01 --run-dir ... \
        --baseline-g5 /path/to/G5_r01/result.json --out /path/to/g6_summary.json
"""

from __future__ import annotations

import argparse
import json
import os

# ---- pre-registered tolerances (frozen before the first run; never relaxed) ----
PEAK_FORCE_REL_TOL = 0.01        # two runs of the same implementation agree within 1 %
FINAL_Z_ABS_TOL_M = 1e-3
FIRST_CONTACT_STEP_TOL = 1
G5_EQUIV_PEAK_REL_TOL = 0.01     # 60-step variant A must reproduce the G5 peak within 1 %


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def tail_support(run_dir, tail_steps=30, mass_kg=20.0, gravity_z=-9.81):
    """SUPPLEMENTARY (post-hoc, NOT part of the pre-registered G6 gate set).

    Compares the mean contact force of the last 30 steps with the static weight
    m*|g|.  Reported separately so that a big landing impulse can never be sold as
    load bearing, and so that the G5L runs (which DID pre-register this gate) can be
    cross-checked against the G6 variants.
    """
    path = os.path.join(run_dir, "raw_poses.jsonl")
    if not os.path.isfile(path):
        return None
    frames = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    tail = frames[-min(tail_steps, len(frames)):]
    chassis = [frame.get("max_contact_force_n") for frame in tail
               if frame.get("max_contact_force_n") is not None]
    per_wheel = {}
    for name in ("wheel1", "wheel2", "wheel3", "wheel4"):
        values = [frame.get("contact_by_sensor_n", {}).get(name) for frame in tail]
        values = [value for value in values if value is not None]
        per_wheel[name] = ("UNKNOWN" if not values else round(sum(values) / len(values), 6))
    expected = mass_kg * abs(gravity_z)
    mean_chassis = (sum(chassis) / len(chassis)) if chassis else None
    return {
        "label": "SUPPLEMENTARY (post-hoc, not a pre-registered G6 gate)",
        "tail_steps": min(tail_steps, len(frames)),
        "tail_mean_chassis_contact_n": (round(mean_chassis, 6)
                                       if mean_chassis is not None else None),
        "expected_static_weight_n": round(expected, 6),
        "relative_error": (round(abs(mean_chassis - expected) / expected, 9)
                           if mean_chassis is not None else None),
        "tail_mean_per_wheel_contact_n": per_wheel,
    }


def rel_delta(a, b):
    if a is None or b is None:
        return None
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def main(argv=None):
    parser = argparse.ArgumentParser(description="G6 offline comparison")
    parser.add_argument("--run-dir", action="append", required=True,
                        help="directory containing result.json; repeat per run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-g5", default=None,
                        help="G5_r01 result.json for the 60-step equivalence control")
    parser.add_argument("--equivalence-run", default=None,
                        help="run_id of the 60-step variant-A equivalence run")
    parser.add_argument("--write-notes", action="store_true",
                        help="write notes.md into every run directory (positives / negatives / unknowns)")
    args = parser.parse_args(argv)

    runs = {}
    for run_dir in args.run_dir:
        result = load(os.path.join(run_dir, "result.json"))
        runs[result["run_id"]] = {"result": result, "run_dir": os.path.abspath(run_dir)}

    by_impl = {}
    for run_id, entry in runs.items():
        by_impl.setdefault(entry["result"]["impl"], []).append(run_id)

    determinism = {}
    for impl, run_ids in by_impl.items():
        run_ids = sorted(run_ids)
        if len(run_ids) < 2:
            determinism[impl] = {"status": "UNKNOWN", "reason": "fewer than 2 run_ids"}
            continue
        first, second = (runs[run_ids[0]]["result"], runs[run_ids[1]]["result"])
        peak_rel = rel_delta(first["metrics"]["peak_contact_force_n"],
                             second["metrics"]["peak_contact_force_n"])
        final_z = abs(first["metrics"]["final_chassis_z_m"] -
                      second["metrics"]["final_chassis_z_m"])
        step_gap = abs((first["metrics"]["first_contact_step"] or -1) -
                       (second["metrics"]["first_contact_step"] or -1))
        checks = {
            "peak_force_within_rel_tol": (peak_rel is not None
                                         and peak_rel <= PEAK_FORCE_REL_TOL),
            "final_z_within_abs_tol": final_z <= FINAL_Z_ABS_TOL_M,
            "first_contact_step_within_tol": step_gap <= FIRST_CONTACT_STEP_TOL,
        }
        determinism[impl] = {
            "run_ids": run_ids,
            "peak_force_rel_delta": (round(peak_rel, 9) if peak_rel is not None else None),
            "final_z_abs_delta_m": round(final_z, 12),
            "first_contact_step_gap": step_gap,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }

    table = {}
    for run_id, entry in sorted(runs.items()):
        result = entry["result"]
        metrics = result["metrics"]
        per_wheel = result.get("per_wheel_peak_contact_n", {})
        per_wheel_known = bool(per_wheel) and all(v is not None for v in per_wheel.values())
        table[run_id] = {
            "impl": result["impl"],
            "gate_result": result["gate_result"],
            "gates": result["gates"],
            "peak_contact_force_n": metrics.get("peak_contact_force_n"),
            "first_contact_step": metrics.get("first_contact_step"),
            "final_chassis_z_m": metrics.get("final_chassis_z_m"),
            "tail_speed_max_m_s": metrics.get("tail_speed_max_m_s"),
            "tail_max_penetration_m": metrics.get("tail_max_penetration_m"),
            "max_tilt_rad": metrics.get("max_tilt_rad"),
            "per_wheel_contact": "MEASURED" if per_wheel_known else "UNKNOWN",
            "per_wheel_peak_contact_n": per_wheel,
            "contact_api": result["diagnostics"].get("contact_api"),
        }

    verdicts = {}
    for impl in sorted(by_impl):
        run_ids = sorted(by_impl[impl])
        results = [runs[run_id]["result"] for run_id in run_ids]
        gates_pass = all(r["gate_result"] == "PASS" for r in results)
        contact = all(r["gates"].get("contact_detected") for r in results)
        det = determinism.get(impl, {}).get("status", "UNKNOWN")
        if not contact:
            verdict = "FAIL"
        elif gates_pass and det == "PASS":
            verdict = "PASS"
        elif contact:
            verdict = "PARTIAL"
        else:
            verdict = "UNKNOWN"
        failed = sorted({name for r in results for name, ok in r["gates"].items() if not ok})
        verdicts[impl] = {
            "run_ids": run_ids,
            "verdict": verdict,
            "all_runs_gate_pass": gates_pass,
            "contact_detected_all_runs": contact,
            "determinism": det,
            "failing_gates": failed,
            "per_wheel_contact_measured": all(
                table[run_id]["per_wheel_contact"] == "MEASURED" for run_id in run_ids),
        }

    equivalence = None
    if args.baseline_g5 and args.equivalence_run:
        baseline = load(args.baseline_g5)
        candidate = runs[args.equivalence_run]["result"]
        # The historical G5 result stored the peak under diagnostics, not metrics.
        base_peak = baseline.get("metrics", {}).get("peak_contact_force_n")
        if base_peak is None:
            base_peak = baseline.get("diagnostics", {}).get("max_contact_force_settle_n")
        cand_peak = candidate["metrics"]["peak_contact_force_n"]
        delta = rel_delta(base_peak, cand_peak)
        equivalence = {
            "baseline_run": baseline["run_id"],
            "baseline_peak_contact_force_n": base_peak,
            "equivalence_run": candidate["run_id"],
            "equivalence_peak_contact_force_n": cand_peak,
            "rel_delta": (round(delta, 9) if delta is not None else None),
            "tolerance": G5_EQUIV_PEAK_REL_TOL,
            "status": ("PASS" if delta is not None and delta <= G5_EQUIV_PEAK_REL_TOL
                       else "FAIL"),
        }

    summary = {
        "preregistration": "management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md",
        "tolerances": {
            "peak_force_rel": PEAK_FORCE_REL_TOL,
            "final_z_abs_m": FINAL_Z_ABS_TOL_M,
            "first_contact_step": FIRST_CONTACT_STEP_TOL,
            "g5_equivalence_peak_rel": G5_EQUIV_PEAK_REL_TOL,
        },
        "table": table,
        "supplementary_support": {run_id: tail_support(runs[run_id]["run_dir"])
                                  for run_id in sorted(runs)},
        "determinism": determinism,
        "verdicts": verdicts,
        "g5_equivalence": equivalence,
        "overall": ("PASS" if verdicts and all(v["verdict"] == "PASS"
                                                for v in verdicts.values())
                    else "PARTIAL"),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    if args.write_notes:
        for run_id, entry in sorted(runs.items()):
            result = entry["result"]
            row = table[run_id]
            impl_det = determinism.get(result["impl"], {})
            failing = sorted(name for name, ok in result["gates"].items() if not ok)
            lines = [
                "# %s · %s" % (run_id, result["impl"]),
                "",
                "- gate_result: **%s** (mode=%s)" % (result["gate_result"], result["gate_mode"]),
                "- peak contact: %s N, first contact step %s" % (
                    row["peak_contact_force_n"], row["first_contact_step"]),
                "- final chassis z: %s m, tail speed max: %s m/s, tail z span: %s m" % (
                    row["final_chassis_z_m"], row["tail_speed_max_m_s"],
                    result["metrics"].get("tail_z_span_m")),
                "- max tilt: %s rad, tail max penetration: %s m" % (
                    row["max_tilt_rad"], row["tail_max_penetration_m"]),
                "- contact api: %s; per-wheel contact: %s" % (
                    row["contact_api"], row["per_wheel_contact"]),
                "",
                "## Positives",
            ]
            lines += ["- gate %s = true" % name for name, ok in result["gates"].items() if ok] or ["- (none)"]
            lines += ["", "## Negatives / failing gates"]
            lines += (["- gate %s = false" % name for name in failing] or ["- none"])
            lines += ["", "## Unknowns"]
            if row["per_wheel_contact"] != "MEASURED":
                lines.append("- per-wheel contact is UNKNOWN for this implementation: only an "
                             "explicit rigid body can host a ContactSensor, so A/C report an "
                             "aggregate chassis value only (recorded as null, not 0 N).")
            lines.append("- cooked PhysX geometry: UNKNOWN (no exposed query API in this build).")
            lines.append("- determinism (same impl, two run_ids): %s" %
                         (impl_det.get("status", "UNKNOWN")))
            lines.append("")
            with open(os.path.join(entry["run_dir"], "notes.md"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("verdicts", "determinism", "g5_equivalence")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
