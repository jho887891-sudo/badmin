"""Offline aggregator for the P4-08 single-factor friction/payload/wheel-speed matrix.

Reads every P4-08 run directory (no Isaac, no GPU), checks that each pair of runs differs in
EXACTLY ONE of {friction, payload, wheel speed}, audits the negative/positive controls, and
emits a machine-readable summary plus a Markdown table. All numbers come from the run
directories; nothing is hand-copied into the summary.

Exit codes: 0 = matrix consistent, 2 = a single-factor rule or a control was violated,
1 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GROUPS = {
    "friction": ["P4f_low", "P4f_base", "P4f_high"],
    "payload": ["P4L_0", "P4L_10"],
    "wheel_speed": ["P4S_1", "P4S_4"],
}
CONTROLS = {"hold_still": "P4N_still", "reverse": "P4R_pos"}


def load(result_path):
    with open(result_path, encoding="utf-8") as handle:
        return json.load(handle)


def summarise(result):
    metrics = result.get("metrics", {})
    diagnostics = result.get("diagnostics", {})
    return {
        "run_id": result["run_id"],
        "profile": result["profile"],
        "gate_result": result["gate_result"],
        "failing_gates": sorted(name for name, ok in result.get("gates", {}).items() if not ok),
        "displacement_m": metrics.get("chassis_displacement_m"),
        "yaw_change_rad": metrics.get("yaw_change_rad"),
        "tail_wheel_omega_rad_s": metrics.get("tail_wheel_omega_rad_s"),
        "wheel_targets_rad_s": metrics.get("wheel_targets_rad_s_per_wheel"),
        "wheel_speed_relative_error": metrics.get("wheel_speed_relative_error"),
        "measured_rolling_radius_m": metrics.get("measured_rolling_radius_m"),
        "measured_rolling_radius_rel_error": metrics.get("measured_rolling_radius_rel_error"),
        "peak_contact_force_n": metrics.get("peak_contact_force_n"),
        "friction": diagnostics.get("friction_coefficient"),
        "payload_kg": diagnostics.get("payload_kg"),
        "root_writes": diagnostics.get("root_writes"),
        "negative_control": diagnostics.get("negative_control"),
    }


def family_of(run_id):
    for control, prefix in CONTROLS.items():
        if run_id.startswith(prefix):
            return ("control", control)
    for group, prefixes in GROUPS.items():
        for prefix in prefixes:
            if run_id.startswith(prefix):
                return (group, prefix)
    return (None, None)


def single_factor_audit(entries):
    """Group members must differ in exactly one knob from the reference member."""
    problems = []
    by_prefix = {}
    for entry in entries:
        family, prefix = family_of(entry["run_id"])
        if family in (None, "control"):
            continue
        by_prefix.setdefault(prefix, []).append(entry)
    for group, prefixes in GROUPS.items():
        reference = None
        for prefix in prefixes:
            runs = by_prefix.get(prefix, [])
            if not runs:
                problems.append("%s: no run for %s" % (group, prefix))
                continue
            values = {(r["friction"], r["payload_kg"], r["profile"],
                       _wheel_speed(r)) for r in runs}
            if len(values) > 1:
                problems.append("%s/%s: runs differ in more than the group factor: %s"
                                % (group, prefix, sorted(values)))
            if reference is None:
                reference = runs[0]
                continue
            changed = _changed_knobs(reference, runs[0])
            expected = {"friction": {"friction"}, "payload": {"payload"},
                        "wheel_speed": {"wheel_speed"}}[group]
            if changed != expected:
                problems.append("%s/%s vs %s: changed %s but only %s may change"
                                % (group, prefix, reference["run_id"], sorted(changed),
                                   sorted(expected)))
    return problems


def _wheel_speed(entry):
    """The COMMANDED wheel speed magnitude is the knob.

    Using the MEASURED tail omega here was wrong: measured speed is an outcome of friction
    and load, so the single-factor audit flagged friction/payload groups as if they had also
    changed the speed. The commanded target is what distinguishes the runs.
    """
    targets = entry.get("wheel_targets_rad_s")
    if targets:
        values = [abs(float(v)) for v in targets if v is not None]
        if values:
            return round(max(values), 3)
    omegas = entry.get("tail_wheel_omega_rad_s") or {}
    values = [abs(v) for v in omegas.values() if v is not None]
    return round(max(values), 3) if values else None


def _changed_knobs(a, b):
    changed = set()
    if a["friction"] != b["friction"]:
        changed.add("friction")
    if a["payload_kg"] != b["payload_kg"]:
        changed.add("payload")
    if a["profile"] != b["profile"]:
        changed.add("profile")
    if _wheel_speed(a) != _wheel_speed(b):
        changed.add("wheel_speed")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="P4-08 matrix aggregator")
    parser.add_argument("--runs-root", required=True,
                        help="directory containing P4f_*/P4L_*/P4S_*/P4N_*/P4R_* run dirs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    root = os.path.abspath(args.runs_root)
    if not os.path.isdir(root):
        print("runs root not found: %s" % root, file=sys.stderr)
        return 1
    entries = []
    for name in sorted(os.listdir(root)):
        result_path = os.path.join(root, name, "result.json")
        if not os.path.isfile(result_path):
            continue
        if not (name.startswith("P4f_") or name.startswith("P4L_")
                or name.startswith("P4S_") or name.startswith("P4N_")
                or name.startswith("P4R_")):
            continue
        try:
            entries.append(summarise(load(result_path)))
        except Exception as exc:
            print("unreadable %s: %s" % (result_path, exc), file=sys.stderr)
            return 1
    if not entries:
        print("no P4-08 runs found under %s" % root, file=sys.stderr)
        return 1

    problems = single_factor_audit(entries)
    # control audit: a no-drive run must pass, a reverse run must displace the other way
    for entry in entries:
        if entry["run_id"].startswith(CONTROLS["hold_still"]):
            if entry["failing_gates"]:
                problems.append("%s: no-drive control failed %s"
                               % (entry["run_id"], entry["failing_gates"]))
        if entry["run_id"].startswith(CONTROLS["reverse"]):
            displacement = entry["displacement_m"] or [0, 0, 0]
            if displacement[1] <= 0:
                problems.append("%s: reverse control did not move +Y (%s)"
                               % (entry["run_id"], displacement))
        if entry["root_writes"] not in (0, None):
            problems.append("%s: root_writes=%s (drive must not write the root)"
                           % (entry["run_id"], entry["root_writes"]))

    radii = {}
    for entry in entries:
        values = entry.get("measured_rolling_radius_m") or {}
        known = [v for v in values.values() if v is not None]
        if known:
            radii[entry["run_id"]] = {"mean": round(sum(known) / len(known), 6),
                                      "max_rel_error": max(
                                          entry.get("measured_rolling_radius_rel_error") or [0])}

    summary = {
        "preregistration": "management/morph_one_rs/PHASE_P4_08_PREREG.md",
        "runs_root": root,
        "run_count": len(entries),
        "runs": entries,
        "measured_rolling_radius": radii,
        "single_factor_problems": problems,
        "controls": {name: [r["run_id"] for r in entries
                            if r["run_id"].startswith(prefix)]
                     for name, prefix in CONTROLS.items()},
        "verdict": "CONSISTENT" if not problems else "VIOLATION",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"run_count": len(entries), "verdict": summary["verdict"],
                      "problems": problems}, ensure_ascii=False, indent=1))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
