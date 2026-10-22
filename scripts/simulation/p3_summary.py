#!/usr/bin/env python3
"""P3 evidence summariser: one authoritative table over every committed P3_* run.

Why this exists
---------------
The P3 gate was revised mid-flight (v1.0 -> v1.1).  v1.0 judged translation by
hypot(x, y) and therefore had NO Z gate at all; v1.1 replaced that with a signed
3D projection (achieved_along_command / |command| >= 0.9) and made
set_joint_targets failures fail-closed.  The two generations are not comparable,
so nobody may average them into one number.  This script reads every run
directory and produces a machine-generated summary instead of hand-copied
numbers in a report.

Generation classification (evidence-based, never by run name)
-------------------------------------------------------------
v1.1-exclusive artefacts (amendment PHASE_P3_GATE_V1_1_AMENDMENT.md section 4):

* metrics.directed_translation - the signed 3D projection, incl. best_frame
* diagnostics.negative_control (non-null) - the freeze-axis switch

The quat-order probe (diagnostics.quat_order_for_write_root_pose with two
probe_measured_yaw_rad entries) is NOT v1.1-exclusive: it is already present in
the v1.0-era P3_*_r09 runs, which still carry the 0.208 m root-pose write defect
(t0_pose_write_error_m == 0.208394530817).  Using it as the sole v1.1 marker
would silently merge v1.0 and v1.1 evidence.  It is therefore recorded as
quat_probe_two_entries and, when it is the only v1.x artefact, flagged as
ambiguous_legacy_quat_probe.  --spec-quat-rule reproduces the literal
"quat probe => v1.1" rule for audit comparison.

Exit codes
----------
0  summary written, every negative control failed as designed, every run has both
   result.json and gate_runner_summary.json
1  evidence integrity failure: missing result.json, missing
   gate_runner_summary.json, or a negative control that PASSED (that is a
   harness failure, not an experiment result)
2  bad invocation (no run directories found)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GENERATION_V10 = "v1.0"
GENERATION_V11 = "v1.1"
GENERATIONS = (GENERATION_V10, GENERATION_V11)

#: Each generation needs this many independent PASSing runs per profile.
MIN_INDEPENDENT_PASS = 2

#: v1.1-exclusive artefacts; absence of both means the run is v1.0-bound.
V11_MARKERS = ("metrics.directed_translation", "diagnostics.negative_control")

#: Motion gate that a frozen axis must break.
FROZEN_AXIS_GATE = {
    "x": "translation_actually_measured",
    "y": "translation_actually_measured",
    "z": "translation_actually_measured",
    "yaw": "yaw_actually_measured",
}

BT = chr(96)  # markdown code span

DIAGNOSTIC_DRIVE_NOTICE = (
    "P3 drive is a DIAGNOSTIC root-pose teleport with the root velocity zeroed every step: "
    "peak_lin_speed_m_s / peak_ang_speed_rad_s / peak_jerk_m_s3 are zero (or teleport "
    "artefacts) BY CONSTRUCTION and are NOT dynamics or stability evidence. "
    "P3 PASS is never P4 wheel-drive evidence."
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATES_DIR = ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "morph_one" / "p2_wc_gates"
DEFAULT_JSON_OUT = DEFAULT_GATES_DIR / "p3_summary.json"
DEFAULT_MD_OUT = DEFAULT_GATES_DIR / "p3_summary.md"


class EvidenceError(Exception):
    """An artefact needed to classify a run is missing or unreadable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise EvidenceError("missing_json", f"missing {path.name} in {path.parent.name}")
    except json.JSONDecodeError as exc:
        raise EvidenceError("bad_json", f"{path} is not valid JSON: {exc}")
    if not isinstance(value, dict):
        raise EvidenceError("bad_json", f"{path} is not a JSON object")
    return value


def _sha256(path: Path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _count_lines(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return None


def _has_marker(metrics: dict, diagnostics: dict, marker: str) -> bool:
    if marker == "metrics.directed_translation":
        return isinstance(metrics.get("directed_translation"), dict)
    if marker == "diagnostics.negative_control":
        return bool(diagnostics.get("negative_control"))
    raise AssertionError(f"unknown marker {marker}")


def classify_generation(result: dict, spec_quat_rule: bool = False) -> dict:
    """Classify one result.json payload as v1.0 or v1.1.

    Returns generation / basis / quat_probe_two_entries /
    ambiguous_legacy_quat_probe.  Classification never looks at the run name:
    only at artefacts the two gate versions actually write differently.
    """
    metrics = result.get("metrics") or {}
    diagnostics = result.get("diagnostics") or {}
    probe = (diagnostics.get("quat_order_for_write_root_pose") or {}).get(
        "probe_measured_yaw_rad")
    quat_two = isinstance(probe, dict) and len(probe) >= 2

    # An EXPLICIT tag wins over every heuristic.  The heuristics below look at artefact
    # shape, and the yaw profile legitimately has no directed_translation block, so
    # P3_yaw_r13/r14 were mis-classified as v1.0 and v1.1/yaw looked uncovered.
    explicit = str(result.get("gate_version") or "").strip().lower()
    if explicit in (GENERATION_V10, GENERATION_V11):
        return {
            "generation": explicit,
            "basis": ["gate_version(%s)" % explicit],
            "quat_probe_two_entries": quat_two,
            "ambiguous_legacy_quat_probe": False,
        }

    basis = [marker for marker in V11_MARKERS if _has_marker(metrics, diagnostics, marker)]
    if spec_quat_rule and quat_two:
        basis.append("diagnostics.quat_order_for_write_root_pose(2 entries)")

    generation = GENERATION_V11 if basis else GENERATION_V10
    ambiguous = bool(quat_two and generation == GENERATION_V10 and not spec_quat_rule)
    return {
        "generation": generation,
        "basis": basis,
        "quat_probe_two_entries": quat_two,
        "ambiguous_legacy_quat_probe": ambiguous,
    }


def load_run(run_dir: Path, spec_quat_rule: bool = False) -> dict:
    """Read one run directory.  A missing result.json raises EvidenceError."""
    result = _read_json(run_dir / "result.json")
    try:
        manifest = _read_json(run_dir / "manifest.json")
    except EvidenceError as exc:
        if exc.code == "missing_json":
            manifest = {}
        else:
            raise
    try:
        runner = _read_json(run_dir / "gate_runner_summary.json")
    except EvidenceError:
        runner = None  # recorded below, and the run becomes an integrity failure

    metrics = result.get("metrics") or {}
    diagnostics = result.get("diagnostics") or {}
    gates = result.get("gates") or {}
    negative = diagnostics.get("negative_control")
    freeze_axis = manifest.get("freeze_axis")
    if freeze_axis is None and isinstance(negative, dict):
        freeze_axis = negative.get("frozen_axis")
    if freeze_axis is None:
        freeze_axis = "none"

    classification = classify_generation(result, spec_quat_rule=spec_quat_rule)
    directed = metrics.get("directed_translation")

    return {
        "run_id": result.get("run_id") or run_dir.name,
        "run_dir": run_dir.name,
        "profile": result.get("profile") or manifest.get("profile"),
        "scene_variant": result.get("scene_variant"),
        "drive_mode": result.get("drive_mode") or manifest.get("drive_mode"),
        "gate_mode": (f"negative_control(freeze_axis={freeze_axis})"
                      if isinstance(negative, dict) else "positive"),
        "freeze_axis": freeze_axis,
        "gate_result": result.get("gate_result"),
        "outer_exit": (runner or {}).get("outer_exit"),
        "child_returncode": (runner or {}).get("child_returncode"),
        "child_ok": (runner or {}).get("child_ok"),
        "gates_ok": (runner or {}).get("gates_ok"),
        "json_gate": (runner or {}).get("json_gate"),
        "timed_out": (runner or {}).get("timed_out"),
        "elapsed_s": (runner or {}).get("elapsed_s"),
        "queued_command": (runner or {}).get("command"),
        "has_gate_runner_summary": runner is not None,
        "gates": {name: bool(value) for name, value in sorted(gates.items())},
        "gates_false": sorted(name for name, value in gates.items() if not value),
        "generation": classification["generation"],
        "generation_basis": classification["basis"],
        "quat_probe_two_entries": classification["quat_probe_two_entries"],
        "ambiguous_legacy_quat_probe": classification["ambiguous_legacy_quat_probe"],
        "metrics": {
            key: metrics.get(key)
            for key in (
                "realized_translation_max_m",
                "realized_yaw_max_rad",
                "tracking_position_error_max_m",
                "mount_position_deviation_max_m",
                "mount_rotation_deviation_max_rad",
                "joint_deviation_max_rad",
                "peak_lin_speed_m_s",
                "peak_ang_speed_rad_s",
                "peak_jerk_m_s3",
                "arm_peak_contact_n",
                "steps",
            )
        },
        "command_amplitude": metrics.get("command_amplitude"),
        "directed_translation": None if not isinstance(directed, dict) else {
            key: directed.get(key)
            for key in ("commanded_displacement_m", "achieved_along_command_max_m",
                        "achieved_along_command_final_m", "best_frame", "achieved_ratio",
                        "final_ratio", "cross_track_m", "pass")
        },
        "negative_control": negative,
        "root_to_chassis_offset_m": diagnostics.get("root_to_chassis_offset_m"),
        "root_to_chassis_offset_root_frame_m": diagnostics.get(
            "root_to_chassis_offset_root_frame_m"),
        "root_pose_is_offset_compensated": diagnostics.get("root_pose_is_offset_compensated"),
        "t0_pose_write_error_m": diagnostics.get("t0_pose_write_error_m"),
        "quat_order_value": (diagnostics.get("quat_order_for_write_root_pose") or {}).get("value"),
        "raw_poses_frames": _count_lines(run_dir / "raw_poses.jsonl"),
        "raw_poses_sha256": _sha256(run_dir / "raw_poses.jsonl"),
        "thresholds": result.get("thresholds"),
    }


def _metric_values(runs: list, key: str) -> list:
    values = []
    for run in runs:
        value = (run.get("metrics") or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def build_summary(gates_dir: Path, extra_dirs=(), *, spec_quat_rule: bool = False) -> dict:
    """Read every P3_* run directory under the given roots."""
    roots = [Path(gates_dir), *[Path(p) for p in extra_dirs]]
    seen = {}
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("P3_") and child.name not in seen:
                seen[child.name] = child

    runs = []
    errors = []
    for name in sorted(seen):
        run_dir = seen[name]
        try:
            runs.append(load_run(run_dir, spec_quat_rule=spec_quat_rule))
        except EvidenceError as exc:
            errors.append({"run_id": name, "run_dir": name, "code": exc.code,
                           "message": exc.message})

    runs.sort(key=lambda run: (run["generation"], str(run["profile"]), run["run_id"]))

    # --- integrity: negative controls must fail, and be observed failing ------
    vacuous_control_warnings = []
    for run in runs:
        if not run["has_gate_runner_summary"]:
            errors.append({"run_id": run["run_id"], "run_dir": run["run_dir"],
                           "code": "missing_gate_runner_summary",
                           "message": f"{run['run_dir']}: gate_runner_summary.json is missing"})
        if not isinstance(run.get("negative_control"), dict):
            continue
        axis = (run["negative_control"].get("frozen_axis") or "").lower()
        expected_gate = FROZEN_AXIS_GATE.get(axis)
        # Resolve vacuity from the run's own command amplitude, not only from the recorded
        # flag: P3_neg_frozen_y_r01 predates the flag and froze Y on a command whose Y was 0.
        amplitude = (run.get("command_amplitude") or {})
        key = {"x": "x_m", "y": "y_m", "z": "z_m", "yaw": "yaw_rad"}.get(axis)
        component = amplitude.get(key) if key else None
        vacuous = (run["negative_control"].get("vacuous") is True
                   or (component is not None and abs(float(component)) < 1e-9))
        if vacuous:
            # The frozen command component was already zero, so the freeze changed nothing
            # and the run cannot serve as a negative control.  P3_neg_frozen_y_r01 (pre-guard
            # evidence) therefore PASSED and P3_neg_frozen_y_r02 (post-guard) was forced to
            # FAIL.  Neither is a harness defect, so this is a warning, not a hard error.
            vacuous_control_warnings.append({
                "run_id": run["run_id"], "run_dir": run["run_dir"],
                "code": "vacuous_negative_control",
                "message": (f"{run['run_dir']} froze '{axis}' but that command component was "
                            f"zero ({component!r}); gate_result={run['gate_result']!r}. "
                            f"The run proves nothing."),
            })
            continue
        reasons = []
        if run["gate_result"] != "FAIL":
            reasons.append(f"gate_result={run['gate_result']!r} (expected FAIL)")
        if run["outer_exit"] != 1:
            reasons.append(f"outer_exit={run['outer_exit']!r} (expected 1)")
        if expected_gate is None:
            reasons.append(f"unknown frozen_axis={axis!r}: no motion gate can be attributed")
        elif run["gates"].get(expected_gate) is not False:
            reasons.append(f"{expected_gate}={run['gates'].get(expected_gate)!r} (expected False)")
        if reasons:
            errors.append({
                "run_id": run["run_id"], "run_dir": run["run_dir"],
                "code": "negative_control_passed",
                "message": f"{run['run_dir']} negative control did not fail: " + "; ".join(reasons),
            })

    # --- duplicate evidence (a copied run is not an independent run) ----------
    content_owner = {}
    duplicates = []
    for run in runs:
        digest = run["raw_poses_sha256"]
        if digest is None:
            continue
        if digest in content_owner:
            duplicates.append({"run_id": run["run_id"], "duplicate_of": content_owner[digest]})
            run["raw_poses_duplicate_of"] = content_owner[digest]
        else:
            content_owner[digest] = run["run_id"]

    # --- coverage audit per generation ---------------------------------------
    profiles = sorted({str(run["profile"]) for run in runs if run.get("profile")})
    coverage = {}
    for generation in GENERATIONS:
        per_profile = {}
        for profile in profiles:
            positive = [run for run in runs
                        if run["generation"] == generation
                        and run["profile"] == profile
                        and run["gate_mode"] == "positive"]
            passes = [run for run in positive if run["gate_result"] == "PASS"
                      and run["outer_exit"] == 0]
            independent = sorted({run.get("raw_poses_duplicate_of") or run["run_id"]
                                  for run in passes})
            per_profile[profile] = {
                "runs": [run["run_id"] for run in positive],
                "passes": [run["run_id"] for run in passes],
                "fails": [run["run_id"] for run in positive if run not in passes],
                "independent_pass": len(independent),
                "independent_pass_ids": independent,
                "required": MIN_INDEPENDENT_PASS,
                "gap": len(independent) < MIN_INDEPENDENT_PASS,
            }
        coverage[generation] = per_profile

    gaps = [{"generation": generation, "profile": profile,
             "independent_pass": data["independent_pass"],
             "required": data["required"], "passes": data["passes"],
             "runs_seen": data["runs"]}
            for generation, per_profile in coverage.items()
            for profile, data in per_profile.items() if data["gap"]]

    # --- per-generation aggregates (never pooled across generations) ---------
    aggregates = {}
    for generation in GENERATIONS:
        subset = [run for run in runs if run["generation"] == generation]
        directed = [run["directed_translation"] for run in subset
                    if isinstance(run.get("directed_translation"), dict)]
        ratios = [float(d["achieved_ratio"]) for d in directed
                  if isinstance(d.get("achieved_ratio"), (int, float))]
        aggregates[generation] = {
            "runs": len(subset),
            "pass": sum(1 for run in subset if run["gate_result"] == "PASS"),
            "fail": sum(1 for run in subset if run["gate_result"] == "FAIL"),
            "run_ids": [run["run_id"] for run in subset],
            "gates_defined": sorted({name for run in subset for name in run["gates"]}),
            "directed_runs": len(directed),
            "directed_achieved_ratio_min": min(ratios) if ratios else None,
            "mount_position_deviation_max_m": max(
                _metric_values(subset, "mount_position_deviation_max_m"), default=None),
            "mount_rotation_deviation_max_rad": max(
                _metric_values(subset, "mount_rotation_deviation_max_rad"), default=None),
            "peak_jerk_m_s3_max": max(_metric_values(subset, "peak_jerk_m_s3"), default=None),
            "peak_lin_speed_m_s_max": max(_metric_values(subset, "peak_lin_speed_m_s"),
                                          default=None),
            "arm_peak_contact_n_max": max(_metric_values(subset, "arm_peak_contact_n"),
                                          default=None),
        }

    # --- diagnostics: root<->chassis offset and mount drift ------------------
    offsets = sorted({run["root_to_chassis_offset_m"] for run in runs
                      if isinstance(run.get("root_to_chassis_offset_m"), (int, float))})
    frames_checked = [run for run in runs if isinstance(run.get("raw_poses_frames"), int)]
    diag = {
        "root_to_chassis_offset_m_values": offsets,
        "root_to_chassis_offset_m_spread": (offsets[-1] - offsets[0]) if offsets else None,
        "runs_with_offset_diagnostic": sum(
            1 for run in runs if isinstance(run.get("root_to_chassis_offset_m"), (int, float))),
        "runs_offset_compensated": sum(
            1 for run in runs if run.get("root_pose_is_offset_compensated") is True),
        "mount_position_deviation_max_m": max(
            _metric_values(runs, "mount_position_deviation_max_m"), default=None),
        "mount_rotation_deviation_max_rad": max(
            _metric_values(runs, "mount_rotation_deviation_max_rad"), default=None),
        "mount_drift_by_generation": {
            generation: {
                "mount_position_deviation_max_m": aggregates[generation][
                    "mount_position_deviation_max_m"],
                "mount_rotation_deviation_max_rad": aggregates[generation][
                    "mount_rotation_deviation_max_rad"],
            } for generation in GENERATIONS
        },
        "raw_poses_frames_expected": sorted({
            int((run["metrics"] or {}).get("steps") or 0) + 1
            for run in frames_checked if (run["metrics"] or {}).get("steps")
        }),
        "raw_poses_frames_observed": sorted({run["raw_poses_frames"] for run in frames_checked}),
        "peak_jerk_m_s3_max": max(_metric_values(runs, "peak_jerk_m_s3"), default=None),
        "duplicate_evidence": duplicates,
    }

    warnings = [DIAGNOSTIC_DRIVE_NOTICE] + vacuous_control_warnings
    for gap in gaps:
        warnings.append(
            f"COVERAGE GAP: {gap['profile']} has {gap['independent_pass']} independent "
            f"PASS under {gap['generation']} (need {gap['required']})")
    for duplicate in duplicates:
        warnings.append(
            f"DUPLICATE EVIDENCE: {duplicate['run_id']} has the same raw_poses.jsonl as "
            f"{duplicate['duplicate_of']} - not an independent run")
    if diag["root_to_chassis_offset_m_spread"] not in (None, 0, 0.0):
        warnings.append(
            "OFFSET DRIFT: root_to_chassis_offset_m differs across runs: "
            f"{diag['root_to_chassis_offset_m_values']}")
    for run in runs:
        if run.get("ambiguous_legacy_quat_probe"):
            warnings.append(
                f"AMBIGUOUS: {run['run_id']} has a two-entry quat probe but no v1.1 marker; "
                "bound to v1.0 per amendment section 4")
    if diag["raw_poses_frames_observed"] and diag["raw_poses_frames_expected"]:
        if diag["raw_poses_frames_observed"] != diag["raw_poses_frames_expected"]:
            warnings.append(
                "FRAME COUNT: observed raw_poses frames "
                f"{diag['raw_poses_frames_observed']} do not match steps+1 "
                f"{diag['raw_poses_frames_expected']}")

    return {
        "schema": "p3_evidence_summary/v1",
        "generated_by": "scripts/simulation/p3_summary.py",
        "gates_dir": Path(gates_dir).name,
        "extra_gates_dirs": [Path(p).name for p in extra_dirs],
        "spec_quat_rule": spec_quat_rule,
        "classification_rule": {
            "v1.1_markers": list(V11_MARKERS),
            "v1.0": "absence of every v1.1 marker",
            "non_exclusive": ["diagnostics.quat_order_for_write_root_pose(2 entries)"],
            "note": ("quat probe also appears in v1.0-era P3_*_r09 runs "
                     "(t0_pose_write_error_m 0.208 m) and therefore cannot separate "
                     "generations"),
        },
        "min_independent_pass": MIN_INDEPENDENT_PASS,
        "caveats": [DIAGNOSTIC_DRIVE_NOTICE],
        "totals": {
            "runs": len(runs),
            "errors": len(errors),
            "by_generation": {generation: aggregates[generation]["runs"]
                              for generation in GENERATIONS},
        },
        "aggregates": aggregates,
        "coverage": coverage,
        "coverage_gaps": gaps,
        "negative_controls": [
            {
                "run_id": run["run_id"],
                "profile": run["profile"],
                "frozen_axis": run["freeze_axis"],
                "gate_result": run["gate_result"],
                "outer_exit": run["outer_exit"],
                "child_returncode": run["child_returncode"],
                "gates_false": run["gates_false"],
                "achieved_ratio": (run["directed_translation"] or {}).get("achieved_ratio"),
                "vacuous": any(w["run_id"] == run["run_id"]
                                    and w["code"] == "vacuous_negative_control"
                                    for w in vacuous_control_warnings),
                # verified means OBSERVED FAILING, not merely "no error was raised": a
                # vacuous control passes without an error and must never look verified.
                "verified": bool(
                    run["gate_result"] == "FAIL"
                    and run["outer_exit"] == 1
                    and (FROZEN_AXIS_GATE.get((run["freeze_axis"] or "").lower()) is None
                         or run["gates"].get(
                             FROZEN_AXIS_GATE[(run["freeze_axis"] or "").lower()]) is False)
                    and not any(w["run_id"] == run["run_id"]
                                and w["code"] == "vacuous_negative_control"
                                for w in vacuous_control_warnings)),
            }
            for run in runs if isinstance(run.get("negative_control"), dict)
        ],
        "diagnostics": diag,
        "warnings": warnings,
        "errors": errors,
        "runs": runs,
    }


def _format(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _code(text: str) -> str:
    return BT + text + BT


def render_markdown(summary: dict) -> str:
    lines = []
    lines.append("<!-- 本文件由 scripts/simulation/p3_summary.py 自动生成，请勿手改 -->")
    lines.append("")
    lines.append("# P3 证据汇总（自动生成，非手抄）")
    lines.append("")
    lines.append("重新生成：" + _code("python scripts/simulation/p3_summary.py")
                 + "（同时写出 " + _code("p3_summary.json") + " 与本表）")
    lines.append("")
    lines.append("## 0. 结论边界（必读）")
    lines.append("")
    lines.append("> " + DIAGNOSTIC_DRIVE_NOTICE)
    lines.append("")
    lines.append("## 1. 每一 run 的证据（按 gate 代际分组，绝不混算）")
    lines.append("")
    for generation in GENERATIONS:
        subset = [run for run in summary["runs"] if run["generation"] == generation]
        lines.append(f"### {generation}（{len(subset)} 个 run）")
        lines.append("")
        if not subset:
            lines.append("（无）")
            lines.append("")
            continue
        lines.append("| run_id | profile | gate 模式 | gate_result | outer_exit | child_rc | 失败门禁 | 逐门禁 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for run in subset:
            per_gate = ", ".join(f"{name}={'T' if value else 'F'}"
                                 for name, value in run["gates"].items())
            lines.append("| {run} | {prof} | {mode} | {gate} | {outer} | {rc} | {false} | {gates} |".format(
                run=run["run_id"], prof=run["profile"], mode=run["gate_mode"],
                gate=run["gate_result"], outer=_format(run["outer_exit"]),
                rc=_format(run["child_returncode"]),
                false=(", ".join(run["gates_false"]) or "-"), gates=per_gate))
        lines.append("")
        directed_runs = [run for run in subset
                         if isinstance(run.get("directed_translation"), dict)]
        if directed_runs:
            lines.append(generation + " 有向位移（directed_translation，v1.1 判据）：")
            lines.append("")
            lines.append("| run_id | profile | 命令位移 (m) | achieved_ratio | final_ratio | best_frame | cross_track (m) | 通过 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for run in directed_runs:
                directed = run["directed_translation"]
                lines.append("| {run} | {prof} | {cmd} | {ratio} | {final} | {frame} | {cross} | {ok} |".format(
                    run=run["run_id"], prof=run["profile"],
                    cmd=_format(directed.get("commanded_displacement_m")),
                    ratio=_format(directed.get("achieved_ratio")),
                    final=_format(directed.get("final_ratio")),
                    frame=_format(directed.get("best_frame")),
                    cross=_format(directed.get("cross_track_m")),
                    ok="PASS" if directed.get("pass") else "FAIL"))
            lines.append("")

    lines.append("## 2. 分代际聚合（不跨代际平均）")
    lines.append("")
    lines.append("| 代际 | run 数 | PASS | FAIL | 有向判据 run 数 | 最小 achieved_ratio | 安装位偏差 max (m) | 安装姿态偏差 max (rad) | jerk max (m/s³) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for generation in GENERATIONS:
        agg = summary["aggregates"][generation]
        lines.append("| {g} | {n} | {p} | {f} | {d} | {r} | {mp} | {mr} | {j} |".format(
            g=generation, n=agg["runs"], p=agg["pass"], f=agg["fail"],
            d=agg["directed_runs"], r=_format(agg["directed_achieved_ratio_min"]),
            mp=_format(agg["mount_position_deviation_max_m"]),
            mr=_format(agg["mount_rotation_deviation_max_rad"]),
            j=_format(agg["peak_jerk_m_s3_max"])))
    lines.append("")
    lines.append("## 3. 覆盖审计（每个剖面每个代际需要 "
                 f"{summary['min_independent_pass']} 个独立 PASS）")
    lines.append("")
    lines.append("| 代际 | profile | 独立 PASS | 要求 | 缺口 | 看到的 run |")
    lines.append("|---|---|---|---|---|---|")
    for generation in GENERATIONS:
        for profile, data in summary["coverage"][generation].items():
            lines.append("| {g} | {p} | {n} | {req} | {gap} | {runs} |".format(
                g=generation, p=profile, n=data["independent_pass"], req=data["required"],
                gap="**缺口**" if data["gap"] else "-",
                runs=", ".join(data["runs"]) or "-"))
    lines.append("")
    lines.append("### 覆盖缺口清单")
    lines.append("")
    if summary["coverage_gaps"]:
        for gap in summary["coverage_gaps"]:
            lines.append(f"- {gap['generation']} / {gap['profile']}：独立 PASS "
                         f"{gap['independent_pass']} < {gap['required']}"
                         f"（PASS: {', '.join(gap['passes']) or '无'}；"
                         f"run: {', '.join(gap['runs_seen']) or '无'}）")
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 4. 负控制审计（缺运动必须 FAIL）")
    lines.append("")
    lines.append("| run_id | profile | frozen_axis | gate_result | outer_exit | 假的门禁 | achieved_ratio | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for control in summary["negative_controls"]:
        lines.append("| {run} | {p} | {axis} | {gate} | {outer} | {false} | {ratio} | {verdict} |".format(
            run=control["run_id"], p=control["profile"], axis=control["frozen_axis"],
            gate=control["gate_result"], outer=_format(control["outer_exit"]),
            false=", ".join(control["gates_false"]) or "-",
            ratio=_format(control["achieved_ratio"]),
            verdict="通过（按设计失败）" if control["verified"] else "**失败：负控制竟然 PASS**"))
    lines.append("")

    diag = summary["diagnostics"]
    lines.append("## 5. 诊断量：root<->chassis 偏置与安装漂移")
    lines.append("")
    lines.append("- " + _code("diagnostics.root_to_chassis_offset_m")
                 + f" 取值：{diag['root_to_chassis_offset_m_values']}"
                 f"（跨 run 极差 {_format(diag['root_to_chassis_offset_m_spread'])}）")
    lines.append(f"- 带该诊断字段的 run：{diag['runs_with_offset_diagnostic']} / "
                 f"{summary['totals']['runs']}；声明做偏置补偿的 run：{diag['runs_offset_compensated']}")
    lines.append(f"- 安装位偏差 max：{_format(diag['mount_position_deviation_max_m'])} m；"
                 f"安装姿态偏差 max：{_format(diag['mount_rotation_deviation_max_rad'])} rad"
                 "（门限 0.001 m / 0.001 rad）")
    lines.append("- " + _code("raw_poses.jsonl")
                 + f" 帧数观测：{diag['raw_poses_frames_observed']}，"
                 f"steps+1 期望：{diag['raw_poses_frames_expected']}")
    for generation in GENERATIONS:
        drift = diag["mount_drift_by_generation"][generation]
        lines.append(f"- {generation} 安装漂移："
                     f"{_format(drift['mount_position_deviation_max_m'])} m / "
                     f"{_format(drift['mount_rotation_deviation_max_rad'])} rad")
    lines.append("")
    lines.append("## 6. 自动告警与证据完整性")
    lines.append("")
    if summary["errors"]:
        for error in summary["errors"]:
            lines.append(f"- **错误**（{error['code']}，{error['run_id']}）：{error['message']}")
    else:
        lines.append("- 无错误：每个 run 都有 " + _code("result.json") + " 与 "
                     + _code("gate_runner_summary.json") + "，每个负控制都按设计失败")
    for warning in summary["warnings"]:
        lines.append(f"- 告警：{warning}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P3 evidence summariser")
    parser.add_argument("--gates-dir", default=str(DEFAULT_GATES_DIR),
                        help="directory holding the P3_* run directories")
    parser.add_argument("--extra-gates-dir", action="append", default=[],
                        help="additional root(s) scanned for P3_* run directories")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--spec-quat-rule", action="store_true",
                        help="treat a two-entry quat probe as a v1.1 marker (literal "
                             "auto-generated spec rule; not the default)")
    parser.add_argument("--no-write", action="store_true", help="do not write output files")
    args = parser.parse_args(argv)

    gates_dir = Path(args.gates_dir)
    summary = build_summary(gates_dir, tuple(Path(p) for p in args.extra_gates_dir),
                            spec_quat_rule=args.spec_quat_rule)

    if not summary["runs"] and not summary["errors"]:
        print(f"no P3_* run directories under {gates_dir}", file=sys.stderr)
        return 2

    if not args.no_write:
        json_out = Path(args.json_out)
        md_out = Path(args.md_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        md_out.write_text(render_markdown(summary) + "\n", encoding="utf-8")

    print(f"runs={summary['totals']['runs']} "
          f"v1.0={summary['totals']['by_generation'][GENERATION_V10]} "
          f"v1.1={summary['totals']['by_generation'][GENERATION_V11]} "
          f"coverage_gaps={len(summary['coverage_gaps'])} "
          f"negative_controls={len(summary['negative_controls'])} "
          f"errors={len(summary['errors'])}")
    for gap in summary["coverage_gaps"]:
        print(f"COVERAGE GAP: {gap['generation']}/{gap['profile']} "
              f"independent_pass={gap['independent_pass']} required={gap['required']}")
    for error in summary["errors"]:
        print(f"ERROR [{error['code']}] {error['message']}", file=sys.stderr)
    print(f"WARNING: {DIAGNOSTIC_DRIVE_NOTICE}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - odd stdio
        pass
    sys.exit(main())
