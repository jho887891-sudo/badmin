#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-11/P7-12 evidence runner: backend comparison + wired closed-loop episodes.

    python tools/p7/run_p7_closed_loop_wiring.py compare  --out <dir>
    python tools/p7/run_p7_closed_loop_wiring.py episodes --out <dir>

The verified backend needs a P6 checkout.  Point it at one with
$BADMIN_P6_ROOT (the p6/dynamic-fk worktree) or $BADMIN_P6_FK_PATH (the fk.py
itself).  If neither resolves, the script EXITS 2 with VERIFIED_FK_UNAVAILABLE in
the message: it never falls back to the proxy DH.

Exit codes: 0 = evidence written, 2 = the verified P6 FK was expected and is
missing, 3 = the committed racket proxy JSON was missing (episodes only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

from src.badminton_brain.harness.wired_closed_loop import (  # noqa: E402
    NOMINAL_SEED_RAD, SCENARIOS, WiredConfig, find_proxy_json, run_episodes,
    run_wired_closed_loop,
)
from src.badminton_brain.reachability import Calibration, assess_reachability  # noqa: E402
from src.badminton_brain.reachability.backends import (  # noqa: E402
    BACKEND_PROXY, BACKEND_VERIFIED, BackendDampedLeastSquaresIk, VerifiedFkUnavailable,
    kinematic_limits_for, select_backend,
)

#: The pre-registered comparison targets (metres, court frame).  Chosen ONCE:
#: three heights on the ball's own line (z = 1.35, 1.05, 0.75, 0.45) so the
#: comparison is about the kinematics, not about a lucky point.
COMPARISON_TARGETS: Tuple[Tuple[float, float, float], ...] = (
    (0.35, 0.05, 1.35),
    (0.42, 0.05, 1.05),
    (0.46, 0.06, 0.75),
    (0.36, 0.05, 0.45),
    (0.70, 0.00, 0.90),
    (1.10, 0.00, 1.20),
)
APPROACHES: Tuple[Tuple[float, float, float], ...] = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
TIME_TO_TARGET_S = 0.448


def reach_bound_sampled_m(backend, samples: int = 4000, seed: int = 20261022) -> float:
    """Largest |frame position| over seeded joint samples: a LOWER-bound estimate."""
    rng = np.random.default_rng(seed)
    lower = np.asarray(backend.lower_rad, dtype=float)
    upper = np.asarray(backend.upper_rad, dtype=float)
    best = 0.0
    for _ in range(samples):
        q = lower + (upper - lower) * rng.random(lower.size)
        best = max(best, float(np.linalg.norm(backend.position(q))))
    return best


def reach_bound_analytic_m(backend) -> Tuple[float, str]:
    """ANALYTIC reach bound: the sum of the rigid link lengths (a sphere bound).

    Rigid link lengths do not depend on q, so summing the consecutive frame
    distances at the zero pose is the true upper bound on |position| - no
    sampling, no lucky configuration.
    """
    if backend.backend_id == BACKEND_PROXY:
        from src.badminton_brain.reachability.kinematics import MAX_FACE_REACH_M
        return float(MAX_FACE_REACH_M), "P7-09 MAX_FACE_REACH_M = sum(|a|+|d|) + 0.50 m tool"
    chain = backend.chain
    names = list(chain.module.LINK_NAMES) if hasattr(chain, "module") else None
    frames = chain.joint_world_frames(np.zeros(backend.dof))
    order = [name for name in ("arm_root", "base_link", "link1", "link2", "link3", "link4",
                               "link5", "link6", "gripper_end") if name in frames]
    total = 0.0
    for previous, current in zip(order, order[1:]):
        total += float(np.linalg.norm(frames[current][:3, 3] - frames[previous][:3, 3]))
    return total, ("P6 q=0 chain: sum of consecutive arm_root->gripper_end link distances "
                   "from the loaded module")


def seeds_for(backend) -> List[np.ndarray]:
    """A DECLARED multi-start: the harness posture first, then the mid-range."""
    lower = np.asarray(backend.lower_rad, dtype=float)
    upper = np.asarray(backend.upper_rad, dtype=float)
    declared = np.asarray(NOMINAL_SEED_RAD[backend.backend_id], dtype=float)
    return [np.clip(declared, lower, upper), 0.5 * (lower + upper)]


def comparison(extra_roots: Sequence[str] = ()) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    backends = [select_backend(BACKEND_PROXY),
                select_backend(BACKEND_VERIFIED, extra_roots=extra_roots)]
    calibration = Calibration(court_from_base=np.eye(4), verified=True,
                              source="ASSUMED identity (P7-11 comparison harness only)")
    for point in COMPARISON_TARGETS:
        for approach in APPROACHES:
            for backend in backends:
                limits = kinematic_limits_for(backend)
                solver = BackendDampedLeastSquaresIk(backend)
                bound_m, bound_source = reach_bound_analytic_m(backend)
                verdict = None
                best_residual = None
                for seed in seeds_for(backend):
                    candidate = assess_reachability(
                        intercept_point_m=point, approach_direction=approach, limits=limits,
                        time_to_target_s=TIME_TO_TARGET_S, seed_rad=seed, solver=solver,
                        calibration=calibration, position_sigma_m=0.01)
                    residual = (None if candidate.ik is None
                                else float(candidate.ik.residual_position_m))
                    if residual is not None and (best_residual is None
                                                 or residual < best_residual):
                        best_residual = residual
                    if verdict is None or verdict.decision.value == "NOT_REACHABLE"                             and candidate.decision.value != "NOT_REACHABLE":
                        verdict = candidate
                rows.append({
                    "target_m": [float(v) for v in point],
                    "approach": [float(v) for v in approach],
                    "backend": backend.backend_id,
                    "source_class": backend.source_class,
                    "frame_name": backend.frame_name,
                    "dof": int(backend.dof),
                    "reach_bound_analytic_m": round(bound_m, 6),
                    "reach_bound_source": bound_source,
                    "reach_bound_sampled_m": round(reach_bound_sampled_m(backend), 6),
                    "decision": verdict.decision.value,
                    "reason": verdict.reason.value,
                    "ik_residual_position_m": None if verdict.ik is None
                    else float(verdict.ik.residual_position_m),
                    "ik_residual_rotation_rad": None if verdict.ik is None
                    else float(verdict.ik.residual_rotation_rad),
                    "joint_margin": verdict.joint_margin_rad,
                    "required_slew_s": verdict.required_slew_s,
                    "detail": verdict.detail,
                })
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        key = "%s|%s" % (row["target_m"], row["approach"])
        by_case.setdefault(key, {})[row["backend"]] = row
    disagreements = []
    for key, entry in by_case.items():
        decisions = {row["decision"] for row in entry.values()}
        if len(decisions) > 1 or len({row["reason"] for row in entry.values()}) > 1:
            disagreements.append({"case": key, "decisions": {
                backend: entry[backend]["reason"] for backend in entry}})
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault(row["backend"], {})
        key = row["decision"] + ":" + row["reason"]
        bucket[key] = bucket.get(key, 0) + 1
    residuals: Dict[str, Any] = {}
    for backend in counts:
        values = [row["ik_residual_position_m"] for row in rows
                  if row["backend"] == backend and row["ik_residual_position_m"] is not None]
        residuals[backend] = None if not values else {
            "count": len(values), "max": float(max(values)),
            "min": float(min(values)), "p50": float(np.percentile(values, 50))}
    return {
        "schema": "badmin.p7.backend_comparison",
        "schema_version": 1,
        "targets": [list(point) for point in COMPARISON_TARGETS],
        "approaches": [list(a) for a in APPROACHES],
        "time_to_target_s": TIME_TO_TARGET_S,
        "rows": rows,
        "verdict_counts": counts,
        "ik_residual_position_m": residuals,
        "disagreements": disagreements,
        "evidence_class": "SIMULATION-ONLY",
        "note": "the two backends answer for DIFFERENT frames: the proxy answers for a "
                "proxy racket-face centre, the verified backend for the authored "
                "gripper_end frame. The racket face is UNKNOWN on the verified chain.",
    }


def _episodes(extra_roots: Sequence[str]) -> Dict[str, Any]:
    calibration = Calibration(court_from_base=np.eye(4), verified=True,
                              source="ASSUMED identity (P7-12 wiring demo only)")
    out: Dict[str, Any] = {"schema": "badmin.p7.wired_episodes", "schema_version": 1,
                           "seeds": [20261022, 7, 8], "episode_reports": {}}
    for backend in (BACKEND_PROXY, BACKEND_VERIFIED):
        out[backend] = run_episodes(backend=backend, seeds=(20261022, 7, 8),
                                    calibration=calibration,
                                    p6_root=(extra_roots[0] if extra_roots else None))
    out["status"] = "WIRED_RUN_COMPLETE"
    out["hit_rate"] = None
    out["real_hit_rate_claimed"] = False
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "episodes", "all"))
    parser.add_argument("--out", default=str(ROOT / "outputs/badminton_brain/p7_closed_loop"))
    parser.add_argument("--p6-root", default=None)
    args = parser.parse_args(argv)
    extra = (args.p6_root,) if args.p6_root else ()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    try:
        if args.mode in ("compare", "all"):
            payload["backend_comparison"] = comparison(extra)
            path = out_dir / "backend_compare.json"
            path.write_text(json.dumps(payload["backend_comparison"], indent=2,
                                       sort_keys=True), encoding="utf-8")
            print("wrote", path)
        if args.mode in ("episodes", "all"):
            if find_proxy_json(extra_roots=extra) is None:
                print("RACKET_COLLISION_PROXY_MISSING: no committed P6 proxy JSON found "
                      "(set $BADMIN_P6_ROOT)", file=sys.stderr)
                return 3
            payload["episodes"] = _episodes(extra)
            path = out_dir / "wired_episodes.json"
            path.write_text(json.dumps(payload["episodes"], indent=2, sort_keys=True),
                            encoding="utf-8")
            print("wrote", path)
    except VerifiedFkUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if "backend_comparison" in payload:
        print(json.dumps(payload["backend_comparison"]["verdict_counts"], indent=2,
                         sort_keys=True))
    if "episodes" in payload:
        for backend in (BACKEND_PROXY, BACKEND_VERIFIED):
            summary = payload["episodes"][backend]
            print(backend, "episodes", summary["episodes"],
                  "in_hold", summary["episodes_ending_in_hold"],
                  "holds", summary["hold_reasons_total"],
                  "verdicts", summary["reachability_verdicts_total"],
                  "geometry", summary["geometry_evaluations"],
                  summary["geometry_contacts"],
                  "hit_rate", summary["hit_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
