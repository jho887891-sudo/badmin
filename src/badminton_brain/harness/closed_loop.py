# -*- coding: utf-8 -*-
"""P7-10 harness: a PRE-REGISTERED simulation closed-loop rig - HARNESS ONLY.

WHAT THIS IS
------------
Three fixed scenarios drive the P7-10 safety FSM with MOCK perception and MOCK
prediction:
    FIXED_BALL          the shuttle sits still at a fixed point
    LOW_SPEED_INCOMING  a deterministic 2 m/s linear incoming ball
    PERTURBED           the same ball with a seeded, reproducible position noise
The point is to exercise the state machine and the FAIL-CLOSED paths end to end,
in a way a reviewer can rerun bit-for-bit.

WHAT THIS IS NOT
----------------
This is NOT a closed-loop hit-rate experiment and it makes NO claim about hitting
a shuttle.  There is no physics engine here, no racket collision mesh, no camera,
no calibration and no robot.  A "contact" in this file means "the mock ball centre
came within the proxy radius of the mock face centre", nothing more.

REFUSAL RULE (pre-registered)
-----------------------------
If the racket collision proxy OR the calibration is missing, run_closed_loop()
REFUSES: it returns status REFUSED, hit_rate None, and computes no contact rate at
all.  A hit rate without a collision model and a calibrated frame would be a
number with no referent.  Even when both are supplied, the report keeps
real_hit_rate_claimed = False and labels the rate SYNTHETIC-MOCK-ONLY.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..reachability import (Calibration, DampedLeastSquaresIk, forward_kinematics_face,
                             target_from_intercept)
from ..safety.interaction_fsm import (FrameInput, FsmState, InteractionFsm, PlanInput,
                                      PredictionInput)

__all__ = ["SCENARIOS", "HarnessRefusal", "RacketCollisionProxy", "HarnessConfig",
           "run_closed_loop", "main"]

SCENARIOS: Tuple[str, ...] = ("FIXED_BALL", "LOW_SPEED_INCOMING", "PERTURBED")

#: Pre-registered scenario constants (fixed before any run).
BALL_START_M = (0.35, 0.05, 1.35)
BALL_VELOCITY_MPS = (0.25, 0.02, -2.0)   # LOW_SPEED_INCOMING: |v| ~ 2 m/s, slow by design
STEP_DT_S = 1.0 / 60.0
STEPS = 12
PERTURBATION_SIGMA_M = 0.004
#: The mock plan always aims at this fixed intercept window, with an approach along
#: -X (toward the robot).  It is a MOCK geometric target: chosen because the PROXY
#: arm can actually converge to it, so the harness exercises the whole FSM instead
#: of latching HOLD on the first IK call.  It says nothing about a real swing.
MOCK_INTERCEPT_M = (0.45, 0.0, 1.0)
MOCK_APPROACH = (-1.0, 0.0, 0.0)
MOCK_LEAD_S = 0.75


class HarnessRefusal(RuntimeError):
    """The harness cannot produce a rate: a required model is missing."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class RacketCollisionProxy:
    """A SPHERE proxy for the racket face - not the racket mesh, not a contact model."""

    face_radius_m: float = 0.11
    name: str = "sphere_proxy/ENGINEERING_V0_1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.face_radius_m) or self.face_radius_m <= 0.0:
            raise ValueError("face_radius_m must be finite and > 0")


@dataclass(frozen=True)
class HarnessConfig:
    scenario: str = "LOW_SPEED_INCOMING"
    seed: int = 20261022
    steps: int = STEPS
    dt_s: float = STEP_DT_S
    calibration: Optional[Calibration] = None
    racket_proxy: Optional[RacketCollisionProxy] = None

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {self.scenario!r}")
        if self.steps <= 0 or not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("steps and dt_s must be positive")


def _ball_position(scenario: str, step: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    start = np.asarray(BALL_START_M, dtype=float)
    if scenario == "FIXED_BALL":
        return start.copy()
    velocity = np.asarray(BALL_VELOCITY_MPS, dtype=float)
    position = start + velocity * (step * dt)
    if scenario == "PERTURBED":
        position = position + rng.normal(0.0, PERTURBATION_SIGMA_M, size=3)
    return position


def _mock_prediction(frame: FrameInput, now_s: float, velocity: np.ndarray,
                     intercept_delay_s: float) -> PredictionInput:
    """Constant-velocity extrapolation to a fixed lead time - a MOCK, no model."""
    return PredictionInput(
        prediction_id=f"mock-{frame.frame_id}",
        frame_id=frame.frame_id,
        reference_time_s=now_s,
        intercept_time_s=now_s + intercept_delay_s,
        intercept_point_m=MOCK_INTERCEPT_M,
        approach_direction=MOCK_APPROACH,
        sigma_start_m=float(frame.sigma_m),
        sigma_growth_mps=0.02,
    )


def _mock_command(seed: np.ndarray) -> np.ndarray:
    """The mock planner's setpoint: the proxy-arm IK solution for the mock window."""
    target = target_from_intercept(MOCK_INTERCEPT_M, MOCK_APPROACH)
    result = DampedLeastSquaresIk().solve(target, seed)
    return np.asarray(result.joints_rad, dtype=float)


def run_closed_loop(config: HarnessConfig) -> Dict[str, Any]:
    """Run one scenario through the FSM and return a report (never a real hit rate)."""
    base: Dict[str, Any] = {
        "schema": "badmin.p7.closed_loop_harness",
        "schema_version": 1,
        "scenario": config.scenario,
        "seed": config.seed,
        "evidence_class": "SYNTHETIC-MOCK-ONLY",
        "real_hit_rate_claimed": False,
        "physics_engine": None,
        "camera": None,
        "robot": None,
    }
    missing: List[str] = []
    if config.racket_proxy is None:
        missing.append("RACKET_COLLISION_PROXY_MISSING")
    if config.calibration is None or not config.calibration.usable():
        missing.append("CALIBRATION_MISSING")
    if missing:
        base.update({
            "status": "REFUSED",
            "refusal_reason": missing,
            "hit_rate": None,
            "synthetic_contact_rate": None,
            "contacts": None,
            "note": "no racket collision proxy and/or no calibration: the harness refuses to "
                    "report a hit rate, because the number would have no referent",
        })
        return base

    rng = np.random.default_rng(config.seed)
    fsm = InteractionFsm(calibration=config.calibration, ik_solver=DampedLeastSquaresIk())
    velocity = (np.zeros(3) if config.scenario == "FIXED_BALL"
                else np.asarray(BALL_VELOCITY_MPS, dtype=float))
    proxy = config.racket_proxy
    contacts = 0
    ball_positions: List[List[float]] = []
    face_positions: List[List[float]] = []
    now_s = 0.0
    seed = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])
    # The MOCK planner's setpoint: the documented proxy-arm solution for the mock
    # intercept window, computed once, off-line, with the same solver the FSM uses.
    command = _mock_command(seed)
    for step in range(config.steps):
        now_s = step * config.dt_s
        ball = _ball_position(config.scenario, step, config.dt_s, rng)
        ball_positions.append([round(float(v), 6) for v in ball])
        frame = FrameInput(frame_id=step, capture_time_s=now_s,
                           received_at_s=now_s + 0.002, detection_dim=3,
                           position_m=ball, sigma_m=0.01)
        state = fsm.on_frame(frame, now_s + 0.002)
        if state is FsmState.HOLD:
            break
        prediction = _mock_prediction(frame, now_s + 0.002, velocity, MOCK_LEAD_S)
        state = fsm.on_prediction(prediction, now_s + 0.002)
        if state is FsmState.HOLD:
            break
        plan = PlanInput(planned_at_s=now_s + 0.002, joints_rad=command,
                         seed_rad=seed, duration_s=0.002)
        state = fsm.on_plan(plan, now_s + 0.002)
        if state is FsmState.HOLD:
            break
        state = fsm.on_execute_command(command, now_s + 0.002)
        if state is FsmState.HOLD:
            break
        # PROXY contact test: the commanded face centre vs the mock ball centre.
        face = forward_kinematics_face(command)[:3, 3]
        face_positions.append([round(float(v), 6) for v in face])
        if float(np.linalg.norm(face - ball)) <= proxy.face_radius_m:
            contacts += 1
        seed = command

    executed = sum(1 for e in fsm.events if e.trigger == "COMMAND")
    report = dict(base)
    report.update({
        "status": "MOCK_RUN_COMPLETE",
        "hit_rate": None,  # never reported as a hit rate: see real_hit_rate_claimed
        "synthetic_contact_rate": (None if executed == 0 else round(contacts / executed, 6)),
        "contacts": contacts,
        "executed_commands": executed,
        "iterations": len(ball_positions),
        "final_state": fsm.state.value,
        "hold_reason": None if fsm.hold_reason is None else fsm.hold_reason.value,
        "racket_proxy": proxy.name,
        "contact_definition": "distance(mock face centre, mock ball centre) <= proxy radius",
        "ball_positions_m": ball_positions,
        "face_positions_m": face_positions,
        "fsm_report": fsm.report(),
        "note": "synthetic_contact_rate is a MOCK geometry statistic. It is NOT a hit rate, "
                "it is NOT measured on hardware, and it MUST NOT be quoted as one.",
    })
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", choices=list(SCENARIOS) + ["all"])
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--without-proxy", action="store_true",
                        help="demo the refusal path (no collision proxy supplied)")
    args = parser.parse_args(argv)
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    out = {}
    for scenario in scenarios:
        config = HarnessConfig(
            scenario=scenario, seed=args.seed,
            calibration=Calibration(court_from_base=np.eye(4), verified=True),
            racket_proxy=None if args.without_proxy else RacketCollisionProxy(),
        )
        out[scenario] = run_closed_loop(config)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
