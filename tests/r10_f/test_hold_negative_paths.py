# -*- coding: utf-8 -*-
"""R10-F3 (P7-10): the safety FSM is FALSIFIABLE - five negative paths, all latching, zero commands.

Each test drives the real InteractionFsm to a stage where one check is the binding constraint,
then asserts four things:
    * the FSM state is HOLD,
    * the typed HOLD reason is the one expected (not "some failure"),
    * ZERO commands were emitted,
    * the latch holds: another stage call is suppressed and does not move the FSM.

The last test joins the two layers: the same over-limit joint vector the FSM refused is handed
to the arm controller API, which refuses it with OutOfLimitTarget and a write count of ZERO.
That is the explicit "no out-of-limit target is written before the refusal" assertion.

Run:
    python -m pytest tests/r10_f/test_hold_negative_paths.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "src", ROOT / "simulation", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import badminton_scene.controller_api as api  # noqa: E402
from src.badminton_brain.reachability import Calibration, IkResult, IkSolver  # noqa: E402
from src.badminton_brain.safety.interaction_fsm import (  # noqa: E402
    CommandSuppressed, FrameInput, FsmState, HoldReason, InteractionFsm, PlanInput,
    PredictionInput, ResetRejected,
)

INTERCEPT = (0.45, 0.0, 1.0)
APPROACH = (-1.0, 0.0, 0.0)
SEED = (0.3, 0.8, -1.2, 0.1, 0.4, 0.2)
NEAR_STOP = (0.3, 0.8, -0.02, 0.1, 0.4, 0.2)      # joint3 = -0.02: inside the -0.05 margin
NOW = 0.002


class PassThroughIk(IkSolver):
    solver_id = "test/pass_through"

    def solve(self, target, seed_rad):
        seed = np.asarray(seed_rad, dtype=float)
        return IkResult(joints_rad=tuple(float(v) for v in seed), residual_position_m=0.0,
                        residual_rotation_rad=0.0, iterations=1,
                        tolerance_position_m=1e-4, tolerance_rotation_rad=1e-3,
                        solver=self.solver_id)


def verified():
    return Calibration(court_from_base=np.eye(4), verified=True)


def make_fsm():
    return InteractionFsm(calibration=verified(), ik_solver=PassThroughIk())


def frame(capture=0.0, position=INTERCEPT, sigma=0.01, dim=3, frame_id=1):
    return FrameInput(frame_id=frame_id, capture_time_s=capture, received_at_s=0.002,
                      detection_dim=dim,
                      position_m=position if dim == 3 else None,
                      sigma_m=sigma if dim == 3 else None)


def prediction(now=NOW, lead=0.30, sigma=0.01, growth=0.02, point=INTERCEPT, pid="p1"):
    return PredictionInput(prediction_id=pid, frame_id=1, reference_time_s=now,
                           intercept_time_s=now + lead, intercept_point_m=point,
                           approach_direction=APPROACH, sigma_start_m=sigma,
                           sigma_growth_mps=growth)


def plan(now=NOW, joints=SEED, seed=SEED, duration=0.002):
    return PlanInput(planned_at_s=now, joints_rad=joints, seed_rad=seed, duration_s=duration)


def drive_to_plan(fsm, now=NOW, point=INTERCEPT, sigma=0.01):
    assert fsm.on_frame(frame(position=point, sigma=sigma), now) is FsmState.TRACK
    assert fsm.on_prediction(prediction(now=now, point=point, sigma=sigma), now) \
        is FsmState.PREDICT
    assert fsm.on_plan(plan(now=now), now) is FsmState.PLAN


def assert_latched(fsm, reason, *, now=NOW):
    assert fsm.state is FsmState.HOLD, fsm.report()["events"]
    assert fsm.hold_reason is reason, fsm.report()["hold"]
    assert fsm.emitted_commands == []
    assert fsm.report()["commands_emitted"] == 0
    # the latch: another stage call is suppressed and does not move the FSM
    before = len(fsm.events)
    fsm.on_frame(frame(position=INTERCEPT), now + 0.01)
    assert fsm.state is FsmState.HOLD
    assert len(fsm.events) > before
    assert fsm.emitted_commands == []


# ---------------------------------------------------------------- 1. joint margin
def test_joint_limit_margin_latches_hold_and_emits_nothing():
    fsm = make_fsm()
    drive_to_plan(fsm)
    state = fsm.on_execute_command(NEAR_STOP, NOW)
    assert state is FsmState.HOLD
    assert_latched(fsm, HoldReason.JOINT_LIMIT_MARGIN)
    assert "margin" in fsm.hold_detail


# ------------------------------------------------------- 2. self-collision proxy
def test_self_collision_proxy_latches_hold_and_emits_nothing():
    fsm = make_fsm()
    fsm.on_frame(frame(position=(0.20, 0.0, 1.0)), NOW)
    fsm.on_prediction(prediction(point=(0.20, 0.0, 1.0)), NOW)
    state = fsm.on_plan(plan(now=NOW), NOW)
    assert state is FsmState.HOLD
    assert_latched(fsm, HoldReason.UNREACHABLE_TARGET)
    assert "SELF_COLLISION_PROXY" in fsm.hold_detail


# ------------------------------------------------------------- 3. stale / late
def test_a_stale_frame_latches_hold():
    fsm = make_fsm()
    state = fsm.on_frame(frame(capture=-0.5), NOW)
    assert state is FsmState.HOLD
    assert_latched(fsm, HoldReason.STALE_FRAME)


def test_a_target_whose_time_has_passed_latches_hold():
    fsm = make_fsm()
    fsm.on_frame(frame(), NOW)
    late = PredictionInput(prediction_id="late", frame_id=1, reference_time_s=NOW,
                           intercept_time_s=NOW - 0.05, intercept_point_m=INTERCEPT,
                           approach_direction=APPROACH, sigma_start_m=0.01,
                           sigma_growth_mps=0.0)
    assert fsm.on_prediction(late, NOW) is FsmState.HOLD
    assert_latched(fsm, HoldReason.TIME_TO_HIT_INVALID)


# --------------------------------------------------------- 4. unreachable target
def test_an_unreachable_target_latches_hold():
    fsm = make_fsm()
    fsm.on_frame(frame(position=(2.4, 0.0, 1.0)), NOW)
    fsm.on_prediction(prediction(point=(2.4, 0.0, 1.0)), NOW)
    assert fsm.on_plan(plan(now=NOW), NOW) is FsmState.HOLD
    assert_latched(fsm, HoldReason.UNREACHABLE_TARGET)
    assert "WORKSPACE_BOUND" in fsm.hold_detail


# ------------------------------------------------------------ 5. low confidence
def test_low_confidence_latches_hold():
    fsm = make_fsm()
    state = fsm.on_frame(frame(sigma=0.40), NOW)
    assert state is FsmState.HOLD
    assert_latched(fsm, HoldReason.COVARIANCE_ABOVE_THRESHOLD)


def test_a_rejected_reset_keeps_the_latch():
    fsm = make_fsm()
    fsm.on_frame(frame(capture=-0.5), NOW)
    assert fsm.state is FsmState.HOLD
    with pytest.raises(ResetRejected):
        fsm.reset(frame=frame(capture=-0.5), prediction=prediction(), now_s=NOW,
                  operator="r10f")
    assert fsm.state is FsmState.HOLD
    assert fsm.hold_reason is HoldReason.STALE_FRAME
    assert fsm.emitted_commands == []


# ------------------------------------- 6. no out-of-limit target written, ever
class _CountingArm(api.ArmController):
    RS_LIMITS = {
        "joint1": (-2.8, 2.8, "rad"), "joint2": (0.0, 3.14, "rad"),
        "joint3": (0.0, 3.14, "rad"), "joint4": (-1.57, 1.57, "rad"),
        "joint5": (-1.57, 1.57, "rad"), "joint6": (-3.14, 3.14, "rad"),
        "gripper_joint1": (0.0, 0.05, "m"), "gripper_joint2": (0.0, 0.0715, "m"),
    }

    def __init__(self):
        super().__init__(tuple(self.RS_LIMITS), limits=self.RS_LIMITS,
                         model="rebot_b601_rs")
        self.writes = []

    def _do_read(self, at_s):
        return (0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575), None

    def _do_write(self, values, at_s):
        self.writes.append(values)

    def _do_reset(self, rows, at_s):
        raise AssertionError("a refusal must never reach the reset hook")


def test_the_arm_api_refuses_before_writing_the_vector_the_fsm_refused():
    fsm = make_fsm()
    drive_to_plan(fsm)
    assert fsm.on_execute_command(NEAR_STOP, NOW) is FsmState.HOLD
    assert fsm.emitted_commands == []

    arm = _CountingArm()
    arm.sample_state(0.0)
    rs_vector = (0.0, 0.3, -0.02, 0.0, 0.0, 0.0, 0.025, 0.03575)   # joint3 < 0.0
    with pytest.raises(api.OutOfLimitTarget) as excinfo:
        arm.command_joint_targets(rs_vector, 0.01)
    assert arm.writes == []
    assert arm.write_count == 0
    assert excinfo.value.context["joint"] == "joint3"
    assert excinfo.value.context["written"] is False
