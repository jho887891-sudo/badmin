# -*- coding: utf-8 -*-
"""R10-F1 (P5-09): the unified controller API contract, and REFUSAL BEFORE WRITE.

The central assertion of this file is a COUNT, not a message: on every refusal path the
subclass' _do_write / _do_reset hook is called exactly zero times.  A refusal that arrives
after the write is not a safety property - it is a log line about a command that already
happened.

Run:
    python -m pytest tests/r10_f/test_controller_api_contract.py -q
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

NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
         "gripper_joint1", "gripper_joint2")
LIMITS = {
    "joint1": (-2.8, 2.8, "rad"), "joint2": (0.0, 3.14, "rad"),
    "joint3": (0.0, 3.14, "rad"), "joint4": (-1.57, 1.57, "rad"),
    "joint5": (-1.57, 1.57, "rad"), "joint6": (-3.14, 3.14, "rad"),
    "gripper_joint1": (0.0, 0.05, "m"), "gripper_joint2": (0.0, 0.0715, "m"),
}
REST = (0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575)
T0 = 10.0


class FakeArm(api.ArmController):
    """A fake adapter: records every hook call.  This is the write counter."""

    def __init__(self, positions=REST, **kwargs):
        super().__init__(NAMES, limits=LIMITS, model="rebot_b601_rs", **kwargs)
        self.positions = tuple(float(v) for v in positions)
        self.writes = []
        self.resets = []
        self.reads = 0

    def _do_read(self, at_s):
        self.reads += 1
        return self.positions, None

    def _do_write(self, values, at_s):
        self.writes.append((values, at_s))

    def _do_reset(self, rows, at_s):
        self.resets.append((rows, at_s))


def _arm(**kwargs):
    arm = FakeArm(**kwargs)
    arm.sample_state(T0)
    return arm


# ------------------------------------------------------------------ interface
def test_the_interface_is_abstract_and_declares_the_five_operations():
    assert issubclass(api.ArmController, object)
    for name in ("sample_state", "state", "command_joint_targets", "reset", "hold",
                 "release", "solve_ik", "describe"):
        assert callable(getattr(api.ArmController, name)), name
    with pytest.raises(TypeError):
        api.ArmController(NAMES, limits=LIMITS)          # abstract hooks not implemented
    assert api.API_VERSION.startswith("r10f/")


def test_the_adapter_is_the_concrete_implementation_of_the_interface():
    assert issubclass(api.B601RSControllerAdapter, api.ArmController)
    assert isinstance(api.B601RSControllerAdapter.__dict__["_do_write"], type(lambda: 0)) or \
        callable(getattr(api.B601RSControllerAdapter, "_do_write"))


def test_every_refusal_is_typed_and_carries_a_stable_code():
    codes = {cls.code for cls in (
        api.UnknownJointTarget, api.TargetWidthError, api.NonFiniteTarget,
        api.OutOfLimitTarget, api.StaleState, api.StateUnavailable, api.DoubleReset,
        api.HoldDuringMotion, api.HoldEngaged, api.NotHolding, api.HoldWithoutTarget,
        api.IkUnavailable)}
    assert len(codes) == 12
    assert len(set(codes)) == len(codes)
    for cls in (api.UnknownJointTarget, api.OutOfLimitTarget, api.StaleState):
        assert issubclass(cls, api.ControllerRefusal)
        assert issubclass(cls, RuntimeError)


# ------------------------------------------------ refusal happens BEFORE the write
def test_out_of_limit_target_is_refused_before_any_write():
    arm = _arm()
    bad = list(REST)
    bad[2] = 3.5                                        # joint3 upper stop is 3.14
    with pytest.raises(api.OutOfLimitTarget) as excinfo:
        arm.command_joint_targets(bad, T0 + 0.01)
    assert arm.write_count == 0
    assert arm.writes == []                             # the sink was never touched
    assert excinfo.value.context["written"] is False
    assert excinfo.value.context["joint"] == "joint3"
    assert "REFUSED BEFORE ANY WRITE" in str(excinfo.value)


def test_out_of_limit_reset_is_refused_before_any_reset_write():
    arm = _arm()
    bad = list(REST)
    bad[6] = 0.06                                       # finger stroke is 0.05 m
    with pytest.raises(api.OutOfLimitTarget):
        arm.reset(bad, T0 + 0.01)
    assert arm.reset_count == 0
    assert arm.resets == []


def test_unknown_joint_is_refused_before_any_write():
    arm = _arm()
    named = {name: value for name, value in zip(NAMES, REST)}
    named["joint7"] = 0.0
    with pytest.raises(api.UnknownJointTarget) as excinfo:
        arm.command_joint_targets(named, T0 + 0.01)
    assert excinfo.value.context["joint"] == "joint7"
    assert arm.writes == []
    del named["joint7"]
    del named["joint5"]
    with pytest.raises(api.UnknownJointTarget):
        arm.command_joint_targets(named, T0 + 0.01)     # a MISSING name is refused too
    assert arm.writes == []


def test_width_and_finiteness_are_refused_before_any_write():
    arm = _arm()
    with pytest.raises(api.TargetWidthError):
        arm.command_joint_targets(REST[:7], T0 + 0.01)
    nan = list(REST)
    nan[0] = float("nan")
    with pytest.raises(api.NonFiniteTarget):
        arm.command_joint_targets(nan, T0 + 0.01)
    inf = list(REST)
    inf[1] = float("inf")
    with pytest.raises(api.NonFiniteTarget):
        arm.command_joint_targets(inf, T0 + 0.01)
    assert arm.writes == []


def test_stale_state_is_refused_and_a_reset_invalidates_the_cached_reading():
    arm = FakeArm(max_state_age_s=0.1)
    with pytest.raises(api.StateUnavailable):
        arm.state(T0)                                   # nothing sampled yet
    arm.sample_state(T0)
    assert arm.state(T0 + 0.05).joint_pos == REST
    with pytest.raises(api.StaleState):
        arm.state(T0 + 0.5)
    arm.reset(REST, T0 + 0.6)
    assert arm.reset_count == 1
    with pytest.raises(api.StateUnavailable):
        arm.state(T0 + 0.61)                            # the reset invalidated it


def test_double_reset_needs_progress_in_between():
    arm = _arm()
    arm.reset(REST, T0 + 0.01)
    with pytest.raises(api.DoubleReset):
        arm.reset(REST, T0 + 0.02)
    assert arm.reset_count == 1
    arm.sample_state(T0 + 0.03)                         # reading the state is progress
    arm.reset(REST, T0 + 0.04)
    assert arm.reset_count == 2


def test_hold_during_motion_is_refused_until_the_arm_settles():
    arm = _arm()
    arm.command_joint_targets(REST, T0 + 0.01)
    with pytest.raises(api.HoldDuringMotion) as excinfo:
        arm.hold(T0 + 0.02)
    assert excinfo.value.context["pending"] == [float(v) for v in REST]
    assert arm.holding is False
    arm.settle(T0 + 0.03)
    ticket = arm.hold(T0 + 0.04)
    assert ticket.joint_pos == tuple(float(v) for v in REST)
    assert arm.holding is True
    with pytest.raises(api.HoldEngaged):
        arm.command_joint_targets(REST, T0 + 0.05)
    with pytest.raises(api.HoldEngaged):
        arm.reset(REST, T0 + 0.06)
    with pytest.raises(api.HoldEngaged):
        arm.hold(T0 + 0.07)
    assert arm.write_count == 1 and arm.reset_count == 0
    arm.release(T0 + 0.08)
    with pytest.raises(api.NotHolding):
        arm.release(T0 + 0.09)


def test_hold_before_any_target_is_refused():
    arm = _arm()
    with pytest.raises(api.HoldWithoutTarget):
        arm.hold(T0 + 0.01)


def test_the_ik_hook_raises_instead_of_returning_a_pose():
    arm = _arm()
    with pytest.raises(api.IkUnavailable):
        arm.solve_ik(object(), T0 + 0.01, seed=REST)
    assert arm.writes == []


def test_an_accepted_target_records_its_limit_margin():
    arm = _arm()
    command = arm.command_joint_targets(REST, T0 + 0.01)
    assert arm.write_count == 1
    assert command.joint_pos == tuple(float(v) for v in REST)
    margins = dict(zip(NAMES, command.limit_margin))
    assert margins["joint2"] == pytest.approx(0.3)
    assert margins["gripper_joint1"] == pytest.approx(0.025)
    assert set(arm.describe()) >= {"api_version", "model", "joint_names", "limits",
                                   "write_count", "envelope_source"}
