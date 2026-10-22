# -*- coding: utf-8 -*-
"""R10-F1 (P5-09) part 2: the CONCRETE adapter over the existing B601RSController.

The contract tests use a fake adapter to prove refusal-before-write.  This file proves the
same ordering on the REAL wiring: a B601RSController bound to a numpy articulation shim and a
recording reset function.  If the adapter delegated an out-of-limit target and let the wrapped
controller refuse it, the reset hook would already have been called - it is not.

Run:
    python -m pytest tests/r10_f/test_controller_adapter_wiring.py -q
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

import badminton_scene.controller as controller_mod  # noqa: E402
import badminton_scene.controller_api as api  # noqa: E402

NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
         "gripper_joint1", "gripper_joint2")
REST = (0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575)
T0 = 3.0


class ShimData:
    def __init__(self, joint_pos, joint_vel=None):
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel


class ShimRobot:
    def __init__(self, names, joint_pos=None):
        self.joint_names = list(names)
        self.num_joints = len(self.joint_names)
        self.data = ShimData(joint_pos)
        self.writes = []

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        self.writes.append({"joint_pos": joint_pos, "joint_vel": joint_vel, "env_ids": env_ids})


def _identity_tensor(rows):
    return [list(row) for row in rows]


@pytest.fixture
def adapter():
    calls = []

    def _reset(env_ids, robot, shuttle, joint_pos=None):
        calls.append({"env_ids": env_ids, "joint_pos": joint_pos})
        return None

    robot = ShimRobot(NAMES, np.array(REST))
    wrapped = controller_mod.B601RSController(robot, reset_fn=_reset,
                                              tensor_factory=_identity_tensor)
    arm = api.B601RSControllerAdapter(wrapped)
    return arm, robot, wrapped, calls


def test_the_adapter_publishes_the_wrapped_model_and_limits(adapter):
    arm, _robot, wrapped, _calls = adapter
    assert arm.model == wrapped.model == "rebot_b601_rs"
    assert arm.joint_names == tuple(wrapped.joint_names)
    assert arm.dof_count == 8
    limits = arm.limits()
    assert limits["joint3"] == (0.0, 3.14, "rad")
    assert limits["gripper_joint2"] == (0.0, 0.0715, "m")


def test_state_comes_from_the_wrapped_controller_by_name(adapter):
    arm, _robot, _wrapped, _calls = adapter
    reading = arm.sample_state(T0)
    assert reading["joint3"] == pytest.approx(1.0)
    assert reading.by_name()["gripper_joint2"] == pytest.approx(0.03575)
    assert reading.sampled_at_s == T0


def test_an_out_of_limit_reset_never_reaches_the_reset_hook(adapter):
    arm, _robot, _wrapped, calls = adapter
    bad = list(REST)
    bad[2] = -0.5                                   # joint3 lower stop is 0.0 rad
    with pytest.raises(api.OutOfLimitTarget) as excinfo:
        arm.reset(bad, T0 + 0.1)
    assert calls == []                              # the write hook was never reached
    assert arm.reset_count == 0
    assert excinfo.value.context["joint"] == "joint3"
    assert excinfo.value.context["written"] is False


def test_an_in_limit_reset_reaches_the_reset_hook_exactly_once(adapter):
    arm, _robot, _wrapped, calls = adapter
    arm.reset(REST, T0 + 0.1)
    assert len(calls) == 1
    assert arm.reset_count == 1
    with pytest.raises(api.DoubleReset):
        arm.reset(REST, T0 + 0.2)
    assert len(calls) == 1


def test_an_unknown_named_target_never_reaches_the_wrapped_controller(adapter):
    arm, _robot, wrapped, _calls = adapter
    named = {name: value for name, value in zip(NAMES, REST)}
    named["joint9"] = 0.0
    before = len(wrapped.commands)
    with pytest.raises(api.UnknownJointTarget):
        arm.command_joint_targets(named, T0 + 0.1)
    assert len(wrapped.commands) == before
    assert arm.write_count == 0


def test_hold_then_a_command_is_refused_and_nothing_reaches_the_wrapped_controller(adapter):
    arm, _robot, wrapped, _calls = adapter
    arm.command_joint_targets(REST, T0 + 0.1)
    assert len(wrapped.commands) == 1
    arm.settle(T0 + 0.2)
    arm.hold(T0 + 0.3, reason="R10_F1_TEST")
    assert arm.held is not None and arm.held.reason == "R10_F1_TEST"
    with pytest.raises(api.HoldEngaged):
        arm.command_joint_targets(REST, T0 + 0.4)
    assert len(wrapped.commands) == 1
    assert arm.write_count == 1
