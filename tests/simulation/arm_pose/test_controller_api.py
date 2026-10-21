# -*- coding: utf-8 -*-
"""P5-09 - the reusable B601-RS controller: state, targets, reset, HOLD, and the IK hook.

What is proven here
    * construction resolves the arm from the RUNTIME joint-name list (P5-04 map, P5B model table)
      and refuses an unknown arm, an extra DOF or a duplicated DOF;
    * state is read and addressed by NAME, and a missing/short state is refused, never padded;
    * targets are validated by the P5-05 gate and reach the sim in runtime order, including from a
      {name: value} mapping, so a table order can never leak into a joint_pos vector;
    * reset reuses the P5B reset path unchanged - the last test in this file drives the REAL
      reset_idx() through the numpy torch shim (fake_torch.py), so the wiring is not a mock story;
    * HOLD freezes the accepted target and refuses commands/resets until released;
    * solve_ik() raises instead of returning a plausible-looking pose.

No Isaac Sim, no GPU, no pxr, no torch install: the only torch-like object in play is the shim.
"""
from __future__ import annotations

import numpy as np
import pytest

from pose_data import PIPER_JOINT_NAMES, REBOT_RS_JOINT_NAMES as NAMES

REST = (0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575)
DT = 1.0 / 60.0


class ShimData:
    def __init__(self, joint_pos, joint_vel=None):
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel


class ShimRobot:
    """A numpy-only articulation stand-in: joint names + a data block + a write log."""

    def __init__(self, names, joint_pos=None, joint_vel=None, num_joints=None):
        self.joint_names = list(names)
        self.num_joints = len(self.joint_names) if num_joints is None else num_joints
        self.data = ShimData(joint_pos, joint_vel)
        self.writes = []

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        self.writes.append({"joint_pos": joint_pos, "joint_vel": joint_vel, "env_ids": env_ids})

    @property
    def last_joint_pos(self):
        assert self.writes, "nothing was written"
        return self.writes[-1]["joint_pos"]


class NoDataRobot(ShimRobot):
    def __init__(self, names):
        super().__init__(names)
        self.data = None


def _recording_resetter(calls):
    def _reset(env_ids, robot, shuttle, joint_pos=None):
        calls.append({"env_ids": env_ids, "robot": robot, "shuttle": shuttle,
                      "joint_pos": joint_pos})
        return None

    return _reset


def _identity_tensor(rows):
    """A tensor factory for tests that never reach the sim: keeps the payload inspectable."""
    return [list(row) for row in rows]


@pytest.fixture
def controller_mod():
    import badminton_scene.controller as module

    return module


@pytest.fixture
def rest_pose():
    import badminton_scene.rest_pose as module

    return module


# --------------------------------------------------------------------------- 1. construction
def test_the_controller_resolves_the_arm_from_the_runtime_names(controller_mod):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    assert arm.model == "rebot_b601_rs"
    assert arm.dof_count == 8
    assert arm.dof_index == {name: position for position, name in enumerate(NAMES)}
    described = arm.describe()
    assert described["dof_count"] == 8 and described["holding"] is False
    assert set(described["position_limits"]) == set(NAMES)
    assert "POLICY CAP" in described["envelope_source"]


def test_a_second_arm_model_is_recognised_by_its_own_names(controller_mod):
    arm = controller_mod.B601RSController(ShimRobot(PIPER_JOINT_NAMES, np.zeros(6)))
    assert arm.model == "piper" and arm.dof_count == 6


def test_an_unknown_arm_raises_instead_of_being_treated_as_the_rs(controller_mod, rest_pose):
    names = NAMES[:-2] + ("finger_a", "finger_b")
    with pytest.raises(rest_pose.UnknownRobotModel) as excinfo:
        controller_mod.B601RSController(ShimRobot(names, np.zeros(8)))
    assert "finger_a" in str(excinfo.value), str(excinfo.value)


def test_a_duplicated_runtime_dof_raises(controller_mod):
    import badminton_scene.dof_index as dof_index

    names = NAMES[:-1] + ("joint1",)
    with pytest.raises(dof_index.DuplicateJointName) as excinfo:
        controller_mod.B601RSController(ShimRobot(names, np.zeros(8)))
    assert "joint1" in str(excinfo.value)


# --------------------------------------------------------------------------- 2. state
def test_state_is_read_by_name_not_by_position(controller_mod):
    values = np.array(REST)
    arm = controller_mod.B601RSController(ShimRobot(NAMES, values, np.zeros(8)))
    state = arm.read_state()
    assert state.joint_pos == REST
    assert state.by_name()["joint3"] == 1.0
    assert state["gripper_joint2"] == 0.03575
    assert state.velocity_by_name()["joint1"] == 0.0
    assert state.joint_vel == (0.0,) * 8


def test_a_state_without_velocity_reports_none_rather_than_zero(controller_mod):
    """A fabricated zero velocity is indistinguishable from a measured one downstream."""
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    assert arm.read_state().joint_vel is None
    assert arm.read_state().velocity_by_name() is None


def test_a_short_state_raises(controller_mod):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.zeros(6)))
    with pytest.raises(controller_mod.StateUnavailable) as excinfo:
        arm.read_state()
    assert "6" in str(excinfo.value) and "8" in str(excinfo.value)


def test_a_robot_without_state_raises(controller_mod):
    arm = controller_mod.B601RSController(NoDataRobot(NAMES))
    with pytest.raises(controller_mod.StateUnavailable):
        arm.read_state()


def test_an_unknown_joint_lookup_in_the_state_raises(controller_mod, rest_pose):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    with pytest.raises(rest_pose.UnknownJoint):
        arm.read_state()["joint9"]


# --------------------------------------------------------------------------- 3. targets
def test_targets_are_accepted_and_latched_in_runtime_order(controller_mod):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    target = tuple(value + 0.01 for value in REST)
    accepted = arm.set_joint_targets(target)
    assert accepted == pytest.approx(target)
    assert arm.commands[-1][0] == accepted
    assert arm.gate.last_valid == accepted


def test_a_named_target_is_ordered_by_the_runtime_index(controller_mod):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    shuffled = dict(reversed(list(zip(NAMES, REST))))  # insertion order is not the DOF order
    accepted = arm.set_joint_targets(shuffled)
    assert accepted == pytest.approx(REST)
    assert arm.dof_index["joint1"] == 0 and accepted[0] == 0.0


def test_a_named_target_with_an_unknown_joint_raises(controller_mod, safety_errors):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    named = {name: value for name, value in zip(NAMES, REST)}
    named["racket"] = 0.0
    with pytest.raises(safety_errors.UnknownCommandJoint) as excinfo:
        arm.set_joint_targets(named)
    assert excinfo.value.joint == "racket"


def test_an_out_of_limit_target_raises_with_the_joint(controller_mod, safety_errors):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    bad = list(REST)
    bad[2] = -1.5
    with pytest.raises(safety_errors.PositionLimitViolation) as excinfo:
        arm.set_joint_targets(bad)
    assert excinfo.value.joint == "joint3" and excinfo.value.value == -1.5


# --------------------------------------------------------------------------- 4. reset
def test_reset_forwards_the_validated_target_in_runtime_order(controller_mod):
    calls = []
    robot = ShimRobot(NAMES, np.array(REST))
    arm = controller_mod.B601RSController(robot, reset_fn=_recording_resetter(calls),
                                         tensor_factory=_identity_tensor)
    arm.reset(REST, env_ids=[0])
    assert len(calls) == 1 and calls[0]["env_ids"] == [0]
    assert calls[0]["joint_pos"] == [list(REST)]
    assert arm.gate.last_valid == pytest.approx(REST), "the rate baseline follows the reset"


def test_reset_from_a_mapping_never_leaks_the_table_order(controller_mod):
    calls = []
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)),
                                         reset_fn=_recording_resetter(calls),
                                         tensor_factory=_identity_tensor)
    named = dict(reversed(list(zip(NAMES, REST))))
    arm.reset(named)
    assert calls[-1]["joint_pos"] == [list(REST)]


def test_a_reset_through_the_real_reset_idx_is_driven_by_the_shim(
        controller_mod, reset_mod, rs_robot, shuttle, env_ids, usd_limits, rest_pose):
    """END-TO-END: controller -> P5B reset_idx -> fake PhysX write, with no torch installed."""
    arm = controller_mod.B601RSController(
        rs_robot, shuttle=shuttle, reset_fn=reset_mod.reset_idx,
        tensor_factory=reset_mod.torch.tensor)
    arm.reset_to_rest(env_ids=env_ids)

    written = rs_robot.joint_pos_by_name()
    assert set(written) == set(NAMES)
    for name, value in written.items():
        lower, upper, _unit = usd_limits[name]
        assert lower < value < upper, (name, value, lower, upper)
    assert written["joint3"] == pytest.approx(1.0) and written["joint2"] == pytest.approx(0.3)
    assert arm.gate.last_valid == pytest.approx([written[name] for name in NAMES])
    limit = rest_pose.JointLimit(name="joint3", kind="revolute", lower=0.0, upper=3.14,
                                 unit="rad", source="pxr export")
    assert rest_pose.margin(limit, written["joint3"]) >= rest_pose.required_margin(limit)


def test_a_reset_target_out_of_limits_never_reaches_reset_idx(controller_mod):
    calls = []
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)),
                                         reset_fn=_recording_resetter(calls),
                                         tensor_factory=_identity_tensor)
    illegal = list(REST)
    illegal[2] = -1.5
    with pytest.raises(Exception) as excinfo:
        arm.reset(illegal)
    assert "joint3" in str(excinfo.value)
    assert calls == []


# --------------------------------------------------------------------------- 5. HOLD
def test_hold_freezes_the_last_target_and_refuses_commands(controller_mod, safety_errors):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    accepted = arm.set_joint_targets(REST)
    held = arm.hold()
    assert held is accepted and arm.held_target is held and arm.holding is True
    nudged = tuple(value + 0.001 for value in REST)
    with pytest.raises(safety_errors.HoldActive):
        arm.set_joint_targets(nudged)
    assert arm.held_target == held
    arm.release()
    assert arm.holding is False and arm.held_target is None
    # A released gate accepts again; dt is passed explicitly so the finger caps are not tripped.
    assert arm.set_joint_targets(nudged, dt=1.0) == pytest.approx(nudged)


def test_a_reset_is_refused_while_holding(controller_mod, safety_errors):
    calls = []
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)),
                                         reset_fn=_recording_resetter(calls),
                                         tensor_factory=_identity_tensor)
    arm.set_joint_targets(REST)
    arm.hold()
    with pytest.raises(safety_errors.HoldActive):
        arm.reset(REST)
    assert calls == []


# --------------------------------------------------------------------------- 6. the IK hook
def test_solve_ik_raises_not_implemented_and_returns_nothing(controller_mod):
    arm = controller_mod.B601RSController(ShimRobot(NAMES, np.array(REST)))
    with pytest.raises(NotImplementedError) as excinfo:
        arm.solve_ik([0.4, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])
    assert isinstance(excinfo.value, controller_mod.IkNotImplemented)
    assert "no IK solver" in str(excinfo.value)
    # Refusing must not have side effects on the command state.
    assert arm.gate.last_valid is None and arm.commands == []


@pytest.fixture
def safety_errors():
    import badminton_scene.safety as module

    return module
