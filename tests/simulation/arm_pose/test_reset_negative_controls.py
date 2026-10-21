# -*- coding: utf-8 -*-
"""P5-06 - reset negative controls: every malformed target must RAISE, never be clamped.

The distinction this file pins
    reset_idx(joint_pos=None) is a DOCUMENTED sentinel (P5B): it means 'write this arm's own rest
    pose'. It is not a clamp and not a default for garbage - it is an explicit branch that was
    validated against the USD. Everything else - NaN, an infinity, an unknown joint name, an
    out-of-limit value, the wrong width, an EMPTY payload, or a controller reset with no target at
    all - must raise, and must raise BEFORE anything is written to the sim.

    The exception types are not new here: they come from badminton_scene.rest_pose (re-exported by
    reset.py), so a caller can catch one family of errors everywhere. The tests assert the exact
    class, which fails if anyone redefines a look-alike.

How it runs without Isaac Sim
    reset.py is imported against the numpy-backed torch stand-in (conftest reset_mod fixture); the
    USD limits come from the committed pxr export.
"""
from __future__ import annotations

import pytest

from pose_data import REBOT_RS_JOINT_NAMES as NAMES

REST = [0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575]


def _tensor(fake_torch, rows):
    return fake_torch.FakeTensor(rows)


# --------------------------------------------------------------------------- 1. reset_idx itself
def test_a_nan_target_raises_and_writes_nothing(reset_mod, rs_robot, shuttle, env_ids, rest_pose):
    import fake_torch

    poisoned = [list(REST)]
    poisoned[0][4] = float("nan")
    with pytest.raises(rest_pose.RestPoseLimitViolation) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle, joint_pos=_tensor(fake_torch, poisoned))
    message = str(excinfo.value)
    assert "nan" in message.lower() and "joint5" in message, message
    assert type(excinfo.value) is rest_pose.RestPoseLimitViolation
    assert rs_robot.joint_state_writes == [], rs_robot.joint_state_writes


def test_an_infinite_target_raises_and_writes_nothing(reset_mod, rs_robot, shuttle, env_ids,
                                                     rest_pose):
    """inf passes the NaN test (inf == inf) but not the range test, so it must still raise."""
    import fake_torch

    poisoned = [list(REST)]
    poisoned[0][1] = float("inf")
    with pytest.raises(rest_pose.RestPoseLimitViolation) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle, joint_pos=_tensor(fake_torch, poisoned))
    assert "joint2" in str(excinfo.value), str(excinfo.value)
    assert rs_robot.joint_state_writes == []


def test_an_unknown_joint_name_raises_before_writing(reset_mod, shuttle, env_ids, rest_pose):
    import fake_torch

    robot = fake_torch.FakeRobot(("joint1", "joint2", "shoulder_pan", "joint4", "joint5", "joint6",
                                  "gripper_joint1", "gripper_joint2"))
    with pytest.raises(rest_pose.UnknownRobotModel) as excinfo:
        reset_mod.reset_idx(env_ids, robot, shuttle)
    assert "shoulder_pan" in str(excinfo.value), str(excinfo.value)
    assert robot.joint_state_writes == []


def test_an_out_of_limit_target_raises_before_writing(reset_mod, rs_robot, shuttle, env_ids,
                                                     rest_pose):
    import fake_torch

    illegal = list(REST)
    illegal[2] = -1.5  # the P5 F1 defect: joint3 on this arm is [0, 3.14]
    with pytest.raises(rest_pose.RestPoseLimitViolation) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle,
                            joint_pos=_tensor(fake_torch, [illegal]))
    message = str(excinfo.value)
    assert "joint3" in message and "-1.5" in message, message
    assert rs_robot.joint_state_writes == []


def test_a_wrong_width_target_raises(reset_mod, rs_robot, shuttle, env_ids, rest_pose):
    import fake_torch

    with pytest.raises(rest_pose.ResetPoseError) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle,
                            joint_pos=_tensor(fake_torch, [REST[:6]]))
    message = str(excinfo.value)
    assert "6" in message and "8" in message, message
    assert type(excinfo.value) is rest_pose.ResetPoseError
    assert rs_robot.joint_state_writes == []


@pytest.mark.parametrize("payload", [[[]], []])
def test_an_empty_target_raises(reset_mod, rs_robot, shuttle, env_ids, rest_pose, payload):
    """Zero columns (one empty row) and zero rows are both refusals, not no-ops."""
    import fake_torch

    with pytest.raises(rest_pose.ResetPoseError) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle,
                            joint_pos=_tensor(fake_torch, payload))
    assert "8" in str(excinfo.value), str(excinfo.value)
    assert rs_robot.joint_state_writes == []


# --------------------------------------------------------------------------- 2. the documented sentinel
def test_none_still_means_the_model_aware_rest_pose_not_a_clamp(reset_mod, rs_robot, shuttle,
                                                               env_ids, usd_limits):
    """Contrast case: None is an explicit branch (P5B), and its result is inside the USD limits.

    This is what keeps the negative controls honest: 'None is allowed' must be a documented sentinel
    with a validated pose, not the absence of a check.
    """
    reset_mod.reset_idx(env_ids, rs_robot, shuttle, joint_pos=None)
    written = rs_robot.joint_pos_by_name()
    assert written["joint3"] == pytest.approx(1.0) and written["joint2"] == pytest.approx(0.3)
    for name, value in written.items():
        lower, upper, _unit = usd_limits[name]
        assert lower < value < upper, (name, value, lower, upper)


# --------------------------------------------------------------------------- 3. the controller API
def test_controller_reset_with_no_target_raises_and_never_reaches_the_sim():
    """A silent 'reset to rest' on None is exactly the ambiguity P5B removed from reset_idx.

    reset_idx keeps None for compatibility with three in-tree scripts; the controller has an
    explicit reset_to_rest(), so None there is a caller bug and must raise.
    """
    import badminton_scene.controller as controller_mod
    from badminton_scene import rest_pose

    calls = []
    robot = _RecordingRobot(NAMES, calls)
    arm = controller_mod.B601RSController(robot, reset_fn=_record_into(calls))
    with pytest.raises(rest_pose.ResetPoseError) as excinfo:
        arm.reset(None)
    assert "None" in str(excinfo.value) or "no joint_pos" in str(excinfo.value), str(excinfo.value)
    assert calls == [], calls
    assert robot.writes == [], robot.writes


def test_controller_reset_with_an_empty_target_raises_and_never_reaches_the_sim():
    import badminton_scene.controller as controller_mod
    from badminton_scene import rest_pose

    calls = []
    robot = _RecordingRobot(NAMES, calls)
    arm = controller_mod.B601RSController(robot, reset_fn=_record_into(calls))
    for empty in ([], [[]]):
        with pytest.raises(rest_pose.ResetPoseError):
            arm.reset(empty)
    assert calls == [], calls


def test_controller_reset_rejects_an_out_of_limit_target_before_writing():
    import badminton_scene.controller as controller_mod
    from badminton_scene import rest_pose

    calls = []
    robot = _RecordingRobot(NAMES, calls)
    arm = controller_mod.B601RSController(robot, reset_fn=_record_into(calls))
    illegal = list(REST)
    illegal[2] = -1.5
    with pytest.raises(rest_pose.RestPoseLimitViolation) as excinfo:
        arm.reset(illegal)
    assert "joint3" in str(excinfo.value)
    assert calls == [], "the sim was written before the target was checked"


def test_controller_reset_rejects_a_nan_target_before_writing():
    import badminton_scene.controller as controller_mod
    from badminton_scene import rest_pose

    calls = []
    robot = _RecordingRobot(NAMES, calls)
    arm = controller_mod.B601RSController(robot, reset_fn=_record_into(calls))
    poisoned = list(REST)
    poisoned[5] = float("nan")
    with pytest.raises(rest_pose.RestPoseLimitViolation):
        arm.reset(poisoned)
    assert calls == []


def test_controller_reset_rejects_the_wrong_width_before_writing():
    import badminton_scene.controller as controller_mod
    from badminton_scene import rest_pose

    calls = []
    robot = _RecordingRobot(NAMES, calls)
    arm = controller_mod.B601RSController(robot, reset_fn=_record_into(calls))
    with pytest.raises(rest_pose.ResetPoseError) as excinfo:
        arm.reset(REST[:6])
    assert "6" in str(excinfo.value) and "8" in str(excinfo.value)
    assert calls == []


# --------------------------------------------------------------------------- helpers
class _RecordingRobot:
    """The minimum articulation surface the controller needs, plus a write log."""

    def __init__(self, names, _calls=None):
        self.joint_names = list(names)
        self.num_joints = len(self.joint_names)
        self.writes = []

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
        self.writes.append({"joint_pos": joint_pos, "joint_vel": joint_vel, "env_ids": env_ids})


def _record_into(calls):
    def _reset(env_ids, robot, shuttle, joint_pos=None):
        calls.append({"env_ids": env_ids, "robot": robot, "shuttle": shuttle,
                      "joint_pos": joint_pos})
        return None

    return _reset
