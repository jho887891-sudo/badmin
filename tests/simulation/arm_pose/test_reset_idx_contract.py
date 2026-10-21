# -*- coding: utf-8 -*-
"""P5B - reset_idx() must write an in-limit pose for the arm it is actually resetting.

The bug this pins
    reset_idx() built its default joint target from one arm-agnostic table and wrote it without
    looking at which arm the scene had loaded. On the reBot B601-RS (joint3 = [0, 3.14] rad) the
    hardcoded joint3 = -1.5 rad is outside the limits, and the three in-tree callers
    (scripts/t06_reset_local.py, t07_local.py, t08_local.py) never pass joint_pos, so every
    --num_envs 1 run on the formal arm went through it. P5 recorded it as F1; this file is the
    regression contract.

HOW IT RUNS WITHOUT ISAAC SIM
    reset.py is imported against a numpy-backed torch stand-in and handed fake robot/shuttle
    objects that record what would have been written. No GPU, no PhysX, no pxr.
"""
from __future__ import annotations

import struct

import pytest

from pose_data import LEGACY_PIPER_REST, PIPER_JOINT_NAMES, REBOT_RS_JOINT_NAMES


def _bits(value):
    return struct.pack("<d", float(value))


def _assert_inside_usd_limits(written, usd_limits, rest_pose):
    """The one assertion this whole file exists for: nothing written may sit outside the limits."""
    assert set(written) == set(usd_limits), (sorted(written), sorted(usd_limits))
    for name, value in written.items():
        lower, upper, unit = usd_limits[name]
        assert lower < value < upper, (
            "reset wrote %s = %r, outside the recorded USD limits [%r, %r] %s"
            % (name, value, lower, upper, unit))
        limit = rest_pose.JointLimit(
            name=name, kind="prismatic" if unit == "m" else "revolute",
            lower=lower, upper=upper, unit=unit, source="pxr export")
        assert rest_pose.margin(limit, value) >= rest_pose.required_margin(limit), (name, value)


# --------------------------------------------------------------------------- 1. the fixed behaviour
def test_reset_on_the_rs_arm_writes_an_in_limit_model_aware_pose(
        reset_mod, rs_robot, shuttle, env_ids, usd_limits, rest_pose):
    reset_mod.reset_idx(env_ids, rs_robot, shuttle)
    written = rs_robot.joint_pos_by_name()
    _assert_inside_usd_limits(written, usd_limits, rest_pose)
    # The old table's joint3 = -1.5 is gone; the RS rest pose lifts the elbow instead.
    assert written["joint3"] == pytest.approx(1.0), written
    assert written["joint2"] == pytest.approx(0.3), written


def test_reset_on_the_rs_arm_does_not_write_the_piper_pose(
        reset_mod, rs_robot, shuttle, env_ids):
    reset_mod.reset_idx(env_ids, rs_robot, shuttle)
    written = rs_robot.joint_pos_by_name()
    assert _bits(written["joint3"]) != _bits(LEGACY_PIPER_REST["joint3"]), written


def test_reset_on_the_piper_arm_is_bit_identical_to_the_stage0_pose(
        reset_mod, piper_robot, shuttle, env_ids):
    """The PiPER regression: these six doubles may not move by one ULP."""
    reset_mod.reset_idx(env_ids, piper_robot, shuttle)
    written = piper_robot.joint_pos_by_name()
    assert tuple(written) == PIPER_JOINT_NAMES
    for name, expected in LEGACY_PIPER_REST.items():
        assert _bits(written[name]) == _bits(expected), (name, written[name], expected)


def test_reset_writes_zero_velocities_and_the_canonical_spawn(
        reset_mod, rs_robot, shuttle, env_ids):
    reset_mod.reset_idx(env_ids, rs_robot, shuttle)
    assert rs_robot.joint_state_writes[-1]["joint_vel"] == [[0.0] * len(REBOT_RS_JOINT_NAMES)]
    # reset_idx() casts the spawn to float32 (the sim dtype), so compare with a tolerance.
    for got, expected in zip(shuttle.root_pose_writes[-1][0][:3], (1.20, 0.00, 1.80)):
        assert got == pytest.approx(expected, abs=1e-6), shuttle.root_pose_writes[-1][0][:3]


def test_reset_offsets_the_spawn_by_the_environment_origin(
        reset_mod, rs_robot, shuttle, usd_limits, rest_pose):
    import fake_torch

    env_origins = fake_torch.FakeTensor([[10.0, 20.0, 30.0]])
    reset_mod.reset_idx(fake_torch.FakeEnvIds([0]), rs_robot, shuttle, env_origins=env_origins)
    for got, expected in zip(shuttle.root_pose_writes[-1][0][:3], (11.20, 20.00, 31.80)):
        assert got == pytest.approx(expected, abs=1e-6), shuttle.root_pose_writes[-1][0][:3]
    _assert_inside_usd_limits(rs_robot.joint_pos_by_name(), usd_limits, rest_pose)


# --------------------------------------------------------------------------- 2. fail loudly
def test_unknown_joint_names_raise_and_nothing_is_written(reset_mod, shuttle, env_ids):
    import fake_torch

    robot = fake_torch.FakeRobot(("joint1", "joint2", "shoulder_pan", "joint4", "joint5", "joint6"))
    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, robot, shuttle)
    assert "shoulder_pan" in str(excinfo.value), str(excinfo.value)
    assert robot.joint_state_writes == [], robot.joint_state_writes
    assert shuttle.root_pose_writes == []


def test_a_piper_shaped_robot_with_an_extra_dof_raises(reset_mod, shuttle, env_ids):
    """A name set that matches no known model is unknown, even if it is 'almost PiPER'."""
    import fake_torch

    robot = fake_torch.FakeRobot(PIPER_JOINT_NAMES + ("joint7",))
    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, robot, shuttle)
    assert "joint7" in str(excinfo.value), str(excinfo.value)
    assert robot.joint_state_writes == []


def test_an_explicit_out_of_limit_joint_pos_raises_before_writing(
        reset_mod, rs_robot, shuttle, env_ids):
    """The precise defect: a caller (or a stale constant) asking for joint3 = -1.5 on the RS arm."""
    import fake_torch

    illegal = [[LEGACY_PIPER_REST[name] for name in REBOT_RS_JOINT_NAMES[:-2]] + [0.025, 0.03575]]
    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle, joint_pos=fake_torch.FakeTensor(illegal))
    message = str(excinfo.value)
    assert "joint3" in message and "-1.5" in message, message
    assert rs_robot.joint_state_writes == [], rs_robot.joint_state_writes


def test_a_joint_pos_containing_nan_raises_before_writing(
        reset_mod, rs_robot, shuttle, env_ids):
    """NaN is invisible to a plain range check: NaN < lo and NaN > hi are BOTH False.

    Without the reflexive test this target would have been written, and PhysX would have propagated
    the NaN through the whole articulation state instead of failing at the reset.
    """
    import fake_torch

    nan = float("nan")
    poisoned = [[0.0, 0.5, 1.2, 0.0, nan, 0.0, 0.025, 0.03575]]
    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle, joint_pos=fake_torch.FakeTensor(poisoned))
    message = str(excinfo.value)
    assert "nan" in message.lower(), message
    assert "joint5" in message, message
    assert rs_robot.joint_state_writes == [], rs_robot.joint_state_writes


def test_an_explicit_in_limit_joint_pos_is_written_unchanged(
        reset_mod, rs_robot, shuttle, env_ids):
    import fake_torch

    legal = [[0.0, 0.5, 1.2, 0.0, 0.0, 0.0, 0.025, 0.03575]]
    reset_mod.reset_idx(env_ids, rs_robot, shuttle, joint_pos=fake_torch.FakeTensor(legal))
    assert rs_robot.joint_pos_by_name() == {
        name: value for name, value in zip(REBOT_RS_JOINT_NAMES, legal[0])}


def test_an_explicit_joint_pos_with_the_wrong_width_raises(
        reset_mod, rs_robot, shuttle, env_ids):
    import fake_torch

    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle,
                            joint_pos=fake_torch.FakeTensor([[0.0] * 6]))
    assert "6" in str(excinfo.value) and "8" in str(excinfo.value), str(excinfo.value)
    assert rs_robot.joint_state_writes == []


def test_a_num_joints_that_disagrees_with_the_names_raises(reset_mod, shuttle, env_ids):
    import fake_torch

    robot = fake_torch.FakeRobot(REBOT_RS_JOINT_NAMES, num_joints=6)
    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, robot, shuttle)
    assert "num_joints" in str(excinfo.value), str(excinfo.value)
    assert robot.joint_state_writes == []


def test_an_explicit_model_key_that_contradicts_the_names_raises(
        reset_mod, rs_robot, shuttle, env_ids):
    with pytest.raises(Exception) as excinfo:
        reset_mod.reset_idx(env_ids, rs_robot, shuttle, robot_key="piper")
    assert "piper" in str(excinfo.value), str(excinfo.value)
    assert rs_robot.joint_state_writes == []


# --------------------------------------------------------------------------- 3. preserved behaviour
def test_no_envs_is_a_noop(reset_mod, rs_robot, shuttle):
    import fake_torch

    assert reset_mod.reset_idx(fake_torch.FakeEnvIds([]), rs_robot, shuttle) is None
    assert rs_robot.joint_state_writes == []
    assert shuttle.root_pose_writes == []


def test_explicit_states_still_win_over_the_canonical_spawn(reset_mod, rs_robot, shuttle, env_ids):
    import fake_torch

    local = fake_torch.FakeTensor([[1.0, 2.0, 3.0]])
    quat = fake_torch.FakeTensor([[1.0, 0.0, 0.0, 0.0]])
    vel = fake_torch.FakeTensor([[0.1, 0.2, 0.3]])
    ang = fake_torch.FakeTensor([[0.4, 0.5, 0.6]])
    reset_mod.reset_idx(env_ids, rs_robot, shuttle, states=(local, quat, vel, ang))
    assert shuttle.root_pose_writes[-1][0][:3] == [1.0, 2.0, 3.0]
    assert shuttle.root_velocity_writes[-1][0] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
