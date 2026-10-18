# -*- coding: utf-8 -*-
"""The scene layer and the robot config must describe the SAME B601-RS, and the rest pose must be legal.

Two sources of truth for one arm is how the PiPER numbers leaked around before. These tests make the
scene layout, the robot cfg and the official limit table agree, and they refuse a rest pose that sits
outside the official joint ranges.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.scene_layout import REBOT_RS  # noqa: E402
from robots.badminton_robot.badminton_robot_cfg import make_default_robot_cfg  # noqa: E402
from src.simulation.rebot_b601_rs.joint_limits import ARM_JOINTS, JOINT_LIMITS  # noqa: E402


def test_scene_layout_and_robot_cfg_describe_the_same_arm():
    cfg = make_default_robot_cfg()
    assert REBOT_RS["rest_joint_pos"] == cfg.rebot_rs.rest_joint_pos
    assert REBOT_RS["usd"] == cfg.rebot_rs.usd_path
    assert tuple(REBOT_RS["mount_rel"]) == tuple(cfg.rebot_rs.mount_translation.value)


def test_rest_pose_is_inside_the_official_joint_ranges():
    for name, q in REBOT_RS["rest_joint_pos"].items():
        spec = JOINT_LIMITS[name]
        assert spec.angle.lower <= q <= spec.angle.upper, name + " rest pose outside the URDF range"


def test_the_piper_rest_pose_would_be_illegal_on_this_arm():
    """Evidence, not opinion: joint3 is [0, 3.14] on the B601-RS, so the PiPER -1.5 is invalid."""
    spec = JOINT_LIMITS["joint3"]
    assert -1.5 < spec.angle.lower
    assert REBOT_RS["rest_joint_pos"]["joint3"] >= spec.angle.lower


def test_rest_pose_covers_exactly_the_six_arm_joints():
    assert set(REBOT_RS["rest_joint_pos"]) == set(ARM_JOINTS)
