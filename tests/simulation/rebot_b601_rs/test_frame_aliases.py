# -*- coding: utf-8 -*-
"""The legacy PiPER frame names must resolve to the B601-RS frames without changing the transform.

Migration rule: the arm identity changes, the algebra does not. A name alias may never alter a
transform, whether it is queried as the source frame or as the base frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from robots.badminton_robot.badminton_robot_cfg import (  # noqa: E402
    AssetStatus,
    Param,
    make_default_robot_cfg,
)
from robots.badminton_robot.frames.robot_frames import (  # noqa: E402
    make_robot_frame_tree,
    quat_from_rpy,
)

BASE = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
LINK6 = ([0.10, 0.0, 0.45], quat_from_rpy(0.0, 0.0, 0.0))


def _tree(cfg=None):
    cfg = cfg or make_default_robot_cfg()
    tree = make_robot_frame_tree(
        cfg,
        base_pose_xyz_quat=BASE,
        link6_pose_in_piper_xyz_quat=LINK6,
        t_link6_tcp_xyz_quat=([0.0, 0.0, 0.02], [1.0, 0.0, 0.0, 0.0]),
        t_tcp_contact_xyz_quat=([0.0, 0.0, 0.50], [1.0, 0.0, 0.0, 0.0]),
    )
    return tree, cfg


def test_alias_as_base_frame_does_not_invert_the_transform():
    tree, _ = _tree()
    real = tree.get_transform("racket_tcp", "rebot_link6")
    alias = tree.get_transform("racket_tcp", "piper_link6")
    np.testing.assert_allclose(alias.translation, real.translation, atol=1e-12)
    np.testing.assert_allclose(alias.quaternion, real.quaternion, atol=1e-12)


def test_alias_as_source_frame_does_not_invert_the_transform():
    tree, _ = _tree()
    np.testing.assert_allclose(
        tree.get_transform("piper_link6", "robot_base").translation,
        tree.get_transform("rebot_link6", "robot_base").translation,
        atol=1e-12,
    )


def test_formal_arm_mount_is_the_b601_rs_mount():
    tree, cfg = _tree()
    np.testing.assert_allclose(
        tree.get_transform("rebot_base", "robot_base").translation,
        cfg.rebot_rs.mount_translation.value,
        atol=1e-12,
    )


def test_default_config_selects_the_formal_arm():
    cfg = make_default_robot_cfg()
    assert cfg.arm_model == "rebot_b601_rs"
    assert cfg.rebot_rs.robot_name == "ReBot_Arm_RS"
    assert cfg.rebot_rs.num_joints == 6
    assert "rebot_b601_rs" in cfg.rebot_rs.usd_path


def test_switching_back_to_piper_uses_the_historical_mount_and_no_rebot_frame():
    cfg = make_default_robot_cfg()
    cfg.arm_model = "piper"
    tree, _ = _tree(cfg)
    np.testing.assert_allclose(
        tree.get_transform("piper_base", "robot_base").translation,
        cfg.piper.mount_translation.value,
        atol=1e-12,
    )
    with pytest.raises(KeyError):
        tree.get_transform("rebot_base", "robot_base")
