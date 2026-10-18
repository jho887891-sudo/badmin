# -*- coding: utf-8 -*-
"""reBot B601-RS formal arm: the limit table must be RS-specific and provenance-gated.

Values come from the OFFICIAL Seeed reBot-DevArm asset (ReBot_Arm_RS.urdf, commit f01a1dc) and the
official Seeed product pages for the RobStride RS00 / RS06 motors. Nothing here may be inherited from
PiPER or from the B601-DM candidate line.
"""
from __future__ import annotations

import pytest


def test_module_exists_and_declares_rs_identity():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    assert jl.ROBOT_NAME == "ReBot_Arm_RS"
    assert jl.IDENTITY == "formal_arm"
    assert jl.MODEL == "reBot B601-RS"


def test_arm_is_six_dof_plus_gripper():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    arm = [n for n in jl.JOINT_LIMITS if n.startswith("joint")]
    assert len(arm) == 6, "the official page states 6 DOF + 1 gripper; do not assume another count"
    assert set(jl.MOTOR_ASSIGNMENT) >= set(arm)


def test_motor_mix_matches_the_official_bom():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    models = [jl.JOINT_LIMITS[n].motor for n in jl.JOINT_LIMITS]
    assert models.count("RS06") == 3
    assert models.count("RS00") == 4
    assert "DM" not in "".join(models)


def test_angle_ranges_are_the_official_urdf_values():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    expected = {
        "joint1": (-2.8, 2.8),
        "joint2": (0.0, 3.14),
        "joint3": (0.0, 3.14),
        "joint4": (-1.57, 1.57),
        "joint5": (-1.57, 1.57),
        "joint6": (-3.14, 3.14),
    }
    for name, (lo, hi) in expected.items():
        spec = jl.JOINT_LIMITS[name]
        assert spec.angle.lower == pytest.approx(lo)
        assert spec.angle.upper == pytest.approx(hi)


def test_angle_is_urdf_provenance_not_verified_hardware():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    for name, spec in jl.JOINT_LIMITS.items():
        if name.startswith("joint"):
            assert spec.angle.provenance is jl.Provenance.OFFICIAL_URDF
            with pytest.raises(jl.LimitNotReal):
                jl.angle_range(name, require_real=True)


def test_peak_torque_is_confirmed_real_and_cross_checked():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    assert jl.MOTORS["RS00"].peak_torque_nm == pytest.approx(14.0)
    assert jl.MOTORS["RS06"].peak_torque_nm == pytest.approx(36.0)
    assert jl.MOTORS["RS00"].reduction_ratio == 10
    assert jl.MOTORS["RS06"].reduction_ratio == 9
    for name, spec in jl.JOINT_LIMITS.items():
        if not name.startswith("joint"):
            continue  # gripper effort is a FORCE in newtons, not a motor torque
        assert spec.torque.provenance is jl.Provenance.CONFIRMED_REAL
        assert spec.torque.value == pytest.approx(jl.MOTORS[spec.motor].peak_torque_nm)
        assert spec.urdf_effort == pytest.approx(jl.MOTORS[spec.motor].peak_torque_nm)


def test_gripper_effort_is_a_force_not_a_motor_torque():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    for name in ("gripper_joint1", "gripper_joint2"):
        spec = jl.JOINT_LIMITS[name]
        assert spec.torque.provenance is jl.Provenance.UNKNOWN
        assert spec.urdf_effort == pytest.approx(500.0)


def test_velocity_is_urdf_only_and_refuses_to_back_a_capability_claim():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    for name in ("joint1", "joint4"):
        spec = jl.JOINT_LIMITS[name]
        assert spec.velocity.provenance is jl.Provenance.OFFICIAL_URDF
        assert spec.velocity.usable_for_capability is False
        with pytest.raises(jl.LimitNotReal):
            jl.joint_velocity(name, require_real=True)


def test_gripper_is_prismatic_and_hardware_coupled():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    g1 = jl.JOINT_LIMITS["gripper_joint1"]
    g2 = jl.JOINT_LIMITS["gripper_joint2"]
    assert g1.kind == "prismatic" and g2.kind == "prismatic"
    assert g1.angle.upper == pytest.approx(0.05)
    assert g2.angle.upper == pytest.approx(0.0715)
    assert jl.GRIPPER_FINGERS_HARDWARE_COUPLED is True


def test_masses_sum_matches_the_urdf_and_the_spec_sheet_is_not_the_model():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    assert sum(jl.LINK_MASSES_KG.values()) == pytest.approx(6.0085, abs=1e-4)
    assert jl.SPEC["weight_kg"] == pytest.approx(6.7)
    assert jl.SPEC["payload_kg"] == pytest.approx(2.5)
    assert jl.SPEC["reach_m"] == pytest.approx(0.754)
    assert jl.SPEC["dof"] == 6


def test_asset_provenance_is_pinned():
    from src.simulation.rebot_b601_rs import joint_limits as jl

    assert jl.ASSET["repo"] == "https://github.com/Seeed-Projects/reBot-DevArm"
    assert jl.ASSET["commit"] == "f01a1dc189ecb74b5435a336a68b287c146c1b60"
    assert len(jl.ASSET["urdf_sha256"]) == 64
    assert "LICENSE" in jl.ASSET["license"]


def test_dm_candidate_line_is_not_reused():
    """Guard rail: the B601-DM figures (reach 767 mm, DM4340/DM4310 motors) must not leak in."""
    from src.simulation.rebot_b601_rs import joint_limits as jl

    assert jl.SPEC["reach_m"] != pytest.approx(0.767), "0.767 is the B601-DM reach"
    assert all("DM" not in m for m in jl.MOTORS)
