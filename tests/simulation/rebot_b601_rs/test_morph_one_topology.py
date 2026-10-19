# -*- coding: utf-8 -*-
"""Topology and parameter-discipline tests for the Morph One four-steer/four-drive interface.

These are pure CPU: they guard the *interface* the physical assembly must implement. They deliberately
say nothing about the real geometry, which phase A could not classify.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene import morph_one_chassis as mc  # noqa: E402


def test_exactly_four_wheel_modules_with_one_steer_and_one_drive_each():
    assert mc.WHEEL_MODULES == ("fl", "fr", "rl", "rr")
    assert mc.JOINT_KINDS == ("steer", "drive")
    assert len(mc.SEMANTIC_JOINTS) == 8
    for module in mc.WHEEL_MODULES:
        assert module + "_steer" in mc.SEMANTIC_JOINTS
        assert module + "_drive" in mc.SEMANTIC_JOINTS


def test_joint_order_is_canonical_and_matches_the_project_vocabulary():
    """The order must agree with the project's existing STEER_DRIVE_JOINT_NAMES builder."""
    expected = tuple(
        str(wheel) .lower() + "_" + kind
        for wheel in ("FL", "FR", "RL", "RR")
        for kind in ("steer", "drive")
    )
    assert mc.SEMANTIC_JOINTS == expected
    assert mc.joint_index_map()["fl_steer"] == 0
    assert mc.joint_index_map()["rr_drive"] == 7


def test_topology_validation_passes_and_rejects_duplicates(monkeypatch):
    mc.validate_topology()
    monkeypatch.setattr(mc, "SEMANTIC_JOINTS", mc.SEMANTIC_JOINTS + ("fl_steer",))
    with pytest.raises(AssertionError):
        mc.validate_topology()


def test_no_parameter_claims_to_be_measured():
    """Phase A classified no geometry, so nothing may be VERIFIED_MEASURED."""
    for name, p in mc.CHASSIS_PARAMETERS.items():
        assert p.status != mc.ChassisParameterStatus.VERIFIED_MEASURED, name


def test_geometric_quantities_stay_unresolved():
    for name in ("wheel_radius_m", "wheel_width_m", "wheel_base_m", "track_width_m",
                 "steer_axis_height_m", "total_mass_kg"):
        p = mc.CHASSIS_PARAMETERS[name]
        assert p.status == mc.ChassisParameterStatus.REQUIRES_MEASUREMENT, name
        assert p.value is None, name
        assert name in mc.unresolved_parameters()


def test_temp_baselines_are_labelled_as_temp_and_point_at_the_spec():
    for name in ("base_length_m", "base_width_m"):
        p = mc.CHASSIS_PARAMETERS[name]
        assert p.status == mc.ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY, name
        assert p.value is not None
        assert "BADMINTON_ROBOT" in p.source


def test_parameter_status_is_enforced_at_construction():
    with pytest.raises(ValueError):
        mc.ChassisParameter("x", None, "m", "PROBABLY_TRUE", "guess")
    with pytest.raises(ValueError):
        mc.ChassisParameter("x", 1.0, "m", mc.ChassisParameterStatus.REQUIRES_MEASUREMENT, "guess")
    with pytest.raises(ValueError):
        mc.ChassisParameter("x", 1.0, "m", mc.ChassisParameterStatus.VERIFIED_MEASURED, "")


def test_required_frames_are_declared_so_transforms_cannot_be_invented():
    for frame in ("chassis_base", "fl_wheel_center", "rr_wheel_center", "arm_mount"):
        assert frame in mc.REQUIRED_FRAMES
    assert len(mc.REQUIRED_FRAMES) == len(set(mc.REQUIRED_FRAMES))


@pytest.mark.parametrize("status,value", [
    (mc.ChassisParameterStatus.VERIFIED_MEASURED, None),
    (mc.ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY, None),
    (mc.ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY, float("nan")),
    (mc.ChassisParameterStatus.VERIFIED_MEASURED, float("inf")),
    (mc.ChassisParameterStatus.VERIFIED_MEASURED, -0.1),
    (mc.ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY, 0.0),
])
def test_parameter_cannot_claim_numeric_fidelity_without_finite_positive_measurement(status, value):
    with pytest.raises(ValueError):
        mc.ChassisParameter("wheel_radius_m", value, "m", status, "explicit source")


def test_parameter_must_have_unit_and_source():
    with pytest.raises(ValueError):
        mc.ChassisParameter("wheel_radius_m", 0.1, "", mc.ChassisParameterStatus.VERIFIED_MEASURED, "test")
    with pytest.raises(ValueError):
        mc.ChassisParameter("wheel_radius_m", 0.1, "m", mc.ChassisParameterStatus.VERIFIED_MEASURED, "")
