# -*- coding: utf-8 -*-
"""P5B - the arm rest pose must belong to the arm that is loaded.

WHY THIS FILE EXISTS
    simulation/badminton_scene/reset.py used to hardcode one rest pose - the PiPER one, joint2 =
    1.5 / joint3 = -1.5 - and write it on whatever articulation the scene had loaded. On the formal
    arm (reBot B601-RS) joint3 is [0, 3.14] rad, so every reset that did not pass joint_pos wrote a
    limit-violating state. P5 recorded the defect (management/morph_one_rs/
    P5_ARM_STATIC_CONSISTENCY.md, F1) and pinned it with a canary; P5B fixes it and this file is the
    contract the fix has to keep:

      1. One rest pose per robot, keyed by JOINT NAME, never by DOF index (the USD traversal order
         is not the DOF order - P5 F2).
      2. Every value strictly inside that robot's limits, with a documented margin.
      3. PiPER keeps its Stage-0 values BIT FOR BIT, so existing PiPER runs do not move.
      4. Unknown robot names or out-of-limit values raise instead of being written.

WHAT THE LIMITS ARE CHECKED AGAINST
    The real B601-RS limits come from the committed pxr export arm_joint_facts.json (degrees in the
    asset, converted to radians once). pxr is not installed here, so this file reads the record; the
    sibling test_rs_rest_pose_against_usda.py re-reads the live asset wherever pxr exists.
"""
from __future__ import annotations

import struct

import pytest

from pose_data import LEGACY_PIPER_REST

#: The asset records revolute limits in degrees, so agreement is float32-conversion agreement
#: (P5 measured 8.3e-8 rad max). 1e-6 rad is that plus headroom, not a licence to move limits.
LIMIT_TOL_RAD = 1e-6
LIMIT_TOL_M = 1e-9


def _bits(value):
    """The exact IEEE-754 double, so "unchanged" means unchanged, not "close"."""
    return struct.pack("<d", float(value))


# --------------------------------------------------------------------------- 1. tables exist
def test_each_known_robot_has_a_name_keyed_rest_pose(rest_pose):
    assert set(rest_pose.REST_POSES) == {rest_pose.MODEL_PIPER, rest_pose.MODEL_REBOT_RS}
    piper = rest_pose.REST_POSES[rest_pose.MODEL_PIPER]
    rs = rest_pose.REST_POSES[rest_pose.MODEL_REBOT_RS]
    assert tuple(piper.joint_names) == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    assert set(rs.joint_names) == {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "gripper_joint1", "gripper_joint2"}
    # Keyed by name, not position: index the table with a name and it still answers.
    for spec in (piper, rs):
        assert set(spec.rest) == set(spec.joint_names)
        assert set(spec.limits) == set(spec.joint_names)
        for name in spec.joint_names:
            assert isinstance(spec.rest[name], float), (spec.key, name, type(spec.rest[name]))


def test_rest_pose_only_depends_on_joint_names_not_on_their_order(rest_pose):
    """The USD stage order (joint6, gripper_joint1, ...) must not change the pose."""
    forward = rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS)
    reversed_names = tuple(reversed(rest_pose.REST_POSES[rest_pose.MODEL_REBOT_RS].joint_names))
    backwards = rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS, reversed_names)
    assert dict(forward) == dict(backwards)
    assert tuple(backwards) == reversed_names  # order follows the request, values follow the name


# --------------------------------------------------------------------------- 2. margin rule
def test_every_rest_value_is_strictly_inside_its_limit_with_the_documented_margin(rest_pose):
    for spec in rest_pose.REST_POSES.values():
        for name in spec.joint_names:
            limit = spec.limits[name]
            value = spec.rest[name]
            required = rest_pose.required_margin(limit)
            got = rest_pose.margin(limit, value)
            assert got >= required, (
                "%s.%s rest=%r has margin %.6g in %s, needs >= %.6g (limits [%r, %r])"
                % (spec.key, name, value, got, limit.unit, required, limit.lower, limit.upper))
            # Strictly inside, not on the boundary.
            assert limit.lower < value < limit.upper, (spec.key, name, value)


def test_the_margin_rule_is_five_percent_or_50_milli_units_whichever_is_smaller(rest_pose):
    revolute = rest_pose.JointLimit(
        name="j", kind="revolute", lower=-1.0, upper=1.0, unit="rad", source="test")
    assert rest_pose.required_margin(revolute) == pytest.approx(rest_pose.MARGIN_ABS)
    narrow = rest_pose.JointLimit(
        name="f", kind="prismatic", lower=0.0, upper=0.05, unit="m", source="test")
    assert rest_pose.required_margin(narrow) == pytest.approx(0.05 * 0.05)


# --------------------------------------------------------------------------- 3. RS limits vs USD
def test_rebot_rs_rest_pose_is_inside_the_recorded_usd_limits_for_all_8_joints(
        rest_pose, usd_limits, usd_facts):
    assert usd_facts["asset"]["usda_sha256_before"] == usd_facts["asset"]["usda_sha256_after"], (
        "the pxr export itself says the asset was modified while it was read")
    assert set(usd_limits) == {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "gripper_joint1", "gripper_joint2"}, sorted(usd_limits)

    spec = rest_pose.REST_POSES[rest_pose.MODEL_REBOT_RS]
    checked = 0
    for name, (lower, upper, unit) in sorted(usd_limits.items()):
        value = spec.rest[name]
        assert lower < value < upper, (
            "B601-RS rest %s = %r is outside the USD limits [%r, %r] %s"
            % (name, value, lower, upper, unit))
        limit = rest_pose.JointLimit(
            name=name, kind="prismatic" if unit == "m" else "revolute",
            lower=lower, upper=upper, unit=unit, source="pxr export")
        assert rest_pose.margin(limit, value) >= rest_pose.required_margin(limit), (name, value)
        checked += 1
    assert checked == 8, checked


def test_the_documented_rs_table_agrees_with_the_recorded_usd_limits(rest_pose, usd_limits):
    spec = rest_pose.REST_POSES[rest_pose.MODEL_REBOT_RS]
    for name, (lower, upper, unit) in sorted(usd_limits.items()):
        limit = spec.limits[name]
        tol = LIMIT_TOL_M if unit == "m" else LIMIT_TOL_RAD
        assert limit.lower == pytest.approx(lower, abs=tol), (name, limit.lower, lower)
        assert limit.upper == pytest.approx(upper, abs=tol), (name, limit.upper, upper)


def test_the_rs_rest_pose_is_not_the_piper_pose(rest_pose):
    """The defect, stated as a fact about the tables: the two arms cannot share a rest pose."""
    rs = rest_pose.REST_POSES[rest_pose.MODEL_REBOT_RS].rest
    piper = rest_pose.REST_POSES[rest_pose.MODEL_PIPER].rest
    assert rs["joint3"] > 0.0, rs["joint3"]
    assert piper["joint3"] < 0.0, piper["joint3"]
    assert any(_bits(rs[name]) != _bits(piper[name]) for name in ("joint2", "joint3")), (rs, piper)


def test_the_piper_pose_is_rejected_on_the_rs_arm(rest_pose):
    """The exact write that P5 called a defect: the PiPER pose on the B601-RS.

    Two things are wrong with it and both raise: the six-joint dict cannot even cover the eight
    DOFs (UnknownJoint), and once the fingers are filled in, joint3 = -1.5 rad is outside
    [0, 3.14] (RestPoseLimitViolation).
    """
    with pytest.raises(rest_pose.ResetPoseError) as excinfo:
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, LEGACY_PIPER_REST)
    assert "joint3" in str(excinfo.value), str(excinfo.value)

    complete = dict(rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS))
    complete["joint3"] = LEGACY_PIPER_REST["joint3"]
    with pytest.raises(rest_pose.RestPoseLimitViolation) as excinfo:
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, complete)
    assert "joint3" in str(excinfo.value) and "-1.5" in str(excinfo.value), str(excinfo.value)
    problems = rest_pose.check_rest_pose(rest_pose.MODEL_REBOT_RS, complete)
    assert any("joint3" in problem for problem in problems), problems


# --------------------------------------------------------------------------- 4. PiPER regression
def test_piper_rest_pose_is_bit_identical_to_the_stage0_values(rest_pose):
    piper = rest_pose.rest_joint_pos(rest_pose.MODEL_PIPER)
    assert tuple(piper) == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    for name, expected in LEGACY_PIPER_REST.items():
        assert _bits(piper[name]) == _bits(expected), (name, piper[name], expected)


def test_piper_rest_pose_itself_passes_validation(rest_pose):
    rest_pose.validate_rest_pose(rest_pose.MODEL_PIPER, LEGACY_PIPER_REST)


def test_piper_limits_are_the_recorded_stage0_urdf_limits(rest_pose):
    """Guard against the PiPER table drifting away from the Stage-0 URDF record.

    simulation/piper_stage0/t02_articulation_checks.py carries the AgileX URDF limits as a literal;
    importing that script starts Isaac Sim, so the values are read out of the source text instead.
    """
    import re

    import pose_data

    source = pose_data.SCENE_DIR / "piper_stage0" / "t02_articulation_checks.py"
    if not source.is_file():
        pytest.skip("no PiPER Stage-0 record in this tree: " + str(source))
    text = source.read_text(encoding="utf-8")
    block = re.search(r"URDF_LIMITS\s*=\s*\{([^}]*)\}", text)
    assert block, "URDF_LIMITS not found in " + str(source)
    recorded = {
        name: (float(lo), float(hi))
        for name, lo, hi in re.findall(
            r'"(joint\d)"\s*:\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)',
            block.group(1))
    }
    assert set(recorded) == {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}, recorded
    limits = rest_pose.limits_for(rest_pose.MODEL_PIPER)
    for name, (lo, hi) in recorded.items():
        assert limits[name].lower == pytest.approx(lo, abs=1e-9), name
        assert limits[name].upper == pytest.approx(hi, abs=1e-9), name


# --------------------------------------------------------------------------- 5. model resolution
PIPER_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
RS_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            "gripper_joint1", "gripper_joint2")


@pytest.mark.parametrize("alias", ["piper", "PiPER", " piper ", "agilex_piper", "piper_stage0"])
def test_piper_aliases_resolve_from_the_joint_names(rest_pose, alias):
    assert rest_pose.resolve_model(PIPER_NAMES, robot_key=alias) == rest_pose.MODEL_PIPER


@pytest.mark.parametrize("alias", ["rebot_b601_rs", "ReBot-B601-RS", "rs", "rebot", "rebot_arm_rs"])
def test_rebot_rs_aliases_resolve_from_the_joint_names(rest_pose, alias):
    assert rest_pose.resolve_model(RS_NAMES, robot_key=alias) == rest_pose.MODEL_REBOT_RS


def test_model_is_inferred_from_the_dof_name_set(rest_pose):
    assert rest_pose.resolve_model(PIPER_NAMES) == rest_pose.MODEL_PIPER
    assert rest_pose.resolve_model(("joint6", "joint1", "gripper_joint1", "joint2", "joint3",
                                    "joint4", "joint5", "gripper_joint2")) == rest_pose.MODEL_REBOT_RS


@pytest.mark.parametrize(
    "names",
    [
        (),
        ("joint1", "joint2"),
        ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint1"),
        ("shoulder", "elbow", "wrist"),
    ],
)
def test_unknown_dof_name_sets_raise_instead_of_guessing(rest_pose, names):
    with pytest.raises(rest_pose.UnknownRobotModel):
        rest_pose.resolve_model(names)


def test_an_explicit_model_that_contradicts_the_names_raises(rest_pose):
    with pytest.raises(rest_pose.ResetPoseError):
        rest_pose.resolve_model(RS_NAMES, robot_key=rest_pose.MODEL_PIPER)


def test_an_unknown_explicit_model_raises(rest_pose):
    with pytest.raises(rest_pose.UnknownRobotModel):
        rest_pose.resolve_model(PIPER_NAMES, robot_key="piper_v2")


# --------------------------------------------------------------------------- 6. validation
def test_validation_rejects_a_missing_joint(rest_pose):
    partial = {name: 0.0 for name in ("joint1", "joint2", "joint3", "joint4", "joint5")}
    with pytest.raises(rest_pose.UnknownJoint):
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, partial)


def test_validation_rejects_an_unknown_joint_name(rest_pose):
    values = dict(rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS))
    values["joint7"] = 0.0
    with pytest.raises(rest_pose.UnknownJoint):
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, values)


def test_validation_rejects_a_boundary_value_when_the_margin_is_required(rest_pose):
    values = dict(rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS))
    values["joint2"] = 0.0  # exactly the lower limit
    with pytest.raises(rest_pose.RestPoseLimitViolation):
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, values, require_margin=True)
    problems = rest_pose.check_rest_pose(rest_pose.MODEL_REBOT_RS, values)
    assert any("joint2" in problem for problem in problems), problems


def test_validation_rejects_a_nan_rest_value(rest_pose):
    """A NaN rest value must raise, not slip through a min/max margin comparison."""
    values = dict(rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS))
    values["joint2"] = float("nan")
    with pytest.raises(rest_pose.RestPoseLimitViolation):
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, values)
    assert any("joint2" in problem for problem in
               rest_pose.check_rest_pose(rest_pose.MODEL_REBOT_RS, values))


def test_validation_accepts_an_explicit_documented_pose(rest_pose):
    posed = dict(rest_pose.rest_joint_pos(rest_pose.MODEL_REBOT_RS))
    posed["joint2"] = 0.5
    posed["joint3"] = 1.2
    rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, posed)


# --------------------------------------------------------------------------- 7. structural canary
def test_rest_pose_module_carries_no_torch_or_pxr_dependency(rest_pose):
    """The table is data plus arithmetic: it must stay importable with neither runtime present."""
    text = open(rest_pose.__file__, encoding="utf-8").read()
    assert "import torch" not in text, "rest_pose.py must not need torch"
    assert "import pxr" not in text, "rest_pose.py must not need pxr"
    assert "import numpy" not in text, "rest_pose.py must not need numpy"
