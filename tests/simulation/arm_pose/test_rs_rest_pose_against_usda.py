# -*- coding: utf-8 -*-
"""P5B - the model-aware rest pose against the LIVE B601-RS USD (pxr only, read-only).

WHY IT IS SEPARATE FROM test_model_aware_rest_pose.py
    That file reads the committed pxr export (arm_joint_facts.json) so it can run on any machine.
    This one opens the real asset and re-derives the limits itself, so the documented table cannot
    drift away from the asset and the export cannot quietly go stale. It needs pxr, and pxr is not
    installed everywhere, so a missing pxr or a missing asset SKIPS rather than fails. It never
    writes the asset: the sha256 is compared before and after the walk.

RUN (CPU only, no Isaac Sim, no torch):

    P5_ARM_USDA=<...>/ReBot_Arm_RS.usda python -m pytest \
        tests/simulation/arm_pose/test_rs_rest_pose_against_usda.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

Usd = pytest.importorskip("pxr.Usd", reason="pxr (USD) not installed: the live-asset checks need it")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics", reason="pxr.UsdPhysics not installed")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulation"))

import badminton_scene.rest_pose as rest_pose  # noqa: E402

DEFAULT_USDA = (
    "/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
    "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda"
)
USDA_PATH = Path(os.environ.get("P5_ARM_USDA", DEFAULT_USDA))

#: The asset hash P5 and P5B both observed. A different hash is not a test failure by itself - it
#: means the asset was re-exported and the recorded numbers must be re-measured, so the test says so.
RECORDED_SHA256 = "96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510"
EVIDENCE_JSON = (
    REPO_ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "morph_one" / "p5b_reset_pose"
    / "rs_rest_pose_probe.json"
)
LIMIT_TOL_RAD = 1e-6
LIMIT_TOL_M = 1e-9


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def asset():
    if not USDA_PATH.is_file():
        pytest.skip("B601-RS USD not staged at " + str(USDA_PATH) + " (set P5_ARM_USDA)")
    before = _sha256(USDA_PATH)
    stage = Usd.Stage.Open(str(USDA_PATH))
    assert stage, "pxr could not open " + str(USDA_PATH)
    yield stage, before
    assert _sha256(USDA_PATH) == before, "the read-only walk changed the asset"


@pytest.fixture(scope="module")
def usd_limits(asset):
    """{joint name: (lower, upper, unit)} read from the live stage, in rad and m."""
    stage, _before = asset
    found = {}
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            continue
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        assert lower is not None and upper is not None, prim.GetName() + " has no authored limits"
        if type_name == "PhysicsRevoluteJoint":
            # UsdPhysics authors revolute limits in DEGREES; the rest pose is in radians.
            found[prim.GetName()] = (math.radians(lower), math.radians(upper), "rad")
        else:
            found[prim.GetName()] = (float(lower), float(upper), "m")
    return found


def test_the_live_asset_has_the_eight_movable_joints_named_by_the_table(asset, usd_limits):
    _stage, _before = asset
    assert set(usd_limits) == {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "gripper_joint1", "gripper_joint2"}, sorted(usd_limits)


def test_the_documented_rs_limits_are_the_live_asset_limits(usd_limits):
    limits = rest_pose.limits_for(rest_pose.MODEL_REBOT_RS)
    for name, (lower, upper, unit) in sorted(usd_limits.items()):
        tolerance = LIMIT_TOL_M if unit == "m" else LIMIT_TOL_RAD
        assert limits[name].lower == pytest.approx(lower, abs=tolerance), name
        assert limits[name].upper == pytest.approx(upper, abs=tolerance), name


def test_the_rs_rest_pose_is_inside_the_live_asset_limits_for_all_eight_joints(usd_limits):
    spec = rest_pose.REST_POSES[rest_pose.MODEL_REBOT_RS]
    for name, (lower, upper, unit) in sorted(usd_limits.items()):
        value = spec.rest[name]
        assert lower < value < upper, (
            "rest %s = %r outside the live asset limits [%r, %r] %s"
            % (name, value, lower, upper, unit))
        limit = rest_pose.JointLimit(
            name=name, kind="prismatic" if unit == "m" else "revolute",
            lower=lower, upper=upper, unit=unit, source="live pxr read")
        assert rest_pose.margin(limit, value) >= rest_pose.required_margin(limit), name


def test_the_asset_hash_is_the_one_this_record_was_measured_on(asset):
    _stage, before = asset
    if before != RECORDED_SHA256:
        pytest.fail(
            "B601-RS asset sha256 is %s, not the recorded %s: re-measure the limits and update "
            "management/morph_one_rs/P5B_RESET_POSE.md (the asset is never modified by this test)"
            % (before, RECORDED_SHA256))


def test_the_probe_evidence_json_agrees_with_this_read(asset, usd_limits):
    """The committed evidence has to be what a live read produces, or it is decoration."""
    stage, before = asset
    if not EVIDENCE_JSON.is_file():
        pytest.skip("no probe evidence at " + str(EVIDENCE_JSON))
    evidence = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    assert evidence["asset"]["usda_sha256_before"] == before
    assert evidence["asset"]["usda_sha256_after"] == before
    assert evidence["asset"]["unmodified"] is True
    for name, (lower, upper, unit) in sorted(usd_limits.items()):
        recorded = evidence["limits"][name]
        tolerance = LIMIT_TOL_M if unit == "m" else LIMIT_TOL_RAD
        assert recorded["lower"] == pytest.approx(lower, abs=tolerance), name
        assert recorded["upper"] == pytest.approx(upper, abs=tolerance), name
