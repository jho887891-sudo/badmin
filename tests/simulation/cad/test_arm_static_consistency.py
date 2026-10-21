# -*- coding: utf-8 -*-
"""P5 - static consistency of the B601-RS arm USD against the repository limit table.

WHY THIS FILE EXISTS
    The arm joint count, order and ranges are the contract every downstream layer leans on
    (rest pose, safety shield, reset, racket mount). Until now that contract was only checked by
    running a simulator on the GPU host, so a plain CI machine could not see it at all. This test
    reads the asset with pxr only - no Isaac Sim, no torch, no cuda - so it runs anywhere USD is
    installed and skips (does not fail) where USD is not.

WHAT IT PINS
    1. The joint census: 9 joints = 1 Fixed + 6 Revolute + 2 Prismatic.
    2. The Fixed joint is the base fix root_joint, NOT the URDF mid-chain j_gripper_end
       (--merge-joints folded that one into link6). Equal counts alone would not prove this.
    3. Revolute limits equal the documented ranges (URDF limits are DEGREES in USD) and stay
       inside them; the two finger ranges match as well.
    4. The joint NAME SET equals joint_limits.JOINT_LIMITS.
    5. The kinematic chain order equals joint_limits.JOINT_ORDER - while the stage own traversal
       order does not, which is why nothing may index DOFs by traversal position.
    6. Drive gains are DEGREE-scaled (stiffness = 100 / 57.2958) and maxForce equals the URDF
       effort, so "the asset asked for 100" never becomes a 57x error.
    7. Total rigid-body mass equals joint_limits.total_model_mass() even though merge-joints
       changes the per-link split (link6 = 0.75 = 0.1 + 0.65).

WHERE THE ASSET LIVES
    The USD package is a 20 MB resource and is NOT in the repository (AGENTS.md: resources stay on
    the host, only results are committed). Point P5_ARM_USDA at a local copy or leave the default,
    which is the staged path on the GPU host. A missing file SKIPS, it does not fail.

REFERENCE TABLE
    src/simulation/rebot_b601_rs/joint_limits.py is the documented authority. It is resolved
    relative to this file, or through P5_JOINT_LIMITS; when it is absent (this file was first
    written on a branch cut from origin/main, which does not carry the B601-RS migration yet) the
    comparison tests SKIP with that reason instead of inventing numbers.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from pathlib import Path

import pytest

# pxr is the whole point of the file: without it there is nothing static to read.
Usd = pytest.importorskip("pxr.Usd", reason="pxr (USD) not installed: the static USD checks need it")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics", reason="pxr.UsdPhysics not installed")

DEFAULT_USDA = (
    "/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
    "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda"
)
USDA_PATH = Path(os.environ.get("P5_ARM_USDA", DEFAULT_USDA))

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]

DEG_PER_RAD = 180.0 / math.pi
#: UsdPhysics authors revolute limits in degrees; the degree tolerance covers the float32 storage of
#: the converted values (0.01 deg = 1.7e-4 rad) and nothing more.
LIMIT_TOL_DEG = 1e-2
LIMIT_TOL_RAD = 1e-6


# --------------------------------------------------------------------------- fixtures
def _load_definition():
    """The documented limit table, loaded by path so this test does not depend on sys.path order."""
    candidates = []
    env_path = os.environ.get("P5_JOINT_LIMITS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(REPO_ROOT / "src" / "simulation" / "rebot_b601_rs" / "joint_limits.py")
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_p5_joint_limits", path)
            module = importlib.util.module_from_spec(spec)
            # Register before exec: dataclasses resolve string annotations through
            # sys.modules[cls.__module__], so an unregistered module raises AttributeError.
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(spec.name, None)
                raise
            module._P5_SOURCE_PATH = str(path)
            return module
    pytest.skip(
        "joint_limits.py not found (tried: "
        + ", ".join(str(p) for p in candidates)
        + "). The B601-RS limit table lives on origin/feature/rebot-rs-sim-asset @ d912202, "
          "not on origin/main; set P5_JOINT_LIMITS to compare against it."
    )


@pytest.fixture(scope="module")
def limits():
    return _load_definition()


@pytest.fixture(scope="module")
def stage():
    if not USDA_PATH.is_file():
        pytest.skip("arm USD not staged at " + str(USDA_PATH) + " (set P5_ARM_USDA)")
    st = Usd.Stage.Open(str(USDA_PATH))
    assert st, "pxr could not open " + str(USDA_PATH)
    return st


@pytest.fixture(scope="module")
def joints(stage):
    """[(prim, path, typeName)] in stage traversal order."""
    out = []
    for prim in stage.Traverse():
        if prim.GetTypeName() in (
            "PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint",
            "PhysicsSphericalJoint", "PhysicsDistanceJoint",
        ):
            out.append((prim, str(prim.GetPath()), prim.GetTypeName()))
    return out


def _by_type(joints, type_name):
    return [(p, path, tn) for (p, path, tn) in joints if tn == type_name]


def _limits_deg(prim):
    lo = prim.GetAttribute("physics:lowerLimit")
    hi = prim.GetAttribute("physics:upperLimit")
    return (lo.Get() if lo else None), (hi.Get() if hi else None)


def _name(path):
    return path.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- 1. census
def test_joint_census_is_one_fixed_six_revolute_two_prismatic(joints):
    counts = {}
    for _p, _path, type_name in joints:
        counts[type_name] = counts.get(type_name, 0) + 1
    assert len(joints) == 9, "expected 9 joints, got %d: %s" % (len(joints), counts)
    assert counts.get("PhysicsFixedJoint") == 1, counts
    assert counts.get("PhysicsRevoluteJoint") == 6, counts
    assert counts.get("PhysicsPrismaticJoint") == 2, counts
    assert set(counts) == {
        "PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}, counts


def test_the_fixed_joint_is_the_base_fix_not_the_merged_gripper_flange(joints):
    """9 joints in the URDF too - but a different 9. A count match is not an identity match."""
    fixed = [_name(path) for (_p, path, _tn) in _by_type(joints, "PhysicsFixedJoint")]
    assert fixed == ["root_joint"], fixed
    names = {_name(path) for (_p, path, _tn) in joints}
    assert "j_gripper_end" not in names, "--merge-joints should have folded j_gripper_end into link6"


# --------------------------------------------------------------------------- 2. revolute limits
def test_revolute_limits_equal_the_documented_ranges(joints, limits):
    seen = 0
    for prim, path, _tn in _by_type(joints, "PhysicsRevoluteJoint"):
        name = _name(path)
        if name not in limits.JOINT_LIMITS:
            continue
        lo_deg, hi_deg = _limits_deg(prim)
        assert lo_deg is not None and hi_deg is not None, name + " has no authored limits"
        expected_lo, expected_hi = limits.angle_range(name)
        # lo_deg/hi_deg are ALREADY degrees (UsdPhysics authors revolute limits in degrees).
        assert lo_deg == pytest.approx(math.degrees(expected_lo), abs=LIMIT_TOL_DEG), name
        assert hi_deg == pytest.approx(math.degrees(expected_hi), abs=LIMIT_TOL_DEG), name
        # USD stores degrees; reading them as radians would be off by 57.3x. Pin the scale.
        assert math.radians(lo_deg) == pytest.approx(expected_lo, abs=LIMIT_TOL_RAD), name
        assert math.radians(hi_deg) == pytest.approx(expected_hi, abs=LIMIT_TOL_RAD), name
        seen += 1
    assert seen == 6, "compared %d revolute joints against the table, expected 6" % seen


def test_every_revolute_limit_stays_inside_the_documented_range(joints, limits):
    """Containment, not equality: a simulator may narrow a range, never widen it silently."""
    for prim, path, _tn in _by_type(joints, "PhysicsRevoluteJoint"):
        name = _name(path)
        if name not in limits.JOINT_LIMITS:
            continue
        lo_deg, hi_deg = _limits_deg(prim)
        expected_lo, expected_hi = limits.angle_range(name)
        lo, hi = math.radians(lo_deg), math.radians(hi_deg)
        assert expected_lo - LIMIT_TOL_RAD <= lo <= hi <= expected_hi + LIMIT_TOL_RAD, (
            "%s: USD [%.6f, %.6f] rad escapes the documented [%.6f, %.6f]"
            % (name, lo, hi, expected_lo, expected_hi))


def test_prismatic_finger_limits_match(joints, limits):
    seen = 0
    for prim, path, _tn in _by_type(joints, "PhysicsPrismaticJoint"):
        name = _name(path)
        lo, hi = _limits_deg(prim)
        expected_lo, expected_hi = limits.angle_range(name)
        assert lo == pytest.approx(expected_lo, abs=1e-6), name
        assert hi == pytest.approx(expected_hi, abs=1e-6), name
        assert limits.JOINT_LIMITS[name].kind == "prismatic", name
        seen += 1
    assert seen == 2, seen


# --------------------------------------------------------------------------- 3. names and order
def test_joint_name_set_matches_joint_limits(joints, limits):
    usd_names = {_name(path) for (_p, path, type_name) in joints
                 if type_name != "PhysicsFixedJoint"}
    assert usd_names == set(limits.JOINT_LIMITS), (
        "USD movable joints %s != joint_limits.JOINT_LIMITS %s"
        % (sorted(usd_names), sorted(limits.JOINT_LIMITS)))


def _chain_order(joints):
    """Order the movable joints by walking body0 -> body1 from the base, never by index.

    A body can carry more than one joint (link6 carries both finger joints), so the map is
    one-to-many; a single-valued dict would silently drop one finger and the comparison would look
    correct for the wrong reason.
    """
    joints_of = {}   # body0 path -> [joint name]
    child_of = {}    # body0 path -> [body1 path]
    for prim, path, type_name in joints:
        if type_name == "PhysicsFixedJoint":
            continue
        b0 = prim.GetRelationship("physics:body0").GetTargets()
        b1 = prim.GetRelationship("physics:body1").GetTargets()
        assert len(b0) == 1 and len(b1) == 1, "joint " + path + " must have one body0 and one body1"
        joints_of.setdefault(str(b0[0]), []).append(prim.GetName())
        child_of.setdefault(str(b0[0]), []).append(str(b1[0]))
    base = [p for p in joints_of if p.endswith("/base_link")]
    assert len(base) == 1, base
    order = []
    node = base[0]
    while node in joints_of:
        names = sorted(joints_of[node])
        order.extend(names)
        if len(child_of.get(node, [])) != 1:
            break           # branch or leaf: every remaining joint hangs off this body
        node = child_of[node][0]
    return order


def test_kinematic_chain_order_matches_joint_limits_order(joints, limits):
    chain = _chain_order(joints)
    assert chain == list(limits.JOINT_ORDER), (
        "USD kinematic chain order %s != joint_limits.JOINT_ORDER %s"
        % (chain, list(limits.JOINT_ORDER)))


def test_stage_traversal_order_is_not_the_dof_order(joints, limits):
    """Canary: the stage walks the layer's child order, PhysX walks the kinematic chain.

    Measured on this asset the traversal is joint6, gripper_joint1, gripper_joint2, joint5, joint4,
    joint3, joint2, joint1, root_joint - neither the URDF order nor alphabetical. If it ever becomes
    the documented order, a positional DOF lookup would start looking correct by accident, so the
    mismatch is pinned deliberately and this message tells the next reader to update the record.
    """
    traversal = [_name(path) for (_p, path, _tn) in joints]
    assert traversal != list(limits.JOINT_ORDER), traversal
    assert len(traversal) == 9, traversal


# --------------------------------------------------------------------------- 4. drives and masses
def test_revolute_drive_gains_are_degree_scaled(joints, limits):
    seen = 0
    for prim, path, _tn in _by_type(joints, "PhysicsRevoluteJoint"):
        name = _name(path)
        if name not in limits.JOINT_LIMITS:
            continue
        stiffness = prim.GetAttribute("drive:angular:physics:stiffness").Get()
        damping = prim.GetAttribute("drive:angular:physics:damping").Get()
        max_force = prim.GetAttribute("drive:angular:physics:maxForce").Get()
        # convert_urdf.py was asked for stiffness 100 / damping 1.0; the asset stores the degree
        # equivalent, i.e. that value divided by 57.2958. Reading it as rad-based overstates by 57x.
        assert stiffness == pytest.approx(100.0 / DEG_PER_RAD, rel=1e-6), (name, stiffness)
        assert damping == pytest.approx(1.0 / DEG_PER_RAD, rel=1e-6), (name, damping)
        assert max_force == pytest.approx(limits.JOINT_LIMITS[name].urdf_effort, rel=1e-6), (name, max_force)
        seen += 1
    assert seen == 6, seen


def test_prismatic_drive_gains_are_not_degree_scaled(joints, limits):
    seen = 0
    for prim, path, _tn in _by_type(joints, "PhysicsPrismaticJoint"):
        name = _name(path)
        if name not in limits.JOINT_LIMITS:
            continue
        assert prim.GetAttribute("drive:linear:physics:stiffness").Get() == pytest.approx(100.0), name
        assert prim.GetAttribute("drive:linear:physics:damping").Get() == pytest.approx(1.0), name
        assert prim.GetAttribute("drive:linear:physics:maxForce").Get() == pytest.approx(
            limits.JOINT_LIMITS[name].urdf_effort), name
        seen += 1
    assert seen == 2, seen


def test_articulation_root_is_unique(stage):
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1, roots
    assert roots[0].endswith("/Geometry"), roots


def test_rigid_body_mass_total_matches_the_table(stage, limits):
    masses = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.MassAPI(prim)
        mass = api.GetMassAttr().Get() if api and api.GetMassAttr() else None
        assert mass is not None, str(prim.GetPath()) + " has RigidBodyAPI but no mass"
        masses.append(mass)
    assert len(masses) == 9, len(masses)
    # merge-joints moved gripper_end 0.65 kg into link6, so the per-link split differs from the
    # table while the total must not.
    # The table carries the URDF masses rounded to 4 decimals, so the agreement is 1e-4 kg.
    assert sum(masses) == pytest.approx(limits.total_model_mass(), abs=1e-4), sum(masses)


# --------------------------------------------------------------------------- 5. pose sanity
def _rest_pose_pairs(path, var):
    """The rest_joint_pos dict of one arm block, as [(joint, value)], or None when the block is absent."""
    text = path.read_text(encoding="utf-8")
    anchor = re.search(re.escape(var) + r"\s*=\s*dict\(", text)
    if not anchor:
        return None
    m = re.search(r"rest_joint_pos=\{([^}]*)\}", text[anchor.end():])
    if not m:
        return None
    return re.findall(r'"(joint\d)"\s*:\s*(-?\d+(?:\.\d+)?)', m.group(1))


def test_formal_arm_rest_pose_is_inside_the_documented_limits(limits):
    """The B601-RS rest pose the scene spawns the arm at must be reachable on this arm."""
    layout = REPO_ROOT / "simulation" / "badminton_scene" / "scene_layout.py"
    if not layout.is_file():
        pytest.skip("no scene_layout.py in this tree: " + str(layout))
    pairs = _rest_pose_pairs(layout, "REBOT_RS")
    if pairs is None:
        pytest.skip("no REBOT_RS block in this tree (it arrives with the B601-RS migration)")
    assert len(pairs) == 6, pairs
    for name, value in pairs:
        lo, hi = limits.angle_range(name)
        assert lo - LIMIT_TOL_RAD <= float(value) <= hi + LIMIT_TOL_RAD, (
            "%s rest value %s rad is outside the documented [%s, %s] (%s)"
            % (name, value, lo, hi, layout))


def test_reset_rest_joint_is_the_piper_pose_canary(limits):
    """DEFECT CANARY (P5 finding): reset.py's arm-agnostic REST_JOINT is the PiPER pose.

    simulation/badminton_scene/reset.py hardcodes joint2 = 1.5, joint3 = -1.5 and reset_idx() uses
    it whenever no explicit joint_pos is passed - on the B601-RS joint3 is [0, 3.14], so that write
    is a limit-violating state. The defect is recorded in management/morph_one_rs/
    P5_ARM_STATIC_CONSISTENCY.md rather than silently patched from here (this branch is a read-only
    audit of the asset). If someone fixes reset.py, THIS test fails on purpose: update the record
    instead of deleting the canary.
    """
    reset_py = REPO_ROOT / "simulation" / "badminton_scene" / "reset.py"
    if not reset_py.is_file():
        pytest.skip("no reset.py in this tree: " + str(reset_py))
    text = reset_py.read_text(encoding="utf-8")
    m = re.search(r"REST_JOINT\s*=\s*\{([^}]*)\}", text)
    assert m, "REST_JOINT not found in " + str(reset_py)
    pairs = dict(re.findall(r'"(joint\d)"\s*:\s*(-?\d+(?:\.\d+)?)', m.group(1)))
    offenders = []
    for name, value in pairs.items():
        lo, hi = limits.angle_range(name)
        if not (lo - LIMIT_TOL_RAD <= float(value) <= hi + LIMIT_TOL_RAD):
            offenders.append((name, float(value), lo, hi))
    assert offenders == [("joint3", -1.5, 0.0, 3.14)] or (
        len(offenders) == 1 and offenders[0][0] == "joint3" and offenders[0][1] == -1.5), (
        "the recorded reset.py defect changed: offenders=%s - update "
        "management/morph_one_rs/P5_ARM_STATIC_CONSISTENCY.md" % (offenders,))
