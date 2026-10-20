"""Regression tests for the orientation probe (pxr only, no GPU).

These pin the two facts the mount decision depends on: which rotation sign turns a
Y-up CAD model into a Z-up world pose, and that the reported up axis follows the
geometry rather than the asset metadata.
"""

import os
import sys

import pytest

pytest.importorskip("pxr")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pxr import Gf, Usd, UsdGeom  # noqa: E402

from scripts.simulation.morph_one_orientation_probe import analyse, rot_x  # noqa: E402


def make_chassis(path, size_mm):
    """A millimetre-authored layer whose smallest extent is Y."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Chassis")
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Chassis/body")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size_mm))
    stage.GetRootLayer().Save()
    return stage


def rounded(vec, digits=9):
    """Compare to 1e-9: a 90 deg rotation legitimately leaves ~6e-17 residue."""
    return [round(float(v), digits) for v in vec]


def test_plus_ninety_about_x_turns_cad_up_into_world_z():
    assert rounded(rot_x([0.0, 1.0, 0.0], 90.0)) == [0.0, 0.0, 1.0]
    assert rounded(rot_x([0.0, 1.0, 0.0], -90.0)) == [0.0, 0.0, -1.0]
    assert rounded(rot_x([1.0, 0.0, 0.0], 90.0)) == [1.0, 0.0, 0.0]
    assert rounded(rot_x([0.0, 0.0, 1.0], 90.0)) == [0.0, -1.0, 0.0]


def test_analyse_reports_the_geometric_up_axis_not_the_metadata(tmp_path):
    path = str(tmp_path / "chassis.usda")
    make_chassis(path, (580.0, 360.0, 800.0))
    facts = analyse(path, "", 90.0)
    assert facts["asset_up_axis"] == "Z", "metadata is Z"
    assert facts["up_axis_by_smallest_extent"] == "y", "geometry is Y-up"
    assert facts["world_up_axis_by_smallest_extent"] == "z"
    assert facts["applied_scale"] == 0.001
    assert facts["applied_rotation_deg_about_x"] == 90.0


def test_analyse_applies_the_scale_and_permutes_the_extents(tmp_path):
    path = str(tmp_path / "chassis.usda")
    make_chassis(path, (580.0, 360.0, 800.0))
    facts = analyse(path, "", 90.0)
    world = facts["world_bbox_m"]
    assert world["size"][1] > world["size"][2], "length ends up along Y after +90 deg"
    assert [round(v, 3) for v in world["size"]] == [0.58, 0.8, 0.36]
    assert facts["world_bbox_size_permuted"] is True


def test_wheel_lookup_reports_missing_mapping_instead_of_inventing_it(tmp_path):
    path = str(tmp_path / "chassis.usda")
    make_chassis(path, (580.0, 360.0, 800.0))
    facts = analyse(path, str(tmp_path / "absent.json"), 90.0)
    assert len(facts["wheel_solids"]) == 4
    assert all(row["found"] is False for row in facts["wheel_solids"])
