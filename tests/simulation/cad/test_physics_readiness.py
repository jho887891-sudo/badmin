"""Tests for the physics-readiness and mount-sandbox tools.

Both tools need pxr, so this module skips cleanly on a machine without USD and runs
for real in the Isaac Sim interpreter.  The sandbox is a *test proxy*: these tests
therefore check authoring facts (one articulation root, expected DOF, resolvable
joint targets), never dynamics.
"""

import json
import os
import sys

import pytest

pxr = pytest.importorskip("pxr")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from scripts.simulation.morph_one_mount_sandbox import (  # noqa: E402
    build_proxy_stage,
    validate_proxy_stage,
)
from scripts.simulation.morph_one_physics_readiness import (  # noqa: E402
    analyse_stage,
    build_findings,
)


def make_stage(path, with_physics=False):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Model")
    stage.SetDefaultPrim(root.GetPrim())
    holder = UsdGeom.Xform.Define(stage, "/Model/part_a")
    mesh = UsdGeom.Mesh.Define(stage, "/Model/part_a/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(1.0), Gf.Vec3f(2.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    if with_physics:
        UsdPhysics.RigidBodyAPI.Apply(holder.GetPrim())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return path


def test_analyse_stage_reports_units_and_counts(tmp_path):
    path = make_stage(str(tmp_path / "model.usda"))
    facts = analyse_stage(path)
    assert facts["default_prim"] == "/Model"
    assert facts["meters_per_unit"] == 0.001
    assert facts["up_axis"] == "Z"
    assert facts["counts"]["type:Mesh"] == 1
    assert facts["mesh_points_total"] == 3
    assert facts["physics_authoring_present"] is False
    assert facts["physics"] == {}


def test_analyse_stage_detects_physics_authoring(tmp_path):
    path = make_stage(str(tmp_path / "model_phys.usda"), with_physics=True)
    facts = analyse_stage(path)
    assert facts["physics"]["rigid_bodies"] == 1
    assert facts["physics"]["colliders"] == 1
    assert facts["physics_authoring_present"] is True


def test_findings_label_missing_data_as_unknown(tmp_path):
    path = make_stage(str(tmp_path / "model.usda"))
    facts = analyse_stage(path)
    findings = build_findings(facts, None)
    by_question = {row["question"][:2]: row for row in findings}
    assert by_question["C1"]["status"] in ("VERIFIED", "UNKNOWN")
    assert by_question["C6"]["status"] == "UNKNOWN"
    assert "no mass" in by_question["C6"]["statement"]


def test_sandbox_proxy_has_one_root_and_expected_dof(tmp_path):
    stage, description = build_proxy_stage()
    out = str(tmp_path / "proxy.usda")
    stage.GetRootLayer().Export(out)
    facts = validate_proxy_stage(out)
    assert facts["articulation_roots"] == ["/World/ArmProxy"], facts
    assert facts["dof_from_joint_types"] == description["proxy_dof"] == 2
    assert facts["all_joint_targets_resolve"] is True
    assert any(row["type"] == "PhysicsFixedJoint" for row in facts["joints"])
    # the chassis plate plus the two proxy links plus the proxy base
    assert facts["rigid_bodies"] == 4, facts["rigid_bodies"]


def test_sandbox_labels_itself_as_a_proxy(tmp_path):
    stage, _ = build_proxy_stage()
    out = str(tmp_path / "proxy.usda")
    stage.GetRootLayer().Export(out)
    stage2 = Usd.Stage.Open(out)
    labels = [prim.GetCustomDataByKey("label") for prim in stage2.Traverse()]
    text = " ".join(str(label) for label in labels if label)
    assert "TEST_PROXY" in text
    assert "not the Morph One mechanical assembly" in text.lower() or "NOT the B601-RS" in text
