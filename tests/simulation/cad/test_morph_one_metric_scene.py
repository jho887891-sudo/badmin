"""Tests for the metres-stage assembly module.

Two layers:

* pure numeric/decision helpers, testable anywhere (no pxr needed);
* composition tests that author a fixture visual USD, reference it into a metres
  stage with and without a 0.001 scale, and check the measured world bbox.  Those
  need pxr, so they skip when the interpreter has no USD (the project's default
  python does not; the Isaac Sim venv does).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation.morph_one_metric_scene import (  # noqa: E402
    DEFAULT_ENVELOPE_M,
    envelope_within,
    evaluate,
    ratio_between,
)


def test_envelope_within_accepts_the_expected_morph_one_envelope():
    assert envelope_within([0.580, 0.373, 0.803], DEFAULT_ENVELOPE_M) is True


def test_envelope_within_rejects_a_thousandfold_or_ten_percent_error():
    assert envelope_within([580.0, 373.0, 803.0], DEFAULT_ENVELOPE_M) is False
    assert envelope_within([0.638, 0.373, 0.803], DEFAULT_ENVELOPE_M) is False


def test_envelope_within_uses_the_absolute_floor_for_tiny_axes():
    # 1.5 mm off on a 3 mm-expected axis is 50 percent, but within the 2 mm floor
    assert envelope_within([0.0035], [0.002], tolerance_ratio=0.02, floor_m=0.002) is True
    assert envelope_within([0.0050], [0.002], tolerance_ratio=0.02, floor_m=0.002) is False


def test_envelope_within_handles_missing_values():
    assert envelope_within(None, DEFAULT_ENVELOPE_M) is False
    assert envelope_within([0.58, 0.373], DEFAULT_ENVELOPE_M) is False


def test_ratio_between_reports_the_scale_factor():
    assert ratio_between([0.58, 0.373, 0.803], [580.0, 373.0, 803.0]) == [0.001, 0.001, 0.001]
    assert ratio_between(None, [1.0]) is None


def test_evaluate_marks_a_good_measurement_as_passing():
    report = {
        "opened": True,
        "bbox_methods_agree": True,
        "mesh_instances": 282,
        "physx_apis": {"CollisionAPI": 0, "RigidBodyAPI": 0, "MassAPI": 0},
        "world_bbox_from_points_meters": [0.580, 0.373, 0.803],
    }
    result = evaluate(report, DEFAULT_ENVELOPE_M)
    assert result["matches_expected_envelope"] is True
    assert all(result["checks"].values())


def test_evaluate_rejects_a_scene_that_kept_physics_apis():
    report = {
        "opened": True,
        "bbox_methods_agree": True,
        "mesh_instances": 282,
        "physx_apis": {"CollisionAPI": 3},
        "world_bbox_from_points_meters": [0.580, 0.373, 0.803],
    }
    result = evaluate(report, DEFAULT_ENVELOPE_M)
    assert result["checks"]["no_physics_apis"] is False


def test_evaluate_reports_the_unscaled_case_as_not_matching():
    report = {
        "opened": True,
        "bbox_methods_agree": True,
        "mesh_instances": 282,
        "physx_apis": {"CollisionAPI": 0},
        "world_bbox_from_points_meters": [580.0, 372.996, 802.852],
    }
    result = evaluate(report, DEFAULT_ENVELOPE_M)
    assert result["matches_expected_envelope"] is False


# ---------------------------------------------------------------- pxr-dependent


@pytest.fixture()
def fixture_visual_usd(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, Vt

    path = tmp_path / "fixture_visual.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    fixture_root = UsdGeom.Xform.Define(stage, "/Fixture")
    stage.SetDefaultPrim(fixture_root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Fixture/Box")
    size = 1000.0  # millimetres
    points = [
        (0.0, 0.0, 0.0), (size, 0.0, 0.0), (size, size * 0.373, 0.0), (0.0, size * 0.373, 0.0),
        (0.0, 0.0, size * 0.803), (size, 0.0, size * 0.803),
        (size, size * 0.373, size * 0.803), (0.0, size * 0.373, size * 0.803),
    ]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([tuple(map(float, p)) for p in points]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4, 4, 4, 4, 4, 4]))
    mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray([0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 7, 3, 1, 5, 4, 0, 2, 6, 5, 1, 3, 7, 6, 2])
    )
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([(0.0, 0.0, 0.0), (size, size * 0.373, size * 0.803)]))
    stage.GetRootLayer().Save()
    return path


def test_unscaled_reference_stays_in_authored_millimetres(tmp_path, fixture_visual_usd):
    pxr = pytest.importorskip("pxr")
    from scripts.simulation.morph_one_metric_scene import build_metric_scene, measure_scene

    scene = build_metric_scene(tmp_path / "A.usda", fixture_visual_usd, scale=None)
    report = measure_scene(scene)
    assert report["stage_meters_per_unit"] == 1.0
    assert report["mesh_instances"] == 1
    assert report["bbox_methods_agree"] is True
    # 1000 authored units stay 1000 units: USD does not convert on reference
    assert report["world_bbox_from_points_stage_units"][0] == pytest.approx(1000.0, abs=1e-6)
    assert report["composed_local_to_world_diagonal"] == [1.0, 1.0, 1.0]


def test_scaled_reference_lands_on_the_expected_metre_envelope(tmp_path, fixture_visual_usd):
    pxr = pytest.importorskip("pxr")
    from scripts.simulation.morph_one_metric_scene import build_metric_scene, measure_scene

    scene = build_metric_scene(
        tmp_path / "B.usda", fixture_visual_usd, scale=(0.001, 0.001, 0.001)
    )
    report = measure_scene(scene)
    assert report["composed_local_to_world_diagonal"] == [0.001, 0.001, 0.001]
    size = report["world_bbox_from_points_meters"]
    assert size[0] == pytest.approx(1.0, abs=1e-6)
    assert size[1] == pytest.approx(0.373, abs=1e-6)
    assert size[2] == pytest.approx(0.803, abs=1e-6)


def test_measurement_reports_no_physics_apis(tmp_path, fixture_visual_usd):
    pxr = pytest.importorskip("pxr")
    from scripts.simulation.morph_one_metric_scene import build_metric_scene, measure_scene

    scene = build_metric_scene(tmp_path / "C.usda", fixture_visual_usd, scale=(0.001,) * 3)
    report = measure_scene(scene)
    assert all(value == 0 for value in report["physx_apis"].values())


# --------------------------------------------------------- acceptance regressions


def _audit_scene_case(*, mesh_instances=282, bbox_agree=True, scale_case=True):
    return {
        "opened": True, "bbox_methods_agree": bbox_agree,
        "mesh_instances": mesh_instances, "stage_meters_per_unit": 1.0,
        "up_axis": "Z",
        "matches_expected_envelope": scale_case,
        "physx_apis": {"CollisionAPI": 0, "RigidBodyAPI": 0,
                       "MassAPI": 0, "ArticulationRootAPI": 0},
    }


def test_acceptance_requires_all_282_meshes_not_just_global_envelope():
    from scripts.simulation.run_metric_scene_validation import acceptance_checks

    a = _audit_scene_case(scale_case=False)
    b = _audit_scene_case(scale_case=True)
    assert acceptance_checks(a, b, [0.001] * 3)["ok"] is True

    b["mesh_instances"] = 1
    verdict = acceptance_checks(a, b, [0.001] * 3)
    assert verdict["ok"] is False
    assert verdict["both_have_expected_mesh_instances"] is False


def test_acceptance_rejects_disagreeing_bbox_methods():
    from scripts.simulation.run_metric_scene_validation import acceptance_checks

    a = _audit_scene_case(scale_case=False)
    b = _audit_scene_case(scale_case=True, bbox_agree=False)
    verdict = acceptance_checks(a, b, [0.001] * 3)
    assert verdict["ok"] is False
    assert verdict["both_world_bbox_methods_agree"] is False


def test_acceptance_checks_reference_stage_units_axis_and_complete_api_keys():
    from scripts.simulation.run_metric_scene_validation import acceptance_checks

    a = _audit_scene_case(scale_case=False)
    b = _audit_scene_case(scale_case=True)
    b["stage_meters_per_unit"] = 0.001
    assert not acceptance_checks(a, b, [0.001] * 3)["ok"]
    b["stage_meters_per_unit"] = 1.0
    b["up_axis"] = "Y"
    assert not acceptance_checks(a, b, [0.001] * 3)["ok"]
    b["up_axis"] = "Z"
    b["physx_apis"].pop("MassAPI")
    assert not acceptance_checks(a, b, [0.001] * 3)["ok"]


def test_acceptance_rejects_reference_scale_error():
    from scripts.simulation.run_metric_scene_validation import acceptance_checks

    a = _audit_scene_case(scale_case=False)
    b = _audit_scene_case(scale_case=True)
    assert not acceptance_checks(a, b, [0.001, 0.001, 0.002])["ok"]
