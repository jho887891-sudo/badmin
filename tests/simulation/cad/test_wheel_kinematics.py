"""Tests for the candidate-axis analysis (pure geometry logic, no pxr, no CAD kernel).

The point of these tests is the honesty rules, not the numbers: co-axial features
must collapse into one candidate axis, parallel-but-offset features must not, and
a cluster only reaches HIGH confidence when several features from several named
parts share it with a real radius.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.simulation.morph_one_wheel_kinematics import (  # noqa: E402
    UP_INDEX,
    _canonical,
    _confidence,
    _corner_key,
    _role_of,
    axial_segments,
    cluster_axes,
    corner_summary,
    detect_up_axis,
    module_of,
    symmetry_check,
)


def feature(direction, point, radius, eid=1, kind="CIRCLE", step_eid=100):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return {
        "entity_id": eid,
        "kind": kind,
        "radius_mm": radius,
        "direction": [float(v) for v in _canonical(d)],
        "point_mm": [float(v) for v in point],
        "step_eid": step_eid,
    }


def segments_of(features, up_axis="y"):
    """Mirror what analyse() does per cluster, for tests of the reporting layer."""
    up_index = UP_INDEX[up_axis]
    out = []
    for cluster in cluster_axes(features):
        direction = cluster["direction"]
        for segment in axial_segments(cluster):
            segment["role"] = _role_of(direction, up_axis)
            segment["direction"] = [float(v) for v in direction]
            segment["step_eids"] = sorted({m["step_eid"] for m in segment.pop("members")})
            segment["corner"] = _corner_key(segment["point_mm"], up_index)
            out.append(segment)
    return out, up_index


def test_module_of_takes_the_second_assembly_segment():
    name = "M1.2-00V3 底盘模块-1/M1.2.2-00V3 舵轮前组件-1/M1.2.2.1-00V3 舵轮转动组件-1/x-solid1"
    assert module_of(name) == "M1.2.2-00V3 舵轮前组件-1"
    assert module_of("M1.1-00V3 主体模块-1/part-solid1") == "M1.1-00V3 主体模块-1"
    assert module_of("NONE") == "UNKNOWN"
    assert module_of("") == "UNKNOWN"


def test_role_of_separates_vertical_horizontal_and_oblique():
    assert _role_of(np.array([0.0, 0.0, 1.0])) == "steering_candidate_vertical"
    assert _role_of(np.array([1.0, 0.0, 0.0])) == "drive_candidate_horizontal"
    assert _role_of(np.array([0.0, 1.0, 0.0])) == "drive_candidate_horizontal"
    assert _role_of(np.array([1.0, 0.0, 1.0])) == "oblique_or_other"


def test_coaxial_features_collapse_to_one_candidate_axis():
    features = [
        feature([0, 0, 1], [10.0, 5.0, 0.0], 20.0, eid=1),
        feature([0, 0, 1], [10.0, 5.0, 12.0], 8.0, eid=2),
        feature([0, 0, -1], [10.0, 5.0, 24.0], 3.5, eid=3),
    ]
    clusters = cluster_axes(features)
    assert len(clusters) == 1, "three features on one line must form one candidate axis"
    cluster = clusters[0]
    assert cluster["feature_count"] == 3
    assert cluster["max_radius_mm"] == 20.0
    assert cluster["extent_along_axis_mm"] == 24.0
    assert abs(float(np.dot(cluster["direction"], np.array([0, 0, 1.0])))) > 0.999999


def test_parallel_but_offset_axes_stay_separate():
    features = [
        feature([0, 0, 1], [0.0, 0.0, 0.0], 20.0, eid=1),
        feature([0, 0, 1], [0.0, 10.0, 0.0], 20.0, eid=2),
    ]
    clusters = cluster_axes(features)
    assert len(clusters) == 2, "10 mm apart is a different physical axis"


def test_different_directions_stay_separate_even_at_the_same_point():
    features = [
        feature([0, 0, 1], [0.0, 0.0, 0.0], 20.0, eid=1),
        feature([1, 0, 0], [0.0, 0.0, 0.0], 20.0, eid=2),
    ]
    assert len(cluster_axes(features)) == 2


def test_canonical_direction_is_sign_independent():
    d = np.array([0.0, 0.0, -1.0])
    assert np.allclose(_canonical(d), _canonical(-d))


def test_confidence_requires_multiple_named_parts_and_a_real_radius():
    strong = {"feature_count": 4, "max_radius_mm": 12.0}
    assert _confidence(strong, parts=[1, 2], named=2) == "HIGH"
    weak_parts = {"feature_count": 4, "max_radius_mm": 12.0}
    assert _confidence(weak_parts, parts=[1], named=1) == "MEDIUM"
    weak_size = {"feature_count": 2, "max_radius_mm": 1.0}
    assert _confidence(weak_size, parts=[1], named=0) == "LOW"


def test_detect_up_axis_uses_the_smallest_extent_and_reports_its_evidence():
    parts = [
        {"vertex_bbox_mm": {"min": [-290.0, -334.0, -400.0], "max": [290.0, 26.0, 400.0],
                            "size": [580.0, 360.0, 800.0]}},
        {"vertex_bbox_mm": {"min": [-250.0, -290.0, 200.0], "max": [250.0, -70.0, 380.0],
                            "size": [500.0, 220.0, 180.0]}},
    ]
    frame = detect_up_axis(parts)
    assert frame["axis"] == "y", "360 mm is the smallest overall extent in this fixture"
    assert frame["method"] == "smallest_overall_extent"
    assert frame["model_extents_mm"] == [580.0, 360.0, 800.0]
    assert "not a gravity direction" in frame["caveat"]


def test_detect_up_axis_honours_an_explicit_request_and_rejects_nonsense():
    parts = [{"vertex_bbox_mm": {"min": [0.0, 0.0, 0.0], "max": [1.0, 2.0, 3.0],
                                 "size": [1.0, 2.0, 3.0]}}]
    frame = detect_up_axis(parts, "z")
    assert frame["axis"] == "z" and frame["method"] == "explicit_request"
    empty = detect_up_axis([])
    assert empty["axis"] == "z" and empty["method"] == "default_no_geometry"
    try:
        detect_up_axis(parts, "w")
    except ValueError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("an unknown up axis must be rejected")


def test_role_of_treats_the_requested_up_axis_as_vertical():
    assert _role_of(np.array([0.0, 1.0, 0.0]), "y") == "steering_candidate_vertical"
    assert _role_of(np.array([1.0, 0.0, 0.0]), "y") == "drive_candidate_horizontal"
    assert _role_of(np.array([0.0, 0.0, 1.0]), "y") == "drive_candidate_horizontal"
    assert _role_of(np.array([0.0, 0.0, 1.0]), "x") == "drive_candidate_horizontal"
    assert _role_of(np.array([1.0, 0.0, 0.0]), "x") == "steering_candidate_vertical"


def test_coaxial_wheels_on_one_axle_line_are_reported_as_two_segments():
    features = [
        feature([1, 0, 0], [-250.0, -244.0, 297.0], 103.0, eid=1, step_eid=11),
        feature([1, 0, 0], [-226.0, -244.0, 297.0], 40.0, eid=2, step_eid=11),
        feature([1, 0, 0], [250.0, -244.0, 297.0], 103.0, eid=3, step_eid=11),
        feature([1, 0, 0], [226.0, -244.0, 297.0], 40.0, eid=4, step_eid=11),
    ]
    clusters = cluster_axes(features)
    assert len(clusters) == 1, "one shared line means one candidate axis"
    segments = axial_segments(clusters[0])
    assert len(segments) == 2, "a 452 mm gap must not be presented as one axle"
    for segment in segments:
        assert segment["feature_count"] == 2
        assert segment["radius_max_mm"] == 103.0


def test_corner_summary_covers_all_four_sign_combinations():
    parts_by_eid = {11: {"named": True, "module": "M1.2.2-00V3 front-1", "name_leaf": "steer-x"}}
    features = []
    eid = 1
    for x in (-250.0, 250.0):
        for z in (-297.0, 297.0):
            features.append(feature([0, 1, 0], [x, -244.0, z], 36.0, eid=eid, step_eid=11))
            features.append(feature([1, 0, 0], [x, -244.0, z], 103.0, eid=eid + 1, step_eid=11))
            eid += 2
    segments, up_index = segments_of(features, "y")
    corners = corner_summary(segments, up_index, parts_by_eid)
    assert [c["corner"] for c in corners] == ["++", "+-", "-+", "--"]
    for corner in corners:
        assert corner["steering_candidates"] == 1, corner
        assert corner["drive_candidates"] == 1, corner
        assert corner["max_steering_circle_radius_mm"] == 36.0
        assert corner["max_drive_circle_radius_mm"] == 103.0
        assert corner["named_parts"] == 1


def test_symmetry_check_accepts_mirror_symmetric_corners():
    parts_by_eid = {11: {"named": True, "module": "m", "name_leaf": "n"}}
    features = []
    eid = 1
    for x in (-250.0, 250.0):
        for z in (-297.0, 297.0):
            features.append(feature([0, 1, 0], [x, -244.0, z], 36.0, eid=eid, step_eid=11))
            eid += 1
    segments, up_index = segments_of(features, "y")
    rows = symmetry_check(corner_summary(segments, up_index, parts_by_eid), up_index)
    assert len(rows) == 4, "four mirror pairs are compared"
    assert all(row["consistent"] for row in rows), rows
    assert all(row["mirror_residual_norm_mm"] == 0.0 for row in rows)


def test_symmetry_check_flags_a_corner_that_does_not_mirror():
    parts_by_eid = {11: {"named": True, "module": "m", "name_leaf": "n"}}
    features = [
        feature([0, 1, 0], [-250.0, -244.0, 297.0], 36.0, eid=1, step_eid=11),
        feature([0, 1, 0], [255.0, -244.0, 297.0], 36.0, eid=2, step_eid=11),
        feature([0, 1, 0], [-250.0, -244.0, -297.0], 36.0, eid=3, step_eid=11),
        feature([0, 1, 0], [250.0, -244.0, -297.0], 36.0, eid=4, step_eid=11),
    ]
    segments, up_index = segments_of(features, "y")
    rows = symmetry_check(corner_summary(segments, up_index, parts_by_eid), up_index)
    bad = [row for row in rows if not row["consistent"]]
    assert bad, "a 5 mm mismatch must not be reported as symmetric"
    assert all(row["mirror_residual_norm_mm"] >= 4.9 for row in bad)

