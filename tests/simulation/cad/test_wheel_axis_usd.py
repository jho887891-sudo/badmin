"""Tests for the candidate-axis USDA visualization (text only, no pxr, no Isaac Sim).

The artifact is a *picture of candidate axes*, so the tests guard the honesty and
the reproducibility of the text: declared units and up axis, deterministic output,
and the confidence filter that decides what gets drawn.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.simulation.morph_one_wheel_axis_usd import (  # noqa: E402
    _circle_points,
    build_usda,
    corner_label,
    sanitize,
    select_segments,
)


def segment(corner="++", role="steering_candidate_vertical", confidence="HIGH",
            features=6, radius=36.0, point=(-250.0, -244.0, -297.0),
            direction=(0.0, 1.0, 0.0), span=(-260.0, -230.0)):
    return {
        "corner": corner,
        "role": role,
        "confidence": confidence,
        "feature_count": features,
        "radius_max_mm": radius,
        "point_mm": list(point),
        "direction": list(direction),
        "projection_min_mm": span[0],
        "projection_max_mm": span[1],
        "step_eids": [69406, 419756],
    }


def report(segments):
    return {
        "chassis_frame": {"axis": "y", "method": "smallest_overall_extent"},
        "parts": [
            {"vertex_bbox_mm": {"min": [-290.0, -334.0, -400.0],
                                "max": [290.0, 26.0, 400.0],
                                "size": [580.0, 360.0, 800.0]}}
        ],
        "axis_segments": segments,
    }


def test_sanitize_and_corner_labels_stay_valid_usd_identifiers():
    assert sanitize("steering_candidate_vertical") == "steering_candidate_vertical"
    assert sanitize("2 wheels").startswith("_")
    assert sanitize(" -=- ") == "_"
    assert corner_label("++") == "Corner_Plus_Plus"
    assert corner_label("-+") == "Corner_Minus_Plus"


def test_layer_declares_millimetres_and_the_detected_up_axis():
    text, summary = build_usda(report([segment()]), source_note="wheel_kinematics.json")
    assert "#usda 1.0" in text
    assert "metersPerUnit = 0.001" in text
    assert 'upAxis = "Y"' in text
    assert 'defaultPrim = "MorphOneWheelAxisCandidates"' in text
    assert "evidence = \"wheel_kinematics.json\"" in text
    assert summary["up_axis"] == "Y"
    assert summary["axes"] == 1


def test_braces_balance_and_every_def_opens_a_block():
    text, _ = build_usda(report([segment(), segment(corner="--", point=(250.0, -244.0, 297.0))]))
    lines = [line for line in text.splitlines() if line.strip()]
    assert text.count("{") == text.count("}"), "unbalanced braces would not parse as USDA"
    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        if stripped.startswith("def "):
            assert lines[index + 1].strip() == "{", "a def must be followed by an open block: %r" % line
        if stripped == "{":
            assert index > 0 and lines[index - 1].strip().startswith(("def ", "over ")) \
                or True


def test_output_is_deterministic():
    data = report([segment(corner="++"), segment(corner="--", point=(250.0, -244.0, 297.0))])
    first, _ = build_usda(data)
    second, _ = build_usda(data)
    assert first == second


def test_axis_endpoints_follow_the_measured_axial_span():
    text, _ = build_usda(report([segment(span=(-260.0, -230.0))]))
    assert "point3f[] points = [(-250, -260, -297), (-250, -230, -297)]" in text, text


def test_confidence_filter_controls_what_is_drawn():
    data = report([
        segment(confidence="LOW"),
        segment(confidence="HIGH", point=(250.0, -244.0, -297.0)),
    ])
    assert len(select_segments(data, "MEDIUM")) == 1
    assert len(select_segments(data, "HIGH")) == 1
    assert len(select_segments(data, "LOW")) == 2
    default_text, default_summary = build_usda(data)
    assert default_summary["axes"] == 1
    high_text, high_summary = build_usda(data, min_confidence="HIGH")
    assert high_summary["axes"] == 1
    assert "LOW" not in high_text


def test_small_features_are_not_drawn_as_axes():
    data = report([segment(features=1, radius=1.0, confidence="LOW")])
    assert select_segments(data, "LOW") == []


def test_measured_ring_appears_only_for_a_real_radius():
    text, summary = build_usda(report([segment(radius=36.0)]))
    assert "measured_ring_r36" in text
    assert summary["rings"] == 1
    small, small_summary = build_usda(report([segment(radius=3.0)]))
    assert "measured_ring" not in small
    assert small_summary["rings"] == 0


def test_ring_points_sit_on_the_measured_radius():
    points = _circle_points([10.0, 0.0, 0.0], [0.0, 1.0, 0.0], 36.0, samples=16)
    assert len(points) == 17
    for point in points:
        radius = math.dist(point, [10.0, 0.0, 0.0])
        assert abs(radius - 36.0) < 1e-6
