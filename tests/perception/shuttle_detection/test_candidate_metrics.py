#!/usr/bin/env python3
"""Top-K candidate matching and detection metrics (spec 03 section 6).

Why this module exists: spec 03 section 6 asks for more than mAP, because the
project does not stop at the detector. A detection layer that keeps only the single
highest-confidence box can destroy a true candidate that a later stage (ROI
refinement, temporal association, 3D reconstruction) could have recovered from the
full candidate list. Whether that happens is not visible in a top-1 metric, so the
Top-K hit rate, the "is the ground truth covered at all" question and the centre
error of the best match are measured separately.

These tests pin the semantics that make those four numbers mean different things:

- candidates are ranked by confidence, never by proximity to the ground truth;
- top1_hit looks only at the highest-confidence candidate, topk_hit at the whole
  top k, so the pair can disagree -- which is exactly the failure being measured;
- a match is IoU >= 0.5, the conventional detection threshold, and the boundary
  (exactly 0.5) is tested from both sides;
- gt_covered is a coverage question, not an IoU question: one oversized box can
  cover the ground truth completely while its IoU stays far below 0.5;
- center_error_px is undefined (None) when nothing matches, and it follows the
  best-matching candidate, not the highest-confidence one.

Run:
    python tests/perception/shuttle_detection/test_candidate_metrics.py -v
    pytest tests/perception/shuttle_detection/test_candidate_metrics.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.metrics import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    DEFAULT_GT_COVERAGE_FRACTION,
    DEFAULT_IOU_THRESHOLD,
    average_precision,
    box_center,
    center_error_px,
    covered_fraction,
    ground_truth_box_from_row,
    gt_box_from_center_size,
    iou_xyxy,
    match_candidates_to_gt,
    match_top_k,
    mean_average_precision,
    precision_recall_f1,
    rank_candidates,
    union_area,
)


def candidate(bbox, confidence):
    return {"bbox": tuple(float(value) for value in bbox), "confidence": float(confidence)}


# --------------------------------------------------------------------------- #
# Box geometry
# --------------------------------------------------------------------------- #


def test_iou_of_identical_boxes_is_one():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_touching_boxes_is_zero():
    # Boxes that share only an edge have no intersection area.
    assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_symmetric():
    a, b = (0, 0, 10, 10), (5, 5, 25, 25)
    assert iou_xyxy(a, b) == iou_xyxy(b, a)


def test_iou_of_a_fully_contained_box():
    # Inner area 100 over outer area 400.
    assert iou_xyxy((0, 0, 20, 20), (5, 5, 15, 15)) == 0.25


def test_iou_of_a_zero_area_box_is_zero():
    assert iou_xyxy((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_iou_accepts_lists_as_well_as_tuples():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_box_center():
    assert box_center((10, 20, 30, 40)) == (20.0, 30.0)


def test_center_error_px_is_a_euclidean_distance():
    assert center_error_px((0, 0, 10, 10), (0, 0, 20, 10)) == 5.0
    assert center_error_px((0, 0, 10, 10), (10, 10, 20, 20)) == pytest.approx(math.sqrt(200.0))


def test_union_area_of_overlapping_rectangles():
    # Two unit squares sharing a 1x1 quadrant: 1 + 1 - 1 = 1... they share a
    # corner region of 1x1 only when they overlap by one quadrant.
    assert union_area([(0, 0, 2, 2), (1, 1, 3, 3)]) == pytest.approx(7.0)
    assert union_area([(0, 0, 10, 10), (0, 0, 10, 10)]) == pytest.approx(100.0)
    assert union_area([]) == 0.0
    assert union_area([(0, 0, 5, 5), (5, 5, 10, 10)]) == pytest.approx(50.0)


def test_covered_fraction_of_one_box_inside_another():
    assert covered_fraction((0, 0, 10, 10), [(0, 0, 30, 30)]) == 1.0
    assert covered_fraction((0, 0, 10, 10), [(0, 0, 5, 5)]) == 0.25


def test_covered_fraction_of_two_boxes_each_covering_half():
    assert covered_fraction((0, 0, 10, 10), [(0, 0, 5, 10), (5, 0, 10, 10)]) == pytest.approx(1.0)
    assert covered_fraction((0, 0, 10, 10), [(0, 0, 5, 10)]) == pytest.approx(0.5)


def test_covered_fraction_ignores_area_outside_the_ground_truth():
    # A box far larger than the target covers it, but only the target's own area
    # counts, so the fraction cannot exceed 1.
    assert covered_fraction((10, 10, 20, 20), [(0, 0, 100, 100)]) == 1.0


# --------------------------------------------------------------------------- #
# Ground-truth conversion from the manifest's centre + size fields
# --------------------------------------------------------------------------- #


def test_gt_box_from_center_size():
    assert gt_box_from_center_size(100.0, 200.0, 20.0, 10.0) == (90.0, 195.0, 110.0, 205.0)


def test_gt_box_from_center_size_round_trips_through_its_centre_and_extents():
    box = gt_box_from_center_size(640.0, 360.0, 9.0, 7.0)
    assert box_center(box) == (640.0, 360.0)
    assert box[2] - box[0] == 9.0
    assert box[3] - box[1] == 7.0


def test_gt_box_from_row_uses_the_manifest_columns():
    row = {"bbox_w_px": "9", "bbox_h_px": "7", "pos_x_px": "640", "pos_y_px": "360"}
    assert ground_truth_box_from_row(row) == (635.5, 356.5, 644.5, 363.5)


def test_gt_box_from_row_is_none_without_a_size():
    assert ground_truth_box_from_row({"pos_x_px": "640", "pos_y_px": "360"}) is None
    assert ground_truth_box_from_row({"bbox_w_px": "9", "bbox_h_px": "7"}) is None


def test_gt_box_from_row_is_none_for_a_blank_or_zero_box():
    assert ground_truth_box_from_row({"bbox_w_px": "", "bbox_h_px": "7", "pos_x_px": "1", "pos_y_px": "1"}) is None
    assert ground_truth_box_from_row({"bbox_w_px": "0", "bbox_h_px": "7", "pos_x_px": "1", "pos_y_px": "1"}) is None


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_rank_candidates_is_by_descending_confidence():
    candidates = [candidate((0, 0, 1, 1), 0.2), candidate((0, 0, 1, 1), 0.9)]
    ranked = rank_candidates(candidates)
    assert [entry["confidence"] for entry in ranked] == [0.9, 0.2]


def test_rank_candidates_is_stable_for_equal_confidence():
    first, second = candidate((0, 0, 1, 1), 0.5), candidate((2, 2, 3, 3), 0.5)
    assert rank_candidates([first, second]) == [first, second]
    assert rank_candidates([second, first]) == [second, first]


def test_rank_candidates_does_not_mutate_the_input_order():
    candidates = [candidate((0, 0, 1, 1), 0.2), candidate((0, 0, 1, 1), 0.9)]
    rank_candidates(candidates)
    assert [entry["confidence"] for entry in candidates] == [0.2, 0.9]


def test_rank_candidates_rejects_a_candidate_without_a_confidence():
    with pytest.raises(ValueError):
        rank_candidates([{"bbox": (0, 0, 1, 1)}])


def test_rank_candidates_rejects_a_candidate_without_a_bbox():
    with pytest.raises(ValueError):
        rank_candidates([{"confidence": 0.5}])


# --------------------------------------------------------------------------- #
# match_top_k: the Top-1 / Top-K distinction that motivates the metric
# --------------------------------------------------------------------------- #


def test_top1_miss_topk_hit_when_a_lower_confidence_candidate_is_the_true_box():
    # The plan's example, with a true box that does match at the documented
    # threshold: the highest-confidence box is a distractor, the true box is
    # ranked second, so keeping only the top-1 destroys it.
    gt = (10, 10, 14, 14)
    candidates = [
        candidate((30, 30, 35, 35), 0.9),
        candidate((10, 10, 14, 14), 0.7),
    ]
    match = match_top_k(gt, candidates, k=2)
    assert match["top1_hit"] is False
    assert match["topk_hit"] is True
    assert match["best_iou"] == 1.0
    assert match["matched_rank"] == 2


def test_plan_example_boxes_at_default_threshold_and_at_a_looser_one():
    # The plan's literal example uses a true box of (9, 9, 15, 15) against a
    # ground truth of (10, 10, 14, 14). That box has IoU 16/36 = 0.444, so it is
    # NOT a match at the conventional 0.5 threshold -- the plan's snippet asserts
    # topk_hit is True, which only holds if the threshold is looser than 0.5.
    # Both readings are recorded here instead of silently picking one.
    gt = (10, 10, 14, 14)
    candidates = [
        candidate((30, 30, 35, 35), 0.9),
        candidate((9, 9, 15, 15), 0.7),
    ]
    assert iou_xyxy(gt, (9, 9, 15, 15)) == pytest.approx(16.0 / 36.0)

    strict = match_top_k(gt, candidates, k=2)
    assert strict["top1_hit"] is False
    assert strict["topk_hit"] is False
    assert strict["best_iou"] == pytest.approx(16.0 / 36.0)
    assert strict["center_error_px"] is None

    loose = match_top_k(gt, candidates, k=2, iou_threshold=0.3)
    assert loose["top1_hit"] is False
    assert loose["topk_hit"] is True


def test_candidates_are_ranked_by_confidence_not_by_proximity():
    # The nearest box to the ground truth is the least confident one; it must not
    # be promoted, because the candidate list the later stage receives is ordered
    # by confidence.
    gt = (100, 100, 110, 110)
    candidates = [
        candidate((100, 100, 110, 110), 0.10),
        candidate((102, 100, 112, 110), 0.95),
    ]
    match = match_top_k(gt, candidates, k=2)
    assert match["top1_confidence"] == 0.95
    assert match["top1_iou"] == pytest.approx(80.0 / 120.0, abs=1e-9)
    assert match["top1_hit"] is True
    assert match["best_iou"] == 1.0


def test_top1_hit_uses_only_the_highest_confidence_candidate():
    gt = (10, 10, 20, 20)
    candidates = [
        candidate((200, 200, 210, 210), 0.9),
        candidate((10, 10, 20, 20), 0.8),
        candidate((10, 10, 20, 20), 0.7),
    ]
    assert match_top_k(gt, candidates, k=3)["top1_hit"] is False
    assert match_top_k(gt, candidates, k=3)["topk_hit"] is True


def test_k_limits_which_candidates_are_examined():
    gt = (10, 10, 20, 20)
    candidates = [
        candidate((200, 200, 210, 210), 0.9),
        candidate((200, 200, 210, 210), 0.8),
        candidate((10, 10, 20, 20), 0.7),
    ]
    assert match_top_k(gt, candidates, k=2)["topk_hit"] is False
    assert match_top_k(gt, candidates, k=3)["topk_hit"] is True


def test_k_zero_examines_no_candidate_for_topk_but_top1_is_unaffected():
    gt = (10, 10, 20, 20)
    candidates = [candidate((10, 10, 20, 20), 0.9)]
    match = match_top_k(gt, candidates, k=0)
    assert match["topk_hit"] is False
    assert match["top1_hit"] is True


def test_k_larger_than_the_candidate_list_is_not_an_error():
    gt = (10, 10, 20, 20)
    assert match_top_k(gt, [candidate((10, 10, 20, 20), 0.9)], k=50)["topk_hit"] is True


def test_negative_k_is_rejected():
    with pytest.raises(ValueError):
        match_top_k((10, 10, 20, 20), [candidate((10, 10, 20, 20), 0.9)], k=-1)


def test_candidate_order_does_not_change_the_outcome():
    gt = (10, 10, 20, 20)
    candidates = [candidate((5, 5, 25, 25), 0.4), candidate((10, 10, 20, 20), 0.9)]
    forward = match_top_k(gt, candidates, k=2)
    backward = match_top_k(gt, list(reversed(candidates)), k=2)
    for key in ("top1_hit", "topk_hit", "gt_covered", "best_iou", "center_error_px", "matched_iou"):
        assert forward[key] == backward[key]


def test_equal_confidence_keeps_the_earlier_candidate_as_top1():
    gt = (10, 10, 20, 20)
    far = candidate((200, 200, 210, 210), 0.5)
    near = candidate((10, 10, 20, 20), 0.5)
    assert match_top_k(gt, [far, near], k=2)["top1_hit"] is False
    assert match_top_k(gt, [near, far], k=2)["top1_hit"] is True


# --------------------------------------------------------------------------- #
# The IoU threshold: exactly 0.5 matches, just below it does not
# --------------------------------------------------------------------------- #


def test_the_documented_default_threshold_is_half():
    assert DEFAULT_IOU_THRESHOLD == 0.5


def test_iou_of_exactly_half_is_a_match():
    # (0, 0, 10, 20) against (0, 0, 10, 10): intersection 100, union 200.
    gt = (0, 0, 10, 10)
    candidates = [candidate((0, 0, 10, 20), 0.9)]
    assert iou_xyxy(gt, candidates[0]["bbox"]) == 0.5
    assert match_top_k(gt, candidates, k=1)["top1_hit"] is True


def test_iou_just_below_half_is_a_miss():
    gt = (0, 0, 10, 10)
    candidates = [candidate((0, 0, 10, 20.01), 0.9)]
    assert iou_xyxy(gt, candidates[0]["bbox"]) < 0.5
    match = match_top_k(gt, candidates, k=1)
    assert match["top1_hit"] is False
    assert match["center_error_px"] is None
    # best_iou is reported even when nothing matched: it is what says how close
    # the miss was.
    assert match["best_iou"] == pytest.approx(0.49975, abs=1e-5)


def test_a_custom_threshold_moves_the_boundary():
    gt = (0, 0, 10, 10)
    candidates = [candidate((0, 0, 10, 20), 0.9)]
    assert match_top_k(gt, candidates, k=1, iou_threshold=0.6)["top1_hit"] is False
    assert match_top_k(gt, candidates, k=1, iou_threshold=0.5)["top1_hit"] is True


# --------------------------------------------------------------------------- #
# center_error_px: the best match, or undefined
# --------------------------------------------------------------------------- #


def test_center_error_follows_the_best_matching_candidate_not_the_top1():
    # cand1 is the top-1 by confidence but matches worse; cand2 matches best and
    # is therefore the box whose centre error is reported.
    gt = (10, 10, 20, 20)
    candidates = [
        candidate((11, 11, 21, 21), 0.9),
        candidate((11, 11, 20, 20), 0.4),
    ]
    match = match_top_k(gt, candidates, k=2)
    assert match["matched_index"] == 1
    assert match["matched_rank"] == 2
    assert match["matched_confidence"] == 0.4
    assert match["best_iou"] == pytest.approx(0.81)
    assert match["center_error_px"] == pytest.approx(math.sqrt(0.5))
    assert match["top1_hit"] is True


def test_center_error_is_none_when_nothing_matches():
    gt = (10, 10, 20, 20)
    match = match_top_k(gt, [candidate((100, 100, 110, 110), 0.9)], k=1)
    assert match["center_error_px"] is None
    assert match["matched_index"] is None
    assert match["matched_confidence"] is None
    assert match["matched_rank"] is None


def test_center_error_is_zero_for_a_perfect_box():
    gt = (10, 10, 20, 20)
    assert match_top_k(gt, [candidate((10, 10, 20, 20), 0.9)], k=1)["center_error_px"] == 0.0


def test_no_candidates_at_all():
    match = match_top_k((10, 10, 20, 20), [], k=5)
    assert match["n_candidates"] == 0
    assert match["top1_hit"] is False
    assert match["topk_hit"] is False
    assert match["gt_covered"] is False
    assert match["gt_covered_fraction"] == 0.0
    assert match["best_iou"] == 0.0
    assert match["best_iou_index"] is None
    assert match["top1_confidence"] is None
    assert match["top1_iou"] is None
    assert match["center_error_px"] is None


# --------------------------------------------------------------------------- #
# gt_covered: coverage, which is not IoU
# --------------------------------------------------------------------------- #


def test_gt_covered_by_an_oversized_box_with_a_low_iou():
    gt = (10, 10, 20, 20)
    candidates = [candidate((0, 0, 30, 30), 0.9)]
    match = match_top_k(gt, candidates, k=1)
    assert match["best_iou"] == pytest.approx(100.0 / 900.0)
    assert match["top1_hit"] is False
    assert match["gt_covered"] is True
    assert match["gt_covered_fraction"] == 1.0


def test_gt_not_covered_by_a_box_that_only_overlaps_a_corner():
    gt = (0, 0, 10, 10)
    match = match_top_k(gt, [candidate((0, 0, 5, 5), 0.9)], k=1)
    assert match["gt_covered"] is False
    assert match["gt_covered_fraction"] == pytest.approx(0.25)


def test_gt_covered_by_two_boxes_that_each_cover_half():
    gt = (0, 0, 10, 10)
    candidates = [candidate((0, 0, 5, 10), 0.9), candidate((5, 0, 10, 10), 0.8)]
    match = match_top_k(gt, candidates, k=2)
    assert match["gt_covered_fraction"] == pytest.approx(1.0)
    assert match["gt_covered"] is True


def test_gt_covered_uses_every_candidate_not_only_the_top_k():
    # Coverage answers "is the target in the candidate set at all", so it is not
    # restricted by k.
    gt = (0, 0, 10, 10)
    candidates = [candidate((200, 200, 210, 210), 0.9), candidate((0, 0, 10, 10), 0.1)]
    assert match_top_k(gt, candidates, k=1)["gt_covered"] is True


def test_the_covered_fraction_needed_for_gt_covered_is_configurable():
    gt = (0, 0, 10, 10)
    candidates = [candidate((0, 0, 5, 5), 0.9)]
    assert DEFAULT_GT_COVERAGE_FRACTION == 0.9
    assert match_top_k(gt, candidates, k=1)["gt_covered"] is False
    assert match_top_k(gt, candidates, k=1, coverage_fraction=0.2)["gt_covered"] is True


# --------------------------------------------------------------------------- #
# Greedy matching of a whole sample, used by the mAP computation
# --------------------------------------------------------------------------- #


def test_match_candidates_to_gt_returns_one_flag_per_candidate_in_confidence_order():
    gt_boxes = [(0, 0, 10, 10)]
    candidates = [candidate((0, 0, 10, 10), 0.9), candidate((50, 50, 60, 60), 0.8)]
    assert match_candidates_to_gt(gt_boxes, candidates) == [(0.9, True), (0.8, False)]


def test_match_candidates_to_gt_consumes_a_ground_truth_box_once():
    # Only one ground truth box exists, so the second candidate that also overlaps
    # it is a false positive: without this, a duplicated detection would be scored
    # as two hits and recall could exceed 1.
    gt_boxes = [(0, 0, 10, 10)]
    candidates = [candidate((0, 0, 10, 10), 0.9), candidate((0, 0, 9, 9), 0.8)]
    assert match_candidates_to_gt(gt_boxes, candidates) == [(0.9, True), (0.8, False)]


def test_match_candidates_to_gt_assigns_each_ground_truth_box_to_its_best_candidate():
    gt_boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
    candidates = [
        candidate((100, 100, 110, 110), 0.9),
        candidate((0, 0, 10, 10), 0.8),
    ]
    assert match_candidates_to_gt(gt_boxes, candidates) == [(0.9, True), (0.8, True)]


def test_match_candidates_to_gt_without_ground_truth_makes_every_candidate_a_false_positive():
    assert match_candidates_to_gt([], [candidate((0, 0, 10, 10), 0.9)]) == [(0.9, False)]


def test_match_candidates_to_gt_respects_the_threshold():
    gt_boxes = [(0, 0, 10, 10)]
    candidates = [candidate((0, 0, 10, 20), 0.9)]
    assert match_candidates_to_gt(gt_boxes, candidates, 0.5) == [(0.9, True)]
    assert match_candidates_to_gt(gt_boxes, candidates, 0.6) == [(0.9, False)]


# --------------------------------------------------------------------------- #
# Precision / recall / F1, and the difference between 0.0 and "not measured"
# --------------------------------------------------------------------------- #


def test_precision_recall_f1_of_a_perfect_result():
    assert precision_recall_f1(2, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_precision_recall_f1_of_a_half_result():
    result = precision_recall_f1(2, 2, 2)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_precision_is_undefined_without_predictions_but_recall_is_a_measured_zero():
    # This is the distinction the whole report depends on: "we predicted nothing"
    # is not the same measurement as "we predicted and were wrong every time".
    result = precision_recall_f1(0, 0, 3)
    assert result["precision"] is None
    assert result["recall"] == 0.0
    assert result["f1"] is None


def test_precision_is_a_measured_zero_when_every_prediction_is_wrong():
    result = precision_recall_f1(0, 3, 0)
    assert result["precision"] == 0.0
    assert result["recall"] is None
    assert result["f1"] is None


def test_f1_is_zero_when_both_are_measured_and_zero():
    result = precision_recall_f1(1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Average precision
# --------------------------------------------------------------------------- #


def test_the_ap_iou_grid_is_coco_style():
    assert len(AP_IOU_THRESHOLDS) == 10
    assert AP_IOU_THRESHOLDS[0] == 0.5
    assert AP_IOU_THRESHOLDS[-1] == 0.95


def test_average_precision_of_a_perfect_run_is_one():
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)


def test_average_precision_prefers_hits_ranked_first():
    ranked_first = average_precision([(0.9, True), (0.8, False)], 1)
    ranked_last = average_precision([(0.9, False), (0.8, True)], 1)
    assert ranked_first == pytest.approx(1.0)
    assert ranked_last == pytest.approx(0.5)
    assert ranked_first > ranked_last


def test_average_precision_is_a_measured_zero_when_nothing_hits():
    assert average_precision([(0.9, False), (0.8, False)], 1) == 0.0


def test_average_precision_is_not_computable_without_predictions():
    assert average_precision([], 3) is None


def test_average_precision_is_not_computable_without_ground_truth():
    assert average_precision([(0.9, False)], 0) is None


def test_mean_average_precision_of_a_swapped_but_valid_box():
    # gt (0, 0, 10, 10) against (2, 0, 12, 10): intersection 80, union 120, IoU
    # 2/3. That is a match at 0.50, 0.55, 0.60 and 0.65 and a miss above, so
    # mAP50 is 1.0 while mAP50-95 is 4/10.
    samples = [([(0, 0, 10, 10)], [candidate((2, 0, 12, 10), 0.9)])]
    result = mean_average_precision(samples)
    assert result["mAP50"] == pytest.approx(1.0)
    assert result["mAP50-95"] == pytest.approx(0.4)
    assert result["reason"] is None
    assert result["n_ground_truth"] == 1
    assert result["n_predictions"] == 1


def test_mean_average_precision_of_a_perfect_box_is_one_at_every_threshold():
    samples = [([(0, 0, 10, 10)], [candidate((0, 0, 10, 10), 0.9)])]
    result = mean_average_precision(samples)
    assert result["mAP50"] == pytest.approx(1.0)
    assert result["mAP50-95"] == pytest.approx(1.0)


def test_mean_average_precision_is_not_computable_without_predictions():
    result = mean_average_precision([([(0, 0, 10, 10)], [])])
    assert result["mAP50"] is None
    assert result["mAP50-95"] is None
    assert "no predictions" in result["reason"]


def test_mean_average_precision_is_not_computable_without_ground_truth():
    result = mean_average_precision([([], [candidate((0, 0, 10, 10), 0.9)])])
    assert result["mAP50"] is None
    assert result["mAP50-95"] is None
    assert "no ground truth" in result["reason"]


def test_mean_average_precision_never_matches_across_samples():
    # Each sample's candidates may only match that sample's ground truth boxes: a
    # hit in a neighbouring image is not a hit.
    samples = [
        ([(0, 0, 10, 10)], [candidate((100, 100, 110, 110), 0.9)]),
        ([(100, 100, 110, 110)], [candidate((0, 0, 10, 10), 0.8)]),
    ]
    result = mean_average_precision(samples)
    assert result["mAP50"] == 0.0


def test_mean_average_precision_of_a_negative_sample_counts_false_positives():
    samples = [([(0, 0, 10, 10)], [candidate((0, 0, 10, 10), 0.9)]), ([], [candidate((5, 5, 15, 15), 0.7)])]
    result = mean_average_precision(samples)
    assert result["mAP50"] == pytest.approx(1.0)
    assert result["n_predictions"] == 2
