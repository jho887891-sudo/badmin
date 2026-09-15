#!/usr/bin/env python3
"""Detection-matching and capability metrics for the shuttlecock detector.

Why this module exists: spec 03 measures one thing -- where the baseline detector's
capability starts to fall away as the target changes -- and every one of those
curves is a function of the same two primitives: how big the target is, and whether
a predicted box is the target. If each reporting script recomputed them, two curves
in the same report could disagree about what "8-12 px" or "hit" means, and the
disagreement would be invisible. Both primitives are therefore declared here once,
straight from docs/superpowers/specs/03_CONTROLLED_CAPABILITY_SPEC.md.

Size (spec 03 section 2): the shuttlecock's bounding box changes aspect ratio as its
3D pose changes, so neither width nor height alone is a stable measure of how large
the target looks. The geometric mean is the single scale:

    equivalent_size_px = sqrt(bbox_width_px * bbox_height_px)

Buckets (spec 03 section 3) are lower-inclusive and upper-exclusive, so 4.0 is in
"4-6" and 8.0 is in "8-12". Edge values therefore never fall into a gap and never
belong to two buckets; a size sweep always produces a curve with no hole in it.

Matching (spec 03 sections 5 and 6): a predicted box counts as a match when its
IoU with the ground-truth box is at least 0.5, the conventional detection
threshold. Candidates are ranked by confidence and never by proximity to the
ground truth, because that is the order a downstream stage receives them in.
Top-1 and Top-K are reported separately and are allowed to disagree: a detection
layer that keeps only the best box can destroy a true candidate that a later stage
could have recovered from the list, and Top-1 alone cannot show that. Coverage
("is the ground truth inside some candidate box at all") and the centre error of
the best match are measured as separate questions, because an oversized box can
cover the target completely while its IoU stays far below the threshold.

This module is standard library only on purpose: the capability sweep has to run on
the training host, next to the weights, without reshaping that host's Python
environment.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_metrics.py -v
    python tests/perception/shuttle_detection/test_candidate_metrics.py -v
"""
from __future__ import annotations

import math
from typing import Any, Mapping

# Spec 03 section 3, in the spec's order. The bucket a size falls into is decided
# by SIZE_BUCKET_EDGES; this tuple is the vocabulary every report writes.
SIZE_BUCKETS: tuple[str, ...] = ("<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", ">32")

# Upper edges, exclusive: a value equal to an edge belongs to the bucket that the
# edge opens, not the one it closes.
SIZE_BUCKET_EDGES: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def _require_number(value: Any, name: str) -> float:
    """Return value as a finite float, or explain why it is not one.

    A manifest column that has been left blank, mistyped or written as text must
    produce an error the caller can report, not a NumberFormatException traceback
    in the middle of a sweep and not a silently mis-bucketed sample.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(name + " must be a real number, got " + repr(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(name + " must be finite, got " + repr(value))
    return number


def equivalent_size_px(width_px: float, height_px: float) -> float:
    """The primary target-size scalar: sqrt(width_px * height_px).

    Both extents are in pixels and describe the same bounding box (spec 03
    section 2). The geometric mean is used rather than the area or the larger
    extent because the shuttlecock's box aspect ratio changes with its 3D pose: a
    cocked shuttlecock seen head-on is tall and narrow, seen from the side it is
    short and wide, and the geometric mean reports those as similar-sized targets
    while the area would over-weight the wide pose and the max extent would
    discard the other axis entirely.

    A negative extent is rejected: the square root of a negative product is not a
    small target, it is a corrupt manifest row.
    """
    width = _require_number(width_px, "width_px")
    height = _require_number(height_px, "height_px")
    if width < 0.0 or height < 0.0:
        raise ValueError(
            "bounding box extents must not be negative, got "
            + repr(width_px)
            + "x"
            + repr(height_px)
        )
    return math.sqrt(width * height)


def size_bucket(eq: float) -> str:
    """Bucket an equivalent size in pixels into the spec 03 section 3 vocabulary.

    Edges are lower-inclusive and upper-exclusive: 3.999 is "<4", 4.0 is "4-6",
    5.0 is "4-6", 6.0 is "6-8", 8.0 is "8-12", and 32.0 is ">32".

    NaN and infinity are rejected rather than bucketed. Every comparison against
    NaN is False, so a NaN would fall through to the final return and be reported
    as a measured ">32 px" target -- the most flattering possible reading of a
    broken manifest row.
    """
    value = _require_number(eq, "equivalent_size_px")
    for index, edge in enumerate(SIZE_BUCKET_EDGES):
        if value < edge:
            return SIZE_BUCKETS[index]
    return SIZE_BUCKETS[-1]


# --------------------------------------------------------------------------- #
# Box geometry
# --------------------------------------------------------------------------- #

Box = tuple[float, float, float, float]

# Manifest columns that carry the ground truth. The controlled manifests record a
# box centre and its extents rather than corners (the renderer knows both), so both
# spellings of each field are accepted.
MANIFEST_EQUIVALENT_SIZE_COLUMNS: tuple[str, ...] = ("equivalent_size_px", "equiv_size_px")
MANIFEST_BBOX_WIDTH_COLUMNS: tuple[str, ...] = ("bbox_w_px", "bbox_width_px")
MANIFEST_BBOX_HEIGHT_COLUMNS: tuple[str, ...] = ("bbox_h_px", "bbox_height_px")
MANIFEST_CENTER_X_COLUMNS: tuple[str, ...] = ("pos_x_px", "bbox_cx_px")
MANIFEST_CENTER_Y_COLUMNS: tuple[str, ...] = ("pos_y_px", "bbox_cy_px")


def _as_box(box: Any) -> Box:
    """Return an (x1, y1, x2, y2) tuple of floats."""
    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError) as error:
        raise ValueError("bounding box must be four numbers, got " + repr(box)) from error
    if len(values) != 4:
        raise ValueError("bounding box must be four numbers, got " + repr(box))
    return (values[0], values[1], values[2], values[3])


def gt_box_from_center_size(
    center_x: float, center_y: float, width_px: float, height_px: float
) -> Box:
    """Convert a manifest box centre and extents into (x1, y1, x2, y2) corners.

    The controlled manifests carry bbox_w_px / bbox_h_px and pos_x_px / pos_y_px,
    where the position is the box centre, so every comparison against a predicted
    box needs this conversion. Spelling it out once, and testing it, is what keeps
    an evaluation from silently comparing corners against centres: such a mistake
    shifts every ground-truth box by half its own size and would look like a
    detector that is merely a little imprecise.
    """
    cx = _require_number(center_x, "center_x")
    cy = _require_number(center_y, "center_y")
    width = _require_number(width_px, "width_px")
    height = _require_number(height_px, "height_px")
    if width < 0.0 or height < 0.0:
        raise ValueError(
            "bounding box extents must not be negative, got "
            + repr(width_px)
            + "x"
            + repr(height_px)
        )
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def manifest_number(row: Any, names: tuple[str, ...]) -> float | None:
    """First readable number among the named columns, or None.

    A blank cell, a missing column and a value that is not a number all return
    None; the caller reports the sample as having no usable ground truth rather
    than inventing a box for it.
    """
    if not isinstance(row, Mapping):
        return None
    for name in names:
        if name not in row:
            continue
        raw = row[name]
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    return None


def ground_truth_box_from_row(row: Any) -> Box | None:
    """Ground-truth corners from a manifest row, or None when the row has no box.

    A row with no positive box is not an error -- a hard negative is a legal
    sample -- but it cannot be matched against, so it is reported as unknown rather
    than as a box at the origin.
    """
    width = manifest_number(row, MANIFEST_BBOX_WIDTH_COLUMNS)
    height = manifest_number(row, MANIFEST_BBOX_HEIGHT_COLUMNS)
    center_x = manifest_number(row, MANIFEST_CENTER_X_COLUMNS)
    center_y = manifest_number(row, MANIFEST_CENTER_Y_COLUMNS)
    if width is None or height is None or center_x is None or center_y is None:
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return gt_box_from_center_size(center_x, center_y, width, height)


def equivalent_size_from_row(row: Any) -> float | None:
    """The target size of a manifest row, or None when the row records no box.

    A manifest that carries equivalent_size_px has already made the choice of size
    metric; otherwise it is recomputed here from the box extents, so a manifest
    written before the column existed still yields a size curve instead of an
    empty one.
    """
    recorded = manifest_number(row, MANIFEST_EQUIVALENT_SIZE_COLUMNS)
    if recorded is not None and recorded > 0.0:
        return recorded
    width = manifest_number(row, MANIFEST_BBOX_WIDTH_COLUMNS)
    height = manifest_number(row, MANIFEST_BBOX_HEIGHT_COLUMNS)
    if width is None or height is None or width <= 0.0 or height <= 0.0:
        return None
    return equivalent_size_px(width, height)


def iou_xyxy(box_a: Any, box_b: Any) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes.

    A degenerate box (zero width or height) has zero area and therefore zero IoU
    against anything, including itself: it marks no target.
    """
    ax1, ay1, ax2, ay2 = _as_box(box_a)
    bx1, by1, bx2, by2 = _as_box(box_b)
    intersection_width = min(ax2, bx2) - max(ax1, bx1)
    intersection_height = min(ay2, by2) - max(ay1, by1)
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return 0.0
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def box_center(box: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = _as_box(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_error_px(box_a: Any, box_b: Any) -> float:
    """Euclidean distance between two box centres, in pixels."""
    a_x, a_y = box_center(box_a)
    b_x, b_y = box_center(box_b)
    return math.hypot(a_x - b_x, a_y - b_y)


def union_area(rectangles: Any) -> float:
    """Total area covered by a set of axis-aligned rectangles, without double counting.

    Coordinate compression rather than pairwise subtraction: the rectangles are
    few (one ground truth against a handful of candidates), and the grid makes
    overlapping candidates covered by two boxes at once count once, which is what
    "is the ground truth covered" has to mean.
    """
    rects = [_as_box(rect) for rect in rectangles]
    rects = [rect for rect in rects if rect[2] > rect[0] and rect[3] > rect[1]]
    if not rects:
        return 0.0
    xs = sorted({rect[0] for rect in rects} | {rect[2] for rect in rects})
    ys = sorted({rect[1] for rect in rects} | {rect[3] for rect in rects})
    total = 0.0
    for x_index in range(len(xs) - 1):
        middle_x = (xs[x_index] + xs[x_index + 1]) / 2.0
        for y_index in range(len(ys) - 1):
            middle_y = (ys[y_index] + ys[y_index + 1]) / 2.0
            if any(
                rect[0] <= middle_x <= rect[2] and rect[1] <= middle_y <= rect[3]
                for rect in rects
            ):
                total += (xs[x_index + 1] - xs[x_index]) * (ys[y_index + 1] - ys[y_index])
    return total


def covered_fraction(gt_box: Any, candidate_boxes: Any) -> float:
    """Fraction of the ground-truth rectangle covered by the candidate boxes.

    This is deliberately not an IoU: a candidate far larger than the target covers
    it completely while its IoU can be arbitrarily small, and for the later stages
    of the pipeline "the target is inside this box" is the question that matters.
    """
    gt = _as_box(gt_box)
    gt_width = gt[2] - gt[0]
    gt_height = gt[3] - gt[1]
    if gt_width <= 0.0 or gt_height <= 0.0:
        return 0.0
    intersections = []
    for box in candidate_boxes:
        candidate = _as_box(box)
        x1 = max(gt[0], candidate[0])
        y1 = max(gt[1], candidate[1])
        x2 = min(gt[2], candidate[2])
        y2 = min(gt[3], candidate[3])
        if x2 > x1 and y2 > y1:
            intersections.append((x1, y1, x2, y2))
    return min(1.0, union_area(intersections) / (gt_width * gt_height))


# --------------------------------------------------------------------------- #
# Candidate ranking, Top-1/Top-K matching and coverage (spec 03 section 6)
# --------------------------------------------------------------------------- #

# A predicted box counts as a match at the conventional detection threshold. The
# same value is used for the standard metrics and for the task metrics so a
# "hit rate" and a "recall" cannot disagree about what a hit is.
DEFAULT_IOU_THRESHOLD: float = 0.5

# COCO's ten IoU thresholds; mAP50-95 is the mean of AP over this grid.
AP_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))

# Fraction of the ground-truth rectangle that candidate boxes must cover for
# gt_covered to be true. 0.9 rather than 1.0 because a box that clips a corner
# pixel of the ground truth still visibly contains the target; the value is a
# parameter of match_top_k so a report can state which one it used.
DEFAULT_GT_COVERAGE_FRACTION: float = 0.9


def _ranked_candidates(candidate_boxes: Any) -> list[dict[str, Any]]:
    """Validate candidates and return them ordered by confidence.

    Ranking is by confidence, descending, and stable: candidates with equal
    confidence keep the order the detector produced them in, so two runs of the
    same evaluation cannot disagree about which box was Top-1.
    """
    prepared: list[dict[str, Any]] = []
    for index, entry in enumerate(candidate_boxes):
        if not isinstance(entry, Mapping):
            raise ValueError("candidate " + str(index) + " is not a mapping: " + repr(entry))
        if "bbox" not in entry or entry["bbox"] is None:
            raise ValueError("candidate " + str(index) + " has no 'bbox'")
        if "confidence" not in entry or entry["confidence"] is None:
            raise ValueError("candidate " + str(index) + " has no 'confidence'")
        prepared.append(
            {
                "index": index,
                "bbox": _as_box(entry["bbox"]),
                "confidence": _require_number(entry["confidence"], "candidate confidence"),
            }
        )
    return sorted(prepared, key=lambda item: -item["confidence"])


def rank_candidates(candidate_boxes: Any) -> list[Any]:
    """Return the candidates ordered by descending confidence.

    The input sequence is neither reordered nor modified; the caller keeps its own
    list in detector order.
    """
    entries = list(candidate_boxes)
    _ranked_candidates(entries)
    return sorted(entries, key=lambda entry: -float(entry["confidence"]))


def match_top_k(
    gt_box: Any,
    candidate_boxes: Any,
    k: int,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    coverage_fraction: float = DEFAULT_GT_COVERAGE_FRACTION,
) -> dict[str, Any]:
    """Match one ground-truth box against a detector's ranked candidate list.

    Candidates are ranked by confidence, never by distance to the ground truth: the
    list a later stage receives is confidence-ordered, so that is the order the
    question "would the pipeline have kept the true box?" has to be asked in.

    Returned keys:

        top1_hit              the highest-confidence candidate matches
        topk_hit              any of the top k candidates matches (k limited by the
                              list length; k=0 examines nothing)
        top1_iou/confidence   what the Top-1 candidate was
        best_iou, best_iou_index
                              the best overlap found anywhere in the candidate list,
                              reported even when nothing matched, because it says how
                              close the miss was
        matched_index/rank/iou/confidence
                              the best candidate at or above the threshold
        gt_covered, gt_covered_fraction
                              whether some candidate covers the ground-truth
                              rectangle, which is a coverage question and not an IoU
                              question (an oversized box covers the target with a low
                              IoU)
        center_error_px       distance between the ground-truth centre and the centre
                              of the best-matching candidate. None -- undefined, not
                              zero -- when no candidate matches: there is no box whose
                              centre could be compared. It follows the best match, not
                              the Top-1 box.

    best_iou, matched_* and center_error_px consider the whole candidate list, while
    topk_hit considers only the top k: they answer different questions ("how good was
    the best box we were given" versus "what would a consumer that kept k boxes have
    seen").
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer, got " + repr(k))
    if k < 0:
        raise ValueError("k must not be negative, got " + repr(k))
    threshold = _require_number(iou_threshold, "iou_threshold")
    coverage_target = _require_number(coverage_fraction, "coverage_fraction")
    gt = _as_box(gt_box)
    ranked = _ranked_candidates(candidate_boxes)

    outcome: dict[str, Any] = {
        "n_candidates": len(ranked),
        "k": k,
        "iou_threshold": threshold,
        "gt_coverage_fraction": coverage_target,
        "top1_hit": False,
        "top1_confidence": None,
        "top1_iou": None,
        "topk_hit": False,
        "gt_covered": False,
        "gt_covered_fraction": 0.0,
        "best_iou": 0.0,
        "best_iou_index": None,
        "matched_index": None,
        "matched_rank": None,
        "matched_iou": None,
        "matched_confidence": None,
        "center_error_px": None,
    }
    if not ranked:
        return outcome

    overlaps = [iou_xyxy(gt, item["bbox"]) for item in ranked]
    outcome["best_iou"] = max(overlaps)
    outcome["best_iou_index"] = ranked[overlaps.index(outcome["best_iou"])]["index"]

    outcome["top1_confidence"] = ranked[0]["confidence"]
    outcome["top1_iou"] = overlaps[0]
    outcome["top1_hit"] = overlaps[0] >= threshold
    outcome["topk_hit"] = any(overlap >= threshold for overlap in overlaps[:k])

    for rank, (item, overlap) in enumerate(zip(ranked, overlaps), start=1):
        if overlap < threshold:
            continue
        if outcome["matched_index"] is not None and overlap <= outcome["matched_iou"]:
            continue
        outcome["matched_index"] = item["index"]
        outcome["matched_rank"] = rank
        outcome["matched_iou"] = overlap
        outcome["matched_confidence"] = item["confidence"]

    if outcome["matched_index"] is not None:
        matched = ranked[outcome["matched_rank"] - 1]
        outcome["center_error_px"] = center_error_px(gt, matched["bbox"])

    outcome["gt_covered_fraction"] = covered_fraction(gt, [item["bbox"] for item in ranked])
    outcome["gt_covered"] = outcome["gt_covered_fraction"] >= coverage_target
    return outcome


def match_candidates_to_gt(
    gt_boxes: Any,
    candidate_boxes: Any,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[tuple[float, bool]]:
    """Greedily match one sample's candidates to its ground-truth boxes.

    Returns (confidence, is_true_positive) per candidate, in the confidence order
    the candidates were examined in. Each ground-truth box is claimed at most once,
    so a duplicated detection is one hit and one false positive rather than two
    hits; without that, recall could exceed 1. Candidates are matched to the best
    still-unclaimed ground-truth box, which is what a detector that has to pick one
    box per target would do.
    """
    threshold = _require_number(iou_threshold, "iou_threshold")
    boxes = [_as_box(box) for box in gt_boxes]
    ranked = _ranked_candidates(candidate_boxes)
    claimed = [False] * len(boxes)
    result: list[tuple[float, bool]] = []
    for item in ranked:
        best_index = None
        best_overlap = 0.0
        for index, box in enumerate(boxes):
            if claimed[index]:
                continue
            overlap = iou_xyxy(box, item["bbox"])
            if overlap < threshold:
                continue
            if best_index is None or overlap > best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index is None:
            result.append((item["confidence"], False))
        else:
            claimed[best_index] = True
            result.append((item["confidence"], True))
    return result


def precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, float | None]:
    """Precision, recall and F1, with None where the quantity is undefined.

    None is not a stand-in for zero. Precision over no predictions at all is
    undefined -- there is no measurement to report -- while recall over three
    ground-truth boxes and no hits is a measured zero. A report that writes 0.0 for
    both makes "we predicted nothing" indistinguishable from "we predicted and were
    wrong every time", which is the difference between a broken run and a real
    capability limit.
    """
    predictions = true_positives + false_positives
    ground_truths = true_positives + false_negatives
    precision = true_positives / predictions if predictions > 0 else None
    recall = true_positives / ground_truths if ground_truths > 0 else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(
    detections: Any, n_ground_truth: int, recall_points: int = 101
) -> float | None:
    """Average precision from one group's (confidence, is_true_positive) pairs.

    101-point interpolated AP, as COCO computes it: precision is collapsed to its
    maximum at or beyond each recall level, so a late hit cannot be paid for by an
    early miss. None is returned only when there is nothing to measure -- no
    predictions, or no ground truth to be recalled.

    detections may be in any order; they are ranked here, so a caller cannot change
    the score by handing the pairs over unsorted.
    """
    if isinstance(recall_points, bool) or not isinstance(recall_points, int) or recall_points < 2:
        raise ValueError("recall_points must be an integer of at least 2, got " + repr(recall_points))
    if isinstance(n_ground_truth, bool) or not isinstance(n_ground_truth, int):
        raise TypeError("n_ground_truth must be an integer, got " + repr(n_ground_truth))
    if n_ground_truth < 0:
        raise ValueError("n_ground_truth must not be negative, got " + repr(n_ground_truth))
    pairs = [(float(confidence), bool(is_true_positive)) for confidence, is_true_positive in detections]
    if n_ground_truth == 0 or not pairs:
        return None

    curve: list[tuple[float, float]] = []
    true_positives = 0
    false_positives = 0
    for _, is_true_positive in sorted(pairs, key=lambda pair: -pair[0]):
        if is_true_positive:
            true_positives += 1
        else:
            false_positives += 1
        curve.append(
            (
                true_positives / n_ground_truth,
                true_positives / (true_positives + false_positives),
            )
        )

    total = 0.0
    for step in range(recall_points):
        level = step / (recall_points - 1)
        best = 0.0
        for recall, precision in curve:
            if recall >= level and precision > best:
                best = precision
        total += best
    return total / recall_points


def mean_average_precision(
    samples: Any, iou_thresholds: Any = AP_IOU_THRESHOLDS
) -> dict[str, Any]:
    """AP at IoU 0.5 and averaged over the IoU grid, for one group of samples.

    samples is a sequence of (ground_truth_boxes, candidate_boxes) pairs, one per
    sample: matching is per image, so a candidate in one image can never be a hit
    against another image's ground truth. The task has a single class, so AP over
    the group is already the mean over classes.

    Returns mAP50, mAP50-95, a reason when they cannot be computed (None, never
    0.0), the counts they were computed from, and the per-threshold AP for
    diagnosis. A group with ground truth and predictions whose detections all miss
    is a measured 0.0, which is a different finding from a group with no
    predictions at all.
    """
    thresholds = [_require_number(value, "iou_threshold") for value in iou_thresholds]
    prepared = [(list(gt_boxes), list(candidates)) for gt_boxes, candidates in samples]
    n_ground_truth = sum(len(gt_boxes) for gt_boxes, _ in prepared)
    n_predictions = sum(len(candidates) for _, candidates in prepared)

    result: dict[str, Any] = {
        "mAP50": None,
        "mAP50-95": None,
        "reason": None,
        "n_ground_truth": n_ground_truth,
        "n_predictions": n_predictions,
        "ap_per_iou": {},
    }
    if n_ground_truth == 0:
        result["reason"] = "no ground truth in this group"
        return result
    if n_predictions == 0:
        result["reason"] = "no predictions in this group"
        return result

    per_threshold: dict[float, float | None] = {}
    for threshold in thresholds:
        detections: list[tuple[float, bool]] = []
        for gt_boxes, candidates in prepared:
            detections.extend(match_candidates_to_gt(gt_boxes, candidates, threshold))
        per_threshold[threshold] = average_precision(detections, n_ground_truth)
    result["ap_per_iou"] = {str(threshold): value for threshold, value in per_threshold.items()}
    if 0.5 in per_threshold:
        result["mAP50"] = per_threshold[0.5]
    averaged = [value for value in per_threshold.values() if value is not None]
    result["mAP50-95"] = sum(averaged) / len(averaged) if averaged else None
    return result
