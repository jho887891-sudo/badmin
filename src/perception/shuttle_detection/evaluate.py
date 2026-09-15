#!/usr/bin/env python3
"""Controlled-capability evaluation of the shuttlecock detector (plan Task 3).

Why this module exists: spec 03 does not ask how high mAP is. It asks where the
baseline detector stops working as the target gets smaller, turns, moves, blurs or
becomes occluded, and that question needs one evaluator that consumes a controlled
manifest, runs a detector over it and reports the same metrics grouped by every
controlled variable. One script per variable is how two curves in one report end up
disagreeing about what a hit is.

The detector is injected as a callable:

    detector(image_path: Path) -> Sequence[{"bbox": (x1, y1, x2, y2), "confidence": float}]

so this module and its tests run with no GPU, no weights and no Ultralytics. The
Ultralytics backend lives in the CLI adapter
(scripts/shuttle_detection/evaluate_controlled.py) and is imported inside a function
body there, never at module scope. The evaluator never opens an image itself: file
access belongs to the detector, which is why a unit test can hand it a path that
does not exist.

Reporting rules, both of which the tests pin:

- A metric that could not be measured is written as the text
  "NOT_COMPUTABLE: <reason>", never as 0.0. Precision over a group with no
  predictions is undefined; recall over the same group is a measured zero (there
  were ground-truth boxes and none was found). Writing 0.0 for both would make a
  detector that returned nothing indistinguishable from one that returned wrong
  boxes -- the difference between a broken run and a capability limit.
- A dimension whose manifest column is absent, and which cannot be derived, is
  listed in summary.json under "not_computed", carried in summary["warnings"], and
  its metrics file is written with a header and no rows. A conditioned curve that
  could not be produced must be visible in the output, not silently missing from it.

Manifest columns understood (any that are absent are reported, not guessed):
    file, split, source_type, bbox_w_px, bbox_h_px, pos_x_px, pos_y_px,
    yaw_deg, pitch_deg, roll_deg, pose_bucket, position_bucket, blur_bucket,
    occlusion_bucket, background, light_azimuth_deg, equivalent_size_px
Ground truth is recorded as a box centre (pos_x_px/pos_y_px) plus extents
(bbox_w_px/bbox_h_px), so it is converted to corners by
metrics.ground_truth_box_from_row before any matching happens.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_controlled_evaluator.py -v
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dataset_audit import UNKNOWN_BUCKET, bucket_pose, read_manifest
from .metrics import (
    AP_IOU_THRESHOLDS,
    DEFAULT_GT_COVERAGE_FRACTION,
    DEFAULT_IOU_THRESHOLD,
    Box,
    equivalent_size_from_row,
    ground_truth_box_from_row,
    manifest_number,
    match_candidates_to_gt,
    match_top_k,
    mean_average_precision,
    precision_recall_f1,
    size_bucket,
)

Detector = Callable[[Path], Sequence[Mapping[str, Any]]]

PREDICTION_FILENAME = "predictions.csv"
SUMMARY_FILENAME = "summary.json"

# Written by every group file when a metric could not be measured. The reason is
# always appended, so a reader never has to guess which of "undefined" and "zero"
# a cell means.
NOT_COMPUTABLE = "NOT_COMPUTABLE"

# Samples whose controlled variable is blank land here instead of being dropped:
# a label that is missing for part of a manifest has to be visible as its own row.
UNLABELLED_GROUP = "<unlabelled>"

STATUS_MANIFEST_COLUMN = "manifest_column"
STATUS_DERIVED = "derived"
STATUS_NOT_COMPUTED = "not_computed"

SAMPLE_OK = "ok"
SAMPLE_NEGATIVE = "negative"
SAMPLE_MISSING_GROUND_TRUTH = "missing_ground_truth"
SAMPLE_DETECTOR_ERROR = "detector_error"

_TRUTHY = {"1", "true", "yes", "y", "t"}

IMAGE_WIDTH_COLUMNS: tuple[str, ...] = ("img_w_px", "image_width_px", "width_px")
IMAGE_HEIGHT_COLUMNS: tuple[str, ...] = ("img_h_px", "image_height_px", "height_px")
IMAGE_SIZE_COLUMNS: tuple[str, ...] = ("imgsz", "img_size", "image_size_px")
POSITION_X_COLUMNS: tuple[str, ...] = ("pos_x_px", "bbox_cx_px")
POSITION_Y_COLUMNS: tuple[str, ...] = ("pos_y_px", "bbox_cy_px")
POSE_ANGLE_COLUMNS: tuple[str, ...] = ("yaw_deg", "pitch_deg", "roll_deg")

# Left/middle/right and top/middle/bottom thirds of the frame. The nine cells carry
# exactly the positions spec 03 section 4.3 requires: center, left, right, top,
# bottom and the four corners.
THIRD_NAMES: tuple[str, ...] = ("first", "middle", "last")
COLUMN_POSITION_NAMES: dict[int, str] = {0: "left", 1: "center", 2: "right"}
ROW_POSITION_NAMES: dict[int, str] = {0: "top", 2: "bottom"}

PREDICTION_COLUMNS: tuple[str, ...] = (
    "index",
    "file",
    "split",
    "source_type",
    "status",
    "image_path",
    "bbox_w_px",
    "bbox_h_px",
    "pos_x_px",
    "pos_y_px",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "gt_width_px",
    "gt_height_px",
    "equivalent_size_px",
    "size_bucket",
    "pose_bucket",
    "position_bucket",
    "blur_bucket",
    "occlusion_bucket",
    "background",
    "light_azimuth_deg",
    "n_candidates",
    "top1_confidence",
    "top1_iou",
    "top1_hit",
    "topk_hit",
    "gt_covered",
    "gt_covered_fraction",
    "center_error_px",
    "center_error_defined",
    "matched_rank",
    "matched_confidence",
    "best_iou",
    "n_true_positives",
    "n_false_positives",
    "n_false_negatives",
    "iou_threshold",
    "top_k",
    "error",
)

GROUP_METRIC_COLUMNS: tuple[str, ...] = (
    "n_samples",
    "n_evaluable",
    "n_negative",
    "n_missing_ground_truth",
    "n_detector_errors",
    "n_ground_truth",
    "n_predictions",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP50-95",
    "top1_hit_rate",
    "topk_hit_rate",
    "gt_covered_rate",
    "mean_center_error_px",
    "n_center_error",
    "mean_confidence",
    "n_confidence",
    "mean_matched_confidence",
    "n_matched_confidence",
    "mean_best_iou",
    "iou_threshold",
    "top_k",
    "notes",
)


@dataclass(frozen=True)
class DimensionSpec:
    """One controlled variable, and the file its grouped metrics are written to."""

    dimension: str
    file_name: str
    label_columns: tuple[str, ...]


# Background has no file of its own in the plan's output list, but spec 03 section
# 4.6 requires background to be recorded and section 8 requires the conditioned
# metrics, so it gets a file of the same shape rather than being folded into another.
DIMENSION_SPECS: tuple[DimensionSpec, ...] = (
    DimensionSpec("size_bucket", "size_bucket_metrics.csv", ("size_bucket",)),
    DimensionSpec("pose", "pose_metrics.csv", ("pose_bucket",)),
    DimensionSpec("position", "position_metrics.csv", ("position_bucket",)),
    DimensionSpec("blur", "blur_metrics.csv", ("blur_bucket", "blur")),
    DimensionSpec("occlusion", "occlusion_metrics.csv", ("occlusion_bucket",)),
    DimensionSpec("background", "background_metrics.csv", ("background",)),
)

DIMENSION_SPECS_BY_NAME: dict[str, DimensionSpec] = {
    spec.dimension: spec for spec in DIMENSION_SPECS
}


@dataclass(frozen=True)
class ControlledEvaluationConfig:
    """What the evaluation is allowed to assume.

    iou_threshold is the match threshold for both the standard and the task metrics,
    so a group's "recall" and its "top-k hit rate" cannot disagree about what a hit
    is. min_confidence is applied by the evaluator on top of whatever the detector
    already returned; it defaults to 0.0 so the evaluator never invents a threshold
    the detector did not use.
    """

    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    top_k: int = 5
    min_confidence: float = 0.0
    expected_class: int | None = 0
    image_root: Path | None = None
    gt_coverage_fraction: float = DEFAULT_GT_COVERAGE_FRACTION


@dataclass(frozen=True)
class DimensionPlan:
    """How one controlled variable will be read from the manifest."""

    spec: DimensionSpec
    status: str
    source: str
    reason: str | None

    @property
    def dimension(self) -> str:
        return self.spec.dimension

    @property
    def resolved(self) -> bool:
        return self.status != STATUS_NOT_COMPUTED


@dataclass(frozen=True)
class ControlledEvaluationReport:
    """Everything the CLI needs to write its output files."""

    predictions: list[dict[str, Any]]
    groups: dict[str, list[dict[str, Any]]]
    summary: dict[str, Any]
    column_plans: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Manifest reading: dimensions, labels and the ground-truth conversion
# --------------------------------------------------------------------------- #


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _truthy(value: Any) -> bool:
    return _text(value).lower() in _TRUTHY


def _first_text(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        if name in row:
            value = _text(row[name])
            if value:
                return value
    return ""


def image_size_from_row(row: Mapping[str, Any]) -> tuple[float, float] | None:
    """Image size in pixels from a manifest row, or None when it is not recorded.

    Needed to turn a target centre into a position class: pos_x_px alone says
    nothing about whether the target is in the middle of the frame or at its edge.
    """
    width = manifest_number(row, IMAGE_WIDTH_COLUMNS)
    height = manifest_number(row, IMAGE_HEIGHT_COLUMNS)
    if width and height and width > 0.0 and height > 0.0:
        return (width, height)
    side = manifest_number(row, IMAGE_SIZE_COLUMNS)
    if side and side > 0.0:
        return (side, side)
    return None


def _third_index(value: float, extent: float) -> int:
    if extent <= 0.0:
        return 1
    index = int(value / (extent / 3.0))
    return min(2, max(0, index))


def position_label(row: Mapping[str, Any]) -> str | None:
    """Position class of the target centre (spec 03 section 4.3).

    The frame is cut into thirds each way, which yields exactly the positions the
    spec names: center, left, right, top, bottom and the four corners.
    """
    center_x = manifest_number(row, POSITION_X_COLUMNS)
    center_y = manifest_number(row, POSITION_Y_COLUMNS)
    size = image_size_from_row(row)
    if center_x is None or center_y is None or size is None:
        return None
    column = _third_index(center_x, size[0])
    row_index = _third_index(center_y, size[1])
    # The names are the ones spec 03 section 4.3 lists: center, left, right, top,
    # bottom and the four corners -- a target in the middle column is "top" rather
    # than "top_center", because that is the vocabulary the curves are plotted in.
    if column == 1:
        return "center" if row_index == 1 else ROW_POSITION_NAMES[row_index]
    if row_index == 1:
        return COLUMN_POSITION_NAMES[column]
    return ROW_POSITION_NAMES[row_index] + "_" + COLUMN_POSITION_NAMES[column]


def plan_dimensions(columns: Sequence[str]) -> list[DimensionPlan]:
    """Decide, from the manifest header alone, which curves this run can produce.

    The decision is made from the header rather than from the rows so that a
    dimension missing everywhere is reported once, instead of appearing as a group
    of unlabelled samples that looks like a measurement.
    """
    present = {_text(column) for column in columns}
    plans: list[DimensionPlan] = []
    for spec in DIMENSION_SPECS:
        for column in spec.label_columns:
            if column in present:
                plans.append(
                    DimensionPlan(
                        spec, STATUS_MANIFEST_COLUMN, "manifest column '" + column + "'", None
                    )
                )
                break
        else:
            plans.append(_derive_plan(spec, present))
    return plans


def _derive_plan(spec: DimensionSpec, present: set[str]) -> DimensionPlan:
    if spec.dimension == "size_bucket":
        for column in ("equivalent_size_px", "equiv_size_px"):
            if column in present:
                return DimensionPlan(
                    spec,
                    STATUS_DERIVED,
                    "derived from manifest column '" + column + "'",
                    None,
                )
        if "bbox_w_px" in present and "bbox_h_px" in present:
            return DimensionPlan(
                spec,
                STATUS_DERIVED,
                "derived from bbox_w_px * bbox_h_px",
                None,
            )
        return DimensionPlan(
            spec,
            STATUS_NOT_COMPUTED,
            "",
            "no size column (size_bucket, equivalent_size_px, bbox_w_px, bbox_h_px) "
            "in the manifest",
        )
    if spec.dimension == "pose":
        if any(column in present for column in POSE_ANGLE_COLUMNS):
            return DimensionPlan(
                spec,
                STATUS_DERIVED,
                "derived from yaw_deg/pitch_deg/roll_deg via dataset_audit.bucket_pose",
                None,
            )
        return DimensionPlan(
            spec,
            STATUS_NOT_COMPUTED,
            "",
            "no pose column (pose_bucket, yaw_deg, pitch_deg, roll_deg) in the manifest; "
            "a named 3D pose cannot be recovered from the image alone",
        )
    if spec.dimension == "position":
        has_position = any(column in present for column in POSITION_X_COLUMNS) and any(
            column in present for column in POSITION_Y_COLUMNS
        )
        has_size = any(column in present for column in IMAGE_SIZE_COLUMNS) or (
            any(column in present for column in IMAGE_WIDTH_COLUMNS)
            and any(column in present for column in IMAGE_HEIGHT_COLUMNS)
        )
        if has_position and has_size:
            return DimensionPlan(
                spec,
                STATUS_DERIVED,
                "derived from pos_x_px/pos_y_px and the recorded image size",
                None,
            )
        missing = []
        if not has_position:
            missing.append("pos_x_px/pos_y_px")
        if not has_size:
            missing.append("an image size (imgsz or img_w_px/img_h_px)")
        return DimensionPlan(
            spec,
            STATUS_NOT_COMPUTED,
            "",
            "position requires " + " and ".join(missing) + " to place the target",
        )
    return DimensionPlan(
        spec,
        STATUS_NOT_COMPUTED,
        "",
        "no " + "/".join(spec.label_columns) + " column in the manifest, and the "
        "value cannot be derived from the image",
    )


def dimension_label(plan: DimensionPlan, row: Mapping[str, Any]) -> str | None:
    """The group label a row belongs to for one dimension, or None when blank."""
    if not plan.resolved:
        return None
    if plan.status == STATUS_MANIFEST_COLUMN:
        return _first_text(row, plan.spec.label_columns) or None
    if plan.dimension == "size_bucket":
        size = equivalent_size_from_row(row)
        return None if size is None else size_bucket(size)
    if plan.dimension == "pose":
        label = bucket_pose(row.get("yaw_deg"), row.get("pitch_deg"), row.get("roll_deg"))
        return None if label == UNKNOWN_BUCKET else label
    if plan.dimension == "position":
        return position_label(row)
    return None


def resolve_image_path(manifest_path: Path, row: Mapping[str, Any], config: ControlledEvaluationConfig) -> Path:
    """Absolute-ish path handed to the detector.

    Relative file names resolve against --images when it is given and against the
    manifest's own directory otherwise; a manifest written next to its images then
    works with no extra argument.
    """
    file_name = _text(row.get("file"))
    path = Path(file_name)
    if path.is_absolute():
        return path
    base = Path(config.image_root) if config.image_root is not None else Path(manifest_path).parent
    return base / path


def _normalise_candidates(raw: Any, config: ControlledEvaluationConfig) -> list[dict[str, Any]]:
    """Validate a detector's output and drop candidates the evaluation excludes.

    A malformed entry raises: the detector contract is part of the run's identity,
    and a silently dropped box would look like a miss.
    """
    candidates: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError("candidate " + str(index) + " is not a mapping: " + repr(entry))
        if entry.get("bbox") is None:
            raise ValueError("candidate " + str(index) + " has no 'bbox'")
        if entry.get("confidence") is None:
            raise ValueError("candidate " + str(index) + " has no 'confidence'")
        class_id = entry.get("class_id", entry.get("cls"))
        if (
            config.expected_class is not None
            and class_id is not None
            and int(class_id) != int(config.expected_class)
        ):
            continue
        confidence = float(entry["confidence"])
        if confidence < config.min_confidence:
            continue
        candidates.append(
            {"bbox": tuple(float(value) for value in entry["bbox"]), "confidence": confidence}
        )
    return candidates


# --------------------------------------------------------------------------- #
# Group aggregation
# --------------------------------------------------------------------------- #


class _GroupAccumulator:
    """Everything one group's metrics are computed from, gathered in one pass."""

    def __init__(self) -> None:
        self.n_samples = 0
        self.n_evaluable = 0
        self.n_negative = 0
        self.n_missing_ground_truth = 0
        self.n_detector_errors = 0
        self.samples: list[tuple[list[Box], list[dict[str, Any]]]] = []
        self.evaluable_outcomes: list[dict[str, Any]] = []
        self.confidences: list[float] = []
        self.matched_confidences: list[float] = []

    def add(
        self,
        status: str,
        gt_box: Box | None,
        candidates: list[dict[str, Any]],
        outcome: dict[str, Any] | None,
    ) -> None:
        self.n_samples += 1
        if status == SAMPLE_DETECTOR_ERROR:
            self.n_detector_errors += 1
            return
        if status == SAMPLE_MISSING_GROUND_TRUTH:
            # The row records no box and does not declare itself a negative, so
            # whether its candidates are false positives is unknown. Counting them
            # either way would invent a metric.
            self.n_missing_ground_truth += 1
            return
        if status == SAMPLE_NEGATIVE:
            self.n_negative += 1
        else:
            self.n_evaluable += 1
            if outcome is not None:
                self.evaluable_outcomes.append(outcome)
        self.samples.append(([] if gt_box is None else [gt_box], candidates))
        if candidates:
            self.confidences.append(max(candidate["confidence"] for candidate in candidates))
        if outcome is not None and outcome["matched_confidence"] is not None:
            self.matched_confidences.append(outcome["matched_confidence"])


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _cell(value: Any, reason: str | None) -> Any:
    """A CSV cell: the number, or the explicit reason it does not exist.

    Floats are rounded to six decimals so a report reads as measurements rather
    than as floating-point noise; six decimals is far finer than any difference a
    capability curve can resolve.
    """
    if value is None:
        return NOT_COMPUTABLE + ": " + (reason or "not measured")
    return round(value, 6) if isinstance(value, float) else value


def _rate(hits: int, total: int, reason: str) -> float | None:
    return hits / total if total > 0 else None


def finalise_group(
    dimension: str,
    label: str,
    accumulator: _GroupAccumulator,
    config: ControlledEvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn one group's accumulations into a report row plus its gaps."""
    detections: list[tuple[float, bool]] = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gt_boxes, candidates in accumulator.samples:
        matched = match_candidates_to_gt(gt_boxes, candidates, config.iou_threshold)
        detections.extend(matched)
        hits = sum(1 for _, is_true_positive in matched if is_true_positive)
        true_positives += hits
        false_positives += len(matched) - hits
        false_negatives += len(gt_boxes) - hits

    scores = precision_recall_f1(true_positives, false_positives, false_negatives)
    maps = mean_average_precision(accumulator.samples, AP_IOU_THRESHOLDS)
    n_ground_truth = sum(len(gt_boxes) for gt_boxes, _ in accumulator.samples)
    n_predictions = sum(len(candidates) for _, candidates in accumulator.samples)

    # Why each metric would be missing, if it is. A reason survives only for a
    # metric that really has no value (cleared below), so a measured number can
    # never sit next to a stale explanation of why it does not exist.
    reasons: dict[str, str | None] = {
        "precision": "no predictions in this group",
        "recall": "no ground truth in this group",
        "f1": (
            "no predictions in this group"
            if scores["precision"] is None
            else "no ground truth in this group"
        ),
        "mAP50": maps["reason"],
        "mAP50-95": maps["reason"],
        "top1_hit_rate": "no evaluable sample in this group",
        "topk_hit_rate": "no evaluable sample in this group",
        "gt_covered_rate": "no evaluable sample in this group",
        "mean_center_error_px": "no matched candidate in this group",
        "mean_confidence": "no predictions in this group",
        "mean_matched_confidence": "no matched candidate in this group",
        "mean_best_iou": "no evaluable sample in this group",
    }

    top1_hits = sum(1 for outcome in accumulator.evaluable_outcomes if outcome["top1_hit"])
    topk_hits = sum(1 for outcome in accumulator.evaluable_outcomes if outcome["topk_hit"])
    covered = sum(1 for outcome in accumulator.evaluable_outcomes if outcome["gt_covered"])
    centre_errors = [
        outcome["center_error_px"]
        for outcome in accumulator.evaluable_outcomes
        if outcome["center_error_px"] is not None
    ]
    best_ious = [outcome["best_iou"] for outcome in accumulator.evaluable_outcomes]

    values: dict[str, Any] = {
        "n_samples": accumulator.n_samples,
        "n_evaluable": accumulator.n_evaluable,
        "n_negative": accumulator.n_negative,
        "n_missing_ground_truth": accumulator.n_missing_ground_truth,
        "n_detector_errors": accumulator.n_detector_errors,
        "n_ground_truth": n_ground_truth,
        "n_predictions": n_predictions,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "mAP50": maps["mAP50"],
        "mAP50-95": maps["mAP50-95"],
        "top1_hit_rate": _rate(top1_hits, accumulator.n_evaluable, "no evaluable sample"),
        "topk_hit_rate": _rate(topk_hits, accumulator.n_evaluable, "no evaluable sample"),
        "gt_covered_rate": _rate(covered, accumulator.n_evaluable, "no evaluable sample"),
        "mean_center_error_px": _mean(centre_errors),
        "n_center_error": len(centre_errors),
        "mean_confidence": _mean(accumulator.confidences),
        "n_confidence": len(accumulator.confidences),
        "mean_matched_confidence": _mean(accumulator.matched_confidences),
        "n_matched_confidence": len(accumulator.matched_confidences),
        "mean_best_iou": _mean(best_ious),
    }
    for metric, value in values.items():
        if value is not None:
            reasons[metric] = None

    row: dict[str, Any] = {dimension: label}
    for metric in GROUP_METRIC_COLUMNS:
        if metric in ("iou_threshold", "top_k", "notes"):
            continue
        row[metric] = _cell(values[metric], reasons.get(metric))
    row["iou_threshold"] = config.iou_threshold
    row["top_k"] = config.top_k
    row["notes"] = "; ".join(
        metric + ": " + str(reason) for metric, reason in sorted(reasons.items()) if reason
    )

    gaps = [
        {"dimension": dimension, "group": label, "metric": metric, "reason": reason}
        for metric, reason in sorted(reasons.items())
        if reason
    ]
    return row, gaps


# --------------------------------------------------------------------------- #
# The evaluation itself
# --------------------------------------------------------------------------- #


def _empty_outcome() -> dict[str, Any]:
    """The match outcome of a sample that has no ground truth to be matched."""
    return {
        "top1_hit": None,
        "top1_iou": None,
        "topk_hit": None,
        "gt_covered": None,
        "gt_covered_fraction": None,
        "center_error_px": None,
        "matched_confidence": None,
        "matched_rank": None,
        "best_iou": None,
    }


def _detect(
    detector: Detector, image_path: Path, config: ControlledEvaluationConfig
) -> tuple[list[dict[str, Any]], str, str]:
    try:
        return _normalise_candidates(detector(image_path), config), "", ""
    except Exception as error:  # a detector failure must not abort a capability sweep
        return [], SAMPLE_DETECTOR_ERROR, type(error).__name__ + ": " + str(error)


def evaluate_controlled(
    manifest_path: Path | str,
    detector: Detector,
    config: ControlledEvaluationConfig | None = None,
) -> ControlledEvaluationReport:
    """Run one detector over one controlled manifest and group every metric.

    Raises ManifestMissingError / ManifestError when the manifest itself is unusable;
    a detector failure on one image is recorded on that sample and does not stop the
    sweep, because losing a whole sweep to one unreadable frame would cost the
    measurements that did succeed.
    """
    settings = config if config is not None else ControlledEvaluationConfig()
    path = Path(manifest_path)
    rows, columns = read_manifest(path)
    plans = plan_dimensions(columns)
    plans_by_dimension = {plan.dimension: plan for plan in plans}

    groups: dict[str, "OrderedDict[str, _GroupAccumulator]"] = {
        plan.dimension: OrderedDict() for plan in plans if plan.resolved
    }
    overall = _GroupAccumulator()
    predictions: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        file_name = _text(row.get("file"))
        gt_box = ground_truth_box_from_row(row)
        size = equivalent_size_from_row(row)
        image_path = resolve_image_path(path, row, settings)
        candidates, error_status, error = _detect(detector, image_path, settings)

        if error_status:
            status = error_status
            outcome = _empty_outcome()
        elif gt_box is None:
            status = SAMPLE_NEGATIVE if _truthy(row.get("is_negative")) else SAMPLE_MISSING_GROUND_TRUTH
            outcome = _empty_outcome()
        else:
            status = SAMPLE_OK
            outcome = match_top_k(
                gt_box,
                candidates,
                settings.top_k,
                settings.iou_threshold,
                settings.gt_coverage_fraction,
            )
            matched = match_candidates_to_gt([gt_box], candidates, settings.iou_threshold)

        ground_truth = [] if gt_box is None else [gt_box]
        matched = match_candidates_to_gt(ground_truth, candidates, settings.iou_threshold)
        true_positives = sum(1 for _, is_true_positive in matched if is_true_positive)
        # A declared negative has a known answer (there is no target), so its
        # candidates are false positives; a row with no ground-truth fields at all
        # does not, and is left unmeasured rather than guessed at.
        ground_truth_known = gt_box is not None or status == SAMPLE_NEGATIVE
        labels = {
            plan.dimension: (dimension_label(plan, row) if plan.resolved else None)
            for plan in plans
        }
        prediction: dict[str, Any] = {
            "index": index,
            "file": file_name,
            "split": _text(row.get("split")),
            "source_type": _text(row.get("source_type")),
            "status": status,
            "image_path": str(image_path),
            "bbox_w_px": _text(row.get("bbox_w_px")),
            "bbox_h_px": _text(row.get("bbox_h_px")),
            "pos_x_px": _text(row.get("pos_x_px")),
            "pos_y_px": _text(row.get("pos_y_px")),
            "gt_x1": "" if gt_box is None else gt_box[0],
            "gt_y1": "" if gt_box is None else gt_box[1],
            "gt_x2": "" if gt_box is None else gt_box[2],
            "gt_y2": "" if gt_box is None else gt_box[3],
            "gt_width_px": "" if gt_box is None else gt_box[2] - gt_box[0],
            "gt_height_px": "" if gt_box is None else gt_box[3] - gt_box[1],
            "equivalent_size_px": size,
            "size_bucket": labels.get("size_bucket") or "",
            "pose_bucket": labels.get("pose") or "",
            "position_bucket": labels.get("position") or "",
            "blur_bucket": labels.get("blur") or "",
            "occlusion_bucket": labels.get("occlusion") or "",
            "background": labels.get("background") or "",
            "light_azimuth_deg": _text(row.get("light_azimuth_deg")),
            "n_candidates": len(candidates),
            "top1_confidence": (
                max(candidate["confidence"] for candidate in candidates) if candidates else None
            ),
            "top1_iou": outcome["top1_iou"],
            "top1_hit": outcome["top1_hit"],
            "topk_hit": outcome["topk_hit"],
            "gt_covered": outcome["gt_covered"],
            "gt_covered_fraction": outcome["gt_covered_fraction"],
            "center_error_px": outcome["center_error_px"],
            "center_error_defined": outcome["center_error_px"] is not None,
            "matched_rank": outcome["matched_rank"],
            "matched_confidence": outcome["matched_confidence"],
            "best_iou": outcome["best_iou"],
            "n_true_positives": true_positives if ground_truth_known else None,
            "n_false_positives": (len(matched) - true_positives) if ground_truth_known else None,
            "n_false_negatives": (
                len(ground_truth) - true_positives if gt_box is not None else None
            ),
            "iou_threshold": settings.iou_threshold,
            "top_k": settings.top_k,
            "error": error,
        }
        predictions.append(prediction)

        for dimension, accumulator_map in groups.items():
            plan = plans_by_dimension[dimension]
            label = dimension_label(plan, row) or UNLABELLED_GROUP
            accumulator = accumulator_map.get(label)
            if accumulator is None:
                accumulator = _GroupAccumulator()
                accumulator_map[label] = accumulator
            accumulator.add(status, gt_box, candidates, outcome)
        overall.add(status, gt_box, candidates, outcome)

    group_rows: dict[str, list[dict[str, Any]]] = {}
    not_computed_metrics: list[dict[str, Any]] = []
    for dimension, accumulator_map in groups.items():
        rows_out: list[dict[str, Any]] = []
        for label, accumulator in accumulator_map.items():
            row_out, gaps = finalise_group(dimension, label, accumulator, settings)
            rows_out.append(row_out)
            not_computed_metrics.extend(gaps)
        group_rows[dimension] = rows_out

    overall_row, overall_gaps = finalise_group("overall", "ALL", overall, settings)
    not_computed_metrics.extend(overall_gaps)

    not_computed = [
        {
            "dimension": plan.dimension,
            "reason": plan.reason,
            "file": plan.spec.file_name,
        }
        for plan in plans
        if not plan.resolved
    ]
    warnings = [
        "WARNING: " + str(entry["reason"]) + "; " + str(entry["file"]) + " will have no data rows"
        for entry in not_computed
    ]

    label_counts = {dimension: {} for dimension in groups}
    for row in rows:
        for dimension, accumulator_map in groups.items():
            plan = plans_by_dimension[dimension]
            label = dimension_label(plan, row) or UNLABELLED_GROUP
            label_counts[dimension][label] = label_counts[dimension].get(label, 0) + 1

    dimensions_summary = {}
    for plan in plans:
        counts = label_counts.get(plan.dimension, {})
        dimensions_summary[plan.dimension] = {
            "status": plan.status,
            "source": plan.source,
            "reason": plan.reason,
            "file": plan.spec.file_name,
            "n_groups": len(counts),
            "n_labelled": sum(count for label, count in counts.items() if label != UNLABELLED_GROUP),
            "n_unlabelled": counts.get(UNLABELLED_GROUP, 0),
        }

    split_counts: dict[str, int] = {}
    for row in rows:
        key = _text(row.get("split")) or "unspecified"
        split_counts[key] = split_counts.get(key, 0) + 1

    summary: dict[str, Any] = {
        "manifest": {
            "path": str(path),
            "sha256": _file_sha256(path),
            "columns": [str(column) for column in columns],
            "n_rows": len(rows),
        },
        "n_samples": len(rows),
        "n_evaluable": overall.n_evaluable,
        "n_negative": overall.n_negative,
        "n_missing_ground_truth": overall.n_missing_ground_truth,
        "n_detector_errors": overall.n_detector_errors,
        "counts_by_split": split_counts,
        "iou_threshold": settings.iou_threshold,
        "top_k": settings.top_k,
        "min_confidence": settings.min_confidence,
        "expected_class": settings.expected_class,
        "gt_coverage_fraction": settings.gt_coverage_fraction,
        "ap_iou_thresholds": list(AP_IOU_THRESHOLDS),
        "dimensions": dimensions_summary,
        "not_computed": not_computed,
        "not_computed_metrics": not_computed_metrics,
        "warnings": warnings,
        "overall": overall_row,
        "outputs": {
            "predictions": PREDICTION_FILENAME,
            "summary": SUMMARY_FILENAME,
            **{plan.dimension: plan.spec.file_name for plan in plans},
        },
    }

    return ControlledEvaluationReport(
        predictions=predictions,
        groups=group_rows,
        summary=summary,
        column_plans={plan.dimension: plan.status for plan in plans},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def write_controlled_reports(
    report: ControlledEvaluationReport, out_dir: Path | str
) -> dict[str, Path]:
    """Write predictions.csv, one metrics file per dimension, and summary.json.

    A dimension whose column is absent still gets its file, with a header and no
    rows: a file that exists and is empty says "this curve was not produced",
    while a missing file says nothing at all.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {
        "predictions": _write_csv(target / PREDICTION_FILENAME, PREDICTION_COLUMNS, report.predictions)
    }
    for spec in DIMENSION_SPECS:
        fieldnames = (spec.dimension,) + GROUP_METRIC_COLUMNS
        written[spec.dimension] = _write_csv(
            target / spec.file_name, fieldnames, report.groups.get(spec.dimension, [])
        )
    summary_path = target / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(report.summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written["summary"] = summary_path
    return written
