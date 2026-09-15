#!/usr/bin/env python3
"""Controlled-capability evaluation of the shuttlecock detector (spec 03, plan Task 3).

Why this module exists: spec 03 does not ask how high mAP is, it asks where the
baseline stops working as the target gets smaller, turns, moves, blurs or becomes
occluded. Answering that needs one evaluator that consumes a controlled manifest,
runs a detector over it, and reports the same metrics grouped by each controlled
variable -- not a script per variable, which is how two curves in one report end up
disagreeing about what a hit is.

The detector is injected as a callable, so this whole test suite runs with no GPU,
no weights and no Ultralytics. Ultralytics is imported only inside functions of the
CLI adapter (scripts/shuttle_detection/evaluate_controlled.py), never at module
scope, so the evaluator can be imported, inspected and tested on a laptop.

Two reporting rules that the tests pin:

- A metric that could not be measured is written as the text
  "NOT_COMPUTABLE: <reason>", never as 0.0. Precision over a group with no
  predictions is undefined; recall over the same group is a measured zero (there
  were ground-truth boxes and none was found). Writing 0.0 for both would make a
  detector that returned nothing indistinguishable from one that returned wrong
  boxes.
- A dimension whose manifest column is absent -- and which cannot be derived -- is
  reported in summary.json under "not_computed", printed as a WARNING by the CLI,
  and its metrics file is written with no data rows. A missing conditioned curve
  must be visible in the output, not missing from it.

Run:
    python tests/perception/shuttle_detection/test_controlled_evaluator.py -v
    pytest tests/perception/shuttle_detection/test_controlled_evaluator.py -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.evaluate import (  # noqa: E402
    NOT_COMPUTABLE,
    PREDICTION_COLUMNS,
    ControlledEvaluationConfig,
    evaluate_controlled,
    write_controlled_reports,
)

EVALUATE_PATH = ROOT / "src" / "perception" / "shuttle_detection" / "evaluate.py"
CLI_PATH = ROOT / "scripts" / "shuttle_detection" / "evaluate_controlled.py"

CONTROLLED_COLUMNS = (
    "file",
    "split",
    "source_type",
    "bbox_w_px",
    "bbox_h_px",
    "pos_x_px",
    "pos_y_px",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "pose_bucket",
    "position_bucket",
    "blur_bucket",
    "occlusion_bucket",
    "background",
    "light_azimuth_deg",
    "equivalent_size_px",
    "imgsz",
)

REQUIRED_OUTPUTS = (
    "predictions.csv",
    "size_bucket_metrics.csv",
    "pose_metrics.csv",
    "position_metrics.csv",
    "blur_metrics.csv",
    "occlusion_metrics.csv",
    "summary.json",
)

# The default test row is a 10x10 box centred at (640, 360) with equivalent size
# 10.0 px, which is bucket "8-12".
TRUE_BOX = (635.0, 355.0, 645.0, 365.0)
DISTRACTOR = {"bbox": (30.0, 30.0, 35.0, 35.0), "confidence": 0.9}
TRUE_CANDIDATE = {"bbox": TRUE_BOX, "confidence": 0.7}
PERFECT_DETECTION = [{"bbox": TRUE_BOX, "confidence": 0.8}]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def manifest_row(file_name, **overrides):
    row = {column: "" for column in CONTROLLED_COLUMNS}
    row.update(
        {
            "file": file_name,
            "split": "fixed_core_test",
            "source_type": "SYNTHETIC_3D",
            "bbox_w_px": "10",
            "bbox_h_px": "10",
            "pos_x_px": "640",
            "pos_y_px": "360",
            "yaw_deg": "0",
            "pitch_deg": "0",
            "roll_deg": "0",
            "pose_bucket": "side",
            "position_bucket": "center",
            "blur_bucket": "clear",
            "occlusion_bucket": "none",
            "background": "bg_001.jpg",
            "light_azimuth_deg": "45",
            "equivalent_size_px": "10.0",
            "imgsz": "1280",
        }
    )
    row.update({key: ("" if value is None else str(value)) for key, value in overrides.items()})
    return row


def write_manifest(path, rows, columns=CONTROLLED_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def by_file(answers):
    """A detector callable that answers from a {file name: candidates} map."""

    def detector(image_path):
        return answers.get(Path(image_path).name, [])

    return detector


def evaluate(tmp_path, rows, detector, columns=CONTROLLED_COLUMNS, config=None, out_name="out"):
    manifest = write_manifest(tmp_path / "manifest.csv", rows, columns)
    report = evaluate_controlled(manifest, detector, config)
    out_dir = tmp_path / out_name
    write_controlled_reports(report, out_dir)
    return report, out_dir


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(out_dir):
    return json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))


def number(text):
    return float(text)


# --------------------------------------------------------------------------- #
# The plan's acceptance test: a fake detector, no GPU
# --------------------------------------------------------------------------- #


def test_required_output_files_are_written(tmp_path):
    rows = [manifest_row("a.png"), manifest_row("b.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION, "b.png": []}))
    for name in REQUIRED_OUTPUTS:
        assert (out_dir / name).is_file(), name


def test_evaluator_reports_size_bucket_topk_hit_and_center_error(tmp_path):
    # plan Task 3 step 1: a fake detector is enough to produce the three numbers
    # the capability curves are made of.
    rows = [manifest_row("a.png")]
    report, out_dir = evaluate(
        tmp_path, rows, by_file({"a.png": [DISTRACTOR, TRUE_CANDIDATE]})
    )
    predictions = read_rows(out_dir / "predictions.csv")
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction["size_bucket"] == "8-12"
    assert prediction["top1_hit"] == "False"
    assert prediction["topk_hit"] == "True"
    assert number(prediction["center_error_px"]) == 0.0
    assert report.predictions[0]["size_bucket"] == "8-12"


def test_top1_and_topk_hit_rates_disagree_in_the_group_row(tmp_path):
    # Spec 03 section 6: keeping only the best box destroys a true candidate that a
    # later stage could have recovered. Two samples each with a confident
    # distractor ranked first and the true box second.
    rows = [manifest_row("a.png"), manifest_row("b.png")]
    detector = by_file({"a.png": [DISTRACTOR, TRUE_CANDIDATE], "b.png": [DISTRACTOR, TRUE_CANDIDATE]})
    _, out_dir = evaluate(tmp_path, rows, detector)
    group = read_rows(out_dir / "size_bucket_metrics.csv")[0]
    assert group["size_bucket"] == "8-12"
    assert number(group["top1_hit_rate"]) == 0.0
    assert number(group["topk_hit_rate"]) == 1.0
    assert int(group["n_samples"]) == 2
    assert int(group["n_evaluable"]) == 2


def test_ground_truth_box_is_converted_from_the_centre_and_size_fields(tmp_path):
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}))
    prediction = read_rows(out_dir / "predictions.csv")[0]
    assert number(prediction["gt_x1"]) == 635.0
    assert number(prediction["gt_y1"]) == 355.0
    assert number(prediction["gt_x2"]) == 645.0
    assert number(prediction["gt_y2"]) == 365.0
    assert number(prediction["gt_width_px"]) == 10.0
    assert number(prediction["gt_height_px"]) == 10.0


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


def test_samples_are_grouped_by_every_controlled_variable(tmp_path):
    rows = [
        manifest_row("a.png", pose_bucket="birdie_to_camera", position_bucket="left", blur_bucket="heavy", occlusion_bucket="partial", background="bg_002.jpg"),
        manifest_row("b.png", pose_bucket="side", position_bucket="corner", blur_bucket="clear", occlusion_bucket="none", background="bg_001.jpg"),
    ]
    detector = by_file({"a.png": PERFECT_DETECTION, "b.png": PERFECT_DETECTION})
    _, out_dir = evaluate(tmp_path, rows, detector)
    for name, column, expected in (
        ("pose_metrics.csv", "pose", {"birdie_to_camera", "side"}),
        ("position_metrics.csv", "position", {"left", "corner"}),
        ("blur_metrics.csv", "blur", {"heavy", "clear"}),
        ("occlusion_metrics.csv", "occlusion", {"partial", "none"}),
        ("background_metrics.csv", "background", {"bg_002.jpg", "bg_001.jpg"}),
    ):
        rows_out = read_rows(out_dir / name)
        assert {row[column] for row in rows_out} == expected


def test_group_metrics_carry_precision_recall_f1_and_map(tmp_path):
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}))
    group = read_rows(out_dir / "size_bucket_metrics.csv")[0]
    assert number(group["precision"]) == 1.0
    assert number(group["recall"]) == 1.0
    assert number(group["f1"]) == 1.0
    assert number(group["mAP50"]) == 1.0
    assert number(group["mAP50-95"]) == 1.0
    assert number(group["mean_center_error_px"]) == 0.0
    assert number(group["mean_confidence"]) == 0.8
    assert int(group["tp"]) == 1
    assert int(group["fp"]) == 0
    assert int(group["fn"]) == 0


def test_group_rows_are_self_describing(tmp_path):
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}))
    group = read_rows(out_dir / "size_bucket_metrics.csv")[0]
    assert number(group["iou_threshold"]) == 0.5
    assert int(group["top_k"]) == 5


def test_multi_threshold_map_is_averaged_over_the_iou_grid(tmp_path):
    # A box swapped by half its own width: IoU 2/3 against the ground truth, which
    # is a hit at 0.50, 0.55, 0.60 and 0.65 and a miss above.
    rows = [manifest_row("a.png")]
    swapped = [{"bbox": (637.0, 355.0, 647.0, 365.0), "confidence": 0.9}]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": swapped}))
    group = read_rows(out_dir / "size_bucket_metrics.csv")[0]
    assert number(group["mAP50"]) == 1.0
    assert number(group["mAP50-95"]) == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# 0.0 versus NOT_COMPUTABLE
# --------------------------------------------------------------------------- #


def test_a_group_with_no_predictions_reports_map_as_not_computable_and_recall_as_zero(tmp_path):
    rows = [
        manifest_row("small_0.png", bbox_w_px=4, bbox_h_px=9, equivalent_size_px="6.0"),
        manifest_row("small_1.png", bbox_w_px=4, bbox_h_px=9, equivalent_size_px="6.0"),
        manifest_row("large_0.png"),
    ]
    detector = by_file({"large_0.png": PERFECT_DETECTION})
    _, out_dir = evaluate(tmp_path, rows, detector)
    groups = {row["size_bucket"]: row for row in read_rows(out_dir / "size_bucket_metrics.csv")}

    empty = groups["6-8"]
    assert empty["mAP50"].startswith(NOT_COMPUTABLE)
    assert empty["mAP50-95"].startswith(NOT_COMPUTABLE)
    assert empty["precision"].startswith(NOT_COMPUTABLE)
    assert empty["f1"].startswith(NOT_COMPUTABLE)
    assert "no predictions" in empty["mAP50"]
    # Recall over a group with ground truth and no hits is a measurement, not a gap.
    assert number(empty["recall"]) == 0.0
    assert int(empty["fn"]) == 2
    assert int(empty["tp"]) == 0
    assert int(empty["fp"]) == 0
    assert empty["mean_center_error_px"].startswith(NOT_COMPUTABLE)

    measured = groups["8-12"]
    assert number(measured["mAP50"]) == 1.0
    assert measured["mean_center_error_px"] == "0.0"


def test_a_group_with_no_ground_truth_reports_recall_as_not_computable(tmp_path):
    rows = [manifest_row("a.png", is_negative="True", bbox_w_px="", bbox_h_px="", equivalent_size_px="")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), columns=CONTROLLED_COLUMNS + ("is_negative",))
    group = read_rows(out_dir / "size_bucket_metrics.csv")[0]
    assert group["recall"].startswith(NOT_COMPUTABLE)
    assert "no ground truth" in group["recall"]
    assert number(group["precision"]) == 0.0
    assert group["mAP50"].startswith(NOT_COMPUTABLE)


def test_not_computable_metrics_are_listed_in_the_summary(tmp_path):
    rows = [
        manifest_row("small.png", bbox_w_px=4, bbox_h_px=9, equivalent_size_px="6.0"),
        manifest_row("large.png"),
    ]
    _, out_dir = evaluate(tmp_path, rows, by_file({"large.png": PERFECT_DETECTION}))
    summary = read_summary(out_dir)
    entries = {(entry["group"], entry["metric"]) for entry in summary["not_computed_metrics"]}
    assert ("6-8", "mAP50") in entries
    assert ("6-8", "precision") in entries


# --------------------------------------------------------------------------- #
# Missing manifest columns
# --------------------------------------------------------------------------- #


def test_every_dimension_is_reported_when_the_manifest_is_complete(tmp_path):
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}))
    summary = read_summary(out_dir)
    assert summary["not_computed"] == []
    assert set(summary["dimensions"]) == {
        "size_bucket",
        "pose",
        "position",
        "blur",
        "occlusion",
        "background",
    }
    # This manifest carries equivalent_size_px but no size_bucket column, so the
    # size curve is derived from the recorded size; every other dimension is read
    # straight from its own column.
    statuses = {name: entry["status"] for name, entry in summary["dimensions"].items()}
    assert statuses["size_bucket"] == "derived"
    assert all(
        status == "manifest_column" for name, status in statuses.items() if name != "size_bucket"
    )


def test_a_missing_column_is_reported_as_not_computed_with_a_warning(tmp_path):
    columns = tuple(column for column in CONTROLLED_COLUMNS if column not in ("occlusion_bucket", "background"))
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(
        tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), columns=columns
    )
    summary = read_summary(out_dir)
    missing = {entry["dimension"] for entry in summary["not_computed"]}
    assert missing == {"occlusion", "background"}
    assert summary["dimensions"]["occlusion"]["status"] == "not_computed"
    assert any("occlusion" in warning for warning in summary["warnings"])
    assert any("background" in warning for warning in summary["warnings"])
    # The file is still written, with a header and no rows: the absent curve is
    # visible in the output rather than missing from it.
    assert read_rows(out_dir / "occlusion_metrics.csv") == []
    assert (out_dir / "occlusion_metrics.csv").is_file()


def test_size_bucket_is_derived_when_both_size_columns_are_absent(tmp_path):
    columns = tuple(
        column
        for column in CONTROLLED_COLUMNS
        if column not in ("equivalent_size_px",)
    )
    rows = [manifest_row("a.png", bbox_w_px=4, bbox_h_px=9)]
    _, out_dir = evaluate(
        tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), columns=columns
    )
    summary = read_summary(out_dir)
    assert summary["dimensions"]["size_bucket"]["status"] == "derived"
    prediction = read_rows(out_dir / "predictions.csv")[0]
    assert number(prediction["equivalent_size_px"]) == pytest.approx(6.0)
    assert prediction["size_bucket"] == "6-8"


def test_size_is_not_computed_without_any_size_field(tmp_path):
    columns = tuple(
        column
        for column in CONTROLLED_COLUMNS
        if column not in ("equivalent_size_px", "bbox_w_px", "bbox_h_px")
    )
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), columns=columns)
    summary = read_summary(out_dir)
    assert summary["dimensions"]["size_bucket"]["status"] == "not_computed"
    assert {entry["dimension"] for entry in summary["not_computed"]} >= {"size_bucket"}


def test_pose_is_derived_from_the_angles_when_the_bucket_column_is_absent(tmp_path):
    columns = tuple(column for column in CONTROLLED_COLUMNS if column != "pose_bucket")
    rows = [
        manifest_row("a.png", yaw_deg="0", pitch_deg="0", roll_deg="0"),
        manifest_row("b.png", yaw_deg="90", pitch_deg="0", roll_deg="0"),
    ]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION, "b.png": PERFECT_DETECTION}), columns=columns)
    summary = read_summary(out_dir)
    assert summary["dimensions"]["pose"]["status"] == "derived"
    assert {row["pose"] for row in read_rows(out_dir / "pose_metrics.csv")} == {
        "axis_aligned",
        "steep",
    }


def test_position_is_derived_from_the_image_size_and_the_target_centre(tmp_path):
    columns = tuple(column for column in CONTROLLED_COLUMNS if column != "position_bucket")
    rows = [
        # The frame is 1280x1280, so its thirds are 0-426, 427-853 and 854-1280.
        manifest_row("center.png", pos_x_px=640, pos_y_px=640),
        manifest_row("left.png", pos_x_px=100, pos_y_px=640),
        manifest_row("right.png", pos_x_px=1180, pos_y_px=640),
        manifest_row("top.png", pos_x_px=640, pos_y_px=60),
        manifest_row("bottom.png", pos_x_px=640, pos_y_px=1220),
        manifest_row("corner.png", pos_x_px=60, pos_y_px=60),
    ]
    detector = by_file({row["file"]: PERFECT_DETECTION for row in rows})
    _, out_dir = evaluate(tmp_path, rows, detector, columns=columns)
    predictions = {row["file"]: row for row in read_rows(out_dir / "predictions.csv")}
    assert predictions["center.png"]["position_bucket"] == "center"
    assert predictions["left.png"]["position_bucket"] == "left"
    assert predictions["right.png"]["position_bucket"] == "right"
    assert predictions["top.png"]["position_bucket"] == "top"
    assert predictions["bottom.png"]["position_bucket"] == "bottom"
    assert predictions["corner.png"]["position_bucket"] == "top_left"


def test_position_is_not_computed_without_an_image_size(tmp_path):
    columns = tuple(
        column
        for column in CONTROLLED_COLUMNS
        if column not in ("position_bucket", "imgsz")
    )
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), columns=columns)
    summary = read_summary(out_dir)
    assert summary["dimensions"]["position"]["status"] == "not_computed"
    assert "image size" in summary["dimensions"]["position"]["reason"]


def test_the_blur_alias_column_is_used_when_blur_bucket_is_absent(tmp_path):
    columns = tuple(
        column for column in CONTROLLED_COLUMNS if column != "blur_bucket"
    ) + ("blur",)
    rows = [manifest_row("a.png", blur="synthetic_sigma2.6")]
    _, out_dir = evaluate(
        tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), columns=columns
    )
    summary = read_summary(out_dir)
    assert summary["dimensions"]["blur"]["status"] == "manifest_column"
    assert "blur" in summary["dimensions"]["blur"]["source"]
    assert read_rows(out_dir / "blur_metrics.csv")[0]["blur"] == "synthetic_sigma2.6"


# --------------------------------------------------------------------------- #
# Per-sample bookkeeping
# --------------------------------------------------------------------------- #


def test_a_detector_error_is_recorded_and_the_run_continues(tmp_path):
    def detector(image_path):
        if Path(image_path).name == "bad.png":
            raise RuntimeError("cannot read image")
        return PERFECT_DETECTION

    rows = [manifest_row("bad.png"), manifest_row("good.png")]
    report, out_dir = evaluate(tmp_path, rows, detector)
    statuses = {row["file"]: row["status"] for row in read_rows(out_dir / "predictions.csv")}
    assert statuses["bad.png"] == "detector_error"
    assert statuses["good.png"] == "ok"
    error_row = read_rows(out_dir / "predictions.csv")[0]
    assert "cannot read image" in error_row["error"]
    assert read_summary(out_dir)["n_detector_errors"] == 1
    assert report.summary["n_samples"] == 2


def test_a_negative_row_keeps_its_predictions_as_false_positives(tmp_path):
    columns = CONTROLLED_COLUMNS + ("is_negative",)
    rows = [
        manifest_row("positive.png"),
        manifest_row("negative.png", is_negative="True", bbox_w_px="", bbox_h_px="", equivalent_size_px=""),
    ]
    detector = by_file({"positive.png": PERFECT_DETECTION, "negative.png": PERFECT_DETECTION})
    _, out_dir = evaluate(tmp_path, rows, detector, columns=columns)
    statuses = {row["file"]: row["status"] for row in read_rows(out_dir / "predictions.csv")}
    assert statuses["negative.png"] == "negative"
    summary = read_summary(out_dir)
    assert summary["n_negative"] == 1
    assert summary["n_evaluable"] == 1
    assert summary["overall"]["fp"] == 1
    assert summary["overall"]["tp"] == 1


def test_a_row_without_any_ground_truth_is_excluded_from_the_metrics(tmp_path):
    rows = [
        manifest_row("positive.png"),
        manifest_row("unknown.png", bbox_w_px="", bbox_h_px="", equivalent_size_px=""),
    ]
    detector = by_file({"positive.png": PERFECT_DETECTION, "unknown.png": PERFECT_DETECTION})
    _, out_dir = evaluate(tmp_path, rows, detector)
    statuses = {row["file"]: row["status"] for row in read_rows(out_dir / "predictions.csv")}
    assert statuses["unknown.png"] == "missing_ground_truth"
    summary = read_summary(out_dir)
    assert summary["n_missing_ground_truth"] == 1
    # The unknown row is not counted as a false positive: we do not know whether it
    # was a negative or a broken row, and guessing either way would invent a metric.
    assert summary["overall"]["fp"] == 0


def test_reported_counts_by_split_are_recorded(tmp_path):
    rows = [manifest_row("a.png", split="fixed_core_test"), manifest_row("b.png", split="val")]
    report, _ = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION, "b.png": PERFECT_DETECTION}))
    assert report.summary["counts_by_split"] == {"fixed_core_test": 1, "val": 1}


def test_min_confidence_filters_candidates_before_ranking(tmp_path):
    rows = [manifest_row("a.png")]
    detector = by_file({"a.png": [DISTRACTOR, TRUE_CANDIDATE]})
    config = ControlledEvaluationConfig(min_confidence=0.8)
    report, _ = evaluate(tmp_path, rows, detector, config=config)
    assert report.predictions[0]["n_candidates"] == 1
    assert report.predictions[0]["top1_hit"] is False
    assert report.predictions[0]["topk_hit"] is False


def test_class_filtering_keeps_only_the_detection_class(tmp_path):
    rows = [manifest_row("a.png")]
    boxes = [
        {"bbox": TRUE_BOX, "confidence": 0.8, "class_id": 32},
        {"bbox": TRUE_BOX, "confidence": 0.7, "class_id": 0},
    ]
    report, _ = evaluate(tmp_path, rows, by_file({"a.png": boxes}))
    assert report.predictions[0]["n_candidates"] == 1
    assert report.predictions[0]["top1_confidence"] == 0.7


def test_the_summary_records_the_evaluation_parameters(tmp_path):
    rows = [manifest_row("a.png")]
    config = ControlledEvaluationConfig(iou_threshold=0.6, top_k=3, min_confidence=0.1, expected_class=None)
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}), config=config)
    summary = read_summary(out_dir)
    assert summary["iou_threshold"] == 0.6
    assert summary["top_k"] == 3
    assert summary["min_confidence"] == 0.1
    assert summary["expected_class"] is None
    assert summary["manifest"]["columns"] == list(CONTROLLED_COLUMNS)
    assert summary["manifest"]["n_rows"] == 1
    assert len(summary["manifest"]["sha256"]) == 64


def test_a_missing_manifest_raises(tmp_path):
    from src.perception.shuttle_detection.dataset_audit import ManifestMissingError

    with pytest.raises(ManifestMissingError):
        evaluate_controlled(tmp_path / "absent.csv", by_file({}))


def test_the_prediction_table_has_a_stable_column_set(tmp_path):
    rows = [manifest_row("a.png")]
    _, out_dir = evaluate(tmp_path, rows, by_file({"a.png": PERFECT_DETECTION}))
    with (out_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        fieldnames = next(csv.reader(handle))
    assert tuple(fieldnames) == PREDICTION_COLUMNS


# --------------------------------------------------------------------------- #
# The Ultralytics boundary
# --------------------------------------------------------------------------- #


def test_ultralytics_is_only_imported_inside_a_function_body():
    # A module-scope "import ultralytics" would make the evaluator unusable on a
    # host without it, and the failure would appear as a collection error rather
    # than as a clear message.
    for path in (EVALUATE_PATH, CLI_PATH):
        assert path.is_file(), path
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("import ultralytics") or stripped.startswith("from ultralytics"):
                assert line[:1].isspace(), (
                    str(path) + ":" + str(line_number) + " imports ultralytics at module scope"
                )


def test_the_cli_module_imports_without_ultralytics():
    code = (
        "import importlib.util, sys;"
        "spec = importlib.util.spec_from_file_location('cli', r'" + str(CLI_PATH) + "');"
        "module = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(module);"
        "print('ultralytics' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False")


def test_the_cli_prints_its_usage():
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0
    for flag in ("--weights", "--manifest", "--out"):
        assert flag in result.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="this host has Ultralytics, so the missing-backend path cannot be exercised",
)
def test_the_cli_fails_cleanly_when_ultralytics_is_missing(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.csv", [manifest_row("a.png")])
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-a-real-checkpoint")
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--weights",
            str(weights),
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 2
    assert "ultralytics" in (result.stdout + result.stderr)
    assert "Traceback" not in result.stderr
