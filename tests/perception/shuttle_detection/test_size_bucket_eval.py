#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-02/03: tests for the size-bucket offline evaluation protocol.

The harness is scored on SYNTHETIC fixtures whose per-bucket recall is declared
in the generator; the tests check that the harness recovers the declaration and
that every malformed input is rejected instead of being scored.

Run:
    python -m pytest tests/perception/shuttle_detection/test_size_bucket_eval.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.size_bucket_eval import (  # noqa: E402
    EVAL_SCHEMA_VERSION,
    GT_SCHEMA_ID,
    PRED_SCHEMA_ID,
    Detection,
    EvalSchemaError,
    GtObject,
    SCHEMA_SOURCE,
    average_precision,
    bucket_for_size,
    centre_distance,
    evaluate,
    iou_xyxy,
    latency_stats,
    match_frame,
    parse_ground_truth,
    parse_predictions,
    validate_inputs,
)

FIXTURE_DIR = ROOT / "outputs" / "p7_eval" / "fixtures"
GT_PATH = FIXTURE_DIR / "ground_truth.jsonl"
PRED_PATH = FIXTURE_DIR / "predictions.jsonl"
META_PATH = FIXTURE_DIR / "fixtures.json"


def gt_record(frame_key="f1", objects=None, labeled=True, split="val",
              image_size=None, camera_id="d455_left"):
    return {
        "schema": GT_SCHEMA_ID, "schema_version": EVAL_SCHEMA_VERSION,
        "frame_key": frame_key, "split": split, "labeled": labeled,
        "camera_id": camera_id,
        "image_size": image_size or {"width_px": 1280, "height_px": 720},
        "timestamp_s": 1.0,
        "objects": objects if objects is not None else [
            {"bbox_xyxy_px": [100.0, 100.0, 105.0, 105.0], "size_px": 5.0}],
    }


def pred_record(frame_key="f1", bbox=None, confidence=0.9, latency_ms=20.0,
                image_size=None, camera_id="d455_left", **extra):
    payload = {
        "schema": PRED_SCHEMA_ID, "schema_version": EVAL_SCHEMA_VERSION,
        "frame_key": frame_key,
        "bbox_xyxy_px": bbox or [100.2, 100.1, 105.3, 105.2],
        "confidence": confidence, "latency_ms": latency_ms,
        "image_size": image_size or {"width_px": 1280, "height_px": 720},
        "camera_id": camera_id, "timestamp_s": 1.0,
        "pixel_covariance": [[1.0, 0.0], [0.0, 1.0]],
    }
    payload.update(extra)
    return payload


class GeometryTests(unittest.TestCase):
    def test_iou_and_distance(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        self.assertAlmostEqual(iou_xyxy(a, a), 1.0)
        self.assertAlmostEqual(iou_xyxy(a, np.array([5.0, 0.0, 15.0, 10.0])), 1 / 3, places=6)
        self.assertEqual(iou_xyxy(a, np.array([20.0, 20.0, 30.0, 30.0])), 0.0)
        self.assertAlmostEqual(centre_distance(a, np.array([5.0, 0.0, 15.0, 10.0])), 5.0)

    def test_bucket_edges_are_half_open(self):
        self.assertEqual(bucket_for_size(3.99), "<4px")
        self.assertEqual(bucket_for_size(4.0), "4-6px")
        self.assertEqual(bucket_for_size(6.0), "6-10px")
        self.assertEqual(bucket_for_size(10.0), "10-20px")
        self.assertEqual(bucket_for_size(20.0), ">20px")
        self.assertEqual(bucket_for_size(float("nan")), "unknown")

    def test_average_precision_of_a_perfect_ranking_is_one(self):
        self.assertAlmostEqual(average_precision([0.9, 0.8], [True, True], 2), 1.0)
        # one FP ranked first caps the AP at 0.5 for one positive
        self.assertAlmostEqual(average_precision([0.9, 0.8], [False, True], 1), 0.5)

    def test_average_precision_is_none_without_positives(self):
        self.assertIsNone(average_precision([0.9], [True], 0))

    def test_latency_stats(self):
        stats = latency_stats([10.0, 20.0, 30.0, None])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["min_ms"], 10.0)
        self.assertEqual(stats["max_ms"], 30.0)
        self.assertEqual(stats["p50_ms"], 20.0)
        self.assertEqual(latency_stats([])["status"], "NO_LATENCY_RECORDED")


class MatchingTests(unittest.TestCase):
    def test_two_gt_one_detection_gives_one_tp_and_one_fn(self):
        gt = [GtObject(np.array([0.0, 0.0, 10.0, 10.0]), 10.0),
              GtObject(np.array([100.0, 100.0, 110.0, 110.0]), 10.0)]
        dets = [Detection("f1", np.array([0.5, 0.5, 10.5, 10.5]), 0.9, 12.0)]
        matches = match_frame(gt, dets)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], 0)

    def test_centre_rule_recovers_a_two_pixel_shift_on_a_tiny_box(self):
        # a 5x5 box shifted by 2 px has IoU = 15/35 = 0.43 -> a full miss at 0.5
        gt = [GtObject(np.array([100.0, 100.0, 105.0, 105.0]), 5.0)]
        dets = [Detection("f1", np.array([102.0, 100.0, 107.0, 105.0]), 0.9)]
        self.assertEqual(match_frame(gt, dets, rule="iou50"), [])
        self.assertEqual(len(match_frame(gt, dets, rule="centre")), 1)
        self.assertAlmostEqual(iou_xyxy(gt[0].bbox_xyxy_px, dets[0].bbox_xyxy_px), 15 / 35, places=6)

    def test_one_pixel_shift_still_passes_iou50_on_a_5px_box(self):
        gt = [GtObject(np.array([100.0, 100.0, 105.0, 105.0]), 5.0)]
        dets = [Detection("f1", np.array([101.0, 100.0, 106.0, 105.0]), 0.9)]
        self.assertEqual(len(match_frame(gt, dets, rule="iou50")), 1)


class RejectionTests(unittest.TestCase):
    """Input-schema validation must REJECT, not score, malformed input."""

    def _gt(self, *records):
        return parse_ground_truth(list(records))

    def test_unlabeled_frame_is_rejected(self):
        with self.assertRaises(EvalSchemaError) as ctx:
            self._gt(gt_record(labeled=False))
        self.assertIn("UNLABELED_FRAME", {code for code, _, _ in ctx.exception.problems})

    def test_labeled_must_be_exactly_true(self):
        for value in (1, "yes", None):
            with self.assertRaises(EvalSchemaError):
                parse_ground_truth([gt_record(labeled=value)])

    def test_duplicate_frame_key_is_rejected(self):
        with self.assertRaises(EvalSchemaError) as ctx:
            self._gt(gt_record("f1"), gt_record("f1"))
        self.assertIn("DUPLICATE_FRAME", {code for code, _, _ in ctx.exception.problems})

    def test_unknown_key_is_rejected_in_strict_mode(self):
        record = gt_record()
        record["split_name"] = "train"
        with self.assertRaises(EvalSchemaError) as ctx:
            parse_ground_truth([record])
        self.assertIn("UNKNOWN_KEY", {code for code, _, _ in ctx.exception.problems})

    def test_bad_bbox_and_numbers_are_rejected(self):
        for bbox in ([0.0, 0.0, 0.0, 5.0], [5.0, 5.0, 1.0, 6.0], [-1.0, 0.0, 5.0, 5.0]):
            with self.assertRaises(EvalSchemaError):
                parse_ground_truth([gt_record(objects=[{"bbox_xyxy_px": bbox}])])

    def test_prediction_on_an_unknown_frame_is_rejected(self):
        gt = self._gt(gt_record("f1"))
        dets = parse_predictions([pred_record("f2")])
        with self.assertRaises(EvalSchemaError) as ctx:
            validate_inputs(gt, dets)
        self.assertIn("UNKNOWN_FRAME", {code for code, _, _ in ctx.exception.problems})

    def test_prediction_camera_or_size_mismatch_is_rejected(self):
        gt = self._gt(gt_record("f1"))
        with self.assertRaises(EvalSchemaError) as ctx:
            validate_inputs(gt, parse_predictions([pred_record("f1", camera_id="other")]))
        self.assertIn("MISMATCHED_FRAME", {code for code, _, _ in ctx.exception.problems})
        with self.assertRaises(EvalSchemaError):
            validate_inputs(gt, parse_predictions([
                pred_record("f1", image_size={"width_px": 1920, "height_px": 1080})]))

    def test_metric_key_in_a_pixel_payload_is_a_unit_confusion(self):
        with self.assertRaises(EvalSchemaError) as ctx:
            parse_predictions([pred_record("f1", depth_m=3.0)])
        codes = {code for code, _, _ in ctx.exception.problems}
        self.assertTrue({"UNIT_CONFUSION", "UNKNOWN_KEY"} & codes)

    def test_metre_squared_covariance_is_rejected_by_the_pixel_floor(self):
        with self.assertRaises(EvalSchemaError) as ctx:
            parse_predictions([pred_record("f1", pixel_covariance=[[1e-4, 0.0], [0.0, 1e-4]])])
        self.assertIn("UNIT_CONFUSION", {code for code, _, _ in ctx.exception.problems})

    def test_confidence_and_latency_ranges(self):
        for confidence in (1.5, -0.1, float("nan")):
            with self.assertRaises(EvalSchemaError):
                parse_predictions([pred_record("f1", confidence=confidence)])
        with self.assertRaises(EvalSchemaError):
            parse_predictions([pred_record("f1", latency_ms=-1.0)])

    def test_wrong_schema_id_or_version_is_rejected(self):
        record = pred_record("f1")
        record["schema"] = "badmin.other"
        with self.assertRaises(EvalSchemaError):
            parse_predictions([record])
        record = pred_record("f1")
        record["schema_version"] = 99
        with self.assertRaises(EvalSchemaError):
            parse_predictions([record])

    def test_a_malformed_jsonl_line_is_a_schema_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            path.write_text('{"schema": "x"}\nnot json\n', encoding="utf-8")
            report = None
            with self.assertRaises(EvalSchemaError):
                from src.perception.shuttle_detection.size_bucket_eval import read_jsonl
                read_jsonl(path)
            self.assertIsNone(report)


@unittest.skipUnless(GT_PATH.is_file() and PRED_PATH.is_file(),
                     "committed synthetic fixtures are not present")
class CommittedFixtureEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate(GT_PATH, PRED_PATH, provenance="SYNTHETIC_FIXTURE", split="val")
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def test_claim_is_offline_synthetic_and_not_an_accuracy_claim(self):
        self.assertEqual(self.report["claim"], "OFFLINE-SYNTHETIC")
        self.assertFalse(self.report["real_detector_accuracy"])
        self.assertIn("NOT a real-detector accuracy claim", " ".join(self.report["notes"]))

    def test_schema_source_is_recorded(self):
        self.assertIn(self.report["detection_schema_source"], {"P7_MODULE", "FALLBACK_COMPAT"})
        self.assertEqual(self.report["detection_schema_source"], SCHEMA_SOURCE)

    def test_harness_recovers_the_declared_per_bucket_recall(self):
        declared = self.meta["declared_recall_by_bucket"]
        per_bucket = self.report["matching_rules"]["iou50"]["per_bucket"]
        for bucket, expected in declared.items():
            block = per_bucket[bucket]
            self.assertIsNotNone(block["recall"], bucket)
            self.assertLess(abs(block["recall"] - expected), 0.20,
                            f"{bucket}: harness recall {block['recall']} vs declared {expected}")

    def test_recall_falls_with_object_size(self):
        per_bucket = self.report["matching_rules"]["iou50"]["per_bucket"]
        recalls = [per_bucket[b]["recall"] for b in ("4-6px", "6-10px", "10-20px", ">20px")]
        self.assertEqual(recalls, sorted(recalls))

    def test_counts_add_up_in_every_bucket(self):
        overall = self.report["matching_rules"]["iou50"]["overall"]
        self.assertEqual(overall["tp"] + overall["fn"], self.report["counts"]["gt_objects"])
        for block in self.report["matching_rules"]["iou50"]["per_bucket"].values():
            self.assertEqual(block["tp"] + block["fn"], block["gt_objects"])

    def test_latency_is_reported_and_labelled_offline(self):
        self.assertGreater(self.report["latency"]["n"], 0)
        self.assertIn("not camera-to-command", self.report["latency"]["note"])

    def test_negative_frames_contribute_false_positives(self):
        self.assertGreater(self.report["counts"]["gt_negative_frames"], 0)
        self.assertGreater(self.report["matching_rules"]["iou50"]["overall"]["fp"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
