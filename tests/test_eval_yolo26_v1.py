#!/usr/bin/env python3
"""Tests for tools/eval_yolo26_v1.py - size-bucketed evaluation.

Pure-function tests: no model, no GPU, no dataset, no live run is touched.
Run:  python tests/test_eval_yolo26_v1.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import eval_yolo26_v1 as ev  # noqa: E402


def gt(box, bucket):
    return {"box": box, "bucket": bucket, "eq640": 0.0}


def det(box, conf):
    return {"box": box, "conf": conf}


class Test1_SingleHit(unittest.TestCase):
    """1 GT (7 px, bucket 6-8), 1 prediction with IoU 0.8 -> GT=1 TP=1 FN=0 Recall=1."""

    def test_single_hit(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "6-8")
        d = det([1.0, 1.0, 11.0, 11.0], 0.9)          # IoU = 81/119 = 0.6807... use a tighter one
        d2 = det([0.0, 0.0, 8.944, 8.944], 0.9)       # area ratio 0.8 -> IoU = 0.8
        self.assertAlmostEqual(ev.iou_xyxy(g["box"], d2["box"]), 0.8, places=3)
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [d2]}], conf_op=0.25)
        b = out["buckets"]["6-8"]
        self.assertEqual(b["gt"], 1)
        self.assertEqual(b["tp"], 1)
        self.assertEqual(b["fn"], 0)
        self.assertAlmostEqual(b["recall"], 1.0, places=9)
        self.assertEqual(out["overall"]["fp"], 0)


class Test2_Miss(unittest.TestCase):
    """1 GT (7 px), no prediction -> GT=1 TP=0 FN=1 Recall=0."""

    def test_miss(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "6-8")
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": []}], conf_op=0.25)
        b = out["buckets"]["6-8"]
        self.assertEqual((b["gt"], b["tp"], b["fn"]), (1, 0, 1))
        self.assertAlmostEqual(b["recall"], 0.0, places=9)
        self.assertEqual(out["overall"]["fp"], 0)


class Test3_Boundaries(unittest.TestCase):
    """Every boundary value lands in exactly one bucket: 0 duplicates, 0 misses."""

    def test_boundaries(self):
        values = [3.9, 4.0, 5.9, 6.0, 7.9, 8.0, 11.9, 12.0, 15.9, 16.0,
                  23.9, 24.0, 31.9, 32.0, 63.9, 64.0]
        expected = ["<4", "4-6", "4-6", "6-8", "6-8", "8-12", "8-12", "12-16",
                    "12-16", "16-24", "16-24", "24-32", "24-32", "32-64", "32-64", ">64"]
        got = [ev.bucket_of(v) for v in values]
        self.assertEqual(got, expected)
        self.assertEqual(ev.bucket_of(4.0), "4-6")
        self.assertEqual(ev.bucket_of(3.999999), "<4")
        self.assertEqual(ev.bucket_of(0.0), "<4")
        self.assertEqual(ev.bucket_of(1e6), ">64")
        # no double counting and no losses: one sample per value
        samples = [{"key": str(i), "gt": [gt([0, 0, 1, 1], ev.bucket_of(v))], "dets": []}
                   for i, v in enumerate(values)]
        out = ev.evaluate_samples(samples, conf_op=0.25)
        total = sum(out["buckets"][lab]["gt"] for lab in ev.BUCKET_LABELS)
        self.assertEqual(total, len(values))
        for lab, want in zip(ev.BUCKET_LABELS, [1, 2, 2, 2, 2, 2, 2, 2, 1]):
            self.assertEqual(out["buckets"][lab]["gt"], want, lab)

    def test_equiv_size_640_is_scale_invariant_in_the_640_frame(self):
        # 1920x1200, box 192x120 px -> eq_orig=151.789 -> x(640/1920) = 50.596
        eq = ev.equiv_size_640(192.0, 120.0, 1920.0, 1200.0)
        self.assertAlmostEqual(eq, 50.5964, places=3)
        self.assertEqual(ev.bucket_of(eq), "32-64")
        # the same physical object at 960x600 must give the same equiv_size_640
        eq2 = ev.equiv_size_640(96.0, 60.0, 960.0, 600.0)
        self.assertAlmostEqual(eq, eq2, places=9)


class Test4_CombinedRecall(unittest.TestCase):
    """6-8: TP=20/GT=25, 8-12: TP=60/GT=100 -> (20+60)/(25+100)=0.64, NOT (0.8+0.6)/2=0.7."""

    def test_combined_recall_is_gt_weighted(self):
        gts, dets = [], []
        for i in range(25):
            box = [float(i * 100), 0.0, float(i * 100 + 7), 7.0]
            gts.append(gt(box, "6-8"))
            if i < 20:
                dets.append(det(box, 0.9))
        for i in range(100):
            box = [float(i * 100), 500.0, float(i * 100 + 10), 510.0]
            gts.append(gt(box, "8-12"))
            if i < 60:
                dets.append(det(box, 0.9))
        out = ev.evaluate_samples([{"key": "a", "gt": gts, "dets": dets}], conf_op=0.25)
        b68, b812 = out["buckets"]["6-8"], out["buckets"]["8-12"]
        self.assertEqual((b68["tp"], b68["gt"]), (20, 25))
        self.assertEqual((b812["tp"], b812["gt"]), (60, 100))
        tp_sum = b68["tp"] + b812["tp"]
        gt_sum = b68["gt"] + b812["gt"]
        combined_6_12 = tp_sum / gt_sum
        self.assertAlmostEqual(combined_6_12, 0.64, places=9)
        self.assertNotAlmostEqual(combined_6_12, 0.7, places=3)          # never the naive mean
        mean_of_recalls = (b68["recall"] + b812["recall"]) / 2.0
        self.assertAlmostEqual(mean_of_recalls, 0.7, places=9)
        self.assertNotAlmostEqual(combined_6_12, mean_of_recalls, places=3)
        self.assertEqual(out["overall"]["tp"], 80)
        self.assertEqual(out["overall"]["fn"], 45)
        # combined_recall_6_16 is the same GT-weighted construction over three buckets
        c = out["combined_recall_6_16"]
        self.assertEqual((c["tp"], c["gt"]), (80, 125))
        self.assertAlmostEqual(c["recall"], 0.64, places=9)


class Matching(unittest.TestCase):
    def test_one_to_one_no_double_tp(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "8-12")
        d1 = det([0.0, 0.0, 10.0, 10.0], 0.9)
        d2 = det([0.0, 0.0, 10.0, 10.0], 0.8)
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [d1, d2]}], conf_op=0.25)
        self.assertEqual(out["buckets"]["8-12"]["tp"], 1)
        self.assertEqual(out["overall"]["fp"], 1)

    def test_greedy_prefers_higher_confidence(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "8-12")
        good = det([0.0, 0.0, 10.0, 10.0], 0.9)
        bad = det([8.0, 8.0, 18.0, 18.0], 0.95)      # IoU 0.22 -> not a match
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [bad, good]}], conf_op=0.25)
        self.assertEqual(out["buckets"]["8-12"]["tp"], 1)
        self.assertEqual(out["overall"]["fp"], 1)

    def test_fp_gets_no_bucket(self):
        s = ev.evaluate_samples([{"key": "a", "gt": [], "dets": [det([0, 0, 10, 10], 0.9)]}], conf_op=0.25)
        self.assertEqual(s["overall"]["fp"], 1)
        for lab in ev.BUCKET_LABELS:
            self.assertEqual(s["buckets"][lab]["gt"], 0)
            self.assertIsNone(s["buckets"][lab]["precision"])   # plan A: FP are never bucketed


class SecondaryMetric(unittest.TestCase):
    def test_center_distance_is_separate_from_iou(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "8-12")
        d = det([20.0, 0.0, 30.0, 10.0], 0.9)        # IoU = 0, centers 20 px apart
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [d]}], conf_op=0.25, center_thr=25.0)
        self.assertEqual(out["buckets"]["8-12"]["tp"], 0)                  # primary: miss
        self.assertEqual(out["secondary_center_distance"]["overall"]["tp"], 1)   # secondary: hit
        self.assertEqual(out["secondary_center_distance"]["buckets"]["8-12"]["tp"], 1)
        self.assertEqual(out["secondary_center_distance"]["threshold_px_original"], 25.0)

    def test_center_distance_threshold_is_respected(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "8-12")
        d = det([40.0, 0.0, 50.0, 10.0], 0.9)        # centers 40 px apart > 25 px
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [d]}], conf_op=0.25, center_thr=25.0)
        self.assertEqual(out["secondary_center_distance"]["overall"]["tp"], 0)


class AveragedPrecision(unittest.TestCase):
    def test_perfect_detection_gives_ap1(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "8-12")
        d = det([0.0, 0.0, 10.0, 10.0], 0.9)
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [d]}], conf_op=0.25)
        self.assertAlmostEqual(out["buckets"]["8-12"]["ap50"], 1.0, places=6)
        self.assertAlmostEqual(out["buckets"]["8-12"]["ap5095"], 1.0, places=6)

    def test_false_positive_lowers_precision_not_ap(self):
        g = gt([0.0, 0.0, 10.0, 10.0], "8-12")
        d1 = det([0.0, 0.0, 10.0, 10.0], 0.9)
        d2 = det([500.0, 500.0, 510.0, 510.0], 0.5)
        out = ev.evaluate_samples([{"key": "a", "gt": [g], "dets": [d1, d2]}], conf_op=0.25)
        self.assertAlmostEqual(out["buckets"]["8-12"]["ap50"], 1.0, places=6)
        self.assertEqual(out["overall"]["fp"], 1)
        self.assertAlmostEqual(out["overall"]["precision"], 0.5, places=6)

    def test_low_sample_flag(self):
        gts = [gt([float(i), 0.0, float(i) + 5, 5.0], "<4") for i in range(5)]
        out = ev.evaluate_samples([{"key": "a", "gt": gts, "dets": []}], conf_op=0.25)
        self.assertTrue(out["buckets"]["<4"]["low_sample"])
        big = [gt([float(i), 0.0, float(i) + 5, 5.0], "<4") for i in range(50)]
        out2 = ev.evaluate_samples([{"key": "a", "gt": big, "dets": []}], conf_op=0.25)
        self.assertFalse(out2["buckets"]["<4"]["low_sample"])


if __name__ == "__main__":
    unittest.main(verbosity=2)