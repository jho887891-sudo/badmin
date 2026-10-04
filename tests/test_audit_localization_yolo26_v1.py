#!/usr/bin/env python3
"""Tests for tools/audit_localization_yolo26_v1.py.

Pure functions plus writers on synthetic boxes: no model, no GPU, no real dataset, no inference.
Run:  python tests/test_audit_localization_yolo26_v1.py -v
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import audit_localization_yolo26_v1 as aud  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402


def gt_box(cx, cy, w, h, W=640, H=640):
    box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    eq = ev.equiv_size_640(w, h, W, H)
    return {"box": box, "eq640": eq, "bucket": ev.bucket_of(eq)}


def sample(key, gts, dets, W=640, H=640):
    return {"key": key, "image": key, "label": "", "img_w": W, "img_h": H, "gt": gts,
            "dets": [{"box": b, "conf": c} for (c, b) in dets]}


def shifted(box, dx, dy):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def fake_result(ap50, ap75, ap90, ap95, iou_med, ctr, norm):
    return {"ap_by_iou": {0.50: ap50, 0.55: None, 0.60: None, 0.65: None, 0.70: None, 0.75: ap75,
                          0.80: None, 0.85: None, 0.90: ap90, 0.95: ap95},
            "iou_dist": {"stats": {"median": iou_med}},
            "errors": {"center_dist_px_640": {"median": ctr}, "normalized_center_error": {"median": norm}}}


class Test1_PerfectLocalization(unittest.TestCase):
    """A perfectly placed prediction is a TP at every IoU threshold with zero error."""

    def test_perfect(self):
        g = gt_box(300, 300, 20, 20)
        s = sample("img", [g], [(0.9, list(g["box"]))])
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["gt"], 1)
        self.assertEqual(r["tp50"], 1)
        self.assertEqual(r["fp50"], 0)
        self.assertEqual(r["fn50"], 0)
        self.assertEqual(r["recall50"], 1.0)
        for t in aud.AUDIT_IOU_THRESHOLDS:
            self.assertAlmostEqual(r["ap_by_iou"][t], 1.0, places=9, msg="AP%.2f" % t)
        self.assertAlmostEqual(r["mAP50-95"], 1.0, places=9)
        self.assertAlmostEqual(r["iou_dist"]["stats"]["median"], 1.0, places=9)
        e = r["errors"]
        self.assertAlmostEqual(e["center_dist_px"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(e["center_dist_px_640"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(e["normalized_center_error"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(e["width_rel_err"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(e["height_rel_err"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(e["area_ratio"]["mean"], 1.0, places=9)
        self.assertAlmostEqual(e["log_width_ratio"]["mean"], 0.0, places=9)
        self.assertEqual(r["near_miss"]["counts"]["A"], 1)
        self.assertEqual(r["near_miss"]["counts"]["B"], 0)


class Test2_TpFpFnAndNearMissNotTp(unittest.TestCase):
    """A detection with IoU in 0.30-0.50 must never be counted as a TP."""

    def test_counts(self):
        g1 = gt_box(100, 100, 100, 100)
        g2 = gt_box(400, 400, 100, 100)
        good = shifted(g1["box"], 5, 0)
        near = shifted(g2["box"], 34, 0)
        iou_near = ev.iou_xyxy(near, g2["box"])
        self.assertTrue(0.30 <= iou_near < 0.50, "expected a B-class IoU, got %.4f" % iou_near)
        s = sample("img", [g1, g2], [(0.9, good), (0.8, near)])
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["gt"], 2)
        self.assertEqual(r["tp50"], 1)
        self.assertEqual(r["fn50"], 1)
        self.assertEqual(r["fp50"], 1)
        self.assertAlmostEqual(r["recall50"], 0.5)
        self.assertAlmostEqual(r["precision_op"], 0.5)
        self.assertEqual(r["near_miss"]["counts"]["A"], 1)
        self.assertEqual(r["near_miss"]["counts"]["B"], 1)
        self.assertEqual(sum(r["near_miss"]["counts"].values()), 2)
        b = r["near_miss"]["records"]["B"][0]
        self.assertAlmostEqual(b["iou"], iou_near, places=9)
        self.assertIsNotNone(b["pred_box"])

    def test_confidence_floor_is_respected(self):
        g = gt_box(100, 100, 100, 100)
        s = sample("img", [g], [(0.10, shifted(g["box"], 2, 2)), (0.90, shifted(g["box"], 40, 0))])
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["detections_op"], 1)
        self.assertEqual(r["detections_all"], 2)
        self.assertEqual(r["tp50"], 0)


class Test3_ApMonotonicAndDrop(unittest.TestCase):
    """AP must be non-increasing in the IoU threshold, and drop to 0 when no box reaches it."""

    def test_monotonic(self):
        g = gt_box(300, 300, 100, 100)
        d1 = shifted(g["box"], 20, 0)
        d2 = shifted(g["box"], 40, 0)
        self.assertTrue(0.65 <= ev.iou_xyxy(d1, g["box"]) < 0.70)
        s = sample("img", [g], [(0.9, d1), (0.5, d2)])
        r = aud.evaluate_localization([s], conf_op=0.25)
        aps = [r["ap_by_iou"][t] for t in aud.AUDIT_IOU_THRESHOLDS]
        for i in range(len(aps) - 1):
            self.assertGreaterEqual(aps[i] + 1e-12, aps[i + 1],
                                    "AP not monotone at %.2f" % aud.AUDIT_IOU_THRESHOLDS[i])
        self.assertAlmostEqual(r["ap_by_iou"][0.50], 1.0, places=9)
        self.assertAlmostEqual(r["ap_by_iou"][0.65], 1.0, places=9)
        self.assertAlmostEqual(r["ap_by_iou"][0.70], 0.0, places=9)
        self.assertAlmostEqual(r["ap_by_iou"][0.95], 0.0, places=9)
        self.assertEqual(r["tp50"], 1)


class Test4_BucketsAndScale(unittest.TestCase):
    """equiv_size_640 bucketing is half-open and error rescaling uses 640/max(W,H)."""

    def test_bucket_boundaries(self):
        self.assertEqual(ev.bucket_of(3.999), "<4")
        self.assertEqual(ev.bucket_of(4.0), "4-6")
        self.assertEqual(ev.bucket_of(5.999), "4-6")
        self.assertEqual(ev.bucket_of(6.0), "6-8")
        self.assertEqual(ev.bucket_of(8.0), "8-12")
        self.assertEqual(ev.bucket_of(16.0), "16-24")
        self.assertEqual(ev.bucket_of(32.0), "32-64")
        self.assertEqual(ev.bucket_of(64.0), ">64")

    def test_scale_640(self):
        self.assertAlmostEqual(aud.scale_error(100.0, 1024, 1024, 640), 62.5, places=9)
        self.assertAlmostEqual(ev.equiv_size_640(100, 100, 1024, 1024), 62.5, places=9)
        self.assertAlmostEqual(aud.scale_error(100.0, 640, 640, 640), 100.0, places=9)
        self.assertAlmostEqual(aud.scale_error(100.0, 2048, 1024, 640), 31.25, places=9)
        self.assertIsNone(aud.scale_error(None, 1024, 1024, 640))

    def test_ratio_error_is_scale_invariant(self):
        g = gt_box(300, 300, 40, 60, W=1024, H=1024)
        p = [g["box"][0] + 3, g["box"][1] + 2, g["box"][2] + 9, g["box"][3] + 8]
        k = 640.0 / 1024.0
        ps = [p[0] * k, p[1] * k, p[2] * k, p[3] * k]
        gs = [g["box"][0] * k, g["box"][1] * k, g["box"][2] * k, g["box"][3] * k]
        self.assertAlmostEqual(aud.width_relative_error(g["box"], p), aud.width_relative_error(gs, ps), places=9)
        self.assertAlmostEqual(aud.height_relative_error(g["box"], p), aud.height_relative_error(gs, ps), places=9)
        self.assertAlmostEqual(aud.area_ratio(g["box"], p), aud.area_ratio(gs, ps), places=9)
        # the 1024-frame equiv size equals the 640-frame size of the rescaled box
        self.assertAlmostEqual(g["eq640"], ev.equiv_size_640(gs[2] - gs[0], gs[3] - gs[1], 640, 640), places=9)

    def test_per_bucket_accumulation(self):
        small = gt_box(100, 100, 6, 6)      # 6-8 bucket at 640x640
        big = gt_box(400, 400, 40, 40)      # 32-64 bucket
        self.assertEqual(small["bucket"], "6-8")
        self.assertEqual(big["bucket"], "32-64")
        s = sample("img", [small, big], [(0.9, list(small["box"])), (0.8, shifted(big["box"], 20, 0))])
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["buckets"]["6-8"]["gt"], 1)
        self.assertEqual(r["buckets"]["6-8"]["tp50"], 1)
        self.assertEqual(r["buckets"]["32-64"]["gt"], 1)
        self.assertEqual(r["buckets"]["32-64"]["tp50"], 0)
        self.assertAlmostEqual(r["buckets"]["6-8"]["recall"], 1.0, places=9)
        self.assertAlmostEqual(r["buckets"]["32-64"]["recall"], 0.0, places=9)
        self.assertAlmostEqual(r["buckets"]["6-8"]["AP50"], 1.0, places=9)
        self.assertAlmostEqual(r["buckets"]["32-64"]["AP50"], 0.0, places=9)
        self.assertEqual(r["buckets"]["<4"]["gt"], 0)
        self.assertIsNone(r["buckets"]["<4"]["recall"])
        tot = sum(r["buckets"][lab]["tp50"] for lab in ev.BUCKET_LABELS)
        self.assertEqual(tot, r["tp50"])


class Test5_ErrorMath(unittest.TestCase):
    """Signed centre error, normalized centre error, size ratios and log ratios."""

    def test_known_values(self):
        g = [100.0, 100.0, 200.0, 200.0]
        p = [110.0, 130.0, 220.0, 240.0]
        dx, dy, dist = aud.center_error(g, p)
        # GT centre (150, 150); pred centre (165, 185)
        self.assertAlmostEqual(dx, 15.0, places=9)
        self.assertAlmostEqual(dy, 35.0, places=9)
        self.assertAlmostEqual(dist, math.sqrt(225.0 + 1225.0), places=9)
        self.assertAlmostEqual(aud.normalized_center_error(g, p), math.sqrt(1450.0) / 100.0, places=9)
        self.assertAlmostEqual(aud.width_relative_error(g, p), 0.10, places=9)
        self.assertAlmostEqual(aud.height_relative_error(g, p), 0.10, places=9)
        self.assertAlmostEqual(aud.area_ratio(g, p), 1.21, places=9)
        wr, hr, lwr, lhr = aud.signed_size_ratios(g, p)
        self.assertAlmostEqual(wr, 1.10, places=9)
        self.assertAlmostEqual(hr, 1.10, places=9)
        self.assertAlmostEqual(lwr, math.log(1.10), places=9)
        self.assertAlmostEqual(lhr, math.log(1.10), places=9)
        self.assertIsNone(aud.width_relative_error([0, 0, 0, 0], p))

    def test_px640_conversion_in_metrics(self):
        g = gt_box(500, 500, 20, 20, W=1024, H=1024)
        p = shifted(g["box"], 2, 0)
        s = sample("img", [g], [(0.9, p)], W=1024, H=1024)
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["tp50"], 1)
        self.assertAlmostEqual(r["errors"]["center_dist_px"]["mean"], 2.0, places=6)
        self.assertAlmostEqual(r["errors"]["center_dist_px_640"]["mean"], 2.0 * 640.0 / 1024.0, places=6)
        self.assertAlmostEqual(r["errors"]["normalized_center_error"]["mean"], 2.0 / 20.0, places=6)
        self.assertAlmostEqual(r["errors"]["center_abs_dx"]["mean"], 2.0, places=6)
        self.assertAlmostEqual(r["errors"]["center_abs_dy"]["mean"], 0.0, places=6)


class Test6_NearMissBands(unittest.TestCase):
    """Band edges A >= 0.50, B 0.30-0.50, C 0.10-0.30, D < 0.10."""

    def test_band_edges(self):
        self.assertEqual(aud.near_miss_class(1.0), "A")
        self.assertEqual(aud.near_miss_class(0.50), "A")
        self.assertEqual(aud.near_miss_class(0.4999), "B")
        self.assertEqual(aud.near_miss_class(0.30), "B")
        self.assertEqual(aud.near_miss_class(0.2999), "C")
        self.assertEqual(aud.near_miss_class(0.10), "C")
        self.assertEqual(aud.near_miss_class(0.0999), "D")
        self.assertEqual(aud.near_miss_class(0.0), "D")

    def test_class_d_when_no_match(self):
        g = gt_box(100, 100, 40, 40)
        far = gt_box(500, 500, 40, 40)
        s = sample("img", [g], [(0.9, list(far["box"]))])
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["near_miss"]["counts"]["D"], 1)
        self.assertEqual(r["near_miss"]["counts"]["A"], 0)
        self.assertEqual(r["tp50"], 0)
        self.assertEqual(sum(r["near_miss"]["counts"].values()), r["gt"])

    def test_empty_gt_set(self):
        s = sample("img", [], [(0.9, [0, 0, 10, 10])])
        r = aud.evaluate_localization([s], conf_op=0.25)
        self.assertEqual(r["gt"], 0)
        self.assertEqual(r["tp50"], 0)
        self.assertEqual(r["fp50"], 1)
        self.assertIsNone(r["recall50"])
        self.assertEqual(r["images"], 1)


class Test7_MatchingAndStats(unittest.TestCase):

    def test_match_by_iou_is_one_to_one(self):
        gts = [[0, 0, 10, 10], [10, 0, 20, 10]]
        dets = [(0.9, [0, 0, 10, 10]), (0.8, [1, 0, 11, 10])]
        m = aud.match_by_iou(gts, dets, 0.1)
        gi = [x[0] for x in m]
        di = [x[1] for x in m]
        self.assertEqual(len(gi), len(set(gi)))
        self.assertEqual(len(di), len(set(di)))
        self.assertEqual(m[0], (0, 0, 1.0))

    def test_iou_histogram_bins(self):
        h = aud.iou_histogram([0.50, 0.5999, 0.60, 0.95, 1.0])
        self.assertEqual(h["0.50-0.60"], 2)
        self.assertEqual(h["0.60-0.70"], 1)
        self.assertEqual(h[">=0.95"], 2)
        self.assertEqual(sum(h.values()), 5)
        self.assertEqual(sum(aud.iou_histogram([]).values()), 0)

    def test_stats_edges(self):
        self.assertEqual(aud.basic_stats([])["n"], 0)
        self.assertEqual(aud.quantiles([]), {})
        st = aud.basic_stats([1.0, 2.0, 3.0])
        self.assertAlmostEqual(st["mean"], 2.0, places=9)
        self.assertAlmostEqual(st["median"], 2.0, places=9)
        q = aud.quantiles([0.0, 1.0], (0.5,))
        self.assertAlmostEqual(q["P50"], 0.5, places=9)


class Test8_DecisionRule(unittest.TestCase):

    def test_case_a_recall_only(self):
        a = fake_result(0.30, 0.20, 0.10, 0.05, 0.70, 8.0, 0.30)
        b = fake_result(0.40, 0.20, 0.10, 0.05, 0.70, 8.0, 0.30)
        v, d = aud.decision_rule(a, b)
        self.assertIn("CASE A", v)
        self.assertAlmostEqual(d["dAP50"], 0.10, places=9)

    def test_case_b_localization(self):
        a = fake_result(0.30, 0.20, 0.10, 0.05, 0.70, 8.0, 0.30)
        b = fake_result(0.305, 0.25, 0.15, 0.10, 0.75, 6.0, 0.22)
        v, _d = aud.decision_rule(a, b)
        self.assertIn("CASE B", v)

    def test_case_c_both(self):
        a = fake_result(0.30, 0.20, 0.10, 0.05, 0.70, 8.0, 0.30)
        b = fake_result(0.40, 0.25, 0.15, 0.10, 0.76, 6.0, 0.22)
        v, _d = aud.decision_rule(a, b)
        self.assertIn("CASE C", v)

    def test_flat_is_indeterminate(self):
        a = fake_result(0.30, 0.20, 0.10, 0.05, 0.70, 8.0, 0.30)
        b = fake_result(0.305, 0.202, 0.101, 0.052, 0.702, 7.95, 0.299)
        v, _d = aud.decision_rule(a, b)
        self.assertIn("INDETERMINATE", v)


class Test9_WritersAndReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        g_small = gt_box(100, 100, 6, 6)
        g_mid = gt_box(300, 300, 40, 40)
        g_big = gt_box(500, 500, 80, 80)
        s1 = sample("a.png", [g_small, g_mid],
                    [(0.9, list(g_small["box"])), (0.7, shifted(g_mid["box"], 25, 0)),
                     (0.4, shifted(g_big["box"], 100, 0)), (0.1, [0, 0, 5, 5])])
        s2 = sample("b.png", [g_big], [(0.8, shifted(g_big["box"], 6, 4))])
        self.samples = [s1, s2]
        self.results = {
            ("a_best", "val"): aud.evaluate_localization(self.samples, conf_op=0.25),
            ("b_best", "val"): aud.evaluate_localization(self.samples, conf_op=0.25),
            ("b_best", "controlled_capability/images"): aud.evaluate_localization(self.samples, conf_op=0.25),
        }

    def test_six_csvs(self):
        m = self.out
        aud.write_ap_by_iou(m / "localization_ap_by_iou.csv", self.results)
        aud.write_iou_distribution(m / "localization_iou_distribution.csv", self.results)
        aud.write_error_summary(m / "localization_error_summary.csv", self.results)
        aud.write_size_buckets(m / "localization_size_buckets.csv", self.results)
        aud.write_checkpoint_compare(m / "localization_checkpoint_compare.csv", self.results)
        aud.write_near_miss(m / "localization_near_miss.csv", self.results)
        expect = {
            "localization_ap_by_iou.csv": ["checkpoint", "set", "AP50", "AP95", "mAP50-95"],
            "localization_iou_distribution.csv": ["n_tp", "median", "bin_0.50-0.60", "bin_>=0.95"],
            "localization_error_summary.csv": ["n_tp", "center_dist_px_median", "center_dist_px_640_median",
                                               "normalized_center_error_median", "width_rel_err_median",
                                               "height_rel_err_median"],
            "localization_size_buckets.csv": ["bucket", "GT", "TP@0.5", "Recall@0.5", "AP50", "AP90"],
            "localization_checkpoint_compare.csv": ["set", "bucket", "d_Recall", "d_iou_med", "d_AP50"],
            "localization_near_miss.csv": ["class", "GT", "count", "share"],
        }
        for fname, cols in expect.items():
            p = m / fname
            self.assertTrue(p.exists(), fname)
            with open(p, encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertGreaterEqual(len(rows), 2, fname)
            for c in cols:
                self.assertIn(c, rows[0], "%s missing column %s" % (fname, c))
        with open(m / "localization_size_buckets.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.results) * len(ev.BUCKET_LABELS))
        with open(m / "localization_checkpoint_compare.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertIn("ALL", [r["bucket"] for r in rows])

    def test_report_written(self):
        args = argparse.Namespace(conf_op=0.25, conf_floor=0.001, compare_a="a_best", compare_b="b_best",
                                  examples_per_kind=30, command_line="python tools/audit_localization_yolo26_v1.py")
        ckpts = [{"name": "a_best", "epoch": 6, "role": "baseline", "sha256": "deadbeef", "bytes": 1,
                  "path": "x.pt"},
                 {"name": "b_best", "epoch": 10, "role": "candidate", "sha256": "cafebabe", "bytes": 1,
                  "path": "y.pt"}]
        p = self.out / "LOCALIZATION_AUDIT.md"
        aud.write_report(p, self.results, ckpts, args, ["val"], [])
        self.assertTrue(p.exists())
        txt = p.read_text(encoding="utf-8")
        for token in ["## 1.", "## 3.", "## 4.", "## 6.", "## 7.", "## 8.", "## 9.", "## 10.",
                      "CASE", "equiv_size_640", "AP95"]:
            self.assertIn(token, txt, token)


if __name__ == "__main__":
    unittest.main()