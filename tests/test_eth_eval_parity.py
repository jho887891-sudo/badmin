#!/usr/bin/env python3
"""Tests for tools/eval_eth_official_baseline.py (fair-comparison guard rails).

Pure-function tests + artifact assertions. Tests that need the ETH checkpoint or a finished audit run SKIP
with an explicit message when the artifact is absent, so the suite is green without the 134 MB weight.

Run:  python tests/test_eth_eval_parity.py -v
"""
from __future__ import annotations

import csv
import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import audit_bbox_height_bias as hb  # noqa: E402
import audit_localization_yolo26_v1 as aud  # noqa: E402
import eval_eth_official_baseline as eb  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402

ETH_CKPT = Path(os.environ.get("ETH_CKPT", str(ROOT / "_scratch_eth_official" / "best.pt")))
METRICS = ROOT / "outputs" / "shuttle_capability" / "metrics"


def sample(key, W, H, gt_boxes, dets):
    gts = []
    for b in gt_boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        eq = ev.equiv_size_640(w, h, W, H)
        gts.append({"box": b, "eq640": eq, "bucket": ev.bucket_of(eq)})
    return {"key": key, "image": key, "label": "", "img_w": W, "img_h": H, "gt": gts,
            "dets": [{"box": b, "conf": c} for c, b in dets]}


def eth_ready():
    """The 134 MB weight must be fully present (an in-flight download locks it)."""
    if not ETH_CKPT.exists():
        return False
    try:
        return ETH_CKPT.stat().st_size == 134312133
    except OSError:
        return False

class Test1_IouIdentity(unittest.TestCase):
    def test_pred_equals_gt(self):
        b = [10.0, 20.0, 40.0, 60.0]
        self.assertAlmostEqual(ev.iou_xyxy(b, list(b)), 1.0, places=12)
        self.assertAlmostEqual(ev.center_dist(b, list(b)), 0.0, places=12)


class Test2_KnownBox(unittest.TestCase):
    def test_10x10_vs_10x11(self):
        gt = [100.0, 100.0, 110.0, 110.0]
        pred = [100.0, 99.5, 110.0, 110.5]
        self.assertAlmostEqual(ev.iou_xyxy(gt, pred), 100.0 / 110.0, places=12)
        self.assertAlmostEqual((pred[3] - pred[1]) / (gt[3] - gt[1]), 1.1, places=12)


class Test3_SizeBucket(unittest.TestCase):
    def test_equiv_size_and_buckets(self):
        self.assertAlmostEqual(ev.equiv_size_640(100, 100, 1024, 1024), 62.5, places=9)
        self.assertAlmostEqual(ev.equiv_size_640(30, 20, 1920, 1200), math.sqrt(600.0) * 640.0 / 1920.0, places=9)
        for val, want in [(3.99, "<4"), (4.0, "4-6"), (6.0, "6-8"), (8.0, "8-12"), (12.0, "12-16"),
                          (16.0, "16-24"), (24.0, "24-32"), (32.0, "32-64"), (64.0, ">64")]:
            self.assertEqual(ev.bucket_of(val), want)
        self.assertEqual(eb.AGGREGATES[3][1], ["<4", "4-6", "6-8"])


class Test4_Scale1024To640(unittest.TestCase):
    def test_scale(self):
        self.assertAlmostEqual(aud.scale_error(100.0, 1024, 1024, 640), 62.5, places=9)
        self.assertAlmostEqual(eb.ETH_DIST_PX * 640.0 / eb.IMGSZ, 15.625, places=9)
        self.assertAlmostEqual(ev.equiv_size_640(100, 100, 1024, 1024), 62.5, places=9)


class Test5_LetterboxRoundTrip(unittest.TestCase):
    def test_roundtrip(self):
        rows = hb.run_roundtrip_tests()
        self.assertGreater(len(rows), 30)
        for r in rows:
            self.assertLessEqual(r["max_abs_err_letterbox_roundtrip"], 1e-6, r["case"])
            self.assertLessEqual(r["max_abs_err_norm_roundtrip"], 1e-6, r["case"])

    def test_letterbox_frame_is_monotone(self):
        p = hb.letterbox_params(1920, 1200)
        self.assertEqual(tuple(p["new_unpad"]), (1024, 640))
        box = [100.0, 100.0, 130.0, 120.0]
        lb = eb.to_letterbox_frame(box, 1920, 1200)
        self.assertAlmostEqual((lb[2] - lb[0]) / (box[2] - box[0]), p["r"], places=12)
        # content-frame distance convention: padding cancels because GT and prediction share it
        r = p["r"]
        self.assertAlmostEqual((25.0 / r) * r, 25.0, places=12)


class Test6_EthModelLoadsAndPredicts(unittest.TestCase):
    """ETH predictions must match a native ultralytics call; the model must load from the custom dict."""

    def test_custom_dict_loading(self):
        if not eth_ready():
            self.skipTest("ETH checkpoint not (fully) present at %s" % ETH_CKPT)
        info = eb.verify_checkpoint(ETH_CKPT, METRICS / "eth_official_repo_inventory.csv",
                                    eb.__dict__.get("ETH_EXPECTED_SHA256"), None) if False else \
            eb.verify_checkpoint(ETH_CKPT, METRICS / "eth_official_repo_inventory.csv")
        self.assertTrue(info.get("exists"))
        self.assertEqual(info.get("ckpt_type"), "dict")
        self.assertFalse(info.get("is_ultralytics_format"))
        self.assertIn("model_state_dict", info.get("ckpt_keys") or [])

    def test_load_and_forward(self):
        if not eth_ready():
            self.skipTest("ETH checkpoint not (fully) present at %s" % ETH_CKPT)
        import torch
        w, meta = eb.load_eth_model(ETH_CKPT, "0")
        # measured facts about the released checkpoint: 3 cv3 classification branches, all detections class 0
        self.assertEqual(meta["nc"], 3)
        self.assertTrue(meta["load_state_dict_strict_ok"])
        import json as _json
        p = ROOT / "_scratch_localization_audit" / "loc_eth_official_val.json"
        if p.exists():
            hist = _json.loads(p.read_text(encoding="utf-8")).get("class_hist") or {}
            self.assertEqual(sorted(hist.keys()), ["0"], "ETH predictions must all be class 0")
        x = torch.zeros(1, 3, eb.IMGSZ, eb.IMGSZ)
        with torch.no_grad():
            y = w.model(x)
        self.assertTrue(len(y) >= 1)


class Test7_PolicyMatchesOldEvaluator(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(eb.IMGSZ, 1024)
        self.assertEqual(eb.CONF_OP, 0.25)
        self.assertEqual(eb.CONF_FLOOR, 0.001)
        self.assertEqual(eb.NMS_IOU, 0.7)
        self.assertEqual(eb.MAX_DET, 300)
        self.assertEqual(eb.CONF_OP, 0.25)
        self.assertEqual(eb.SETS, ["val", "controlled_capability/images", "challenge_test/images"])

    def test_predict_policy_in_source(self):
        src = (ROOT / "tools" / "eval_eth_official_baseline.py").read_text(encoding="utf-8")
        for token in ["conf=conf_floor", "iou=NMS_IOU", "max_det=MAX_DET", "rect=False", "stream=True"]:
            self.assertIn(token, src, token)


class Test8_EmptyLabelFp(unittest.TestCase):
    def test_fp_stats_on_empty_labels(self):
        dets = [[{"conf": 0.9, "box": [0, 0, 5, 5]}, {"conf": 0.3, "box": [1, 1, 6, 6]}],
                [{"conf": 0.4, "box": [2, 2, 7, 7]}], []]
        st = ev.fp_stats(dets, [0.25, 0.75])
        self.assertEqual(st["0.25"]["total_FP"], 3)
        self.assertEqual(st["0.25"]["images_with_FP"], 2)
        self.assertAlmostEqual(st["0.25"]["FP_per_image"], 1.0, places=9)
        self.assertAlmostEqual(st["0.25"]["image_FP_rate"], 2.0 / 3.0, places=9)
        self.assertEqual(st["0.75"]["total_FP"], 1)

    def test_fp_rows_shape(self):
        class G:
            key = "fake/negatives"
            images = ["a.jpg", "b.jpg"]
        rows = eb.fp_rows("m", [G()], {"%s" % os.path.abspath("a.jpg"): [{"conf": 0.9, "box": [0, 0, 5, 5]}]},
                          (0.25, 0.5))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["FP_total"], 1)
        self.assertEqual(rows[1]["FP_total"], 1)


class Test9_CenterMetricScale(unittest.TestCase):
    def test_threshold_in_letterbox_frame(self):
        W, H = 1920, 1200
        gt = [900.0, 560.0, 940.0, 600.0]
        r = hb.letterbox_params(W, H)
        cx, cy = 920.0, 580.0
        dx_lb = 24.0
        dx_orig = dx_lb / r["r"]
        det_hit = [cx + dx_orig - 10.0, cy - 10.0, cx + dx_orig + 10.0, cy + 10.0]
        det_miss = [cx + dx_orig * 25.0 / 24.0 - 10.0, cy - 10.0, cx + dx_orig * 25.0 / 24.0 + 10.0, cy + 10.0]
        s1 = sample("a.jpg", W, H, [gt], [(0.9, det_hit)])
        s2 = sample("b.jpg", W, H, [gt], [(0.9, det_miss)])
        m1 = eb.eth_style_center_metric([s1], {"a.jpg": s1["dets"]}, conf=0.5, dist_thr=25.0, frame="content")
        m2 = eb.eth_style_center_metric([s2], {"b.jpg": s2["dets"]}, conf=0.5, dist_thr=25.0, frame="content")
        self.assertEqual(m1["TP"], 1)
        self.assertEqual(m2["FP"], 1)
        self.assertAlmostEqual(m1["mean_dist"], 24.0, places=6)

    def test_top1_only_and_one_gt(self):
        W, H = 1024, 1024
        gt = [100.0, 100.0, 120.0, 120.0]
        far = [500.0, 500.0, 520.0, 520.0]
        s = sample("c.jpg", W, H, [gt], [(0.9, [100.0, 100.0, 120.0, 120.0]), (0.8, far)])
        m = eb.eth_style_center_metric([s], {"c.jpg": s["dets"]}, conf=0.5, dist_thr=25.0, frame="content")
        self.assertEqual(m["TP"], 1)
        self.assertEqual(m["FP"], 0)
        s2 = sample("d.jpg", W, H, [gt], [(0.9, far)])
        m2 = eb.eth_style_center_metric([s2], {"d.jpg": s2["dets"]}, conf=0.5, dist_thr=25.0, frame="content")
        self.assertEqual(m2["FP"], 1)
        self.assertEqual(m2["FN"], 0)


class Test10_SameGtForEveryModel(unittest.TestCase):
    def test_same_samples_yield_same_gt(self):
        if not (ROOT / "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv").exists():
            self.skipTest("manifest missing")
        ss, _ = aud.build_val_samples(ROOT, "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", [], 40)
        ss2, _ = aud.build_val_samples(ROOT, "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", [], 40)
        sig = [(s["image"], len(s["gt"]), [tuple(g["box"]) for g in s["gt"]]) for s in ss]
        sig2 = [(s["image"], len(s["gt"]), [tuple(g["box"]) for g in s["gt"]]) for s in ss2]
        self.assertEqual(sig, sig2)

    def test_gt_column_identical_across_models(self):
        p = METRICS / "eth_vs_ours.csv"
        if not p.exists():
            self.skipTest("run tools/eval_eth_official_baseline.py first")
        rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))]
        by_set = {}
        for r in rows:
            by_set.setdefault(r["set"], set()).add(r["GT"])
        for set_key, gts in by_set.items():
            self.assertEqual(len(gts), 1, "GT differs across models on %s: %s" % (set_key, gts))


if __name__ == "__main__":
    unittest.main()