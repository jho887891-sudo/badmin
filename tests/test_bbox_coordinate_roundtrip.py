#!/usr/bin/env python3
"""Coordinate round-trip tests for tools/audit_bbox_height_bias.py.

Pure maths + the installed ultralytics implementation: no GPU, no dataset, no training.

Run:  python tests/test_bbox_coordinate_roundtrip.py -v
"""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import audit_bbox_height_bias as hb  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402

TOL = 1e-6


class Test1_NormPixelNorm(unittest.TestCase):
    """normalized YOLO label -> pixel xyxy -> normalized must close."""

    def test_roundtrip_all_resolutions(self):
        for (W, H) in [(1920, 1200), (1920, 1080), (960, 960), (2048, 1000), (640, 480)]:
            for box in ([W / 2 - 15, H / 2 - 10, W / 2 + 15, H / 2 + 10],
                        [2.0, 2.0, 32.0, 22.0],
                        [W - 32.0, H - 22.0, W - 2.0, H - 2.0],
                        [W / 2 - 2, H / 2 - 3, W / 2 + 2, H / 2 + 3]):
                cx, cy, bw, bh = hb.pixel_to_yolo_norm(box, W, H)
                back = hb.yolo_norm_to_pixel(cx, cy, bw, bh, W, H)
                self.assertLessEqual(max(abs(a - b) for a, b in zip(box, back)), TOL,
                                     "%dx%d %s" % (W, H, box))


class Test2_LetterboxInverse(unittest.TestCase):
    """original -> letterbox -> inverse -> original must close (nominal r path)."""

    def test_roundtrip(self):
        for (name, _W, _H, box, _kind) in hb.roundtrip_cases():
            p = hb.letterbox_params(_W, _H)
            f = hb.letterbox_forward_box(box, p)
            inv = hb.scale_boxes_inverse(f, p, _W, _H)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(box, inv)), TOL, name)

    def test_gain_is_isotropic(self):
        p = hb.letterbox_params(2048, 1000)
        self.assertEqual(p["ratio"][0], p["ratio"][1])
        self.assertAlmostEqual(p["r"], 1024 / 2048.0, places=12)


class Test3_Resolutions(unittest.TestCase):
    """1920x1200 / 1920x1080 / 960x960 explicit checks, including the padded shape."""

    def test_shapes_and_roundtrip(self):
        for (W, H, exp_unpad) in [(1920, 1200, (1024, 640)), (1920, 1080, (1024, 576)), (960, 960, (1024, 1024))]:
            p = hb.letterbox_params(W, H)
            self.assertEqual(tuple(p["new_unpad"]), exp_unpad, "%dx%d" % (W, H))
            box = [W * 0.25, H * 0.25, W * 0.25 + 30.0, H * 0.25 + 20.0]
            inv = hb.scale_boxes_inverse(hb.letterbox_forward_box(box, p), p, W, H)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(box, inv)), TOL)
            for a, b in (("left", 1024 - exp_unpad[0]), ("top", 1024 - exp_unpad[1])):
                self.assertEqual(p[a] + p["right" if a == "left" else "bottom"], b)


class Test4_TinyBoxesNoHeightScaling(unittest.TestCase):
    """4x6 / 6x8 / 8x12 must come back with the SAME height: no extra height scaling anywhere."""

    def test_no_extra_height_scaling(self):
        for (W, H) in [(1920, 1200), (1920, 1080), (960, 960), (2048, 1000)]:
            p = hb.letterbox_params(W, H)
            for (w, h) in [(4, 6), (6, 8), (8, 12), (4, 12), (12, 4)]:
                box = [W / 2 - w / 2.0, H / 2 - h / 2.0, W / 2 + w / 2.0, H / 2 + h / 2.0]
                inv = hb.scale_boxes_inverse(hb.letterbox_forward_box(box, p), p, W, H)
                h_out = inv[3] - inv[1]
                w_out = inv[2] - inv[0]
                self.assertLessEqual(abs(h_out - h), TOL, "%dx%d box %dx%d" % (W, H, w, h))
                self.assertLessEqual(abs(w_out - w), TOL, "%dx%d box %dx%d" % (W, H, w, h))
                # the realistic content path may differ only by the round(new_unpad) residual
                inv_c = hb.scale_boxes_inverse(hb.letterbox_forward_box(box, p, content_scale=True), p, W, H)
                ratio = (inv_c[3] - inv_c[1]) / h
                self.assertLess(abs(ratio - 1.0), 2e-3, "%dx%d box %dx%d ratio %.5f" % (W, H, w, h, ratio))
                self.assertGreater(abs(ratio - 1.092), 0.05, "content path must not carry the +9.2% bias")


class Test5_PerfectPrediction(unittest.TestCase):
    def test_identity(self):
        gt = [100.0, 100.0, 110.0, 110.0]
        pred = list(gt)
        self.assertAlmostEqual(ev.iou_xyxy(gt, pred), 1.0, places=12)
        self.assertAlmostEqual(ev.center_dist(gt, pred), 0.0, places=12)
        gw, gh = gt[2] - gt[0], gt[3] - gt[1]
        pw, ph = pred[2] - pred[0], pred[3] - pred[1]
        self.assertAlmostEqual(pw / gw, 1.0, places=12)
        self.assertAlmostEqual(ph / gh, 1.0, places=12)


class Test6_KnownHeightRatio(unittest.TestCase):
    """GT 10x10, pred 10x11 with the same centre: h_ratio = 1.1 and IoU = 100/110 by hand."""

    def test_hand_computed_iou(self):
        gt = [100.0, 100.0, 110.0, 110.0]
        pred = [100.0, 99.5, 110.0, 110.5]
        self.assertAlmostEqual((gt[0] + gt[2]) / 2, (pred[0] + pred[2]) / 2, places=12)
        self.assertAlmostEqual((gt[1] + gt[3]) / 2, (pred[1] + pred[3]) / 2, places=12)
        h_ratio = (pred[3] - pred[1]) / (gt[3] - gt[1])
        w_ratio = (pred[2] - pred[0]) / (gt[2] - gt[0])
        self.assertAlmostEqual(h_ratio, 1.1, places=12)
        self.assertAlmostEqual(w_ratio, 1.0, places=12)
        inter = 10.0 * 10.0
        union = 100.0 + 110.0 - inter
        self.assertAlmostEqual(ev.iou_xyxy(gt, pred), inter / union, places=12)
        self.assertAlmostEqual(ev.iou_xyxy(gt, pred), 100.0 / 110.0, places=12)


class Test7_UltralyticsParity(unittest.TestCase):
    """Native ultralytics boxes vs the cached pipeline boxes.

    Exact bitwise equality is NOT expected: the two call paths batch differently, so the integer padding
    of the letterbox can differ by a fraction of a pixel. The claim under test is sub-pixel agreement on
    the boxes both runs agree on, plus the exactness of the coordinate FORMULA (Test7b, delta 0.0).
    """

    def test_parity_artifact(self):
        p = ROOT / "outputs" / "shuttle_capability" / "metrics" / "bbox_ultralytics_parity.csv"
        if not p.exists():
            self.skipTest("run tools/audit_bbox_height_bias.py --parity to produce %s" % p.name)
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        matched = [r for r in rows if not r.get("note")]
        extra = [r for r in rows if r.get("note")]
        self.assertGreater(len(matched), 5)
        worst = 0.0
        for r in matched:
            d = max(abs(float(r[k])) for k in ("dx1", "dy1", "dx2", "dy2"))
            worst = max(worst, d)
        self.assertLessEqual(worst, 1.0, "matched boxes must agree to sub-pixel accuracy")
        self.assertLessEqual(len(extra), max(2, len(matched) // 5), "too many unmatched boxes")

    def test_ultralytics_scale_boxes_matches(self):
        cc = hb.ultralytics_cross_check()
        if not cc.get("available"):
            self.skipTest("ultralytics not importable: %s" % cc.get("error"))
        self.assertEqual(cc["max_abs_diff_inverse"], 0.0, "my scale-back must equal utils.ops.scale_boxes")
        self.assertEqual(cc["max_abs_diff_letterbox_params"], 0.0, "letterbox params must match get_params")
        self.assertEqual(cc["max_abs_diff_padded_shape"], 0.0, "padded output must be exactly imgsz x imgsz")
class Test8_RunRoundtripSuite(unittest.TestCase):
    def test_suite_passes(self):
        rows = hb.run_roundtrip_tests()
        self.assertGreater(len(rows), 30)
        for r in rows:
            self.assertLessEqual(r["max_abs_err_norm_roundtrip"], TOL, r["case"])
            self.assertLessEqual(r["max_abs_err_letterbox_roundtrip"], TOL, r["case"])
            self.assertLessEqual(r["max_abs_err_content_roundtrip"], 1.0, r["case"])
            self.assertLess(abs(r["h_ratio_content"] - 1.0), 2e-3, r["case"])


if __name__ == "__main__":
    unittest.main()
