#!/usr/bin/env python3
"""Tests for the frozen-test path of tools/eval_yolo26_v1.py.

Pure-function tests: no model, no GPU, no real dataset, no live run. Everything builds its own
temporary tree / synthetic boxes. Run:  python tests/test_eval_yolo26_v1_frozen.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import eval_hard_negative as ehn  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402


def write_image(path: Path, w: int = 100, h: int = 100) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (30, 30, 30)).save(path)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class Test1_DiscoverGroups(unittest.TestCase):
    """labeled set (images/+labels/), multi-image-dir set, flat set, missing set."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "outputs" / "shuttle_capability"
        self.addCleanup(self.tmp.cleanup)
        # labeled: images/ + labels/
        for i in range(2):
            write_image(self.root / "controlled_capability" / "images" / ("a%d.png" % i))
            touch(self.root / "controlled_capability" / "labels" / ("a%d.txt" % i))
        # unlabeled, two image dirs
        write_image(self.root / "real_images" / "backgrounds" / "bg_001.jpg")
        write_image(self.root / "real_images" / "raw" / "real_001.png")

    def test_labeled_group_is_detected(self):
        gs = {g.key: g for g in ev.discover_frozen_groups(self.root)}
        self.assertIn("controlled_capability/images", gs)
        g = gs["controlled_capability/images"]
        self.assertTrue(g.labeled)
        self.assertEqual(len(g.images), 2)
        self.assertEqual(g.label_dir.name, "labels")

    def test_set_with_two_image_dirs_yields_two_unlabeled_groups(self):
        gs = {g.key: g for g in ev.discover_frozen_groups(self.root)}
        self.assertIn("real_images/backgrounds", gs)
        self.assertIn("real_images/raw", gs)
        self.assertFalse(gs["real_images/backgrounds"].labeled)
        self.assertFalse(gs["real_images/raw"].labeled)

    def test_flat_set_uses_set_name_as_key(self):
        write_image(self.root / "synthetic_3d" / "syn_0001.png")
        gs = {g.key: g for g in ev.discover_frozen_groups(self.root)}
        self.assertIn("synthetic_3d", gs)
        self.assertEqual(gs["synthetic_3d"].group, "")
        self.assertFalse(gs["synthetic_3d"].labeled)

    def test_missing_set_is_skipped_and_only_filters(self):
        keys = [g.key for g in ev.discover_frozen_groups(self.root)]
        self.assertNotIn("challenge_test/images", keys)
        only = [g.key for g in ev.discover_frozen_groups(self.root, only=["real_images"])]
        self.assertEqual(sorted(only), ["real_images/backgrounds", "real_images/raw"])

    def test_non_image_files_are_ignored(self):
        (self.root / "synthetic_3d").mkdir(parents=True, exist_ok=True)
        (self.root / "synthetic_3d" / "notes.txt").write_text("x", encoding="utf-8")
        self.assertNotIn("synthetic_3d", [g.key for g in ev.discover_frozen_groups(self.root)])


class Test2_LabelDirResolution(unittest.TestCase):
    """labels/ beside images, labels/ with only foreign stems, labels beside (same dir)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_label_dir_with_only_foreign_stems_is_unlabeled(self):
        write_image(self.root / "challenge_test" / "images" / "c1.jpg")
        touch(self.root / "challenge_test" / "labels" / "other.txt")   # stem matches no image
        gs = {g.key: g for g in ev.discover_frozen_groups(self.root)}
        self.assertIn("challenge_test/images", gs)
        self.assertFalse(gs["challenge_test/images"].labeled)
        self.assertIsNone(gs["challenge_test/images"].label_dir)

    def test_labels_beside_images(self):
        d = self.root / "controlled_capability" / "images"
        write_image(d / "s1.jpg")
        touch(d / "s1.txt")
        gs = {g.key: g for g in ev.discover_frozen_groups(self.root)}
        self.assertTrue(gs["controlled_capability/images"].labeled)
        self.assertEqual(gs["controlled_capability/images"].label_dir, d)


class Test3_LoadGtBoxes(unittest.TestCase):
    """YOLO label parsing -> boxes + equiv_size_640 bucket; robust to junk and missing files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.img = self.d / "img.png"
        write_image(self.img, 100, 100)

    def test_single_box_geometry_and_bucket(self):
        lab = self.d / "img.txt"
        lab.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        gts = ev.load_gt_boxes(self.img, lab)
        self.assertEqual(len(gts), 1)
        self.assertEqual([round(v, 3) for v in gts[0]["box"]], [40.0, 40.0, 60.0, 60.0])
        self.assertAlmostEqual(gts[0]["eq640"], 128.0, places=6)   # 20 px at 6.4x letterbox scale
        self.assertEqual(gts[0]["bucket"], ">64")

    def test_missing_label_is_a_hard_negative(self):
        self.assertEqual(ev.load_gt_boxes(self.img, self.d / "nope.txt"), [])

    def test_unreadable_image_returns_none(self):
        lab = self.d / "img.txt"
        lab.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
        self.assertIsNone(ev.load_gt_boxes(self.d / "missing.png", lab))

    def test_junk_lines_are_ignored(self):
        lab = self.d / "junk.txt"
        lab.write_text(chr(10).join(["", "0 1 2", "0 a b c d", "0 0.5 0.5 0.1 0.1", "garbage"]), encoding="utf-8")
        gts = ev.load_gt_boxes(self.img, lab)
        self.assertEqual(len(gts), 1)
        self.assertAlmostEqual(gts[0]["eq640"], 64.0, places=6)     # 10 px * 6.4


class Test4_FpStats(unittest.TestCase):
    """FP/image and image FP rate for all-negative sets, at several thresholds."""

    def setUp(self):
        self.dets = [[{"conf": 0.9}], [], [{"conf": 0.3}, {"conf": 0.1}]]
        self.stats = ev.fp_stats(self.dets, [0.25, 0.5, 0.75])

    def test_counts_and_rates(self):
        s = self.stats["0.25"]
        self.assertEqual(s["total_images"], 3)
        self.assertEqual(s["total_FP"], 2)
        self.assertAlmostEqual(s["FP_per_image"], 2 / 3, places=9)
        self.assertEqual(s["images_with_FP"], 2)
        self.assertAlmostEqual(s["image_FP_rate"], 2 / 3, places=9)
        self.assertAlmostEqual(s["max_confidence_FP"], 0.9, places=9)

    def test_higher_threshold_removes_low_confidence_fp(self):
        self.assertEqual(self.stats["0.5"]["total_FP"], 1)
        self.assertEqual(self.stats["0.75"]["total_FP"], 1)
        self.assertAlmostEqual(self.stats["0.5"]["image_FP_rate"], 1 / 3, places=9)

    def test_threshold_is_inclusive(self):
        s = ev.fp_stats([[{"conf": 0.25}]], [0.25])
        self.assertEqual(s["0.25"]["total_FP"], 1)

    def test_empty_set_has_no_rates(self):
        s = ev.fp_stats([], [0.25])["0.25"]
        self.assertIsNone(s["FP_per_image"])
        self.assertIsNone(s["image_FP_rate"])
        self.assertIsNone(s["max_confidence_FP"])

    def test_confidence_distribution_fields(self):
        s = ev.fp_stats([[{"conf": 0.1}, {"conf": 0.2}, {"conf": 0.9}]], [0.05])["0.05"]
        self.assertAlmostEqual(s["mean_confidence_FP"], 0.4, places=9)
        self.assertAlmostEqual(s["median_confidence_FP"], 0.2, places=9)
        self.assertAlmostEqual(s["P95_FP_confidence"], 0.83, places=6)


class Test5_PercentileConvention(unittest.TestCase):
    """Keep the same percentile convention as tools/eval_hard_negative.py."""

    def test_matches_hard_negative_tool(self):
        for xs in ([0.1, 0.2, 0.9], [1.0], [0.0, 1.0, 2.0, 3.0], [0.5, 0.5]):
            for p in (0.5, 0.95):
                self.assertAlmostEqual(ev.percentile(xs, p), ehn.percentile(xs, p), places=12)

    def test_empty_is_none(self):
        self.assertIsNone(ev.percentile([], 0.95))


class Test6_ResolveCheckpoints(unittest.TestCase):
    """Unified interface: registry names + ad-hoc NAME=PATH + placeholders that must not crash."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.reg = self.d / "checkpoints.yaml"
        self.reg.write_text(chr(10).join([
            "checkpoints:",
            "  a_best:",
            "    path: /ckpt/a.pt",
            "    run: runA",
            "    epoch: 6",
            "    role: baseline",
            "  c_best:",
            "    path: \"\"",
            "    epoch: null",
            "    role: future",
        ]), encoding="utf-8")

    def test_registry_order_and_metadata(self):
        got = ev.resolve_checkpoints([], self.reg, ["a_best"])
        self.assertEqual([g["name"] for g in got], ["a_best"])
        self.assertEqual(got[0]["path"], "/ckpt/a.pt")
        self.assertEqual(got[0]["epoch"], 6)
        self.assertEqual(got[0]["role"], "baseline")

    def test_placeholder_is_skipped_not_fatal(self):
        got = ev.resolve_checkpoints([], self.reg, ["c_best", "a_best"])
        self.assertEqual([g["name"] for g in got], ["a_best"])

    def test_adhoc_pair_wins_and_dedupes(self):
        got = ev.resolve_checkpoints(["a_best=/other.pt"], self.reg, ["a_best"])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["path"], "/other.pt")

    def test_unknown_name_raises(self):
        with self.assertRaises(SystemExit):
            ev.resolve_checkpoints([], self.reg, ["z_best"])

    def test_bad_pair_raises(self):
        with self.assertRaises(SystemExit):
            ev.resolve_checkpoints(["no-equals-sign"], None, [])

    def test_missing_registry_yields_empty_dict(self):
        self.assertEqual(ev.load_checkpoint_registry(self.d / "nope.yaml"), {})

    def test_json_registry_is_supported(self):
        p = self.d / "reg.json"
        p.write_text(json.dumps({"checkpoints": {"b_best": {"path": "/ckpt/b.pt", "epoch": 10}}}), encoding="utf-8")
        got = ev.resolve_checkpoints([], p, ["b_best"])
        self.assertEqual(got[0]["path"], "/ckpt/b.pt")
        self.assertEqual(got[0]["epoch"], 10)


class Test7_RepoRegistryResolves(unittest.TestCase):
    """The shipped registry must resolve a_best/b_best/b_last and tolerate the c_best placeholder."""

    def test_shipped_registry(self):
        reg = ROOT / "configs" / "shuttle_detection" / "checkpoints_v1.yaml"
        self.assertTrue(reg.exists())
        got = ev.resolve_checkpoints([], reg, ["a_best", "b_best", "b_last", "c_best"])
        self.assertEqual([g["name"] for g in got], ["a_best", "b_best", "b_last"])
        self.assertEqual(got[1]["epoch"], 10)
        self.assertEqual(got[2]["role"], "overfit_control")


if __name__ == "__main__":
    unittest.main(verbosity=2)
