#!/usr/bin/env python3
"""Tests for the GLB-driven shuttlecock renderer (tools/shuttle_render.py).

Why this module exists: the archived shuttle GLB (Obj_Feather + Obj_Cork) carries
authored vertex NORMALs but NO TEXCOORD_0 and NO texture for the shuttlecock parts
(material 'White' has no baseColorTexture), so appearance cannot come from the asset.
What CAN come from the asset is geometry + normals; everything else must be modelled.

Run:
    python tests/tools/test_shuttle_render.py -v
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from shuttle_render import (  # noqa: E402
    AMBIENT,
    DEFAULT_GLB,
    FEATHER_ALBEDO,
    KEY,
    SUPPORT_COVERAGE,
    calibrate_distance,
    composite,
    find_blank_backgrounds,
    find_leaked_backgrounds,
    imread_unicode,
    imwrite_unicode,
    load_shuttle_parts,
    load_shuttle_spec,
    prepare_background,
    render_sample,
    mask_from_coverage,
    render_shuttle,
    shade,
    split_backgrounds,
    yolo_bbox_from_mask,
)

# Camera used throughout: pinhole, +z forward, y down (matches the experiment scripts).
K = (700.0, 700.0, 480.0, 480.0)
WH = 960
SHUTTLE_EXTENT_M = 0.0778  # measured from the GLB (combined bbox, metres)


def light_from_above_and_behind() -> np.ndarray:
    v = np.array([0.40, -0.50, -0.75], dtype=np.float64)
    return v / np.linalg.norm(v)


class ShuttleSpecTests(unittest.TestCase):
    """The renderer must take the shuttlecock's size from the project model, not from a
    constant of its own: this repository logs duplicated constants as a defect (ISSUE-013),
    and the visual asset is an EXTERNAL_REFERENCE that nothing else cross-checks."""

    def test_reference_geometry_comes_from_the_project_config(self) -> None:
        spec = load_shuttle_spec()
        self.assertEqual(spec.feathers_count, 16)
        self.assertAlmostEqual(spec.feather_length_m, 0.066, places=9)
        self.assertAlmostEqual(spec.cork_diameter_m, 0.0265, places=9)
        self.assertAlmostEqual(spec.total_length_m, 0.01325 + 0.066, places=9)

    def test_configured_skirt_diameter_is_inside_the_bwf_range(self) -> None:
        spec = load_shuttle_spec()
        self.assertGreaterEqual(spec.skirt_tip_diameter_m, 0.058)
        self.assertLessEqual(spec.skirt_tip_diameter_m, 0.068)

    def test_imported_visual_is_dimensionally_consistent_with_the_model(self) -> None:
        """A swapped visual would otherwise silently mis-size every sample."""
        spec = load_shuttle_spec()
        feather, cork = load_shuttle_parts()
        allv = np.vstack([feather.verts, cork.verts])
        length = float(allv[:, 2].max() - allv[:, 2].min())
        skirt = feather.verts[:, :2].max(axis=0) - feather.verts[:, :2].min(axis=0)
        skirt_dia = float(np.mean(skirt))
        self.assertLessEqual(abs(skirt_dia - spec.skirt_tip_diameter_m) / spec.skirt_tip_diameter_m,
                             0.05, f"visual skirt {skirt_dia:.4f} m vs model "
                                   f"{spec.skirt_tip_diameter_m:.4f} m")
        self.assertLessEqual(abs(length - spec.total_length_m) / spec.total_length_m, 0.05,
                             f"visual length {length:.4f} m vs model "
                             f"{spec.total_length_m:.4f} m")


class LoadPartsTests(unittest.TestCase):
    """The renderer must consume the real asset, not a stand-in."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parts = load_shuttle_parts(DEFAULT_GLB)

    def test_default_glb_is_the_archived_shuttlecock_asset(self) -> None:
        self.assertTrue(DEFAULT_GLB.is_file(), str(DEFAULT_GLB))
        self.assertEqual(DEFAULT_GLB.name, "badminton_racket_and_shuttlecock_low_poly.glb")

    def test_parts_are_feather_and_cork_with_measured_triangle_counts(self) -> None:
        self.assertEqual([p.name for p in self.parts], ["Obj_Feather", "Obj_Cork"])
        self.assertEqual(sum(len(p.verts) for p in self.parts), 6498)
        self.assertEqual(sum(len(p.faces) for p in self.parts), 3422)

    def test_parts_carry_the_authored_unit_normals(self) -> None:
        for part in self.parts:
            self.assertEqual(part.normals.shape, part.verts.shape)
            np.testing.assert_allclose(
                np.linalg.norm(part.normals, axis=1), 1.0, atol=1e-4,
                err_msg=f"{part.name} normals are not unit length",
            )

    def test_face_indices_stay_inside_the_vertex_array(self) -> None:
        for part in self.parts:
            self.assertGreaterEqual(int(part.faces.min()), 0)
            self.assertLess(int(part.faces.max()), len(part.verts))

    def test_feather_is_brighter_than_cork(self) -> None:
        feather, cork = self.parts
        self.assertGreater(float(feather.albedo.mean()), float(cork.albedo.mean()))


class ShadeTests(unittest.TestCase):
    """A white shuttlecock must never render black - without a hand-tuned fudge."""

    def test_white_shuttle_never_shades_below_the_ambient_floor(self) -> None:
        rng = np.random.default_rng(0)
        n = rng.normal(size=(3000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        rgb = shade(n, light_from_above_and_behind(), albedo=np.array([0.93, 0.94, 0.90]),
                    ambient=0.34, key=0.66)
        lum = rgb.mean(axis=1)
        self.assertGreaterEqual(float(lum.min()), 0.34 * 0.92 * 255.0 * 0.98)
        self.assertLessEqual(float(lum.max()), 255.0)

    def test_turning_the_normal_toward_the_light_brightens_it(self) -> None:
        light = np.array([0.0, 0.0, -1.0])
        albedo = np.array([0.9, 0.9, 0.9])
        lit = shade(np.array([[0.0, 0.0, -1.0]]), light, albedo, ambient=0.3, key=0.7)[0]
        unlit = shade(np.array([[0.0, 0.0, 1.0]]), light, albedo, ambient=0.3, key=0.7)[0]
        self.assertGreater(float(lit.mean()), float(unlit.mean()))

    def test_feathers_transmit_backlight_so_a_backlit_face_is_not_dark(self) -> None:
        light = np.array([0.0, 0.0, -1.0])
        albedo = np.array([0.93, 0.94, 0.90])
        away = np.array([[0.0, 0.0, 1.0]])
        opaque = shade(away, light, albedo, ambient=0.34, key=0.66, translucency=0.0)[0]
        feather = shade(away, light, albedo, ambient=0.34, key=0.66, translucency=0.30)[0]
        self.assertGreater(float(feather.mean()), float(opaque.mean()))


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parts = load_shuttle_parts(DEFAULT_GLB)
        cls.light = light_from_above_and_behind()

    def _render(self, target_px: float, ss: int = 3):
        dist = calibrate_distance(self.parts, np.eye(3), target_px,
                                  (WH / 2.0, WH / 2.0), K, WH, WH)
        return render_shuttle(
            self.parts, np.eye(3), dist, (WH / 2.0, WH / 2.0), K, WH, WH,
            light_dir=self.light, supersample=ss,
        )

    def _gt_bbox(self, alpha):
        """Ground truth uses the object footprint, the same rule as the generator."""
        return yolo_bbox_from_mask(mask_from_coverage(alpha, SUPPORT_COVERAGE), WH, WH)

    def test_render_returns_colour_and_coverage_of_the_requested_size(self) -> None:
        rgb, alpha = self._render(24.0)
        self.assertEqual(rgb.shape, (WH, WH, 3))
        self.assertEqual(alpha.shape, (WH, WH))
        self.assertGreater(float(alpha.max()), 0.98)

    def test_edges_are_anti_aliased_rather_than_a_hard_silhouette(self) -> None:
        _rgb, alpha = self._render(24.0)
        partial = ((alpha > 0.05) & (alpha < 0.95)).sum()
        self.assertGreater(int(partial), 20, "expected soft edge pixels from supersampling")

    def test_rendered_object_is_bright_not_a_flat_single_colour(self) -> None:
        rgb, alpha = self._render(24.0)
        inside = alpha > 0.9
        self.assertGreater(int(inside.sum()), 50)
        values = rgb[inside]
        self.assertGreater(float(values.mean()), 120.0)
        self.assertGreater(float(values.std()), 5.0, "shading should vary across the object")

    def test_shading_uses_the_authored_normals_not_a_constant_colour(self) -> None:
        rgb_smooth, alpha = self._render(28.0, ss=3)
        flat = np.broadcast_to(np.array([245.0, 245.0, 235.0]), rgb_smooth.shape)
        inside = alpha > 0.9
        self.assertGreater(float(np.abs(rgb_smooth[inside] - flat[inside]).mean()), 5.0)

    def test_a_two_pixel_target_is_a_fractional_smear_not_a_solid_blob(self) -> None:
        """Measured fact: at 2 px the shuttle touches pixels without filling any of them.

        This is why ground truth uses the object footprint rather than a half-coverage
        rule - the latter would erase the target exactly where recall is already zero.
        """
        _rgb, alpha = self._render(2.0)
        self.assertIsNotNone(self._gt_bbox(alpha), "a 2 px target must stay labelable")
        self.assertGreater(int((alpha > 0).sum()), 0)
        self.assertLess(float(alpha.max()), 0.8, "2 px should not resolve to a solid pixel")

    def test_calibrated_distance_reaches_the_requested_pixel_size(self) -> None:
        """Tight on purpose: the naive formula lands ~0.83x, so a loose delta cannot tell
        calibration from no calibration at all (mutation testing proved exactly that)."""
        for target in (4.0, 8.0, 16.0):
            _rgb, alpha = self._render(target)
            bbox = self._gt_bbox(alpha)
            self.assertIsNotNone(bbox, f"no ground truth at target {target} px")
            achieved = math.sqrt(bbox[2] * WH * bbox[3] * WH)
            self.assertLessEqual(abs(achieved / target - 1.0), 0.12,
                                 f"target {target} px produced {achieved:.2f} px "
                                 f"(ratio {achieved / target:.3f})")

    def test_calibration_never_loses_to_the_naive_formula(self) -> None:
        """At 8 px the naive formula gave 6.48 px while calibration gave 10.0 px, because
        calibration measured at one supersample and the shipped image used another."""
        for target in (4.0, 8.0, 16.0, 24.0):
            _rgb, alpha = self._render(target)
            bbox = self._gt_bbox(alpha)
            achieved = math.sqrt(bbox[2] * WH * bbox[3] * WH)
            naive = K[0] * SHUTTLE_EXTENT_M / target
            _rgb2, alpha2 = render_shuttle(self.parts, np.eye(3), naive,
                                           (WH / 2.0, WH / 2.0), K, WH, WH,
                                           light_dir=self.light, supersample=3)
            boxt = self._gt_bbox(alpha2)
            achieved_naive = math.sqrt(boxt[2] * WH * boxt[3] * WH)
            self.assertLessEqual(abs(achieved - target), abs(achieved_naive - target),
                                 f"target {target}: calibrated {achieved:.2f} px is no better "
                                 f"than naive {achieved_naive:.2f} px")

    def test_calibrated_distance_beats_naive_extent_over_target(self) -> None:
        target = 12.0
        naive = K[0] * SHUTTLE_EXTENT_M / target
        calibrated = calibrate_distance(self.parts, np.eye(3), target,
                                        (WH / 2.0, WH / 2.0), K, WH, WH)
        _rgb, alpha = render_shuttle(self.parts, np.eye(3), calibrated,
                                     (WH / 2.0, WH / 2.0), K, WH, WH,
                                     light_dir=self.light, supersample=3)
        bbox = self._gt_bbox(alpha)
        achieved = math.sqrt(bbox[2] * WH * bbox[3] * WH)
        self.assertLess(abs(calibrated - naive), naive,
                        "calibration should move the distance, not ignore it")
        self.assertAlmostEqual(achieved / target, 1.0, delta=0.25)


class MaskAndGroundTruthTests(unittest.TestCase):
    def test_mask_thresholds_coverage_at_one_half(self) -> None:
        alpha = np.array([[0.0, 0.49], [0.5, 1.0]])
        np.testing.assert_array_equal(mask_from_coverage(alpha), [[False, False], [True, True]])

    def test_bbox_is_exactly_tight_around_the_visible_pixels(self) -> None:
        mask = np.zeros((20, 20), bool)
        mask[5:9, 3:11] = True
        bbox = yolo_bbox_from_mask(mask, 20, 20)
        cx, cy, w, h = bbox
        self.assertAlmostEqual(cx, 7.0 / 20.0, places=9)
        self.assertAlmostEqual(cy, 7.0 / 20.0, places=9)
        self.assertAlmostEqual(w, 8.0 / 20.0, places=9)
        self.assertAlmostEqual(h, 4.0 / 20.0, places=9)

    def test_bbox_round_trips_back_to_the_exact_pixel_box(self) -> None:
        mask = np.zeros((32, 32), bool)
        mask[11:19, 4:7] = True
        cx, cy, w, h = yolo_bbox_from_mask(mask, 32, 32)
        x0 = round((cx - w / 2.0) * 32)
        x1 = round((cx + w / 2.0) * 32)
        y0 = round((cy - h / 2.0) * 32)
        y1 = round((cy + h / 2.0) * 32)
        self.assertEqual((x0, x1, y0, y1), (4, 7, 11, 19))

    def test_empty_mask_has_no_bbox(self) -> None:
        self.assertIsNone(yolo_bbox_from_mask(np.zeros((4, 4), bool), 4, 4))


class SplitBackgroundTests(unittest.TestCase):
    def test_train_and_val_backgrounds_never_share_an_image(self) -> None:
        bgs = [Path(f"bg_{i:03d}.png") for i in range(40)]
        train, val = split_backgrounds(bgs, val_count=7, seed=3)
        self.assertEqual(len(val), 7)
        self.assertEqual(len(train), 33)
        self.assertEqual(set(train) & set(val), set())

    def test_split_is_deterministic_for_a_seed_and_covers_every_input(self) -> None:
        bgs = [Path(f"bg_{i:03d}.png") for i in range(40)]
        a = split_backgrounds(bgs, val_count=7, seed=3)
        b = split_backgrounds(bgs, val_count=7, seed=3)
        self.assertEqual(a, b)
        self.assertEqual(set(a[0]) | set(a[1]), set(bgs))

    def test_split_refuses_a_val_count_larger_than_the_pool(self) -> None:
        with self.assertRaises(ValueError):
            split_backgrounds([Path("only.png")], val_count=2, seed=0)

class LeakageTests(unittest.TestCase):
    """A training background that is also a frozen-test background silently inflates every
    capability number measured on that test set - so the check must be automatic."""

    def _write(self, path: Path, seed: int) -> None:
        img = np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        imwrite_unicode(path, img, quality=95)

    def test_finds_a_background_shared_with_the_frozen_pool(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "train").mkdir()
            (root / "frozen").mkdir()
            self._write(root / "train" / "tbg_011.png", 5)
            self._write(root / "frozen" / "bg_021.jpg", 5)   # same picture, new name/format
            self._write(root / "frozen" / "bg_022.jpg", 6)
            leaks = find_leaked_backgrounds(sorted((root / "train").glob("*")),
                                            sorted((root / "frozen").glob("*")))
            self.assertEqual([c.name for c, _f, _r in leaks], ["tbg_011.png"])
            self.assertEqual(leaks[0][1].name, "bg_021.jpg")
            self.assertGreater(leaks[0][2], 0.95)

    def test_keeps_visually_unrelated_backgrounds(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "train").mkdir()
            (root / "frozen").mkdir()
            self._write(root / "train" / "tbg_001.png", 11)
            self._write(root / "frozen" / "bg_001.jpg", 12)
            self.assertEqual(find_leaked_backgrounds(sorted((root / "train").glob("*")),
                                                     sorted((root / "frozen").glob("*"))), [])

    def test_empty_frozen_pool_reports_no_leak(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "train").mkdir()
            self._write(root / "train" / "tbg_001.png", 3)
            self.assertEqual(find_leaked_backgrounds(sorted((root / "train").glob("*")), []), [])


class BlankBackgroundTests(unittest.TestCase):
    def _write(self, path: Path, image: np.ndarray) -> None:
        imwrite_unicode(path, image, quality=95)

    def test_flags_a_fully_black_background(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write(root / "black.png", np.zeros((32, 32, 3), np.uint8))
            self._write(root / "scene.jpg",
                        np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8))
            flagged = find_blank_backgrounds(sorted(root.glob("*")))
            self.assertEqual([p.name for p in flagged], ["black.png"])

    def test_flags_a_flat_single_colour_frame(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write(root / "flat.png", np.full((32, 32, 3), 200, np.uint8))
            self.assertEqual([p.name for p in find_blank_backgrounds(sorted(root.glob("*")))],
                             ["flat.png"])

    def test_keeps_a_real_scene(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write(root / "scene.jpg",
                        np.random.default_rng(2).integers(0, 255, (48, 48, 3), dtype=np.uint8))
            self.assertEqual(find_blank_backgrounds(sorted(root.glob("*"))), [])


class CompositeAndSampleTests(unittest.TestCase):
    """Compositing must record exactly what it applied - a manifest that says None while
    noise was added makes every later claim about the data unverifiable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parts = load_shuttle_parts(DEFAULT_GLB)
        cls.bg = np.full((WH, WH, 3), 90, np.uint8)

    def test_contact_shadow_never_darkens_the_shuttle_itself(self) -> None:
        """A shadow is cast onto the background; it cannot fall on the object casting it.
        Measured before the fix: shadow_gain=0.319 darkened the object itself by 26.1%."""
        rgb, alpha = render_shuttle(self.parts, np.eye(3), 3.0, (WH / 2.0, WH / 2.0), K,
                                    WH, WH, light_dir=light_from_above_and_behind())
        plain = composite(self.bg, rgb, alpha)
        shadowed = composite(self.bg, rgb, alpha, shadow_gain=0.5)
        solid = alpha > 0.5
        self.assertGreater(int(solid.sum()), 10)
        np.testing.assert_array_equal(shadowed[solid], plain[solid])
        self.assertLess(float(shadowed[~solid].mean()), float(plain[~solid].mean()),
                        "a shadow must still be visible on the background")

    def test_the_darkest_white_feather_face_is_not_darker_than_mid_grey(self) -> None:
        """A white feather shuttle is the bright object on court; a fully back-facing face
        must not render darker than mid-grey (128), which is what ambient=0.34 did."""
        away = np.array([[0.0, 0.0, 1.0]])
        light = np.array([0.0, 0.0, -1.0])
        darkest = shade(away, light, FEATHER_ALBEDO, ambient=AMBIENT, key=KEY)[0]
        self.assertGreaterEqual(float(darkest.mean()), 128.0,
                                f"darkest white face renders at {float(darkest.mean()):.1f}")

    def test_composite_is_deterministic_for_explicit_parameters(self) -> None:
        rgb, alpha = render_shuttle(self.parts, np.eye(3), 3.0, (WH / 2.0, WH / 2.0), K,
                                    WH, WH, light_dir=light_from_above_and_behind())
        a = composite(self.bg, rgb, alpha, shadow_gain=0.2, motion_px=2.0,
                      motion_angle_deg=30.0, noise_sigma=3.0, noise_seed=7)
        b = composite(self.bg, rgb, alpha, shadow_gain=0.2, motion_px=2.0,
                      motion_angle_deg=30.0, noise_sigma=3.0, noise_seed=7)
        np.testing.assert_array_equal(a, b)

    def test_composite_returns_a_uint8_image_of_the_background_shape(self) -> None:
        rgb, alpha = render_shuttle(self.parts, np.eye(3), 3.0, (WH / 2.0, WH / 2.0), K,
                                    WH, WH, light_dir=light_from_above_and_behind())
        out = composite(self.bg, rgb, alpha, noise_sigma=2.0, noise_seed=1)
        self.assertEqual(out.shape, self.bg.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_composite_noise_actually_changes_pixels(self) -> None:
        rgb, alpha = render_shuttle(self.parts, np.eye(3), 3.0, (WH / 2.0, WH / 2.0), K,
                                    WH, WH, light_dir=light_from_above_and_behind())
        clean = composite(self.bg, rgb, alpha)
        noisy = composite(self.bg, rgb, alpha, noise_sigma=4.0, noise_seed=2)
        self.assertGreater(int(np.abs(noisy.astype(int) - clean.astype(int)).max()), 0)

    def test_sample_record_reproduces_the_exact_image(self) -> None:
        sample = render_sample(self.parts, self.bg, target_px=10.0, seed=11,
                               split="train", name="train_00000")
        again = composite(sample.background, sample.rgb, sample.alpha,
                          shadow_gain=sample.record["shadow_gain"],
                          motion_px=sample.record["motion_px"],
                          motion_angle_deg=sample.record["motion_angle_deg"],
                          noise_sigma=sample.record["noise_sigma"],
                          noise_seed=sample.record["noise_seed"])
        np.testing.assert_array_equal(again, sample.image)

    def test_sample_record_logs_the_noise_it_really_applied(self) -> None:
        sample = render_sample(self.parts, self.bg, target_px=10.0, seed=12,
                               split="train", name="train_00001")
        self.assertIsInstance(sample.record["noise_sigma"], float)
        self.assertGreater(sample.record["noise_sigma"], 0.0)
        self.assertIn("noise_seed", sample.record)

    def test_sample_record_stores_replay_parameters_exactly(self) -> None:
        """target_px drives the calibration loop, so it is a replay input like noise_sigma.
        Recorded rounded to 3 decimals it changed a pixel at train_00399 (mean|d| 1.03e-4)."""
        target = 10.6975706472
        sample = render_sample(self.parts, self.bg, target_px=target, seed=17,
                               split="train", name="n")
        self.assertEqual(sample.record["target_px"], target)

    def test_replaying_a_sample_from_its_record_is_bit_exact(self) -> None:
        target = 18.4597529756   # a target that demonstrably flipped on rounding
        first = render_sample(self.parts, self.bg, target_px=target, seed=18,
                              split="train", name="n")
        again = render_sample(self.parts, self.bg, target_px=first.record["target_px"],
                              seed=18, split="train", name="n")
        np.testing.assert_array_equal(again.alpha, first.alpha)
        np.testing.assert_array_equal(again.image, first.image)

    def test_sample_is_deterministic_for_a_seed(self) -> None:
        a = render_sample(self.parts, self.bg, target_px=10.0, seed=13,
                          split="train", name="n")
        b = render_sample(self.parts, self.bg, target_px=10.0, seed=13,
                          split="train", name="n")
        np.testing.assert_array_equal(a.image, b.image)
        self.assertEqual(a.record, b.record)

    def test_sample_label_matches_the_recorded_box(self) -> None:
        sample = render_sample(self.parts, self.bg, target_px=10.0, seed=14,
                               split="train", name="n")
        cls_id, cx, cy, bw, bh = sample.label.split()
        self.assertEqual(cls_id, "0")
        self.assertAlmostEqual(float(bw) * WH,
                               sample.record["bbox_w_px"], delta=0.5)
        self.assertAlmostEqual(float(bh) * WH,
                               sample.record["bbox_h_px"], delta=0.5)

    def _object_mean_luma(self, sample) -> float:
        """Mean over pixels the shuttle actually fills, so background bleed cannot fake it."""
        solid = np.asarray(sample.alpha) > 0.5
        self.assertGreater(int(solid.sum()), 0, "no filled pixels to measure")
        return float(np.asarray(sample.image)[solid].mean())

    def test_shuttle_luminance_tracks_the_scene_illumination(self) -> None:
        """A white shuttle is lit by the same hall as everything else, not by a fixed lamp."""
        dark = np.full((WH, WH, 3), 55, np.uint8)
        bright = np.full((WH, WH, 3), 215, np.uint8)
        in_dark = self._object_mean_luma(render_sample(
            self.parts, dark, target_px=14.0, seed=21, split="train", name="d"))
        in_bright = self._object_mean_luma(render_sample(
            self.parts, bright, target_px=14.0, seed=21, split="train", name="b"))
        self.assertGreater(in_bright, in_dark + 40.0,
                           f"bright scene {in_bright:.1f} vs dark scene {in_dark:.1f}")

    def test_a_white_shuttle_outshines_a_mid_grey_scene(self) -> None:
        """With no illumination coupling a white shuttle rendered darker than the scene."""
        scene = np.full((WH, WH, 3), 140, np.uint8)
        sample = render_sample(self.parts, scene, target_px=14.0, seed=22,
                               split="train", name="m")
        self.assertGreater(self._object_mean_luma(sample), 140.0)

    def test_sample_record_names_the_background_it_used(self) -> None:
        """Without the real background name a train/val leak audit is impossible."""
        sample = render_sample(self.parts, self.bg, target_px=10.0, seed=15,
                               split="train", name="train_00007",
                               background_name="tbg_003.png")
        self.assertEqual(sample.record["background"], "tbg_003.png")

    def test_imread_unicode_reads_an_image_at_a_non_ascii_path(self) -> None:
        """cv2.imread returns None for this repository path on Windows; work around it."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "背景_001.png"
            img = np.zeros((8, 8, 3), np.uint8)
            img[..., 0] = 200
            imwrite_unicode(path, img, quality=95)
            back = imread_unicode(path)
            self.assertIsNotNone(back)
            self.assertEqual(back.shape, (8, 8, 3))

    def test_imwrite_unicode_round_trips_a_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "样本_07.jpg"
            img = np.full((16, 16, 3), 120, np.uint8)
            imwrite_unicode(path, img, quality=92)
            back = imread_unicode(path)
            self.assertEqual(back.shape, (16, 16, 3))
            self.assertLess(int(np.abs(back.astype(int) - 120).max()), 12)

    def test_imread_unicode_returns_none_for_a_missing_file(self) -> None:
        self.assertIsNone(imread_unicode(Path("不存在_404.png")))

    def test_prepare_background_crops_to_the_requested_size(self) -> None:
        big = np.zeros((1400, 1600, 3), np.uint8)
        out = prepare_background(big, 960, 960, np.random.default_rng(0))
        self.assertEqual(out.shape, (960, 960, 3))

    def test_prepare_background_upscales_a_smaller_image(self) -> None:
        small = np.zeros((300, 400, 3), np.uint8)
        out = prepare_background(small, 960, 960, np.random.default_rng(0))
        self.assertEqual(out.shape, (960, 960, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
