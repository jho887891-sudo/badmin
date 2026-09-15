#!/usr/bin/env python3
"""Controlled single-variable sweeps for the shuttlecock capability test set.

Why this module exists: spec 03 section 7 says one experiment changes one variable,
and that is a property of the SAMPLE LIST, not of the renderer. render_sample()
randomises pose, position, blur, noise and background crop internally, so a sweep
built on top of it cannot be attributed to any single cause. The list of samples
that will be rendered -- which variable each row moves, and every value that is
held -- is therefore data this module owns, pure and testable, and the generator
CLI (tools/generate_controlled_capability.py) only walks it.

The vocabulary written here is the one the evaluator reads
(src/perception/shuttle_detection/evaluate.py): when a manifest carries
size_bucket / pose_bucket / position_bucket / blur_bucket / occlusion_bucket /
background, those exact strings become the group labels of the capability curves.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_controlled_capability.py -v
    pytest tests/perception/shuttle_detection/test_controlled_capability.py -v
"""
from __future__ import annotations

import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from src.perception.shuttle_detection.controlled_capability import (  # noqa: E402
    BACKGROUND_DIR,
    BLUR_LEVELS,
    NEAR_FIELD_MARGIN_PX,
    ORIENTATIONS_PER_FAMILY,
    REPLICATES_PER_GROUP,
    TARGET_HALF_WIDTH,
    NEAR_FIELD_MIN_DISTANCE_M,
    LIGHT_AZIMUTHS_DEG,
    NEAR_FIELD_SIZES_PX,
    NEAR_FIELD_SUPERSAMPLE,
    OCCLUDER_SIDE,
    MANIFEST_COLUMNS,
    OCCLUSION_LEVELS,
    OCCLUSION_TARGET_FRACTIONS,
    POSE_BUCKETS,
    POSE_FAMILIES,
    POSITION_TARGETS,
    REQUIRED_MANIFEST_COLUMNS,
    SIZE_BUCKETS,
    SIZE_BUCKET_TARGETS_PX,
    SWEEP_IDS,
    ControlSettings,
    apply_occlusion,
    bbox_inside_frame,
    build_plan,
    distribution,
    end_toward_camera,
    light_vector,
    long_axis_camera_space,
    expected_conditions,
    measured_condition_counts,
    merge_manifest_rows,
    near_field_camera,
    occlusion_bucket,
    planned_condition_counts,
    proportion_half_width,
    required_half_width,
    verify_group_counts,
    verify_matched_nuisance,
    position_pixel_centre,
    position_stays_inside_frame,
    size_bucket,
    verify_measured_pins,
    verify_single_variable,
)

# The camera the frozen P4-A synthetic set was rendered with, so a controlled
# image is comparable with the data the baseline was trained on.
WIDTH = HEIGHT = 960
FOCAL_PX = 700.0
BACKGROUNDS = ("bg_001.jpg", "bg_002.jpg", "bg_003.jpg", "bg_004.jpg")


def settings() -> ControlSettings:
    return ControlSettings(
        width=WIDTH,
        height=HEIGHT,
        focal_px=FOCAL_PX,
        background="bg_001.jpg",
        backgrounds=BACKGROUNDS,
    )


def plan():
    return build_plan(settings())


class SizeBucketTests(unittest.TestCase):
    """Spec 03 section 3: the eight buckets, and every bucket reachable."""

    def test_bucket_vocabulary_is_the_spec_vocabulary_in_order(self) -> None:
        self.assertEqual(
            SIZE_BUCKETS,
            ("<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", ">32"),
        )

    def test_edges_are_lower_inclusive_and_upper_exclusive(self) -> None:
        for value, expected in (
            (3.999, "<4"),
            (4.0, "4-6"),
            (5.99, "4-6"),
            (6.0, "6-8"),
            (8.0, "8-12"),
            (12.0, "12-16"),
            (16.0, "16-24"),
            (24.0, "24-32"),
            (32.0, ">32"),
            (120.0, ">32"),
        ):
            self.assertEqual(size_bucket(value), expected, str(value))

    def test_every_bucket_offers_at_least_three_targets_and_they_land_in_it(self) -> None:
        """A bucket with no achievable target is a hole in the capability curve.

        The renderer quantises the measured footprint to whole pixels, so a
        requested target is not always achievable near a bucket edge; several
        candidates per bucket are what let the generator stay inside the bucket
        without moving the edge.
        """
        self.assertEqual(set(SIZE_BUCKET_TARGETS_PX), set(SIZE_BUCKETS))
        for bucket, targets in SIZE_BUCKET_TARGETS_PX.items():
            self.assertGreaterEqual(len(set(targets)), 3, bucket)
            for target in targets:
                self.assertEqual(size_bucket(target), bucket, f"{bucket}: {target}")


class SharedVocabularyTests(unittest.TestCase):
    """The manifest, the audit and the curves must agree on what a bucket is.

    size_bucket is written into the manifest here and re-derived by the evaluator
    from equivalent_size_px there; two vocabularies would silently produce two
    different size axes in one report.
    """

    def test_size_buckets_match_the_evaluator(self) -> None:
        from src.perception.shuttle_detection import metrics

        self.assertEqual(SIZE_BUCKETS, tuple(metrics.SIZE_BUCKETS))

    def test_size_bucketing_agrees_with_the_evaluator_on_every_edge(self) -> None:
        from src.perception.shuttle_detection import metrics

        for value in (0.1, 3.999, 4.0, 5.999, 6.0, 7.999, 8.0, 11.999, 12.0, 15.999,
                      16.0, 23.999, 24.0, 31.999, 32.0, 400.0):
            self.assertEqual(size_bucket(value), metrics.size_bucket(value), str(value))


class PoseFamilyTests(unittest.TestCase):
    """Spec 03 section 4.2: every pose condition, as a true 3D orientation."""

    REQUIRED = (
        "cork_end_on",
        "feather_end_on",
        "side",
        "oblique",
        "yaw_only",
        "pitch_only",
        "roll_only",
        "flight_rotation",
    )

    def test_every_required_pose_condition_has_a_family(self) -> None:
        self.assertEqual(set(POSE_FAMILIES), set(self.REQUIRED) | set(POSE_FAMILIES))
        for name in self.REQUIRED:
            self.assertIn(name, POSE_FAMILIES, name)
            self.assertGreaterEqual(len(POSE_FAMILIES[name]), 1, name)

    def test_a_single_axis_family_moves_exactly_its_own_axis(self) -> None:
        for family, pinned in (
            ("yaw_only", ("pitch_deg", "roll_deg")),
            ("pitch_only", ("yaw_deg", "roll_deg")),
            ("roll_only", ("yaw_deg", "pitch_deg")),
        ):
            rows = POSE_FAMILIES[family]
            moved = {"yaw_deg", "pitch_deg", "roll_deg"} - set(pinned)
            for axis in pinned:
                values = {row[axis] for row in rows}
                self.assertEqual(len(values), 1, f"{family} moved {axis}: {values}")
            for axis in moved:
                values = {row[axis] for row in rows}
                self.assertGreaterEqual(len(values), 3, f"{family} barely moved {axis}")

    def test_the_two_end_on_cases_are_half_a_turn_apart(self) -> None:
        cork = POSE_FAMILIES["cork_end_on"][0]
        feather = POSE_FAMILIES["feather_end_on"][0]
        a = long_axis_camera_space(cork["yaw_deg"], cork["pitch_deg"], cork["roll_deg"])
        b = long_axis_camera_space(feather["yaw_deg"], feather["pitch_deg"], feather["roll_deg"])
        self.assertAlmostEqual(float(np.dot(a, b)), -1.0, places=6)

    def test_roll_only_rotates_about_the_shuttles_own_long_axis(self) -> None:
        rows = POSE_FAMILIES["roll_only"]
        axes = [
            long_axis_camera_space(r["yaw_deg"], r["pitch_deg"], r["roll_deg"]) for r in rows
        ]
        for axis in axes[1:]:
            self.assertAlmostEqual(float(np.dot(axes[0], axis)), 1.0, places=6)

    def test_flight_rotation_is_a_tumble_through_intermediate_orientations(self) -> None:
        rolls = {r["roll_deg"] for r in POSE_FAMILIES["flight_rotation"]}
        self.assertGreaterEqual(len(rolls), 8)
        for row in POSE_FAMILIES["flight_rotation"]:
            self.assertNotIn(round(row["roll_deg"], 6), (0.0,), "needs non-axis-aligned poses")


class MeshPoseClaimTests(unittest.TestCase):
    """The end-on claim must come from the mesh, not from a comment."""

    @classmethod
    def setUpClass(cls) -> None:
        from shuttle_render import load_shuttle_parts

        cls.parts = load_shuttle_parts()

    def test_identity_pose_puts_the_cork_end_at_the_camera(self) -> None:
        self.assertEqual(end_toward_camera(self.parts, 0.0, 0.0, 0.0), "cork")

    def test_half_turn_about_x_puts_the_feather_end_at_the_camera(self) -> None:
        self.assertEqual(end_toward_camera(self.parts, 0.0, 180.0, 0.0), "feather")

    def test_the_two_end_on_families_agree_with_the_mesh(self) -> None:
        cork = POSE_FAMILIES["cork_end_on"][0]
        feather = POSE_FAMILIES["feather_end_on"][0]
        self.assertEqual(
            end_toward_camera(
                self.parts, cork["yaw_deg"], cork["pitch_deg"], cork["roll_deg"]
            ),
            "cork",
        )
        self.assertEqual(
            end_toward_camera(
                self.parts, feather["yaw_deg"], feather["pitch_deg"], feather["roll_deg"]
            ),
            "feather",
        )


class PositionTests(unittest.TestCase):
    """Spec 03 section 4.3: center, left, right, top, bottom and four corners."""

    REQUIRED = {
        "center",
        "left",
        "right",
        "top",
        "bottom",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }

    def test_the_nine_required_positions_are_covered(self) -> None:
        self.assertEqual(set(POSITION_TARGETS), self.REQUIRED)

    def test_nominal_centres_are_ordered_left_to_right_and_top_to_bottom(self) -> None:
        px = {name: position_pixel_centre(name, WIDTH, HEIGHT)[0] for name in POSITION_TARGETS}
        py = {name: position_pixel_centre(name, WIDTH, HEIGHT)[1] for name in POSITION_TARGETS}
        self.assertLess(px["left"], px["center"])
        self.assertLess(px["center"], px["right"])
        self.assertLess(py["top"], py["center"])
        self.assertLess(py["center"], py["bottom"])

    def test_every_position_keeps_the_worst_case_box_inside_the_frame(self) -> None:
        """Spec 03 section 4.3 wants the whole target in frame, not a clipped one.

        The largest target any sweep pins is the top of the ">32" bucket, so that
        is the box the positions have to hold.
        """
        biggest = max(max(v) for v in SIZE_BUCKET_TARGETS_PX.values())
        for name in POSITION_TARGETS:
            self.assertTrue(
                position_stays_inside_frame(name, biggest, WIDTH, HEIGHT),
                f"{name} clips a {biggest} px target",
            )


class LightAndDegradationTests(unittest.TestCase):
    def test_blur_levels_are_none_light_medium_heavy(self) -> None:
        self.assertEqual([name for name, _ in BLUR_LEVELS], ["none", "light", "medium", "heavy"])
        self.assertEqual(BLUR_LEVELS[0][1], 0.0)
        for (_, smaller), (_, larger) in zip(BLUR_LEVELS, BLUR_LEVELS[1:]):
            self.assertLess(smaller, larger)

    def test_occlusion_levels_are_none_light_partial(self) -> None:
        self.assertEqual(
            [name for name, _ in OCCLUSION_LEVELS], ["none", "light", "partial"]
        )
        self.assertEqual(OCCLUSION_LEVELS[0][1], 0.0)
        for (_, smaller), (_, larger) in zip(OCCLUSION_LEVELS, OCCLUSION_LEVELS[1:]):
            self.assertLess(smaller, larger)

    def test_light_azimuths_give_distinct_directions(self) -> None:
        directions = [light_vector(a) for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)]
        for a in directions:
            self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=9)
        for i, a in enumerate(directions):
            for b in directions[i + 1:]:
                self.assertLess(float(np.dot(a, b)), 0.999, "two azimuths share a direction")

    def test_opposite_azimuths_light_from_opposite_sides(self) -> None:
        right = light_vector(0.0)
        left = light_vector(180.0)
        self.assertGreater(right[0], 0.0)
        self.assertLess(left[0], 0.0)
        self.assertAlmostEqual(right[2], left[2], places=9)


class PlanTests(unittest.TestCase):
    """The plan is the control: one variable per sweep, everything else pinned."""

    def test_the_plan_covers_every_sweep_the_spec_asks_for(self) -> None:
        self.assertEqual(
            SWEEP_IDS, ("S1", "S2", "S3", "S4", "S5", "S6a", "S6b", "S7")
        )
        # S7 is the near-field ladder, not a spec 03 section 4 variable: spec 03's own
        # sweeps are S1 to S6b, and S7 exists because the real positives are 838-1579 px
        # while this set used to stop at 44 px.
        self.assertEqual({s.sweep for s in plan()}, set(SWEEP_IDS))

    def test_every_sweep_varies_exactly_one_variable(self) -> None:
        self.assertEqual(verify_single_variable(plan()), [])

    def test_every_sweep_has_at_least_two_distinct_values_of_its_variable(self) -> None:
        for sweep in SWEEP_IDS:
            rows = [s for s in plan() if s.sweep == sweep]
            joined = [tuple(row.control_inputs()[f] for f in row.varied) for row in rows]
            self.assertGreaterEqual(len(set(joined)), 2, sweep)

    def test_the_checker_reports_a_pinned_value_that_drifted(self) -> None:
        """The checker must fail on the defect it exists to catch."""
        mutated = []
        drifted = False
        for sample in plan():
            if sample.sweep == "S1" and not drifted:
                drifted = True
                mutated.append(
                    sample.__class__(
                        **{**sample.__dict__, "background": BACKGROUNDS[1]}
                    )
                )
            else:
                mutated.append(sample)
        violations = verify_single_variable(mutated)
        self.assertTrue(violations, "a drifted pinned background was not reported")
        self.assertTrue(any("background" in v for v in violations))

    def test_size_sweep_targets_every_bucket_with_several_repeats(self) -> None:
        rows = [s for s in plan() if s.sweep == "S1"]
        by_bucket: dict[str, int] = {}
        for row in rows:
            bucket = size_bucket(row.target_px)
            by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
            self.assertEqual(row.varied, ("target_px",))
        self.assertEqual(set(by_bucket), set(SIZE_BUCKETS))
        for bucket, count in by_bucket.items():
            self.assertGreaterEqual(count, 3, bucket)

    def test_pose_sweep_pins_size_position_background_light_and_blur(self) -> None:
        rows = [s for s in plan() if s.sweep == "S2"]
        for field in ("target_px", "position_bucket", "background", "light_azimuth_deg", "motion_px"):
            self.assertEqual(len({r.control_inputs()[field] for r in rows}), 1, field)
        self.assertGreaterEqual(len(rows), len(PoseFamilyTests.REQUIRED))

    def test_pose_sweep_touches_every_pose_family(self) -> None:
        rows = [s for s in plan() if s.sweep == "S2"]
        self.assertEqual({r.pose_bucket for r in rows}, set(PoseFamilyTests.REQUIRED))

    def test_position_sweep_pins_size_pose_background_light_and_blur(self) -> None:
        rows = [s for s in plan() if s.sweep == "S3"]
        for field in ("target_px", "yaw_deg", "pitch_deg", "roll_deg", "background",
                      "light_azimuth_deg", "motion_px"):
            self.assertEqual(len({r.control_inputs()[field] for r in rows}), 1, field)
        self.assertEqual({r.position_bucket for r in rows}, set(POSITION_TARGETS))

    def test_background_sweep_pins_the_light_and_the_light_sweep_pins_the_background(self) -> None:
        by_sweep = {sweep: [s for s in plan() if s.sweep == sweep] for sweep in SWEEP_IDS}
        self.assertEqual(
            len({r.control_inputs()["light_azimuth_deg"] for r in by_sweep["S6a"]}), 1
        )
        self.assertEqual({r.background for r in by_sweep["S6a"]}, set(BACKGROUNDS))
        self.assertEqual(len({r.control_inputs()["background"] for r in by_sweep["S6b"]}), 1)
        self.assertGreaterEqual(len({r.light_azimuth_deg for r in by_sweep["S6b"]}), 8)

    def test_blur_sweep_is_driven_by_motion_px(self) -> None:
        rows = [s for s in plan() if s.sweep == "S4"]
        self.assertEqual({r.motion_px for r in rows}, {px for _, px in BLUR_LEVELS})
        self.assertEqual({r.control_inputs()["blur_bucket"] for r in rows},
                         {name for name, _ in BLUR_LEVELS})

    def test_occlusion_sweep_is_driven_by_the_occluder(self) -> None:
        rows = [s for s in plan() if s.sweep == "S5"]
        self.assertEqual({r.occlusion_bucket for r in rows},
                         {name for name, _ in OCCLUSION_LEVELS})
        for bucket, fraction in OCCLUSION_LEVELS:
            self.assertTrue(any(r.occlusion_bucket == bucket for r in rows), bucket)
            if fraction > 0.0:
                self.assertGreaterEqual(
                    len([r for r in rows if r.occlusion_bucket == bucket]), 2, bucket
                )

    def test_names_are_unique_ascii_and_file_system_safe(self) -> None:
        names = [s.name for s in plan()]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(name.isascii(), name)
            self.assertRegex(name, r"^[A-Za-z0-9_.-]+$")

    def test_every_row_carries_the_columns_the_evaluator_reads(self) -> None:
        for sample in plan():
            row = sample.as_manifest_row()
            for column in REQUIRED_MANIFEST_COLUMNS:
                self.assertIn(column, row, f"{sample.name}: {column}")
            self.assertEqual(row["split"], "fixed_core_test")
            self.assertEqual(row["file"], "images/" + sample.name + ".jpg")
        self.assertEqual(
            tuple(MANIFEST_COLUMNS[: len(REQUIRED_MANIFEST_COLUMNS)]),
            REQUIRED_MANIFEST_COLUMNS,
        )

    def test_seeds_are_deterministic_and_pinned_statistics_are_recorded(self) -> None:
        first, second = plan(), plan()
        self.assertEqual([s.seed for s in first], [s.seed for s in second])
        self.assertEqual(len({s.seed for s in first}), len(first))
        self.assertEqual(len({s.noise_sigma for s in first}), 1)


class MeasuredPinTests(unittest.TestCase):
    """The plan says what should be pinned; the manifest says what was.

    Rendering happens after the plan, so the only evidence that a sweep really held
    its other variables is the recorded rows themselves. These are the checks the
    generator runs over the finished manifest.
    """

    @staticmethod
    def rendered_rows():
        rows = []
        for index, sample in enumerate(plan()):
            row = sample.as_manifest_row()
            row["bbox_w_px"] = 16
            row["bbox_h_px"] = 16
            row["pos_x_px"] = 480
            row["pos_y_px"] = 480
            row["equivalent_size_px"] = 16.0
            row["equiv_size_px"] = 16.0
            row["distance_m"] = 2.0 + 0.001 * index
            row["occlusion_fraction"] = sample.occlusion_target_fraction
            rows.append(row)
        return rows

    def test_a_clean_manifest_has_no_violations(self) -> None:
        self.assertEqual(verify_measured_pins(self.rendered_rows()), [])

    def test_a_pinned_background_that_drifted_is_reported(self) -> None:
        rows = self.rendered_rows()
        for row in rows:
            if row["sweep"] == "S1":
                row["background"] = "bg_009.jpg"
                break
        violations = verify_measured_pins(rows)
        self.assertTrue(any("S1" in v and "background" in v for v in violations), violations)

    def test_a_pinned_pose_that_drifted_is_reported(self) -> None:
        rows = self.rendered_rows()
        for row in rows:
            if row["sweep"] == "S3":
                row["pitch_deg"] = 31.0
                break
        violations = verify_measured_pins(rows)
        self.assertTrue(any("S3" in v and "pitch_deg" in v for v in violations), violations)

    def test_the_measured_occlusion_fraction_may_move_in_the_occlusion_sweep(self) -> None:
        rows = self.rendered_rows()
        for index, row in enumerate(rows):
            if row["sweep"] == "S5":
                row["occlusion_fraction"] = 0.1 + 0.05 * index
        self.assertEqual(verify_measured_pins(rows), [])

    def test_the_near_field_rows_are_checked_like_every_other_sweep(self) -> None:
        rows = self.rendered_rows()
        near = [row for row in rows if row["sweep"] == "S7"]
        self.assertTrue(near)
        self.assertEqual({row["imgsz"] for row in near}, {1280})
        self.assertEqual({row["supersample"] for row in near}, {1})
        self.assertEqual(verify_measured_pins(rows), [])

    def test_a_near_field_row_rendered_in_another_frame_is_reported(self) -> None:
        rows = self.rendered_rows()
        for row in rows:
            if row["sweep"] == "S7":
                row["imgsz"] = 960
                break
        violations = verify_measured_pins(rows)
        self.assertTrue(any("S7" in v and "imgsz" in v for v in violations), violations)

    def test_an_occlusion_fraction_outside_its_sweep_is_reported(self) -> None:
        rows = self.rendered_rows()
        for row in rows:
            if row["sweep"] == "S2":
                row["occlusion_fraction"] = 0.3
                break
        violations = verify_measured_pins(rows)
        self.assertTrue(any("occlusion_fraction" in v for v in violations), violations)


class FrameAndDistributionTests(unittest.TestCase):
    """Two small checks the post-generation report is built on."""

    def test_a_box_that_fits_is_accepted(self) -> None:
        row = {"pos_x_px": 480, "pos_y_px": 480, "bbox_w_px": 16, "bbox_h_px": 16}
        self.assertTrue(bbox_inside_frame(row, 960, 960))

    def test_a_box_touching_each_edge_is_still_inside(self) -> None:
        for centre_x, centre_y in ((7, 480), (951, 480), (480, 7), (480, 951)):
            row = {"pos_x_px": centre_x, "pos_y_px": centre_y, "bbox_w_px": 16, "bbox_h_px": 16}
            self.assertTrue(bbox_inside_frame(row, 960, 960), f"{centre_x},{centre_y}")

    def test_a_clipped_box_is_rejected(self) -> None:
        for centre_x, centre_y in ((6, 480), (952, 480), (480, 6), (480, 952)):
            row = {"pos_x_px": centre_x, "pos_y_px": centre_y, "bbox_w_px": 16, "bbox_h_px": 16}
            self.assertFalse(bbox_inside_frame(row, 960, 960), f"{centre_x},{centre_y}")

    def test_a_row_without_a_box_is_rejected(self) -> None:
        self.assertFalse(bbox_inside_frame({"pos_x_px": 480}, 960, 960))

    def test_distribution_counts_by_column(self) -> None:
        rows = [{"sweep": "S1"}, {"sweep": "S1"}, {"sweep": "S2"}]
        self.assertEqual(distribution(rows, "sweep"), {"S1": 2, "S2": 1})
        self.assertEqual(distribution(rows, "missing"), {"": 3})


class NearFieldTests(unittest.TestCase):
    """The large-target ladder the size sweep could not reach.

    Measured elsewhere: the largest target in the training set is 32.86 px while the
    real verified positives are 838-1579 px, and the trained baseline finds none of
    them. Whether that is a size-coverage gap or a domain gap can only be told apart
    by putting SYNTHETIC shuttles of that size in front of the detector, which is what
    this ladder exists for. The spec 03 section 3 buckets are untouched: every rung of
    this ladder falls in ">32" and is identified by its sweep and its measured size.
    """

    REAL_POSITIVE_RANGE_PX = (837.8, 1578.8)

    def test_the_ladder_reaches_past_the_bottom_of_the_real_range(self) -> None:
        self.assertGreaterEqual(max(NEAR_FIELD_SIZES_PX), self.REAL_POSITIVE_RANGE_PX[0])

    def test_at_least_one_rung_sits_comfortably_inside_the_real_range(self) -> None:
        low, high = self.REAL_POSITIVE_RANGE_PX
        inside = [size for size in NEAR_FIELD_SIZES_PX if low <= size <= high]
        self.assertTrue(inside, "no rung lands inside the real 838-1579 px range")
        self.assertTrue(
            any(size >= low * 1.1 for size in inside),
            "no rung is comfortably above the low edge of the real range",
        )

    def test_the_ladder_rises_without_repeats(self) -> None:
        self.assertEqual(list(NEAR_FIELD_SIZES_PX), sorted(NEAR_FIELD_SIZES_PX))
        self.assertEqual(len(set(NEAR_FIELD_SIZES_PX)), len(NEAR_FIELD_SIZES_PX))
        self.assertGreater(len(NEAR_FIELD_SIZES_PX), 1)

    def test_every_rung_fits_the_frame_with_the_required_margin(self) -> None:
        camera = near_field_camera()
        for target in NEAR_FIELD_SIZES_PX:
            margin = camera.frame_px / 2.0 - target / 2.0
            self.assertGreaterEqual(margin, NEAR_FIELD_MARGIN_PX, str(target))

    def test_the_focal_keeps_every_rung_well_in_front_of_the_camera(self) -> None:
        """Too short a lens needs a camera distance inside the mesh, which drops faces."""
        camera = near_field_camera()
        for target in NEAR_FIELD_SIZES_PX:
            self.assertGreater(
                camera.naive_distance_m(target), NEAR_FIELD_MIN_DISTANCE_M, str(target)
            )

    def test_one_camera_serves_the_whole_ladder(self) -> None:
        """A per-rung frame or focal would be a second variable inside a size sweep."""
        camera = near_field_camera()
        self.assertEqual(camera.supersample, NEAR_FIELD_SUPERSAMPLE)
        self.assertEqual(camera, near_field_camera())

    def test_a_frame_too_small_for_the_ladder_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            near_field_camera(frame_px=int(max(NEAR_FIELD_SIZES_PX)))

    def test_a_focal_too_short_for_the_ladder_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            near_field_camera(focal_px=700.0)

    def test_the_supersample_is_the_cheap_one_because_the_target_is_huge(self) -> None:
        self.assertEqual(NEAR_FIELD_SUPERSAMPLE, 1)
        self.assertGreater(min(NEAR_FIELD_SIZES_PX), 32.0)


class NearFieldPlanTests(unittest.TestCase):
    """The ladder has to be a controlled experiment like every other sweep."""

    @staticmethod
    def rows():
        return [sample for sample in plan() if sample.sweep == "S7"]

    def test_the_near_field_sweep_varies_only_the_size(self) -> None:
        rows = self.rows()
        self.assertTrue(rows)
        self.assertEqual({row.varied for row in rows}, {("target_px",)})
        for field in ("background", "light_azimuth_deg", "motion_px", "position_bucket",
                      "yaw_deg", "pitch_deg", "roll_deg", "occlusion_bucket", "pose_bucket"):
            self.assertEqual(len({row.control_inputs()[field] for row in rows}), 1, field)

    def test_the_near_field_sweep_pins_one_camera(self) -> None:
        rows = self.rows()
        for field in ("width", "height", "focal_px", "supersample"):
            self.assertEqual(len({getattr(row, field) for row in rows}), 1, field)
        camera = near_field_camera()
        self.assertEqual(rows[0].focal_px, camera.focal_px)
        self.assertEqual(rows[0].width, camera.frame_px)
        self.assertEqual(rows[0].supersample, camera.supersample)

    def test_every_rung_has_several_repeats(self) -> None:
        counts = Counter(row.target_px for row in self.rows())
        self.assertEqual(set(counts), set(NEAR_FIELD_SIZES_PX))
        for target, count in counts.items():
            self.assertGreaterEqual(count, REPLICATES_PER_GROUP["S7"], str(target))

    def test_every_near_field_box_stays_inside_its_own_frame(self) -> None:
        rows = self.rows()
        biggest = max(row.target_px for row in rows)
        for row in rows:
            self.assertTrue(
                position_stays_inside_frame(row.position_bucket, biggest, row.width, row.height),
                row.name,
            )

    def test_the_existing_sweeps_are_untouched_by_the_new_one(self) -> None:
        for sample in plan():
            if sample.sweep == "S7":
                continue
            self.assertEqual((sample.width, sample.height), (WIDTH, HEIGHT), sample.name)
            self.assertEqual(sample.focal_px, FOCAL_PX, sample.name)
            self.assertEqual(sample.supersample, 3, sample.name)


class ManifestMergeTests(unittest.TestCase):
    """Adding a sweep to a finished set must not disturb the rows already shipped."""

    def test_appended_rows_keep_the_existing_rows_and_their_order(self) -> None:
        existing = [{"file": "images/a.jpg"}, {"file": "images/b.jpg"}]
        merged = merge_manifest_rows(existing, [{"file": "images/c.jpg"}])
        self.assertEqual(
            [row["file"] for row in merged],
            ["images/a.jpg", "images/b.jpg", "images/c.jpg"],
        )

    def test_a_row_already_present_is_replaced_rather_than_duplicated(self) -> None:
        existing = [{"file": "images/a.jpg", "sweep": "S1"}, {"file": "images/b.jpg", "sweep": "S1"}]
        merged = merge_manifest_rows(existing, [{"file": "images/a.jpg", "sweep": "S7"}])
        self.assertEqual(len(merged), 2)
        self.assertEqual([row["file"] for row in merged], ["images/a.jpg", "images/b.jpg"])
        self.assertEqual(merged[0]["sweep"], "S7")

    def test_an_empty_existing_manifest_is_just_the_new_rows(self) -> None:
        merged = merge_manifest_rows([], [{"file": "images/a.jpg"}])
        self.assertEqual([row["file"] for row in merged], ["images/a.jpg"])

    def test_merging_does_not_mutate_its_inputs(self) -> None:
        existing = [{"file": "images/a.jpg", "sweep": "S1"}]
        new = [{"file": "images/a.jpg", "sweep": "S7"}]
        merge_manifest_rows(existing, new)
        self.assertEqual(existing[0]["sweep"], "S1")
        self.assertEqual(new[0]["sweep"], "S7")


class ReplicateTests(unittest.TestCase):
    """Error bars, not just counts.

    A conditioned group of one is not a measurement: a binomial proportion at n=3 has
    a 95% half-width of 57%, so "cork_end_on recall 0.0" from a single row is noise.
    These tests hold the set to the replicate counts that make a curve readable, and
    to the nuisance design that makes the replicates independent.
    """

    def test_the_required_replicates_meet_the_precision_target(self) -> None:
        self.assertEqual(
            REPLICATES_PER_GROUP,
            {"S1": 40, "S2": 40, "S3": 40, "S4": 20, "S5": 20, "S6a": 3, "S6b": 20, "S7": 40},
        )
        # The two sizes the brief asks for: ~15% at n=40, and 20 rows for the level
        # sweeps, which buys 21.9% and is reported as such.
        self.assertLessEqual(required_half_width("S1"), 0.16)
        self.assertLessEqual(required_half_width("S2"), 0.16)
        self.assertLessEqual(required_half_width("S3"), 0.16)
        self.assertLessEqual(required_half_width("S7"), 0.16)
        self.assertLessEqual(required_half_width("S4"), TARGET_HALF_WIDTH)
        self.assertLessEqual(required_half_width("S5"), TARGET_HALF_WIDTH)
        self.assertLessEqual(required_half_width("S6b"), TARGET_HALF_WIDTH)

    def test_the_half_width_is_the_worst_case_binomial_one(self) -> None:
        self.assertAlmostEqual(proportion_half_width(40), 1.96 * math.sqrt(0.25 / 40), places=12)
        self.assertAlmostEqual(proportion_half_width(3), 0.566, places=3)
        for smaller, larger in zip((3, 10, 20, 40), (10, 20, 40, 400)):
            self.assertGreater(
                proportion_half_width(smaller), proportion_half_width(larger)
            )

    def test_every_conditioned_group_in_the_plan_reaches_its_required_count(self) -> None:
        counts = planned_condition_counts(plan())
        self.assertEqual(
            verify_group_counts(counts, expected_conditions(settings())), []
        )
        total = sum(sum(groups.values()) for groups in counts.values())
        self.assertEqual(total, len(plan()))
        # Every condition times its required replicates. With the pool's thirty
        # backgrounds this is the 1750-image set; the test settings carry four.
        self.assertEqual(
            total,
            len(SIZE_BUCKETS) * REPLICATES_PER_GROUP["S1"]
            + len(POSE_BUCKETS) * REPLICATES_PER_GROUP["S2"]
            + len(POSITION_TARGETS) * REPLICATES_PER_GROUP["S3"]
            + len(BLUR_LEVELS) * REPLICATES_PER_GROUP["S4"]
            + len(OCCLUSION_LEVELS) * REPLICATES_PER_GROUP["S5"]
            + len(BACKGROUNDS) * REPLICATES_PER_GROUP["S6a"]
            + len(LIGHT_AZIMUTHS_DEG) * REPLICATES_PER_GROUP["S6b"]
            + len(NEAR_FIELD_SIZES_PX) * REPLICATES_PER_GROUP["S7"],
        )

    def test_the_conditioned_groups_are_the_conditioned_columns(self) -> None:
        counts = planned_condition_counts(plan())
        # Eight size buckets, eight pose families, nine positions, four blur levels,
        # three occlusion levels, thirty backgrounds, eight azimuths, nine rungs.
        # The test settings carry four backgrounds rather than the pool's thirty.
        self.assertEqual(
            [len(counts[sweep]) for sweep in SWEEP_IDS], [8, 8, 9, 4, 3, 4, 8, 9]
        )

    def test_verify_group_counts_reports_a_short_group(self) -> None:
        counts = planned_condition_counts(plan())
        counts["S3"]["bottom"] = 1
        violations = verify_group_counts(counts, expected_conditions(settings()))
        self.assertTrue(any("S3" in v and "bottom" in v for v in violations), violations)

    def test_verify_group_counts_reports_a_missing_and_an_unexpected_group(self) -> None:
        counts = planned_condition_counts(plan())
        del counts["S4"]["heavy"]
        counts["S4"]["nonexistent"] = 20
        violations = verify_group_counts(counts, expected_conditions(settings()))
        self.assertTrue(any("heavy" in v for v in violations), violations)
        self.assertTrue(any("nonexistent" in v for v in violations), violations)

    def test_repeats_share_the_crop_sequence_and_differ_in_noise(self) -> None:
        """The nuisance realisation must be matched across conditions, not the same.

        Every conditioned group draws the SAME sequence of background crops, so no
        bucket is measured on a luckier scene than another; within a group the rows
        differ by the noise draw, so the replicates are not copies of one image.
        """
        rows = plan()
        self.assertEqual(verify_matched_nuisance(rows), [])
        for sweep in SWEEP_IDS:
            group = [s for s in rows if s.sweep == sweep]
            self.assertGreater(len({s.crop_seed for s in group}), 1, sweep)
            self.assertEqual(len({s.noise_seed for s in group}), len(group), sweep)

    def test_verify_matched_nuisance_reports_a_group_on_its_own_scene(self) -> None:
        rows = list(plan())
        for index, sample in enumerate(rows):
            if sample.sweep == "S3" and sample.repeat == 0:
                rows[index] = sample.__class__(
                    **{**sample.__dict__, "crop_seed": sample.crop_seed + 1}
                )
                break
        violations = verify_matched_nuisance(rows)
        self.assertTrue(any("S3" in v and "crop" in v for v in violations), violations)

    def test_measured_condition_counts_group_by_the_measured_condition(self) -> None:
        rows = [
            {"sweep": "S1", "equivalent_size_px": "3.0", "target_px": "3.0"},
            {"sweep": "S1", "equivalent_size_px": "5.0", "target_px": "5.0"},
            {"sweep": "S2", "pose_bucket": "side"},
        ]
        counts = measured_condition_counts(rows)
        self.assertEqual(counts["S1"], {"<4": 1, "4-6": 1})
        self.assertEqual(counts["S2"], {"side": 1})

    def test_every_pose_family_offers_enough_distinct_orientations(self) -> None:
        for family, rows in POSE_FAMILIES.items():
            self.assertGreaterEqual(len(rows), ORIENTATIONS_PER_FAMILY, family)
            seen = {
                (row["yaw_deg"], row["pitch_deg"], row["roll_deg"]) for row in rows
            }
            self.assertEqual(len(seen), len(rows), family)


class CommittedManifestTests(unittest.TestCase):
    """The shipped manifest has to carry the replicates its curves need.

    This is the check that turns "enough samples" from a hope into a test: it reads
    the manifest that is actually committed and holds every conditioned group to its
    required count and to an error bar a reader can use.
    """

    MANIFEST = ROOT / "outputs" / "shuttle_capability" / "controlled_capability" / "manifest.csv"

    @classmethod
    def setUpClass(cls) -> None:
        import csv

        if not cls.MANIFEST.is_file():
            raise unittest.SkipTest("controlled capability manifest is not present")
        with cls.MANIFEST.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        # The expectations come from the frozen inputs - the audited background pool and
        # the ladder constants - never from the manifest being checked: reading the
        # vocabulary out of the file under test would accept a condition that is simply
        # missing from it.
        pool = tuple(sorted(path.name for path in BACKGROUND_DIR.glob("*.jpg")))
        if len(pool) < 2:
            raise unittest.SkipTest("the audited background pool is not present")
        cls.settings = ControlSettings(background=pool[0], backgrounds=pool)

    def test_every_conditioned_group_reaches_its_required_count(self) -> None:
        counts = measured_condition_counts(self.rows)
        expected = expected_conditions(self.settings)
        violations = verify_group_counts(counts, expected)
        self.assertEqual(violations, [], "; ".join(violations))

    def test_every_conditioned_group_carries_a_usable_error_bar(self) -> None:
        counts = measured_condition_counts(self.rows)
        for sweep, groups in counts.items():
            if sweep == "S6a":
                # The background sweep is a nuisance variable, cut deliberately; it is the
                # only group that does not carry a per-condition error bar worth reading.
                continue
            for label, count in groups.items():
                self.assertLessEqual(
                    proportion_half_width(count),
                    required_half_width(sweep),
                    f"{sweep}/{label} has n={count}",
                )

    def test_the_shipped_rows_keep_their_controls(self) -> None:
        self.assertEqual(verify_measured_pins(self.rows), [])
        self.assertEqual({row["split"] for row in self.rows}, {"fixed_core_test"})


class OcclusionTests(unittest.TestCase):
    """Spec 03 section 4.5: none / light / partial, measured rather than assumed.

    The renderer has no occlusion support, so the occluder is drawn over the
    composited image and the fraction of the OBJECT FOOTPRINT it hides is measured.
    A sweep that records the fraction it asked for instead of the fraction it drew
    would report a curve against a variable nobody controlled.
    """

    @staticmethod
    def scene(size: int = 240, radius: int = 45):
        yy, xx = np.mgrid[0:size, 0:size]
        footprint = ((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2) <= radius ** 2
        image = np.zeros((size, size, 3), np.uint8)
        image[..., 0] = 200
        image[..., 1] = 180
        image[..., 2] = 160
        return image, footprint

    def test_measured_fraction_is_the_measured_fraction(self) -> None:
        image, footprint = self.scene()
        for target in (0.15, 0.25, 0.35, 0.45, 0.55, 0.70):
            result = apply_occlusion(image, footprint, target_fraction=target, seed=7)
            covered = np.count_nonzero((result.occluder_alpha >= 0.5) & footprint)
            actual = covered / float(np.count_nonzero(footprint))
            self.assertAlmostEqual(result.measured_fraction, actual, places=9)
            self.assertLessEqual(abs(result.measured_fraction - target), 0.03, str(target))

    def test_the_reported_bucket_follows_the_measured_fraction(self) -> None:
        image, footprint = self.scene()
        for target, expected in ((0.0, "none"), (0.15, "light"), (0.35, "light"), (0.55, "partial")):
            result = apply_occlusion(image, footprint, target_fraction=target, seed=3)
            self.assertEqual(result.bucket, expected, str(target))
            self.assertEqual(result.bucket, occlusion_bucket(result.measured_fraction))

    def test_an_unoccluded_run_returns_the_image_untouched(self) -> None:
        image, footprint = self.scene()
        result = apply_occlusion(image, footprint, target_fraction=0.0, seed=1)
        self.assertEqual(result.measured_fraction, 0.0)
        self.assertEqual(result.bucket, "none")
        np.testing.assert_array_equal(result.image, image)
        self.assertEqual(int(np.count_nonzero(result.occluder_alpha)), 0)

    def test_pixels_the_occluder_does_not_reach_are_unchanged(self) -> None:
        image, footprint = self.scene()
        result = apply_occlusion(image, footprint, target_fraction=0.35, seed=5)
        untouched = result.occluder_alpha <= 0.0
        self.assertGreater(int(untouched.sum()), 0)
        np.testing.assert_array_equal(result.image[untouched], image[untouched])

    def test_the_occluder_actually_darkens_the_object_it_covers(self) -> None:
        image, footprint = self.scene()
        result = apply_occlusion(image, footprint, target_fraction=0.55, seed=5)
        hidden = (result.occluder_alpha >= 0.5) & footprint
        self.assertGreater(int(hidden.sum()), 0)
        before = image[hidden].astype(np.int32).mean()
        after = result.image[hidden].astype(np.int32).mean()
        self.assertLess(after, before)

    def test_the_side_decides_which_edge_is_covered(self) -> None:
        image, footprint = self.scene()
        left = apply_occlusion(image, footprint, target_fraction=0.25, side="left", seed=2)
        right = apply_occlusion(image, footprint, target_fraction=0.25, side="right", seed=2)
        centre_x = np.nonzero(footprint)[1].mean()
        left_x = np.nonzero((left.occluder_alpha >= 0.5) & footprint)[1].mean()
        right_x = np.nonzero((right.occluder_alpha >= 0.5) & footprint)[1].mean()
        self.assertLess(left_x, centre_x)
        self.assertGreater(right_x, centre_x)

    def test_the_sweeps_occluder_side_is_a_constant(self) -> None:
        self.assertIn(OCCLUDER_SIDE, ("left", "right", "top", "bottom"))

    def test_occlusion_is_reproducible_for_a_seed(self) -> None:
        image, footprint = self.scene()
        first = apply_occlusion(image, footprint, target_fraction=0.45, seed=11)
        second = apply_occlusion(image, footprint, target_fraction=0.45, seed=11)
        third = apply_occlusion(image, footprint, target_fraction=0.45, seed=12)
        np.testing.assert_array_equal(first.image, second.image)
        self.assertEqual(first.measured_fraction, second.measured_fraction)
        self.assertFalse(np.array_equal(first.image, third.image))

    def test_no_swept_row_hides_the_whole_object(self) -> None:
        """A fully hidden target is not an occluded target, it is a deleted one."""
        biggest = max(max(values) for values in OCCLUSION_TARGET_FRACTIONS.values())
        self.assertLess(biggest, 0.80)
        image, footprint = self.scene()
        result = apply_occlusion(image, footprint, target_fraction=biggest, seed=4)
        self.assertLessEqual(result.measured_fraction, 0.80)
        self.assertGreater(result.measured_fraction, 0.60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
