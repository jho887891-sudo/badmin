#!/usr/bin/env python3
"""Tests for the CHALLENGE_TEST split (src/perception/shuttle_detection/challenge_test.py).

Why this module exists: spec 08 section 4.2 requires a challenge split that is separate
from the frozen core, grows over time and never enters training, and spec 08 section 12
gates acceptance on "challenge test = NO UNEXPLAINED SYSTEMIC HARD FAILURE". The split did
not exist, so that gate had nothing to run against. These tests hold the split to the
properties that make it worth running: it is built from the sources the spec names, it
shares no scene and no image with the frozen core, and the training path refuses it.

Run:
    python tests/perception/shuttle_detection/test_challenge_test.py -v
    pytest tests/perception/shuttle_detection/test_challenge_test.py -v
"""
from __future__ import annotations

import csv
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from src.perception.shuttle_detection.challenge_test import (  # noqa: E402
    CHALLENGE_SWEEPS,
    CHALLENGE_SPLIT,
    COMPOSITION_COLUMNS,
    FAST_BLUR_PX,
    HARD_NEGATIVE_DIR,
    HISTORICAL_SIZES_PX,
    SWEEP_SOURCE,
    TINY_TARGETS_PX,
    build_challenge_plan,
    challenge_composition,
    challenge_varied_columns,
    negative_share,
    select_challenge_backgrounds,
)
from src.perception.shuttle_detection.controlled_capability import (  # noqa: E402
    BACKGROUND_DIR,
    ControlSettings,
    SIZE_BUCKETS,
    SWEEP_IDS,
    verify_measured_pins,
)

SCENES = ("hn_001.jpg", "hn_002.jpg", "hn_003.jpg", "hn_004.jpg")
CORE_POOL = ROOT / "outputs" / "shuttle_capability" / "real_images" / "backgrounds"
CORE_MANIFEST = (
    ROOT / "outputs" / "shuttle_capability" / "controlled_capability" / "manifest.csv"
)
CHALLENGE_MANIFEST = (
    ROOT / "outputs" / "shuttle_capability" / "challenge_test" / "manifest.csv"
)


def settings() -> ControlSettings:
    pool = ("bg_001.jpg", "bg_002.jpg", "bg_003.jpg", "bg_004.jpg", "bg_005.jpg")
    return ControlSettings(
        background=pool[0], backgrounds=pool, block_backgrounds=pool
    )


def plan():
    return build_challenge_plan(settings(), SCENES)


class SourceCategoryTests(unittest.TestCase):
    """Every spec 08 section 4.2 source is either built or reported as a gap."""

    def test_every_challenge_sweep_names_a_spec_source(self) -> None:
        self.assertEqual(set(SWEEP_SOURCE), set(CHALLENGE_SWEEPS))
        for sweep, source in SWEEP_SOURCE.items():
            self.assertTrue(source, sweep)

    def test_the_sweeps_are_not_the_core_sweeps(self) -> None:
        """A challenge row that reused a core sweep id would be read as core material."""
        for sweep in CHALLENGE_SWEEPS:
            self.assertNotIn(sweep, SWEEP_IDS)

    def test_the_categories_covered_and_the_one_that_cannot_be(self) -> None:
        from src.perception.shuttle_detection.challenge_test import CHALLENGE_SOURCES

        names = [name for name, _ in CHALLENGE_SOURCES]
        for covered in ("extreme_background", "very_small_targets", "fast_blur",
                        "extreme_pose", "historical_failure"):
            self.assertIn(covered, names)
        self.assertIn("real_machine_footage", names)
        note = dict(CHALLENGE_SOURCES)["real_machine_footage"]
        # The gap is stated, not silently dropped: no real-machine footage exists here.
        self.assertIn("NOT COVERED", note)


class ChallengePlanTests(unittest.TestCase):
    def test_every_row_is_challenge_test_and_stands_on_a_hard_scene(self) -> None:
        rows = plan()
        self.assertTrue(rows)
        for sample in rows:
            self.assertEqual(sample.split, CHALLENGE_SPLIT)
            self.assertIn(sample.background, SCENES)

    def test_no_row_reuses_a_frozen_core_scene(self) -> None:
        core = {path.name for path in CORE_POOL.glob("*.jpg")}
        for sample in plan():
            self.assertNotIn(sample.background, core, sample.name)

    def test_the_plan_covers_every_source_with_the_expected_counts(self) -> None:
        counts = Counter(sample.sweep for sample in plan())
        self.assertEqual(
            counts,
            {
                "C1": len(SCENES) * 2,
                "C1n": len(SCENES),
                "C2": len(TINY_TARGETS_PX) * 10,
                "C3": len(FAST_BLUR_PX) * 8,
                "C4": 8 * 5,
                "C5": len(HISTORICAL_SIZES_PX) * 8,
            },
        )

    def test_the_negatives_carry_no_object(self) -> None:
        negatives = [sample for sample in plan() if sample.is_negative]
        self.assertEqual(len(negatives), len(SCENES))
        for sample in negatives:
            self.assertEqual(sample.target_px, 0.0)
            self.assertEqual(sample.sweep, "C1n")
        self.assertLess(negative_share(plan()), 0.5)
        self.assertGreater(negative_share(plan()), 0.0)

    def test_the_tiny_targets_go_to_and_below_the_measured_floor(self) -> None:
        tiny = [sample.target_px for sample in plan() if sample.sweep == "C2"]
        self.assertLess(min(tiny), 2.0)
        self.assertEqual(sorted(set(tiny)), sorted(TINY_TARGETS_PX))

    def test_the_blur_goes_past_the_core_sweeps_heaviest_level(self) -> None:
        fast = {sample.motion_px for sample in plan() if sample.sweep == "C3"}
        self.assertEqual(fast, set(FAST_BLUR_PX))
        self.assertGreater(min(fast), 7.0)
        # Its own group label, not folded into the core's "heavy": the evaluator groups
        # by this column and 20 px of smear is not the same stimulus as 7 px.
        self.assertEqual(
            {sample.blur_bucket for sample in plan() if sample.sweep == "C3"}, {"extreme"}
        )

    def test_the_pose_sweep_is_the_family_measured_hardest(self) -> None:
        poses = [sample for sample in plan() if sample.sweep == "C4"]
        self.assertEqual({sample.pose_bucket for sample in poses}, {"side"})
        self.assertEqual(len({(s.yaw_deg, s.pitch_deg, s.roll_deg) for s in poses}), 8)

    def test_the_historical_sizes_go_above_the_core_ceiling(self) -> None:
        far = [sample for sample in plan() if sample.sweep == "C5"]
        self.assertGreater(min(sample.target_px for sample in far), 1024.0)
        self.assertGreaterEqual(max(sample.target_px for sample in far), 1578.0)
        for sample in far:
            self.assertGreater(sample.width, 1578)
            self.assertEqual(sample.width, sample.height)

    def test_every_challenge_sweep_declares_what_it_moves(self) -> None:
        varied = challenge_varied_columns()
        self.assertEqual(set(varied), set(CHALLENGE_SWEEPS))
        # A varied column is either an input the plan controls or the measured outcome of
        # one of those inputs; anything else would mean a sweep declared something it does
        # not drive.
        measured_outcomes = {"equivalent_size_px", "occlusion_fraction", "blur_bucket"}
        for sample in plan():
            inputs = sample.control_inputs()
            for column in varied[sample.sweep]:
                self.assertTrue(
                    column in inputs or column in measured_outcomes,
                    sample.name + ": " + column,
                )

    def test_the_planned_rows_keep_their_pinned_columns(self) -> None:
        rows = [sample.as_manifest_row() for sample in plan()]
        self.assertEqual(
            verify_measured_pins(rows, challenge_varied_columns()), []
        )

    def test_a_pinned_column_that_drifted_is_still_reported(self) -> None:
        """The challenge split gets the same control check as the core, not a weaker one."""
        rows = [sample.as_manifest_row() for sample in plan()]
        for row in rows:
            if row["sweep"] == "C3":
                row["light_azimuth_deg"] = 45.0
                break
        violations = verify_measured_pins(rows, challenge_varied_columns())
        self.assertTrue(any("C3" in v and "light_azimuth_deg" in v for v in violations), violations)


class CompositionTests(unittest.TestCase):
    def test_the_composition_reports_measured_values_per_source(self) -> None:
        rows = [sample.as_manifest_row() for sample in plan()]
        for row in rows:
            row["equivalent_size_px"] = row["target_px"]
            row["is_negative"] = str(row["is_negative"])
        table = challenge_composition(rows)
        self.assertEqual([entry["sweep"] for entry in table], list(CHALLENGE_SWEEPS))
        for entry in table:
            self.assertEqual(set(entry), set(COMPOSITION_COLUMNS))
            self.assertGreater(entry["n_rows"], 0)
        tiny = next(entry for entry in table if entry["sweep"] == "C2")
        self.assertAlmostEqual(tiny["size_min_px"], min(TINY_TARGETS_PX), places=6)
        far = next(entry for entry in table if entry["sweep"] == "C5")
        self.assertAlmostEqual(far["size_max_px"], max(HISTORICAL_SIZES_PX), places=6)
        blur = next(entry for entry in table if entry["sweep"] == "C3")
        self.assertAlmostEqual(blur["max_motion_px"], max(FAST_BLUR_PX), places=6)


class HardSceneSelectionTests(unittest.TestCase):
    """A scene the core already renders on cannot be challenge material."""

    def test_the_pool_is_split_by_measured_content_not_by_name(self) -> None:
        candidates = sorted(HARD_NEGATIVE_DIR.glob("*.jpg"))
        forbidden = sorted(CORE_POOL.glob("*.jpg"))
        if not candidates or not forbidden:
            self.skipTest("the hard-negative pool or the frozen pool is not present")
        usable, excluded = select_challenge_backgrounds(candidates, forbidden)
        self.assertEqual(len(usable) + len(excluded), len(candidates))
        self.assertEqual(sorted(usable + tuple(name for name, _, _ in excluded)),
                         sorted(path.name for path in candidates))
        for name, match, score in excluded:
            self.assertGreaterEqual(score, 0.95)
            self.assertTrue(match)
        # Whatever the pool is, no usable scene may be the same picture as a core scene.
        self.assertEqual(
            _renderer_leaks([HARD_NEGATIVE_DIR / name for name in usable], forbidden), []
        )

    def test_a_pool_with_no_overlap_keeps_everything(self) -> None:
        candidates = sorted(HARD_NEGATIVE_DIR.glob("*.jpg"))[:4]
        if not candidates:
            self.skipTest("the hard-negative pool is not present")
        usable, excluded = select_challenge_backgrounds(candidates, candidates)
        # The same pool compared with itself is a full overlap, which is the degenerate
        # case: every scene is excluded rather than silently reused.
        self.assertEqual(usable, ())
        self.assertEqual(len(excluded), len(candidates))


def _renderer_leaks(candidates, forbidden):
    from shuttle_render import find_leaked_backgrounds

    return find_leaked_backgrounds(candidates, forbidden)


class CommittedChallengeManifestTests(unittest.TestCase):
    """What is actually shipped, checked against the frozen core."""

    @classmethod
    def setUpClass(cls) -> None:
        if not CHALLENGE_MANIFEST.is_file():
            raise unittest.SkipTest("the challenge manifest has not been generated yet")
        with CHALLENGE_MANIFEST.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.core_rows = []
        if CORE_MANIFEST.is_file():
            with CORE_MANIFEST.open(newline="", encoding="utf-8") as handle:
                cls.core_rows = list(csv.DictReader(handle))

    def test_every_shipped_row_is_challenge_test(self) -> None:
        self.assertTrue(self.rows)
        self.assertEqual({row["split"] for row in self.rows}, {CHALLENGE_SPLIT})

    def test_the_schema_is_the_core_schema(self) -> None:
        """The evaluator and the audit run over this file unchanged or not at all."""
        with CORE_MANIFEST.open(newline="", encoding="utf-8") as handle:
            core_header = next(csv.reader(handle))
        with CHALLENGE_MANIFEST.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, core_header)

    def test_no_scene_is_shared_with_the_frozen_core(self) -> None:
        core_scenes = {row["background"] for row in self.core_rows}
        challenge_scenes = {row["background"] for row in self.rows}
        self.assertTrue(challenge_scenes)
        self.assertEqual(core_scenes & challenge_scenes, set())

    def test_no_image_content_is_shared_with_the_frozen_core(self) -> None:
        import hashlib

        def digest(row, manifest):
            return hashlib.sha256((manifest.parent / row["file"]).read_bytes()).hexdigest()

        core = {digest(row, CORE_MANIFEST) for row in self.core_rows}
        challenge = {digest(row, CHALLENGE_MANIFEST) for row in self.rows}
        self.assertEqual(len(challenge), len(self.rows))
        self.assertEqual(core & challenge, set())

    def test_the_shipped_rows_keep_their_pinned_columns(self) -> None:
        self.assertEqual(verify_measured_pins(self.rows, challenge_varied_columns()), [])

    def test_every_source_category_is_present_in_the_shipped_manifest(self) -> None:
        counts = Counter(row["sweep"] for row in self.rows)
        for sweep in CHALLENGE_SWEEPS:
            self.assertGreater(counts.get(sweep, 0), 0, sweep)


class TrainingGuardTests(unittest.TestCase):
    """Spec 08 section 4.2: the challenge pool never enters training."""

    def test_the_guard_names_both_held_out_splits(self) -> None:
        from src.perception.shuttle_detection.ultralytics_dataset import (
            HELD_OUT_SPLITS,
            held_out_reason,
        )

        self.assertIn(CHALLENGE_SPLIT, HELD_OUT_SPLITS)
        for spelling in ("challenge_test", "CHALLENGE TEST", "challenge-test", "ChallengeTest"):
            self.assertIsNotNone(held_out_reason(spelling), spelling)

    def test_the_builder_refuses_a_challenge_manifest(self) -> None:
        if not CHALLENGE_MANIFEST.is_file():
            raise unittest.SkipTest("the challenge manifest has not been generated yet")
        import tempfile

        from src.perception.shuttle_detection.ultralytics_dataset import (
            ForbiddenSplitError,
            write_dataset_yaml,
        )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ForbiddenSplitError) as caught:
                write_dataset_yaml(
                    temporary,
                    CHALLENGE_MANIFEST,
                    CHALLENGE_MANIFEST,
                    data_root="/",
                )
            self.assertIn(CHALLENGE_SPLIT, str(caught.exception))
            # Nothing may be written when the refusal fires.
            self.assertEqual(sorted(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
