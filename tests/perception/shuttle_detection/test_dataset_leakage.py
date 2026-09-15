#!/usr/bin/env python3
"""Split-leakage detection and distribution reporting for shuttlecock datasets.

Why this module exists: spec section 6 forbids the same real video segment, the
same synthetic sequence or the same source group from crossing train / val / test
boundaries, and the 400/120 P4-A split currently relies on the background pools
bg_train and bg_val being disjoint by construction. Nothing verified that except
the generator's own bookkeeping, so it is verified here from the manifests and the
pixel bytes themselves, and the size/blur/pose coverage of the set is quantified
at the same time (spec section 7).

Run:
    python tests/perception/shuttle_detection/test_dataset_leakage.py -v
    pytest tests/perception/shuttle_detection/test_dataset_leakage.py -v
"""
from __future__ import annotations

import base64
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.dataset_audit import (  # noqa: E402
    BUCKET_DIMENSIONS,
    audit_dataset,
    bucket_blur,
    bucket_pose,
    bucket_size,
    find_cross_split_duplicates,
    find_source_group_leakage,
    summarize_distribution,
)

REAL_DATA_DIR = ROOT / "outputs" / "shuttle_capability" / "train_data"
REAL_MANIFEST_TRAIN = REAL_DATA_DIR / "manifest_train.csv"
REAL_MANIFEST_VAL = REAL_DATA_DIR / "manifest_val.csv"

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
# A second, different 1x1 PNG so two samples can be made non-identical.
OTHER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class FindCrossSplitDuplicatesTests(unittest.TestCase):
    def test_same_content_cannot_cross_train_and_fixed_test(self) -> None:
        rows = [
            ("a", "train", "abc123"),
            ("b", "fixed_core_test", "abc123"),
        ]
        issues = find_cross_split_duplicates(rows)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "CROSS_SPLIT_DUPLICATE")

    def test_the_duplicate_is_an_error(self) -> None:
        issues = find_cross_split_duplicates([("a", "train", "h"), ("b", "val", "h")])
        self.assertEqual(issues[0].severity, "ERROR")

    def test_the_issue_names_both_splits_and_both_samples(self) -> None:
        issues = find_cross_split_duplicates([("a", "train", "h"), ("b", "val", "h")])
        self.assertIn("train", issues[0].detail)
        self.assertIn("val", issues[0].detail)
        self.assertIn("a", issues[0].sample_id)
        self.assertIn("b", issues[0].sample_id)

    def test_duplicates_inside_one_split_are_not_split_leakage(self) -> None:
        rows = [("a", "train", "h"), ("b", "train", "h")]
        self.assertEqual(find_cross_split_duplicates(rows), [])

    def test_distinct_content_produces_no_issues(self) -> None:
        rows = [("a", "train", "h1"), ("b", "val", "h2")]
        self.assertEqual(find_cross_split_duplicates(rows), [])

    def test_three_splits_sharing_content_produce_one_issue(self) -> None:
        rows = [("a", "train", "h"), ("b", "val", "h"), ("c", "challenge_test", "h")]
        self.assertEqual(len(find_cross_split_duplicates(rows)), 1)

    def test_empty_hashes_are_ignored(self) -> None:
        """Hashing can be switched off; that must not turn every row into a twin."""
        rows = [("a", "train", ""), ("b", "val", "")]
        self.assertEqual(find_cross_split_duplicates(rows), [])


class FindSourceGroupLeakageTests(unittest.TestCase):
    def test_a_source_group_may_not_cross_train_and_test(self) -> None:
        rows = [
            ("a", "train", "tbg_008.jpg"),
            ("b", "fixed_core_test", "tbg_008.jpg"),
        ]
        issues = find_source_group_leakage(rows)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "SOURCE_GROUP_LEAKAGE")
        self.assertEqual(issues[0].severity, "ERROR")

    def test_a_group_inside_one_split_is_fine(self) -> None:
        rows = [("a", "train", "bg1"), ("b", "train", "bg1")]
        self.assertEqual(find_source_group_leakage(rows), [])

    def test_disjoint_groups_produce_no_issues(self) -> None:
        rows = [("a", "train", "bg1"), ("b", "val", "bg2")]
        self.assertEqual(find_source_group_leakage(rows), [])

    def test_unknown_groups_are_ignored(self) -> None:
        rows = [("a", "train", ""), ("b", "val", "")]
        self.assertEqual(find_source_group_leakage(rows), [])


class BucketTests(unittest.TestCase):
    def test_size_buckets_follow_the_measured_p4a_range(self) -> None:
        self.assertEqual(bucket_size(2.449), "<4px")
        self.assertEqual(bucket_size(3.999), "<4px")
        self.assertEqual(bucket_size(4.0), "4-8px")
        self.assertEqual(bucket_size(8.944), "8-16px")
        self.assertEqual(bucket_size(16.0), "16-32px")
        self.assertEqual(bucket_size(32.863), ">=32px")

    def test_size_bucket_of_a_missing_value_is_unknown(self) -> None:
        self.assertEqual(bucket_size(None), "unknown")
        self.assertEqual(bucket_size(""), "unknown")
        self.assertEqual(bucket_size("not a number"), "unknown")

    def test_blur_buckets_follow_the_motion_px_column(self) -> None:
        self.assertEqual(bucket_blur(0.0), "sharp")
        self.assertEqual(bucket_blur(1.0), "slight")
        self.assertEqual(bucket_blur(2.0), "slight")
        self.assertEqual(bucket_blur(4.0), "moderate")
        self.assertEqual(bucket_blur(7.0), "heavy")

    def test_blur_bucket_of_a_missing_value_is_unknown(self) -> None:
        self.assertEqual(bucket_blur(None), "unknown")

    def test_pose_bucket_uses_the_largest_orientation_deviation(self) -> None:
        self.assertEqual(bucket_pose(0.0, 0.0, 0.0), "axis_aligned")
        self.assertEqual(bucket_pose(15.0, 0.0, 0.0), "axis_aligned")
        self.assertEqual(bucket_pose(0.0, 30.0, 0.0), "tilted")
        self.assertEqual(bucket_pose(0.0, 0.0, 90.0), "steep")

    def test_pose_bucket_folds_angles_into_one_half_turn(self) -> None:
        """-170 deg and +10 deg are the same orientation of an axisymmetric body."""
        self.assertEqual(bucket_pose(-170.0, 0.0, 0.0), bucket_pose(10.0, 0.0, 0.0))

    def test_pose_bucket_of_a_missing_value_is_unknown(self) -> None:
        self.assertEqual(bucket_pose(None, None, None), "unknown")


class SummarizeDistributionTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "split": "train",
                "source_type": "SYNTHETIC_HIFI_3D",
                "camera_id": "",
                "equiv_size_px": "10.0",
                "motion_px": "0.0",
                "yaw_deg": "0.0",
                "pitch_deg": "0.0",
                "roll_deg": "0.0",
                "is_negative": "False",
            },
            {
                "split": "val",
                "source_type": "REAL_IMAGE",
                "camera_id": "left",
                "equiv_size_px": "3.0",
                "motion_px": "7.0",
                "yaw_deg": "0.0",
                "pitch_deg": "90.0",
                "roll_deg": "0.0",
                "is_negative": "True",
            },
        ]

    def test_covers_every_required_dimension(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(list(summary), list(BUCKET_DIMENSIONS))

    def test_counts_by_split(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["split"], {"train": 1, "val": 1})

    def test_counts_by_canonical_source_type(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["source_type"], {"SYNTHETIC_3D": 1, "REAL_IMAGE": 1})

    def test_counts_by_camera_id_with_an_unspecified_bucket(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["camera_id"], {"unspecified": 1, "left": 1})

    def test_counts_by_size_bucket(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["size_bucket"], {"8-16px": 1, "<4px": 1})

    def test_counts_by_blur_bucket(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["blur_bucket"], {"sharp": 1, "heavy": 1})

    def test_counts_by_pose_bucket(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["pose_bucket"], {"axis_aligned": 1, "steep": 1})

    def test_counts_by_is_negative(self) -> None:
        summary = summarize_distribution(self.rows())
        self.assertEqual(summary["is_negative"], {"False": 1, "True": 1})

    def test_precomputed_bucket_columns_win(self) -> None:
        rows = [dict(self.rows()[0], size_bucket="custom", blur_bucket="custom", pose_bucket="custom")]
        summary = summarize_distribution(rows)
        self.assertEqual(summary["size_bucket"], {"custom": 1})
        self.assertEqual(summary["blur_bucket"], {"custom": 1})
        self.assertEqual(summary["pose_bucket"], {"custom": 1})

    def test_missing_optional_fields_do_not_crash(self) -> None:
        summary = summarize_distribution([{"split": "train", "source_type": "REAL_VIDEO"}])
        self.assertEqual(summary["split"], {"train": 1})
        self.assertEqual(summary["size_bucket"], {"unknown": 1})
        self.assertEqual(summary["camera_id"], {"unspecified": 1})
        self.assertEqual(summary["is_negative"], {"False": 1})

    def test_object_rows_are_accepted_as_well_as_mappings(self) -> None:
        class Row:
            split = "train"
            source_family = "SYNTHETIC_3D"
            camera_id = None
            is_negative = False
            size_bucket = "<4px"
            blur_bucket = "sharp"
            pose_bucket = "steep"

        summary = summarize_distribution([Row()])
        self.assertEqual(summary["split"], {"train": 1})
        self.assertEqual(summary["source_type"], {"SYNTHETIC_3D": 1})
        self.assertEqual(summary["is_negative"], {"False": 1})

    def test_empty_input_gives_empty_dimensions(self) -> None:
        summary = summarize_distribution([])
        self.assertEqual(list(summary), list(BUCKET_DIMENSIONS))
        self.assertTrue(all(counts == {} for counts in summary.values()))


class RealManifestDistributionTests(unittest.TestCase):
    """The reports must describe the data we actually have, not only fixtures."""

    @unittest.skipUnless(REAL_MANIFEST_TRAIN.exists(), "frozen P4-A manifest not present")
    def test_equivalence_pixel_sizes_are_bucketed_across_the_measured_range(self) -> None:
        from src.perception.shuttle_detection.dataset_audit import read_manifest

        rows, _ = read_manifest(REAL_MANIFEST_TRAIN)
        summary = summarize_distribution(rows)
        self.assertEqual(sum(summary["split"].values()), 400)
        self.assertEqual(sum(summary["size_bucket"].values()), 400)
        sized = sum(
            count for name, count in summary["size_bucket"].items() if name != "unknown"
        )
        self.assertEqual(sized, 400)

    @unittest.skipUnless(REAL_MANIFEST_TRAIN.exists(), "frozen P4-A manifest not present")
    def test_train_and_val_background_pools_are_disjoint(self) -> None:
        from src.perception.shuttle_detection.dataset_audit import read_manifest

        train_rows, _ = read_manifest(REAL_MANIFEST_TRAIN)
        val_rows, _ = read_manifest(REAL_MANIFEST_VAL)
        rows = [
            (row["file"], "train", row["background"]) for row in train_rows
        ] + [(row["file"], "val", row["background"]) for row in val_rows]
        self.assertEqual(find_source_group_leakage(rows), [])

    @unittest.skipUnless(REAL_MANIFEST_TRAIN.exists(), "frozen P4-A manifest not present")
    def test_the_two_frozen_manifests_share_no_sample_id(self) -> None:
        from src.perception.shuttle_detection.dataset_audit import read_manifest

        train_rows, _ = read_manifest(REAL_MANIFEST_TRAIN)
        val_rows, _ = read_manifest(REAL_MANIFEST_VAL)
        train_ids = {Path(row["file"]).stem for row in train_rows}
        val_ids = {Path(row["file"]).stem for row in val_rows}
        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(len(train_ids), 400)
        self.assertEqual(len(val_ids), 120)


def write_manifest(path: Path, rows, fieldnames=("file", "split", "source_type", "background")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(fieldnames)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in columns})
    return path


def write_sample(
    root: Path,
    name: str,
    split: str,
    payload: bytes = TINY_PNG,
    background: str = "bg_shared",
    label: str = "0 0.500000 0.500000 0.200000 0.200000\n",
    source_type: str = "SYNTHETIC_3D",
) -> dict:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "images" / (name + ".png")).write_bytes(payload)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    (root / split / "labels" / (name + ".txt")).write_text(label, encoding="utf-8")
    return {
        "file": name + ".png",
        "split": split,
        "source_type": source_type,
        "background": background,
    }


class AuditDatasetTests(unittest.TestCase):
    """The manifest-level audit must tie every check into one machine-readable report."""

    def build_two_pool_dataset(self, root: Path, train_payload=TINY_PNG, val_payload=OTHER_PNG,
                               train_background="bg_train_1", val_background="bg_val_1"):
        train_row = write_sample(root, "train_00000", "train", train_payload, train_background)
        val_row = write_sample(root, "val_00000", "val", val_payload, val_background)
        train_manifest = write_manifest(root / "manifest_train.csv", [train_row])
        val_manifest = write_manifest(root / "manifest_val.csv", [val_row])
        return train_manifest, val_manifest

    def test_missing_manifest_raises_a_typed_error(self) -> None:
        from src.perception.shuttle_detection.dataset_audit import ManifestMissingError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ManifestMissingError):
                audit_dataset([root / "missing.csv"], dataset_root=root)

    def test_isolated_pools_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root)
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertEqual(report.result.error_count, 0)
            self.assertTrue(report.result.passed)
            self.assertEqual(report.leakage, [])
            self.assertEqual(report.summary["checklist"]["pool_isolation"], "PASS")

    def test_cross_split_duplicate_content_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root, train_payload=TINY_PNG, val_payload=TINY_PNG)
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertFalse(report.result.passed)
            self.assertIn("CROSS_SPLIT_DUPLICATE", [issue.code for issue in report.issues])
            self.assertEqual(
                [row["code"] for row in report.leakage], ["CROSS_SPLIT_DUPLICATE"]
            )
            self.assertEqual(report.summary["checklist"]["pool_isolation"], "FAIL")

    def test_source_group_leakage_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(
                root, train_background="bg_shared", val_background="bg_shared"
            )
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertIn("SOURCE_GROUP_LEAKAGE", [issue.code for issue in report.issues])
            self.assertEqual([row["code"] for row in report.leakage], ["SOURCE_GROUP_LEAKAGE"])

    def test_inventory_has_one_row_per_manifest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root)
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertEqual(len(report.inventory), 2)
            self.assertEqual(report.summary["n_samples"], 2)
            self.assertEqual(report.summary["counts_by_split"], {"train": 1, "val": 1})
            self.assertEqual(
                sorted(row["sample_id"] for row in report.inventory),
                ["train_00000", "val_00000"],
            )

    def test_summary_carries_the_manifest_hash_and_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root)
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertEqual(len(report.summary["manifests"]), 2)
            for info in report.summary["manifests"]:
                self.assertEqual(len(info["sha256"]), 64)
                self.assertEqual(info["rows"], 1)
            self.assertTrue(report.summary["dataset_version"].startswith("sha256:"))

    def test_dataset_version_is_stable_across_manifest_order(self) -> None:
        """The version identifies the data, not the order the CLI happened to list."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root)
            forward = audit_dataset(list(manifests), dataset_root=root)
            backward = audit_dataset(list(reversed(manifests)), dataset_root=root)
            self.assertEqual(
                forward.summary["dataset_version"], backward.summary["dataset_version"]
            )

    def test_distribution_covers_every_required_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root)
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertEqual(list(report.distribution), list(BUCKET_DIMENSIONS))
            self.assertEqual(report.summary["distribution_dimensions"], list(BUCKET_DIMENSIONS))

    def test_labels_are_validated_through_the_manifest_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self.build_two_pool_dataset(root)
            write_sample(root, "train_00000", "train", TINY_PNG, "bg_train_1", label="0 0.5 0.5 1.4 0.2\n")
            report = audit_dataset(list(manifests), dataset_root=root)
            self.assertIn("BBOX_OUT_OF_RANGE", [issue.code for issue in report.issues])
            self.assertEqual(report.summary["checklist"]["label_integrity"], "FAIL")
            row = [r for r in report.inventory if r["sample_id"] == "train_00000"][0]
            self.assertEqual(row["error_count"], 1)

    def test_duplicate_sample_ids_across_manifests_are_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_manifest, val_manifest = self.build_two_pool_dataset(root)
            write_sample(root, "val_00000", "train", OTHER_PNG, "bg_val_1")
            write_manifest(
                train_manifest,
                [{"file": "val_00000.png", "split": "train", "source_type": "SYNTHETIC_3D", "background": "bg_train_1"}],
            )
            report = audit_dataset([train_manifest, val_manifest], dataset_root=root)
            self.assertIn("DUPLICATE_SAMPLE_ID", [issue.code for issue in report.issues])

    def test_a_manifest_without_a_group_column_warns_but_does_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train")
            manifest = write_manifest(
                root / "manifest_train.csv", [row], fieldnames=("file", "split", "source_type")
            )
            report = audit_dataset([manifest], dataset_root=root)
            codes = [issue.code for issue in report.issues]
            self.assertIn("SOURCE_GROUP_UNKNOWN", codes)
            self.assertEqual(report.result.error_count, 0)

    def test_a_manifest_without_rows_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root / "manifest_train.csv", [])
            report = audit_dataset([manifest], dataset_root=root)
            self.assertIn("MANIFEST_EMPTY", [issue.code for issue in report.issues])

    def test_qualified_source_type_is_reported_once_per_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                write_sample(root, "train_00000", "train", source_type="SYNTHETIC_HIFI_3D"),
                write_sample(root, "train_00001", "train", source_type="SYNTHETIC_HIFI_3D"),
            ]
            manifest = write_manifest(root / "manifest_train.csv", rows)
            report = audit_dataset([manifest], dataset_root=root)
            qualified = [i for i in report.issues if i.code == "SOURCE_TYPE_NON_CANONICAL"]
            self.assertEqual(len(qualified), 1)
            self.assertIn("2 sample(s)", qualified[0].detail)

    @unittest.skipUnless(REAL_MANIFEST_TRAIN.exists(), "frozen P4-A manifest not present")
    def test_frozen_p4a_train_and_val_pass_the_audit(self) -> None:
        """Evidence for the readiness checklist: the two frozen pools are isolated."""
        report = audit_dataset(
            [REAL_MANIFEST_TRAIN, REAL_MANIFEST_VAL], dataset_root=REAL_DATA_DIR
        )
        self.assertEqual(report.result.error_count, 0)
        self.assertTrue(report.result.passed)
        self.assertEqual(report.summary["n_samples"], 520)
        self.assertEqual(report.summary["counts_by_split"], {"train": 400, "val": 120})
        self.assertEqual(report.leakage, [])
        self.assertEqual(report.summary["checklist"]["pool_isolation"], "PASS")
        self.assertEqual(report.summary["checklist"]["label_integrity"], "PASS")
        # 2 manifests x (camera_id missing, is_negative missing, qualified source_type)
        self.assertEqual(report.summary["warning_count"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
