#!/usr/bin/env python3
"""Manifest and label validation for the shuttlecock dataset audit.

Why this module exists: the P4-A training set (400 train / 120 val) was generated
by a renderer, and the only evidence that a rendered sample is usable is its
manifest row plus its label file. A silently empty label, a bbox outside [0,1] or
a manifest row pointing at an image that was never written would be discovered
only after a wasted training run, so they are checked here, before training.

Run:
    python tests/perception/shuttle_detection/test_dataset_audit.py -v
    pytest tests/perception/shuttle_detection/test_dataset_audit.py -v
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
    ManifestError,
    ManifestMissingError,
    audit_sample,
    canonical_source_family,
    check_manifest_columns,
    find_duplicate_sample_ids,
    read_manifest,
    validate_yolo_label,
)

# Frozen P4-A artifacts, used read-only as an integration fixture. They are the
# real inputs of this audit, so the audit must cope with their actual headers.
REAL_DATA_DIR = ROOT / "outputs" / "shuttle_capability" / "train_data"
REAL_MANIFEST = REAL_DATA_DIR / "manifest_train.csv"

# 1x1 RGB PNG: minimal bytes that any image library can open.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

GOOD_LABEL = "0 0.500000 0.500000 0.200000 0.200000\n"


def write_image(path: Path, payload: bytes = TINY_PNG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def write_label(path: Path, text: str = GOOD_LABEL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_manifest(path: Path, rows, fieldnames=("file", "split", "source_type")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def build_sample(root: Path, name: str = "train_00000", split: str = "train", label: str = GOOD_LABEL):
    """Create the real on-disk layout: <root>/<split>/images and .../labels."""
    write_image(root / split / "images" / (name + ".png"))
    write_label(root / split / "labels" / (name + ".txt"), label)
    return {
        "file": name + ".png",
        "split": split,
        "source_type": "SYNTHETIC_3D",
    }


class ValidateYoloLabelTests(unittest.TestCase):
    def test_rejects_out_of_range_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.txt"
            p.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")
            issues = validate_yolo_label("s1", p, expected_class=0)
            self.assertTrue(any(i.code == "BBOX_OUT_OF_RANGE" for i in issues))

    def test_accepts_a_well_formed_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "ok.txt")
            self.assertEqual(validate_yolo_label("s1", p, expected_class=0), [])

    def test_missing_label_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            issues = validate_yolo_label("s1", Path(tmp) / "gone.txt", expected_class=0)
            self.assertEqual([i.code for i in issues], ["LABEL_MISSING"])
            self.assertEqual(issues[0].severity, "ERROR")

    def test_wrong_field_count_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "short.txt", "0 0.5 0.5 0.2\n")
            codes = [i.code for i in validate_yolo_label("s1", p, expected_class=0)]
            self.assertIn("LABEL_FORMAT", codes)

    def test_wrong_class_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "cls.txt", "1 0.5 0.5 0.2 0.2\n")
            codes = [i.code for i in validate_yolo_label("s1", p, expected_class=0)]
            self.assertIn("WRONG_CLASS", codes)

    def test_zero_width_box_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "zero.txt", "0 0.5 0.5 0.0 0.2\n")
            codes = [i.code for i in validate_yolo_label("s1", p, expected_class=0)]
            self.assertIn("BBOX_OUT_OF_RANGE", codes)

    def test_non_numeric_field_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "nan.txt", "0 0.5 abc 0.2 0.2\n")
            codes = [i.code for i in validate_yolo_label("s1", p, expected_class=0)]
            self.assertIn("LABEL_NON_NUMERIC", codes)

    def test_empty_label_is_a_legal_hard_negative(self) -> None:
        """Spec section 5: a genuinely hard sample must not be flagged as dirt."""
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "empty.txt", "")
            self.assertEqual(validate_yolo_label("s1", p, expected_class=0), [])

    def test_reports_the_offending_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = write_label(Path(tmp) / "two.txt", "0 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\n")
            issues = validate_yolo_label("s1", p, expected_class=0)
            self.assertEqual(len(issues), 1)
            self.assertIn("line=2", issues[0].detail)


class ReadManifestTests(unittest.TestCase):
    def test_reads_rows_and_fieldnames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(
                Path(tmp) / "manifest_train.csv",
                [{"file": "a.png", "split": "train", "source_type": "SYNTHETIC_3D"}],
            )
            rows, fieldnames = read_manifest(path)
            self.assertEqual(list(fieldnames), ["file", "split", "source_type"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["file"], "a.png")

    def test_missing_manifest_raises_a_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ManifestMissingError):
                read_manifest(Path(tmp) / "missing.csv")

    def test_manifest_missing_error_is_a_manifest_error(self) -> None:
        self.assertTrue(issubclass(ManifestMissingError, ManifestError))

    def test_missing_manifest_error_is_not_a_bare_file_not_found(self) -> None:
        """The CLI turns this into a clean non-zero exit instead of a traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ManifestError):
                read_manifest(Path(tmp) / "missing.csv")

    def test_header_only_manifest_returns_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp) / "manifest_train.csv", [])
            rows, fieldnames = read_manifest(path)
            self.assertEqual(rows, [])
            self.assertEqual(list(fieldnames), ["file", "split", "source_type"])


class CanonicalSourceFamilyTests(unittest.TestCase):
    def test_canonical_values_map_to_themselves(self) -> None:
        for value in ("SYNTHETIC_3D", "REAL_IMAGE", "REAL_VIDEO", "NEGATIVE"):
            self.assertEqual(canonical_source_family(value), value)

    def test_qualified_synthetic_value_maps_to_the_synthetic_family(self) -> None:
        """The frozen P4-A manifests say SYNTHETIC_HIFI_3D, not SYNTHETIC_3D."""
        self.assertEqual(canonical_source_family("SYNTHETIC_HIFI_3D"), "SYNTHETIC_3D")

    def test_unknown_value_has_no_family(self) -> None:
        self.assertIsNone(canonical_source_family("SOMETHING_ELSE"))

    def test_empty_value_has_no_family(self) -> None:
        self.assertIsNone(canonical_source_family(""))

    def test_matching_is_case_insensitive(self) -> None:
        self.assertEqual(canonical_source_family("synthetic_hifi_3d"), "SYNTHETIC_3D")


class CheckManifestColumnsTests(unittest.TestCase):
    def test_missing_optional_column_is_a_warning(self) -> None:
        issues = check_manifest_columns(["file", "split", "source_type", "background"])
        codes = {i.code for i in issues}
        self.assertIn("MANIFEST_COLUMN_MISSING", codes)
        self.assertTrue(all(i.severity == "WARNING" for i in issues))
        self.assertTrue(any("camera_id" in i.detail for i in issues))

    def test_missing_required_column_is_an_error(self) -> None:
        issues = check_manifest_columns(["split", "source_type"])
        errors = [i for i in issues if i.severity == "ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertIn("file", errors[0].detail)

    def test_complete_header_produces_no_issues(self) -> None:
        header = [
            "file",
            "split",
            "source_type",
            "camera_id",
            "is_negative",
            "background",
            "equiv_size_px",
            "target_px",
            "motion_px",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
            "pos_x_px",
            "pos_y_px",
            "imgsz",
        ]
        self.assertEqual(check_manifest_columns(header), [])

    @unittest.skipUnless(REAL_MANIFEST.exists(), "frozen P4-A manifest not present")
    def test_frozen_p4a_header_produces_no_errors(self) -> None:
        """The real manifests lack camera_id/is_negative; that must only warn."""
        _, fieldnames = read_manifest(REAL_MANIFEST)
        issues = check_manifest_columns(fieldnames)
        self.assertEqual([i for i in issues if i.severity == "ERROR"], [])
        self.assertTrue(issues)


class AuditSampleTests(unittest.TestCase):
    def test_accepts_a_complete_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertEqual(audit.issues, [])
            self.assertEqual(audit.box_count, 1)
            self.assertFalse(audit.is_negative)
            self.assertEqual(audit.sample_id, "train_00000")
            self.assertEqual(audit.camera_id, None)

    def test_missing_image_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            (root / "train" / "images" / "train_00000.png").unlink()
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertIn("IMAGE_MISSING", [i.code for i in audit.issues])
            self.assertEqual(
                [i.severity for i in audit.issues if i.code == "IMAGE_MISSING"], ["ERROR"]
            )

    def test_missing_label_is_an_error_for_a_positive_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            (root / "train" / "labels" / "train_00000.txt").unlink()
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            missing = [i for i in audit.issues if i.code == "LABEL_MISSING"]
            self.assertEqual([i.severity for i in missing], ["ERROR"])

    def test_missing_label_is_only_a_warning_for_a_hard_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            row["source_type"] = "NEGATIVE"
            (root / "train" / "labels" / "train_00000.txt").unlink()
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            missing = [i for i in audit.issues if i.code == "LABEL_MISSING"]
            self.assertEqual([i.severity for i in missing], ["WARNING"])
            self.assertTrue(audit.is_negative)

    def test_empty_label_file_marks_a_negative_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root, label="")
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertEqual(audit.box_count, 0)
            self.assertTrue(audit.is_negative)

    def test_bad_bbox_is_reported_through_the_sample_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root, label="0 0.5 0.5 1.4 0.2\n")
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertIn("BBOX_OUT_OF_RANGE", [i.code for i in audit.issues])

    def test_illegal_split_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            row["split"] = "training"
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            codes = [i.code for i in audit.issues]
            self.assertIn("SPLIT_ILLEGAL", codes)

    def test_illegal_source_type_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            row["source_type"] = "MYSTERY_SOURCE"
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            errors = [i for i in audit.issues if i.code == "SOURCE_TYPE_ILLEGAL"]
            self.assertEqual([i.severity for i in errors], ["ERROR"])

    def test_qualified_synthetic_source_type_only_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            row["source_type"] = "SYNTHETIC_HIFI_3D"
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            qualified = [i for i in audit.issues if i.code == "SOURCE_TYPE_NON_CANONICAL"]
            self.assertEqual([i.severity for i in qualified], ["WARNING"])
            self.assertEqual(audit.source_family, "SYNTHETIC_3D")

    def test_split_directory_disagreement_is_an_error(self) -> None:
        """A train row that resolves under val/ would silently leak the split."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "val" / "images" / "train_00000.png")
            write_label(root / "val" / "labels" / "train_00000.txt")
            row = {"file": "train_00000.png", "split": "train", "source_type": "SYNTHETIC_3D"}
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertIn("SPLIT_MISMATCH", [i.code for i in audit.issues])

    def test_empty_image_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            write_image(root / "train" / "images" / "train_00000.png", b"")
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertIn("IMAGE_EMPTY", [i.code for i in audit.issues])

    def test_corrupt_image_bytes_are_an_error_when_a_decoder_is_available(self) -> None:
        try:
            import PIL  # noqa: F401
        except Exception:  # pragma: no cover - Pillow is optional
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            write_image(root / "train" / "images" / "train_00000.png", b"not an image at all")
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertIn("IMAGE_UNREADABLE", [i.code for i in audit.issues])

    def test_manifest_row_without_a_file_column_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = audit_sample(
                {"file": "", "split": "train", "source_type": "SYNTHETIC_3D"},
                dataset_root=root,
                manifest_dir=root,
            )
            self.assertIn("MANIFEST_ROW_INCOMPLETE", [i.code for i in audit.issues])

    def test_camera_id_is_taken_from_the_manifest_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = build_sample(root)
            row["camera_id"] = "left"
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertEqual(audit.camera_id, "left")

    def test_images_next_to_the_manifest_are_found(self) -> None:
        """Layout variant: <root>/images/... with the manifest beside it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "images" / "solo.png")
            write_label(root / "labels" / "solo.txt")
            row = {"file": "solo.png", "split": "train", "source_type": "REAL_IMAGE"}
            audit = audit_sample(row, dataset_root=root, manifest_dir=root)
            self.assertEqual([i for i in audit.issues if i.severity == "ERROR"], [])


class FindDuplicateSampleIdsTests(unittest.TestCase):
    def test_detects_a_repeated_sample_id(self) -> None:
        issues = find_duplicate_sample_ids([("s1", "a.csv"), ("s1", "b.csv"), ("s2", "a.csv")])
        self.assertEqual([i.code for i in issues], ["DUPLICATE_SAMPLE_ID"])
        self.assertEqual(issues[0].severity, "ERROR")
        self.assertEqual(issues[0].sample_id, "s1")

    def test_unique_sample_ids_produce_no_issues(self) -> None:
        self.assertEqual(find_duplicate_sample_ids([("s1", "a.csv"), ("s2", "a.csv")]), [])

    def test_empty_input_produces_no_issues(self) -> None:
        self.assertEqual(find_duplicate_sample_ids([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
