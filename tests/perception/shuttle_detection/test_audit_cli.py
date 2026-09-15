#!/usr/bin/env python3
"""The shuttle dataset audit CLI must be usable as the training gate.

Why this module exists: every training run in this project starts with the same
question -- is this dataset safe to train on? The CLI answers it with five
machine-readable files and an exit code, so a training script (or a human) can
refuse to start without reading prose. Exit code 0 therefore means one thing only:
the audit found zero ERRORs.

Run:
    python tests/perception/shuttle_detection/test_audit_cli.py -v
    pytest tests/perception/shuttle_detection/test_audit_cli.py -v
"""
from __future__ import annotations

import base64
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shuttle_detection.audit_dataset import (  # noqa: E402
    DISTRIBUTION_REPORT,
    INVENTORY_REPORT,
    LEAKAGE_REPORT,
    REPORT_FILES,
    SUMMARY_REPORT,
    main,
)

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
OTHER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

CLEANING_REPORT = "dataset_cleaning_report.csv"


def write_sample(root: Path, name: str, split: str, payload: bytes = TINY_PNG,
                 background: str = "bg_1", label: str = "0 0.500000 0.500000 0.200000 0.200000\n") -> dict:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "images" / (name + ".png")).write_bytes(payload)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    (root / split / "labels" / (name + ".txt")).write_text(label, encoding="utf-8")
    return {"file": name + ".png", "split": split, "source_type": "SYNTHETIC_3D", "background": background}


def write_manifest(path: Path, rows, fieldnames=("file", "split", "source_type", "background")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in columns})
    return path


def build_clean_dataset(root: Path) -> Path:
    train_row = write_sample(root, "train_00000", "train", TINY_PNG, "bg_train_1")
    val_row = write_sample(root, "val_00000", "val", OTHER_PNG, "bg_val_1")
    return write_manifest(root / "manifest_train.csv", [train_row, val_row])


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class MissingManifestTests(unittest.TestCase):
    def test_cli_returns_nonzero_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = main(
                [
                    "--manifest", str(root / "missing.csv"),
                    "--dataset-root", str(root),
                    "--out", str(root / "out"),
                ]
            )
            self.assertNotEqual(rc, 0)

    def test_missing_manifest_prints_a_message_instead_of_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = io.StringIO()
            with redirect_stderr(captured):
                rc = main(
                    [
                        "--manifest", str(root / "missing.csv"),
                        "--dataset-root", str(root),
                        "--out", str(root / "out"),
                    ]
                )
            self.assertNotEqual(rc, 0)
            self.assertIn("missing.csv", captured.getvalue())

    def test_missing_manifest_writes_no_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            main(
                [
                    "--manifest", str(root / "missing.csv"),
                    "--dataset-root", str(root),
                    "--out", str(out),
                ]
            )
            self.assertEqual(list(out.glob("*")) if out.exists() else [], [])


class ReportWritingTests(unittest.TestCase):
    def test_clean_dataset_exits_zero_and_writes_every_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
            )
            self.assertEqual(rc, 0)
            for name in REPORT_FILES:
                self.assertTrue((out / name).is_file(), name + " was not written")

    def test_nested_output_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "deep" / "nested" / "out"
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out / SUMMARY_REPORT).is_file())

    def test_inventory_has_one_row_per_manifest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)])
            rows = read_csv(out / INVENTORY_REPORT)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(row["sample_id"] for row in rows), ["train_00000", "val_00000"])
            self.assertEqual(sorted(row["split"] for row in rows), ["train", "val"])
            self.assertTrue(all(len(row["image_sha256"]) == 64 for row in rows))

    def test_summary_json_reports_counts_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)])
            summary = json.loads((out / SUMMARY_REPORT).read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["n_samples"], 2)
            self.assertEqual(summary["n_manifests"], 1)
            self.assertEqual(summary["counts_by_split"], {"train": 1, "val": 1})
            self.assertTrue(summary["dataset_version"].startswith("sha256:"))
            self.assertEqual(len(summary["manifests"][0]["sha256"]), 64)

    def test_distribution_report_contains_every_required_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)])
            rows = read_csv(out / DISTRIBUTION_REPORT)
            dimensions = {row["dimension"] for row in rows}
            self.assertEqual(
                dimensions,
                {
                    "split",
                    "source_type",
                    "camera_id",
                    "size_bucket",
                    "blur_bucket",
                    "pose_bucket",
                    "is_negative",
                },
            )
            self.assertTrue(all(row["count"].isdigit() for row in rows))
            for dimension in dimensions:
                counts = sum(int(row["count"]) for row in rows if row["dimension"] == dimension)
                self.assertEqual(counts, 2, dimension)

    def test_cleaning_report_is_a_csv_with_a_header_even_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)])
            text = (out / CLEANING_REPORT).read_text(encoding="utf-8")
            self.assertEqual(text.splitlines()[0], "sample_id,code,severity,detail")

    def test_leakage_report_is_a_csv_with_a_header_even_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)])
            rows = read_csv(out / LEAKAGE_REPORT)
            self.assertEqual(rows, [])
            self.assertEqual(
                (out / LEAKAGE_REPORT).read_text(encoding="utf-8").splitlines()[0],
                "code,key,splits,sample_ids,detail",
            )


class ErrorExitTests(unittest.TestCase):
    def test_a_bad_bbox_makes_the_cli_exit_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train", label="0 0.5 0.5 1.4 0.2\n")
            manifest = write_manifest(root / "manifest_train.csv", [row])
            out = root / "out"
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
            )
            self.assertNotEqual(rc, 0)
            summary = json.loads((out / SUMMARY_REPORT).read_text(encoding="utf-8"))
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["error_count"], 1)
            codes = {row["code"] for row in read_csv(out / CLEANING_REPORT)}
            self.assertIn("BBOX_OUT_OF_RANGE", codes)

    def test_a_missing_image_makes_the_cli_exit_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train")
            (root / "train" / "images" / "train_00000.png").unlink()
            manifest = write_manifest(root / "manifest_train.csv", [row])
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(root / "out")]
            )
            self.assertNotEqual(rc, 0)

    def test_warnings_alone_still_exit_zero(self) -> None:
        """A manifest without camera_id cannot describe cameras; that is a warning."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train")
            manifest = write_manifest(
                root / "manifest_train.csv", [row], fieldnames=("file", "split", "source_type")
            )
            out = root / "out"
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
            )
            self.assertEqual(rc, 0)
            summary = json.loads((out / SUMMARY_REPORT).read_text(encoding="utf-8"))
            self.assertGreater(summary["warning_count"], 0)
            self.assertEqual(summary["error_count"], 0)


class MultiManifestTests(unittest.TestCase):
    def test_cross_split_leakage_is_detected_and_fails_the_audit(self) -> None:
        """Leakage only exists between manifests, so the CLI must accept several."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_row = write_sample(root, "train_00000", "train", TINY_PNG, "bg_train_1")
            val_row = write_sample(root, "val_00000", "val", TINY_PNG, "bg_train_1")
            train_manifest = write_manifest(root / "manifest_train.csv", [train_row])
            val_manifest = write_manifest(root / "manifest_val.csv", [val_row])
            out = root / "out"
            rc = main(
                [
                    "--manifest", str(train_manifest),
                    "--manifest", str(val_manifest),
                    "--dataset-root", str(root),
                    "--out", str(out),
                ]
            )
            self.assertNotEqual(rc, 0)
            codes = {row["code"] for row in read_csv(out / LEAKAGE_REPORT)}
            self.assertIn("CROSS_SPLIT_DUPLICATE", codes)
            self.assertIn("SOURCE_GROUP_LEAKAGE", codes)
            summary = json.loads((out / SUMMARY_REPORT).read_text(encoding="utf-8"))
            self.assertEqual(summary["n_manifests"], 2)
            self.assertEqual(summary["n_samples"], 2)
            self.assertEqual(summary["checklist"]["pool_isolation"], "FAIL")

    def test_one_manifest_argument_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(root / "out")]
            )
            self.assertEqual(rc, 0)


class CommandLineTests(unittest.TestCase):
    """The documented command line must work, not just the imported main()."""

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "shuttle_detection" / "audit_dataset.py"), *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

    def test_documented_command_line_exits_zero_on_a_clean_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            out = root / "out"
            result = self.run_cli(
                "--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / SUMMARY_REPORT).is_file())

    def test_documented_command_line_exits_non_zero_on_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train", label="0 0.5 0.5 1.4 0.2\n")
            manifest = write_manifest(root / "manifest_train.csv", [row])
            result = self.run_cli(
                "--manifest", str(manifest), "--dataset-root", str(root), "--out", str(root / "out")
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)


LATIN1_LABEL = b"0 0.500000 0.500000 0.200000 0.200000\n" + b"# caf\xe9\n"
LATIN1_MANIFEST = (
    b"file,split,source_type,background\n"
    b"train_00000.png,train,SYNTHETIC_3D,caf\xe9_bg.jpg\n"
)


class UndecodableInputTests(unittest.TestCase):
    """F3: an undecodable file must not stop the CLI from doing its job.

    Before the fix these inputs raised UnicodeDecodeError out of main(), which
    produced a traceback, no reports at all, and an exit code that described the
    crash rather than the dataset.
    """

    def test_non_utf8_label_still_writes_reports_and_exits_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train")
            (root / "train" / "labels" / "train_00000.txt").write_bytes(LATIN1_LABEL)
            manifest = write_manifest(root / "manifest_train.csv", [row])
            out = root / "out"
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
            )
            self.assertEqual(rc, 1)
            for name in REPORT_FILES:
                self.assertTrue((out / name).is_file(), name + " was not written")
            codes = {entry["code"] for entry in read_csv(out / CLEANING_REPORT)}
            self.assertIn("LABEL_NOT_UTF8", codes)

    def test_non_utf8_manifest_still_writes_reports_and_exits_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sample(root, "train_00000", "train")
            manifest = root / "manifest_train.csv"
            manifest.write_bytes(LATIN1_MANIFEST)
            out = root / "out"
            rc = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
            )
            self.assertEqual(rc, 1)
            for name in REPORT_FILES:
                self.assertTrue((out / name).is_file(), name + " was not written")
            codes = {entry["code"] for entry in read_csv(out / CLEANING_REPORT)}
            self.assertIn("MANIFEST_NOT_UTF8", codes)
            summary = json.loads((out / SUMMARY_REPORT).read_text(encoding="utf-8"))
            self.assertEqual(summary["n_samples"], 1)
            self.assertFalse(summary["passed"])

    def test_documented_command_line_does_not_traceback_on_non_utf8_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train")
            (root / "train" / "labels" / "train_00000.txt").write_bytes(LATIN1_LABEL)
            manifest = write_manifest(root / "manifest_train.csv", [row])
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "shuttle_detection" / "audit_dataset.py"),
                    "--manifest", str(manifest),
                    "--dataset-root", str(root),
                    "--out", str(root / "out"),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class OutputDestinationTests(unittest.TestCase):
    """F4: an unusable --out is an error message, not a traceback."""

    def test_out_pointing_at_an_existing_file_is_a_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            occupied = root / "not_a_directory.txt"
            occupied.write_text("occupied", encoding="utf-8")
            captured = io.StringIO()
            with redirect_stderr(captured):
                rc = main(
                    ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(occupied)]
                )
            self.assertEqual(rc, 2)
            self.assertIn(str(occupied), captured.getvalue())
            self.assertEqual(occupied.read_text(encoding="utf-8"), "occupied")

    def test_write_reports_rejects_a_file_destination_with_a_typed_error(self) -> None:
        """Library callers get a typed error rather than a bare FileExistsError."""
        from scripts.shuttle_detection.audit_dataset import ReportDestinationError, write_reports
        from src.perception.shuttle_detection.dataset_audit import audit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            report = audit_dataset([manifest], dataset_root=root)
            occupied = root / "file.txt"
            occupied.write_text("x", encoding="utf-8")
            with self.assertRaises(ReportDestinationError):
                write_reports(report, occupied)

    def test_documented_command_line_reports_a_bad_out_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            occupied = root / "not_a_directory.txt"
            occupied.write_text("occupied", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "shuttle_detection" / "audit_dataset.py"),
                    "--manifest", str(manifest),
                    "--dataset-root", str(root),
                    "--out", str(occupied),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class TrainingReadinessVisibilityTests(unittest.TestCase):
    """F1: the CLI's own one-line summary must not read as "safe to train"."""

    def test_a_mixed_audit_prints_trainable_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_row = write_sample(root, "train_00000", "train", TINY_PNG, "bg_train_1")
            core_row = write_sample(root, "core_00000", "fixed_core_test", OTHER_PNG, "bg_core_1")
            train_manifest = write_manifest(root / "manifest_train.csv", [train_row])
            core_manifest = write_manifest(root / "manifest_core.csv", [core_row])
            captured = io.StringIO()
            with redirect_stdout(captured):
                rc = main(
                    [
                        "--manifest", str(train_manifest),
                        "--manifest", str(core_manifest),
                        "--dataset-root", str(root),
                        "--out", str(root / "out"),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("trainable=False", captured.getvalue())

    def test_a_trainable_audit_prints_trainable_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            captured = io.StringIO()
            with redirect_stdout(captured):
                rc = main(
                    ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(root / "out")]
                )
            self.assertEqual(rc, 0)
            self.assertIn("trainable=True", captured.getvalue())


def build_leaky_pair(root: Path) -> list[Path]:
    """Two pools holding byte-identical content, with disjoint source groups.

    The backgrounds are deliberately different: identical images plus a shared
    background would leak twice, and the --no-hash-images test measures exactly one
    of those two checks.
    """
    train_row = write_sample(root, "train_00000", "train", TINY_PNG, "bg_train_1")
    val_row = write_sample(root, "val_00000", "val", TINY_PNG, "bg_val_1")
    return [
        write_manifest(root / "manifest_train.csv", [train_row]),
        write_manifest(root / "manifest_val.csv", [val_row]),
    ]


def read_summary(out: Path) -> dict:
    return json.loads((out / SUMMARY_REPORT).read_text(encoding="utf-8"))


class CliOptionCoverageTests(unittest.TestCase):
    """F5: options and verdicts that no assertion used to constrain.

    Each test here is written so that removing or ignoring the behaviour it names
    makes it fail, rather than merely exercising the code path.
    """

    def test_no_hash_images_disables_duplicate_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = build_leaky_pair(root)
            arguments = []
            for manifest in manifests:
                arguments += ["--manifest", str(manifest)]

            detected = root / "out_detected"
            rc_detected = main(
                [*arguments, "--dataset-root", str(root), "--out", str(detected)]
            )
            self.assertEqual(rc_detected, 1)
            codes = {row["code"] for row in read_csv(detected / LEAKAGE_REPORT)}
            self.assertIn("CROSS_SPLIT_DUPLICATE", codes)

            skipped = root / "out_skipped"
            rc_skipped = main(
                [*arguments, "--dataset-root", str(root), "--out", str(skipped), "--no-hash-images"]
            )
            self.assertEqual(rc_skipped, 0)
            self.assertEqual(read_csv(skipped / LEAKAGE_REPORT), [])
            summary = read_summary(skipped)
            self.assertEqual(summary["checks"]["cross_split_duplicate_content"], "NOT_RUN")
            self.assertEqual(summary["checklist"]["pool_isolation"], "NOT_RUN")
            self.assertTrue(
                all(row["image_sha256"] == "" for row in read_csv(skipped / INVENTORY_REPORT))
            )
            self.assertNotEqual(
                read_summary(detected)["dataset_version"], summary["dataset_version"]
            )

    def test_no_decode_images_disables_the_corrupt_image_check(self) -> None:
        try:
            import PIL  # noqa: F401
        except Exception:  # pragma: no cover - Pillow is optional
            self.skipTest("Pillow not installed: the check cannot run here at all")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train")
            manifest = write_manifest(root / "manifest_train.csv", [row])
            (root / "train" / "images" / "train_00000.png").write_bytes(b"not an image at all")

            decoded = root / "out_decoded"
            rc_decoded = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(decoded)]
            )
            self.assertEqual(rc_decoded, 1)
            codes = {entry["code"] for entry in read_csv(decoded / CLEANING_REPORT)}
            self.assertIn("IMAGE_UNREADABLE", codes)

            skipped = root / "out_skipped"
            rc_skipped = main(
                [
                    "--manifest", str(manifest),
                    "--dataset-root", str(root),
                    "--out", str(skipped),
                    "--no-decode-images",
                ]
            )
            self.assertEqual(rc_skipped, 0)
            skipped_codes = {entry["code"] for entry in read_csv(skipped / CLEANING_REPORT)}
            self.assertNotIn("IMAGE_UNREADABLE", skipped_codes)
            summary = read_summary(skipped)
            self.assertEqual(summary["checks"]["corrupt_image_decode"], "NOT_RUN")
            self.assertEqual(summary["checklist"]["data_cleaning"], "NOT_RUN")

    def test_expected_class_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = write_sample(root, "train_00000", "train", label="1 0.500000 0.500000 0.200000 0.200000\n")
            manifest = write_manifest(root / "manifest_train.csv", [row])

            rc_default = main(
                ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(root / "out_default")]
            )
            self.assertEqual(rc_default, 1)
            codes = {entry["code"] for entry in read_csv(root / "out_default" / CLEANING_REPORT)}
            self.assertIn("WRONG_CLASS", codes)

            rc_matching = main(
                [
                    "--manifest", str(manifest),
                    "--dataset-root", str(root),
                    "--out", str(root / "out_matching"),
                    "--expected-class", "1",
                ]
            )
            self.assertEqual(rc_matching, 0)
            matching_codes = {
                entry["code"] for entry in read_csv(root / "out_matching" / CLEANING_REPORT)
            }
            self.assertNotIn("WRONG_CLASS", matching_codes)

    def test_missing_manifest_exits_with_the_usage_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = io.StringIO()
            with redirect_stderr(captured):
                rc = main(
                    [
                        "--manifest", str(root / "missing.csv"),
                        "--dataset-root", str(root),
                        "--out", str(root / "out"),
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertTrue(captured.getvalue().strip())

    def test_headerless_manifest_exits_with_the_usage_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest_train.csv"
            manifest.write_bytes(b"")
            out = root / "out"
            captured = io.StringIO()
            with redirect_stderr(captured):
                rc = main(
                    ["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]
                )
            self.assertEqual(rc, 2)
            self.assertFalse(out.exists())
            self.assertTrue(captured.getvalue().strip())

    def test_data_cleaning_reports_pass_fail_and_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_manifest = build_clean_dataset(root)
            clean_out = root / "out_clean"
            self.assertEqual(
                main(["--manifest", str(clean_manifest), "--dataset-root", str(root), "--out", str(clean_out)]),
                0,
            )
            self.assertEqual(read_summary(clean_out)["checklist"]["data_cleaning"], "PASS")

            bad_out = root / "out_bad"
            bad_row = write_sample(root, "bad_00000", "train", label="0 0.5 0.5 1.4 0.2\n")
            bad_manifest = write_manifest(root / "manifest_bad.csv", [bad_row])
            self.assertNotEqual(
                main(["--manifest", str(bad_manifest), "--dataset-root", str(root), "--out", str(bad_out)]),
                0,
            )
            self.assertEqual(read_summary(bad_out)["checklist"]["data_cleaning"], "FAIL")

            not_run_out = root / "out_not_run"
            self.assertEqual(
                main(
                    [
                        "--manifest", str(clean_manifest),
                        "--dataset-root", str(root),
                        "--out", str(not_run_out),
                        "--no-decode-images",
                    ]
                ),
                0,
            )
            self.assertEqual(read_summary(not_run_out)["checklist"]["data_cleaning"], "NOT_RUN")

    def test_dataset_version_is_stable_for_unchanged_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            first = root / "out_first"
            second = root / "out_second"
            for out in (first, second):
                self.assertEqual(
                    main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(out)]),
                    0,
                )
            self.assertEqual(
                read_summary(first)["dataset_version"], read_summary(second)["dataset_version"]
            )

    def test_dataset_version_changes_when_the_manifest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            before = root / "out_before"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(before)])
            version_before = read_summary(before)["dataset_version"]

            rows = read_csv(manifest)
            rows[0]["background"] = "bg_renamed"
            write_manifest(manifest, rows)
            after = root / "out_after"
            main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(after)])
            self.assertNotEqual(version_before, read_summary(after)["dataset_version"])

    def test_the_printed_summary_states_the_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_clean_dataset(root)
            captured = io.StringIO()
            with redirect_stdout(captured):
                main(["--manifest", str(manifest), "--dataset-root", str(root), "--out", str(root / "out")])
            self.assertIn(read_summary(root / "out")["dataset_version"], captured.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
