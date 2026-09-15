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


if __name__ == "__main__":
    unittest.main(verbosity=2)
