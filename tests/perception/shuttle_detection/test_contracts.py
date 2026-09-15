#!/usr/bin/env python3
"""Contracts for the shuttle dataset audit.

Why this module exists: every later audit step (label checks, leakage checks,
distribution reports) must agree on one vocabulary for splits, source types and
issue severities. Publishing that vocabulary as frozen dataclasses keeps the CLI,
the tests and the reports from drifting apart.

Run:
    python tests/perception/shuttle_detection/test_contracts.py -v
    pytest tests/perception/shuttle_detection/test_contracts.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.contracts import (  # noqa: E402
    SOURCE_TYPES,
    SPLITS,
    AuditIssue,
    AuditResult,
    DatasetRecord,
)


class DatasetRecordTests(unittest.TestCase):
    def test_dataset_record_keeps_source_and_split(self) -> None:
        r = DatasetRecord(
            sample_id="train_00001",
            image_path="images/train_00001.png",
            label_path="labels/train_00001.txt",
            split="train",
            source_type="SYNTHETIC_3D",
            camera_id=None,
        )
        self.assertEqual(r.split, "train")
        self.assertEqual(r.source_type, "SYNTHETIC_3D")

    def test_dataset_record_is_frozen(self) -> None:
        """A record is evidence: nothing downstream may rewrite it in place."""
        r = DatasetRecord(
            sample_id="s1",
            image_path="images/s1.jpg",
            label_path=None,
            split="val",
            source_type="NEGATIVE",
            camera_id="left",
        )
        with self.assertRaises(Exception):
            r.split = "train"  # type: ignore[misc]

    def test_dataset_record_accepts_a_missing_label_path(self) -> None:
        """Hard negatives have no label file; that is legal, not a type error."""
        r = DatasetRecord(
            sample_id="n1",
            image_path="images/n1.jpg",
            label_path=None,
            split="challenge_test",
            source_type="NEGATIVE",
            camera_id=None,
        )
        self.assertIsNone(r.label_path)

    def test_dataset_record_accepts_a_left_or_right_camera_id(self) -> None:
        for camera_id in ("left", "right", None):
            r = DatasetRecord(
                sample_id="f1",
                image_path="images/f1.jpg",
                label_path=None,
                split="fixed_core_test",
                source_type="REAL_VIDEO",
                camera_id=camera_id,
            )
            self.assertEqual(r.camera_id, camera_id)


class AuditIssueTests(unittest.TestCase):
    def test_issue_carries_sample_code_severity_and_detail(self) -> None:
        issue = AuditIssue(
            sample_id="train_00001",
            code="BBOX_OUT_OF_RANGE",
            severity="ERROR",
            detail="line=1",
        )
        self.assertEqual(issue.sample_id, "train_00001")
        self.assertEqual(issue.code, "BBOX_OUT_OF_RANGE")
        self.assertEqual(issue.severity, "ERROR")
        self.assertEqual(issue.detail, "line=1")

    def test_issue_is_frozen(self) -> None:
        issue = AuditIssue(sample_id="s1", code="LABEL_MISSING", severity="ERROR", detail="")
        with self.assertRaises(Exception):
            issue.code = "OTHER"  # type: ignore[misc]


class AuditResultTests(unittest.TestCase):
    def test_result_reports_counts_and_pass_state(self) -> None:
        result = AuditResult(passed=True, issue_count=0, error_count=0)
        self.assertTrue(result.passed)
        self.assertEqual(result.issue_count, 0)
        self.assertEqual(result.error_count, 0)

    def test_from_issues_passes_when_only_warnings_were_found(self) -> None:
        result = AuditResult.from_issues(
            [
                AuditIssue("s1", "MANIFEST_COLUMN_MISSING", "WARNING", "camera_id"),
                AuditIssue("s2", "MANIFEST_COLUMN_MISSING", "WARNING", "camera_id"),
            ]
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.issue_count, 2)
        self.assertEqual(result.error_count, 0)

    def test_from_issues_fails_when_an_error_was_found(self) -> None:
        result = AuditResult.from_issues(
            [
                AuditIssue("s1", "MANIFEST_COLUMN_MISSING", "WARNING", "camera_id"),
                AuditIssue("s2", "IMAGE_MISSING", "ERROR", "images/s2.jpg"),
            ]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.issue_count, 2)
        self.assertEqual(result.error_count, 1)

    def test_from_issues_on_an_empty_list_passes(self) -> None:
        result = AuditResult.from_issues([])
        self.assertTrue(result.passed)
        self.assertEqual(result.issue_count, 0)


class VocabularyTests(unittest.TestCase):
    def test_split_vocabulary_matches_the_data_pool_spec(self) -> None:
        self.assertEqual(
            tuple(SPLITS), ("train", "val", "fixed_core_test", "challenge_test")
        )

    def test_source_type_vocabulary_matches_the_data_pool_spec(self) -> None:
        self.assertEqual(
            tuple(SOURCE_TYPES), ("SYNTHETIC_3D", "REAL_IMAGE", "REAL_VIDEO", "NEGATIVE")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
