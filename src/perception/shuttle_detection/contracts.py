"""Frozen data contracts for the shuttlecock dataset audit.

Why this module exists: the audit spans several steps (manifest reading, label
validation, leakage detection, distribution reporting) and a CLI. If those steps
each invented their own vocabulary for splits, source types and severities, the
reports they produce could not be compared with each other, and a PASS from one
step would say nothing about the others. The vocabulary is therefore declared once
here, straight from docs/superpowers/specs/01_DATA_ASSET_READINESS_SPEC.md
(sections 3 and 6), and every other module imports it.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_contracts.py -v
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

# Data pools required by spec section 3. "train"/"val" are the trainable pools;
# "fixed_core_test" never enters training (spec section 6) and "challenge_test"
# is held out for the three-level adjustment gate.
Split = Literal["train", "val", "fixed_core_test", "challenge_test"]
SPLITS: tuple[str, ...] = ("train", "val", "fixed_core_test", "challenge_test")

# Provenance of a sample, as required by spec section 7 (synthetic vs real).
SourceType = Literal["SYNTHETIC_3D", "REAL_IMAGE", "REAL_VIDEO", "NEGATIVE"]
SOURCE_TYPES: tuple[str, ...] = ("SYNTHETIC_3D", "REAL_IMAGE", "REAL_VIDEO", "NEGATIVE")

# An ERROR blocks training; a WARNING is recorded and does not change the exit code.
Severity = Literal["ERROR", "WARNING"]


@dataclass(frozen=True)
class DatasetRecord:
    """One auditable dataset sample.

    image_path / label_path are stored as strings so a record can be written into
    a report without dragging platform-specific Path objects along. label_path is
    None for hard negatives, which have no label file.
    """

    sample_id: str
    image_path: str
    label_path: str | None
    split: Split
    source_type: SourceType
    camera_id: str | None


@dataclass(frozen=True)
class AuditIssue:
    """One finding. sample_id may name a manifest-level pseudo sample."""

    sample_id: str
    code: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class AuditResult:
    """Aggregate verdict of an audit run."""

    passed: bool
    issue_count: int
    error_count: int

    @classmethod
    def from_issues(cls, issues: Iterable[AuditIssue]) -> "AuditResult":
        """Summarise an issue list: only ERRORs can fail an audit."""
        materialised = list(issues)
        error_count = sum(1 for issue in materialised if issue.severity == "ERROR")
        return cls(
            passed=error_count == 0,
            issue_count=len(materialised),
            error_count=error_count,
        )
