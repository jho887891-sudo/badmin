#!/usr/bin/env python3
"""Audit a shuttlecock detection dataset and write the readiness reports.

Usage:
    python scripts/shuttle_detection/audit_dataset.py \
        --manifest manifest_train.csv \
        --manifest manifest_val.csv \
        --dataset-root outputs/shuttle_capability/train_data \
        --out outputs/shuttle_detection/dataset_audit

Exit codes:
    0  the audit found no ERROR (WARNINGs are reported, not fatal)
    1  the audit found at least one ERROR: do not train on this data
    2  the audit could not run (manifest missing or unreadable)

Writes into --out:
    dataset_inventory.csv
    dataset_cleaning_report.csv
    dataset_distribution_report.csv
    dataset_leakage_report.csv
    audit_summary.json

Passing the same --manifest flag more than once audits several pools together,
which is the only way to see leakage between train and val/test.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.dataset_audit import (  # noqa: E402
    BUCKET_DIMENSIONS,
    CLEANING_COLUMNS,
    DISTRIBUTION_COLUMNS,
    INVENTORY_COLUMNS,
    LEAKAGE_COLUMNS,
    DatasetAuditReport,
    ManifestError,
    audit_dataset,
)

EXIT_OK = 0
EXIT_AUDIT_FAILED = 1
EXIT_UNUSABLE_INPUT = 2

INVENTORY_REPORT = "dataset_inventory.csv"
CLEANING_REPORT = "dataset_cleaning_report.csv"
DISTRIBUTION_REPORT = "dataset_distribution_report.csv"
LEAKAGE_REPORT = "dataset_leakage_report.csv"
SUMMARY_REPORT = "audit_summary.json"

REPORT_FILES: tuple[str, ...] = (
    INVENTORY_REPORT,
    CLEANING_REPORT,
    DISTRIBUTION_REPORT,
    LEAKAGE_REPORT,
    SUMMARY_REPORT,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a shuttlecock detection dataset before training."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        metavar="CSV",
        help="manifest CSV to audit; repeat the flag to audit several pools together",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        metavar="DIR",
        help="directory that contains the split directories referenced by the manifests",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="directory the five reports are written into (created if missing)",
    )
    parser.add_argument(
        "--expected-class",
        type=int,
        default=0,
        help="class index every label must carry (detection task is nc=1, class 0)",
    )
    parser.add_argument(
        "--no-hash-images",
        action="store_true",
        help="skip image content hashing, which also disables cross-split duplicate detection",
    )
    parser.add_argument(
        "--no-decode-images",
        action="store_true",
        help="skip decoding image pixels (faster; no corrupt-image check)",
    )
    return parser.parse_args(argv)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reports(report: DatasetAuditReport, out_dir: Path | str) -> list[Path]:
    """Write the five reports and return their paths."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    inventory_path = destination / INVENTORY_REPORT
    cleaning_path = destination / CLEANING_REPORT
    distribution_path = destination / DISTRIBUTION_REPORT
    leakage_path = destination / LEAKAGE_REPORT
    summary_path = destination / SUMMARY_REPORT

    _write_csv(inventory_path, INVENTORY_COLUMNS, report.inventory)

    _write_csv(
        cleaning_path,
        CLEANING_COLUMNS,
        [
            {
                "sample_id": issue.sample_id,
                "code": issue.code,
                "severity": issue.severity,
                "detail": issue.detail,
            }
            for issue in report.issues
        ],
    )

    distribution_rows: list[dict[str, Any]] = []
    for dimension in BUCKET_DIMENSIONS:
        counts = report.distribution.get(dimension, {})
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            distribution_rows.append({"dimension": dimension, "value": value, "count": count})
    _write_csv(distribution_path, DISTRIBUTION_COLUMNS, distribution_rows)

    _write_csv(leakage_path, LEAKAGE_COLUMNS, report.leakage)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report.summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return [inventory_path, cleaning_path, distribution_path, leakage_path, summary_path]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_dataset(
            args.manifest,
            dataset_root=args.dataset_root,
            expected_class=args.expected_class,
            hash_images=not args.no_hash_images,
            decode_images=not args.no_decode_images,
        )
    except ManifestError as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT
    except OSError as error:
        print("ERROR: cannot read input: " + str(error), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    written = write_reports(report, args.out)
    summary = report.summary
    print(
        "audit_dataset: samples={0} errors={1} warnings={2} passed={3} version={4}".format(
            summary["n_samples"],
            summary["error_count"],
            summary["warning_count"],
            summary["passed"],
            summary["dataset_version"],
        )
    )
    for path in written:
        print("  wrote " + path.name)

    return EXIT_OK if report.result.error_count == 0 else EXIT_AUDIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
