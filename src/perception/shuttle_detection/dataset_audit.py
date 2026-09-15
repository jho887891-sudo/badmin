"""Manifest, label, leakage and distribution auditing for shuttlecock datasets.

Why this module exists: the detection dataset is produced by a renderer and by
frame extraction, so nothing in the pipeline guarantees that a manifest row points
at an image that exists, that its label is a legal YOLO row, or that the train and
test pools are disjoint. This module is the single place that answers those
questions from the manifests themselves -- it embeds no dataset path of its own --
and it fails loudly (ERROR) only for defects that would corrupt a training run.
Findings that are merely unusual for the current data, such as a manifest written
without a camera_id column, are WARNINGs so the audit still runs on real data.

Layouts understood for the frozen P4-A set and its successors:
    <root>/<split>/images/<file>  +  <root>/<split>/labels/<stem>.txt
    <root>/<split>/<file>         +  <root>/<split>/<stem>.txt
    <root>/images/<file>          +  <root>/<stem>.txt
with <manifest_dir> accepted as an alternative base directory.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_dataset_audit.py -v
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .contracts import SPLITS, SOURCE_TYPES, AuditIssue

# Columns whose absence makes the manifest unusable rather than merely terse.
REQUIRED_COLUMNS: tuple[str, ...] = ("file", "split", "source_type")

# Columns the audit can derive distribution statistics from. Missing ones are
# reported as WARNINGs: the frozen P4-A manifests have no camera_id and no
# is_negative column, and refusing to audit them would be worse than reporting it.
OPTIONAL_COLUMNS: tuple[str, ...] = (
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
)

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Codes emitted by this module. Kept as constants so reports and tests agree.
CODE_MANIFEST_COLUMN_MISSING = "MANIFEST_COLUMN_MISSING"
CODE_MANIFEST_ROW_INCOMPLETE = "MANIFEST_ROW_INCOMPLETE"
CODE_MANIFEST_EMPTY = "MANIFEST_EMPTY"
CODE_LABEL_MISSING = "LABEL_MISSING"
CODE_LABEL_FORMAT = "LABEL_FORMAT"
CODE_LABEL_NON_NUMERIC = "LABEL_NON_NUMERIC"
CODE_WRONG_CLASS = "WRONG_CLASS"
CODE_BBOX_OUT_OF_RANGE = "BBOX_OUT_OF_RANGE"
CODE_IMAGE_MISSING = "IMAGE_MISSING"
CODE_IMAGE_EMPTY = "IMAGE_EMPTY"
CODE_IMAGE_UNREADABLE = "IMAGE_UNREADABLE"
CODE_IMAGE_SIZE_MISMATCH = "IMAGE_SIZE_MISMATCH"
CODE_SPLIT_ILLEGAL = "SPLIT_ILLEGAL"
CODE_SPLIT_MISMATCH = "SPLIT_MISMATCH"
CODE_SOURCE_TYPE_ILLEGAL = "SOURCE_TYPE_ILLEGAL"
CODE_SOURCE_TYPE_NON_CANONICAL = "SOURCE_TYPE_NON_CANONICAL"
CODE_DUPLICATE_SAMPLE_ID = "DUPLICATE_SAMPLE_ID"
CODE_CROSS_SPLIT_DUPLICATE = "CROSS_SPLIT_DUPLICATE"
CODE_SOURCE_GROUP_LEAKAGE = "SOURCE_GROUP_LEAKAGE"
CODE_COLUMN_VALUE_UNPARSEABLE = "COLUMN_VALUE_UNPARSEABLE"

# A manifest-level finding has no single sample to blame.
MANIFEST_SCOPE = "<manifest>"

_TRUTHY = {"1", "true", "yes", "y", "t"}


class ManifestError(RuntimeError):
    """The manifest cannot be audited at all (missing or headerless)."""


class ManifestMissingError(ManifestError):
    """The manifest path does not exist."""


def canonical_source_family(source_type: str) -> str | None:
    """Map a manifest source_type value onto the spec vocabulary.

    The frozen P4-A manifests record SYNTHETIC_HIFI_3D, which is a qualified form
    of the spec's SYNTHETIC_3D. Treating the qualifier as an illegal value would
    block the audit of the only dataset we have, so qualified values keep their
    family and are reported separately as WARNINGs.
    """
    key = (source_type or "").strip().upper()
    if not key:
        return None
    if key in SOURCE_TYPES:
        return key
    if key.startswith("SYNTHETIC"):
        return "SYNTHETIC_3D"
    if key.startswith("REAL_VIDEO"):
        return "REAL_VIDEO"
    if key.startswith("REAL_IMAGE"):
        return "REAL_IMAGE"
    if key.startswith("REAL"):
        return "REAL_IMAGE"
    if "NEGATIVE" in key:
        return "NEGATIVE"
    return None


def read_manifest(path: Path | str) -> tuple[list[dict[str, str]], list[str]]:
    """Read a manifest into plain row dicts, plus its header.

    csv is used instead of pandas on purpose: the audit must run in environments
    (including the remote training host) where only the standard library exists.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestMissingError("manifest not found: " + str(manifest_path))
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestError("manifest has no header row: " + str(manifest_path))
        fieldnames = [name.strip() for name in reader.fieldnames]
        rows = [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, fieldnames


def validate_yolo_label(sample_id: str, path: Path, expected_class: int = 0) -> list[AuditIssue]:
    """Validate one YOLO label file against the detection task (nc=1, class 0).

    An empty file is legal: it is a hard negative. A missing file is not, because
    a missing label usually means a crashed renderer rather than a negative.
    """
    issues: list[AuditIssue] = []
    label_path = Path(path)
    if not label_path.exists():
        return [AuditIssue(sample_id, CODE_LABEL_MISSING, "ERROR", str(label_path))]
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                AuditIssue(sample_id, CODE_LABEL_FORMAT, "ERROR", "line=" + str(line_no))
            )
            continue
        try:
            cls, xc, yc, w, h = (float(value) for value in parts)
        except ValueError:
            issues.append(
                AuditIssue(
                    sample_id,
                    CODE_LABEL_NON_NUMERIC,
                    "ERROR",
                    "line=" + str(line_no) + " raw=" + line.strip(),
                )
            )
            continue
        if int(cls) != expected_class:
            issues.append(
                AuditIssue(
                    sample_id, CODE_WRONG_CLASS, "ERROR", "line=" + str(line_no) + " class=" + parts[0]
                )
            )
        if not all(0.0 <= value <= 1.0 for value in (xc, yc, w, h)) or w <= 0 or h <= 0:
            issues.append(
                AuditIssue(
                    sample_id, CODE_BBOX_OUT_OF_RANGE, "ERROR", "line=" + str(line_no)
                )
            )
    return issues


def check_manifest_columns(fieldnames: Sequence[str]) -> list[AuditIssue]:
    """Report missing required columns (ERROR) and missing optional ones (WARNING)."""
    present = {name.strip() for name in fieldnames}
    issues: list[AuditIssue] = []
    for column in REQUIRED_COLUMNS:
        if column not in present:
            issues.append(
                AuditIssue(
                    MANIFEST_SCOPE,
                    CODE_MANIFEST_COLUMN_MISSING,
                    "ERROR",
                    "required column missing: " + column,
                )
            )
    for column in OPTIONAL_COLUMNS:
        if column not in present:
            issues.append(
                AuditIssue(
                    MANIFEST_SCOPE,
                    CODE_MANIFEST_COLUMN_MISSING,
                    "WARNING",
                    "optional column missing: " + column,
                )
            )
    return issues


@dataclass(frozen=True)
class SampleAudit:
    """Per-sample audit outcome, reused as the dataset inventory row."""

    sample_id: str
    split: str
    source_type: str
    source_family: str
    camera_id: str | None
    image_path: str | None
    label_path: str | None
    box_count: int
    is_negative: bool
    issues: list[AuditIssue] = field(default_factory=list)


def _candidate_image_dirs(
    dataset_root: Path, manifest_dir: Path, split: str
) -> list[Path]:
    """Directories searched for a manifest row's image, declared split first.

    The other pools are searched last and only so that a row whose file sits in
    the wrong pool is reported as SPLIT_MISMATCH instead of a bare IMAGE_MISSING:
    that shape is leakage, and calling it a missing file would hide it.
    """
    directories = [
        dataset_root / split / "images",
        manifest_dir / split / "images",
        dataset_root / split,
        manifest_dir / split,
        dataset_root / "images",
        dataset_root,
        manifest_dir,
    ]
    for other in SPLITS:
        if other == split:
            continue
        directories.append(dataset_root / other / "images")
        directories.append(manifest_dir / other / "images")
    return directories


def _resolve_image_path(
    dataset_root: Path, manifest_dir: Path, split: str, file_name: str
) -> Path | None:
    names = [file_name]
    if not Path(file_name).suffix:
        names = [file_name + extension for extension in IMAGE_EXTENSIONS]
    for directory in _candidate_image_dirs(dataset_root, manifest_dir, split):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _resolve_label_path(
    dataset_root: Path,
    manifest_dir: Path,
    split: str,
    file_name: str,
    image_path: Path | None,
) -> Path | None:
    stem = Path(file_name).stem
    directories: list[Path] = []
    if image_path is not None and image_path.parent.name == "images":
        directories.append(image_path.parent.parent / "labels")
    if image_path is not None:
        directories.append(image_path.parent)
    directories.extend(
        [
            dataset_root / split / "labels",
            manifest_dir / split / "labels",
            dataset_root / "labels",
            manifest_dir / "labels",
        ]
    )
    for directory in directories:
        candidate = directory / (stem + ".txt")
        if candidate.is_file():
            return candidate
    return None


def _detect_split_mismatch(path: Path, dataset_root: Path, manifest_dir: Path, split: str) -> str | None:
    """A file that physically lives under another split directory is leakage."""
    for base in (dataset_root, manifest_dir):
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        for part in relative.parts[:-1]:
            if part in SPLITS and part != split:
                return part
    return None


def _as_bool(value: str | None) -> bool:
    return bool(value) and str(value).strip().lower() in _TRUTHY


def _as_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def audit_sample(
    row: Mapping[str, str],
    dataset_root: Path | str,
    manifest_dir: Path | str,
    expected_class: int = 0,
    decode_images: bool = True,
) -> SampleAudit:
    """Audit one manifest row: existence, layout, provenance and label content."""
    root = Path(dataset_root)
    manifest_base = Path(manifest_dir)
    issues: list[AuditIssue] = []

    file_name = str(row.get("file", "") or "").strip()
    split = str(row.get("split", "") or "").strip()
    source_type = str(row.get("source_type", "") or "").strip()
    camera_raw = str(row.get("camera_id", "") or "").strip()
    camera_id = camera_raw or None
    sample_id = Path(file_name).stem if file_name else ""

    if not file_name:
        issues.append(
            AuditIssue(
                sample_id or MANIFEST_SCOPE,
                CODE_MANIFEST_ROW_INCOMPLETE,
                "ERROR",
                "row has no file column value",
            )
        )

    if split not in SPLITS:
        issues.append(
            AuditIssue(sample_id, CODE_SPLIT_ILLEGAL, "ERROR", "split=" + repr(split))
        )

    family = canonical_source_family(source_type)
    if family is None:
        issues.append(
            AuditIssue(
                sample_id, CODE_SOURCE_TYPE_ILLEGAL, "ERROR", "source_type=" + repr(source_type)
            )
        )
    elif family != source_type.strip().upper():
        issues.append(
            AuditIssue(
                sample_id,
                CODE_SOURCE_TYPE_NON_CANONICAL,
                "WARNING",
                "source_type=" + repr(source_type) + " family=" + family,
            )
        )

    image_path = (
        _resolve_image_path(root, manifest_base, split, file_name) if file_name else None
    )
    if file_name and image_path is None:
        issues.append(
            AuditIssue(
                sample_id,
                CODE_IMAGE_MISSING,
                "ERROR",
                "no candidate matched " + repr(file_name) + " under " + str(root),
            )
        )
    elif image_path is not None:
        mismatch = _detect_split_mismatch(image_path, root, manifest_base, split)
        if mismatch is not None:
            issues.append(
                AuditIssue(
                    sample_id,
                    CODE_SPLIT_MISMATCH,
                    "ERROR",
                    "file lives under split directory " + repr(mismatch) + " but split=" + repr(split),
                )
            )
        if image_path.stat().st_size == 0:
            issues.append(AuditIssue(sample_id, CODE_IMAGE_EMPTY, "ERROR", str(image_path)))
        elif decode_images:
            issues.extend(_decode_image_issues(sample_id, image_path, row))

    label_path = (
        _resolve_label_path(root, manifest_base, split, file_name, image_path) if file_name else None
    )

    negative_hint = _as_bool(row.get("is_negative")) or family == "NEGATIVE"
    if label_path is None:
        severity = "WARNING" if negative_hint else "ERROR"
        issues.append(
            AuditIssue(
                sample_id,
                CODE_LABEL_MISSING,
                severity,
                "no label file for " + repr(file_name),
            )
        )
        box_count = 0
    else:
        issues.extend(validate_yolo_label(sample_id, label_path, expected_class=expected_class))
        box_count = _count_label_boxes(label_path)

    is_negative = negative_hint or (label_path is not None and box_count == 0)

    return SampleAudit(
        sample_id=sample_id,
        split=split,
        source_type=source_type,
        source_family=family or "",
        camera_id=camera_id,
        image_path=None if image_path is None else str(image_path),
        label_path=None if label_path is None else str(label_path),
        box_count=box_count,
        is_negative=is_negative,
        issues=issues,
    )


def _count_label_boxes(label_path: Path) -> int:
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def _decode_image_issues(sample_id: str, image_path: Path, row: Mapping[str, str]) -> list[AuditIssue]:
    """Decode the raster when Pillow is importable, and cross-check its size.

    Pillow is optional: the audit must run on hosts that only have the standard
    library, where this check is skipped rather than failed.
    """
    issues: list[AuditIssue] = []
    try:
        from PIL import Image  # noqa: PLC0415 - optional dependency
    except Exception:
        return issues
    try:
        with Image.open(image_path) as image:
            size = image.size
    except Exception as error:  # Pillow raises many exception types
        return [
            AuditIssue(
                sample_id, CODE_IMAGE_UNREADABLE, "ERROR", str(image_path) + ": " + str(error)
            )
        ]
    expected_side = _as_float(row.get("imgsz"))
    if expected_side and expected_side > 0 and size != (int(expected_side), int(expected_side)):
        issues.append(
            AuditIssue(
                sample_id,
                CODE_IMAGE_SIZE_MISMATCH,
                "WARNING",
                "image is " + str(size[0]) + "x" + str(size[1]) + " but imgsz=" + str(int(expected_side)),
            )
        )
    return issues


def find_duplicate_sample_ids(rows: Iterable[tuple[str, str]]) -> list[AuditIssue]:
    """A sample id may appear once in the whole dataset, across all manifests."""
    seen: dict[str, list[str]] = {}
    order: list[str] = []
    for sample_id, source in rows:
        if sample_id not in seen:
            seen[sample_id] = []
            order.append(sample_id)
        seen[sample_id].append(source)
    issues: list[AuditIssue] = []
    for sample_id in order:
        sources = seen[sample_id]
        if len(sources) > 1:
            issues.append(
                AuditIssue(
                    sample_id,
                    CODE_DUPLICATE_SAMPLE_ID,
                    "ERROR",
                    "appears " + str(len(sources)) + " times in: " + ", ".join(sorted(set(sources))),
                )
            )
    return issues
