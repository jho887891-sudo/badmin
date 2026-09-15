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
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import SPLITS, SOURCE_TYPES, AuditIssue, AuditResult

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
CODE_HELD_OUT_SPLIT_MIXED = "HELD_OUT_SPLIT_WITH_TRAINING_SPLIT"
CODE_IMAGE_DECODE_SKIPPED = "IMAGE_DECODE_CHECK_SKIPPED"

# A manifest-level finding has no single sample to blame.
MANIFEST_SCOPE = "<manifest>"

# Splits a training run may draw from, and splits that must never be trained on as
# a whole (spec section 6). Auditing both together is legitimate -- it produces one
# inventory -- but the pooled result is then a report, not a training set, which is
# what summary["bound_for_training"] records.
TRAINING_SPLITS: tuple[str, ...] = ("train", "val")
HELD_OUT_SPLITS: tuple[str, ...] = ("fixed_core_test", "challenge_test")

# Status vocabulary for summary["checks"]. A check that did not run must be
# visible as NOT_RUN: leaving a composite at PASS would claim verification that
# never happened.
CHECK_RAN = "RAN"
CHECK_NOT_RUN = "NOT_RUN"
CHECK_NOT_CHECKED = "NOT_CHECKED_BY_THIS_AUDIT"

# What summary["dataset_version"] identifies. Hashing the manifest alone cannot
# distinguish two datasets that share identical manifests but different images, so
# the scope is part of the token and is stated next to it.
DATASET_VERSION_SCOPE_IMAGE = "manifest+image-content"
DATASET_VERSION_SCOPE_MANIFEST = "manifest-only"

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


_DECODER_AVAILABLE: bool | None = None


def image_decoder_available() -> bool:
    """Whether this host can decode images at all (Pillow), memoised.

    Pillow is optional: the audit must run on hosts that only have the standard
    library. A skipped corrupt-image check is then reported as NOT_RUN instead of
    being silently dropped, so the check runner and the summary both ask here.
    """
    global _DECODER_AVAILABLE
    if _DECODER_AVAILABLE is None:
        try:
            from PIL import Image  # noqa: F401,PLC0415 - optional dependency

            _DECODER_AVAILABLE = True
        except Exception:
            _DECODER_AVAILABLE = False
    return _DECODER_AVAILABLE


def _decode_image_issues(sample_id: str, image_path: Path, row: Mapping[str, str]) -> list[AuditIssue]:
    """Decode the raster when a decoder exists, and cross-check its size."""
    issues: list[AuditIssue] = []
    if not image_decoder_available():
        return issues
    from PIL import Image  # noqa: PLC0415 - guarded by image_decoder_available()
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


# --------------------------------------------------------------------------- #
# Leakage detection (spec section 6)
# --------------------------------------------------------------------------- #


def _scan_cross_split(rows: Iterable[tuple[str, str, str]]) -> list[tuple[str, list[str], list[str]]]:
    """Group rows by key and return the keys that span more than one split.

    Rows are (sample_id, split, key). Rows without a key are skipped: a manifest
    that cannot supply a source group or a content hash must not be reported as if
    every one of its samples were a duplicate of every other.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for sample_id, split, key in rows:
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((sample_id, split))
    findings: list[tuple[str, list[str], list[str]]] = []
    for key in order:
        members = groups[key]
        splits = sorted({split for _, split in members})
        if len(splits) > 1:
            findings.append((key, splits, [sample_id for sample_id, _ in members]))
    return findings


def _summarise_ids(sample_ids: Sequence[str], limit: int = 5) -> str:
    if len(sample_ids) <= limit:
        return ", ".join(sample_ids)
    return ", ".join(sample_ids[:limit]) + " (+" + str(len(sample_ids) - limit) + " more)"


def find_cross_split_duplicates(rows: Iterable[tuple[str, str, str]]) -> list[AuditIssue]:
    """Report byte-identical images that appear in more than one split.

    Rows are (sample_id, split, content_hash). Spec section 6: the same synthetic
    sequence or the same real frame must never straddle train / val / test, so any
    content hash seen in two splits is an ERROR.
    """
    return [
        AuditIssue(
            _summarise_ids(sample_ids),
            CODE_CROSS_SPLIT_DUPLICATE,
            "ERROR",
            "identical content in splits " + ", ".join(splits) + " (hash " + key + ")",
        )
        for key, splits, sample_ids in _scan_cross_split(rows)
    ]


def find_source_group_leakage(rows: Iterable[tuple[str, str, str]]) -> list[AuditIssue]:
    """Report a source group (background, clip, scene) that spans two splits.

    Rows are (sample_id, split, group). Using one background or one video clip on
    both sides of a split boundary lets the model memorise the source instead of
    the object, which the pool isolation of spec section 3 exists to prevent.
    """
    return [
        AuditIssue(
            _summarise_ids(sample_ids),
            CODE_SOURCE_GROUP_LEAKAGE,
            "ERROR",
            "source group " + key + " in splits " + ", ".join(splits),
        )
        for key, splits, sample_ids in _scan_cross_split(rows)
    ]


# --------------------------------------------------------------------------- #
# Distribution reporting (spec sections 7 and 8)
# --------------------------------------------------------------------------- #

# Bucket edges chosen from the measured P4-A range (2.45 px to 32.86 px, median
# 8.94 px) so that every rendered size class keeps its own bucket instead of the
# whole set collapsing into one "small" bin.
SIZE_BUCKETS: tuple[str, ...] = ("<4px", "4-8px", "8-16px", "16-32px", ">=32px")
# motion_px is the rendered motion blur length in pixels; the frozen set uses
# 0, 1, 2, 4 and 7 px.
BLUR_BUCKETS: tuple[str, ...] = ("sharp", "slight", "moderate", "heavy")
# Orientation deviation from the reference orientation, folded into one half turn
# because the shuttlecock is (near) axisymmetric: 0 deg and 180 deg look alike.
POSE_BUCKETS: tuple[str, ...] = ("axis_aligned", "tilted", "steep")

UNKNOWN_BUCKET = "unknown"
UNSPECIFIED = "unspecified"

# Dimensions required in dataset_distribution_report.csv.
BUCKET_DIMENSIONS: tuple[str, ...] = (
    "split",
    "source_type",
    "camera_id",
    "size_bucket",
    "blur_bucket",
    "pose_bucket",
    "is_negative",
)


def bucket_size(equiv_size_px: Any) -> str:
    """Bucket an equivalent target size in pixels."""
    value = _as_float(equiv_size_px)
    if value is None:
        return UNKNOWN_BUCKET
    if value < 4.0:
        return SIZE_BUCKETS[0]
    if value < 8.0:
        return SIZE_BUCKETS[1]
    if value < 16.0:
        return SIZE_BUCKETS[2]
    if value < 32.0:
        return SIZE_BUCKETS[3]
    return SIZE_BUCKETS[4]


def bucket_blur(motion_px: Any) -> str:
    """Bucket the rendered motion blur length in pixels."""
    value = _as_float(motion_px)
    if value is None:
        return UNKNOWN_BUCKET
    if value <= 0.0:
        return BLUR_BUCKETS[0]
    if value <= 2.0:
        return BLUR_BUCKETS[1]
    if value <= 5.0:
        return BLUR_BUCKETS[2]
    return BLUR_BUCKETS[3]


def _fold_angle(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return abs(((number + 90.0) % 180.0) - 90.0)


def bucket_pose(yaw_deg: Any, pitch_deg: Any, roll_deg: Any) -> str:
    """Bucket the largest folded orientation deviation of the three pose angles."""
    folded = [_fold_angle(value) for value in (yaw_deg, pitch_deg, roll_deg)]
    known = [value for value in folded if value is not None]
    if not known:
        return UNKNOWN_BUCKET
    deviation = max(known)
    if deviation <= 15.0:
        return POSE_BUCKETS[0]
    if deviation <= 45.0:
        return POSE_BUCKETS[1]
    return POSE_BUCKETS[2]


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping row or an attribute-carrying object."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _first_field(record: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = _field(record, name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _normalize_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None or str(value).strip() == "":
        return "False"
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return "True"
    if text in {"0", "false", "no", "n", "f"}:
        return "False"
    return UNKNOWN_BUCKET


def summarize_distribution(records: Iterable[Any]) -> dict[str, dict[str, int]]:
    """Count records by every dimension required in the distribution report.

    Accepts manifest row dicts, inventory row dicts, or record objects; precomputed
    bucket columns are honoured so a caller can override the bucketing policy.
    """
    counters: dict[str, Counter] = {dimension: Counter() for dimension in BUCKET_DIMENSIONS}
    for record in records:
        split = _field(record, "split")
        counters["split"][
            str(split).strip() if split is not None and str(split).strip() else UNSPECIFIED
        ] += 1

        source = _field(record, "source_family")
        if source is None or str(source).strip() == "":
            source = canonical_source_family(str(_field(record, "source_type") or ""))
        counters["source_type"][str(source) if source else UNSPECIFIED] += 1

        camera = _field(record, "camera_id")
        counters["camera_id"][
            str(camera).strip() if camera is not None and str(camera).strip() else UNSPECIFIED
        ] += 1

        size_value = _field(record, "size_bucket")
        if size_value is None or str(size_value).strip() == "":
            size_value = bucket_size(_first_field(record, ("equiv_size_px", "target_px")))
        counters["size_bucket"][str(size_value)] += 1

        blur_value = _field(record, "blur_bucket")
        if blur_value is None or str(blur_value).strip() == "":
            blur_value = bucket_blur(_field(record, "motion_px"))
        counters["blur_bucket"][str(blur_value)] += 1

        pose_value = _field(record, "pose_bucket")
        if pose_value is None or str(pose_value).strip() == "":
            pose_value = bucket_pose(
                _field(record, "yaw_deg"), _field(record, "pitch_deg"), _field(record, "roll_deg")
            )
        counters["pose_bucket"][str(pose_value)] += 1

        counters["is_negative"][_normalize_bool_text(_field(record, "is_negative"))] += 1

    return {dimension: dict(counters[dimension]) for dimension in BUCKET_DIMENSIONS}


# --------------------------------------------------------------------------- #
# Manifest-level audit: ties the checks above together into one report
# --------------------------------------------------------------------------- #

# Column consulted for the "same source" of a sample, most specific first. A
# manifest that has none of these cannot be checked for source-group leakage.
GROUP_COLUMN_PREFERENCE: tuple[str, ...] = (
    "source_group",
    "source_id",
    "clip_id",
    "video_id",
    "sequence_id",
    "scene_id",
    "background",
)

CODE_SOURCE_GROUP_UNKNOWN = "SOURCE_GROUP_UNKNOWN"

INVENTORY_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "split",
    "source_type",
    "source_family",
    "camera_id",
    "is_negative",
    "box_count",
    "equiv_size_px",
    "size_bucket",
    "blur_bucket",
    "pose_bucket",
    "background",
    "image_sha256",
    "image_path",
    "label_path",
    "error_count",
    "warning_count",
)

CLEANING_COLUMNS: tuple[str, ...] = ("sample_id", "code", "severity", "detail")

DISTRIBUTION_COLUMNS: tuple[str, ...] = ("dimension", "value", "count")

LEAKAGE_COLUMNS: tuple[str, ...] = ("code", "key", "splits", "sample_ids", "detail")


@dataclass(frozen=True)
class DatasetAuditReport:
    """Everything the CLI needs to write its five output files."""

    result: AuditResult
    issues: list[AuditIssue]
    inventory: list[dict[str, Any]]
    leakage: list[dict[str, Any]]
    distribution: dict[str, dict[str, int]]
    summary: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_group_column(columns: Sequence[str]) -> str | None:
    present = {column.strip() for column in columns}
    for candidate in GROUP_COLUMN_PREFERENCE:
        if candidate in present:
            return candidate
    return None


def _dataset_version(
    manifest_infos: Sequence[Mapping[str, Any]],
    image_pairs: Sequence[tuple[str, str]],
    scope: str,
) -> str:
    """Order-independent identity of what was audited.

    The manifest bytes alone identify the annotations, not the pixels: two runs
    that share a manifest and differ only in the rendered images would collide. The
    per-sample content hashes are therefore part of the token, and the scope marker
    guarantees that a manifest-only token can never be mistaken for an
    image-bound one.
    """
    lines = [scope]
    lines.extend(
        sorted(
            str(info["name"]) + ":" + str(info["sha256"]) + ":" + str(info["rows"])
            for info in manifest_infos
        )
    )
    lines.extend(sorted(sample_id + ":" + digest for sample_id, digest in image_pairs))
    return "sha256:" + hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def _composite_status(found_problem: bool, check_ran: bool) -> str:
    """PASS/FAIL for a composite checklist entry, or NOT_RUN if it could not run.

    A real defect always wins: FAIL is reported even when a sibling check was
    skipped, because the finding does not depend on the skipped check.
    """
    if found_problem:
        return "FAIL"
    return "PASS" if check_ran else CHECK_NOT_RUN


def _leakage_rows(
    hash_rows: Sequence[tuple[str, str, str]], group_rows: Sequence[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, splits, sample_ids in _scan_cross_split(hash_rows):
        rows.append(
            {
                "code": CODE_CROSS_SPLIT_DUPLICATE,
                "key": key,
                "splits": ", ".join(splits),
                "sample_ids": _summarise_ids(sample_ids),
                "detail": "identical image content in more than one split",
            }
        )
    for key, splits, sample_ids in _scan_cross_split(group_rows):
        rows.append(
            {
                "code": CODE_SOURCE_GROUP_LEAKAGE,
                "key": key,
                "splits": ", ".join(splits),
                "sample_ids": _summarise_ids(sample_ids),
                "detail": "source group reused in more than one split",
            }
        )
    return rows


def audit_dataset(
    manifests: Path | str | Iterable[Path | str],
    dataset_root: Path | str | None = None,
    expected_class: int = 0,
    hash_images: bool = True,
    decode_images: bool = True,
) -> DatasetAuditReport:
    """Audit one or more manifests and return every report the spec asks for.

    Raises ManifestMissingError when a manifest does not exist, so a caller (the
    CLI) can turn that into a clean non-zero exit instead of a traceback.
    """
    if isinstance(manifests, (str, Path)):
        manifest_paths = [Path(manifests)]
    else:
        manifest_paths = [Path(path) for path in manifests]
    if not manifest_paths:
        raise ManifestError("no manifest given to audit")

    root = Path(dataset_root) if dataset_root is not None else manifest_paths[0].parent

    issues: list[AuditIssue] = []
    inventory: list[dict[str, Any]] = []
    hash_rows: list[tuple[str, str, str]] = []
    group_rows: list[tuple[str, str, str]] = []
    id_rows: list[tuple[str, str]] = []
    manifest_infos: list[dict[str, Any]] = []
    group_columns: set[str] = set()
    image_pairs: list[tuple[str, str]] = []
    total_rows = 0

    decode_check_ran = bool(decode_images and image_decoder_available())
    if decode_images and not decode_check_ran:
        issues.append(
            AuditIssue(
                MANIFEST_SCOPE,
                CODE_IMAGE_DECODE_SKIPPED,
                "WARNING",
                "no image decoder (Pillow) on this host: corrupt-image detection did not run",
            )
        )

    for manifest_path in manifest_paths:
        rows, columns = read_manifest(manifest_path)
        manifest_infos.append(
            {
                "path": str(manifest_path),
                "name": manifest_path.name,
                "sha256": _file_sha256(manifest_path),
                "rows": len(rows),
                "columns": list(columns),
            }
        )
        issues.extend(check_manifest_columns(columns))
        group_column = _pick_group_column(columns)
        if group_column is None:
            issues.append(
                AuditIssue(
                    MANIFEST_SCOPE,
                    CODE_SOURCE_GROUP_UNKNOWN,
                    "WARNING",
                    "no source-group column in "
                    + manifest_path.name
                    + "; source leakage cannot be checked",
                )
            )
        else:
            group_columns.add(group_column)
        total_rows += len(rows)

        # A qualified source_type is a property of the whole manifest, not of each
        # sample: reporting it 400 times would bury the findings that matter, so it
        # is counted here and emitted once per manifest.
        non_canonical_sources: Counter[str] = Counter()

        for row in rows:
            sample = audit_sample(
                row,
                dataset_root=root,
                manifest_dir=manifest_path.parent,
                expected_class=expected_class,
                decode_images=decode_images,
            )
            for issue in sample.issues:
                if issue.code == CODE_SOURCE_TYPE_NON_CANONICAL:
                    non_canonical_sources[sample.source_type] += 1
                    continue
                issues.append(issue)

            content_hash = ""
            if hash_images and sample.image_path:
                content_hash = _file_sha256(Path(sample.image_path))
            if content_hash:
                image_pairs.append((sample.sample_id, content_hash))
            hash_rows.append((sample.sample_id, sample.split, content_hash))
            group_rows.append(
                (
                    sample.sample_id,
                    sample.split,
                    str(row.get(group_column, "")) if group_column else "",
                )
            )
            if sample.sample_id:
                id_rows.append((sample.sample_id, manifest_path.name))

            sample_error_count = sum(
                1 for issue in sample.issues if issue.severity == "ERROR"
            )
            sample_warning_count = sum(
                1
                for issue in sample.issues
                if issue.severity == "WARNING" and issue.code != CODE_SOURCE_TYPE_NON_CANONICAL
            )
            inventory.append(
                {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "source_type": sample.source_type,
                    "source_family": sample.source_family,
                    "camera_id": sample.camera_id,
                    "is_negative": sample.is_negative,
                    "box_count": sample.box_count,
                    "equiv_size_px": row.get("equiv_size_px", ""),
                    "size_bucket": bucket_size(_first_field(row, ("equiv_size_px", "target_px"))),
                    "blur_bucket": bucket_blur(row.get("motion_px")),
                    "pose_bucket": bucket_pose(
                        row.get("yaw_deg"), row.get("pitch_deg"), row.get("roll_deg")
                    ),
                    "background": row.get(group_column, "") if group_column else "",
                    "image_sha256": content_hash,
                    "image_path": sample.image_path or "",
                    "label_path": sample.label_path or "",
                    "error_count": sample_error_count,
                    "warning_count": sample_warning_count,
                }
            )

        for source_value, count in non_canonical_sources.items():
            issues.append(
                AuditIssue(
                    MANIFEST_SCOPE,
                    CODE_SOURCE_TYPE_NON_CANONICAL,
                    "WARNING",
                    "source_type="
                    + repr(source_value)
                    + " family="
                    + str(canonical_source_family(source_value))
                    + " in "
                    + str(count)
                    + " sample(s) of "
                    + manifest_path.name,
                )
            )

    if total_rows == 0:
        issues.append(
            AuditIssue(MANIFEST_SCOPE, CODE_MANIFEST_EMPTY, "ERROR", "no manifest row to audit")
        )

    issues.extend(find_duplicate_sample_ids(id_rows))
    leakage_issues = find_cross_split_duplicates(hash_rows) + find_source_group_leakage(group_rows)
    issues.extend(leakage_issues)

    split_counts = Counter(str(row["split"]) for row in inventory)
    splits_present = [split for split in SPLITS if split in split_counts] + sorted(
        split for split in split_counts if split and split not in SPLITS
    )
    training_splits_present = [split for split in splits_present if split in TRAINING_SPLITS]
    held_out_splits_present = [split for split in splits_present if split in HELD_OUT_SPLITS]

    # Auditing several pools together is how a complete inventory is produced, so a
    # held-out row is not an ERROR here. It does mean the pooled set must never be
    # trained on as a whole, which the WARNING and bound_for_training both state --
    # the hard refusal belongs to the training-path dataset builder, not to the audit.
    bound_for_training = not (training_splits_present and held_out_splits_present)
    if not bound_for_training:
        issues.append(
            AuditIssue(
                MANIFEST_SCOPE,
                CODE_HELD_OUT_SPLIT_MIXED,
                "WARNING",
                "this audit mixes training splits ("
                + ", ".join(training_splits_present)
                + ") with held-out splits ("
                + ", ".join(held_out_splits_present)
                + "); the pooled inventory is for reporting only and is not a training set",
            )
        )

    result = AuditResult.from_issues(issues)
    distribution = summarize_distribution(inventory)
    severity_counts = Counter(issue.severity for issue in issues)
    code_counts = Counter(issue.code for issue in issues)
    group_check_ran = bool(group_columns)
    version_scope = (
        DATASET_VERSION_SCOPE_IMAGE if hash_images else DATASET_VERSION_SCOPE_MANIFEST
    )
    version_note = (
        "identifies the audited manifest bytes and the content hash of every audited image"
        if hash_images
        else "identifies the manifest bytes only: image hashing was disabled, so two "
        "datasets that differ only in image bytes share this version"
    )
    checks = {
        "asset_consistency": CHECK_NOT_CHECKED,
        "near_duplicate_content": CHECK_NOT_CHECKED,
        "label_integrity": CHECK_RAN,
        "corrupt_image_decode": CHECK_RAN if decode_check_ran else CHECK_NOT_RUN,
        "cross_split_duplicate_content": CHECK_RAN if hash_images else CHECK_NOT_RUN,
        "within_split_exact_duplicate": CHECK_RAN if hash_images else CHECK_NOT_RUN,
        "source_group_leakage": CHECK_RAN if group_check_ran else CHECK_NOT_RUN,
        "distribution": CHECK_RAN,
    }
    label_codes = {
        CODE_LABEL_MISSING,
        CODE_LABEL_FORMAT,
        CODE_LABEL_NON_NUMERIC,
        CODE_WRONG_CLASS,
        CODE_BBOX_OUT_OF_RANGE,
    }
    label_ok = not any(issue.code in label_codes for issue in issues)

    summary = {
        "passed": result.passed,
        "issue_count": result.issue_count,
        "error_count": result.error_count,
        "warning_count": severity_counts.get("WARNING", 0),
        "n_samples": len(inventory),
        "n_manifests": len(manifest_infos),
        "dataset_root": str(root),
        "dataset_version": _dataset_version(manifest_infos, image_pairs, version_scope),
        "dataset_version_scope": version_scope,
        "dataset_version_note": version_note,
        "manifests": manifest_infos,
        "source_group_columns": sorted(group_columns),
        "splits_present": splits_present,
        "bound_for_training": bound_for_training,
        "counts_by_split": dict(split_counts),
        "counts_by_severity": dict(severity_counts),
        "counts_by_code": dict(code_counts),
        "distribution_dimensions": list(BUCKET_DIMENSIONS),
        "checks": checks,
        "checklist": {
            "asset_consistency": CHECK_NOT_CHECKED,
            "label_integrity": _composite_status(not label_ok, True),
            "data_cleaning": _composite_status(result.error_count > 0, decode_check_ran),
            "pool_isolation": _composite_status(
                bool(leakage_issues), hash_images and group_check_ran
            ),
            "distribution_quantified": _composite_status(not inventory, True),
        },
    }

    return DatasetAuditReport(
        result=result,
        issues=list(issues),
        inventory=inventory,
        leakage=_leakage_rows(hash_rows, group_rows),
        distribution=distribution,
        summary=summary,
    )
