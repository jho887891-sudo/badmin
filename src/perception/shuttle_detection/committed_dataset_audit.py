# -*- coding: utf-8 -*-
"""P7-01: committed-manifest-only dataset audit (split / duplicate / group / size).

Why this module exists
----------------------
The P4-A/P5 dataset audits (dataset_audit.py) verify image CONTENT: they open
every image and label.  That is the right audit when the resources are mounted,
and it is impossible here: the ETH images live on D:\\_eth_data, the Isaac pool is
gone (435 dangling manifest rows, management/shuttle_detection/
DATASET_RESOLUTION.md section 4), and the training host is the only place some
pools ever existed.  A dataset audit that crashes on a missing file reports
nothing about the 30k rows that DO have manifests.

This module therefore audits ONLY what is committed to git: the manifest CSV
rows.  It answers four questions with no resource dependency:

  1. per-split counts (train / val / test, plus the sources and sizes behind them);
  2. duplicate / overlapping frame identifiers across splits (same path, same
     basename, same video clip + frame index);
  3. video/session grouping: frames of one recording session or clip must not
     straddle a split boundary (that is memorisation, not generalisation);
  4. negative-sample counts and the small-object size distribution.

Missing files are counted as DANGLING (a separate number, with a bounded sample),
never raised.  What this audit does NOT do is stated in
checks_not_performed: image-content duplicates need bytes; native-pixel sizes
need labels; a test split needs a test split.

Scope vocabulary (also written into the JSON):
  scope                 = "COMMITTED_MANIFEST_ONLY"
  file_existence_checked= True   (a stat per row; a missing file is DATA, not an error)
  image_content_checked = False
  native_px_sizes       = False unless the manifest records native px
  basis                 = whatever column the manifest records (see size_basis)

Run:
    python -m pytest tests/perception/shuttle_detection/test_committed_dataset_audit.py -q
    python scripts/shuttle_detection/audit_committed_datasets.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "AUDIT_SCHEMA_ID", "AUDIT_SCHEMA_VERSION", "AUDIT_SCOPE",
    "EVAL_SIZE_BUCKETS", "UNBUCKETED_SMALL", "UNKNOWN_SIZE_BUCKET",
    "FRAME_MARKER_DIRS", "MANIFEST_KINDS", "KNOWN_SESSION_CROSSINGS",
    "FrameIdentity", "ManifestRows", "audit_manifest", "audit_manifests",
    "bucket_size_eval", "read_manifest_rows", "derive_groups",
    "find_cross_split_overlaps", "find_group_leakage", "summarise_manifest",
]

AUDIT_SCHEMA_ID = "badmin.p7.dataset_audit"
AUDIT_SCHEMA_VERSION = 1
AUDIT_SCOPE = "COMMITTED_MANIFEST_ONLY"

# P7-02 evaluation buckets (management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md).
# The audit reports the SAME edges so that "the dataset has no 4-6 px samples" and
# "the detector is bad at 4-6 px" are stated in the same units.
EVAL_SIZE_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("4-6px", 4.0, 6.0),
    ("6-10px", 6.0, 10.0),
    ("10-20px", 10.0, 20.0),
    (">20px", 20.0, math.inf),
)
UNBUCKETED_SMALL = "<4px"
UNKNOWN_SIZE_BUCKET = "unknown"

FRAME_MARKER_DIRS: Tuple[str, ...] = ("images", "labels", "raw")

# Column maps per committed manifest family.  Every key is a role, every value is
# the CSV column that fills it; a missing key means "this manifest cannot answer
# that question" and is reported as null instead of being invented.
MANIFEST_KINDS: Dict[str, Dict[str, Optional[str]]] = {
    # outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv (30321 rows):
    # the only committed manifest that records a split for the real ETH data.
    "v1_dataset": {
        "image": "image", "label": "label", "split": "split", "source": "source",
        "group": "location", "n_boxes": "n_boxes", "size_lo": "eq640_min",
        "size_hi": "eq640_max", "bucket": "bucket", "width": "width",
        "height": "height", "image_root": None,
    },
    # outputs/shuttle_capability/train_data/manifest_{train,val}.csv: synthetic
    # renders keyed by a background asset, so the GROUP is the background.
    "synthetic_render": {
        "image": "file", "label": None, "split": "split", "source": "source_type",
        "group": "background", "n_boxes": None, "size_lo": "equiv_size_px",
        "size_hi": "equiv_size_px", "bucket": None, "width": None,
        "height": None, "image_root": "outputs/shuttle_capability/train_data/{split}/images",
    },
}

# Session crossings that are known, reviewed and NOT silenced: the test below
# fails on any crossing that is not listed here, and fails again if a listed one
# disappears (so the list cannot rot).  status documents what is actually true.
KNOWN_SESSION_CROSSINGS: Dict[str, Dict[str, Any]] = {
    "_eth_data/eth_shuttle_detection/old/iphone_20251015": {
        "status": "OPEN_LEAK_UNRESOLVED",
        "why": (
            "One iPhone recording session (old/iphone_20251015) has frames in both "
            "train and val. Per-clip identities do not cross, so this is session "
            "level, not frame-duplicate level, but the scene/lighting of that "
            "recording is shared by both sides. Resolving it requires re-splitting "
            "the committed v1 manifest (a dataset change, out of scope for a CPU-only "
            "audit slice) or excluding the session; both must be decided by the "
            "dataset owner."
        ),
    },
}

# A video frame counter is long (cab_1_00000, iphone ...000000001); a pool asset
# id is short (bg_008, hn2_001).  Requiring four digits keeps pool assets out of
# the clip grouping and stops two background pools colliding as one "clip".
_FRAME_INDEX_RE = re.compile(r"^(?P<prefix>.*?)[._-]?(?P<index>\d{4,})$")


class ManifestAuditError(RuntimeError):
    """A manifest cannot be read at all (missing file / no header)."""


@dataclass(frozen=True)
class FrameIdentity:
    """Normalised identity of one manifest row, computed from its path only."""

    relpath: str          # volume-less, forward-slash path as recorded
    basename: str
    stem: str
    frame_index: Optional[int]
    session: str          # directory that owns the frames (video / scene / pool)
    clip: str             # session + filename clip prefix (the video itself)

    @property
    def session_group(self) -> str:
        return "session:" + self.session

    @property
    def clip_group(self) -> str:
        return "clip:" + self.clip


def _components(raw: Any) -> List[str]:
    text = str(raw or "").strip().replace("\\", "/")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if parts and parts[0].endswith(":"):
        parts = parts[1:]  # drop a Windows volume: roots differ per machine
    return parts


def frame_identity(raw_path: Any) -> FrameIdentity:
    """Normalise a recorded image path into a comparable frame identity.

    The volume prefix is dropped on purpose: the ETH pool is recorded as
    D:\\_eth_data\\... on one host and as a mount path on another; two rows that
    differ only by the drive must still be detected as the same frame.
    """
    parts = _components(raw_path)
    if not parts:
        return FrameIdentity("", "", "", None, "", "")
    basename = parts[-1]
    stem = basename.rsplit(".", 1)[0]
    match = _FRAME_INDEX_RE.match(stem)
    if match:
        prefix = match.group("prefix") or stem
        index: Optional[int] = int(match.group("index"))
    else:
        prefix, index = stem, None
    marker_positions = [i for i, p in enumerate(parts) for m in FRAME_MARKER_DIRS
                        if p.lower() == m]
    if marker_positions:
        cut = min(marker_positions)
        session_parts = parts[:cut]
    else:
        session_parts = parts[:-1]
    session = "/".join(session_parts) if session_parts else "."
    clip = session + "/" + prefix
    return FrameIdentity("/".join(parts), basename, stem, index, session, clip)


def derive_groups(raw_path: Any, extra_group: Optional[str] = None) -> Dict[str, str]:
    """Return every grouping key the audit uses for one row."""
    ident = frame_identity(raw_path)
    groups = {"session": ident.session_group, "clip": ident.clip_group}
    if extra_group:
        groups["declared_group"] = "declared:" + str(extra_group)
    return groups


def bucket_size_eval(size_px: Any) -> str:
    """Bucket a size in pixels onto the P7-02 evaluation buckets.

    Buckets are half-open [lo, hi): a 6.0 px box is 6-10px, not 4-6px, and the
    convention is asserted in the tests so two reports cannot disagree.
    """
    try:
        value = float(size_px)
    except (TypeError, ValueError):
        return UNKNOWN_SIZE_BUCKET
    if not math.isfinite(value) or value <= 0.0:
        return UNKNOWN_SIZE_BUCKET
    if value < EVAL_SIZE_BUCKETS[0][1]:
        return UNBUCKETED_SMALL
    for name, lo, hi in EVAL_SIZE_BUCKETS:
        if lo <= value < hi:
            return name
    return EVAL_SIZE_BUCKETS[-1][0]


def _empty_buckets() -> Dict[str, int]:
    counts = {name: 0 for name, _, _ in EVAL_SIZE_BUCKETS}
    counts[UNBUCKETED_SMALL] = 0
    counts[UNKNOWN_SIZE_BUCKET] = 0
    return counts


@dataclass
class ManifestRows:
    """One manifest file, parsed and normalised. Nothing here touches image bytes."""

    path: Path
    kind: str
    rows: List[Dict[str, str]] = field(default_factory=list)
    header: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    sha256: Optional[str] = None


def read_manifest_rows(path: Path, kind: str, *, compute_sha256: bool = True) -> ManifestRows:
    """Read a manifest CSV into plain row dicts, recording which roles are missing.

    Decoding falls back to latin-1 so one non-UTF-8 byte cannot abort an audit
    whose whole job is to report problems.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestAuditError("manifest not found: " + str(manifest_path))
    raw = manifest_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ManifestAuditError("manifest has no header: " + str(manifest_path))
    header = [name.strip() for name in reader.fieldnames]
    rows = [
        {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        for row in reader
    ]
    columns = MANIFEST_KINDS[kind]
    missing = [role for role, column in columns.items()
               if column and column not in header]
    return ManifestRows(
        path=manifest_path, kind=kind, rows=rows, header=header,
        missing_columns=missing,
        sha256=hashlib.sha256(raw).hexdigest() if compute_sha256 else None,
    )


def _resolve_image(raw: Any, repo_root: Path, image_root: Optional[str],
                   split: str) -> Tuple[bool, str]:
    """Locate a recorded image without ever depending on it being there.

    Candidates, in order: the recorded path (it may exist on the machine that
    recorded it), the repo-root-relative tail of the recorded path, and the
    manifest-kind image root.  Returns (present, candidate_that_matched).
    """
    parts = _components(raw)
    candidates: List[Path] = []
    if str(raw).strip():
        candidates.append(Path(str(raw).strip()))
    if parts:
        candidates.append(repo_root.joinpath(*parts))
        if len(parts) > 2:
            candidates.append(repo_root.joinpath(*parts[2:]))
    if image_root:
        leaf = parts[-1] if parts else str(raw)
        candidates.append(repo_root / image_root.format(split=split) / leaf)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return True, str(candidate)
        except OSError:
            continue
    return False, ""


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    if number is None or number != int(number):
        return None
    return int(number)


def summarise_manifest(manifest: ManifestRows, repo_root: Path,
                       *, sample_limit: int = 10) -> Dict[str, Any]:
    """Summarise ONE manifest: splits, negatives, sizes, dangling, identities."""
    columns = MANIFEST_KINDS[manifest.kind]
    present_rows = 0
    dangling_rows = 0
    dangling_sample: List[str] = []
    label_dangling = 0
    split_counts: Counter[str] = Counter()
    split_negatives: Counter[str] = Counter()
    split_positives: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_split: Dict[str, Counter[str]] = defaultdict(Counter)
    sizes = _empty_buckets()
    size_split: Dict[str, Dict[str, int]] = defaultdict(_empty_buckets)
    sizes_raw: List[float] = []
    unknown_size = 0
    negative_bucket_disagreements = 0
    identities: Dict[str, FrameIdentity] = {}
    rows_by_split: Dict[str, set] = defaultdict(set)
    rows_by_basename: Dict[str, Counter[str]] = defaultdict(Counter)
    groups: Dict[str, Dict[str, set]] = {"session": defaultdict(set), "clip": defaultdict(set),
                                         "declared_group": defaultdict(set)}
    group_rows: Dict[str, Dict[str, Counter[str]]] = {
        "session": defaultdict(Counter), "clip": defaultdict(Counter),
        "declared_group": defaultdict(Counter)}

    for row in manifest.rows:
        split = row.get(columns["split"] or "", "") or "unspecified"
        split_counts[split] += 1
        source = row.get(columns["source"] or "", "") or "unspecified"
        source_counts[source] += 1
        source_split[source][split] += 1
        raw_image = row.get(columns["image"] or "", "")

        ident = frame_identity(raw_image)
        identities[ident.relpath] = ident
        rows_by_split[split].add(ident.relpath)
        rows_by_basename[ident.basename.lower()][split] += 1
        for name, key in derive_groups(raw_image, row.get(columns["group"] or "", "") or None).items():
            if name in groups:
                groups[name][key].add(split)
                group_rows[name][key][split] += 1

        n_boxes = _as_int(row.get(columns["n_boxes"] or "", "")) if columns["n_boxes"] else None
        bucket_text = (row.get(columns["bucket"] or "", "") or "").lower() if columns["bucket"] else ""
        is_negative: Optional[bool]
        if n_boxes is not None:
            is_negative = n_boxes == 0
            if bucket_text and ((bucket_text == "negative") != is_negative):
                negative_bucket_disagreements += 1
        elif bucket_text:
            is_negative = bucket_text == "negative"
        else:
            is_negative = None
        if is_negative is True:
            split_negatives[split] += 1
        elif is_negative is False:
            split_positives[split] += 1

        size = _as_float(row.get(columns["size_lo"] or "", "")) if columns["size_lo"] else None
        bucket = bucket_size_eval(size) if size is not None else UNKNOWN_SIZE_BUCKET
        if size is not None:
            sizes_raw.append(size)
            sizes[bucket] += 1
            size_split[split][bucket] += 1
        else:
            sizes[UNKNOWN_SIZE_BUCKET] += 1
            unknown_size += 1

        image_present, _ = _resolve_image(
            raw_image, repo_root, columns["image_root"], split)
        if image_present:
            present_rows += 1
        else:
            dangling_rows += 1
            if len(dangling_sample) < sample_limit:
                dangling_sample.append(ident.relpath)
        if columns["label"]:
            raw_label = row.get(columns["label"], "")
            if raw_label:
                label_present, _ = _resolve_image(
                    raw_label, repo_root, None, split)
                if not label_present:
                    label_dangling += 1

    sizes_raw.sort()

    def _quantile(values: Sequence[float], q: float) -> Optional[float]:
        if not values:
            return None
        pos = (len(values) - 1) * q
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return round(values[lo], 4)
        return round(values[lo] + (values[hi] - values[lo]) * (pos - lo), 4)

    cross_split_paths = sorted(
        path for path in identities
        if len(rows_by_split_of(path, rows_by_split)) > 1)
    return {
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "kind": manifest.kind,
        "rows": len(manifest.rows),
        "missing_columns": manifest.missing_columns,
        "scope": {
            "file_existence_checked": True,
            "image_content_checked": False,
            "native_px_sizes": False,
        },
        "splits": dict(sorted(split_counts.items())),
        "test_split_present": "test" in split_counts,
        "positives": dict(sorted(split_positives.items())),
        "negatives": dict(sorted(split_negatives.items())),
        "negatives_total": sum(split_negatives.values()),
        "negative_bucket_disagreements": negative_bucket_disagreements,
        "sources": dict(sorted(source_counts.items())),
        "source_by_split": {k: dict(sorted(v.items())) for k, v in sorted(source_split.items())},
        "size_distribution": sizes,
        "size_distribution_by_split": {k: v for k, v in sorted(size_split.items())},
        "size_stats": {
            "n": len(sizes_raw),
            "min": round(sizes_raw[0], 4) if sizes_raw else None,
            "p50": _quantile(sizes_raw, 0.5),
            "p95": _quantile(sizes_raw, 0.95),
            "max": round(sizes_raw[-1], 4) if sizes_raw else None,
            "unknown": unknown_size,
            "basis": columns["size_lo"],
            "basis_definition_in_repo": "UNDOCUMENTED (no code or doc defines the column)",
        },
        "dangling": {
            "rows": dangling_rows,
            "present": present_rows,
            "sample": dangling_sample,
            "label_dangling": label_dangling,
        },
        "frame_identities": len(identities),
        "cross_split_paths": cross_split_paths[:sample_limit],
        "cross_split_path_count": len(cross_split_paths),
        "groups": {name: {"n_groups": len(keys), "crossings": sorted(
            key for key, splits in keys.items() if len(splits) > 1)}
            for name, keys in groups.items()},
    }


def rows_by_split_of(path: str, rows_by_split: Mapping[str, set]) -> set:
    """Which splits contain this normalised path (helper kept for readability)."""
    return {split for split, paths in rows_by_split.items() if path in paths}


def find_cross_split_overlaps(manifest: ManifestRows) -> Dict[str, Any]:
    """Duplicate / overlapping frame identifiers across splits, three ways.

    * same normalised path in two splits        -> the same file is on both sides;
    * same basename (case-folded) in two splits -> two different records of one frame;
    * same clip + same trailing frame index     -> the same video frame re-used.
    """
    columns = MANIFEST_KINDS[manifest.kind]
    split_of: Dict[str, set] = defaultdict(set)
    basename_splits: Dict[str, set] = defaultdict(set)
    basename_rows: Dict[str, int] = Counter()
    clip_index: Dict[Tuple[str, Optional[int]], set] = defaultdict(set)
    for row in manifest.rows:
        split = row.get(columns["split"] or "", "") or "unspecified"
        ident = frame_identity(row.get(columns["image"] or "", ""))
        split_of[ident.relpath].add(split)
        basename_splits[ident.basename.lower()].add(split)
        basename_rows[ident.basename.lower()] += 1
        if ident.frame_index is not None:
            clip_index[(ident.clip, ident.frame_index)].add(split)
    return {
        "same_path_cross_split": sorted(p for p, s in split_of.items() if len(s) > 1),
        "same_basename_cross_split": sorted(b for b, s in basename_splits.items() if len(s) > 1),
        "same_basename_repeated_rows": sum(1 for b, n in basename_rows.items()
                                           if n > 1 and len(basename_splits[b]) > 1),
        "same_clip_frame_index_cross_split": sorted(
            clip + "#" + str(index) for (clip, index), s in clip_index.items() if len(s) > 1),
    }


def find_group_leakage(manifest: ManifestRows) -> Dict[str, List[Dict[str, Any]]]:
    """Groups (session / clip / declared background) whose frames straddle splits."""
    columns = MANIFEST_KINDS[manifest.kind]
    groups: Dict[str, Dict[str, Counter[str]]] = {
        "session": defaultdict(Counter), "clip": defaultdict(Counter),
        "declared_group": defaultdict(Counter)}
    for row in manifest.rows:
        split = row.get(columns["split"] or "", "") or "unspecified"
        declared = row.get(columns["group"] or "", "") or None if columns["group"] else None
        for name, key in derive_groups(row.get(columns["image"] or "", ""), declared).items():
            if name in groups:
                groups[name][key][split] += 1
    findings: Dict[str, List[Dict[str, Any]]] = {}
    for name, keys in groups.items():
        findings[name] = [
            {"group": key, "splits": dict(sorted(counts.items())),
             "rows": sum(counts.values())}
            for key, counts in sorted(keys.items()) if len(counts) > 1
        ]
    return findings


def audit_manifest(manifest_ref: Any, *, kind: str, repo_root: Path,
                   compute_sha256: bool = True) -> Dict[str, Any]:
    """Audit one manifest and return the (JSON-safe) report block."""
    rows = read_manifest_rows(Path(manifest_ref), kind, compute_sha256=compute_sha256)
    summary = summarise_manifest(rows, repo_root)
    summary["overlaps"] = find_cross_split_overlaps(rows)
    summary["group_leakage"] = find_group_leakage(rows)
    return summary


def audit_manifests(
    manifests: Sequence[Tuple[str, str]],
    *, repo_root: Path,
    evidence_class: str = "COMMITTED_MANIFESTS",
) -> Dict[str, Any]:
    """Audit any number of (path, kind) pairs into one machine-readable report."""
    blocks = [audit_manifest(path, kind=kind, repo_root=repo_root) for path, kind in manifests]
    session_crossings = sorted({
        finding["group"].replace("session:", "", 1)
        for block in blocks for finding in block["group_leakage"]["session"]})
    clip_crossings = sorted({
        finding["group"].replace("clip:", "", 1)
        for block in blocks for finding in block["group_leakage"]["clip"]})
    known = set(KNOWN_SESSION_CROSSINGS)
    return {
        "schema": AUDIT_SCHEMA_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": AUDIT_SCOPE,
        "evidence_class": evidence_class,
        "repo_root": str(repo_root),
        "manifests": blocks,
        "leakage_summary": {
            "session_crossings": session_crossings,
            "clip_crossings": clip_crossings,
            "known_session_crossings": KNOWN_SESSION_CROSSINGS,
            "unexplained_session_crossings": sorted(set(session_crossings) - known),
            "unexplained_clip_crossings": sorted(set(clip_crossings)),
            "known_crossings_absent_now": sorted(known - set(session_crossings)),
        },
        "checks_not_performed": [
            "image-content duplicates (needs image bytes: scope is manifest-only)",
            "label-content errors (bbox out of range, wrong class: needs label bytes)",
            "native-pixel object sizes (the recorded size column has no in-repo definition)",
            "test-split leakage with a test split (no committed manifest has a test split)",
        ],
        "caveats": [
            "A missing file is counted as DANGLING and never raised; a dangling row "
            "cannot be judged for content.",
            "Grouping is derived from the RECORDED PATH (session directory + filename "
            "clip prefix). It is a heuristic, not dataset metadata: two recordings "
            "sharing a directory look like one session.",
            "No detector accuracy is measured or implied by this file.",
        ],
    }


def dumps(report: Mapping[str, Any]) -> str:
    """Serialise deterministically (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
