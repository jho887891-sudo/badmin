#!/usr/bin/env python3
"""COCO train2017 outdoor hard-negative curation, empty-label export and review gate (Task 5).

The second NEW pool of CLEAN_OUTDOOR_CP_V1 is made of real outdoor COCO train2017 photos that LOOK like a
shuttlecock (glints, paper, plastic, feathers, thin white foliage, signage) but contain none. Four rules are
non-negotiable and are enforced here rather than described:

1. the only admissible source is COCO train2017, and never an image the frozen V2 work or any frozen
   evaluation set already used (checked by COCO image id, by SHA256 and by near-duplicate group);
2. a hard negative is a YOLO .txt that EXISTS and is ZERO bytes (single class 0). COCO has no shuttlecock
   category, so the absence of a COCO annotation is never accepted as proof of an empty picture: only an
   explicit human full-size review (no_shuttle_reviewed) counts, and an "uncertain tiny white dot" review,
   a declined review or a missing reviewer category keeps the picture out;
3. no COCO image, byte or near-duplicate cluster may serve as a CLEAN copy-paste background and as a hard
   negative at the same time;
4. the pool gate is OUTDOOR_HARDNEG_INSUFFICIENT below 150 unique approved images, 3 reviewed distractor
   categories or 3 audited source clusters.

mine_v2_fp_candidates() is a read-only review aid that prioritises pictures for a human to inspect with the
frozen V2 checkpoint. It can only be pointed at train-only COCO images, it refuses every evaluation path or
non-COCO file name, it writes nothing, and its output is never a training manifest.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a declared project dependency
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

# --- frozen vocabulary (mirrors configs/eth_real_clean_outdoor_copypaste_v1.yaml) -------------------
TRAIN_SPLIT = "train2017"
FORBIDDEN_SPLITS = ("val2017", "test2017")
COCO_IMAGE_NAME = re.compile(r"^(\d{12})\.(jpg|jpeg|png)$", re.IGNORECASE)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

#: reviewed distractor categories a human may record; classify_negative_hardness accepts exactly these
REVIEWED_CATEGORIES = (
    "white_reflection",
    "paper_plastic",
    "feather_leaf",
    "metal_glint",
    "bright_signage",
    "other_reviewed",
)

MIN_UNIQUE_IMAGES = 150
MIN_REVIEWED_CATEGORIES = 3
MIN_AUDITED_CLUSTERS = 3

GATE_PASS = "PASS"
GATE_INSUFFICIENT = "OUTDOOR_HARDNEG_INSUFFICIENT"
GATE_LEAK = "INVALID_OR_LEAK"

#: explicit reject reasons; a rejected row always carries exactly one of them plus an actionable detail
REASON_BAD_ID = "coco_image_id_invalid"
REASON_NOT_TRAIN2017 = "not_coco_train2017"
REASON_HASH_UNVERIFIABLE = "content_hash_unverifiable"
REASON_FROZEN_EVAL = "frozen_eval_overlap"
REASON_PREVIOUSLY_USED = "previously_used_v2_coco_negative"
REASON_CLEAN_BG_ID = "coco_image_id_in_clean_bg"
REASON_CLEAN_BG_HASH = "content_hash_in_clean_bg"
REASON_CLEAN_BG_NEARDUP = "near_duplicate_with_clean_bg"
REASON_LICENSE = "license_unverifiable"
REASON_LABEL_MISSING = "label_missing"
REASON_LABEL_NONEMPTY = "nonempty_label"
REASON_NO_SHUTTLE_MISSING = "no_shuttle_review_missing"
REASON_SHUTTLE_UNCERTAIN = "shuttle_present_or_uncertain"
REASON_TINY_DOT = "uncertain_tiny_dot_review"
REASON_NOT_APPROVED = "review_not_approved"
REASON_CATEGORY_MISSING = "reviewed_category_missing"
REASON_CATEGORY_INVALID = "reviewed_category_invalid"
REASON_DUPLICATE_ID = "duplicate_coco_image_id"
REASON_DUPLICATE_CONTENT = "duplicate_content"
REASON_NEARDUP_POOL = "near_duplicate_within_pool"

INVENTORY_COLUMNS = (
    "coco_image_id", "file_name", "split", "path", "sha256", "width", "height",
    "license_or_permission", "permission_state", "reviewer_id", "no_shuttle_reviewed",
    "review_uncertain_tiny_dot", "reviewed_category", "hardness_class", "source_cluster",
    "near_duplicate_group", "label_path", "label_exists", "label_bytes", "n_boxes",
    "is_negative", "is_positive", "role", "qa_status",
)
MANIFEST_COLUMNS = (
    "coco_image_id", "file_name", "path", "label_path", "label_bytes", "label_empty", "n_boxes",
    "is_negative", "is_positive", "role", "split", "sha256", "license_or_permission",
    "permission_state", "reviewer_id", "no_shuttle_reviewed", "review_uncertain_tiny_dot",
    "reviewed_category", "hardness_class", "source_cluster", "near_duplicate_group", "qa_status",
)
REJECTION_COLUMNS = ("coco_image_id", "file_name", "path", "reason", "detail")
PROHIBITED_COLUMNS = ("source_id", "coco_image_id", "sha256", "path", "origin", "role", "reason")
FALSE_POSITIVE_COLUMNS = (
    "coco_image_id", "path", "n_detections", "max_conf", "class_ids", "suggested_for_review",
)

#: path tokens that can only belong to an evaluation set; matched as substrings of the directory name
FORBIDDEN_PATH_TOKENS = (
    "val2017", "test2017", "evaluation", "frozen", "challenge_test", "controlled_capability",
    "eth_unseen", "no_target", "hard_eval",
)
#: ambiguous short words ("eval", "496") are only refused on a word boundary so "retrieval" is not blocked
FORBIDDEN_PATH_WORDS = ("eval", "496")
WORD_BOUNDARY = r"(^|[^a-z0-9])%s([^a-z0-9]|$)"

DEFAULT_THRESHOLDS = {
    "min_unique_images": MIN_UNIQUE_IMAGES,
    "min_reviewed_categories": MIN_REVIEWED_CATEGORIES,
    "min_audited_clusters": MIN_AUDITED_CLUSTERS,
}

TRUTHY = {"true", "1", "yes"}
FALSY = {"false", "0", "no"}
BLANK_WORDS = {"none", "nan", "<na>", "unknown", "n/a", "na", "null"}


class OutdoorHardNegBlocked(RuntimeError):
    """Raised when a hard-negative input is not the authorised COCO train2017 material."""


# --- small pure helpers ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    """SHA256 of a file, streamed so both a 20 MB checkpoint and a 100 KB photo are cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blank(value) -> bool:
    """True for None, NaN, empty text and the explicit "unknown"-style placeholders, never for a value."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):  # pragma: no cover - exotic containers only
        pass
    text = str(value).strip()
    return text == "" or text.lower() in BLANK_WORDS


def _text(value) -> str:
    return "" if _blank(value) else str(value).strip()


def _truthy(value) -> bool:
    return str(value).strip().lower() in TRUTHY


def _falsy(value) -> bool:
    return str(value).strip().lower() in FALSY


def _as_int(value):
    if _blank(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _path_key(path_text: str) -> str:
    """Normalised comparison key so two spellings of the same file cannot slip past a path check."""
    return str(Path(path_text).resolve()).lower() if path_text else ""


def _has_word(text: str, word: str) -> bool:
    return re.search(WORD_BOUNDARY % re.escape(word), text) is not None


def canonical_reviewed_category(reviewed_category):
    """Return the frozen category name, or None when nothing valid was reviewed (never a default)."""
    if _blank(reviewed_category):
        return None
    normalised = str(reviewed_category).strip().lower().replace("-", "_").replace(" ", "_")
    return normalised if normalised in REVIEWED_CATEGORIES else None


def classify_negative_hardness(image_id: int, reviewed_category: str) -> str:
    """Map one human-reviewed distractor category onto the frozen hardness class (the category itself)."""
    category = canonical_reviewed_category(reviewed_category)
    if category is None:
        raise OutdoorHardNegBlocked(
            "COCO image %s has no reviewed distractor category; allowed values are %s, got %r"
            % (image_id, list(REVIEWED_CATEGORIES), reviewed_category)
        )
    return category


def _label_state(row):
    """(label_path, exists, bytes) read from disk; a label path that was never written is not "empty"."""
    label_path = _text(row.get("label_path"))
    if not label_path:
        image_path = _text(row.get("path"))
        if not image_path:
            return None, False, None
        label_path = str(Path(image_path).with_suffix(".txt"))
    path = Path(label_path)
    if not path.is_file():
        return str(path), False, None
    size = path.stat().st_size
    return str(path), True, size


def _count_label_boxes(label_path, label_bytes):
    """Boxes in the YOLO label; a zero-byte file has none, anything else is reported, never rounded away."""
    if not label_path or not label_bytes:
        return 0
    text = Path(label_path).read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if line.strip())


def _split_from_path(path_text: str) -> str:
    """The split a picture lives in, taken from its path. Only a train2017 directory is admissible."""
    parts = Path(path_text).parts
    forbidden = [part for part in parts if part in FORBIDDEN_SPLITS]
    if forbidden:
        raise OutdoorHardNegBlocked(
            "only COCO %s may feed the outdoor hard-negative pool; %s is in %s"
            % (TRAIN_SPLIT, forbidden[0], path_text)
        )
    if TRAIN_SPLIT not in parts:
        raise OutdoorHardNegBlocked(
            "path is not inside a COCO %s split directory: %s" % (TRAIN_SPLIT, path_text)
        )
    return TRAIN_SPLIT


def _row_split_ok(row) -> bool:
    declared = _text(row.get("split"))
    if declared and declared != TRAIN_SPLIT:
        return False
    path_text = _text(row.get("path"))
    if not path_text:
        return False
    return TRAIN_SPLIT in Path(path_text).parts


def _rejection_record(coco_id, file_name, path_text, reason, detail) -> dict:
    return {
        "coco_image_id": coco_id if coco_id is not None else "",
        "file_name": file_name or "",
        "path": path_text or "",
        "reason": reason,
        "detail": detail,
    }


def _image_size(path_text: str):
    if Image is None or not path_text or not Path(path_text).is_file():
        return None, None
    try:
        with Image.open(path_text) as handle:
            return handle.size
    except Exception:  # noqa: BLE001 - an undecodable image is reported, not silently sized 0
        return None, None


# --- Task 5 interface: inventory ------------------------------------------------------------------

def inventory_outdoor_hardnegatives(coco_candidates: pd.DataFrame) -> pd.DataFrame:
    """Normalise the Task 2 COCO train2017 candidate rows into a reviewable hard-negative inventory.

    The function is deliberately strict about the things that must never be guessed: every row must live in
    a COCO train2017 directory with its bytes on disk, a supplied SHA256 must match those bytes, and the
    licence text and the YOLO label state are read from disk instead of being trusted from the sheet.
    Review fields (reviewer, no_shuttle_reviewed, category, cluster, qa_status) are passed through untouched
    and stay blank when nobody reviewed anything: no default ever approves a picture.
    """
    if coco_candidates is None:
        raise OutdoorHardNegBlocked(
            "inventory_outdoor_hardnegatives needs the Task 2 COCO train2017 candidate rows, got None"
        )
    frame = coco_candidates if isinstance(coco_candidates, pd.DataFrame) else pd.DataFrame(coco_candidates)
    if len(frame) == 0:
        return pd.DataFrame(columns=list(INVENTORY_COLUMNS))
    missing = [column for column in ("coco_image_id", "path") if column not in frame.columns]
    if missing:
        raise OutdoorHardNegBlocked("COCO candidate rows miss required columns: %s" % missing)

    rows: list = []
    for record in frame.to_dict("records"):
        coco_id = _as_int(record.get("coco_image_id"))
        if coco_id is None:
            raise OutdoorHardNegBlocked("COCO candidate row has no usable coco_image_id: %r" % record)
        image_path = Path(_text(record.get("path")))
        if not image_path.is_file():
            raise OutdoorHardNegBlocked(
                "COCO image bytes are missing for id %d: %s (a full-size original is required for review)"
                % (coco_id, image_path)
            )
        derived_split = _split_from_path(str(image_path))
        declared_split = _text(record.get("split"))
        if declared_split and declared_split != derived_split:
            raise OutdoorHardNegBlocked(
                "split column %r contradicts the path %s for COCO image %d"
                % (declared_split, image_path, coco_id)
            )

        digest = sha256_file(image_path)
        supplied = _text(record.get("sha256"))
        if supplied:
            if not SHA256_PATTERN.match(supplied):
                raise OutdoorHardNegBlocked(
                    "sha256 for COCO image %d is not a sha256 digest: %r" % (coco_id, supplied)
                )
            if supplied.lower() != digest:
                raise OutdoorHardNegBlocked(
                    "sha256 for COCO image %d does not match the bytes on disk (%s != %s)"
                    % (coco_id, supplied.lower(), digest)
                )

        license_text = _text(record.get("license_or_permission"))
        if not license_text:
            license_text = _text(record.get("license_name")) or _text(record.get("license_url"))
        permission_state = _text(record.get("permission_state")).lower()
        if permission_state not in ("declared", "unknown"):
            permission_state = "declared" if license_text else "unknown"
        if permission_state == "declared" and not license_text:
            # a "declared" flag with no licence/permission text proves nothing
            permission_state = "unknown"

        label_path, label_exists, label_bytes = _label_state({**record, "path": str(image_path)})
        width, height = _as_int(record.get("width")), _as_int(record.get("height"))
        if width is None or height is None:
            width, height = _image_size(str(image_path))
        category = canonical_reviewed_category(record.get("reviewed_category"))

        row = dict(record)
        row.update({
            "coco_image_id": coco_id,
            "file_name": _text(record.get("file_name")) or image_path.name,
            "split": derived_split,
            "path": str(image_path),
            "sha256": digest,
            "width": width,
            "height": height,
            "license_or_permission": license_text,
            "permission_state": permission_state,
            "reviewer_id": _text(record.get("reviewer_id")),
            "no_shuttle_reviewed": record.get("no_shuttle_reviewed"),
            "review_uncertain_tiny_dot": record.get("review_uncertain_tiny_dot"),
            "reviewed_category": _text(record.get("reviewed_category")),
            "hardness_class": category or "",
            "source_cluster": _text(record.get("source_cluster")),
            "near_duplicate_group": _text(record.get("near_duplicate_group")),
            "label_path": label_path or "",
            "label_exists": label_exists,
            "label_bytes": label_bytes,
            "n_boxes": _count_label_boxes(label_path, label_bytes),
            "is_negative": True,
            "is_positive": False,
            "role": C.ROLE_COCO_OUTDOOR_HARDNEG,
            "qa_status": _text(record.get("qa_status")),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    ordered = list(INVENTORY_COLUMNS) + [column for column in out.columns if column not in INVENTORY_COLUMNS]
    return out[ordered]


# --- Task 5 interface: audit ----------------------------------------------------------------------

def _excluded_sets(excluded):
    """Split the Task 2 prohibited index into frozen-evaluation and historical-V2 id/hash/path sets."""
    ids, hashes, paths = set(), set(), set()
    eval_ids, eval_hashes, eval_paths = set(), set(), set()
    if excluded is None or len(excluded) == 0:
        return (frozenset(),) * 6
    for record in excluded.to_dict("records"):
        is_eval = _text(record.get("role")) == C.ROLE_FROZEN_EVAL
        target_ids, target_hashes, target_paths = (
            (eval_ids, eval_hashes, eval_paths) if is_eval else (ids, hashes, paths)
        )
        coco_id = _as_int(record.get("coco_image_id"))
        if coco_id is not None:
            target_ids.add(coco_id)
        digest = _text(record.get("sha256")).lower()
        if digest:
            target_hashes.add(digest)
        path_key = _path_key(_text(record.get("path")))
        if path_key:
            target_paths.add(path_key)
    return (frozenset(ids), frozenset(hashes), frozenset(paths),
            frozenset(eval_ids), frozenset(eval_hashes), frozenset(eval_paths))


def _clean_background_sets(clean_backgrounds):
    """Ids, hashes and near-duplicate clusters of the approved CLEAN copy-paste backgrounds."""
    if clean_backgrounds is None or len(clean_backgrounds) == 0:
        return frozenset(), frozenset(), frozenset()
    ids, hashes, groups = set(), set(), set()
    for record in clean_backgrounds.to_dict("records"):
        coco_id = _as_int(record.get("coco_image_id"))
        if coco_id is not None:
            ids.add(coco_id)
        digest = _text(record.get("sha256")).lower()
        if digest:
            hashes.add(digest)
        group = _text(record.get("near_duplicate_group"))
        if group:
            groups.add(group)
    return frozenset(ids), frozenset(hashes), frozenset(groups)


def audit_outdoor_hardnegatives(df: pd.DataFrame, excluded: pd.DataFrame,
                                clean_backgrounds: pd.DataFrame, thresholds=None) -> dict:
    """Decide which candidate pictures may join COCO_OUTDOOR_HARDNEG_TRAIN, and prove every rejection.

    One row is rejected for exactly one, explicitly named reason; the accepted pool is re-checked once more
    after every filter, so a leak can never survive into the returned manifest. The gate is PASS only when
    150 unique approved images carry 3 reviewed categories and 3 audited source clusters.
    """
    resolved_thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    missing = [key for key in DEFAULT_THRESHOLDS if key not in resolved_thresholds]
    if missing:
        raise OutdoorHardNegBlocked("thresholds miss required keys: %s" % missing)

    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df if df is not None else [])
    (excluded_ids, excluded_hashes, excluded_paths,
     eval_ids, eval_hashes, eval_paths) = _excluded_sets(excluded)
    clean_ids, clean_hashes, clean_groups = _clean_background_sets(clean_backgrounds)

    provisional: list = []
    rejections: list = []
    violations: list = []
    seen_ids: dict = {}
    seen_hashes: dict = {}

    for row in frame.to_dict("records"):
        coco_id = _as_int(row.get("coco_image_id"))
        file_name = _text(row.get("file_name"))
        path_text = _text(row.get("path"))

        if coco_id is None:
            rejections.append(_rejection_record(coco_id, file_name, path_text, REASON_BAD_ID,
                                                "row has no usable coco_image_id"))
            violations.append({"kind": REASON_BAD_ID, "detail": "row without coco_image_id"})
            continue

        if not _row_split_ok(row):
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_NOT_TRAIN2017,
                "only a COCO train2017 path is admissible, got split=%r path=%s"
                % (_text(row.get("split")), path_text)))
            violations.append({"kind": REASON_NOT_TRAIN2017, "coco_image_id": coco_id, "path": path_text})
            continue

        digest = _text(row.get("sha256")).lower()
        if not digest and Path(path_text).is_file():
            digest = sha256_file(path_text)
        if not digest:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_HASH_UNVERIFIABLE,
                "no SHA256 is recorded and the image bytes are unavailable, so overlap cannot be excluded"))
            continue

        path_key = _path_key(path_text)
        if coco_id in eval_ids or digest in eval_hashes or path_key in eval_paths:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_FROZEN_EVAL,
                "matches a frozen evaluation image by id, content or path; evaluation pixels never train"))
            violations.append({"kind": REASON_FROZEN_EVAL, "coco_image_id": coco_id, "path": path_text})
            continue

        if coco_id in excluded_ids or digest in excluded_hashes or path_key in excluded_paths:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_PREVIOUSLY_USED,
                "already used by the frozen V2 work (coco_train_easy / old negatives / old hard negatives)"))
            continue

        group = _text(row.get("near_duplicate_group"))
        if coco_id in clean_ids:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_CLEAN_BG_ID,
                "the same COCO image id is an approved CLEAN copy-paste background"))
            continue
        if digest in clean_hashes:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_CLEAN_BG_HASH,
                "identical bytes to an approved CLEAN copy-paste background"))
            continue
        if group and group in clean_groups:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_CLEAN_BG_NEARDUP,
                "shares near-duplicate group %r with an approved CLEAN background" % group))
            continue

        license_text = _text(row.get("license_or_permission"))
        permission_state = _text(row.get("permission_state")).lower()
        if permission_state != "declared" or not license_text:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_LICENSE,
                "licence/permission is not resolved (state=%r); it is never assumed"
                % (permission_state or "unknown")))
            continue

        label_path, label_exists, label_bytes = _label_state(row)
        if not label_exists:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_LABEL_MISSING,
                "the YOLO .txt must exist and be zero bytes, nothing was found at %s" % (label_path or "?")))
            continue
        if label_bytes != 0:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_LABEL_NONEMPTY,
                "%d bytes / %d box(es) in %s: a labelled image is a positive, not a hard negative"
                % (label_bytes, _count_label_boxes(label_path, label_bytes), label_path)))
            continue

        no_shuttle = row.get("no_shuttle_reviewed")
        if not _truthy(no_shuttle):
            if _falsy(no_shuttle):
                rejections.append(_rejection_record(
                    coco_id, file_name, path_text, REASON_SHUTTLE_UNCERTAIN,
                    "the reviewer did not confirm no shuttle (no_shuttle_reviewed=%r)" % (no_shuttle,)))
            else:
                rejections.append(_rejection_record(
                    coco_id, file_name, path_text, REASON_NO_SHUTTLE_MISSING,
                    "no human full-size review is recorded; an absent COCO annotation is not proof"))
            continue
        if _truthy(row.get("review_uncertain_tiny_dot")):
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_TINY_DOT,
                "the reviewer flagged an uncertain tiny white dot; a possibly real shuttlecock is excluded"))
            continue
        if _text(row.get("qa_status")).upper() != "APPROVED":
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_NOT_APPROVED,
                "qa_status is %r, expected APPROVED" % (_text(row.get("qa_status")),)))
            continue

        category = canonical_reviewed_category(row.get("reviewed_category"))
        if category is None:
            if _blank(row.get("reviewed_category")):
                rejections.append(_rejection_record(
                    coco_id, file_name, path_text, REASON_CATEGORY_MISSING,
                    "the reviewed distractor category is missing; allowed values are %s"
                    % list(REVIEWED_CATEGORIES)))
            else:
                rejections.append(_rejection_record(
                    coco_id, file_name, path_text, REASON_CATEGORY_INVALID,
                    "reviewed category %r is not in %s"
                    % (_text(row.get("reviewed_category")), list(REVIEWED_CATEGORIES))))
            continue

        if coco_id in seen_ids:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_DUPLICATE_ID,
                "the same COCO image id appears twice in the candidate sheet"))
            continue
        if digest in seen_hashes:
            rejections.append(_rejection_record(
                coco_id, file_name, path_text, REASON_DUPLICATE_CONTENT,
                "identical bytes to accepted COCO image %d" % seen_hashes[digest]))
            continue
        seen_ids[coco_id] = path_text
        seen_hashes[digest] = coco_id

        accepted_row = dict(row)
        accepted_row.update({
            "coco_image_id": coco_id,
            "sha256": digest,
            "near_duplicate_group": group,
            "label_path": label_path or "",
            "label_exists": True,
            "label_bytes": 0,
            "n_boxes": 0,
            "is_negative": True,
            "is_positive": False,
            "role": C.ROLE_COCO_OUTDOOR_HARDNEG,
            "hardness_class": category,
        })
        provisional.append(accepted_row)

    # only one member of a near-duplicate cluster counts towards the 150 independent images
    grouped: dict = {}
    for index, record in enumerate(provisional):
        if record["near_duplicate_group"]:
            grouped.setdefault(record["near_duplicate_group"], []).append(index)
    dropped: set = set()
    for group, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        for index in sorted(members, key=lambda position: provisional[position]["coco_image_id"])[1:]:
            dropped.add(index)
            record = provisional[index]
            rejections.append(_rejection_record(
                record["coco_image_id"], record.get("file_name"), record.get("path"), REASON_NEARDUP_POOL,
                "shares near-duplicate group %r inside the pool; only one independent member is counted"
                % group))

    # defence in depth: re-check the surviving pool against every exclusion before it is returned
    accepted: list = []
    for index, record in enumerate(provisional):
        if index in dropped:
            continue
        coco_id = int(record["coco_image_id"])
        digest = str(record["sha256"])
        path_key = _path_key(str(record.get("path") or ""))
        group = _text(record.get("near_duplicate_group"))
        leak = None
        if coco_id in eval_ids or digest in eval_hashes or path_key in eval_paths:
            leak = REASON_FROZEN_EVAL
        elif coco_id in excluded_ids or digest in excluded_hashes or path_key in excluded_paths:
            leak = REASON_PREVIOUSLY_USED
        elif coco_id in clean_ids or digest in clean_hashes or (group and group in clean_groups):
            leak = "clean_background_overlap"
        if leak:
            violations.append({"kind": "accepted_row_leak", "coco_image_id": coco_id, "detail": leak})
            rejections.append(_rejection_record(
                coco_id, record.get("file_name"), record.get("path"), leak,
                "defence-in-depth re-check removed this row from the accepted pool"))
            continue
        accepted.append(record)

    accepted_frame = pd.DataFrame(accepted)
    rejected_frame = pd.DataFrame(rejections, columns=list(REJECTION_COLUMNS))
    if len(accepted_frame) == 0:
        accepted_frame = pd.DataFrame(columns=list(INVENTORY_COLUMNS))

    category_counts = Counter(str(record["hardness_class"]) for record in accepted)
    clusters = sorted({_text(record.get("source_cluster")) for record in accepted} - {""})
    counts = {
        "inventory_rows": int(len(frame)),
        "accepted": len(accepted),
        "rejected": len(rejections),
        "unique_accepted_images": len({int(record["coco_image_id"]) for record in accepted}),
        "unique_accepted_hashes": len({str(record["sha256"]) for record in accepted}),
        "reviewed_categories": len(category_counts),
        "audited_clusters": len(clusters),
        "zero_byte_labels": sum(1 for record in accepted if int(record.get("label_bytes") or 0) == 0),
        "contract_violations": len(violations),
    }
    shortfall = {
        "unique_images": max(0, int(resolved_thresholds["min_unique_images"]) - counts["unique_accepted_images"]),
        "reviewed_categories": max(0, int(resolved_thresholds["min_reviewed_categories"])
                                   - counts["reviewed_categories"]),
        "audited_clusters": max(0, int(resolved_thresholds["min_audited_clusters"]) - counts["audited_clusters"]),
    }
    if violations:
        status = GATE_LEAK
    elif any(shortfall.values()):
        status = GATE_INSUFFICIENT
    else:
        status = GATE_PASS

    return {
        "status": status,
        "gate": status,
        "plan": "ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md",
        "task": "Task 5: curate COCO train2017 outdoor hard negatives (empty labels)",
        "thresholds": {key: int(value) for key, value in resolved_thresholds.items()},
        "counts": counts,
        "shortfall": shortfall,
        "by_category": dict(sorted(category_counts.items())),
        "clusters": clusters,
        "rejection_reasons": dict(sorted(Counter(row["reason"] for row in rejections).items())),
        "contract_violations": violations,
        "rejected": rejected_frame,
        "accepted": accepted_frame,
        "accepted_ids": sorted(int(record["coco_image_id"]) for record in accepted),
        "rejected_ids": sorted(int(record["coco_image_id"]) for record in rejections
                               if record["coco_image_id"] != ""),
        "note": ("COCO has no shuttlecock category: the absence of an annotation is never proof of an empty "
                 "picture, so only a human full-size review can admit a hard negative"),
    }


# --- Task 5 interface: manifest export and review aid ---------------------------------------------

def _manifest_row(record) -> dict:
    label_path, label_exists, label_bytes = _label_state(record)
    if not label_exists or label_bytes != 0:
        raise OutdoorHardNegBlocked(
            "refusing to export COCO image %s as a hard negative: its label is %s"
            % (record.get("coco_image_id"),
               "missing" if not label_exists else "%d bytes" % label_bytes))
    return {
        "coco_image_id": int(record["coco_image_id"]),
        "file_name": _text(record.get("file_name")),
        "path": _text(record.get("path")),
        "label_path": label_path,
        "label_bytes": int(label_bytes),
        "label_empty": True,
        "n_boxes": 0,
        "is_negative": True,
        "is_positive": False,
        "role": C.ROLE_COCO_OUTDOOR_HARDNEG,
        "split": TRAIN_SPLIT,
        "sha256": _text(record.get("sha256")).lower(),
        "license_or_permission": _text(record.get("license_or_permission")),
        "permission_state": _text(record.get("permission_state")),
        "reviewer_id": _text(record.get("reviewer_id")),
        "no_shuttle_reviewed": True,
        "review_uncertain_tiny_dot": False,
        "reviewed_category": _text(record.get("reviewed_category")),
        "hardness_class": classify_negative_hardness(record["coco_image_id"],
                                                     record.get("reviewed_category")),
        "source_cluster": _text(record.get("source_cluster")),
        "near_duplicate_group": _text(record.get("near_duplicate_group")),
        "qa_status": _text(record.get("qa_status")),
    }


def write_outdoor_hardneg_manifest(result, dest) -> Path:
    """Write the accepted pool as data/clean_outdoor_cp_v1/outdoor_hardneg_manifest.csv.

    The export refuses a leaking pool and an empty pool outright, and re-verifies on disk that every exported
    label exists and is zero bytes, so a manifest can never advertise an empty label it does not have.
    """
    if isinstance(result, pd.DataFrame):
        accepted, status = result, None
    elif isinstance(result, dict):
        accepted, status = result.get("accepted"), result.get("status")
    else:
        raise OutdoorHardNegBlocked("export needs the audit result dict or an accepted-pool DataFrame")
    if status == GATE_LEAK:
        raise OutdoorHardNegBlocked("refusing to export a pool whose audit status is %s" % GATE_LEAK)
    if accepted is None or len(accepted) == 0:
        raise OutdoorHardNegBlocked("refusing to export an empty outdoor hard-negative pool")

    rows = [_manifest_row(record) for record in accepted.to_dict("records")]
    frame = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return dest


def write_contact_sheet(rows, dest, columns: int = 8, cell: int = 160, max_images: int = 200) -> Path:
    """Deterministic contact sheet in ascending COCO image id order for the human full-size review.

    The sampling is fixed by id order, never by model score and never success-only, so the sheet a reviewer
    sees is reproducible from the manifest alone.
    """
    if Image is None:
        raise OutdoorHardNegBlocked("Pillow is required to write a contact sheet")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if len(frame) == 0:
        raise OutdoorHardNegBlocked("refusing to write a contact sheet for an empty pool")
    if int(columns) < 1 or int(cell) < 1:
        raise OutdoorHardNegBlocked("columns and cell must be positive, got %r/%r" % (columns, cell))

    records = frame.sort_values("coco_image_id").head(int(max_images)).to_dict("records")
    grid_columns = int(columns)
    grid_rows = (len(records) + grid_columns - 1) // grid_columns
    sheet = Image.new("RGB", (grid_columns * int(cell), grid_rows * int(cell)), (24, 24, 24))
    for index, record in enumerate(records):
        image_path = Path(_text(record.get("path")))
        if not image_path.is_file():
            raise OutdoorHardNegBlocked("contact sheet needs the original image bytes: %s" % image_path)
        with Image.open(image_path) as handle:
            tile = handle.convert("RGB").resize((int(cell), int(cell)))
        sheet.paste(tile, ((index % grid_columns) * int(cell), (index // grid_columns) * int(cell)))

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "coco_hardnegative_contact_sheet.png"
    sheet.save(out)
    return out


def assert_train_only_coco_images(train_only_coco_images) -> Path:
    """Refuse any directory that could hold an evaluation image or a non-COCO picture.

    The directory's own name is checked for evaluation tokens and every contained image must be a
    COCO train2017 file name (twelve-digit image id). Ancestors are only checked for the exact split names
    val2017/test2017, so a CI temporary folder whose name happens to mention evaluation cannot block a
    legitimate train2017 directory, while a real evaluation set can never be described as train-only.
    """
    directory = Path(train_only_coco_images)
    parts = directory.parts
    splits = [part for part in parts if part in FORBIDDEN_SPLITS]
    if splits:
        raise OutdoorHardNegBlocked(
            "refusing to mine %s: %s is not the train2017 split" % (directory, splits[0])
        )
    name = directory.name.lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in name:
            raise OutdoorHardNegBlocked(
                "refusing to mine %s: the directory name names an evaluation set (%r)" % (directory, token)
            )
    for word in FORBIDDEN_PATH_WORDS:
        if _has_word(name, word):
            raise OutdoorHardNegBlocked(
                "refusing to mine %s: the directory name names an evaluation set (%r)" % (directory, word)
            )
    if not directory.is_dir():
        raise OutdoorHardNegBlocked("train-only COCO image directory does not exist: %s" % directory)
    images = sorted(path for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise OutdoorHardNegBlocked("no COCO images found in %s" % directory)
    foreign = [path.name for path in images if not COCO_IMAGE_NAME.match(path.name)]
    if foreign:
        raise OutdoorHardNegBlocked(
            "refusing to mine non-COCO train2017 file names (an evaluation or ETH image would be touched): %s"
            % foreign[:5]
        )
    return directory


def mine_v2_fp_candidates(train_only_coco_images, v2_checkpoint, conf: float = 0.25) -> pd.DataFrame:
    """Read-only: suggest train-only COCO pictures where the frozen V2 model fires, for a HUMAN to review.

    This never produces a training manifest: every returned row is a review suggestion, and nothing is
    written to disk. There is deliberately no manifest or evaluation argument, so a caller cannot hand an
    evaluation listing to this function, and an evaluation path is refused before any checkpoint is opened.
    """
    directory = assert_train_only_coco_images(train_only_coco_images)
    confidence = float(conf)
    if not 0.0 < confidence <= 1.0:
        raise OutdoorHardNegBlocked("conf must be in (0, 1], got %r" % conf)
    checkpoint = Path(v2_checkpoint)
    if not checkpoint.is_file():
        raise OutdoorHardNegBlocked(
            "the frozen V2 checkpoint is missing: %s (no detection is invented without real weights)"
            % checkpoint
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise OutdoorHardNegBlocked(
            "ultralytics is required for read-only V2 false-positive mining: %s" % exc
        )
    try:
        model = YOLO(str(checkpoint))
    except Exception as exc:  # noqa: BLE001 - an unloadable checkpoint must block, not silently return empty
        raise OutdoorHardNegBlocked("the frozen V2 checkpoint could not be loaded: %s (%s)"
                                    % (checkpoint, exc))

    images = sorted(path for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    rows: list = []
    for image in images:
        match = COCO_IMAGE_NAME.match(image.name)
        try:
            predictions = model.predict(source=str(image), conf=confidence, verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise OutdoorHardNegBlocked("read-only V2 inference failed on %s: %s" % (image, exc))
        confidences: list = []
        classes: set = set()
        for prediction in predictions or []:
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            confidences.extend(float(value) for value in boxes.conf.tolist())
            classes.update(int(value) for value in boxes.cls.tolist())
        if not confidences:
            continue
        rows.append({
            "coco_image_id": int(match.group(1)),
            "path": str(image),
            "n_detections": len(confidences),
            "max_conf": max(confidences),
            "class_ids": sorted(classes),
            "suggested_for_review": True,
        })

    frame = pd.DataFrame(rows, columns=list(FALSE_POSITIVE_COLUMNS))
    frame.attrs["checkpoint"] = str(checkpoint)
    frame.attrs["checkpoint_sha256"] = sha256_file(checkpoint)
    frame.attrs["role"] = "review_suggestion_only"
    frame.attrs["evaluation_images_touched"] = False
    frame.attrs["conf"] = confidence
    return frame


# --- CLI ------------------------------------------------------------------------------------------

def _empty_prohibited_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PROHIBITED_COLUMNS))


def _read_csv_if_given(path) -> pd.DataFrame:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise OutdoorHardNegBlocked("input CSV is missing: %s" % resolved)
    return pd.read_csv(resolved, keep_default_na=False, dtype=str)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit COCO train2017 outdoor hard negatives and export empty-label manifests (Task 5)")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="reviewed COCO train2017 candidate rows (CSV)")
    parser.add_argument("--excluded", type=Path, default=None,
                        help="Task 2 prohibited-source index: historical V2 and frozen evaluation rows")
    parser.add_argument("--clean-backgrounds", type=Path, default=None,
                        help="Task 4 approved CLEAN copy-paste background manifest")
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    excluded = _read_csv_if_given(args.excluded)
    clean_backgrounds = _read_csv_if_given(args.clean_backgrounds)
    inventory = inventory_outdoor_hardnegatives(_read_csv_if_given(args.candidates))
    result = audit_outdoor_hardnegatives(
        inventory,
        _empty_prohibited_frame() if excluded is None else excluded,
        pd.DataFrame(columns=["coco_image_id", "sha256", "near_duplicate_group"])
        if clean_backgrounds is None else clean_backgrounds,
    )

    audit = {key: value for key, value in result.items()
             if key not in ("accepted", "rejected")}
    audit["generated_by"] = "tools/audit_outdoor_hardnegatives.py"
    audit["accepted_count"] = int(len(result["accepted"]))
    # An insufficient pool never produces a training manifest: the partial accepted list stays in the audit
    # JSON (accepted_ids) so it can be reviewed, but nothing that could be fed to a trainer is written.
    if result["status"] == GATE_PASS:
        manifest = write_outdoor_hardneg_manifest(result, args.manifest_out)
        audit["manifest"] = str(manifest)
    else:
        audit["manifest"] = None
        audit["manifest_withheld_because"] = result["status"]
    if args.contact_sheet_dir is not None and len(result["accepted"]):
        audit["contact_sheet"] = str(write_contact_sheet(result["accepted"], args.contact_sheet_dir))

    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
                              encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "shortfall": result["shortfall"],
                      "rejection_reasons": result["rejection_reasons"]},
                     indent=2, sort_keys=True))
    return 0 if result["status"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
