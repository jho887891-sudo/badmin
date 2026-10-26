#!/usr/bin/env python3
"""Real ETH train-only foreground instances and mandatory-review mask QA (Task 3).

The copy-paste positives of CLEAN_OUTDOOR_CP_V1 may only be built from ETH *train-only* real shuttlecock
RGB plus a human-approved instance mask. Four things are therefore refused as a matter of contract:

* a detection bbox is not a segmentation mask (an alpha equal to the box is rejected);
* a mask that does not share the RGB frame geometry cannot be pasted into that frame;
* a small source crop may not be magnified into invented feather detail (the upscale factor is checked,
  and an unknown source size is rejected rather than assumed to be native);
* an unreviewed proposal is never approved - reviewed must be the explicit boolean True of a human
  sign-off, never a truthy string, 1 or a missing field.

The module also owns the honest stock gate: fewer than 100 approved instances across at least 2 train
locations/sessions reports FOREGROUND_INSUFFICIENT instead of handing a thin pool to the generator.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 3).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: The plan's stock gate for the real foreground pool (Task 3 Step 4 / Preflight matrix).
GATE_PASS = "PASS"
GATE_INSUFFICIENT = "FOREGROUND_INSUFFICIENT"
MIN_APPROVED_INSTANCES = 100
MIN_DISTINCT_LOCATIONS = 2

DEFAULT_MIN_SHORT = 10
DEFAULT_MIN_LONG = 24
DEFAULT_MAX_UPSCALE = 1.0

STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_PENDING_REVIEW = "PENDING_REVIEW"

#: The repository's frozen evaluation index (Task 2 output): every frozen eval image with its content hash.
FROZEN_EVAL_INDEX_PATH = "data/clean_outdoor_cp_v1/frozen_eval_index.csv"

#: The frozen V2 manifest columns this task derives its candidates from.
REQUIRED_MANIFEST_COLUMNS = (
    "image", "label", "location", "split", "set_dir", "role", "source", "is_negative", "is_positive",
    "n_boxes", "width", "height", "sha256",
)

INVENTORY_COLUMNS = (
    "instance_id", "image", "label", "location", "set_dir", "split", "source", "sha256", "role",
    "frame_width", "frame_height", "bbox_xyxy", "bbox_wh", "bbox_short", "bbox_long", "n_boxes",
    "status", "reviewed", "reviewer_id", "review_method", "mask_path", "mask_sha256",
)

#: Review columns that are legitimately blank before a human has looked at the mask; the geometry columns
#: are not in this set, so a candidate without a real box can never be written silently.
REVIEW_COLUMNS = frozenset({
    "status", "reviewed", "reviewer_id", "review_method", "mask_path", "mask_sha256", "role",
})

#: set_dir / location / file-stem tokens that mark a frozen evaluation set (plan Global Constraints 4 and 9).
EVAL_TOKENS = frozenset({
    "val", "valid", "validation", "test", "tests", "eval", "evaluation", "unseen", "eth_unseen",
    "challenge", "challenge_test", "controlled", "controlled_capability", "no_target",
    "real_no_target_496", "ood", "holdout", "hold_out",
})
#: Provenance tokens that are not an ETH real capture (no COCO, iPhone, D455, Roboflow or 3D asset).
NON_ETH_SOURCE_TOKENS = frozenset({
    "coco", "roboflow", "iphone", "d455", "isaac", "synthetic", "render", "rendered", "blender", "3d",
    "unreal", "gazebo", "mujoco", "sim",
})
#: A row's source must name the frozen ETH capture; anything else is not an ETH real positive.
ETH_SOURCE_PREFIX = "eth"

UPSAMPLE_TOLERANCE = 1e-9
RECTANGLE_FILL_RATIO = 0.98
BBOX_TOLERANCE_PX = 1


class EthForegroundBlocked(RuntimeError):
    """Raised when the frozen V2 manifest or the ETH label mirror cannot support the selection at all."""


class ForegroundInsufficient(RuntimeError):
    """Raised when the approved foreground pool is too thin to proceed (FOREGROUND_INSUFFICIENT)."""


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _tokens(*values) -> set:
    tokens: set = set()
    for value in values:
        if value is None:
            continue
        tokens.update(part for part in re.split(r"[^a-z0-9]+", str(value).strip().lower()) if part)
    return tokens


def _require_columns(fieldnames, path) -> None:
    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(fieldnames or ()))
    if missing:
        raise EthForegroundBlocked(
            "V2 manifest %s misses required columns %s; Task 3 reads the frozen V2 manifest schema, not a "
            "hand-made list" % (path, missing)
        )


def read_manifest(path) -> list:
    """Read the frozen V2 manifest as raw row dicts, refusing any schema drift before a row is used."""
    resolved = Path(path)
    if not resolved.is_file():
        raise EthForegroundBlocked("V2 train manifest is missing: %s" % resolved)
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, resolved)
        rows = list(reader)
    if not rows:
        raise EthForegroundBlocked("V2 train manifest is empty: %s" % resolved)
    return rows


def load_frozen_eval_index(manifests) -> dict:
    """Collect every frozen evaluation image the new foregrounds must never touch: path, sha256 and stem."""
    paths: set = set()
    digests: set = set()
    stems: set = set()
    for manifest in manifests or ():
        for row in read_manifest_loose(manifest):
            image = str(row.get("image") or row.get("abs_path") or row.get("path")
                        or row.get("source_id") or "").strip()
            digest = str(row.get("sha256") or "").strip()
            if image:
                paths.add(image)
                stems.add(Path(image).stem)
            if digest:
                digests.add(digest)
    return {"paths": paths, "sha256": digests, "stems": stems}


def read_manifest_loose(path) -> list:
    """Read any manifest CSV for the leakage firewall: the frozen eval manifests do not use the V2 schema."""
    resolved = Path(path)
    if not resolved.is_file():
        raise EthForegroundBlocked("frozen evaluation manifest is missing: %s" % resolved)
    with open(resolved, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_label_path(row: dict, label_root) -> Path:
    """Locate the YOLO label of one ETH image. An unmounted mirror is a blocker, never an empty selection."""
    raw = str(row.get("label") or row.get("abs_label_path") or "").strip()
    tried: list = []
    if raw:
        tried.append(Path(raw))
    if label_root is not None:
        base = Path(label_root)
        if raw:
            tried.append(base / Path(raw).name)
        set_dir = str(row.get("set_dir") or "").strip()
        if raw and set_dir:
            tried.append(base / set_dir / "labels" / "train" / Path(raw).name)
    for candidate in tried:
        if candidate.is_file():
            return candidate
    raise EthForegroundBlocked(
        "no readable YOLO label for image %s; looked at %s. The frozen V2 manifest points at the remote "
        "/home/T7 ETH mirror, so the label bytes must be mounted locally and passed as label_root=... "
        "(BLOCKED_FOREGROUND_LABELS_MISSING)" % (row.get("image"), [str(item) for item in tried])
    )


def parse_yolo_label(path) -> list:
    """Parse a single-class YOLO label into normalised (cx, cy, w, h). Malformed content is a hard error."""
    boxes: list = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise EthForegroundBlocked(
                "%s line %d is not a 'class cx cy w h' row: %r" % (path, number, stripped)
            )
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            raise EthForegroundBlocked("%s line %d is not numeric: %r" % (path, number, stripped))
        if class_id != 0:
            raise EthForegroundBlocked(
                "%s line %d uses class id %d; the single-class contract allows only class 0"
                % (path, number, class_id)
            )
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            raise EthForegroundBlocked("%s line %d has a non-finite coordinate" % (path, number))
        if width <= 0.0 or height <= 0.0:
            raise EthForegroundBlocked("%s line %d has a non-positive box size" % (path, number))
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise EthForegroundBlocked("%s line %d has a centre outside the image" % (path, number))
        boxes.append((cx, cy, width, height))
    return boxes


def _pixel_box(box, frame_width: int, frame_height: int) -> list:
    cx, cy, width, height = box
    x1 = int(round((cx - width / 2.0) * frame_width))
    y1 = int(round((cy - height / 2.0) * frame_height))
    x2 = int(round((cx + width / 2.0) * frame_width))
    y2 = int(round((cy + height / 2.0) * frame_height))
    return [max(0, min(frame_width, x1)), max(0, min(frame_height, y1)),
            max(0, min(frame_width, x2)), max(0, min(frame_height, y2))]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_eth_train_instances(v2_manifest, min_short: int = DEFAULT_MIN_SHORT, min_long: int = DEFAULT_MIN_LONG,
                              *, label_root=None, frozen_eval_manifests=(),
                              include_all_sizes: bool = False, require_image_files: bool = False) -> dict:
    """Select ETH train-only real positives and record a reason for every row that was not selected.

    Every rejection is enumerated, so an evaluation image that never becomes a foreground leaves a trace
    instead of disappearing. A candidate needs an existing mask review before it is approved; this function
    only produces PENDING_REVIEW instances.
    """
    if min_short <= 0 or min_long <= 0:
        raise EthForegroundBlocked("min_short/min_long must be positive, got %r/%r" % (min_short, min_long))
    if min_short > min_long:
        raise EthForegroundBlocked(
            "min_short=%d is larger than min_long=%d: the short side can never exceed the long side"
            % (min_short, min_long)
        )
    manifest_path = Path(v2_manifest)
    rows = read_manifest(manifest_path)
    frozen = load_frozen_eval_index(frozen_eval_manifests)

    instances: list = []
    rejections: list = []
    eligible_train_label_boxes = 0
    invalid_box_count = 0
    for row in rows:
        image = str(row.get("image") or "").strip()
        stem = Path(image).stem if image else ""

        def reject(reason: str, detail: str = "") -> None:
            rejections.append({"image": image, "reason": reason, "detail": detail})

        if _truthy(row.get("is_negative")) or not _truthy(row.get("is_positive")):
            reject("not_positive", "row is not a positive ETH shuttlecock instance")
            continue
        split = str(row.get("split") or "").strip().lower()
        if not split:
            reject("split_missing", "row carries no split, so train-only selection cannot be proven")
            continue
        if split != "train":
            reject("eval_split", "split %r is not the frozen train split" % split)
            continue
        eval_tokens = _tokens(row.get("set_dir"), row.get("location"), stem) & EVAL_TOKENS
        if eval_tokens:
            reject("eval_set", "evaluation token(s) %s in set_dir/location/file name" % sorted(eval_tokens))
            continue
        source = str(row.get("source") or "").strip()
        foreign = _tokens(source, row.get("set_dir")) & NON_ETH_SOURCE_TOKENS
        if foreign or ETH_SOURCE_PREFIX not in _tokens(source):
            reject("non_eth_source",
                   "source=%r is not an ETH real capture (offending tokens: %s)" % (source, sorted(foreign)))
            continue
        digest = str(row.get("sha256") or "").strip()
        if (image and image in frozen["paths"]) or (digest and digest in frozen["sha256"]) or \
                (stem and stem in frozen["stems"]):
            reject("eval_content", "image path, content hash or file stem matches a frozen evaluation image")
            continue

        try:
            frame_width = int(float(row.get("width") or 0))
            frame_height = int(float(row.get("height") or 0))
        except ValueError:
            raise EthForegroundBlocked("image %s has a non-numeric width/height" % image)
        if frame_width <= 0 or frame_height <= 0:
            raise EthForegroundBlocked("image %s has no usable frame size: %dx%d" % (image, frame_width, frame_height))

        if include_all_sizes or require_image_files:
            image_path = Path(image)
            if not image_path.is_file():
                raise EthForegroundBlocked("raw ETH image missing: %s (BLOCKED_FOREGROUND_IMAGES_MISSING)" % image)
            try:
                with Image.open(image_path) as image_obj:
                    actual_width, actual_height = image_obj.size
                    image_obj.verify()
            except Exception as exc:
                raise EthForegroundBlocked("raw ETH image unreadable: %s (%s)" % (image, exc))
            if (actual_width, actual_height) != (frame_width, frame_height):
                raise EthForegroundBlocked("manifest/image dimensions disagree for %s: manifest=%dx%d actual=%dx%d" %
                                           (image, frame_width, frame_height, actual_width, actual_height))
            actual_digest = _sha256_path(image_path)
            if include_all_sizes:
                if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    raise EthForegroundBlocked("missing valid SHA256 in V2 manifest for %s" % image)
                if actual_digest.lower() != digest.lower():
                    raise EthForegroundBlocked("raw ETH image SHA256 mismatch for %s: manifest=%s actual=%s" %
                                               (image, digest, actual_digest))
            if actual_digest in frozen["sha256"] or image in frozen["paths"] or stem in frozen["stems"]:
                reject("eval_content", "actual raw-image SHA256/path/stem matches a frozen evaluation image")
                continue
            digest = actual_digest

        label_path = resolve_label_path(row, label_root)
        boxes = parse_yolo_label(label_path)
        declared = str(row.get("n_boxes") or "").strip()
        if declared and int(float(declared)) != len(boxes):
            raise EthForegroundBlocked(
                "manifest n_boxes=%s disagrees with %d parsed box(es) in %s" % (declared, len(boxes), label_path)
            )
        if not boxes:
            reject("no_boxes", "label %s carries no instance" % label_path)
            continue

        eligible_train_label_boxes += len(boxes)
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = _pixel_box(box, frame_width, frame_height)
            box_width, box_height = x2 - x1, y2 - y1
            short, long_side = min(box_width, box_height), max(box_width, box_height)
            instance_id = ("%s#%d" % (digest, index)) if include_all_sizes else ("%s#%d" % (stem, index))
            if box_width <= 0 or box_height <= 0:
                invalid_box_count += 1
                rejections.append({"image": image, "instance_id": instance_id,
                                   "reason": "quantized_empty_bbox",
                                   "detail": "box %dx%d px after pixel quantization" % (box_width, box_height),
                                   "bbox_xyxy": [x1, y1, x2, y2]})
                continue
            if not include_all_sizes and (short < min_short or long_side < min_long):
                reject("below_min_size",
                       "%s#%d is %dx%d px, below the %d/%d px floors" % (stem, index, box_width, box_height,
                                                                        min_short, min_long))
                continue
            instances.append({
                "instance_id": instance_id,
                "image": image,
                "label": str(label_path),
                "location": str(row.get("location") or "").strip(),
                "set_dir": str(row.get("set_dir") or "").strip(),
                "split": split,
                "source": source,
                "sha256": digest,
                "role": C.ROLE_ETH_TRAIN_FOREGROUND,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_wh": [box_width, box_height],
                "bbox_short": short,
                "bbox_long": long_side,
                "n_boxes": len(boxes),
                "status": STATUS_PENDING_REVIEW,
                "reviewed": False,
                "reviewer_id": "",
                "review_method": "",
                "mask_path": "",
                "mask_sha256": "",
            })

    by_reason: dict = {}
    for rejection in rejections:
        by_reason[rejection["reason"]] = by_reason.get(rejection["reason"], 0) + 1

    return {
        "manifest": str(manifest_path),
        "min_short": int(min_short),
        "min_long": int(min_long),
        "include_all_sizes": bool(include_all_sizes),
        "rows": len(rows),
        "eligible_train_label_boxes": eligible_train_label_boxes,
        "invalid_box_count": invalid_box_count,
        "instances": instances,
        "rejections": rejections,
        "rejections_by_reason": by_reason,
        "locations": sorted({item["location"] for item in instances if item["location"]}),
        "gate": GATE_INSUFFICIENT if not instances else "CANDIDATES_PENDING_REVIEW",
        "note": "candidates are PENDING_REVIEW until a human mask review approves them; selection alone "
                "never authorises training",
    }


def select_eth_train_instances(v2_manifest, min_short: int = DEFAULT_MIN_SHORT,
                               min_long: int = DEFAULT_MIN_LONG, *, label_root=None,
                               frozen_eval_manifests=()) -> list:
    """ETH train-only real positives above the pixel floors; never an evaluation image or val split."""
    return audit_eth_train_instances(
        v2_manifest, min_short=min_short, min_long=min_long,
        label_root=label_root, frozen_eval_manifests=frozen_eval_manifests,
    )["instances"]


def _validate_bbox(image, bbox) -> tuple:
    try:
        x1, y1, x2, y2 = (int(value) for value in bbox)
    except (TypeError, ValueError):
        raise EthForegroundBlocked("bbox must be four integers (x1, y1, x2, y2), got %r" % (bbox,))
    height, width = image.shape[:2]
    if x2 <= x1 or y2 <= y1:
        raise EthForegroundBlocked("bbox %r has no positive area" % (bbox,))
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise EthForegroundBlocked("bbox %r falls outside the %dx%d RGB frame" % (bbox, width, height))
    return x1, y1, x2, y2


def validate_foreground_mask(image, bbox, alpha, reviewed, *, source_size=None,
                             max_upscale: float = DEFAULT_MAX_UPSCALE) -> dict:
    """QA one proposed foreground mask. Returns APPROVED only when every contract clause holds.

    The result always carries the measurements behind the verdict (mask footprint, fill ratio, upscale
    factor) so a rejection can be audited without re-running the reviewer.
    """
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise EthForegroundBlocked("image must be an HxWx3 RGB array, got shape %r" % (frame.shape,))
    mask = np.asarray(alpha)
    if mask.ndim != 2:
        raise EthForegroundBlocked("alpha must be a 2-D array, got shape %r" % (mask.shape,))
    x1, y1, x2, y2 = _validate_bbox(frame, bbox)
    box_width, box_height = x2 - x1, y2 - y1

    reasons: list = []
    image_size = (int(frame.shape[1]), int(frame.shape[0]))
    mask_size = (int(mask.shape[1]), int(mask.shape[0]))
    if mask_size != image_size:
        reasons.append("mask_dimensions_match_rgb")

    if reviewed is not True:
        reasons.append("reviewed_required")

    foreground = mask > 0
    alpha_bbox = None
    foreground_pixels = 0
    fill_ratio = 0.0
    matches_bbox = False
    if not foreground.any():
        reasons.append("mask_empty")
    else:
        rows, columns = np.nonzero(foreground)
        alpha_bbox = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        foreground_pixels = int(foreground.sum())
        footprint_area = (alpha_bbox[2] - alpha_bbox[0]) * (alpha_bbox[3] - alpha_bbox[1])
        fill_ratio = foreground_pixels / float(footprint_area) if footprint_area else 0.0
        matches_bbox = alpha_bbox == (x1, y1, x2, y2)
        outside = (alpha_bbox[0] < x1 - BBOX_TOLERANCE_PX or alpha_bbox[1] < y1 - BBOX_TOLERANCE_PX
                   or alpha_bbox[2] > x2 + BBOX_TOLERANCE_PX or alpha_bbox[3] > y2 + BBOX_TOLERANCE_PX)
        if outside:
            reasons.append("mask_outside_bbox")
        if matches_bbox and fill_ratio >= RECTANGLE_FILL_RATIO:
            # The alpha is exactly the detection rectangle: this is a bbox crop, not a segmentation.
            reasons.append("bbox_crop_not_mask")
        elif fill_ratio >= RECTANGLE_FILL_RATIO:
            reasons.append("opaque_rectangular_mask")

    upscale_factor = None
    if source_size is None:
        reasons.append("source_size_unknown")
    else:
        try:
            source_width, source_height = (int(value) for value in source_size)
        except (TypeError, ValueError):
            raise EthForegroundBlocked("source_size must be (width, height) in pixels, got %r" % (source_size,))
        if source_width <= 0 or source_height <= 0:
            raise EthForegroundBlocked("source_size must be positive, got %r" % (source_size,))
        upscale_factor = max(box_width / float(source_width), box_height / float(source_height))
        if upscale_factor > float(max_upscale) + UPSAMPLE_TOLERANCE:
            reasons.append("upscaled_source_crop")

    return {
        "status": STATUS_APPROVED if not reasons else STATUS_REJECTED,
        "reasons": reasons,
        "image_size": image_size,
        "mask_size": mask_size,
        "bbox": (x1, y1, x2, y2),
        "bbox_wh": (box_width, box_height),
        "source_size": (int(source_size[0]), int(source_size[1])) if source_size is not None else None,
        "max_upscale": float(max_upscale),
        "upscale_factor": upscale_factor,
        "alpha_bbox": alpha_bbox,
        "foreground_pixels": foreground_pixels,
        "bbox_fill_ratio": fill_ratio,
        "alpha_matches_bbox_rectangle": bool(matches_bbox),
        "reviewed": bool(reviewed is True),
    }


def _require_candidate(candidate) -> dict:
    if not isinstance(candidate, dict):
        raise EthForegroundBlocked("every foreground candidate must be a mapping, got %r" % (candidate,))
    instance_id = str(candidate.get("instance_id") or "").strip()
    image = str(candidate.get("image") or "").strip()
    if not instance_id or not image:
        raise EthForegroundBlocked("a foreground candidate needs instance_id and image: %r" % (candidate,))
    missing = [column for column in INVENTORY_COLUMNS
               if column not in REVIEW_COLUMNS and candidate.get(column) in (None, "")]
    if missing:
        raise EthForegroundBlocked(
            "foreground candidate %s misses geometry column(s) %s; no field is defaulted silently"
            % (instance_id, missing)
        )
    if str(candidate.get("status") or "").upper() == STATUS_APPROVED:
        if not str(candidate.get("reviewer_id") or "").strip():
            raise EthForegroundBlocked(
                "candidate %s claims APPROVED without a reviewer_id: a mask cannot be approved by a program"
                % instance_id
            )
        if not str(candidate.get("review_method") or "").strip():
            raise EthForegroundBlocked(
                "candidate %s claims APPROVED without a review_method" % instance_id
            )
    return candidate


def _cell(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return "" if value is None else str(value)


def write_foreground_inventory(candidates: list, dest) -> Path:
    """Write the foreground inventory CSV. Refuses duplicated instances and unreviewed APPROVED claims."""
    rows = [_require_candidate(candidate) for candidate in candidates]
    seen: set = set()
    for row in rows:
        instance_id = str(row["instance_id"]).strip()
        if instance_id in seen:
            raise EthForegroundBlocked("duplicate foreground instance_id in the inventory: %s" % instance_id)
        seen.add(instance_id)

    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INVENTORY_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in INVENTORY_COLUMNS})
    return destination


def _read_inventory(inventory) -> list:
    if isinstance(inventory, (str, Path)):
        path = Path(inventory)
        if not path.is_file():
            raise EthForegroundBlocked("foreground inventory is missing: %s" % path)
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if isinstance(inventory, dict):
        raise EthForegroundBlocked("the inventory must be a list of rows or a CSV path, got a mapping")
    return [row for row in inventory]


def foreground_gate(inventory, mask_root=None) -> dict:
    """The honest stock gate: below 100 reviewed instances (or < 2 train locations) it stays insufficient.

    An APPROVED row only counts when a reviewer and a review method are recorded and, when mask_root is
    given, when the mask file exists and still hashes to the value recorded at review time.
    """
    rows = _read_inventory(inventory)
    root = Path(mask_root) if mask_root is not None else None
    approved: list = []
    unverified: list = []
    for row in rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if str(row.get("status") or "").upper() != STATUS_APPROVED:
            continue
        if str(row.get("reviewed") or "").strip().lower() not in {"true", "1", "yes"}:
            unverified.append({"instance_id": instance_id, "reason": "reviewed_flag_missing"})
            continue
        if not str(row.get("reviewer_id") or "").strip():
            unverified.append({"instance_id": instance_id, "reason": "reviewer_missing"})
            continue
        if not str(row.get("review_method") or "").strip():
            unverified.append({"instance_id": instance_id, "reason": "review_method_missing"})
            continue
        if root is not None:
            mask_path = str(row.get("mask_path") or "").strip()
            if not mask_path:
                unverified.append({"instance_id": instance_id, "reason": "mask_path_missing"})
                continue
            candidate = Path(mask_path)
            if not candidate.is_file():
                candidate = root / candidate.name
            if not candidate.is_file():
                unverified.append({"instance_id": instance_id, "reason": "mask_missing",
                                   "mask_path": mask_path})
                continue
            recorded = str(row.get("mask_sha256") or "").strip()
            actual = C.sha256_file(candidate)
            if not recorded or recorded != actual:
                unverified.append({"instance_id": instance_id, "reason": "mask_sha_mismatch",
                                   "mask_path": str(candidate), "recorded": recorded, "actual": actual})
                continue
        approved.append(row)

    locations = sorted({str(row.get("location") or "").strip() for row in approved
                        if str(row.get("location") or "").strip()})
    reasons: list = []
    if len(approved) < MIN_APPROVED_INSTANCES:
        reasons.append("only %d approved instance(s), %d required" % (len(approved), MIN_APPROVED_INSTANCES))
    if len(locations) < MIN_DISTINCT_LOCATIONS:
        reasons.append("only %d distinct train location(s), %d required" % (len(locations), MIN_DISTINCT_LOCATIONS))
    if unverified:
        reasons.append("%d APPROVED row(s) could not be verified" % len(unverified))

    return {
        "gate": GATE_PASS if not reasons else GATE_INSUFFICIENT,
        "reasons": reasons,
        "rows": len(rows),
        "approved": len(approved),
        "approved_required": MIN_APPROVED_INSTANCES,
        "distinct_locations": locations,
        "locations_required": MIN_DISTINCT_LOCATIONS,
        "unverified": unverified,
        "rule": ">=100 human-approved ETH train-only masks across >=2 train locations/sessions, each mask "
                "hash-verified against the review record",
    }


def assert_foreground_sufficient(inventory, mask_root=None) -> dict:
    """Raise ForegroundInsufficient instead of letting a thin foreground pool reach the generator."""
    report = foreground_gate(inventory, mask_root=mask_root)
    if report["gate"] != GATE_PASS:
        raise ForegroundInsufficient(
            "%s: %s (approved=%d/%d, locations=%s)"
            % (GATE_INSUFFICIENT, "; ".join(report["reasons"]) or "gate not satisfied",
               report["approved"], report["approved_required"], report["distinct_locations"])
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Select ETH train-only real foreground instances with mandatory mask review (Task 3)"
    )
    parser.add_argument("--manifest", type=Path, default=REPO / "data/eth_real_hardneg_v2_train_manifest.csv")
    parser.add_argument("--label-root", type=Path, default=None,
                        help="local mirror of the ETH labels (the V2 manifest points at the remote mirror)")
    parser.add_argument("--frozen-eval-manifest", type=Path, action="append",
                        default=[REPO / FROZEN_EVAL_INDEX_PATH],
                        help="frozen evaluation manifest whose images/paths/hashes must never be selectable; "
                             "the repository's frozen eval index is the default and a missing index blocks")
    parser.add_argument("--min-short", type=int, default=DEFAULT_MIN_SHORT)
    parser.add_argument("--min-long", type=int, default=DEFAULT_MIN_LONG)
    parser.add_argument("--out-inventory", type=Path,
                        default=REPO / "data/clean_outdoor_cp_v1/foreground_manifest.csv")
    parser.add_argument("--out-audit", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None,
                        help="an existing reviewed inventory to gate (no selection is run)")
    parser.add_argument("--mask-root", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.inventory is not None:
        report = foreground_gate(args.inventory, mask_root=args.mask_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["gate"] == GATE_PASS else 1

    try:
        audit = audit_eth_train_instances(
            args.manifest, min_short=args.min_short, min_long=args.min_long,
            label_root=args.label_root, frozen_eval_manifests=args.frozen_eval_manifest,
        )
    except EthForegroundBlocked as exc:
        print(json.dumps({"gate": "BLOCKED_FOREGROUND_SELECTION", "error": str(exc)}, indent=2))
        return 2

    dest = write_foreground_inventory(audit["instances"], args.out_inventory)
    report = foreground_gate(audit["instances"])
    report["inventory"] = str(dest)
    report["manifest"] = audit["manifest"]
    report["rejections_by_reason"] = audit["rejections_by_reason"]
    report["candidate_instances"] = len(audit["instances"])
    report["candidate_locations"] = audit["locations"]
    report["contact_sheet_dir"] = str(REPO / "outputs/shuttle_capability/clean_outdoor_cp_v1/contact_sheets"
                                             "/foreground_masks")
    if args.out_audit is not None:
        args.out_audit.parent.mkdir(parents=True, exist_ok=True)
        args.out_audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
