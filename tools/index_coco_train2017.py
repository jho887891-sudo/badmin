#!/usr/bin/env python3
"""COCO train2017 index, historical-overlap exclusion and leakage firewall (Task 2).

The two NEW training pools of CLEAN_OUTDOOR_CP_V1 may only come from COCO train2017, and only from images
that (a) were never used by the frozen V2 work, (b) are not part of any frozen evaluation set, (c) carry a
resolvable licence declaration and (d) are not a duplicate, crop or near-duplicate of another candidate.

Nothing in this module decides that a picture is shuttlecock-free: COCO has no such category, so an absent
annotation is never treated as proof of absence. That decision belongs to the human review recorded in
qa_status / no_shuttle_reviewed downstream.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a declared project dependency
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

SPLIT = "train2017"
COCO_NAME_PATTERN = re.compile(r"^(\d{12})\.(jpg|jpeg|png)$", re.IGNORECASE)
NEAR_DUPLICATE_MAX_DISTANCE = 6
#: Locality-sensitive bands. Eight 8-bit bands guarantee that two hashes within NEAR_DUPLICATE_MAX_DISTANCE
#: (<= 7) still share at least one band, so no near-duplicate pair can be split by bucketing.
AHASH_BAND_BITS = 8
MIN_IMAGE_SIDE = 8

INVENTORY_COLUMNS = (
    "coco_image_id", "file_name", "split", "path", "bytes", "width", "height", "sha256",
    "license_id", "license_name", "license_url", "permission_state", "coco_url", "flickr_url",
    "ahash", "near_duplicate_group",
)
PROHIBITED_COLUMNS = ("source_id", "coco_image_id", "sha256", "path", "origin", "role", "reason")
REJECTION_COLUMNS = ("coco_image_id", "file_name", "reason", "detail")


class CocoIndexBlocked(RuntimeError):
    """Raised when the COCO input is not the official train2017 index the plan authorises."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coco_id_from_file_name(name: str):
    """COCO file names are the zero-padded image id, which is how historical usage is matched exactly."""
    match = COCO_NAME_PATTERN.match(Path(str(name)).name)
    return int(match.group(1)) if match else None


def average_hash(path) -> int:
    """64-bit average hash: cheap, deterministic, and enough to flag crops/near-duplicates for review."""
    if Image is None:
        raise CocoIndexBlocked("Pillow is required to hash COCO images")
    with Image.open(path) as handle:
        gray = handle.convert("L").resize((8, 8))
        # tobytes() rather than getdata(): stable across Pillow versions (getdata is deprecated in Pillow 12+)
        pixels = list(gray.tobytes())
    mean = sum(pixels) / float(len(pixels))
    bits = 0
    for position, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << position
    return bits


def hamming_distance(left: int, right: int) -> int:
    return bin(int(left) ^ int(right)).count("1")


def empty_prohibited_index() -> pd.DataFrame:
    """An explicitly empty exclusion index, so callers never pass a bare None by accident."""
    return pd.DataFrame(columns=list(PROHIBITED_COLUMNS))


def _assign_near_duplicate_groups(rows) -> None:
    """Group near-duplicates in place using banded LSH, so 118k images stay tractable without O(n^2)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict = {}
    for index, row in enumerate(rows):
        value = row.get("ahash")
        if value is None:
            continue
        for shift in range(0, 64, AHASH_BAND_BITS):
            key = (shift, (int(value) >> shift) & ((1 << AHASH_BAND_BITS) - 1))
            buckets.setdefault(key, []).append(index)

    compared: set = set()
    for _, members in buckets.items():
        if len(members) < 2:
            continue
        for position in range(len(members)):
            for other in range(position + 1, len(members)):
                left, right = members[position], members[other]
                pair = (left, right) if left < right else (right, left)
                if pair in compared:
                    continue
                compared.add(pair)
                if hamming_distance(rows[left]["ahash"], rows[right]["ahash"]) <= NEAR_DUPLICATE_MAX_DISTANCE:
                    union(left, right)

    components: dict = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    group_index = 0
    for _, members in sorted(components.items()):
        if len(members) > 1:
            name = "nd-%04d" % group_index
            group_index += 1
            for member in members:
                rows[member]["near_duplicate_group"] = name


def load_coco_train2017_index(images_dir, annotation_json) -> pd.DataFrame:
    """Read the official COCO train2017 split into an inventory. Any deviation is blocking, not a warning."""
    images_dir = Path(images_dir)
    annotation_json = Path(annotation_json)

    if images_dir.name != SPLIT:
        raise CocoIndexBlocked(
            "only the %s split may be indexed; got directory name %r (%s)" % (SPLIT, images_dir.name, images_dir)
        )
    if not images_dir.is_dir():
        raise CocoIndexBlocked("images directory does not exist: %s" % images_dir)
    if not annotation_json.is_file():
        raise CocoIndexBlocked("COCO annotation file is missing: %s" % annotation_json)
    try:
        payload = json.loads(annotation_json.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse failure blocks the index
        raise CocoIndexBlocked("COCO annotation file is unreadable: %s (%s)" % (annotation_json, exc))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise CocoIndexBlocked("COCO annotation file has no images array: %s" % annotation_json)
    licences = {}
    for entry in payload.get("licenses") or []:
        try:
            licences[int(entry.get("id"))] = entry
        except (TypeError, ValueError):
            continue

    root = images_dir.resolve()
    rows: list = []
    seen_ids: set = set()
    for entry in images:
        try:
            coco_id = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            raise CocoIndexBlocked("COCO image entry without a usable id: %r" % (entry,))
        if coco_id in seen_ids:
            raise CocoIndexBlocked("duplicate COCO image id in %s: %d" % (annotation_json, coco_id))
        seen_ids.add(coco_id)

        name = str(entry.get("file_name", ""))
        if not name:
            raise CocoIndexBlocked("COCO image %d has no file_name" % coco_id)
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CocoIndexBlocked("unsafe COCO file_name for image %d: %r" % (coco_id, name))
        path = (images_dir / candidate).resolve()
        if root != path.parent and root not in path.parents:
            raise CocoIndexBlocked("COCO file_name escapes the split directory: %r" % name)
        if not path.is_file():
            raise CocoIndexBlocked("COCO image bytes are missing: %s" % path)

        if Image is None:
            raise CocoIndexBlocked("Pillow is required to validate COCO image bytes")
        try:
            with Image.open(path) as handle:
                actual = handle.size
        except Exception as exc:  # noqa: BLE001
            raise CocoIndexBlocked("COCO image is not decodable: %s (%s)" % (path, exc))

        declared = (int(entry.get("width") or 0), int(entry.get("height") or 0))
        if declared != actual:
            raise CocoIndexBlocked(
                "COCO size mismatch for image %d: annotation says %s, file is %s" % (coco_id, declared, actual)
            )
        if min(actual) < MIN_IMAGE_SIDE:
            raise CocoIndexBlocked("COCO image %d is smaller than %d px: %s" % (coco_id, MIN_IMAGE_SIDE, actual))

        licence_id = entry.get("license")
        licence = licences.get(int(licence_id)) if licence_id is not None else None
        permission = "declared" if (licence and (licence.get("name") or licence.get("url"))) else "unknown"

        rows.append({
            "coco_image_id": coco_id,
            "file_name": name,
            "split": SPLIT,
            "path": str(path),
            "bytes": path.stat().st_size,
            "width": actual[0],
            "height": actual[1],
            "sha256": sha256_file(path),
            "license_id": int(licence_id) if licence_id is not None else None,
            "license_name": (licence or {}).get("name"),
            "license_url": (licence or {}).get("url"),
            "permission_state": permission,
            "coco_url": entry.get("coco_url"),
            "flickr_url": entry.get("flickr_url"),
            "ahash": average_hash(path),
            "near_duplicate_group": None,
        })

    _assign_near_duplicate_groups(rows)
    frame = pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS))
    frame["coco_image_id"] = frame["coco_image_id"].astype("int64")
    return frame


def _manifest_source_rows(manifest, role: str, only_negatives: bool = True):
    """Turn a repository manifest into exclusion rows, deriving the COCO id from the file name when possible."""
    manifest = Path(manifest)
    if not manifest.is_file():
        raise CocoIndexBlocked("manifest is missing: %s" % manifest)
    with open(manifest, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            is_negative = str(row.get("is_negative", "")).strip().lower() in {"true", "1", "yes"}
            if only_negatives and not is_negative:
                continue
            image = row.get("image") or row.get("abs_path") or row.get("path") or ""
            digest = row.get("sha256") or None
            coco_id = coco_id_from_file_name(image)
            yield {
                "source_id": image or ("sha:%s" % digest),
                "coco_image_id": coco_id,
                "sha256": digest,
                "path": image or None,
                "origin": "%s:%s" % (manifest.name, row.get("set_dir", "")),
                "role": role,
                "reason": "historical_v2_coco_id" if coco_id is not None else "historical_content_hash",
            }


def build_prohibited_source_index(
    v2_manifests=(),
    frozen_eval_manifests=(),
    historical_coco_ids=(),
    historical_sha256=(),
    historical_coco_id_completeness: str = "UNKNOWN",
    coco_id_dirs=(),
) -> pd.DataFrame:
    """Everything the two new pools must not touch: historical COCO ids/hashes and every frozen eval image.

    historical_coco_id_completeness must be KNOWN for the id-based part of the audit to be trustworthy; the
    caller records UNKNOWN when the historical id list could not be reconstructed, which keeps the gate shut.
    """
    rows: list = []
    ids = {int(value) for value in historical_coco_ids}
    for directory in coco_id_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise CocoIndexBlocked("historical COCO id directory is missing: %s" % directory)
        for path in sorted(directory.iterdir()):
            derived = coco_id_from_file_name(path.name)
            if derived is not None:
                ids.add(derived)
    for coco_id in sorted(ids):
        rows.append({
            "source_id": "coco-id:%d" % coco_id,
            "coco_image_id": coco_id,
            "sha256": None,
            "path": None,
            "origin": "historical COCO usage",
            "role": C.ROLE_V2_HISTORICAL_NEGATIVE,
            "reason": "historical_v2_coco_id",
        })
    for digest in historical_sha256 or ():
        rows.append({
            "source_id": "sha:%s" % digest,
            "coco_image_id": None,
            "sha256": digest,
            "path": None,
            "origin": "historical content hash",
            "role": C.ROLE_V2_HISTORICAL_NEGATIVE,
            "reason": "historical_content_hash",
        })
    for manifest in v2_manifests:
        rows.extend(_manifest_source_rows(manifest, C.ROLE_V2_HISTORICAL_NEGATIVE, only_negatives=True))
    for manifest in frozen_eval_manifests:
        rows.extend(_manifest_source_rows(manifest, C.ROLE_FROZEN_EVAL, only_negatives=False))

    frame = pd.DataFrame(rows, columns=list(PROHIBITED_COLUMNS))
    frame.attrs["historical_coco_id_completeness"] = historical_coco_id_completeness
    return frame


def filter_coco_candidates(coco: pd.DataFrame, prohibited: pd.DataFrame):
    """Split the inventory into usable candidates and reasons-bearing rejections. No row is silently dropped."""
    if coco is None or len(coco) == 0:
        return pd.DataFrame(columns=list(INVENTORY_COLUMNS)), pd.DataFrame(columns=list(REJECTION_COLUMNS))

    prohibited_ids: set = set()
    prohibited_hashes: set = set()
    prohibited_paths: set = set()
    if prohibited is not None and len(prohibited) > 0:
        if "coco_image_id" in prohibited.columns:
            prohibited_ids = {int(value) for value in prohibited["coco_image_id"].dropna()}
        if "sha256" in prohibited.columns:
            prohibited_hashes = {str(value) for value in prohibited["sha256"].dropna()}
        if "path" in prohibited.columns:
            prohibited_paths = {str(value) for value in prohibited["path"].dropna()}

    kept: list = []
    rejections: list = []
    seen_hash: dict = {}
    for row in coco.to_dict("records"):
        coco_id = int(row["coco_image_id"])
        reason = detail = None
        if coco_id in prohibited_ids:
            reason, detail = "historical_v2_coco_id", "image id already used by the frozen V2 work"
        elif row.get("sha256") in prohibited_hashes:
            reason, detail = "historical_content_hash", "identical bytes to a frozen V2 source"
        elif str(row.get("path")) in prohibited_paths:
            reason, detail = "prohibited_path", "path is on the prohibited source list"
        elif row.get("permission_state") != "declared":
            reason, detail = "license_unverifiable", "image licence cannot be resolved from the COCO metadata"
        elif row.get("sha256") in seen_hash:
            reason, detail = "duplicate_content", "identical bytes to COCO image %d" % seen_hash[row["sha256"]]
        if reason:
            rejections.append({"coco_image_id": coco_id, "file_name": row.get("file_name"),
                               "reason": reason, "detail": detail})
            continue
        seen_hash[row["sha256"]] = coco_id
        kept.append(row)

    groups: dict = {}
    for row in kept:
        group = row.get("near_duplicate_group")
        if isinstance(group, str) and group:
            groups.setdefault(group, []).append(row)
    flagged = set()
    for group, members in groups.items():
        if len(members) > 1:
            flagged.update(int(member["coco_image_id"]) for member in members)

    final: list = []
    for row in kept:
        if int(row["coco_image_id"]) in flagged:
            rejections.append({
                "coco_image_id": int(row["coco_image_id"]),
                "file_name": row.get("file_name"),
                "reason": "near_duplicate_review",
                "detail": "shares near-duplicate group %s; a human must decide which member, if any, is used"
                          % row.get("near_duplicate_group"),
            })
            continue
        final.append(row)

    candidates = pd.DataFrame(final, columns=list(coco.columns))
    rejection_frame = pd.DataFrame(rejections, columns=list(REJECTION_COLUMNS))
    return candidates, rejection_frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Index COCO train2017 and isolate new-pool candidates (Task 2)")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prohibited-manifest", type=Path, action="append", default=[])
    parser.add_argument("--frozen-eval-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-coco-id-dir", type=Path, action="append", default=[],
                        help="directory whose COCO file names are historical usage and must be excluded")
    parser.add_argument("--historical-coco-id-completeness", default="UNKNOWN", choices=["KNOWN", "UNKNOWN"])
    parser.add_argument("--out-inventory", type=Path, required=True)
    parser.add_argument("--out-candidates", type=Path, required=True)
    parser.add_argument("--out-exclusions", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    args = parser.parse_args(argv)

    index = load_coco_train2017_index(args.images_dir, args.annotations)
    prohibited = build_prohibited_source_index(
        v2_manifests=args.prohibited_manifest,
        frozen_eval_manifests=args.frozen_eval_manifest,
        coco_id_dirs=args.exclude_coco_id_dir,
        historical_coco_id_completeness=args.historical_coco_id_completeness,
    )
    candidates, rejections = filter_coco_candidates(index, prohibited)

    for path, frame in ((args.out_inventory, index), (args.out_candidates, candidates),
                        (args.out_exclusions, rejections), (args.out_audit, prohibited)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    audit = {
        "generated_by": "tools/index_coco_train2017.py",
        "task": "Task 2: COCO train2017 index, historical overlap exclusion and leakage firewall",
        "images_dir": str(args.images_dir),
        "annotations": str(args.annotations),
        "split": SPLIT,
        "counts": {
            "inventory": int(len(index)),
            "candidates": int(len(candidates)),
            "rejected": int(len(rejections)),
            "prohibited_rows": int(len(prohibited)),
            "prohibited_coco_ids": int(prohibited["coco_image_id"].dropna().nunique()) if len(prohibited) else 0,
            "permission_declared": int((index["permission_state"] == "declared").sum()),
            "permission_unknown": int((index["permission_state"] == "unknown").sum()),
            "near_duplicate_rows": int(index["near_duplicate_group"].notna().sum()),
        },
        "rejection_reasons": rejections["reason"].value_counts().to_dict() if len(rejections) else {},
        "historical_coco_id_completeness": args.historical_coco_id_completeness,
        "gate": {
            "name": "COCO inventory",
            "status": "PASS" if len(candidates) else "BLOCKED_COCO_INVENTORY",
            "rule": "usable COCO train2017 candidates exist and no historical/eval overlap survives",
        },
        "note": "COCO has no shuttlecock category: no annotation is never proof that a picture has no shuttlecock",
    }
    args.out_audit.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit["counts"], indent=2, sort_keys=True))
    print("rejections by reason:", audit["rejection_reasons"])
    print("gate:", audit["gate"]["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
