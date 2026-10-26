"""Independent re-derivation of the ETH all-box inventory.

Deliberately does NOT import tools/eth_four_stage_inventory.py or prepare_eth_real_foregrounds.py:
it re-implements the selection from the raw CSVs so the tool's own numbers can be contradicted.
Exits non-zero if any cross-check fails.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

MANIFEST = Path("/home/dgut/eth_audit/repo/data/eth_real_hardneg_v2_train_manifest.csv")
VAL = Path("/home/dgut/eth_audit/repo/data/eth_real_hardneg_v2_val_manifest.csv")
FROZEN = Path("/home/dgut/eth_audit/repo/data/clean_outdoor_cp_v1/frozen_eval_index.csv")
INVENTORY = Path("/home/dgut/eth_audit/out/inventory_20261026T074352Z/instances.csv")

EVAL_TOKENS = {"val", "valid", "validation", "test", "tests", "eval", "evaluation", "unseen",
               "eth_unseen", "challenge", "challenge_test", "controlled", "controlled_capability",
               "no_target", "real_no_target_496", "ood", "holdout", "hold_out"}
NON_ETH_TOKENS = {"coco", "roboflow", "iphone", "d455", "isaac", "synthetic", "render", "rendered",
                  "blender", "3d", "unreal", "gazebo", "mujoco", "sim"}
TRUTHY = {"true", "1", "yes"}

failures: list[str] = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def tokens(*vals):
    out = set()
    for v in vals:
        if v is None:
            continue
        out.update(p for p in re.split(r"[^a-z0-9]+", str(v).strip().lower()) if p)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def loose(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        train = list(csv.DictReader(f))
    val = loose(VAL)
    frozen = loose(FROZEN)

    val_paths = {r["image"] for r in val}
    val_sha = {r["sha256"] for r in val}
    fr_paths = {r.get("source_id", "") for r in frozen}
    fr_sha = {r.get("sha256", "") for r in frozen if r.get("sha256")}

    # ---- independent re-selection: positives, split=train, ETH source, no eval tokens, not in firewalls
    expected = []
    rejected = Counter()
    for r in train:
        if str(r.get("is_negative", "")).strip().lower() in TRUTHY or \
                str(r.get("is_positive", "")).strip().lower() not in TRUTHY:
            rejected["not_positive"] += 1
            continue
        if str(r.get("split", "")).strip().lower() != "train":
            rejected["not_train_split"] += 1
            continue
        if tokens(r.get("set_dir"), r.get("location"), Path(r["image"]).stem) & EVAL_TOKENS:
            rejected["eval_token"] += 1
            continue
        if tokens(r.get("source"), r.get("set_dir")) & NON_ETH_TOKENS or \
                "eth" not in tokens(r.get("source")):
            rejected["non_eth_source"] += 1
            continue
        if r["image"] in val_paths or r["sha256"] in val_sha:
            rejected["in_val_manifest"] += 1
            continue
        if r["image"] in fr_paths or r["sha256"] in fr_sha:
            rejected["in_frozen_eval"] += 1
            continue
        expected.append(r)

    expected_boxes = sum(int(float(r["n_boxes"] or 0)) for r in expected)

    with INVENTORY.open("r", encoding="utf-8", newline="") as f:
        inv = list(csv.DictReader(f))

    check(len(inv) == expected_boxes,
          f"inventory rows {len(inv)} != independently re-derived boxes {expected_boxes}")
    ids = [r["instance_id"] for r in inv]
    check(len(set(ids)) == len(ids), "instance_id values are not unique")

    # ---- per-instance integrity: format, bbox sanity, image/label bytes, source isolation
    bad_id = 0
    bad_bbox = 0
    checked_images = 0
    mismatched_sha = 0
    leaked = 0
    seen_images = set()
    for row in inv:
        m = re.fullmatch(r"([0-9a-f]{64})#(\d+)", row["instance_id"])
        if not m:
            bad_id += 1
            continue
        digest, idx = m.group(1), int(m.group(2))
        img = Path(row["image"])
        lab = Path(row["label"])
        if not img.is_file() or not lab.is_file():
            failures.append(f"missing file for {row['instance_id']}")
            continue
        x1, y1, x2, y2 = json.loads(row["bbox_xyxy"])
        w, h = json.loads(row["bbox_wh"])
        if not (0 <= x1 < x2 <= 30000 and 0 <= y1 < y2 <= 30000 and w == x2 - x1 and h == y2 - y1):
            bad_bbox += 1
        if row["source_sha256"] != digest:
            mismatched_sha += 1
        if str(row["image"]) in val_paths or str(row["source_sha256"]) in val_sha:
            leaked += 1
        if str(row["image"]) in fr_paths or str(row["source_sha256"]) in fr_sha:
            leaked += 1
        if row["split"] != "train" or row["source"] != "eth_main":
            leaked += 1
        if row["status"] != "PENDING_REVIEW" or row["reason"] != "mask_not_reviewed":
            failures.append(f"unexpected review state {row['status']}/{row['reason']}")
        # verify real image bytes once per distinct file
        if row["image"] not in seen_images:
            seen_images.add(row["image"])
            actual = sha256(img)
            checked_images += 1
            if actual != digest:
                failures.append(f"image sha mismatch {img}: {actual} != {digest}")
                if checked_images > 5:
                    break

    check(bad_id == 0, f"{bad_id} instance ids are not sha256#index")
    check(bad_bbox == 0, f"{bad_bbox} bboxes are degenerate/out of range")
    check(mismatched_sha == 0, f"{mismatched_sha} rows have source_sha256 != instance id digest")
    check(leaked == 0, f"{leaked} rows violate split/source/frozen-eval isolation")

    # ---- coverage: every distinct positive train image must appear
    exp_imgs = {r["image"] for r in expected}
    inv_imgs = {r["image"] for r in inv}
    check(exp_imgs == inv_imgs, f"image coverage differs: only-expected={len(exp_imgs - inv_imgs)} "
                               f"only-inventory={len(inv_imgs - exp_imgs)}")

    summary = {
        "independent_re_derivation": {
            "manifest_rows": len(train),
            "positives_after_firewalls": len(expected),
            "expected_boxes": expected_boxes,
            "rejected": dict(rejected),
            "distinct_positive_images": len(exp_imgs),
            "locations": sorted({r["location"] for r in expected}),
            "set_dirs": sorted({r["set_dir"] for r in expected}),
            "box_short_min": min(int(float(r["n_boxes"] or 0)) for r in expected) if expected else None,
        },
        "inventory": {
            "rows": len(inv),
            "distinct_instance_ids": len(set(ids)),
            "images_byte_verified": checked_images,
            "image_sha_mismatches": len([f for f in failures if "sha mismatch" in f]),
        },
        "cross_check": {
            "row_count_matches": len(inv) == expected_boxes,
            "image_set_matches": exp_imgs == inv_imgs,
            "isolation_leaks": leaked,
            "bad_instance_ids": bad_id,
            "bad_bboxes": bad_bbox,
        },
        "failures": failures[:20],
        "verdict": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
