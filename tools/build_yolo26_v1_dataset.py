#!/usr/bin/env python3
"""Gate 4 of the YOLO26 V1 spec: build the unified manifest + frozen location-disjoint split.

Data roles (spec sections 5-8):
  eth_main      11 recording locations, 20508 real boxes      -> primary real positives
  eth_iphone    old/iphone_20251015, 1068 real boxes          -> larger-target supplement
  synthetic     outputs/shuttle_capability/train_data         -> scale compensation
  coco_bg       ETH coco_train_easy / coco_val_easy           -> ordinary real negatives
  hard_negative hard_negatives{,2}                            -> upweighted negatives

Frozen sets (NEVER train/val): controlled_capability, challenge_test, synthetic_on_real_bg,
synthetic_3d, real_images, real_video, real_match_frames, real_train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 10 ** 9)]
BUCKET_LABELS = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FROZEN_DIRS = ["controlled_capability", "challenge_test", "synthetic_on_real_bg", "synthetic_3d",
               "real_images", "real_video", "real_match_frames", "real_train"]


def bucket(v):
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        if lo <= v < hi:
            return lab
    return BUCKET_LABELS[-1]


def read_size(path):
    """Read width/height from the file header without decoding pixels."""
    import struct
    with open(path, "rb") as f:
        head = f.read(32)
    if head[:2] == b"\xff\xd8":
        with open(path, "rb") as f:
            f.read(2)
            while True:
                b = f.read(1)
                if not b:
                    raise ValueError("no SOF")
                if b != b"\xff":
                    continue
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                m = marker[0]
                if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                    continue
                ln = struct.unpack(">H", f.read(2))[0]
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    data = f.read(5)
                    h, w = struct.unpack(">HH", data[1:5])
                    return w, h
                f.seek(ln - 2, 1)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return w, h
    raise ValueError("unsupported image header")


def parse_label(label_path):
    """Return (n_lines, n_boxes, [(cx, cy, w, h)]) for a YOLO label file."""
    if not label_path.exists():
        return 0, 0, []
    rows = []
    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append(tuple(float(x) for x in parts[1:5]))
            except ValueError:
                continue
    return len(rows), len(rows), rows


def label_for(image_path: Path) -> Path:
    s = str(image_path)
    s = s.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    p = Path(s)
    return p.with_suffix(".txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--eth-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-out", required=True)
    ap.add_argument("--remote-eth-root", default=None)
    ap.add_argument("--remote-repo", default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    eth = Path(args.eth_root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- frozen location split: the 3 locations with the fewest frames go to val ----
    def is_location_dir(name: str) -> bool:
        if name.startswith("coco") or name == "old":
            return False
        parts = name.rsplit("_", 1)
        return len(parts) == 2 and parts[1] in ("easy", "medium", "hard")

    loc_counts = {}
    for locdiff in sorted(os.listdir(eth)):
        d = eth / locdiff
        idir = d / "images" / "train"
        if not is_location_dir(locdiff) or not d.is_dir() or not idir.is_dir():
            continue
        loc = locdiff.rsplit("_", 1)[0]
        loc_counts[loc] = loc_counts.get(loc, 0) + len([f for f in os.listdir(idir) if f.lower().endswith(IMG_EXT)])
    ordered = sorted(loc_counts.items(), key=lambda kv: (kv[1], kv[0]))
    val_locations = sorted([k for k, _ in ordered[:3]])
    split_doc = {
        "version": "eth_location_split_v1",
        "rule": "val = the 3 recording locations with the fewest frames (ties broken by name); frozen",
        "train_locations": sorted([k for k, _ in ordered[3:]]),
        "val_locations": val_locations,
        "location_frame_counts": dict(ordered),
    }
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    import yaml
    Path(args.split_out).write_text(yaml.safe_dump(split_doc, allow_unicode=True, sort_keys=False), encoding="utf-8")

    rows = []
    problems = []

    def add(image_path: Path, source: str, split: str, location: str, difficulty: str):
        try:
            w, h = read_size(image_path)
        except Exception as e:
            problems.append({"path": str(image_path), "error": repr(e)})
            return
        lp = label_for(image_path)
        n_lines, n_boxes, boxes = parse_label(lp)
        scale = 640.0 / max(w, h)
        eqs = []
        for (cx, cy, bw, bh) in boxes:
            import math
            eqs.append(math.sqrt(bw * w * bh * h) * scale)
        rec = {
            "image": str(image_path),
            "label": str(lp) if lp.exists() else "",
            "source": source,
            "split": split,
            "location": location,
            "difficulty": difficulty,
            "width": w,
            "height": h,
            "n_boxes": n_boxes,
            "eq640_min": round(min(eqs), 3) if eqs else "",
            "eq640_max": round(max(eqs), 3) if eqs else "",
            "bucket": bucket(max(eqs)) if eqs else "negative",
        }
        rows.append(rec)

    # ETH main + iphone
    for locdiff in sorted(os.listdir(eth)):
        d = eth / locdiff
        idir = d / "images" / "train"
        if not is_location_dir(locdiff) or not d.is_dir() or not idir.is_dir():
            continue
        loc = locdiff.rsplit("_", 1)[0]
        diff = locdiff.rsplit("_", 1)[1]
        split = "val" if loc in val_locations else "train"
        for f in sorted(os.listdir(idir)):
            if f.lower().endswith(IMG_EXT):
                add(idir / f, "eth_main", split, loc, diff)
    iphone = eth / "old" / "iphone_20251015"
    if iphone.is_dir():
        for sub in ("train", "val"):
            idir = iphone / "images" / sub
            if not idir.is_dir():
                continue
            for f in sorted(os.listdir(idir)):
                if f.lower().endswith(IMG_EXT):
                    add(idir / f, "eth_iphone", sub, "iphone_20251015", "n/a")
    for name, split in (("coco_train_easy", "train"), ("coco_val_easy", "val")):
        idir = eth / name / "images" / "train"
        if not idir.is_dir():
            idir = eth / name / "images" / "val"
        if idir.is_dir():
            for f in sorted(os.listdir(idir)):
                if f.lower().endswith(IMG_EXT):
                    add(idir / f, "coco_bg", split, name, "n/a")

    # synthetic + backgrounds + hard negatives (repo-local)
    cap = repo / "outputs" / "shuttle_capability"
    syn = cap / "train_data"
    for sub, split in (("train", "train"), ("val", "val")):
        idir = syn / sub / "images"
        if idir.is_dir():
            for f in sorted(os.listdir(idir)):
                if f.lower().endswith(IMG_EXT):
                    add(idir / f, "synthetic", split, "synthetic", sub)
    for sub, split in (("bg_train", "train"), ("bg_val", "val")):
        idir = syn / sub
        if idir.is_dir():
            for f in sorted(os.listdir(idir)):
                if f.lower().endswith(IMG_EXT):
                    add(idir / f, "bg_negative", split, "backgrounds", sub)
    for hn in ("hard_negatives", "hard_negatives2"):
        idir = cap / hn
        if idir.is_dir():
            for f in sorted(idir.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    add(f, "hard_negative", "train", hn, "n/a")

    # ---- frozen guard: refuse to ship a manifest that contains frozen data ----
    for r in rows:
        p = r["image"].replace(os.sep, "/")
        for frozen in FROZEN_DIRS:
            if ("/shuttle_capability/" + frozen + "/") in p:
                raise SystemExit("FROZEN DATA LEAKED INTO MANIFEST: " + p)

    fields = ["image", "label", "source", "split", "location", "difficulty", "width", "height",
              "n_boxes", "eq640_min", "eq640_max", "bucket"]
    man = out / "v1_dataset_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for r in sorted(rows, key=lambda x: (x["split"], x["source"], x["image"])):
            wtr.writerow(r)

    # ---- ultralytics path lists (absolute paths, remapped for the training host) ----
    def remap(p: str) -> str:
        if args.remote_eth_root and p.startswith(str(eth)):
            return args.remote_eth_root + p[len(str(eth)):].replace(os.sep, "/")
        if args.remote_repo and p.startswith(str(repo)):
            return args.remote_repo + p[len(str(repo)):].replace(os.sep, "/")
        return p.replace(os.sep, "/")

    for split in ("train", "val"):
        sel = [r for r in rows if r["split"] == split]
        (out / (split + ".txt")).write_text(chr(10).join(remap(r["image"]) for r in sel) + chr(10), encoding="utf-8")

    # ---- scale-aware sampler + positive/negative ratio, expressed as manifest repeat weights ----
    tr = [r for r in rows if r["split"] == "train"]
    pos = [r for r in tr if r["n_boxes"] > 0]
    neg = [r for r in tr if r["n_boxes"] == 0]
    bucket_counts = Counter(r["bucket"] for r in pos)
    n_max = max(bucket_counts.values()) if bucket_counts else 1
    weights = {}
    for b, n in bucket_counts.items():
        weights[b] = min(3.0, (float(n_max) / float(n)) ** 0.5) if n else 1.0
    target_neg_ratio = 0.20
    n_pos_slots = sum(max(1, int(round(weights[r["bucket"]]))) for r in pos)
    n_neg_slots = int(round(n_pos_slots * target_neg_ratio / (1.0 - target_neg_ratio)))
    hard = [r for r in neg if r["source"] == "hard_negative"]
    plain = [r for r in neg if r["source"] != "hard_negative"]
    hard_weight = 3.0
    hard_slots = min(len(hard) * int(hard_weight), n_neg_slots // 2)
    plain_slots = max(0, n_neg_slots - hard_slots)
    weighted = []
    for r in pos:
        weighted.extend([r["image"]] * max(1, int(round(weights[r["bucket"]]))))
    for i in range(hard_slots):
        weighted.append(hard[i % len(hard)]["image"])
    for i in range(plain_slots):
        weighted.append(plain[i % len(plain)]["image"])
    (out / "train_weighted.txt").write_text(chr(10).join(remap(p) for p in weighted) + chr(10), encoding="utf-8")
    sampler = {
        "bucket_counts_train": dict(bucket_counts),
        "bucket_weight": {k: round(v, 4) for k, v in weights.items()},
        "n_pos_images_train": len(pos),
        "n_neg_images_train": len(neg),
        "pos_slots": n_pos_slots,
        "neg_slots": n_neg_slots,
        "negative_ratio_target": target_neg_ratio,
        "hard_negative_images": len(hard),
        "hard_negative_weight": hard_weight,
        "epoch_length": len(weighted),
    }
    (out / "v1_sampler.json").write_text(json.dumps(sampler, indent=1, ensure_ascii=False), encoding="utf-8")
    print("sampler:", json.dumps(sampler, ensure_ascii=False))

    # ---- digest ----
    digest = {
        "manifest": str(man),
        "rows": len(rows),
        "problems": problems[:50],
        "frozen_dirs_excluded": FROZEN_DIRS,
        "val_locations": val_locations,
        "train_locations": split_doc["train_locations"],
        "by_source_split": {},
        "by_bucket": {},
        "by_source": {},
        "positives": sum(1 for r in rows if r["n_boxes"] > 0),
        "negatives": sum(1 for r in rows if r["n_boxes"] == 0),
    }
    for r in rows:
        digest["by_source_split"][r["source"] + "_" + r["split"]] = digest["by_source_split"].get(r["source"] + "_" + r["split"], 0) + 1
        digest["by_source"][r["source"]] = digest["by_source"].get(r["source"], 0) + 1
        key = r["bucket"] + "_" + r["split"]
        digest["by_bucket"][key] = digest["by_bucket"].get(key, 0) + 1
    h = hashlib.sha256()
    with open(man, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest["manifest_sha256"] = h.hexdigest()
    (out / "v1_dataset_digest.json").write_text(json.dumps(digest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: digest[k] for k in ("rows", "positives", "negatives", "val_locations",
                                             "by_source", "by_source_split", "by_bucket",
                                             "manifest_sha256")}, indent=1, ensure_ascii=False))
    print("problems:", len(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())