#!/usr/bin/env python3
"""Build the YOLO26S_V2_FULL_ETH_CONTINUATION_V1 training pool (spec sections 5, 6, 7).

Positives : every <location>_easy / <location>_medium official ETH real image with a non-empty label.
Negatives : the official ETH background pool coco_train_easy, every one checked to be label-empty.
Excluded  : *_hard (ineligible difficulty), coco_val_easy (not the official train pool), old/ (project iPhone
            extras), anything else outside the official ETH train pool - all enumerated in the audit.

Every image is read ONCE: the bytes are hashed and the size is parsed from the same buffer. Work is spread over a
thread pool because the ETH pool lives on a fuseblk (NTFS) mount whose per-request latency, not bandwidth, is the
bottleneck. Rows are appended in the deterministic plan order, so re-running resumes where the previous pass stopped.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SET_RE = re.compile(r"^(?P<loc>[a-z0-9]+(?:_[0-9]+)?)_(?P<diff>easy|medium|hard)$")
NEGATIVE_SETS = ("coco_train_easy",)
PROHIBITED_TOKENS = ("synthetic", "isaac", "iphone", "near_field", "d455", "roboflow",
                     "controlled_capability", "challenge_test", "old/")
BUCKETS = [(0.0, 4.0, "<4"), (4.0, 6.0, "4-6"), (6.0, 8.0, "6-8"), (8.0, 12.0, "8-12"),
           (12.0, 16.0, "12-16"), (16.0, 24.0, "16-24"), (24.0, 32.0, "24-32"),
           (32.0, 64.0, "32-64"), (64.0, float("inf"), ">64")]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
FIELDS = ["path", "abs_path", "source", "set_dir", "location", "difficulty", "label_path", "abs_label_path",
          "has_label", "bbox_count", "equiv_size_640", "eq_min", "eq_median", "eq_max", "size_bucket",
          "sha256", "bytes", "width", "height", "is_positive", "is_negative"]


def bucket_of(eq):
    for lo, hi, lab in BUCKETS:
        if lo <= eq < hi:
            return lab
    return BUCKETS[-1][2]


def equiv_size_640(w_px, h_px, img_w, img_h):
    return math.sqrt(max(w_px, 0.0) * max(h_px, 0.0)) * (640.0 / max(img_w, img_h))


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label(path):
    boxes, errors = [], []
    with open(str(path), encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh.read().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                errors.append("line %d: %d fields" % (ln, len(parts)))
                continue
            try:
                boxes.append((int(float(parts[0])), *(float(x) for x in parts[1:5])))
            except ValueError:
                errors.append("line %d: unparsable" % ln)
    return boxes, errors


def discover(root: Path):
    positives, negatives, excluded = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name = d.name
        img_dir = d / "images" / "train"
        if name in NEGATIVE_SETS:
            negatives.append(d)
            continue
        m = SET_RE.match(name)
        if m and m.group("diff") in ("easy", "medium") and img_dir.is_dir():
            positives.append(d)
            continue
        if name == "old":
            reason = "project iPhone extras (spec section 6 prohibits ETH iPhone extras)"
        elif name == "coco_val_easy":
            reason = "official background val split, not part of the official train pool"
        elif m and m.group("diff") == "hard":
            reason = "difficulty not eligible (spec section 5.1: easy + medium only)"
        elif not img_dir.is_dir():
            reason = "no images/train directory"
        else:
            reason = "not part of the official ETH train pool"
        excluded.append({"set_dir": name, "reason": reason,
                         "images": len(list(img_dir.iterdir())) if img_dir.is_dir() else 0})
    return positives, negatives, excluded


def process_one(item, root):
    """Read one image exactly once and return (row, problems)."""
    idx, kind, set_dir, loc, diff, img = item
    problems = {}
    lab = root / set_dir / "labels" / "train" / (img.stem + ".txt")
    row = {"path": str(img.relative_to(root)), "abs_path": str(img),
           "source": "eth_main" if kind == "positive" else "eth_negative", "set_dir": set_dir,
           "location": loc, "difficulty": diff,
           "label_path": str(lab.relative_to(root)), "abs_label_path": str(lab),
           "has_label": lab.is_file(), "is_positive": kind == "positive", "is_negative": kind == "negative"}
    try:
        data = img.read_bytes()
    except Exception as exc:
        problems["unreadable"] = {"path": row["path"], "error": "%s: %s" % (type(exc).__name__, exc)}
        return None, problems
    sha = hashlib.sha256(data).hexdigest()
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            W, H = im.size
    except Exception as exc:
        problems["unreadable"] = {"path": row["path"], "error": "%s: %s" % (type(exc).__name__, exc)}
        return None, problems
    boxes, errors = ([], [])
    if lab.is_file():
        boxes, errors = read_label(lab)
    elif kind == "positive":
        problems["missing_label"] = row["path"]
        return None, problems
    if errors:
        problems["parse_errors"] = {"path": row["path"], "errors": errors}
    if kind == "negative" and boxes:
        problems["non_empty_negative"] = {"path": row["path"], "boxes": len(boxes)}
    eqs = [equiv_size_640(b[3] * W, b[4] * H, W, H) for b in boxes]
    row["bbox_count"] = len(boxes)
    row["equiv_size_640"] = round(max(eqs), 6) if eqs else ""
    row["eq_min"] = round(min(eqs), 6) if eqs else ""
    row["eq_median"] = round(sorted(eqs)[len(eqs) // 2], 6) if eqs else ""
    row["eq_max"] = row["equiv_size_640"]
    row["size_bucket"] = bucket_of(max(eqs)) if eqs else ""
    row["sha256"] = sha
    row["bytes"] = len(data)
    row["width"] = W
    row["height"] = H
    return row, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-audit", required=True)
    ap.add_argument("--lists-dir", default=None)
    ap.add_argument("--val-stride", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_manifest)
    audit_path = Path(args.out_audit)
    positives, negatives, excluded = discover(root)
    print("positive sets : %d  negative sets: %d" % (len(positives), len(negatives)), flush=True)
    print("excluded sets : %s" % json.dumps([[e["set_dir"], e["images"], e["reason"][:40]] for e in excluded]),
          flush=True)

    plan = []
    for kind, dirs in (("positive", positives), ("negative", negatives)):
        for d in dirs:
            m = SET_RE.match(d.name)
            loc, diff = (m.group("loc"), m.group("diff")) if m else (d.name, "n/a")
            img_dir = d / "images" / "train"
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
                plan.append((kind, d.name, loc, diff, img))
    print("planned rows  : %d" % len(plan), flush=True)

    done = 0
    if out.is_file():
        with out.open("r", encoding="utf-8") as fh:
            done = max(0, sum(1 for _ in fh) - 1)
        print("resume: %d rows already present" % done, flush=True)
        if done > len(plan):
            print("STOP: existing manifest is longer than the plan")
            return 3
    fh = out.open("a" if done else "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if not done:
        w.writeheader()

    audit = {"root": str(root), "plan_rows": len(plan), "excluded_sets": excluded,
             "positive_set_dirs": [d.name for d in positives], "negative_set_dirs": [d.name for d in negatives],
             "missing_images": [], "unreadable_images": [], "missing_labels": [], "label_parse_errors": [],
             "non_empty_negative_labels": [], "duplicate_sha256": [], "resume_from": done,
             "workers": args.workers}
    sha_seen = defaultdict(list)
    by_set, boxes_by_set = Counter(), Counter()
    bucket_boxes, bucket_images = Counter(), Counter()
    work = [(i, kind, sd, loc, diff, img)
            for i, (kind, sd, loc, diff, img) in enumerate(plan, 1) if i > done]
    if args.limit:
        work = work[:args.limit]
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row, problems in ex.map(lambda it: process_one(it, root), work, chunksize=8):
            n += 1
            if "unreadable" in problems:
                audit["unreadable_images"].append(problems["unreadable"])
            if "missing_label" in problems:
                audit["missing_labels"].append(problems["missing_label"])
            if "parse_errors" in problems:
                audit["label_parse_errors"].append(problems["parse_errors"])
            if "non_empty_negative" in problems:
                audit["non_empty_negative_labels"].append(problems["non_empty_negative"])
            if row is None:
                continue
            w.writerow(row)
            sha_seen[row["sha256"]].append(row["path"])
            by_set[row["set_dir"]] += 1
            boxes_by_set[row["set_dir"]] += int(row["bbox_count"])
            for _ in range(int(row["bbox_count"])):
                pass
            if row["size_bucket"]:
                bucket_images[row["size_bucket"]] += 1
            if n % 500 == 0:
                fh.flush()
                rate = n / max(1e-9, time.time() - t0)
                print("  %d/%d (%.1f rows/s, eta %.0f min)" % (n, len(work), rate,
                                                               (len(work) - n) / max(rate, 1e-9) / 60), flush=True)
    fh.close()

    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    for r in rows:
        if r["equiv_size_640"]:
            bucket_boxes[bucket_of(float(r["equiv_size_640"]))] += 1
    audit.update({"manifest": str(out), "manifest_rows": len(rows), "manifest_sha256": sha256_file(out),
                  "images_total": len(plan), "positive_images": sum(1 for r in rows if r["is_positive"] == "True"),
                  "negative_images": sum(1 for r in rows if r["is_negative"] == "True"),
                  "boxes_total": sum(int(r["bbox_count"]) for r in rows),
                  "images_by_set": dict(sorted(by_set.items())), "boxes_by_set": dict(sorted(boxes_by_set.items())),
                  "location_counts": dict(sorted(Counter(r["location"] for r in rows).items())),
                  "difficulty_counts": dict(sorted(Counter(r["difficulty"] for r in rows).items())),
                  "size_bucket_counts_boxes_max_eq_per_image": {lab: bucket_boxes.get(lab, 0) for _, _, lab in BUCKETS},
                  "size_bucket_counts_images": {lab: bucket_images.get(lab, 0) for _, _, lab in BUCKETS},
                  "duplicate_sha256_groups": sum(1 for v in sha_seen.values() if len(v) > 1),
                  "duplicate_sha256_extra_rows": sum(len(v) - 1 for v in sha_seen.values() if len(v) > 1),
                  "duplicate_sha256_examples": [v for v in sha_seen.values() if len(v) > 1][:5],
                  "prohibited_source_hits": [r["path"] for r in rows
                                             if any(t in r["path"].lower() for t in PROHIBITED_TOKENS)],
                  "elapsed_s": round(time.time() - t0, 1)})
    for k, v in (("GATE_missing_images_zero", "missing_images"), ("GATE_unreadable_images_zero", "unreadable_images"),
                 ("GATE_non_empty_negative_labels_zero", "non_empty_negative_labels"),
                 ("GATE_prohibited_hits_zero", "prohibited_source_hits")):
        audit[k] = not audit[v]
    audit["GATE_all_pass"] = all(audit[k] for k in ("GATE_missing_images_zero", "GATE_unreadable_images_zero",
                                                    "GATE_non_empty_negative_labels_zero", "GATE_prohibited_hits_zero"))
    audit["GATE_manifest_complete"] = audit["manifest_rows"] == len(plan)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    if args.lists_dir:
        ld = Path(args.lists_dir)
        ld.mkdir(parents=True, exist_ok=True)
        ordered = sorted(rows, key=lambda r: r["sha256"])
        val = [r for i, r in enumerate(ordered) if i % args.val_stride == 0]
        vp = {r["path"] for r in val}
        train = [r for r in ordered if r["path"] not in vp]
        (ld / "train.txt").write_text("".join(r["abs_path"] + chr(10) for r in train), encoding="utf-8")
        (ld / "val.txt").write_text("".join(r["abs_path"] + chr(10) for r in val), encoding="utf-8")
        (ld / "dataset.yaml").write_text("path: %s%strain: train.txt%sval: val.txt%snc: 1%snames:%s  0: shuttlecock%s"
                                         % (ld, chr(10), chr(10), chr(10), chr(10), chr(10), chr(10)), encoding="utf-8")
        audit["lists"] = {"dir": str(ld), "train_rows": len(train), "val_rows": len(val),
                          "val_stride": args.val_stride, "internal_val_overlaps_train_pool": True,
                          "internal_val_used_for_selection": False}
    audit_path.write_text(json.dumps(audit, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("manifest_rows", "positive_images", "negative_images", "boxes_total",
                                            "images_by_set", "difficulty_counts", "location_counts",
                                            "size_bucket_counts_images", "duplicate_sha256_groups",
                                            "duplicate_sha256_extra_rows", "missing_labels", "unreadable_images",
                                            "non_empty_negative_labels", "prohibited_source_hits", "GATE_all_pass",
                                            "GATE_manifest_complete", "elapsed_s")}, indent=1))
    print("WROTE %s (%d B) sha256=%s" % (out, out.stat().st_size, audit["manifest_sha256"]))
    print("WROTE %s (%d B)" % (audit_path, audit_path.stat().st_size))
    return 0 if (audit["GATE_all_pass"] and audit["GATE_manifest_complete"]) else 4


if __name__ == "__main__":
    raise SystemExit(main())
