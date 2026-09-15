#!/usr/bin/env python3
"""Rebuild the hard-negative pool as WHOLE SCENES rather than centre crops.

Measured reason: 4 of the 33 challenge C1n scenes overlap the training negatives, yet the model still
places about one box on each of them. The training negatives are 960x960 CENTRE CROPS, so the model
learned the centre of those scenes and not their edges. Whole scenes give the model the whole frame at
training time, since Ultralytics letterboxes each image to the training imgsz regardless of its size.

Both batches are combined and de-duplicated against the frozen core, the training backgrounds and each
other, so this pool is a clean replacement rather than an addition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import cv2

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/home/T7/dgut/robot_sim")
PREFIX = "whole_neg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "tools"))
    from shuttle_render import imread_unicode, imwrite_unicode

    base_dir = repo / "outputs" / "shuttle_capability" / "train_data"
    core = repo / "outputs" / "shuttle_capability" / "real_images" / "backgrounds"
    batch1 = repo / "outputs" / "shuttle_capability" / "hard_negatives" / "raw"
    batch2 = repo / "outputs" / "shuttle_capability" / "hard_negatives2" / "raw"

    def sig(path, n=48):
        im = imread_unicode(path)
        if im is None:
            return None
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
        g -= g.mean()
        s = g.std()
        return g / s if s > 1e-9 else None

    refs = {}
    for p in sorted(core.glob("*.jpg")) + sorted(core.glob("*.png")):
        s = sig(p)
        if s is not None:
            refs["core:" + p.name] = s
    for p in sorted((base_dir / "bg_train").glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            s = sig(p)
            if s is not None:
                refs["bg:" + p.name] = s
    print("leak reference set:", len(refs))

    rows = list(csv.DictReader((base_dir / args.base_manifest).open(encoding="utf-8")))
    columns = list(rows[0].keys())
    # drop the previous hard negatives entirely: this is a replacement, not an addition
    kept = [r for r in rows if not str(r.get("file", "")).startswith(("hardneg_train_", "hardneg2_train_"))]
    removed = len(rows) - len(kept)
    print("dropped {} crop-style hard negatives".format(removed))

    seen = set()
    added = 0
    excluded = 0
    for src_dir, tag in ((batch1, "b1"), (batch2, "b2")):
        for p in sorted(src_dir.glob("*.jpg")):
            s = sig(p)
            if s is None:
                continue
            worst = max(((float((s * r).mean()), n) for n, r in refs.items()), key=lambda t: t[0])
            if worst[0] >= 0.90:
                excluded += 1
                continue
            key = hashlib.sha256(p.read_bytes()).hexdigest()
            if key in seen:
                excluded += 1
                continue
            seen.add(key)
            img = imread_unicode(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            name = "{}_{}_{:05d}.jpg".format(PREFIX, tag, added)
            # WHOLE scene, resized to fit 960 on the long side with the aspect preserved; Ultralytics
            # letterboxes to the training size afterwards, so no information is cropped away here.
            scale = 960.0 / max(h, w)
            nw, nh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
            imwrite_unicode(base_dir / "train" / "images" / name,
                            cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), quality=92)
            (base_dir / "train" / "labels" / (name.rsplit(".", 1)[0] + ".txt")).write_text("", encoding="utf-8")
            row = {c: "" for c in columns}
            row.update({"file": name, "split": "train", "source_type": "NEGATIVE",
                        "background": p.name, "imgsz": "960", "is_negative": "True"})
            kept.append(row)
            added += 1

    out = base_dir / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in kept:
            w.writerow({c: r.get(c, "") for c in columns})
    pos = sum(1 for r in kept if r.get("is_negative") == "False")
    neg = sum(1 for r in kept if r.get("is_negative") == "True")
    print("added {} WHOLE-SCENE negatives, excluded {}".format(added, excluded))
    print("  manifest now {} rows: {} positives, {} negatives ({:.1%})".format(
        len(kept), pos, neg, neg / len(kept)))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())