#!/usr/bin/env python3
"""14_render_hifi.py - build a calibrated, textured-looking shuttle dataset.

Thin CLI over tools/shuttle_render.py (unit-tested in tests/tools/test_shuttle_render.py).
All rendering, sizing, ground-truth and compositing logic lives in that module; this file
only walks the sample list, writes files, and refuses to run on a leaky background split.

Source type is SYNTHETIC_HIFI_3D: the shuttlecock geometry comes from the archived GLB
(Obj_Feather + Obj_Cork), appearance is modelled (that asset carries no texture for the
shuttle), and the ground-truth box is the rendered footprint - exact, not eyeballed.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_DEFAULT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO_DEFAULT))
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-px", type=float, default=2.0)
    p.add_argument("--max-px", type=float, default=32.0)
    p.add_argument("--median-px", type=float, default=9.0)
    p.add_argument("--supersample", type=int, default=3)
    p.add_argument("--bg-dir", default=None)
    p.add_argument("--frozen-bg-dir", default=None,
                   help="P3 frozen capability-set backgrounds; the run aborts on a match")
    p.add_argument("--other-bg-dir", default=None,
                   help="the other split background pool; render aborts if it overlaps")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo / "tools"))
    from shuttle_render import (find_blank_backgrounds, find_leaked_backgrounds,
                               imread_unicode, imwrite_unicode, load_shuttle_parts,
                               render_sample)

    out = repo / "outputs" / "shuttle_capability" / "train_data"
    data_dir = out / a.split
    img_dir = data_dir / "images"
    lab_dir = data_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    other = "val" if a.split == "train" else "train"
    bg_dir = Path(a.bg_dir) if a.bg_dir else (out / ("bg_" + a.split))
    other_bg_dir = Path(a.other_bg_dir) if a.other_bg_dir else (out / ("bg_" + other))
    bgs = sorted(p for p in bg_dir.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not bgs:
        print("no backgrounds in", bg_dir)
        return 2
    if other_bg_dir.is_dir():
        mine = {p.name for p in bgs}
        theirs = {p.name for p in other_bg_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        overlap = mine & theirs
        if overlap:
            print("LEAK: " + a.split + " shares backgrounds with " + other + ": "
                  + ", ".join(sorted(overlap)))
            return 3
        print("background isolation ok: " + str(len(mine)) + " " + a.split
              + " / " + str(len(theirs)) + " " + other + " / 0 shared")

    blank = find_blank_backgrounds(bgs)
    if blank:
        print("BLANK background(s) in the pool (no scene to composite onto):")
        for path in blank:
            print("   " + path.name)
        return 6

    frozen_dir = (Path(a.frozen_bg_dir) if a.frozen_bg_dir
                  else repo / "outputs" / "shuttle_capability" / "real_images" / "backgrounds")
    if frozen_dir.is_dir():
        frozen = sorted(p for p in frozen_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        leaks = find_leaked_backgrounds(bgs, frozen)
        if leaks:
            print("LEAK: training backgrounds duplicated in the frozen capability set:")
            for candidate, match, corr in leaks:
                print("   " + candidate.name + "  ==  " + match.name
                      + "  (r=" + f"{corr:.4f}" + ")")
            return 5
        print("frozen-set isolation ok: " + str(len(bgs)) + " candidates vs "
              + str(len(frozen)) + " frozen backgrounds, 0 duplicated")
    else:
        print("WARNING: frozen background pool not found at " + str(frozen_dir)
              + " - leakage gate SKIPPED (this is not a pass)")

    parts = load_shuttle_parts()
    rng = np.random.default_rng(a.seed)
    raw = rng.lognormal(math.log(a.median_px), 0.5, a.n)
    targets = np.clip(raw, a.min_px, a.max_px)
    rows = []
    for i, target in enumerate(targets):
        bg_path = bgs[i % len(bgs)]
        bg = imread_unicode(bg_path)
        if bg is None:
            print("unreadable background", bg_path)
            continue
        name = a.split + "_" + f"{i:05d}"
        sample = render_sample(parts, bg, target_px=float(target), seed=a.seed * 100003 + i,
                               split=a.split, name=name, width=a.imgsz, height=a.imgsz,
                               supersample=a.supersample, background_name=bg_path.name)
        imwrite_unicode(img_dir / (name + ".jpg"), sample.image, quality=92)
        (lab_dir / (name + ".txt")).write_text(sample.label)
        rows.append(sample.record)

    if not rows:
        print("nothing rendered")
        return 4
    manifest = out / ("manifest_" + a.split + ".csv")
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    sizes = np.array([r["equiv_size_px"] for r in rows])
    print(a.split + ": " + str(len(rows)) + " images | measured eq px min="
          + f"{sizes.min():.2f}" + " median=" + f"{np.median(sizes):.2f}"
          + " max=" + f"{sizes.max():.2f}")
    for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 999)):
        n = int(((sizes >= lo) & (sizes < hi)).sum())
        print("  " + f"[{lo:>2d},{hi:>3d}) px : {n:4d}")
    print("wrote", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
