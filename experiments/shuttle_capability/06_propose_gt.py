#!/usr/bin/env python3
"""06_propose_gt.py - propose GT boxes for real images (threshold + component), for VISUAL review.

Nothing here is treated as GT until a human/agent looks at the drawn box: the script only
produces review sheets.  Accepted boxes are written to annotations/real_images/ by 07_accept_gt.py.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import cv2
import numpy as np


def propose(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]; s = hsv[:, :, 1]
    mask = ((v > 170) & (s < 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 40 or bw < 4 or bh < 4:
            continue
        if bw > 0.9 * w or bh > 0.9 * h:
            continue
        cands.append((int(x), int(y), int(bw), int(bh), int(area)))
    cands.sort(key=lambda c: -c[4])
    return cands[:3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--per-sheet', type=int, default=4)
    args = ap.parse_args()
    repo = Path(args.repo) if args.repo else Path('/home/T7/ojh/robot_sim')
    src = repo / 'outputs/shuttle_capability/real_images/raw'
    viz = repo / 'outputs/shuttle_capability/visualizations/gt_review'
    viz.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    proposals = {}
    thumbs = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        cands = propose(img)
        proposals[f.name] = cands
        t = img.copy()
        for k, (x, y, bw, bh, a) in enumerate(cands):
            col = [(0, 0, 255), (0, 200, 255), (255, 0, 0)][k]
            cv2.rectangle(t, (x, y), (x + bw, y + bh), col, max(2, t.shape[1] // 400))
        h = 420; s = h / t.shape[0]
        thumbs.append((f.name, cv2.resize(t, (int(t.shape[1] * s), h)), cands))
    per = args.per_sheet
    for i in range(0, len(thumbs), per):
        chunk = thumbs[i:i + per]
        W = sum(t[1].shape[1] for t in chunk)
        sheet = np.full((440, W, 3), 25, np.uint8)
        x = 0
        for name, t, _ in chunk:
            sheet[10:10 + t.shape[0], x:x + t.shape[1]] = t
            cv2.putText(sheet, name, (x + 6, 434), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x += t.shape[1]
        cv2.imwrite(str(viz / f'proposal_{i // per:02d}.png'), sheet)
    (repo / 'outputs/shuttle_capability/annotations').mkdir(parents=True, exist_ok=True)
    (repo / 'outputs/shuttle_capability/annotations/proposals_real_images.json').write_text(
        json.dumps(proposals, indent=1))
    print(f'images={len(files)} sheets={math.ceil(len(thumbs) / per)} proposals written')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
