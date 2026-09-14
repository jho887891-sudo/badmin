#!/usr/bin/env python3
"""03_analyze_before.py - WHY did anything match?  Class-level analysis + negatives + visuals.

A COCO model has no shuttlecock class, so a raw IoU match is meaningless until we know which
class produced it.  This script dumps every detection with its class/conf/IoU, measures false
positives on shuttle-free frames, and writes annotated visual evidence.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def read_label(path: Path, W, H):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        _, xc, yc, bw, bh = int(p[0]), *[float(v) for v in p[1:]]
        out.append(((xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H))
    return out


def make_negatives(out: Path, n: int, imgsz: int, seed: int):
    import cv2
    rng = np.random.default_rng(seed)
    d = out / 'synthetic_3d' / 'negatives'
    d.mkdir(parents=True, exist_ok=True)
    files = []
    H = W = imgsz
    for i in range(n):
        style = i % 4
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        if style == 0:
            base = np.full((H, W), 120, np.float32)
        elif style == 1:
            base = 60 + 120 * (yy / (H - 1))
        elif style == 2:
            base = np.full((H, W), 70, np.float32) + 25 * (yy / (H - 1))
        else:
            base = 100 + 40 * rng.standard_normal((H, W)).astype(np.float32)
        img = np.clip(np.stack([base, base, base], -1) * (0.9 + 0.2 * rng.random()), 0, 255)
        cv2.imwrite(str(d / f'neg_{i:04d}.png'), img.astype('uint8'))
        files.append(f'neg_{i:04d}.png')
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--n-neg', type=int, default=40)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'shuttle_capability'
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments' / 'shuttle_capability' / '_scratch' / 'ultralytics_cfg')
    import cv2
    from ultralytics import YOLO

    with (out / 'metrics' / 'synthetic_manifest.csv').open() as fh:
        manifest = list(csv.DictReader(fh))
    model = YOLO(str(repo / 'assets/external/_staging/F_yolo/weights/yolo26s.pt'))
    names = model.names
    neg_files = make_negatives(out, args.n_neg, args.imgsz, 0)
    print(f'positives={len(manifest)}  negatives={len(neg_files)}  imgsz={args.imgsz} conf={args.conf}')

    det_rows, class_hits = [], Counter()
    hit_conf = defaultdict(list)
    for row in manifest:
        p = out / 'synthetic_3d' / 'images' / row['file']
        img = cv2.imread(str(p)); H, W = img.shape[:2]
        gts = read_label(out / 'annotations' / 'synthetic_3d' / row['gt_label'], W, H)
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=0.7, verbose=False, device=args.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            det_rows.append({'file': row['file'], 'gt_size_px': row['equivalent_size_px'],
                             'class_id': -1, 'class_name': 'NONE', 'conf': 0.0, 'iou_to_gt': 0.0, 'hit': 0})
            continue
        boxes = res.boxes.xyxy.cpu().numpy(); scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        best = (0.0, -1, 0.0)
        for b, s, c in zip(boxes, scores, classes):
            v = max([iou(g, tuple(b)) for g in gts], default=0.0)
            det_rows.append({'file': row['file'], 'gt_size_px': row['equivalent_size_px'],
                             'class_id': int(c), 'class_name': str(names[int(c)]),
                             'conf': round(float(s), 4), 'iou_to_gt': round(float(v), 4),
                             'hit': int(v >= args.iou)})
            if v >= args.iou and v > best[0]:
                best = (v, int(c), float(s))
        if best[1] >= 0:
            class_hits[str(names[best[1]])] += 1
            hit_conf[str(names[best[1]])].append(best[2])

    det_csv = out / 'predictions_before' / 'detections_before.csv'
    with det_csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(det_rows[0].keys())); w.writeheader(); w.writerows(det_rows)

    neg_boxes, neg_confs = 0, []
    for f in neg_files:
        img = cv2.imread(str(out / 'synthetic_3d' / 'negatives' / f))
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=0.7, verbose=False, device=args.device)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        neg_boxes += n
        if n:
            neg_confs.extend(res.boxes.conf.cpu().numpy().tolist())

    print('\n=== which classes produced the GT matches (IoU>=%.2f) ===' % args.iou)
    for cls, n in class_hits.most_common():
        print(f'  {cls:16s} hits={n:3d}  mean_conf={np.mean(hit_conf[cls]):.3f}')
    if not class_hits:
        print('  (none)')
    print(f'\nnegatives: {len(neg_files)} shuttle-free frames -> {neg_boxes} boxes at conf>={args.conf}'
          f' ({neg_boxes / max(len(neg_files), 1):.2f} FP/frame), mean_conf={np.mean(neg_confs) if neg_confs else 0:.3f}')
    print('wrote', det_csv)
    (out / 'metrics' / 'class_hits_before.txt').write_text(
        '\n'.join([f'{k}\t{v}' for k, v in class_hits.most_common()]) +
        f'\nnegatives_fp_per_frame\t{neg_boxes / max(len(neg_files), 1):.4f}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
