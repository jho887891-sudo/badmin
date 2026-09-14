#!/usr/bin/env python3
"""02_eval_capability.py - measure what the CURRENT model can detect (protocol sections 2,6,20,28).

For every capability image: run the detector, match GT by IoU, and record the detection
outcome per target size bucket.  Reports both ANY-CLASS recall (what a coarse ROI proposal
needs) and named-class recall (e.g. the COCO 'sports ball' class).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

BUCKETS = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 1e9)]


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def bucket_of(size):
    for lo, hi in BUCKETS:
        if lo <= size < hi:
            return f'{lo}-{hi}' if hi < 1e9 else f'>{lo}'
    return 'unknown'


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def read_yolo_label(path: Path, W, H):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls, xc, yc, bw, bh = int(p[0]), *[float(v) for v in p[1:]]
        x1 = (xc - bw / 2) * W; y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W; y2 = (yc + bh / 2) * H
        rows.append((cls, x1, y1, x2, y2))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--weights', default=None)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--imgsz-list', type=int, nargs='+', default=[640, 960, 1280])
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--tag', default='before')
    ap.add_argument('--device', default='0')
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'shuttle_capability'
    man_path = Path(args.manifest) if args.manifest else out / 'metrics' / 'synthetic_manifest.csv'
    weights = Path(args.weights) if args.weights else (
        repo / 'assets' / 'external' / '_staging' / 'F_yolo' / 'weights' / 'yolo26s.pt')
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments' / 'shuttle_capability' / '_scratch' / 'ultralytics_cfg')

    import cv2
    from ultralytics import YOLO

    with man_path.open() as fh:
        manifest = list(csv.DictReader(fh))
    model = YOLO(str(weights))
    names = model.names
    sports = [k for k, v in names.items() if str(v) == 'sports ball']
    print(f'weights={weights.name} conf={args.conf} iou={args.iou} images={len(manifest)}')

    per_image, viz = [], []
    for imgsz in args.imgsz_list:
        for row in manifest:
            img_path = out / 'synthetic_3d' / 'images' / row['file']
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            gts = read_yolo_label(out / 'annotations' / 'synthetic_3d' / row['gt_label'], W, H)
            res = model.predict(img, imgsz=imgsz, conf=args.conf, iou=0.7, verbose=False, device=args.device)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
            classes = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), int)
            best_any, best_named = 0.0, 0.0
            for _cls, x1, y1, x2, y2 in gts:
                for b, s, c in zip(boxes, scores, classes):
                    v = iou((x1, y1, x2, y2), tuple(b))
                    best_any = max(best_any, v)
                    if int(c) in sports:
                        best_named = max(best_named, v)
            per_image.append({
                'imgsz': imgsz, 'file': row['file'], 'source_type': row['source_type'],
                'distance_m': row['distance_m'], 'yaw_deg': row['yaw_deg'], 'pitch_deg': row['pitch_deg'],
                'roll_deg': row['roll_deg'], 'pos_x_px': row['pos_x_px'], 'pos_y_px': row['pos_y_px'],
                'bbox_w_px': row['bbox_w_px'], 'bbox_h_px': row['bbox_h_px'],
                'equivalent_size_px': row['equivalent_size_px'], 'blur': row['blur'],
                'n_pred': len(boxes), 'n_gt': len(gts),
                'best_iou_any': round(best_any, 4), 'best_iou_sports_ball': round(best_named, 4),
                'hit_any': int(best_any >= args.iou), 'hit_named': int(best_named >= args.iou),
                'max_conf': round(float(scores.max()), 4) if len(scores) else 0.0,
            })
            if len(viz) < 12 and imgsz == args.imgsz_list[-1]:
                viz.append((row['file'], float(row['equivalent_size_px']), best_any, len(boxes)))

    pred_csv = out / f'predictions_{args.tag}' / f'predictions_{args.tag}.csv'
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with pred_csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        w.writeheader(); w.writerows(per_image)

    buckets = []
    for lo, hi in BUCKETS:
        label = f'{lo}-{hi}' if hi < 1e9 else f'>{lo}'
        for imgsz in args.imgsz_list:
            sel = [r for r in per_image if r['imgsz'] == imgsz
                   and lo <= float(r['equivalent_size_px']) < hi]
            if not sel:
                buckets.append({'bucket': label, 'imgsz': imgsz, 'images': 0, 'recall_any': None,
                                'recall_sports_ball': None})
                continue
            buckets.append({'bucket': label, 'imgsz': imgsz, 'images': len(sel),
                            'recall_any': round(sum(r['hit_any'] for r in sel) / len(sel), 4),
                            'recall_sports_ball': round(sum(r['hit_named'] for r in sel) / len(sel), 4)})
    b_csv = out / 'metrics' / f'size_bucket_metrics_{args.tag}.csv'
    with b_csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(buckets[0].keys()))
        w.writeheader(); w.writerows(buckets)

    summary = {'weights': weights.name, 'conf': args.conf, 'iou': args.iou,
               'imgsz_list': args.imgsz_list, 'images': len(manifest),
               'overall_recall_any': {str(s): round(float(np.mean([r['hit_any'] for r in per_image
                                                                   if r['imgsz'] == s])), 4) for s in args.imgsz_list},
               'overall_recall_sports_ball': {str(s): round(float(np.mean([r['hit_named'] for r in per_image
                                                                           if r['imgsz'] == s])), 4) for s in args.imgsz_list}}
    (out / 'metrics' / f'capability_{args.tag}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('\nbucket table (imgsz=%d):' % args.imgsz_list[-1])
    for b in buckets:
        if b['imgsz'] == args.imgsz_list[-1]:
            print(f"  {b['bucket']:>7s} imgs={b['images']:3d}  recall_any={b['recall_any']}  recall_sports_ball={b['recall_sports_ball']}")
    print('wrote', pred_csv, '|', b_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
