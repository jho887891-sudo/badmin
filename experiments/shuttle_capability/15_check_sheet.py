#!/usr/bin/env python3
"""15_check_sheet.py - build a labelled check sheet from a train/val split."""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/home/T7/ojh/robot_sim')
p.add_argument('--split', default='val')
p.add_argument('--tile', type=int, default=400)
p.add_argument('--cols', type=int, default=4)
p.add_argument('--max', type=int, default=16)
a = p.parse_args()
repo = Path(a.repo)
sys.path.insert(0, str(repo / 'tools'))
from shuttle_render import imread_unicode, imwrite_unicode  # noqa: E402
src = repo / 'outputs/shuttle_capability/train_data' / a.split / 'images'
lab = repo / 'outputs/shuttle_capability/train_data' / a.split / 'labels'
files = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
files = files[:a.max]
rows = math.ceil(len(files)/a.cols)
sheet = np.full((rows*a.tile, a.cols*a.tile, 3), 20, np.uint8)
for i, f in enumerate(files):
    im = imread_unicode(f)
    if im is None: continue
    t = lab / (f.stem + '.txt')
    if t.exists():
        parts = t.read_text().split()
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        H, W = im.shape[:2]
        x1, y1 = int((xc-bw/2)*W), int((yc-bh/2)*H)
        x2, y2 = int((xc+bw/2)*W), int((yc+bh/2)*H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        eq = math.sqrt(bw*W*bh*H)
        cv2.putText(im, f'{eq:.1f}px', (max(2,x1), max(14, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    s = a.tile / max(im.shape[:2])
    im = cv2.resize(im, (int(im.shape[1]*s), int(im.shape[0]*s)))
    y, x = (i//a.cols)*a.tile, (i%a.cols)*a.tile
    sheet[y:y+im.shape[0], x:x+im.shape[1]] = im
out = repo / 'outputs/shuttle_capability/visualizations' / f'check_{a.split}.png'
imwrite_unicode(out, sheet)
print('sheet:', out, sheet.shape, 'n=', len(files))