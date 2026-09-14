#!/usr/bin/env python3
"""11_motion_candidates.py - find small moving blobs (shuttle candidates) in real badminton video.

Broadcast footage has multi-pixel shuttles; a tile view cannot resolve them.  This proposes
candidates by frame differencing + compactness, then emits ZOOM CROPS for visual verification.
Nothing is called GT until the crops are looked at.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/home/T7/ojh/robot_sim')
p.add_argument('--clip', default='eurogames_ccbysa4_619.webm')
p.add_argument('--start-s', type=float, default=0.0)
p.add_argument('--span-s', type=float, default=12.0)
p.add_argument('--top', type=int, default=6)
p.add_argument('--zoom', type=int, default=160)
p.add_argument('--out-scale', type=int, default=420)
a = p.parse_args()
repo = Path(a.repo)
clip = repo / 'outputs/shuttle_capability/real_video/clips' / a.clip
outdir = repo / 'outputs/shuttle_capability/real_video/candidates'; outdir.mkdir(parents=True, exist_ok=True)
viz = repo / 'outputs/shuttle_capability/visualizations/shuttle_candidates'; viz.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(clip))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start_s * fps))
n = int(a.span_s * fps)
prev = None; cands = []
for i in range(n):
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        d = cv2.absdiff(g, prev)
        _, th = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        nn, lab, st, ce = cv2.connectedComponentsWithStats(th, 8)
        for k in range(1, nn):
            x,y,bw,bh,area = st[k]
            if 4 <= area <= 400 and 2 <= bw <= 30 and 2 <= bh <= 30:
                fill = area / float(bw*bh)
                if fill > 0.25:
                    cands.append((area, x, y, bw, bh, i, float(fill)))
    prev = g
cap.release()
cands.sort(key=lambda c: -c[0])
print(f'candidates found: {len(cands)} in {n} frames')
rows = []; tiles = []
seen = set()
for area,x,y,bw,bh,fi,fill in cands:
    key = (fi//15, x//60, y//60)
    if key in seen: continue
    seen.add(key)
    rows.append({'frame_index': fi, 't_s': round(fi/fps,3), 'x':int(x), 'y':int(y),
                 'w':int(bw), 'h':int(bh), 'area':int(area), 'fill':round(fill,3)})
    if len(rows) >= a.top*3: break
cap = cv2.VideoCapture(str(clip)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start_s*fps))
for r in rows[:a.top]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, r['frame_index'])
    ok, fr = cap.read()
    if not ok: continue
    cx, cy = r['x']+r['w']//2, r['y']+r['h']//2
    half = a.zoom
    x0,y0 = max(0,cx-half), max(0,cy-half)
    crop = fr[y0:y0+2*half, x0:x0+2*half].copy()
    if crop.size == 0: continue
    cv2.rectangle(crop, (r['x']-x0-4, r['y']-y0-4), (r['x']-x0+r['w']+4, r['y']-y0+r['h']+4), (0,0,255), 2)
    s = a.out_scale / crop.shape[0]
    crop = cv2.resize(crop, (a.out_scale, a.out_scale))
    tiles.append((f"f{r['frame_index']} {r['w']}x{r['h']}px", crop))
cap.release()
cols = 3; per = 9
for si in range(0, len(tiles), per):
    ch = tiles[si:si+per]; rws = math.ceil(len(ch)/cols)
    sheet = np.full((rws*a.out_scale, cols*a.out_scale, 3), 20, np.uint8)
    for k,(lab,im) in enumerate(ch):
        yy, xx = (k//cols)*a.out_scale, (k%cols)*a.out_scale
        sheet[yy:yy+im.shape[0], xx:xx+im.shape[1]] = im
        cv2.putText(sheet, lab, (xx+6, yy+24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imwrite(str(viz / f'{clip.stem}_cand{si//per:02d}.png'), sheet)
with (outdir / f'{clip.stem}_candidates.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('candidate crops written:', math.ceil(len(tiles)/per), 'sheets; csv rows:', len(rows))