#!/usr/bin/env python3
"""08_verify_sheets.py - build compact verification sheets (union box + coords) for P0 review."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2, numpy as np

p = argparse.ArgumentParser(); p.add_argument('--repo', default='/home/T7/ojh/robot_sim'); p.add_argument('--per-sheet', type=int, default=9); p.add_argument('--tile', type=int, default=430)
a = p.parse_args(); repo = Path(a.repo)
props = json.loads((repo / 'outputs/shuttle_capability/annotations/proposals_real_images.json').read_text())
meta = {r['file']: r for r in __import__('csv').DictReader((repo / 'outputs/shuttle_capability/real_images/real_images_metadata.csv').open())}
src = repo / 'outputs/shuttle_capability/real_images/raw'
viz = repo / 'outputs/shuttle_capability/visualizations/gt_verify'; viz.mkdir(parents=True, exist_ok=True)
names = sorted(props.keys())
entries = []
for name in names:
    img = cv2.imread(str(src / name))
    if img is None: continue
    H, W = img.shape[:2]
    big = [c for c in props[name] if c[4] >= 400]
    if big:
        x1 = min(c[0] for c in big); y1 = min(c[1] for c in big)
        x2 = max(c[0]+c[2] for c in big); y2 = max(c[1]+c[3] for c in big)
        ux1,uy1,ux2,uy2 = max(0,x1-4), max(0,y1-4), min(W,x2+4), min(H,y2+4)
    else:
        ux1=uy1=ux2=uy2=0
    bw,bh = ux2-ux1, uy2-uy1
    eq = float(np.sqrt(bw*bh)) if bw>0 and bh>0 else 0.0
    entries.append((name, img, (ux1,uy1,ux2,uy2), bw, bh, eq, len(big)))
print('entries:', len(entries))
per = a.per_sheet; tile = a.tile; cols = 3
for si in range(0, len(entries), per):
    chunk = entries[si:si+per]
    rows = math.ceil(len(chunk)/cols)
    sheet = np.full((rows*tile, cols*tile, 3), 25, np.uint8)
    for k,(name,img,box,bw,bh,eq,nbig) in enumerate(chunk):
        t = img.copy()
        if bw>0: cv2.rectangle(t, (box[0],box[1]), (box[2],box[3]), (0,0,255), max(2, t.shape[1]//500))
        s = min(tile/t.shape[0], tile/t.shape[1])
        t = cv2.resize(t, (int(t.shape[1]*s), int(t.shape[0]*s)))
        y, x = (k//cols)*tile, (k%cols)*tile
        sheet[y:y+t.shape[0], x:x+t.shape[1]] = t
        label = f'{name} {bw}x{bh} eq={eq:.0f}px'
        cv2.putText(sheet, label, (x+6, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.imwrite(str(viz / f'verify_{si//per:02d}.png'), sheet)
print('sheets written:', math.ceil(len(entries)/per), 'to', viz)