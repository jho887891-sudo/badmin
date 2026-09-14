#!/usr/bin/env python3
"""10_extract_frames.py - fixed-rate frame extraction + contact sheets for P2 (real video)."""
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/home/T7/ojh/robot_sim')
p.add_argument('--fps', type=float, default=2.0, help='sampling rate')
p.add_argument('--max-per-video', type=int, default=60)
p.add_argument('--tile', type=int, default=320)
a = p.parse_args()
repo = Path(a.repo)
vdir = repo / 'outputs/shuttle_capability/real_video/clips'
fdir = repo / 'outputs/shuttle_capability/real_video/frames'
sdir = repo / 'outputs/shuttle_capability/visualizations/video_sheets'
for d in (fdir, sdir): d.mkdir(parents=True, exist_ok=True)
rows = []
for clip in sorted(vdir.glob('*.webm')) + sorted(vdir.glob('*.mp4')):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        print('cannot open', clip.name); continue
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(int(round(src_fps / a.fps)), 1)
    print(f'{clip.name}: {w}x{h} fps={src_fps:.1f} frames={n_total} step={step}')
    idx = saved = 0
    thumbs = []
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            name = f'{clip.stem}_f{idx:05d}.png'
            cv2.imwrite(str(fdir / name), frame)
            rows.append({'clip': clip.name, 'frame_file': name, 'frame_index': idx,
                         't_s': round(idx / src_fps, 3), 'width': w, 'height': h})
            t = frame.copy()
            s = a.tile / max(t.shape[:2])
            thumbs.append((name, cv2.resize(t, (int(t.shape[1]*s), int(t.shape[0]*s)))))
            saved += 1
            if saved >= a.max_per_video: break
        idx += 1
    cap.release()
    cols = 6
    per = 24
    for si in range(0, len(thumbs), per):
        chunk = thumbs[si:si+per]
        rws = math.ceil(len(chunk)/cols)
        sheet = np.full((rws*a.tile, cols*a.tile, 3), 20, np.uint8)
        for k,(nm,th) in enumerate(chunk):
            y, x = (k//cols)*a.tile, (k%cols)*a.tile
            sheet[y:y+th.shape[0], x:x+th.shape[1]] = th
            cv2.putText(sheet, nm.split('_f')[-1].replace('.png',''), (x+4,y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imwrite(str(sdir / f'{clip.stem}_sheet{si//per:02d}.png'), sheet)
    print(f'  saved {saved} frames')
with (repo / 'outputs/shuttle_capability/real_video/frames_manifest.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('total frames:', len(rows))