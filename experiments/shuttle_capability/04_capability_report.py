#!/usr/bin/env python3
"""04_capability_report.py - capability_before table with a confidence sweep (protocol 6,14,28)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BUCKETS = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 1e9)]
CONFS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70, 0.90]


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def bucket_label(size):
    for lo, hi in BUCKETS:
        if lo <= size < hi:
            return f'{lo}-{hi}px' if hi < 1e9 else f'>{lo}px'
    return 'unknown'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'shuttle_capability'
    with (out / 'metrics' / 'synthetic_manifest.csv').open() as fh:
        manifest = {r['file']: r for r in csv.DictReader(fh)}
    with (out / 'predictions_before' / 'detections_before.csv').open() as fh:
        dets = list(csv.DictReader(fh))

    per_file = defaultdict(list)
    for d in dets:
        per_file[d['file']].append(d)

    rows = []
    for conf in CONFS:
        for name, meta in manifest.items():
            size = float(meta['equivalent_size_px'])
            ds = per_file.get(name, [])
            live = [d for d in ds if float(d['conf']) >= conf]
            hit_any = any(int(d['hit']) == 1 for d in live)
            hit_named = any(int(d['hit']) == 1 and d['class_name'] == 'sports ball' for d in live)
            rows.append({'conf': conf, 'file': name, 'bucket': bucket_label(size),
                         'equivalent_size_px': size, 'blur': meta['blur'],
                         'distance_m': meta['distance_m'], 'hit_any': int(hit_any),
                         'hit_sports_ball': int(hit_named)})

    cap = out / 'metrics' / 'capability_before.csv'
    with cap.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lines = ['# Capability BEFORE training (current yolo26s.pt = COCO 80 classes)',
             '', '## Recall_any (any class, IoU>=0.5) vs confidence threshold and target size', 
             '', '| bucket | ' + ' | '.join(f'c>={c}' for c in CONFS) + ' |',
             '|---|' + '---|' * len(CONFS)]
    for lo, hi in BUCKETS:
        label = f'{lo}-{hi}px' if hi < 1e9 else f'>{lo}px'
        cells = []
        for conf in CONFS:
            sel = [r for r in rows if r['bucket'] == label and r['conf'] == conf]
            cells.append('n/a' if not sel else f"{sum(r['hit_any'] for r in sel) / len(sel):.2f} ({len(sel)})")
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Recall_sports_ball (COCO class 32) vs confidence threshold',
              '', '| bucket | ' + ' | '.join(f'c>={c}' for c in CONFS) + ' |',
              '|---|' + '---|' * len(CONFS)]
    for lo, hi in BUCKETS:
        label = f'{lo}-{hi}px' if hi < 1e9 else f'>{lo}px'
        cells = []
        for conf in CONFS:
            sel = [r for r in rows if r['bucket'] == label and r['conf'] == conf]
            cells.append('n/a' if not sel else f"{sum(r['hit_sports_ball'] for r in sel) / len(sel):.2f}")
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    (out / 'reports' / 'CAPABILITY_BEFORE.md').write_text(text)
    print(text)
    print('wrote', cap, '|', out / 'reports' / 'CAPABILITY_BEFORE.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
