#!/usr/bin/env python3
"""03_dataset_audit.py - find a shuttlecock detection dataset (pruned walk, fast).

Reports dataset YAML / image counts / YOLO labels, or DATASET_NOT_AVAILABLE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SKIP_DIRS = {'env_isaaclab', 'IsaacLab', 'cache', 'home', '.git', '__pycache__', 'assets',
             '_wheel_extract', '_deps', 'node_modules', '.pytest_cache'}
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = Path(args.out) if args.out else repo / 'outputs' / 'yolo26_p2_ab_test'
    out.mkdir(parents=True, exist_ok=True)

    dataset_yamls, images, label_txt, label_dirs = [], [], [], []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = Path(dirpath).relative_to(repo)
        if any(part in SKIP_DIRS for part in rel_dir.parts):
            continue
        for name in filenames:
            scanned += 1
            p = Path(dirpath) / name
            rel = str(p.relative_to(repo))
            if name.endswith('.yaml'):
                try:
                    txt = p.read_text(errors='ignore')
                except Exception:
                    txt = ''
                if re.search(r'^\s*(train|val|test)\s*:', txt, re.M) and ('nc' in txt or 'names' in txt):
                    dataset_yamls.append(rel)
            if p.suffix.lower() in IMG_EXT:
                images.append(rel)
            if p.suffix == '.txt' and name != 'requirements.txt':
                try:
                    first = p.read_text(errors='ignore').splitlines()[:1]
                except Exception:
                    first = []
                if first and re.match(r'^\s*\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$', first[0]):
                    label_txt.append(rel)
            if p.is_dir() and name in ('labels', 'labels_train', 'labels_val'):
                label_dirs.append(rel)

    summary = {
        'scanned_files': scanned,
        'dataset_yamls': dataset_yamls[:20],
        'dataset_yaml_count': len(dataset_yamls),
        'image_count': len(images),
        'image_examples': images[:10],
        'yolo_label_files': len(label_txt),
        'yolo_label_examples': label_txt[:10],
        'verdict': 'DATASET_PRESENT' if (images and label_txt) else 'DATASET_NOT_AVAILABLE',
        'note': 'assets/ and cache/ are excluded by design; no shuttlecock dataset is expected there',
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / 'dataset_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('wrote', out / 'dataset_audit.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
