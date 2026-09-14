#!/usr/bin/env python3
"""00_target_check.py - what can the current model actually detect? (protocol section 1)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--weights', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'shuttle_capability'
    (out / 'reports').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments' / 'shuttle_capability' / '_scratch' / 'ultralytics_cfg')
    w = Path(args.weights) if args.weights else (
        repo / 'assets' / 'external' / '_staging' / 'F_yolo' / 'weights' / 'yolo26s.pt')

    from ultralytics import YOLO
    model = YOLO(str(w))
    names = model.names
    nc = len(names)
    hits = {k: v for k, v in names.items() if any(
        t in str(v).lower() for t in ('shuttle', 'badminton', 'birdie', 'ball'))}
    report = {
        'weights': w.name,
        'nc': nc,
        'names': names,
        'shuttlecock_class_present': any('shuttle' in str(v).lower() for v in names.values()),
        'badminton_class_present': any('badminton' in str(v).lower() for v in names.values()),
        'ball_like_classes': hits,
        'task': getattr(model, 'task', None),
        'verdict': 'TARGET_CLASS_NOT_PRESENT' if nc == 80 and 'shuttle' not in str(names).lower() else 'SEE_NAMES',
        'target_2': 'UNKNOWN / NOT_SPECIFIED',
    }
    print(json.dumps({k: report[k] for k in ('weights', 'nc', 'task',
                                            'shuttlecock_class_present', 'badminton_class_present',
                                            'ball_like_classes', 'verdict', 'target_2')},
                     indent=2, ensure_ascii=False))
    print('full names:', json.dumps(names))
    (out / 'reports' / 'target_check.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('wrote', out / 'reports' / 'target_check.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
