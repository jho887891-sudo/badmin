#!/usr/bin/env python3
"""01_discover.py - model/config discovery without installing anything.

Extracts the archived ultralytics wheel into a scratch directory and enumerates the OFFICIAL
model configs it ships, so that the baseline vs P2 detection levels are read from the package
itself (not assumed).  READ-ONLY with respect to the project and the environment.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path


def find_repo(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'env_isaaclab').is_dir() and (candidate / 'src').is_dir():
            return candidate
    raise SystemExit('repository root not found (needs env_isaaclab/ and src/)')


def extract_wheel(wheel: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(dest)
    return dest


def detection_levels(yaml_text: str) -> dict:
    """Crude, dependency-free read of the head of a model YAML."""
    info = {}
    for key in ('nc', 'scales', 'backbone', 'head'):
        m = re.search(rf'^{key}:', yaml_text, re.M)
        info[key + '_present'] = bool(m)
    head = yaml_text.split('head:', 1)[1] if 'head:' in yaml_text else ''
    detect = re.findall(r'\[\s*\d+\s*,\s*1\s*,\s*Detect\s*,\s*\[([^\]]*)\]\s*\]', head)
    info['detect_from_layers'] = detect
    levels = re.findall(r'\[\s*\d+\s*,\s*1\s*,(?:Detect|Segment|Pose|OBB)\s*,\s*\[([^\]]*)\]\s*\]', head)
    info['head_from_layers'] = levels
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = Path(args.out) if args.out else repo / 'outputs' / 'yolo26_p2_ab_test'
    out.mkdir(parents=True, exist_ok=True)
    scratch = repo / 'experiments' / 'yolo26_p2_ab' / '_wheel_extract'

    wheels = sorted((repo / 'assets' / 'external' / '_staging' / 'F_yolo' / 'wheels').glob('ultralytics-*.whl'))
    weights = sorted((repo / 'assets' / 'external' / '_staging' / 'F_yolo' / 'weights').glob('yolo26*'))
    print('ultralytics wheels found:', [w.name for w in wheels])
    print('yolo26 weights found    :', [w.name for w in weights])
    if not wheels:
        print('BLOCKED: no ultralytics wheel archived')
        return 2

    root = extract_wheel(wheels[-1], scratch)
    cfg_root = root / 'ultralytics' / 'cfg' / 'models'
    print('extracted wheel to:', scratch)
    print('model config families:', sorted(p.name for p in cfg_root.iterdir() if p.is_dir()))

    y26 = sorted(p for p in cfg_root.rglob('*.yaml') if '26' in p.name)
    print(f'\nYOLO26 configs shipped by this package: {len(y26)}')
    for p in y26:
        print('  ', p.relative_to(cfg_root))

    p2_names = [p for p in y26 if 'p2' in p.name.lower()]
    print('\nYOLO26 configs with P2 in the name:', [p.name for p in p2_names] or 'NONE')

    for p in y26:
        text = p.read_text()
        info = detection_levels(text)
        print(f'\n--- {p.name} ---')
        print('   detect head from-layers:', info['detect_from_layers'] or info['head_from_layers'])
        print('   has scales:', info['scales_present'], '| has backbone:', info['backbone_present'],
              '| has head:', info['head_present'])

    report = out / 'model_configs_discovered.txt'
    with report.open('w') as fh:
        fh.write(f'wheel: {wheels[-1].name}\n')
        fh.write('yolo26 configs:\n')
        for p in y26:
            fh.write('  ' + str(p.relative_to(cfg_root)) + '\n')
        fh.write('p2-named configs: ' + str([p.name for p in p2_names]) + '\n')
    print('\nwrote', report)
    return 0 if p2_names else 3


if __name__ == '__main__':
    raise SystemExit(main())
