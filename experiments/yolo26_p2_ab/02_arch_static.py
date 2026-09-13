#!/usr/bin/env python3
"""02_arch_static.py - static architecture comparison from the OFFICIAL configs.

Prints the head sections verbatim and derives the detection feature levels of
yolo26.yaml (baseline) vs yolo26-p2.yaml (P2).  No dependencies beyond the stdlib.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def head_block(text: str) -> list:
    lines = text.splitlines()
    out, inside = [], False
    for ln in lines:
        if re.match(r'^head:', ln):
            inside = True
            out.append(ln)
            continue
        if inside:
            if ln and not ln.startswith((' ', '\t', '-')) and not ln.startswith('#'):
                break
            out.append(ln)
    return [l for l in out if l.strip() and not l.strip().startswith('#')]


def detect_layers(head_lines: list) -> list:
    """Return [[sources], layer_name, comment] for the terminal detection layer(s)."""
    out = []
    for ln in head_lines:
        m = re.search(r'\[\s*\[([^\]]+)\]\s*,\s*\d+\s*,\s*(Detect|Segment|Pose|OBB)', ln)
        if not m:
            continue
        sources = [int(x) for x in re.findall(r'-?\d+', m.group(1))]
        comment = ln.split('#', 1)[1].strip() if '#' in ln else ''
        out.append([sources, m.group(2), comment])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    cfg = repo / 'experiments' / 'yolo26_p2_ab' / '_wheel_extract' / 'ultralytics' / 'cfg' / 'models' / '26'
    if not cfg.is_dir():
        print('BLOCKED: extracted configs not found at', cfg, '- run 01_discover.py first')
        return 2
    print('config dir:', cfg)
    print('all yolo26 configs:', sorted(p.name for p in cfg.glob('yolo26*.yaml')))
    for name in ('yolo26.yaml', 'yolo26-p2.yaml'):
        p = cfg / name
        text = p.read_text()
        print('\n' + '=' * 72)
        print('# ' + name)
        print('=' * 72)
        hb = head_block(text)
        for ln in hb:
            print(ln)
        det = detect_layers(hb)
        for sources, name, comment in det:
            print(f'\n-> DETECTION LAYER: {name} from head layers {sources}  [{comment}]')
            print(f'-> detection levels: {len(sources)}')
        m = re.search(r'^scales:\s*\n((?:\s+\[[^\]]+\]\s*\n)+)', text, re.M)
        if m:
            print('-> scales block:')
            print(m.group(1).rstrip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
