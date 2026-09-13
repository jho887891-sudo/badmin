#!/usr/bin/env python3
"""05_weight_transfer.py - how much of the COCO-pretrained baseline transfers into P2.

Reports matched / missing / unexpected tensors and their parameter counts.
This is PARTIAL_PRETRAIN_TRANSFER, never "pretrained weights fully loaded".
"""
from __future__ import annotations

import argparse
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
    ap.add_argument('--scale', default='s')
    ap.add_argument('--weights', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'yolo26_p2_ab_test'
    out.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments' / 'yolo26_p2_ab' / '_scratch' / 'ultralytics_cfg')
    weights = Path(args.weights) if args.weights else (
        repo / 'assets' / 'external' / '_staging' / 'F_yolo' / 'weights' / f'yolo26{args.scale}.pt')

    import torch
    from ultralytics import YOLO

    print('pretrained weights:', weights, '| exists:', weights.exists())
    base = YOLO(str(weights))
    base_sd = base.model.state_dict()
    p2 = YOLO(f'yolo26{args.scale}-p2.yaml')
    p2_sd = p2.model.state_dict()

    base_keys, p2_keys = set(base_sd), set(p2_sd)
    matched = sorted(base_keys & p2_keys)
    missing = sorted(p2_keys - base_keys)
    unexpected = sorted(base_keys - p2_keys)
    matched_shapes_ok = [k for k in matched if tuple(base_sd[k].shape) == tuple(p2_sd[k].shape)]
    shape_mismatch = [k for k in matched if tuple(base_sd[k].shape) != tuple(p2_sd[k].shape)]
    n_matched = sum(base_sd[k].numel() for k in matched_shapes_ok)
    n_missing = sum(p2_sd[k].numel() for k in missing)
    n_unexpected = sum(base_sd[k].numel() for k in unexpected)
    total_p2 = sum(v.numel() for v in p2_sd.values())

    report = [
        'weight transfer: baseline -> P2',
        f'source weights        : {weights.name}',
        f'target config         : yolo26{args.scale}-p2.yaml',
        f'VERDICT               : PARTIAL_PRETRAIN_TRANSFER',
        '',
        f'tensors in baseline   : {len(base_keys)}',
        f'tensors in P2         : {len(p2_keys)}',
        f'shape-compatible      : {len(matched_shapes_ok)}  ({n_matched:,} params)',
        f'shape mismatch        : {len(shape_mismatch)}  {shape_mismatch[:5]}',
        f'missing (P2 only)     : {len(missing)}  ({n_missing:,} params)',
        f'unexpected(only base) : {len(unexpected)}  ({n_unexpected:,} params)',
        f'P2 total params       : {total_p2:,}',
        f'coverage of P2 params : {100.0 * n_matched / total_p2:.2f}%',
        '',
        'missing keys (first 20):',
    ] + [f'  - {k}' for k in missing[:20]] + ['', 'unexpected keys (first 20):'] + [
        f'  - {k}' for k in unexpected[:20]]
    text = '\n'.join(report)
    print(text)
    (out / 'weight_transfer_report.txt').write_text(text + '\n')

    load_result = p2.model.load_state_dict(base_sd, strict=False)
    loaded = len(base_sd) - len(load_result.unexpected_keys)
    print(f'\nload_state_dict(strict=False): loaded={loaded}/{len(base_sd)} tensors '
          f'missing_keys={len(load_result.missing_keys)} unexpected_keys={len(load_result.unexpected_keys)}')
    print('wrote', out / 'weight_transfer_report.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
