#!/usr/bin/env python3
"""04_model_summary.py - build BOTH official configs at the requested scale and compare.

Scale is explicit (default s = YOLO26s) so the two variants differ ONLY by the P2 branch.
Nothing is installed into the project venv; the wheel is used through PYTHONPATH.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def describe(obj, depth=0):
    """Structure of a possibly nested prediction output."""
    if hasattr(obj, 'shape'):
        return f'tensor{tuple(obj.shape)}'
    if isinstance(obj, dict):
        return '{' + ', '.join(f'{k}: {describe(v, depth + 1)}' for k, v in obj.items()) + '}'
    if isinstance(obj, (list, tuple)):
        inner = ', '.join(describe(v, depth + 1) for v in obj)
        return f'[{inner}]'
    return type(obj).__name__


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--scale', default='s')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'yolo26_p2_ab_test'
    out.mkdir(parents=True, exist_ok=True)
    scratch = repo / 'experiments' / 'yolo26_p2_ab' / '_scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(scratch / 'ultralytics_cfg')

    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load

    combos = [('baseline', f'yolo26{args.scale}.yaml'), ('p2', f'yolo26{args.scale}-p2.yaml')]
    rows = []
    for label, cfg_name in combos:
        resolved = None
        try:
            d = yaml_model_load(cfg_name)
            resolved = d.get('yaml_file', None)
        except Exception as exc:
            print(f'[warn] yaml_model_load({cfg_name}) failed: {exc!r}')
        model_obj = YOLO(cfg_name)
        model = model_obj.model.to('cpu').eval()
        params = count_params(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        det = model.model[-1]
        nl = getattr(det, 'nl', None)
        strides = [float(s) for s in getattr(det, 'stride', [])]
        with torch.no_grad():
            pred = model(torch.zeros(1, 3, args.imgsz, args.imgsz))
        structure = describe(pred)
        gflops = None
        try:
            from ultralytics.utils.torch_utils import get_flops
            gflops = round(float(get_flops(model, imgsz=args.imgsz)), 3)
        except Exception as exc:
            gflops = f'UNAVAILABLE({type(exc).__name__})'
        size_mb = round(params * 4 / 1e6, 2)
        rows.append({'variant': label, 'config_requested': cfg_name,
                     'config_resolved': str(Path(resolved).name) if resolved else 'UNKNOWN',
                     'scale': args.scale, 'imgsz': args.imgsz,
                     'detect_levels': nl, 'detect_strides_px': strides,
                     'params_M': round(params / 1e6, 3),
                     'trainable_params_M': round(trainable / 1e6, 3),
                     'fp32_weight_size_MB': size_mb, 'gflops': gflops,
                     'output_structure_at_imgsz': structure})
        print(f"{label:8s} cfg={cfg_name:18s} -> {rows[-1]['config_resolved']:16s} "
              f"params={rows[-1]['params_M']}M levels={nl} strides={strides} "
              f"gflops={gflops} size={size_mb}MB")
        print(f"         output: {structure}")

    csv_path = out / 'model_summary.csv'
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', csv_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
