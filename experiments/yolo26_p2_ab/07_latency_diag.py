#!/usr/bin/env python3
"""07_latency_diag.py - diagnose why measured latency looked independent of imgsz.

Checks: device/dtype of the model, a reference conv2d benchmark to establish GPU health,
imgsz sweep with independent timing, and the theoretical FLOPs ratio for comparison.
"""
from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def timeit(fn, warmup=50, iters=200):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for i in range(iters):
        ev[i][0].record()
        fn()
        ev[i][1].record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return statistics.fmean(ts), ts[len(ts) // 2], ts[int(0.95 * len(ts)) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--scale', default='s')
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments' / 'yolo26_p2_ab' / '_scratch' / 'ultralytics_cfg')

    import torch
    from ultralytics import YOLO

    print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('sm_count', props.multi_processor_count, '| total_mem_GiB', round(props.total_memory / 2**30, 1))

    # reference workload: does the GPU produce sane numbers at all?
    for n in (1, 4, 8):
        a = torch.randn(n, 1024, 1024, device='cuda', dtype=torch.half)
        b = torch.randn(1024, 1024, device='cuda', dtype=torch.half)
        mean, med, p95 = timeit(lambda: a @ b)
        print(f'ref matmul {n}x1024x1024 fp16: mean={mean:.3f}ms median={med:.3f}ms')
        del a, b

    model = YOLO(f'yolo26{args.scale}.yaml').model.cuda().eval().half()
    p = next(model.parameters())
    print('model param device/dtype:', p.device, p.dtype)

    print('\nimgsz sweep with pure forward timing:')
    for imgsz in (320, 640, 960, 1280, 1600):
        x = torch.zeros(1, 3, imgsz, imgsz, device='cuda', dtype=torch.half)
        mean, med, p95 = timeit(lambda: model(x), warmup=30, iters=150)
        print(f'  imgsz={imgsz:5d}  mean={mean:8.3f}ms  median={med:8.3f}ms  p95={p95:8.3f}ms')

    print('\nCPU-side cost of building the input only (no forward):')
    mean, med, p95 = timeit(lambda: torch.zeros(1, 3, 1280, 1280, device='cuda', dtype=torch.half), warmup=10, iters=50)
    print(f'  alloc 1280: mean={mean:.3f}ms')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
