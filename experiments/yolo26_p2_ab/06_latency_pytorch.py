#!/usr/bin/env python3
"""06_latency_pytorch.py - FP16 latency + peak VRAM for both variants at several imgsz.

Protocol: warmup >= 100, measure >= 500, torch.cuda.synchronize() around every timed step,
model loading excluded from the timing, VRAM measured with reset_peak_memory_stats.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
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
    ap.add_argument('--imgsz', type=int, nargs='+', default=[640, 960, 1280])
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--batch', type=int, default=1)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'yolo26_p2_ab_test'
    out.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments' / 'yolo26_p2_ab' / '_scratch' / 'ultralytics_cfg')

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print('BLOCKED: CUDA not available')
        return 2
    print('gpu:', torch.cuda.get_device_name(0))
    free_before, total = torch.cuda.mem_get_info()
    print(f'vram before: free={free_before/2**20:.0f} MiB total={total/2**20:.0f} MiB')

    import subprocess
    def gpu_state():
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,power.draw,utilization.gpu,memory.used',
                                    '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
            return out.stdout.strip()
        except Exception as exc:
            return f'unavailable ({exc!r})'

    gpu_clocks_before = gpu_state()
    print('gpu state before (clocks.sm, power, util, mem):', gpu_clocks_before)

    rows = []
    for label, cfg in (('baseline', f'yolo26{args.scale}.yaml'), ('p2', f'yolo26{args.scale}-p2.yaml')):
        model = YOLO(cfg).model.cuda().eval().half()
        for imgsz in args.imgsz:
            x = torch.zeros(args.batch, 3, imgsz, imgsz, device='cuda', dtype=torch.half)
            with torch.no_grad():
                for _ in range(args.warmup):
                    model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    model(x)
                    t1.record()
                    torch.cuda.synchronize()
                    times.append(t0.elapsed_time(t1))
            peak = torch.cuda.max_memory_allocated() / 2**20
            times.sort()
            mean = statistics.fmean(times)
            median = statistics.median(times)
            p95 = times[int(0.95 * len(times)) - 1]
            p99 = times[int(0.99 * len(times)) - 1]
            rows.append({'variant': label, 'cfg': cfg, 'imgsz': imgsz, 'batch': args.batch,
                         'precision': 'fp16', 'warmup': args.warmup, 'iters': args.iters,
                         'mean_ms': round(mean, 3), 'median_ms': round(median, 3),
                         'p95_ms': round(p95, 3), 'p99_ms': round(p99, 3),
                         'fps': round(1000.0 / mean, 2), 'peak_vram_MiB': round(peak, 1)})
            print(f"{label:8s} imgsz={imgsz:5d} mean={mean:7.3f}ms median={median:7.3f} "
                  f"p95={p95:7.3f} p99={p99:7.3f} fps={1000.0/mean:7.2f} peakVRAM={peak:7.1f}MiB")
        del model
        torch.cuda.empty_cache()

    print('gpu state after :', gpu_state())

    csv_path = out / f'latency_pytorch_b{args.batch}.csv'
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', csv_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
