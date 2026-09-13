#!/usr/bin/env python3
"""08_write_report.py - assemble experiment_config.yaml, BLOCKED records, evidence plots and REPORT.md.

Only writes what the measurements support; every unavailable experiment is recorded as BLOCKED
with its concrete reason (never fabricated).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def blocked(out: Path, name: str, reason: str, needs: str) -> None:
    p = out / name
    with p.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['status', 'reason', 'what_unblocks_it'])
        w.writerow(['BLOCKED', reason, needs])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = repo / 'outputs' / 'yolo26_p2_ab_test'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'evidence').mkdir(exist_ok=True)

    summary = read_csv(out / 'model_summary.csv')
    lat1 = read_csv(out / 'latency_pytorch_b1.csv')
    lat8 = read_csv(out / 'latency_pytorch_b8.csv')
    ds = json.loads((out / 'dataset_audit.json').read_text()) if (out / 'dataset_audit.json').exists() else {}

    cfg = {
        'experiment': 'yolo26s vs yolo26s-p2 ablation for the 01 Perception coarse shuttle detector',
        'generated_utc': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'variants': {
            'A_baseline': 'yolo26s.yaml  (official, Detect P3/P4/P5)',
            'B_p2': 'yolo26s-p2.yaml  (official, Detect P2/P3/P4/P5)',
        },
        'scale': 's',
        'weights_source': 'assets/external/_staging/F_yolo/weights/yolo26s.pt (COCO, AGPL-3.0)',
        'runtime': {'ultralytics': '8.4.150 (used via PYTHONPATH from the extracted wheel; NOT installed)',
                    'torch': '2.10.0+cu128', 'cuda_build': '12.8', 'precision': 'fp16',
                    'device': 'NVIDIA RTX A6000 (driver 550.163.01)'},
        'imgsz_tested': [640, 960, 1280],
        'latency_protocol': {'warmup': 100, 'iters': 300, 'batch_tested': [1, 8],
                             'timing': 'cuda events + torch.cuda.synchronize() around every step',
                             'load_time_excluded': True},
        'seed': 0,
        'no_env_modification': True,
    }
    (out / 'experiment_config.yaml').write_text(
        '\n'.join(f'{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}' for k, v in cfg.items()) + '\n')

    blocked(out, 'accuracy_overall.csv', 'DATASET_NOT_AVAILABLE',
            'same shuttlecock dataset + both models trained under the identical protocol (3 seeds)')
    blocked(out, 'accuracy_by_size.csv', 'DATASET_NOT_AVAILABLE', 'labelled tiny-target dataset with size buckets')
    blocked(out, 'threshold_sweep.csv', 'DATASET_NOT_AVAILABLE', 'trained models + validation split for PR curves')
    blocked(out, 'hard_negative.csv', 'DATASET_NOT_AVAILABLE', 'shuttle-free frames with nets/lines/lights/shoes/reflections')
    blocked(out, 'video_continuity.csv', 'DATASET_NOT_AVAILABLE', 'real shuttle video with per-frame GT')
    blocked(out, 'occlusion_reacquisition.csv', 'DATASET_NOT_AVAILABLE', 'sequences with visible->occluded->visible structure')
    blocked(out, 'roi_coverage.csv', 'DATASET_NOT_AVAILABLE', 'GT boxes to test ROI expansion 1.25x/1.5x/2.0x')
    blocked(out, 'stereo_usability.csv', 'DATASET_NOT_AVAILABLE', 'synchronised left/right frames with GT')
    blocked(out, 'downstream_measurement.csv', 'DOWNSTREAM_NOT_AVAILABLE',
            'images + calibrated rig feeding the (implemented) subpixel localizer, triangulation and UKF')
    blocked(out, 'annotation_stats.csv', 'DATASET_NOT_AVAILABLE', 'labelled bboxes to compute size percentiles')
    blocked(out, 'latency_end_to_end.csv', 'DOWNSTREAM_NOT_AVAILABLE', 'real frames through decode/letterbox/H2D/ROI/localizer/stereo')
    blocked(out, 'latency_tensorrt.csv', 'TENSORRT_BLOCKED',
            'TensorRT+onnxruntime installed in a separate env (must not be force-installed into the shared Isaac env)')
    blocked(out, 'statistical_analysis.csv', 'DATASET_NOT_AVAILABLE', 'per-seed accuracy metrics to bootstrap')

    (out / 'dataset_audit.md').write_text(
        '# Dataset audit\n\n' +
        f"verdict: **{ds.get('verdict', 'UNKNOWN')}**\n\n" +
        f"- scanned files: {ds.get('scanned_files', '?')}\n" +
        f"- dataset YAMLs found: {ds.get('dataset_yaml_count', '?')}\n" +
        f"- images found: {ds.get('image_count', '?')} (all under outputs/evidence, i.e. renders)\n" +
        f"- YOLO label files found: {ds.get('yolo_label_files', '?')}\n\n" +
        'Consequence: no shuttlecock dataset exists in this repository, therefore every '
        'accuracy / tiny-object / hard-negative / video / stereo experiment is BLOCKED and '
        '**no recall or mAP number may be reported**.\n')  # noqa: E501

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figs = []
        for label, rows, fname in (('batch=1', lat1, 'latency_vs_imgsz_b1.png'),
                                   ('batch=8', lat8, 'latency_vs_imgsz_b8.png')):
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for variant in ('baseline', 'p2'):
                sel = [r for r in rows if r['variant'] == variant]
                xs = [int(r['imgsz']) for r in sel]
                ys = [float(r['mean_ms']) for r in sel]
                ax.plot(xs, ys, marker='o', label=variant)
            ax.set_title(f'FP16 forward latency vs imgsz ({label})')
            ax.set_xlabel('imgsz'); ax.set_ylabel('mean latency [ms]'); ax.grid(True); ax.legend()
            fig.tight_layout(); fig.savefig(out / 'evidence' / fname, dpi=140); plt.close(fig)
            figs.append(fname)
        if summary:
            fig, ax = plt.subplots(figsize=(6, 4))
            names = [f"{r['variant']}\n{r['config_resolved']}" for r in summary]
            ax.bar(names, [float(r['gflops']) for r in summary], color=['#4477aa', '#cc6677'])
            ax.set_ylabel('GFLOPs @640'); ax.set_title('Structural cost of the P2 branch')
            fig.tight_layout(); fig.savefig(out / 'evidence' / 'gflops_compare.png', dpi=140); plt.close(fig)
            figs.append('gflops_compare.png')
        print('evidence figures:', figs)
    except Exception as exc:
        print('evidence plots unavailable:', repr(exc))

    print('wrote experiment_config.yaml, dataset_audit.md, BLOCKED records')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
