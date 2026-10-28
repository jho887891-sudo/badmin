#!/usr/bin/env python3
"""Train YOLO26s detection Stage-1 on *full, scene/time-balanced* ETH BBox-only composites.

Requires exact source hashes, independent internal-validation *source* hashes,
verified model checkpoint bytes, and newly allocated output root.
Does not train until --execute. Does not alter four-stage Mask-based path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class TrainingBlocked(RuntimeError):
    pass


REQUIRED_MODEL_FILENAME = 'yolo26s.pt'


def validate_checkpoint_filename(model: Path) -> None:
    """Fail closed before any GPU use: this experiment starts from YOLO26s."""
    if model.name.lower() != REQUIRED_MODEL_FILENAME:
        raise TrainingBlocked('Stage 1 requires an explicitly verified yolo26s.pt; '
                              'got ' + model.name)


def inspect_loaded_yolo26s(loaded: Any) -> dict:
    """Inspect the actual parsed network, not just the checkpoint filename.

    Ultralytics stores original YAML metadata in loaded.model.yaml.
    A renamed YOLOv8s or YOLO26 segmentation model must not pass.
    """
    network = getattr(loaded, 'model', None)
    model_yaml = getattr(network, 'yaml', None)
    if not isinstance(model_yaml, dict):
        raise TrainingBlocked('YOLO26s architecture YAML metadata unavailable')
    task = str(getattr(loaded, 'task', ''))
    config_name = str(model_yaml.get('yaml_file', '')).lower()
    scale = str(model_yaml.get('scale', '')).lower()
    if task != 'detect' or 'yolo26' not in Path(config_name).stem or scale != 's':
        raise TrainingBlocked('checkpoint is not YOLO26s detection: '
                              f'task={task!r}, yaml_file={config_name!r}, scale={scale!r}')
    return {'task': task, 'yaml_file': config_name, 'scale': scale,
            'architecture_verified': True}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise TrainingBlocked(f'CSV missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        records = list(csv.DictReader(f))
    if not records:
        raise TrainingBlocked(f'CSV empty: {path}')
    return records


def resolve_source(raw: str, root: Path | None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return root / p if root is not None else p


def _get_sha(row: dict) -> str:
    for key in ('source_sha256', 'sha256', 'image_sha256'):
        value = (row.get(key) or '').strip().lower()
        if len(value) == 64 and all(c in '0123456789abcdef' for c in value):
            return value
    raise TrainingBlocked('validation record has no explicit source SHA256')


def _get_path(row: dict, *keys: str) -> str:
    for key in keys:
        if row.get(key):
            return row[key]
    raise TrainingBlocked('missing required path field: '+','.join(keys))


def check_synthetic_dataset(dataset: Path, *, expected_unique: int = 13992,
                            bins: int = 10, verify_output_hashes: bool = True) -> dict:
    audit_path = dataset / 'audit.json'
    if not audit_path.is_file():
        raise TrainingBlocked('synthetic audit.json absent')
    report = json.loads(audit_path.read_text(encoding='utf-8'))
    checks = {
        'status': 'PASS_DATA_GENERATION_NOT_TRAINING',
        'mode': 'balanced-full', 'inventory_instances': expected_unique,
        'unique_instances_covered': expected_unique, 'full_unique_coverage': True,
        'mask_used': False, 'model_trained': False, 'optimizer_steps': 0,
        'scale': 1.0, 'image_size': [1024, 1024],
        'foreground_method': 'bbox_crop_no_mask_no_resize',
        'time_bins_per_sequence': bins,
    }
    for field, required in checks.items():
        if report.get(field) != required:
            raise TrainingBlocked(f'synthetic audit mismatch: {field} expected {required!r}, got {report.get(field)!r}')
    records = read_csv(dataset/'manifest.csv')
    if len(records) != report.get('generated_images') or len(records) < expected_unique:
        raise TrainingBlocked('synthetic images count inconsistent with audit')
    scene_time = Counter()
    seen_names: set[str] = set()
    source_ids: set[str] = set()
    source_hashes: set[str] = set()
    for row in records:
        if row.get('foreground_method') != 'bbox_crop_no_mask' or row.get('scale') != '1.0':
            raise TrainingBlocked('mask or resized synthetic record')
        scene, time_bin = row.get('location', ''), row.get('time_bin', '')
        if scene not in {'cab_1', 'cab_2', 'glc_1', 'ml_4', 'ticino_1', 'ticino_2'}:
            raise TrainingBlocked('unexpected scene '+scene)
        if not time_bin.isdigit() or not 0 <= int(time_bin) < bins:
            raise TrainingBlocked('invalid time bin')
        scene_time[(scene, int(time_bin))] += 1
        iid = row.get('instance_id', '')
        if not iid:
            raise TrainingBlocked('missing source instance')
        source_ids.add(iid)
        source_sha = (row.get('source_sha256') or '').lower()
        if len(source_sha) != 64:
            raise TrainingBlocked('missing synthetic parent SHA')
        source_hashes.add(source_sha)
        image_rel = Path(row.get('image_path', ''))
        label_rel = Path(row.get('label_path', ''))
        if (image_rel.is_absolute() or label_rel.is_absolute() or
                len(image_rel.parts) != 2 or image_rel.parts[0] != 'images' or
                len(label_rel.parts) != 2 or label_rel.parts[0] != 'labels' or
                image_rel.stem != label_rel.stem or image_rel.name in seen_names):
            raise TrainingBlocked('invalid/duplicate YOLO output path')
        seen_names.add(image_rel.name)
        image, label = dataset/image_rel, dataset/label_rel
        if not image.is_file() or not label.is_file():
            raise TrainingBlocked('generated image/label missing: '+str(image))
        if verify_output_hashes and (sha256(image) != row.get('image_sha256') or
                                     sha256(label) != row.get('output_label_sha256')):
            raise TrainingBlocked('generated image/label SHA mismatch: '+str(image))
    if len(source_ids) != expected_unique or len(scene_time) != 6*bins or len(set(scene_time.values())) != 1:
        raise TrainingBlocked('full source coverage or equal scene/time exposure not satisfied')
    if report.get('by_scene_time_bin') != {f'{s}|T{t}': n for (s,t),n in sorted(scene_time.items())}:
        raise TrainingBlocked('scene-time report does not match manifest')
    if sha256(dataset/'manifest.csv') != report.get('manifest_sha256'):
        raise TrainingBlocked('manifest hash mismatch')
    return {'source_ids': source_ids, 'source_hashes': source_hashes,
            'images': len(records), 'scene_time_exposures': next(iter(scene_time.values())),
            'duplicates': len(records) - len(source_ids), 'audit_sha256': sha256(audit_path),
            'manifest_sha256': sha256(dataset/'manifest.csv')}


def check_real_val(val_csv: Path, *, val_root: Path | None,
                   forbidden_hashes: set[str], min_images: int = 1) -> list[dict]:
    rows = read_csv(val_csv)
    if len(rows) < min_images:
        raise TrainingBlocked('not enough internal validation images')
    checked = []
    unique_sha = set()
    for row in rows:
        imgraw = _get_path(row, 'image', 'abs_path', 'image_path', 'path')
        labelraw = _get_path(row, 'label', 'abs_label_path', 'label_path')
        img = resolve_source(imgraw, val_root)
        label = resolve_source(labelraw, val_root)
        source_sha = _get_sha(row)
        if source_sha in forbidden_hashes:
            raise TrainingBlocked('train/validation SHA overlap: '+source_sha)
        if source_sha in unique_sha:
            raise TrainingBlocked('duplicate validation source SHA: '+source_sha)
        unique_sha.add(source_sha)
        if not img.is_file() or sha256(img).lower() != source_sha:
            raise TrainingBlocked('validation source missing/SHA mismatch: '+str(img))
        if not label.is_file():
            raise TrainingBlocked('validation label missing: '+str(label))
        if row.get('label_sha256') and sha256(label).lower() != row['label_sha256'].lower():
            raise TrainingBlocked('validation label SHA mismatch: '+str(label))
        lines = label.read_text(encoding='utf-8').strip().splitlines()
        if not lines:
            raise TrainingBlocked('empty validation label for positive-only internal val: '+str(label))
        for line in lines:
            parts = line.split()
            if len(parts) != 5 or parts[0] != '0':
                raise TrainingBlocked('invalid single-class YOLO label: '+str(label))
            try:
                cx, cy, w, h = map(float, parts[1:])
            except ValueError as exc:
                raise TrainingBlocked('invalid numerical validation label: '+str(label)) from exc
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise TrainingBlocked('validation YOLO label outside [0,1]: '+str(label))
        checked.append({'image': img, 'label': label, 'sha256': source_sha})
    return checked


def preflight(dataset: Path, val_csv: Path, model: Path, model_sha256: str,
              run_root: Path, *, val_root: Path | None = None, bins: int = 10,
              expected_unique: int = 13992, verify_output_hashes: bool = True) -> tuple[dict, list[dict]]:
    if run_root.exists():
        raise TrainingBlocked('run root already exists: '+str(run_root))
    validate_checkpoint_filename(model)
    data = check_synthetic_dataset(dataset, bins=bins, expected_unique=expected_unique,
                                   verify_output_hashes=verify_output_hashes)
    real_val = check_real_val(val_csv, val_root=val_root,
                              forbidden_hashes=data['source_hashes'])
    target_sha = model_sha256.lower()
    if len(target_sha) != 64 or not model.is_file() or sha256(model) != target_sha:
        raise TrainingBlocked('checkpoint path or SHA256 mismatch')
    result = {'status': 'PASS', 'model': str(model), 'model_sha256': target_sha,
              'model_family_requested': 'YOLO26s', 'model_filename_checked': True,
              'architecture_loaded_and_checked': False,
              'stage': 'stage1_bbox_only', 'imgsz': 1024,
              'train_images': data['images'], 'train_unique_instances': len(data['source_ids']),
              'train_duplicate_exposures': data['duplicates'], 'per_scene_time_exposures': data['scene_time_exposures'],
              'internal_real_val_images': len(real_val),
              'internal_val_only_not_independent_final_test': True,
              'train_audit_sha256': data['audit_sha256'], 'train_manifest_sha256': data['manifest_sha256'],
              'source_val_sha_overlap': 0, 'model_trained': False, 'optimizer_steps': 0,
              'time_bins_proxy_not_physical_time': True,
              'visual_quality_review': 'UNKNOWN_UNLESS_SEPARATELY_SIGNED_OFF',
              'checkpoint_selection_scope': 'internal_real_val_only',
              'optimizer': 'AdamW', 'lr0': 1e-4,
              'warmup_epochs': 3.0, 'warmup_bias_lr': 0.0,
              'warmup_bias_lr_contract': 'bias_lr_starts_at_zero_then_warms_up_to_scheduler_target',
              'dataset_root': str(dataset), 'val_manifest': str(val_csv), 'run_root': str(run_root)}
    return result, real_val


def materialize_dataset_config(dataset: Path, val: list[dict], run_root: Path) -> Path:
    import yaml
    if run_root.exists():
        raise TrainingBlocked('run root exists; refuse overwrite')
    root = run_root/'validation'
    imgdir, labeldir = root/'images', root/'labels'
    imgdir.mkdir(parents=True)
    labeldir.mkdir()
    for i, rec in enumerate(val):
        stem = f'{i:06d}'
        (imgdir/(stem+rec['image'].suffix.lower())).symlink_to(rec['image'].resolve())
        (labeldir/(stem+'.txt')).symlink_to(rec['label'].resolve())
    yaml_path = run_root/'dataset.yaml'
    yaml_path.write_text(yaml.safe_dump({
        'path': str(run_root.resolve()),
        'train': str((dataset/'images').resolve()),
        'val': str(imgdir.resolve()),
        'names': {0: 'shuttlecock'}, 'nc': 1,
    }, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return yaml_path


def build_stage1_train_kwargs(yaml_path: Path, run_root: Path, *, epochs: int, batch: int,
                              seed: int, device: str, workers: int, lr0: float) -> dict:
    """Explicitly control Ultralytics warmup for all parameter groups.

    With optimizer=AdamW, Ultralytics normally uses warmup_bias_lr=0.1.
    For lr0=1e-4 this can yield bias-group rates 300-700x higher in
    epochs 1-2. Setting it to 0 makes bias warm up from zero like other
    groups; this is a deliberate change from the previous 9-epoch run.
    """
    if epochs < 1 or batch < 1 or workers < 0 or lr0 <= 0:
        raise TrainingBlocked('invalid training hyperparameters')
    return dict(data=str(yaml_path), imgsz=1024, epochs=epochs, batch=batch,
                optimizer='AdamW', lr0=lr0, seed=seed, deterministic=True,
                # Explicit warmup settings: no hidden 0.1 bias LR.
                warmup_epochs=3.0, warmup_bias_lr=0.0,
                device=device, workers=workers, project=str(run_root), name='training',
                exist_ok=False, val=True, save=True, save_period=1,
                patience=epochs, cache=False, multi_scale=False,
                mosaic=0.0, mixup=0.0, copy_paste=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                perspective=0.0, flipud=0.0, fliplr=0.0,
                close_mosaic=0)


def launch_train(trainer: Any, yaml_path: Path, run_root: Path, *, epochs: int, batch: int,
                 seed: int, device: str, workers: int, lr0: float) -> None:
    kwargs = build_stage1_train_kwargs(yaml_path, run_root, epochs=epochs, batch=batch,
                                       seed=seed, device=device, workers=workers, lr0=lr0)
    import torch
    if not torch.cuda.is_available():
        raise TrainingBlocked('CUDA not available; refuse GPU training')
    inspect_loaded_yolo26s(trainer)
    trainer.train(**kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--val-root', type=Path, default=None)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-sha256', required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--expected-unique', type=int, default=13992)
    p.add_argument('--time-bins', type=int, default=10)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr0', type=float, default=1e-4)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    try:
        summary, valid = preflight(args.dataset_root, args.val_manifest,
                                   args.model, args.model_sha256, args.run_root,
                                   val_root=args.val_root, bins=args.time_bins,
                                   expected_unique=args.expected_unique)
        if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.lr0 <= 0:
            raise TrainingBlocked('invalid hyperparameters')
        summary['requested_epochs'] = args.epochs
        summary['batch'] = args.batch
        summary['lr0'] = args.lr0
        summary['seed'] = args.seed
        summary['execute_requested'] = args.execute
        if not args.execute:
            summary['status'] = 'DRY_RUN_PASS_NO_TRAINING'
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        import torch
        import ultralytics
        if not torch.cuda.is_available():
            raise TrainingBlocked('CUDA unavailable on training host')
        if args.device != '0':
            raise TrainingBlocked('this stage1 training contract expects device 0 only')
        summary['cuda_name'] = torch.cuda.get_device_name(0)
        try:
            from ultralytics import YOLO
            import ultralytics
            trainer = YOLO(str(args.model))
            summary['loaded_architecture'] = inspect_loaded_yolo26s(trainer)
            summary['ultralytics_version'] = str(ultralytics.__version__)
            summary['architecture_loaded_and_checked'] = True
        except TrainingBlocked:
            raise
        except Exception as exc:
            raise TrainingBlocked('YOLO26s load/Ultralytics compatibility failed: '+repr(exc)) from exc
        yaml_path = materialize_dataset_config(args.dataset_root, valid, args.run_root)
        (args.run_root/'preflight.json').write_text(json.dumps(summary, ensure_ascii=False,
                                                              indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'status': 'TRAINING_LAUNCH', 'preflight': str(args.run_root/'preflight.json'),
                          'data_yaml': str(yaml_path)}, ensure_ascii=False), flush=True)
        launch_train(trainer, yaml_path, args.run_root, epochs=args.epochs,
                     batch=args.batch, seed=args.seed, device=args.device,
                     workers=args.workers, lr0=args.lr0)
        return 0
    except (TrainingBlocked, OSError, ValueError, ImportError) as exc:
        print(json.dumps({'status':'BLOCKED', 'reason':str(exc)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
