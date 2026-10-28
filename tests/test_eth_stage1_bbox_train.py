"""CPU-only tests for Stage1 BBox-only YOLO training gates; never launch GPU."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from generate_eth_stage1_bbox_preview import LOCATIONS, run
from train_eth_stage1_bbox import (
    TrainingBlocked, check_synthetic_dataset, check_real_val,
    materialize_dataset_config, preflight, sha256, inspect_loaded_yolo26s,
    build_stage1_train_kwargs,
)


def make_input(root: Path):
    original = []
    for scene in LOCATIONS:
        for i in range(4):
            im = Image.new('RGB', (32, 32), (i + 15, 60, 115))
            im.putpixel((0, 0), (LOCATIONS.index(scene), i, 0))
            image = root/f'{scene}_{i:05d}.png'
            label = root/f'{scene}_{i:05d}.txt'
            im.save(image)
            label.write_text('0 0.5 0.5 0.25 0.25\n', encoding='utf-8')
            original.append({'instance_id': hashlib.sha256(image.read_bytes()).hexdigest()+f'#{i}',
                             'image': str(image), 'label': str(label),
                             'source_sha256': sha256(image), 'label_sha256': sha256(label),
                             'split': 'train', 'source': 'eth_main', 'role': 'eth_train',
                             'location': scene, 'set_dir': scene+'/images/train',
                             'bbox_xyxy': json.dumps([10, 10, 18, 18])})
    path = root/'inventory.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(original[0])); writer.writeheader(); writer.writerows(original)
    return path


def make_val(root: Path, n: int = 2, *, overlap: Path | None = None):
    rows = []
    for i in range(n):
        image = root / f'validation_{i:05d}.png'
        label = root / f'validation_{i:05d}.txt'
        Image.new('RGB', (24, 24), (10+i, 10+i, 25+i)).save(image)
        label.write_text('0 0.500 0.500 0.250 0.250\n', encoding='utf-8')
        if overlap is not None and i == 0: image = overlap
        rows.append({'image': str(image), 'label': str(label),
                     'source_sha256': sha256(image), 'label_sha256': sha256(label)})
    path = root/'val.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return path


def prepared(tmp_path: Path):
    inv = make_input(tmp_path)
    root = tmp_path/'synthetic'
    report = run(inv, root, mode='balanced-full', expected_count=0, bins=2,
                 size=(1024,1024), seed=1)
    model = tmp_path/'yolo26s.pt'
    model.write_bytes(b'model-fixture-only')
    return root, model, inv, report


def test_training_dataset_full_balanced_pass(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    got = check_synthetic_dataset(root, expected_unique=24, bins=2)
    assert got['images'] >= 24
    assert got['scene_time_exposures'] > 0
    assert got['duplicates'] == got['images']-24


def test_training_gate_blocks_unbalanced_full(tmp_path):
    inv = make_input(tmp_path)
    root = tmp_path/'full'
    run(inv, root, mode='full', expected_count=0, bins=2, size=(1024,1024), seed=1)
    with pytest.raises(TrainingBlocked, match='mode'):
        check_synthetic_dataset(root, expected_unique=24, bins=2)


def test_training_gate_blocks_corrupted_image(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    next((root/'images').glob('*.png')).write_bytes(b'tampered')
    with pytest.raises(TrainingBlocked, match='SHA mismatch'):
        check_synthetic_dataset(root, expected_unique=24, bins=2)


def test_training_gate_blocks_val_leak(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    rows = list(csv.DictReader(inv.open()))
    val = make_val(tmp_path, overlap=Path(rows[0]['image']))
    data = check_synthetic_dataset(root, expected_unique=24, bins=2)
    with pytest.raises(TrainingBlocked, match='overlap'):
        check_real_val(val, val_root=None, forbidden_hashes=data['source_hashes'])


def test_training_preflight_passes_with_verified_identity(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    val = make_val(tmp_path)
    result, checked = preflight(root, val, model, sha256(model), tmp_path/'new_run',
                                expected_unique=24, bins=2)
    assert result['status'] == 'PASS'
    assert result['source_val_sha_overlap'] == 0
    assert result['train_unique_instances'] == 24
    assert len(checked) == 2
    assert not (tmp_path/'new_run').exists()


def test_training_gate_blocks_bad_model_sha(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    val = make_val(tmp_path)
    with pytest.raises(TrainingBlocked, match='checkpoint'):
        preflight(root, val, model, '0'*64, tmp_path/'new_run',
                  expected_unique=24, bins=2)


def test_training_gate_blocks_overwrite(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    val = make_val(tmp_path)
    (tmp_path/'new_run').mkdir()
    with pytest.raises(TrainingBlocked, match='already exists'):
        preflight(root, val, model, sha256(model), tmp_path/'new_run',
                  expected_unique=24, bins=2)


def test_dataset_yaml_has_real_val_symlinks(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    val = make_val(tmp_path)
    _, checked = preflight(root, val, model, sha256(model), tmp_path/'new_run',
                           expected_unique=24, bins=2)
    config = materialize_dataset_config(root, checked, tmp_path/'new_run')
    assert config.is_file()
    assert (tmp_path/'new_run/validation/images/000000.png').is_symlink()
    assert (tmp_path/'new_run/validation/labels/000000.txt').is_symlink()
    with pytest.raises(TrainingBlocked, match='exists'):
        materialize_dataset_config(root, checked, tmp_path/'new_run')


def test_yolov8s_checkpoint_filename_is_rejected(tmp_path):
    root, model, inv, report = prepared(tmp_path)
    val = make_val(tmp_path)
    old_model = tmp_path/'yolov8s.pt'
    old_model.write_bytes(model.read_bytes())
    with pytest.raises(TrainingBlocked, match='yolo26s'):
        preflight(root, val, old_model, sha256(old_model), tmp_path/'new_run',
                  expected_unique=24, bins=2)


def test_yolo26s_model_identity_checks_loaded_architecture():
    def stub(yaml, task='detect'):
        return SimpleNamespace(task=task, model=SimpleNamespace(yaml=yaml))
    valid = stub({'yaml_file':'yolo26.yaml', 'scale':'s'})
    assert inspect_loaded_yolo26s(valid)['architecture_verified']
    for invalid in (
        stub({'yaml_file':'yolov8.yaml', 'scale':'s'}),
        stub({'yaml_file':'yolo26.yaml', 'scale':'n'}),
        stub({'yaml_file':'yolo26-seg.yaml', 'scale':'s'}, task='segment'),
        stub({}, task='detect'),
    ):
        with pytest.raises(TrainingBlocked, match='YOLO26s'):
            inspect_loaded_yolo26s(invalid)


def test_explicit_warmup_prevents_bias_learning_rate_spike(tmp_path):
    kwargs = build_stage1_train_kwargs(tmp_path/'dataset.yaml', tmp_path/'run',
                                       epochs=30, batch=8, seed=20261027,
                                       device='0', workers=4, lr0=1e-4)
    assert kwargs['optimizer'] == 'AdamW'
    assert kwargs['lr0'] == pytest.approx(1e-4)
    assert kwargs['warmup_epochs'] == pytest.approx(3.0)
    assert kwargs['warmup_bias_lr'] == pytest.approx(0.0)
    assert kwargs['imgsz'] == 1024
    assert kwargs['scale'] == 0.0
    assert kwargs['mosaic'] == 0.0


def test_warmup_settings_apply_to_smoke_and_formal_runs(tmp_path):
    for epochs, batch in ((1, 4), (30, 8)):
        kwargs = build_stage1_train_kwargs(tmp_path/'data.yaml', tmp_path/'run',
                                           epochs=epochs, batch=batch, seed=1,
                                           device='0', workers=0, lr0=1e-4)
        assert kwargs['warmup_bias_lr'] == 0.0
        assert kwargs['warmup_epochs'] == 3.0
        assert kwargs['epochs'] == epochs and kwargs['batch'] == batch


def test_training_hyperparameters_reject_nonpositive_lr(tmp_path):
    with pytest.raises(TrainingBlocked, match='invalid training hyperparameters'):
        build_stage1_train_kwargs(tmp_path/'data.yaml', tmp_path/'run',
                                  epochs=30, batch=8, seed=1,
                                  device='0', workers=4, lr0=0.0)
