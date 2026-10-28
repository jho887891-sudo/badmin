"""CPU-only checks: source-frame holdout and synthetic YOLO26s validation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_eth_stage1_synthetic_holdout import build,source_split,HoldoutBlocked
from generate_eth_stage1_bbox_preview import LOCATIONS,load_inventory,sha256,PALETTE
from train_eth_stage1_synthetic_holdout import (
    check_synthetic_validation,preflight,make_dataset_yaml,TrainingBlocked,
)

def fixture_inventory(root:Path, frames:int=32)->Path:
    rows=[]
    for scene_idx,scene in enumerate(LOCATIONS):
        for frame in range(frames):
            image=root/f'{scene}_{frame:05d}.png'
            label=root/f'{scene}_{frame:05d}.txt'
            img=Image.new('RGB',(32,32),(scene_idx+15,frame+10,80))
            img.putpixel((10,10),(255,255,255))
            img.save(image)
            label.write_text('0 0.375 0.375 0.25 0.25\n',encoding='utf-8')
            rows.append(dict(instance_id=f'{scene}-{frame}',image=str(image),
                label=str(label),source_sha256=sha256(image),label_sha256=sha256(label),
                split='train',source='eth_main',role='eth_train',
                location=scene,set_dir=scene+'/images/train',
                bbox_xyxy=json.dumps([8,8,16,16])))
    path=root/'inventory.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return path

def test_temporal_holdout_has_no_source_overlap_and_is_reproducible(tmp_path):
    inv=fixture_inventory(tmp_path)
    rows=load_inventory(inv,expected_count=192,bins=10)
    first,audit=source_split(rows,train_fraction=.8,embargo_frames=2)
    second,_=source_split(rows,train_fraction=.8,embargo_frames=2)
    assert [r['instance_id'] for r in first['train']]==[r['instance_id'] for r in second['train']]
    assert sum(len(v) for v in first.values())==192
    assert audit['val_unique_instances']>0
    assert audit['embargoed_instances']>0
    assert all(x['first_val_frame']-x['last_train_frame']>2 for x in audit['sequences'])

def test_cross_split_duplicate_source_sha_is_blocked(tmp_path):
    rows=load_inventory(fixture_inventory(tmp_path),expected_count=192,bins=10)
    rows[-1]['source_sha256']=rows[0]['source_sha256']
    with pytest.raises(HoldoutBlocked,match='SHA leaks'):
        source_split(rows,embargo_frames=2)

def test_excessive_temporal_embargo_blocks(tmp_path):
    rows=load_inventory(fixture_inventory(tmp_path,frames=3),expected_count=18,bins=1)
    with pytest.raises(HoldoutBlocked,match='embargo exhausted'):
        source_split(rows,embargo_frames=20)

def test_invalid_train_fraction_blocks():
    with pytest.raises(HoldoutBlocked,match='train-fraction'):
        source_split([],train_fraction=1.0)

@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    root=tmp_path_factory.mktemp('stage1_synthetic_holdout')
    inv=fixture_inventory(root)
    split=root/'split'
    result=build(inv,split,expected_instances=192,seed=20261027,time_bins=10,
                 embargo_frames=2)
    model=root/'yolo26s.pt'
    model.write_bytes(b'cpu-fixture-only-not-real-yolo26')
    return split,result,inv,model

def test_both_datasets_generated_after_source_split(generated):
    root,audit,inv,model=generated
    assert audit['original_instances']==192
    assert sum(audit[k] for k in ('train_unique_instances','val_unique_instances',
                                  'embargoed_instances'))==192
    assert audit['train_generated_images']>=audit['train_unique_instances']
    assert audit['val_generated_images']==audit['val_unique_instances']
    assert audit['train_val_source_sha_overlap']==0
    assert (root/'split_audit.json').is_file()
    assert (root/'train_synthetic'/'contact_sheet.png').is_file()
    assert (root/'val_synthetic'/'contact_sheet.png').is_file()
    val=next(csv.DictReader((root/'val_synthetic'/'manifest.csv').open()))
    assert tuple(json.loads(val['background_rgb'])) in PALETTE
    with Image.open(val['source_image']) as source, Image.open(root/'val_synthetic'/val['image_path']) as result:
        assert source.crop(json.loads(val['source_bbox_xyxy'])).tobytes() == \
               result.crop(json.loads(val['output_bbox_xyxy'])).tobytes()

def test_synthetic_val_label_and_sha_gate(generated):
    root,audit,inv,model=generated
    checked=check_synthetic_validation(root/'val_synthetic',
        expected_unique=audit['val_unique_instances'],forbidden_hashes=set())
    assert checked['images']==audit['val_unique_instances']

def test_synthetic_val_rejects_train_source_overlap(generated):
    root,audit,inv,model=generated
    source=next(csv.DictReader((root/'val_inventory.csv').open()))['source_sha256']
    with pytest.raises(TrainingBlocked,match='overlap'):
        check_synthetic_validation(root/'val_synthetic',
            expected_unique=audit['val_unique_instances'],forbidden_hashes={source})

def test_preflight_reads_both_synthetic_datasets_without_gpu(generated):
    root,audit,inv,model=generated
    report=preflight(root,model,sha256(model),root.parent/'fresh_training',
                     expected_original=192,time_bins=10)
    assert report['source_sha_overlap']==0
    assert report['val_domain']=='synthetic_solid'
    assert report['train_unique_instances']==audit['train_unique_instances']
    assert report['val_unique_instances']==audit['val_unique_instances']
    assert not (root.parent/'fresh_training').exists()

def test_yaml_points_to_synthetic_val_not_real_images(generated):
    root,audit,inv,model=generated
    output=root.parent/'yaml_output'
    yaml=make_dataset_yaml(root,output)
    import yaml as y
    data=y.safe_load(yaml.read_text())
    assert data['val']==str((root/'val_synthetic'/'images').resolve())
    assert data['train']==str((root/'train_synthetic'/'images').resolve())

def test_reject_existing_output_and_check_model_sha(generated):
    root,audit,inv,model=generated
    with pytest.raises(HoldoutBlocked,match='already exists'):
        build(inv,root,expected_instances=192,embargo_frames=2)
    with pytest.raises(TrainingBlocked,match='SHA mismatch'):
        preflight(root,model,'0'*64,root.parent/'new_run',expected_original=192)
