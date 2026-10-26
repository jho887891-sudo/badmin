import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_eth_four_stage_foregrounds import run_inventory


def build_manifests(tmp_path):
    label=tmp_path/'labels.txt'
    # Two source boxes: one tiny but non-degenerate and one ordinary.
    label.write_text('0 0.5000 0.5000 0.0010 0.0010\n0 0.3000 0.4000 0.0120 0.0200\n',encoding='utf-8')
    columns=['image','label','location','split','set_dir','role','source','is_negative','is_positive',
             'n_boxes','width','height','sha256']
    rows=[
        {'image':'/remote/eth/cab_1_easy/images/train/a.jpg','label':str(label),'location':'cab_1',
         'split':'train','set_dir':'cab_1_easy','role':'official_positives','source':'eth_main',
         'is_negative':'False','is_positive':'True','n_boxes':'2','width':'1920','height':'1200','sha256':'TRAIN-A'},
        {'image':'/remote/eth/cab_1_easy/images/val/b.jpg','label':str(label),'location':'cab_1',
         'split':'val','set_dir':'cab_1_easy','role':'official_positives','source':'eth_main',
         'is_negative':'False','is_positive':'True','n_boxes':'2','width':'1920','height':'1200','sha256':'VAL-B'},
        {'image':'/remote/eth/cab_1_easy/images/train/c.jpg','label':str(label),'location':'cab_1',
         'split':'train','set_dir':'cab_1_easy','role':'official_positives','source':'eth_main',
         'is_negative':'False','is_positive':'True','n_boxes':'2','width':'1920','height':'1200','sha256':'FROZEN-C'},
    ]
    manifest=tmp_path/'train_manifest.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    frozen=tmp_path/'frozen_eval.csv'
    with frozen.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image','sha256']);w.writeheader()
        w.writerow({'image':'/frozen/test/c.jpg','sha256':'FROZEN-C'})
    return manifest,frozen


def test_includes_tiny_nonzero_boxes_without_10_24_px_filter(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    report=run_inventory(manifest,None,[frozen],tmp_path/'out/inventory.csv',
                         tmp_path/'out/audit.json',tmp_path/'out/audit.jsonl')
    assert report['candidate_instances']==2
    sizes=[row['bbox_short'] for row in report and []]
    rows=list(csv.DictReader((tmp_path/'out/inventory.csv').open(encoding='utf-8')))
    assert len(rows)==2
    assert min(int(r['bbox_short']) for r in rows)<=2
    assert all(r['status']=='PENDING_REVIEW' for r in rows)


def test_eval_split_and_frozen_hash_are_recorded_as_rejections(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    report=run_inventory(manifest,None,[frozen],tmp_path/'out/inventory.csv',
                         tmp_path/'out/audit.json',tmp_path/'out/audit.jsonl')
    assert report['rejections_by_reason']['eval_split']==1
    assert report['rejections_by_reason']['eval_content']==1
    records=[json.loads(x) for x in (tmp_path/'out/audit.jsonl').read_text().splitlines()]
    assert sum(r['record_type']=='candidate' for r in records)==2
    assert sum(r['record_type']=='rejection' for r in records)==2


def test_size_distribution_is_descriptive_and_at_1024(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    report=run_inventory(manifest,None,[frozen],tmp_path/'out/inventory.csv',
                         tmp_path/'out/audit.json',tmp_path/'out/audit.jsonl')
    s=report['size_distribution_at_1024']
    assert s['count']==2
    assert 'no size bucket is used to select or discard instances' in s['purpose']
    assert sum(s['count_by_bucket'].values())==2


def test_source_manifest_and_eval_manifest_hashes_are_recorded(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    report=run_inventory(manifest,None,[frozen],tmp_path/'out/inventory.csv',
                         tmp_path/'out/audit.json',tmp_path/'out/audit.jsonl')
    assert len(report['source_manifest_sha256'])==64
    assert len(report['frozen_eval_manifests'][0]['sha256'])==64


def test_existing_outputs_are_never_overwritten(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    out=tmp_path/'out';out.mkdir();existing=out/'inventory.csv';existing.write_text('keep')
    with pytest.raises(FileExistsError):
        run_inventory(manifest,None,[frozen],existing,out/'audit.json',out/'audit.jsonl')
    assert existing.read_text()=='keep'


def test_missing_frozen_eval_index_blocks_inventory(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_inventory(manifest,None,[tmp_path/'missing.csv'],tmp_path/'out/inventory.csv',
                      tmp_path/'out/audit.json',tmp_path/'out/audit.jsonl')


def test_all_candidates_remain_unreviewed_until_human_signoff(tmp_path):
    manifest,frozen=build_manifests(tmp_path)
    report=run_inventory(manifest,None,[frozen],tmp_path/'out/inventory.csv',
                         tmp_path/'out/audit.json',tmp_path/'out/audit.jsonl')
    rows=list(csv.DictReader((tmp_path/'out/inventory.csv').open(encoding='utf-8')))
    assert all(r['reviewed']=='False' and r['reviewer_id']=='' for r in rows)
    assert report['mask_approval']=='PENDING'
    assert report['training_started'] is False
