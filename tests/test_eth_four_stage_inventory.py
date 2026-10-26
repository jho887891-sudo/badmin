import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import pytest
from eth_four_stage_inventory import enumerate_eth_train_instances, InventoryBlocked


def fixture(tmp_path):
    tr=tmp_path/'train';tr.mkdir()
    tiny=tr/'source.jpg';tiny.write_bytes(b'real bytes')
    lab=tr/'source.txt';lab.write_text('0 0.5 0.5 0.025 0.0833\n0 0.5 0.5 0.15 0.5\n',encoding='utf8')
    ev=tr/'eval.jpg';ev.write_bytes(b'eval')
    (tr/'eval.txt').write_text('0 0.5 0.5 0.15 0.5\n',encoding='utf8')
    rows=[{'image':str(tiny),'label':str(lab),'location':'a','split':'train','set_dir':'court_a','source':'eth_main','is_positive':'True','is_negative':'False','n_boxes':'2','width':'200','height':'120','sha256':'SHA1'},
          {'image':str(ev),'label':str(tr/'eval.txt'),'location':'b','split':'val','set_dir':'val','source':'eth_main','is_positive':'True','is_negative':'False','n_boxes':'1','width':'200','height':'120','sha256':'SHA2'}]
    f=tmp_path/'manifest.csv'
    with f.open('w',newline='') as o:
        w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return f


def test_two_boxes_same_frame_have_distinct_ids(tmp_path):
    r=enumerate_eth_train_instances(fixture(tmp_path),label_root=None,frozen_eval_manifests=())
    assert len(r['instances'])==2
    assert len({x['instance_id'] for x in r['instances']})==2


def test_tiny_box_included_not_filtered_by_old_thresholds(tmp_path):
    r=enumerate_eth_train_instances(fixture(tmp_path),label_root=None,frozen_eval_manifests=())
    assert any(x['bbox_short']<10 for x in r['instances'])


def test_eval_only_frame_excluded_with_reason(tmp_path):
    r=enumerate_eth_train_instances(fixture(tmp_path),label_root=None,frozen_eval_manifests=())
    assert r['counts']['excluded_non_train_rows']==1
    assert any(x['reason']=='non_train_split' for x in r['rejections'])


def test_inventory_reconciles_all_train_label_boxes(tmp_path):
    r=enumerate_eth_train_instances(fixture(tmp_path),label_root=None,frozen_eval_manifests=())
    assert r['counts']['train_label_boxes_total']==2
    assert r['counts']['usable_instances']+r['counts']['unusable_instances']==2


def test_zero_area_roundoff_has_explicit_reason(tmp_path):
    f=fixture(tmp_path)
    (tmp_path/'train'/'source.txt').write_text('0 0.5 0.5 0.001 0.002\n',encoding='utf8')
    lines=list(csv.DictReader(f.open()));lines[0]['n_boxes']='1'
    with f.open('w',newline='') as o:
        w=csv.DictWriter(o,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
    r=enumerate_eth_train_instances(f,label_root=None,frozen_eval_manifests=())
    assert r['counts']['unusable_instances']==1
    assert 'quantized_empty_bbox' in [x['reason'] for x in r['rejections']]


def test_missing_label_is_blocker(tmp_path):
    f=fixture(tmp_path)
    (tmp_path/'train'/'source.txt').unlink()
    with pytest.raises(InventoryBlocked):
        enumerate_eth_train_instances(f,label_root=None,frozen_eval_manifests=())


def test_eval_sha_excluded_even_if_split_train(tmp_path):
    f=fixture(tmp_path)
    ev=tmp_path/'frozen.csv'
    ev.write_text('image,sha256\n/frozen/irrelevant.png,SHA1\n',encoding='utf8')
    r=enumerate_eth_train_instances(f,label_root=None,frozen_eval_manifests=(ev,))
    assert len(r['instances'])==0
    assert any(x['reason']=='frozen_eval_match' for x in r['rejections'])
