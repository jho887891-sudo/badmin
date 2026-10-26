import csv,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from eth_four_stage_inventory import enumerate_eth_train_instances, InventoryBlocked


COLUMNS=['image','label','location','split','set_dir','role','source','is_negative','is_positive',
         'n_boxes','width','height','sha256']


def _sample(root,name,boxes,*,split='train',set_dir='court_a',source='eth_main'):
    image=root/'images'/f'{name}.jpg'; image.parent.mkdir(parents=True,exist_ok=True)
    arr=np.zeros((120,200,3),dtype=np.uint8)
    arr[:,:,0]=np.arange(200,dtype=np.uint8)[None,:]
    arr[:,:,1]=np.arange(120,dtype=np.uint8)[:,None]
    arr[::9,:,2]=150
    Image.fromarray(arr).save(image,quality=95)
    label=root/'labels'/'train'/f'{name}.txt';label.parent.mkdir(parents=True,exist_ok=True)
    label.write_text(''.join('0 %.8f %.8f %.8f %.8f\n'%tuple(b) for b in boxes),encoding='utf-8')
    sha=hashlib.sha256(image.read_bytes()).hexdigest()
    return {'image':str(image),'label':str(label),'location':'court_a','split':split,
            'set_dir':set_dir,'role':'official_positives','source':source,'is_negative':'False',
            'is_positive':'True','n_boxes':str(len(boxes)),'width':'200','height':'120','sha256':sha}


def _manifest(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
    return path


def _boxes():
    return [(0.5,0.5,0.025,0.0833333),(0.5,0.5,0.15,0.5)]


def _data(tmp_path):
    row=_sample(tmp_path,'source',_boxes())
    return _manifest(tmp_path/'manifest.csv',[row]),row


def test_two_boxes_same_frame_have_distinct_source_hash_and_box_ids(tmp_path):
    manifest,row=_data(tmp_path)
    r=enumerate_eth_train_instances(manifest)
    assert len(r['instances'])==2
    assert len({x['instance_id'] for x in r['instances']})==2
    assert all(x['instance_id'].startswith(row['sha256']+'#') for x in r['instances'])


def test_tiny_box_included_not_filtered_by_old_thresholds(tmp_path):
    manifest,_=_data(tmp_path)
    r=enumerate_eth_train_instances(manifest)
    assert any(x['bbox_short']<=5 for x in r['instances'])
    assert r['counts']['train_label_boxes_total']==2


def test_eval_only_frame_excluded_with_reason(tmp_path):
    train=_sample(tmp_path,'train',_boxes())
    val=_sample(tmp_path,'val',_boxes(),split='val',set_dir='validation')
    r=enumerate_eth_train_instances(_manifest(tmp_path/'manifest.csv',[train,val]))
    assert r['counts']['excluded_non_train_rows']==1
    assert any(x['reason']=='eval_split' for x in r['rejections'])
    assert all(x['split']=='train' for x in r['instances'])


def test_inventory_reconciles_all_train_label_boxes(tmp_path):
    manifest,_=_data(tmp_path)
    r=enumerate_eth_train_instances(manifest)
    assert r['counts']['train_label_boxes_total']==2
    assert r['counts']['usable_instances']+r['counts']['unusable_instances']==2
    assert r['source_gate']=='PASS'


def test_zero_area_roundoff_has_explicit_reason(tmp_path):
    tiny=_sample(tmp_path,'zero',[(0.5,0.5,0.001,0.002)])
    r=enumerate_eth_train_instances(_manifest(tmp_path/'manifest.csv',[tiny]))
    assert r['counts']['unusable_instances']==1
    assert 'quantized_empty_bbox' in [x['reason'] for x in r['rejections']]
    assert r['counts']['train_label_boxes_total']==1


def test_missing_label_is_blocker(tmp_path):
    manifest,row=_data(tmp_path)
    Path(row['label']).unlink()
    with pytest.raises(InventoryBlocked):
        enumerate_eth_train_instances(manifest)


def test_eval_sha_excluded_even_if_split_train(tmp_path):
    manifest,row=_data(tmp_path)
    frozen=tmp_path/'frozen.csv'
    frozen.write_text('image,sha256\n/frozen/other.jpg,'+row['sha256']+'\n',encoding='utf-8')
    r=enumerate_eth_train_instances(manifest,frozen_eval_manifests=(frozen,))
    assert len(r['instances'])==0
    assert any(x['reason']=='eval_content' for x in r['rejections'])


def test_raw_image_missing_blocks_instead_of_empty_success(tmp_path):
    manifest,row=_data(tmp_path)
    Path(row['image']).unlink()
    with pytest.raises(InventoryBlocked):
        enumerate_eth_train_instances(manifest)


def test_manifest_sha_must_match_actual_image_bytes(tmp_path):
    manifest,row=_data(tmp_path)
    rows=list(csv.DictReader(manifest.open(encoding='utf-8')))
    rows[0]['sha256']='0'*64
    _manifest(manifest,rows)
    with pytest.raises(InventoryBlocked,match='SHA256 mismatch'):
        enumerate_eth_train_instances(manifest)


def test_duplicate_source_row_is_rejected_and_reconciled(tmp_path):
    manifest,row=_data(tmp_path)
    r=enumerate_eth_train_instances(_manifest(tmp_path/'manifest.csv',[row,row]))
    assert len(r['instances'])==2
    assert r['counts']['unusable_instances']==2
    assert sum(x['reason']=='duplicate_source_instance' for x in r['rejections'])==2
    assert r['source_gate']=='PASS'
