import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_eth_mask_candidates import run_mask_candidates,MaskRunBlocked


FIELDS=['instance_id','image','label','source_sha256','label_sha256','split','source','role',
        'bbox_xyxy','bbox_short','bbox_long']


def make_input(root,*,bbox=(10,10,22,26),bad_sha=False):
    img=root/'images'/'a.png';img.parent.mkdir(parents=True,exist_ok=True)
    a=np.full((40,40,3),25,dtype=np.uint8);a[10:26,10:22]=220
    Image.fromarray(a).save(img)
    lab=root/'labels'/'a.txt';lab.parent.mkdir(parents=True,exist_ok=True);lab.write_text('0 0.4 0.45 0.3 0.4\n')
    ih=hashlib.sha256(img.read_bytes()).hexdigest()
    lh=hashlib.sha256(lab.read_bytes()).hexdigest()
    row={'instance_id':ih+'#0','image':str(img),'label':str(lab),
         'source_sha256':('0'*64 if bad_sha else ih),'label_sha256':lh,
         'split':'train','source':'eth_main','role':'eth_train',
         'bbox_xyxy':json.dumps(list(bbox)),'bbox_short':min(bbox[2]-bbox[0],bbox[3]-bbox[1]),
         'bbox_long':max(bbox[2]-bbox[0],bbox[3]-bbox[1])}
    inv=root/'inventory.csv'
    with inv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerow(row)
    return inv,row


def test_every_row_retained_and_never_approved(tmp_path):
    inv,row=make_input(tmp_path)
    out=run_mask_candidates(inv,tmp_path/'out',sheet_size=1)
    assert out['input_rows']==1
    assert out['approved_masks']==0 and out['training_permitted'] is False
    review=list(csv.DictReader((tmp_path/'out'/'review.csv').open()))
    assert len(review)==1 and review[0]['reviewed']==''
    assert review[0]['decision']==''
    assert (tmp_path/'out'/'review_sheets'/'review_0001.jpg').is_file()


def test_source_sha_mismatch_blocks_run(tmp_path):
    inv,row=make_input(tmp_path,bad_sha=True)
    out=run_mask_candidates(inv,tmp_path/'out')
    assert out['status']=='BLOCKED_SOURCE_INTEGRITY'
    assert out['source_integrity_failures']==1
    assert out['approved_masks']==0


def test_missing_label_is_recorded_as_blocker(tmp_path):
    inv,row=make_input(tmp_path)
    Path(row['label']).unlink()
    out=run_mask_candidates(inv,tmp_path/'out')
    assert out['status']=='BLOCKED_SOURCE_INTEGRITY'
    assert out['source_integrity_failures']==1


def test_tiny_target_is_not_silently_dropped(tmp_path):
    inv,row=make_input(tmp_path,bbox=(10,10,16,16))
    out=run_mask_candidates(inv,tmp_path/'out')
    rec=json.loads((tmp_path/'out'/'candidate_manifest.jsonl').read_text().splitlines()[0])
    assert out['input_rows']==1
    assert rec['instance_id']==row['instance_id']
    assert rec['status'] in {'PENDING_REVIEW','UNRECOVERABLE'}
    assert out['approved_masks']==0


def test_duplicate_inventory_id_never_silently_overwrites(tmp_path):
    inv,row=make_input(tmp_path)
    with inv.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writerow(row)
    out=run_mask_candidates(inv,tmp_path/'out')
    assert out['status']=='BLOCKED_SOURCE_INTEGRITY'
    assert out['input_rows']==2
    assert out['source_integrity_failures']==1


def test_existing_output_directory_refused(tmp_path):
    inv,row=make_input(tmp_path);dest=tmp_path/'out';dest.mkdir()
    with pytest.raises(MaskRunBlocked,match='already_exists'):
        run_mask_candidates(inv,dest)


def test_inventory_source_role_and_split_are_required(tmp_path):
    inv,row=make_input(tmp_path)
    rows=list(csv.DictReader(inv.open()))
    rows[0]['split']='val'
    with inv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    out=run_mask_candidates(inv,tmp_path/'out')
    assert out['status']=='BLOCKED_SOURCE_INTEGRITY'
    assert out['source_integrity_failures']==1


def test_missing_grabcut_backend_is_a_top_level_blocker(tmp_path,monkeypatch):
    import run_eth_mask_candidates as runner
    inv,row=make_input(tmp_path)
    monkeypatch.setattr(runner,'propose_instance_mask',
        lambda *a,**k:{'candidate_alpha':None,'status':'UNRECOVERABLE',
                       'reason':'BLOCKED_MASK_BACKEND','method':'grabcut'})
    out=run_mask_candidates(inv,tmp_path/'out')
    assert out['status']=='BLOCKED_MASK_BACKEND'
    assert out['approved_masks']==0
    assert out['training_permitted'] is False
