import csv
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from select_eth_mask_pilot import select_mask_pilot, PilotBlocked

COLS = ['instance_id','image','label','source_sha256','label_sha256',
        'split','source','role','location','bbox_short','bbox_long','bbox_xyxy','status']

def inputs(tmp_path, sizes=(2,3,4,5,10,12,20,24,25,31,50,60)):
    root=tmp_path/'src';root.mkdir()
    inv=root/'instances.csv'
    with inv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader()
        for i,s in enumerate(sizes):
            w.writerow({'instance_id':f'{i:064x}#0','image':f'/remote/{i}.png',
                'label':f'/remote/{i}.txt','source_sha256':f'{i:064x}',
                'label_sha256':f'{i+1:064x}','split':'train','source':'eth_main',
                'role':'eth_train','location':f'court_{i%3}', 'bbox_short':s,
                'bbox_long':s+5,'bbox_xyxy':'[1,1,6,7]','status':'PENDING_REVIEW'})
    audit=root/'audit.json'
    audit.write_text(json.dumps({'status':'CANDIDATES_PENDING_REVIEW','source_gate':'PASS',
        'inventory_csv_sha256':hashlib.sha256(inv.read_bytes()).hexdigest(),
        'next_gate':'HUMAN_MASK_REVIEW_REQUIRED',
        'counts':{'usable_instances':len(sizes)}}))
    return inv,audit

def test_stratified_120_pilot_contract_and_no_mask_approval(tmp_path):
    inv,audit=inputs(tmp_path)
    report=select_mask_pilot(inv,audit,tmp_path/'pilot',pilot_count=8,seed=42)
    assert report['status']=='PILOT_READY_FOR_MASK_PROPOSALS'
    assert report['selected_by_band']=={'lt4':2,'4to9':2,'10to23':2,'ge24':2}
    assert report['human_approved_masks']==0
    assert report['training_permitted'] is False
    rows=list(csv.DictReader((tmp_path/'pilot'/'pilot_instances.csv').open()))
    assert len(rows)==8
    assert len({x['instance_id'] for x in rows})==8
    assert all(x['split']=='train' and x['source']=='eth_main' for x in rows)

def test_same_seed_gives_same_hash(tmp_path):
    inv,audit=inputs(tmp_path)
    a=select_mask_pilot(inv,audit,tmp_path/'a',pilot_count=8,seed=7)
    b=select_mask_pilot(inv,audit,tmp_path/'b',pilot_count=8,seed=7)
    assert a['pilot_sha256']==b['pilot_sha256']

def test_corrupted_inventory_blocks_even_if_rows_look_valid(tmp_path):
    inv,audit=inputs(tmp_path)
    inv.write_text(inv.read_text()+'corruption\n')
    with pytest.raises(PilotBlocked,match='sha256'):
        select_mask_pilot(inv,audit,tmp_path/'out',pilot_count=4)

def test_blocked_r0_never_selects(tmp_path):
    inv,audit=inputs(tmp_path)
    data=json.loads(audit.read_text());data['status']='BLOCKED_RECONCILIATION'
    audit.write_text(json.dumps(data))
    with pytest.raises(PilotBlocked,match='R0_gate'):
        select_mask_pilot(inv,audit,tmp_path/'out',pilot_count=4)

def test_output_overwrite_refused(tmp_path):
    inv,audit=inputs(tmp_path);dest=tmp_path/'out';dest.mkdir()
    with pytest.raises(PilotBlocked,match='already_exists'):
        select_mask_pilot(inv,audit,dest,pilot_count=4)

def test_duplicate_source_instance_blocks(tmp_path):
    inv,audit=inputs(tmp_path)
    rows=list(csv.DictReader(inv.open()));rows[1]['instance_id']=rows[0]['instance_id']
    with inv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader();w.writerows(rows)
    data=json.loads(audit.read_text())
    data['inventory_csv_sha256']=hashlib.sha256(inv.read_bytes()).hexdigest()
    audit.write_text(json.dumps(data))
    with pytest.raises(PilotBlocked,match='duplicate'):
        select_mask_pilot(inv,audit,tmp_path/'out',pilot_count=4)

def test_nontrain_role_or_missing_label_hash_blocks(tmp_path):
    inv,audit=inputs(tmp_path)
    rows=list(csv.DictReader(inv.open()))
    rows[0]['split']='val'
    with inv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader();w.writerows(rows)
    data=json.loads(audit.read_text())
    data['inventory_csv_sha256']=hashlib.sha256(inv.read_bytes()).hexdigest()
    audit.write_text(json.dumps(data))
    with pytest.raises(PilotBlocked,match='unauthorized'):
        select_mask_pilot(inv,audit,tmp_path/'out',pilot_count=4)

def test_pilot_size_larger_than_inventory_rejected(tmp_path):
    inv,audit=inputs(tmp_path)
    with pytest.raises(PilotBlocked,match='too_small'):
        select_mask_pilot(inv,audit,tmp_path/'out',pilot_count=120)
