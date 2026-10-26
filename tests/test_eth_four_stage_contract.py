import sys
import hashlib
from pathlib import Path

import pytest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from eth_four_stage_contract import (
    ContractViolation, load_stage_contract, audit_training_sources,
    assert_stage_preflight,
)


def contract():
    return {'run_id':'ETH_FOUR_STAGE_CP_V1',
            'stages':['stage1','stage2','stage4','stage3'],
            'imgsz':1024,'palette_rgb':[[30,40,80],[17,75,30]],
            'output_root':'/remote/eth_four_stage',
            'seed':42,'checkpoint':{'path':'/remote/weights.pt','sha256':'a'*64},
            'train_source_splits':['train']}


def row(id, role='eth_train', sha='a', split='train', parent=None, **kw):
    r={'id':id,'role':role,'source_sha256':sha,'split':split,'path':f'/train/{id}.png'}
    if parent: r['parent_sha256']=parent
    r.update(kw)
    return r


def test_rejects_eval_derived_source_even_with_new_path():
    train=[row('composite','stage4_composite','unique',parent='evalsha')]
    result=audit_training_sources(train,[row('eval','frozen_eval','evalsha',split='test')])
    assert result['status']=='INVALID_OR_LEAK'
    assert 'frozen_source_or_parent_sha' in [x['reason'] for x in result['violations']]


def test_stage_ids_and_palette_required(tmp_path):
    good=contract()
    file=tmp_path/'config.yaml';file.write_text(yaml.safe_dump(good), encoding='utf-8')
    assert load_stage_contract(file)['stages']==['stage1','stage2','stage4','stage3']
    for bad in ({**good,'stages':['stage1','stage2','stage3']},
                {**good,'palette_rgb':[[255,255,255]]},
                {**good,'palette_rgb':[[250,249,251]]},
                {**good,'imgsz':640}):
        file.write_text(yaml.safe_dump(bad), encoding='utf-8')
        with pytest.raises(ContractViolation):load_stage_contract(file)


def test_missing_checkpoint_blocks_training_not_inventory():
    x=contract();x['checkpoint']={}
    assert assert_stage_preflight(x,{'status':'PASS'},require_checkpoint=False)['status']=='PASS'
    assert assert_stage_preflight(x,{'status':'PASS'},require_checkpoint=True)['status']=='BLOCKED_PREFLIGHT'


def test_old_and_new_source_roles_block_eval_and_duplicates():
    frozen=[row('testpic','frozen_eval','X',split='test')]
    bad=[row('a','eth_train','A'),row('b','eth_train','A'),row('c','eth_train','X')]
    report=audit_training_sources(bad,frozen)
    reasons={x['reason'] for x in report['violations']}
    assert reasons=={'frozen_source_or_parent_sha'}


def test_eval_split_cannot_enter_training_pool():
    report=audit_training_sources([row('a','eth_train','XYZ',split='val')], [])
    assert report['status']=='INVALID_OR_LEAK'
    assert report['violations'][0]['reason']=='non_train_split'


def test_clean_preflight_allows_valid_metadata():
    assert assert_stage_preflight(contract(),{'status':'PASS'},require_checkpoint=True)['status']=='PASS'


def test_rejects_unknown_source_role():
    report=audit_training_sources([row('a','invented_role','Y')], [])
    assert report['status']=='INVALID_OR_LEAK'


def test_legacy_frozen_config_unchanged():
    legacy = Path(__file__).resolve().parents[1] / 'tools/clean_cp_contract.py'
    if not legacy.exists():
        pytest.skip('isolated partial checkout: legacy file not mounted')
    raw = legacy.read_bytes()
    git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert git_blob == 'c68b35544a757884be3699a53051c8903e626fd0'



def test_composite_must_have_parent_and_background_provenance():
    audit=audit_training_sources([row('c','stage4_composite','newsha')], [])
    assert 'missing_lineage_hash' in [v['reason'] for v in audit['violations']]


def test_multiple_train_boxes_from_same_frame_are_not_duplicates():
    records=[row('source#0','eth_train','same'),row('source#1','eth_train','same')]
    assert audit_training_sources(records,[])['status']=='PASS'
