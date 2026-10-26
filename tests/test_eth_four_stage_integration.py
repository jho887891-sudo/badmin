import sys,json,hashlib
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_eth_four_stage_pilot import generate_fixture_pilot, save_pilot


def test_tiny_fixture_pilot_stage1_stage2_stage4_stage3_manifest(tmp_path):
    samples=generate_fixture_pilot()
    assert set(samples)=={'stage1','stage2','stage4'}
    assert samples['stage1']['label'].startswith('0 ')
    assert samples['stage2']['label'].startswith('0 ')
    assert samples['stage4']['label'].startswith('0 ')
    output=save_pilot(samples,tmp_path/'artifacts')
    assert output['status']=='CPU_FIXTURE_ONLY'
    assert (tmp_path/'artifacts'/'manifest.json').exists()
    assert len(output['records'])==3


def test_replay_same_seed_same_hash(tmp_path):
    a=save_pilot(generate_fixture_pilot(seed=12),tmp_path/'run1')
    b=save_pilot(generate_fixture_pilot(seed=12),tmp_path/'run2')
    assert [x['image_sha256'] for x in a['records']]==[x['image_sha256'] for x in b['records']]
    assert [x['label_sha256'] for x in a['records']]==[x['label_sha256'] for x in b['records']]


def test_all_gates_block_without_remote_real_data(tmp_path):
    from train_eth_four_stage import training_preflight
    config={'stages':['stage1','stage2','stage4','stage3'],'imgsz':1024,'seed':42,
            'palette_rgb':[[20,30,40],[60,70,80]],'train_source_splits':['train'],
            'run_id':'ETH_FOUR_STAGE_CP_V1','output_root':str(tmp_path),'checkpoint':{}}
    result=training_preflight(config,{'source_report':{'status':'BLOCKED_NO_REAL_DATA'},
             'stage':'stage3_e4','checkpoint_path':'/absent/checkpoint.pt','dataset_yaml':'/absent/data.yaml',
             'output_dir':str(tmp_path/'train'),'execute':True})
    assert result['status']=='BLOCKED_TRAINING_PREFLIGHT'


def test_stage3_wording_preserved_in_runbook():
    text=(Path(__file__).resolve().parents[1]/'docs/shuttle_detection/ETH_FOUR_STAGE_RUNBOOK.md').read_text(encoding='utf8')
    assert '回到真实背景，做最终训练 / 微调' in text
    assert '灯具误检是否下降' in text


def test_old_frozen_configs_checksums_unmodified():
    p=Path(__file__).resolve().parents[1]/'tools/clean_cp_contract.py'
    if not p.exists():pytest.skip('isolated checkout lacks legacy source')
    raw=p.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='c68b35544a757884be3699a53051c8903e626fd0'
