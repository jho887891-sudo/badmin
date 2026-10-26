import sys,csv
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_eth_four_stage_train import build_stage_training_manifests,assert_comparison_parity,TrainingSetBlocked


def write(path,role,n):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','image','label','source_sha256','parent_sha256','background_sha256','role','split'])
        w.writeheader()
        for i in range(n):
            w.writerow({'id':f'{role}-{i}','image':f'/data/{role}-{i}.jpg','label':f'/data/{role}-{i}.txt',
                        'source_sha256':f'{role}-hash-{i}','parent_sha256':'ETH_PARENT' if role!='eth_train' else '',
                        'background_sha256':'BG_PARENT' if role=='stage4_composite' else '',
                        'role':role,'split':'train'})
    return str(path)


def data(tmp_path):
    return {'real_train':write(tmp_path/'real.csv','eth_train',3),
            'stage2':write(tmp_path/'s2.csv','stage2_degraded',3),
            'stage4':write(tmp_path/'s4.csv','stage4_composite',3)}


def config():
    return {'synthetic_slots_per_epoch':2,'seed':42,'checkpoint_sha256':'abc','optimizer_steps':100,
            'frozen_sources':[]}


def test_stage3_retains_all_allowed_original_train_positives(tmp_path):
    out=build_stage_training_manifests(config(),data(tmp_path))
    assert [r['id'] for r in out['e3']['rows'] if r['role']=='eth_train']==[f'eth_train-{i}' for i in range(3)]
    assert [r['id'] for r in out['e4']['rows'] if r['role']=='eth_train']==[f'eth_train-{i}' for i in range(3)]


def test_unrecoverable_mask_sample_keeps_original_real_image(tmp_path):
    d=data(tmp_path);out=build_stage_training_manifests(config(),d)
    assert 'eth_train-2' in [r['id'] for r in out['e4']['rows']]


def test_e3_e4_use_identical_real_image_ids(tmp_path):
    out=build_stage_training_manifests(config(),data(tmp_path))
    assert assert_comparison_parity(out['e3'],out['e4'])['status']=='PASS'


def test_e3_e4_equal_exposure_and_steps(tmp_path):
    out=build_stage_training_manifests(config(),data(tmp_path))
    assert len(out['e3']['rows'])==len(out['e4']['rows'])==5
    assert out['e3']['optimizer_steps']==out['e4']['optimizer_steps']==100


def test_e4_only_adds_stage4_pool(tmp_path):
    out=build_stage_training_manifests(config(),data(tmp_path))
    assert {r['role'] for r in out['e3']['rows']}=={'eth_train','stage2_degraded'}
    assert {r['role'] for r in out['e4']['rows']}=={'eth_train','stage4_composite'}


def test_eval_image_derivative_rejected(tmp_path):
    d=data(tmp_path);cfg=config();cfg['frozen_sources']=[{'source_sha256':'ETH_PARENT'}]
    with pytest.raises(TrainingSetBlocked):build_stage_training_manifests(cfg,d)


def test_parity_mismatch_is_blocker(tmp_path):
    out=build_stage_training_manifests(config(),data(tmp_path))
    out['e4']['optimizer_steps']=101
    assert assert_comparison_parity(out['e3'],out['e4'])['status']=='BLOCKED_PARITY'


def test_not_enough_synthetic_candidates_blocks(tmp_path):
    d=data(tmp_path);cfg=config();cfg['synthetic_slots_per_epoch']=4
    with pytest.raises(TrainingSetBlocked):build_stage_training_manifests(cfg,d)


def test_stage3_allows_approved_real_venue_training_images(tmp_path):
    d=data(tmp_path)
    p=Path(d['real_train'])
    rows=list(csv.DictReader(p.open()))
    rows[0]['role']='real_venue_train'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=build_stage_training_manifests(config(),d)
    assert any(r['role']=='real_venue_train' for r in out['e4']['rows'])
    assert out['parity']['real_exposures']==3
