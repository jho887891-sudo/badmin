import sys,hashlib
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_eth_four_stage import build_train_command,training_preflight,TrainingLaunchBlocked


def config():
    return {'stages':['stage1','stage2','stage4','stage3'],'imgsz':1024,
            'palette_rgb':[[15,25,35],[45,55,65]],'train_source_splits':['train'],
            'run_id':'ETH_FOUR_STAGE_CP_V1','seed':42,'output_root':'/remote/eth_four_stage',
            'checkpoint':{'path':'/remote/missing.pt','sha256':'0'*64},
            'epochs':50,'batch':8,'optimizer':'AdamW','lr0':1e-4}


def test_default_dry_run_never_imports_gpu_stack_or_trains(tmp_path):
    c=config();argv=build_train_command(c,'stage1',str(tmp_path/'stage1.yaml'),'/weights/init.pt',str(tmp_path/'run'))
    assert argv[0]=='yolo'
    assert 'imgsz=1024' in argv
    assert 'exist_ok=False' in argv


def test_missing_checkpoint_hash_blocks_execute(tmp_path):
    c=config();c['checkpoint']={}
    out=training_preflight(c,{'source_report':{'status':'PASS'},'stage':'stage1',
                              'checkpoint_path':str(tmp_path/'missing.pt'),'dataset_yaml':str(tmp_path/'data.yaml'),
                              'output_dir':str(tmp_path/'new'),'execute':True})
    assert out['status']=='BLOCKED_TRAINING_PREFLIGHT'


def test_failed_mask_or_source_gate_blocks_execute(tmp_path):
    c=config();out=training_preflight(c,{'source_report':{'status':'INVALID_OR_LEAK'},'stage':'stage1',
                              'checkpoint_path':'/remote/missing.pt','dataset_yaml':'/missing.yaml',
                              'output_dir':str(tmp_path/'run'),'execute':True})
    assert out['status']=='BLOCKED_TRAINING_PREFLIGHT'
    assert 'source_gate_not_pass' in out['reasons']


def test_stage2_warm_starts_from_stage1():
    c=config();a=build_train_command(c,'stage2','/data/s2.yaml','/trained/stage1/best.pt','/runs/s2')
    assert 'model=/trained/stage1/best.pt' in a


def test_stage3_warm_starts_from_stage2():
    c=config();a=build_train_command(c,'stage3_e4','/data/e4.yaml','/trained/stage2/best.pt','/runs/e4')
    assert 'model=/trained/stage2/best.pt' in a


def test_eval_sets_cannot_select_checkpoint(tmp_path):
    c=config();c['selection_scope']='challenge_test'
    with pytest.raises(TrainingLaunchBlocked):
        build_train_command(c,'stage3_e4','/data/e4.yaml','/trained/s2.pt','/runs/e4')


def test_distinct_output_dirs_no_overwrite(tmp_path):
    c=config();o=tmp_path/'exists';o.mkdir()
    z=training_preflight(c,{'source_report':{'status':'PASS'},'stage':'stage1',
                             'checkpoint_path':'/remote/missing.pt','dataset_yaml':'/missing.yaml',
                             'output_dir':str(o),'execute':True})
    assert 'output_directory_already_exists' in z['reasons']


def test_actual_checkpoint_bytes_must_match_metadata(tmp_path):
    c=config();data=tmp_path/'data.yaml';data.write_text('train: images/train\nval: images/val\nnames: [shuttlecock]\n')
    weight=tmp_path/'weight.pt';weight.write_bytes(b'not a real model')
    c['checkpoint']={'path':str(weight),'sha256':'f'*64}
    out=training_preflight(c,{'source_report':{'status':'PASS'},'stage':'stage1',
                             'checkpoint_path':str(weight),'dataset_yaml':str(data),
                             'output_dir':str(tmp_path/'new'),'execute':True})
    assert out['status']=='BLOCKED_TRAINING_PREFLIGHT'
    assert 'checkpoint_sha_mismatch' in out['reasons']
