#!/usr/bin/env python3
"""Gated four-stage Ultralytics launcher. Default is preflight/dry-run.

Never deploys a robot or trains without an explicit --execute plus verified
checkpoint SHA, approved source report, readable YOLO dataset YAML and
nonexisting output directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from eth_four_stage_contract import validate_stage_contract,ContractViolation


class TrainingLaunchBlocked(RuntimeError):
    pass


ALLOWED_STAGES={'stage1','stage2','stage3_e3','stage3_e4'}


def build_train_command(config:dict,stage:str,manifest_path:str,checkpoint_path:str,output_dir:str)->list[str]:
    if stage not in ALLOWED_STAGES:raise TrainingLaunchBlocked('unsupported training stage')
    if config.get('selection_scope','internal_val') not in ('internal_val','train_internal_val'):
        raise TrainingLaunchBlocked('checkpoint selection must use internal validation only')
    if int(config.get('imgsz',0))!=1024:raise TrainingLaunchBlocked('input size contract requires 1024')
    p=Path(output_dir)
    if not str(p.name).strip() or not str(checkpoint_path) or not str(manifest_path):
        raise TrainingLaunchBlocked('missing explicit train paths')
    options={'model':checkpoint_path,'data':manifest_path,'imgsz':1024,
             'epochs':int(config.get('epochs',50)),'batch':int(config.get('batch',8)),
             'optimizer':config.get('optimizer','AdamW'),'lr0':float(config.get('lr0',1e-4)),
             'seed':int(config['seed']),'project':str(p.parent),'name':p.name,
             'exist_ok':False,'pretrained':True}
    return ['yolo','detect','train']+[f'{k}={v}' for k,v in options.items()]


def _sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()


def training_preflight(config:dict, inputs:dict)->dict:
    problems=[]
    try:validate_stage_contract(config)
    except (ContractViolation,TypeError,ValueError) as exc:problems.append('invalid_contract:'+str(exc))
    if inputs.get('source_report',{}).get('status')!='PASS':problems.append('source_gate_not_pass')
    stage=inputs.get('stage')
    if stage not in ALLOWED_STAGES:problems.append('invalid_stage')
    if config.get('selection_scope','internal_val') not in ('internal_val','train_internal_val'):
        problems.append('test_set_checkpoint_selection_forbidden')
    if not inputs.get('execute',False):
        return {'status':'DRY_RUN_ONLY' if not problems else 'BLOCKED_TRAINING_PREFLIGHT',
                'reasons':problems,'optimizer_steps':0,'executed':False}
    ck=config.get('checkpoint') or {}
    path=Path(inputs.get('checkpoint_path') or ck.get('path') or '/invalid/missing.pt')
    sha=str(ck.get('sha256') or '').lower()
    if not ck.get('path') or str(ck.get('path'))!=str(path) or len(sha)!=64:
        problems.append('checkpoint_identity_missing')
    if not path.is_file():problems.append('checkpoint_missing')
    elif _sha256_file(path)!=sha:problems.append('checkpoint_sha_mismatch')
    data=Path(inputs.get('dataset_yaml') or '/invalid/missing.yaml')
    if not data.is_file():problems.append('dataset_yaml_missing')
    else:
        import yaml
        payload=yaml.safe_load(data.read_text(encoding='utf-8'))
        if not isinstance(payload,dict) or not {'train','val','names'}<=set(payload):
            problems.append('dataset_yaml_invalid')
    out=Path(inputs.get('output_dir') or '/invalid')
    if out.exists():problems.append('output_directory_already_exists')
    if not str(inputs.get('output_dir') or ''):problems.append('output_directory_missing')
    return {'status':'BLOCKED_TRAINING_PREFLIGHT' if problems else 'PASS',
            'reasons':problems,'optimizer_steps':0,'executed':False,
            'checkpoint_verified':not any(x.startswith('checkpoint_') for x in problems)}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--stage',required=True,choices=sorted(ALLOWED_STAGES))
    ap.add_argument('--checkpoint-path',required=True)
    ap.add_argument('--dataset-yaml',required=True)
    ap.add_argument('--source-audit-json',required=True)
    ap.add_argument('--output-dir',required=True)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    import yaml
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    source=json.loads(Path(args.source_audit_json).read_text(encoding='utf-8'))
    inputs={'stage':args.stage,'source_report':source,'checkpoint_path':args.checkpoint_path,
            'dataset_yaml':args.dataset_yaml,'output_dir':args.output_dir,'execute':args.execute}
    report=training_preflight(config,inputs)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if report['status']=='BLOCKED_TRAINING_PREFLIGHT':return 2
    if not args.execute:return 0
    if report['status']!='PASS':return 2
    if shutil.which('yolo') is None:raise TrainingLaunchBlocked('Ultralytics yolo CLI not installed')
    cmd=build_train_command(config,args.stage,args.dataset_yaml,args.checkpoint_path,args.output_dir)
    result=subprocess.run(cmd,check=False)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
