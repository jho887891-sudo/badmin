"""Stage 3 real-background comparison manifests for E3/E4.

Keep the user's Stage 3 definition. E3 uses Stage 2 as synthetic control,
E4 substitutes Stage 4 at the same synthetic exposure budget.
"""
from __future__ import annotations
import csv
import random
from pathlib import Path
from eth_four_stage_contract import audit_training_sources


class TrainingSetBlocked(ValueError):
    pass


def _read_rows(path):
    p=Path(path)
    if not p.is_file():raise TrainingSetBlocked('manifest missing: '+str(p))
    with p.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise TrainingSetBlocked('empty dataset: '+str(p))
    return rows


def _arm(name, rows, config):
    return {'name':name,'rows':rows,'seed':int(config['seed']),
            'checkpoint_sha256':str(config['checkpoint_sha256']),
            'optimizer_steps':int(config['optimizer_steps'])}


def assert_comparison_parity(e3:dict,e4:dict)->dict:
    violations=[]
    fields=('seed','checkpoint_sha256','optimizer_steps')
    for field in fields:
        if e3.get(field)!=e4.get(field):violations.append(field+'_mismatch')
    a=e3.get('rows',[]);b=e4.get('rows',[])
    if len(a)!=len(b):violations.append('unequal_exposure_slots')
    a_real=[r['id'] for r in a if r.get('role') in {'eth_train','real_venue_train'}]
    b_real=[r['id'] for r in b if r.get('role') in {'eth_train','real_venue_train'}]
    if a_real!=b_real:violations.append('unequal_real_exposures')
    if any(r.get('role') not in {'eth_train','real_venue_train','stage2_degraded'} for r in a):
        violations.append('invalid_e3_role')
    if any(r.get('role') not in {'eth_train','real_venue_train','stage4_composite'} for r in b):
        violations.append('invalid_e4_role')
    return {'status':'BLOCKED_PARITY' if violations else 'PASS','violations':violations,
            'real_exposures':len(a_real),'slots_e3':len(a),'slots_e4':len(b)}


def build_stage_training_manifests(config:dict,pool_manifests:dict[str,str])->dict:
    required=('real_train','stage2','stage4')
    missing=[k for k in required if k not in pool_manifests]
    if missing:raise TrainingSetBlocked('missing pools: '+','.join(missing))
    real=_read_rows(pool_manifests['real_train'])
    s2=_read_rows(pool_manifests['stage2']);s4=_read_rows(pool_manifests['stage4'])
    if any(r.get('role') not in {'eth_train','real_venue_train'} or r.get('split')!='train' for r in real):
        raise TrainingSetBlocked('stage3 real must be approved ETH or real venue train')
    if any(r.get('role')!='stage2_degraded' or r.get('split')!='train' for r in s2):
        raise TrainingSetBlocked('stage2 pool must be approved train-only synthetic')
    if any(r.get('role')!='stage4_composite' or r.get('split')!='train' for r in s4):
        raise TrainingSetBlocked('stage4 pool must be approved train-only synthetic')
    frozen=config.get('frozen_sources',[])
    for pool in (real,s2,s4):
        if audit_training_sources(pool,frozen)['status']!='PASS':
            raise TrainingSetBlocked('provenance/leakage gate failed')
    n=int(config['synthetic_slots_per_epoch'])
    if n<0 or n>len(s2) or n>len(s4):
        raise TrainingSetBlocked('not enough approved synthetic candidates for exposure budget')
    seed=int(config['seed'])
    e3=real+random.Random(seed).sample(s2,n)
    e4=real+random.Random(seed).sample(s4,n)
    arms={'e3':_arm('E3_real_plus_stage2',e3,config),
          'e4':_arm('E4_real_plus_stage4',e4,config)}
    parity=assert_comparison_parity(arms['e3'],arms['e4'])
    if parity['status']!='PASS':raise TrainingSetBlocked('E3/E4 parity: '+','.join(parity['violations']))
    return {**arms,'parity':parity}
