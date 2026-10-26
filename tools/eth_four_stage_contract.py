"""Isolated source/recipe guards for ETH four-stage Copy-Paste.

This module deliberately never imports the frozen CLEAN_OUTDOOR_CP_V1 contract.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import yaml

STAGES = ('stage1', 'stage2', 'stage4', 'stage3')
TRAIN_ROLES = frozenset({'eth_train','stage1_solid','stage2_degraded','stage4_composite',
                         'real_venue_train','approved_background','hard_negative_train'})
SHA256 = re.compile(r'^[0-9a-f]{64}$')


class ContractViolation(RuntimeError):
    pass


def _required(config: dict, key: str):
    if key not in config:
        raise ContractViolation('missing configuration key: ' + key)
    return config[key]


def _is_near_white(rgb) -> bool:
    return min(rgb) >= 235


def validate_stage_contract(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ContractViolation('configuration must be a mapping')
    if list(_required(config,'stages')) != list(STAGES):
        raise ContractViolation('stages must be stage1,stage2,stage4,stage3 (execution dependencies)')
    if _required(config,'imgsz') != 1024:
        raise ContractViolation('imgsz must be 1024 for this experiment')
    if not str(_required(config,'run_id')).strip():
        raise ContractViolation('run_id must not be blank')
    palette = _required(config,'palette_rgb')
    if not isinstance(palette,list) or len({tuple(map(int,c)) for c in palette}) < 2:
        raise ContractViolation('palette requires at least 2 distinct non-white colors')
    for color in palette:
        if (not isinstance(color,(list,tuple)) or len(color)!=3 or
           any(not isinstance(v,int) or v<0 or v>255 for v in color) or _is_near_white(color)):
            raise ContractViolation(f'illegal non-white RGB palette color: {color}')
    if list(_required(config,'train_source_splits'))!=['train']:
        raise ContractViolation('only ETH train may supply positive training instances')
    if not str(_required(config,'output_root')).strip():
        raise ContractViolation('output_root is required')
    _required(config,'seed')
    return config


def load_stage_contract(path: str | Path) -> dict:
    with open(path,encoding='utf-8') as f:
        return validate_stage_contract(yaml.safe_load(f))


def audit_training_sources(rows: list[dict], frozen_rows: list[dict]) -> dict:
    """Reject eval origins and their derivatives even after renaming.

    Several boxes from the same ETH *train* frame may legitimately share source SHA.
    So identity is record ID, not source hash, except against forbidden sources.
    """
    forbidden_hashes = {r.get('source_sha256') for r in frozen_rows if r.get('source_sha256')}
    forbidden_paths = {str(r.get('path')) for r in frozen_rows if r.get('path')}
    reasons = []
    seen_ids = set()
    for row in rows:
        key = str(row.get('id') or '').strip()
        if not key or key in seen_ids:
            reasons.append({'id':key,'reason':'duplicate_or_missing_instance_id'})
        seen_ids.add(key)
        if row.get('role') not in TRAIN_ROLES:
            reasons.append({'id':key,'reason':'invalid_training_role'})
        if row.get('split')!='train':
            reasons.append({'id':key,'reason':'non_train_split'})
        if row.get('role') in {'stage1_solid','stage2_degraded','stage4_composite'}:
            if not row.get('parent_sha256') or (row.get('role')=='stage4_composite' and not row.get('background_sha256')):
                reasons.append({'id':key,'reason':'missing_lineage_hash'})
        sha_values=(row.get('source_sha256'),row.get('parent_sha256'),row.get('background_sha256'))
        if any(sha and sha in forbidden_hashes for sha in sha_values) or str(row.get('path')) in forbidden_paths:
            reasons.append({'id':key,'reason':'frozen_source_or_parent_sha'})
    return {'status':'INVALID_OR_LEAK' if reasons else 'PASS',
            'rows':len(rows), 'violations':reasons}


def _sha256_file(path: str|Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def assert_stage_preflight(config: dict, source_report: dict,
                           checkpoint: dict | None = None,
                           require_checkpoint: bool = False) -> dict:
    """Metadata-only preflight unless checkpoint is actually mounted.

    Passing a well-formed checkpoint *description* is not verification of bytes.
    To require on-disk verification provide checkpoint={'verify_on_disk':True,...}.
    """
    reasons=[]
    try:
        validate_stage_contract(config)
    except (ValueError, TypeError, ContractViolation) as exc:
        reasons.append('invalid_contract: '+str(exc))
    if source_report.get('status')!='PASS':
        reasons.append('source_gate_not_pass')
    if require_checkpoint:
        ck=checkpoint or config.get('checkpoint') or {}
        path=ck.get('path')
        sha=str(ck.get('sha256') or '').lower()
        if not path or not SHA256.fullmatch(sha):
            reasons.append('missing_checkpoint_sha_or_path')
        elif ck.get('verify_on_disk'):
            if not Path(path).is_file() or _sha256_file(path)!=sha:
                reasons.append('checkpoint_missing_or_sha_mismatch')
    return {'status':'BLOCKED_PREFLIGHT' if reasons else 'PASS','reasons':reasons,
            'checkpoint_verified_on_disk':bool(checkpoint and checkpoint.get('verify_on_disk') and not reasons)}
