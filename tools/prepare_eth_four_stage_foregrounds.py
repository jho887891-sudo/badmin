#!/usr/bin/env python3
"""Create the ETH four-stage *all-box* foreground inventory, without resizing or training.

This intentionally reuses the repository's frozen ETH train-only selection,
evaluation firewall, label parser and human-mask review contract. The only
size floor is 1x1 source pixel after coordinate quantization, solely to reject
degenerate zero-area boxes; sizes are reported, never filtered into a target band.
All outputs go to a new experiment directory, never CLEAN_OUTDOOR_CP_V1.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_eth_real_foregrounds import (  # noqa: E402
    audit_eth_train_instances, write_foreground_inventory, EthForegroundBlocked,
    FROZEN_EVAL_INDEX_PATH,
)


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def size_summary_1024(instances:list[dict])->dict:
    values=[]
    for row in instances:
        w,h=map(float,row['bbox_wh'])
        fw,fh=int(row['frame_width']),int(row['frame_height'])
        values.append(math.sqrt(w*h)*1024.0/max(fw,fh))
    ordered=sorted(values)
    def quantile(q):
        if not ordered:return None
        return round(ordered[min(len(ordered)-1,max(0,math.ceil(q*len(ordered))-1))],4)
    bounds=[0,4,6,8,12,16,24,32,64,float('inf')]
    labels=['<4','4-6','6-8','8-12','12-16','16-24','24-32','32-64','>=64']
    buckets={k:0 for k in labels}
    for x in values:
        for lo,hi,label in zip(bounds[:-1],bounds[1:],labels):
            if lo<=x<hi:
                buckets[label]+=1;break
    return {'definition':'sqrt(bbox_width_px * bbox_height_px) * 1024 / max(frame_width, frame_height)',
            'purpose':'descriptive only; no size bucket is used to select or discard instances',
            'count':len(ordered),'min':round(ordered[0],4) if ordered else None,
            'p1':quantile(.01),'p5':quantile(.05),'p25':quantile(.25),
            'median':quantile(.5),'p75':quantile(.75),'p95':quantile(.95),
            'p99':quantile(.99),'max':round(ordered[-1],4) if ordered else None,
            'count_by_bucket':buckets}


def run_inventory(manifest:Path,label_root:Path|None,frozen_eval_manifests:list[Path],
                  out_inventory:Path,out_audit:Path,out_jsonl:Path)->dict:
    outputs=[out_inventory,out_audit,out_jsonl]
    if len({str(p.resolve()) for p in outputs})!=3:
        raise ValueError('output paths must be distinct')
    existing=[str(p) for p in outputs if p.exists()]
    if existing:raise FileExistsError('refusing to overwrite existing outputs: '+', '.join(existing))
    if not manifest.is_file():raise FileNotFoundError('ETH source manifest missing: '+str(manifest))
    for p in frozen_eval_manifests:
        if not p.is_file():raise FileNotFoundError('frozen evaluation manifest missing: '+str(p))
    audit=audit_eth_train_instances(
        manifest,min_short=1,min_long=1,label_root=label_root,
        frozen_eval_manifests=frozen_eval_manifests)
    instances=audit['instances']
    rejections=audit['rejections']
    out_inventory.parent.mkdir(parents=True,exist_ok=True)
    out_audit.parent.mkdir(parents=True,exist_ok=True)
    out_jsonl.parent.mkdir(parents=True,exist_ok=True)
    write_foreground_inventory(instances,out_inventory)
    with out_jsonl.open('w',encoding='utf-8') as f:
        for row in instances:
            f.write(json.dumps({'record_type':'candidate','status':'PENDING_REVIEW',**row},ensure_ascii=False)+'\n')
        for row in rejections:
            f.write(json.dumps({'record_type':'rejection',**row},ensure_ascii=False)+'\n')
    reason_counts={}
    for row in rejections:
        reason=str(row.get('reason') or 'unknown')
        reason_counts[reason]=reason_counts.get(reason,0)+1
    report={
        'status':'CANDIDATES_PENDING_REVIEW' if instances else 'NO_ELIGIBLE_INSTANCES',
        'source_manifest':str(manifest),
        'source_manifest_sha256':sha256_file(manifest),
        'frozen_eval_manifests':[{'path':str(p),'sha256':sha256_file(p)} for p in frozen_eval_manifests],
        'label_root':str(label_root) if label_root else None,
        'selection_contract':{
            'allowed_source':'ETH real train positives only',
            'split':'train',
            'min_short_px_after_rounding':1,
            'min_long_px_after_rounding':1,
            'size_floor_reason':'non-zero geometry only; not a target-size filter',
            'human_mask_review_required':True,
            'bbox_crop_is_not_a_mask':True,
            'upscaled_source_crop_forbidden':True,
            'frozen_eval_path_hash_and_stem_exclusion':True},
        'manifest_rows':int(audit['rows']),
        'candidate_instances':len(instances),
        'rejected_rows_or_boxes':len(rejections),
        'rejections_by_reason':reason_counts,
        'candidate_locations':audit['locations'],
        'size_distribution_at_1024':size_summary_1024(instances),
        'outputs':{'inventory_csv':str(out_inventory),'audit_json':str(out_audit),'audit_jsonl':str(out_jsonl)},
        'training_started':False,
        'gpu_steps':0,
        'mask_approval':'PENDING',
        'real_video_recall':'UNKNOWN'}
    out_audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=REPO/'data/eth_real_hardneg_v2_train_manifest.csv')
    p.add_argument('--label-root',type=Path,default=None)
    p.add_argument('--frozen-eval-manifest',type=Path,action='append',default=None)
    p.add_argument('--out-inventory',type=Path,default=REPO/'data/eth_four_stage_copypaste_v1/foreground_manifest.csv')
    p.add_argument('--out-audit',type=Path,default=REPO/'outputs/shuttle_detection/eth_four_stage_copypaste_v1/foreground_audit.json')
    p.add_argument('--out-jsonl',type=Path,default=REPO/'outputs/shuttle_detection/eth_four_stage_copypaste_v1/foreground_audit.jsonl')
    args=p.parse_args(argv)
    frozen=args.frozen_eval_manifest or [REPO/FROZEN_EVAL_INDEX_PATH]
    try:
        report=run_inventory(args.manifest,args.label_root,frozen,args.out_inventory,args.out_audit,args.out_jsonl)
    except (EthForegroundBlocked,FileNotFoundError,FileExistsError,ValueError) as exc:
        print(json.dumps({'status':'BLOCKED_ETH_INVENTORY','error':str(exc)},ensure_ascii=False,indent=2))
        return 2
    print(json.dumps({k:v for k,v in report.items() if k not in {'outputs','frozen_eval_manifests'}},ensure_ascii=False,indent=2))
    return 0 if report['candidate_instances'] else 1


if __name__=='__main__':raise SystemExit(main())
