#!/usr/bin/env python3
"""ETH YOLO26s Stage 1: source-frame split before synthetic train/validation.

Deterministic, frame-index embargoed 80/20 holdout per (location,set_dir).
This is a temporal split, NOT independent-video validation. Both partitions
use the existing pixel-exact, unscaled BBox solid-background generator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from generate_eth_stage1_bbox_preview import (
    PreviewBlocked, load_inventory, run, sha256, LOCATIONS,
)

class HoldoutBlocked(ValueError):
    pass

FIELDS = ('instance_id','image','label','source_sha256','label_sha256',
          'split','source','role','location','set_dir','bbox_xyxy')

def source_split(rows: list[dict], *, train_fraction: float = .8,
                 embargo_frames: int = 10) -> tuple[dict[str,list[dict]],dict]:
    if not 0.5 <= train_fraction <= 0.95 or embargo_frames < 0:
        raise HoldoutBlocked('train-fraction must be 0.5..0.95 and embargo >=0')
    sequences = defaultdict(list)
    seen_ids = set()
    for row in rows:
        if row['instance_id'] in seen_ids:
            raise HoldoutBlocked('duplicate source instance')
        seen_ids.add(row['instance_id'])
        sequences[(row['location'],row['set_dir'])].append(row)
    groups = {'train':[], 'val':[], 'embargo':[]}
    per_sequence = []
    for (scene,sequence),items in sorted(sequences.items()):
        frames = sorted({r['_frame_index'] for r in items})
        if len(frames)<3:
            raise HoldoutBlocked('not enough distinct frames: '+scene+'/'+sequence)
        cut = max(1,min(len(frames)-1,math.floor(len(frames)*train_fraction)))
        left,right = frames[cut-1],frames[cut]
        selected = {'train':[],'val':[],'embargo':[]}
        for row in items:
            idx = row['_frame_index']
            partition = ('train' if idx <= left-embargo_frames else
                         'val' if idx >= right+embargo_frames else 'embargo')
            selected[partition].append(row)
        if not selected['train'] or not selected['val']:
            raise HoldoutBlocked('embargo exhausted train/val: '+scene+'/'+sequence)
        for key,part in selected.items():
            groups[key].extend(part)
        min_val=min(x['_frame_index'] for x in selected['val'])
        max_train=max(x['_frame_index'] for x in selected['train'])
        if min_val-max_train<=embargo_frames:
            raise HoldoutBlocked('frame-index gap failure')
        per_sequence.append(dict(location=scene,set_dir=sequence,
             unique_frames=len(frames),cut_frame_left=left,cut_frame_right=right,
             train_frames=len({r['_frame_index'] for r in selected['train']}),
             val_frames=len({r['_frame_index'] for r in selected['val']}),
             embargo_frames=len({r['_frame_index'] for r in selected['embargo']}),
             last_train_frame=max_train,first_val_frame=min_val))
    source_to_partition={}
    for name,part in groups.items():
        for row in part:
            digest=row['source_sha256'].lower()
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise HoldoutBlocked('invalid original source SHA')
            existing=source_to_partition.setdefault(digest,name)
            if existing != name:
                raise HoldoutBlocked('original source SHA leaks across '+existing+' and '+name+': '+digest)
    all_ids={r['instance_id'] for part in groups.values() for r in part}
    if len(all_ids)!=len(rows) or len(rows)!=sum(len(v) for v in groups.values()):
        raise HoldoutBlocked('split coverage failure')
    if ({r['location'] for r in groups['train']} != set(LOCATIONS) or
        {r['location'] for r in groups['val']} != set(LOCATIONS)):
        raise HoldoutBlocked('missing ETH scene from train or val')
    report=dict(split_rule='per_location_set_dir_chronological_source_frame_holdout',
        interpretation='frame-index holdout with embargo, NOT independent-video validation',
        train_fraction_requested=train_fraction,
        embargo_frame_indices=embargo_frames,original_instances=len(rows),
        train_unique_instances=len(groups['train']),
        val_unique_instances=len(groups['val']),
        embargoed_instances=len(groups['embargo']),
        train_val_source_sha_overlap=0,train_val_source_instance_overlap=0,
        by_split_scene={k:dict(sorted(Counter(r['location'] for r in v).items()))
                        for k,v in groups.items()},
        sequences=per_sequence)
    return groups,report

def _write_inventory(path:Path,rows:list[dict])->None:
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows,key=lambda x:x['instance_id']):
            writer.writerow({key:row[key] for key in FIELDS})

def build(inventory:Path, output_root:Path, *, image_root:Path|None=None,
          expected_instances:int=13992, seed:int=20261027, time_bins:int=10,
          train_fraction:float=.8, embargo_frames:int=10,
          max_train_outputs:int=100000, size:tuple[int,int]=(1024,1024))->dict:
    if output_root.exists():
        raise HoldoutBlocked('output root already exists: '+str(output_root))
    if size!=(1024,1024) or max_train_outputs<1:
        raise HoldoutBlocked('requires 1024x1024 and positive max-train-outputs')
    rows=load_inventory(inventory,expected_count=expected_instances,bins=time_bins)
    partitions,report=source_split(rows,train_fraction=train_fraction,
                                    embargo_frames=embargo_frames)
    output_root.mkdir(parents=True)
    for name in ('train','val','embargo'):
        _write_inventory(output_root/f'{name}_inventory.csv',partitions[name])
    t=run(output_root/'train_inventory.csv',output_root/'train_synthetic',
          mode='balanced-full',seed=seed,bins=time_bins,size=size,
          image_root=image_root,expected_count=len(partitions['train']),
          max_outputs=max_train_outputs)
    v=run(output_root/'val_inventory.csv',output_root/'val_synthetic',
          mode='full',seed=seed+1,bins=time_bins,size=size,
          image_root=image_root,expected_count=len(partitions['val']))
    if not (t['full_unique_coverage'] and v['full_unique_coverage'] and
            t['unique_instances_covered']==len(partitions['train']) and
            v['unique_instances_covered']==len(partitions['val'])):
        raise HoldoutBlocked('generated source coverage mismatch')
    report.update(status='PASS_SPLIT_AND_SYNTHETIC_GENERATION_NO_TRAINING',
       source_inventory_sha256=sha256(inventory),
       generator='generate_eth_stage1_bbox_preview.py',
       training_synthetic_mode='balanced-full',validation_synthetic_mode='full',
       source_pixel_resize=False,foreground_method='bbox_crop_no_mask_no_resize',
       train_and_val_background_policy='same_solid_background_palette',
       model_trained=False,optimizer_steps=0,seed=seed,image_size=list(size),
       time_bins_per_sequence=time_bins,
       train_generated_images=t['generated_images'],
       val_generated_images=v['generated_images'],
       train_manifest_sha256=t['manifest_sha256'],
       val_manifest_sha256=v['manifest_sha256'],
       train_audit_sha256=sha256(output_root/'train_synthetic'/'audit.json'),
       val_audit_sha256=sha256(output_root/'val_synthetic'/'audit.json'),
       train_inventory_sha256=sha256(output_root/'train_inventory.csv'),
       val_inventory_sha256=sha256(output_root/'val_inventory.csv'),
       embargo_inventory_sha256=sha256(output_root/'embargo_inventory.csv'),
       visual_quality_review='UNKNOWN_PENDING_HUMAN_REVIEW')
    (output_root/'split_audit.json').write_text(
       json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',
       encoding='utf-8')
    with (output_root/'SHA256SUMS').open('x',encoding='utf-8') as stream:
        for name in ('train_inventory.csv','val_inventory.csv',
                     'embargo_inventory.csv','split_audit.json'):
            path=output_root/name
            stream.write(f'{sha256(path)}  {name}\n')
    return report

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--image-root',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--expected-instances',type=int,default=13992)
    p.add_argument('--train-fraction',type=float,default=.8)
    p.add_argument('--embargo-frames',type=int,default=10)
    p.add_argument('--time-bins',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--max-train-outputs',type=int,default=100000)
    args=p.parse_args(argv)
    try:
        result=build(args.inventory,args.output_root,image_root=args.image_root,
            expected_instances=args.expected_instances,seed=args.seed,
            train_fraction=args.train_fraction,embargo_frames=args.embargo_frames,
            time_bins=args.time_bins,max_train_outputs=args.max_train_outputs)
    except (HoldoutBlocked,PreviewBlocked,OSError,ValueError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)},ensure_ascii=False))
        return 2
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
