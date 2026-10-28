#!/usr/bin/env python3
"""YOLO26s Stage1 training with both train/val synthetic, leakage-checked.

Requires tools/build_eth_stage1_synthetic_holdout.py outputs. Without --execute
performs CPU-only dry run, does not create run root or launch GPU. Original
real-validation launcher remains unchanged for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from generate_eth_stage1_bbox_preview import PALETTE, LOCATIONS, _frame_index
from train_eth_stage1_bbox import (
    TrainingBlocked, check_synthetic_dataset, inspect_loaded_yolo26s,
    launch_train, read_csv, sha256, validate_checkpoint_filename,
    build_stage1_train_kwargs,
)

def _audit(path:Path)->dict:
    if not path.is_file():
        raise TrainingBlocked('required audit missing: '+str(path))
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        raise TrainingBlocked('invalid JSON audit: '+str(path)) from exc
    if not isinstance(report,dict):
        raise TrainingBlocked('audit must be JSON object')
    return report

def _require_sha(path:Path, expected:str, what:str)->None:
    if not path.is_file() or sha256(path)!=expected:
        raise TrainingBlocked(f'{what} missing or SHA256 mismatch: {path}')

def _check_split_sources(root:Path,report:dict)->tuple[dict,dict]:
    inv={}
    for name in ('train','val','embargo'):
        path=root/f'{name}_inventory.csv'
        _require_sha(path,report.get(f'{name}_inventory_sha256',''),name+' inventory')
        if name=='embargo':
            with path.open(newline='',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
        else:
            rows=read_csv(path)
        for row in rows:
            if (row.get('split'),row.get('source'),row.get('role'))!=('train','eth_main','eth_train'):
                raise TrainingBlocked('non ETH-TRAIN inventory record: '+name)
            if row.get('location') not in LOCATIONS:
                raise TrainingBlocked('unknown scene in inventory')
        inv[name]=rows
    ids={k:{r['instance_id'] for r in rows} for k,rows in inv.items()}
    hashes={k:{r['source_sha256'].lower() for r in rows} for k,rows in inv.items()}
    if any(len(ids[k])!=len(inv[k]) for k in inv):
        raise TrainingBlocked('duplicated original source instance')
    if sum(map(len,ids.values()))!=report.get('original_instances'):
        raise TrainingBlocked('split instance count mismatch')
    for k in inv:
        expected=report.get('embargoed_instances' if k=='embargo' else k+'_unique_instances')
        if len(inv[k])!=expected:
            raise TrainingBlocked('split inventory count mismatch: '+k)
    for a,b in (('train','val'),('train','embargo'),('val','embargo')):
        if ids[a]&ids[b] or hashes[a]&hashes[b]:
            raise TrainingBlocked(f'original source SHA/instance overlap: {a}/{b}')
    if {r['location'] for r in inv['train']}!=set(LOCATIONS) or \
       {r['location'] for r in inv['val']}!=set(LOCATIONS):
        raise TrainingBlocked('train/val missing ETH scenes')
    per_seq=defaultdict(lambda:defaultdict(set))
    for name,rows in inv.items():
        for row in rows:
            per_seq[(row['location'],row['set_dir'])][name].add(_frame_index(row['image']))
    embargo=report.get('embargo_frame_indices',-1)
    if not isinstance(embargo,int) or embargo<0:
        raise TrainingBlocked('invalid frame-index embargo')
    entries=report.get('sequences',[])
    if len(per_seq)!=len(entries):
        raise TrainingBlocked('video sequence count audit mismatch')
    audited={(x['location'],x['set_dir']):x for x in entries}
    if set(audited)!=set(per_seq):
        raise TrainingBlocked('video sequence identities audit mismatch')
    for key,parts in per_seq.items():
        if not parts['train'] or not parts['val']:
            raise TrainingBlocked('empty temporal holdout sequence')
        tr=max(parts['train']); val=min(parts['val'])
        if val-tr<=embargo:
            raise TrainingBlocked('temporal train-val gap violated: '+repr(key))
        if audited[key]['last_train_frame']!=tr or audited[key]['first_val_frame']!=val:
            raise TrainingBlocked('temporal audit boundaries corrupted')
        if any(parts[x]&parts[y] for x,y in (('train','val'),('train','embargo'),('val','embargo'))):
            raise TrainingBlocked('same source frame appears across partitions')
    return inv,hashes

# PNG/label generation writes nine decimal places. Its rounded YOLO center and
# size can place a mathematically valid border box ~5e-10 outside the canvas.
# Tolerate only that quantization error; still reject materially invalid boxes.
_NORMALIZED_LABEL_EPS = 1e-9


def is_valid_normalized_box(cx:float,cy:float,w:float,h:float)->bool:
    if not all(math.isfinite(q) for q in (cx,cy,w,h)):
        return False
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    return (0 <= cx <= 1 and 0 <= cy <= 1
            and cx-w/2 >= -_NORMALIZED_LABEL_EPS
            and cy-h/2 >= -_NORMALIZED_LABEL_EPS
            and cx+w/2 <= 1+_NORMALIZED_LABEL_EPS
            and cy+h/2 <= 1+_NORMALIZED_LABEL_EPS)


def check_synthetic_validation(val_dir:Path, *, expected_unique:int,
                               forbidden_hashes:set[str],
                               verify_output_hashes:bool=True)->dict:
    report=_audit(val_dir/'audit.json')
    expected={'status':'PASS_DATA_GENERATION_NOT_TRAINING','mode':'full',
        'inventory_instances':expected_unique,'unique_instances_covered':expected_unique,
        'generated_images':expected_unique,'full_unique_coverage':True,
        'duplicate_exposures':0,'image_size':[1024,1024],
        'foreground_method':'bbox_crop_no_mask_no_resize','scale':1.0,
        'mask_used':False,'model_trained':False,'optimizer_steps':0}
    for key,value in expected.items():
        if report.get(key)!=value:
            raise TrainingBlocked(f'synthetic val audit mismatch {key}: {report.get(key)!r}')
    rows=read_csv(val_dir/'manifest.csv')
    if len(rows)!=expected_unique:
        raise TrainingBlocked('val manifest row count mismatch')
    if sha256(val_dir/'manifest.csv')!=report.get('manifest_sha256'):
        raise TrainingBlocked('val manifest SHA mismatch')
    ids=set(); stems=set(); hashes=set(); scene_count=Counter()
    for row in rows:
        iid=row.get('instance_id','')
        original_sha=(row.get('source_sha256') or '').lower()
        if not iid or iid in ids or len(original_sha)!=64 or \
           any(c not in '0123456789abcdef' for c in original_sha):
            raise TrainingBlocked('duplicate or invalid original source in synthetic validation')
        ids.add(iid);hashes.add(original_sha)
        if original_sha in forbidden_hashes:
            raise TrainingBlocked('train/validation source SHA overlap: '+original_sha)
        if row.get('foreground_method')!='bbox_crop_no_mask' or row.get('scale')!='1.0':
            raise TrainingBlocked('validation altered original crop pixels or scale')
        if row.get('location') not in LOCATIONS:
            raise TrainingBlocked('invalid synthetic validation scene')
        scene_count[row['location']]+=1
        try:
            color=tuple(json.loads(row.get('background_rgb','')))
        except (TypeError,ValueError) as exc:
            raise TrainingBlocked('invalid synthetic validation RGB background') from exc
        if color not in PALETTE:
            raise TrainingBlocked('validation not using same solid palette')
        image_rel=Path(row.get('image_path',''))
        label_rel=Path(row.get('label_path',''))
        if (image_rel.is_absolute() or label_rel.is_absolute() or
            len(image_rel.parts)!=2 or image_rel.parts[0]!='images' or
            len(label_rel.parts)!=2 or label_rel.parts[0]!='labels' or
            image_rel.stem!=label_rel.stem or image_rel.stem in stems):
            raise TrainingBlocked('invalid synthetic val image/label pathname')
        stems.add(image_rel.stem)
        image=val_dir/image_rel; label=val_dir/label_rel
        if not image.is_file() or not label.is_file():
            raise TrainingBlocked('synthetic val missing image or label')
        if verify_output_hashes:
            _require_sha(image,row.get('image_sha256',''),'synthetic val image')
            _require_sha(label,row.get('output_label_sha256',''),'synthetic val label')
        lines=label.read_text(encoding='utf-8').strip().splitlines()
        if len(lines)!=1:
            raise TrainingBlocked('synthetic val should have exactly one bbox')
        parts=lines[0].split()
        if len(parts)!=5 or parts[0]!='0':
            raise TrainingBlocked('synthetic val label invalid: class / columns')
        try:
            cx,cy,w,h=map(float,parts[1:])
        except ValueError as exc:
            raise TrainingBlocked('synthetic val label has bad coordinates') from exc
        if not is_valid_normalized_box(cx,cy,w,h):
            raise TrainingBlocked('invalid synthetic val BBox normalized coordinates')
    if len(ids)!=expected_unique or dict(sorted(scene_count.items()))!=report.get('by_scene'):
        raise TrainingBlocked('val scene/source coverage mismatch')
    return {'images':len(rows),'source_ids':ids,'source_hashes':hashes,
            'manifest_sha256':sha256(val_dir/'manifest.csv'),
            'audit_sha256':sha256(val_dir/'audit.json')}

def preflight(split_root:Path,model:Path,model_sha256:str,run_root:Path, *,
              expected_original:int=13992,time_bins:int=10,
              verify_output_hashes:bool=True)->dict:
    if run_root.exists():
        raise TrainingBlocked('run root exists; refusing overwrite: '+str(run_root))
    validate_checkpoint_filename(model)
    if (not model.is_file() or len(model_sha256)!=64 or
        sha256(model)!=model_sha256.lower()):
        raise TrainingBlocked('YOLO26s checkpoint path/SHA mismatch')
    split=_audit(split_root/'split_audit.json')
    if (split.get('status')!='PASS_SPLIT_AND_SYNTHETIC_GENERATION_NO_TRAINING' or
        split.get('original_instances')!=expected_original or
        split.get('train_val_source_sha_overlap')!=0 or
        split.get('train_val_source_instance_overlap')!=0 or
        split.get('training_synthetic_mode')!='balanced-full' or
        split.get('validation_synthetic_mode')!='full' or
        split.get('train_and_val_background_policy')!='same_solid_background_palette' or
        split.get('image_size')!=[1024,1024] or
        split.get('time_bins_per_sequence')!=time_bins):
        raise TrainingBlocked('split audit contract mismatch')
    inventory,hashes=_check_split_sources(split_root,split)
    train_dir=split_root/'train_synthetic'
    val_dir=split_root/'val_synthetic'
    _require_sha(train_dir/'audit.json',split.get('train_audit_sha256',''),'train audit')
    _require_sha(val_dir/'audit.json',split.get('val_audit_sha256',''),'validation audit')
    train=check_synthetic_dataset(train_dir,expected_unique=len(inventory['train']),
          bins=time_bins,verify_output_hashes=verify_output_hashes)
    val=check_synthetic_validation(val_dir,expected_unique=len(inventory['val']),
          forbidden_hashes=train['source_hashes'],verify_output_hashes=verify_output_hashes)
    if (train['source_ids']!={r['instance_id'] for r in inventory['train']} or
        val['source_ids']!={r['instance_id'] for r in inventory['val']} or
        train['source_hashes']!=hashes['train'] or
        val['source_hashes']!=hashes['val']):
        raise TrainingBlocked('generated samples do not match source partition inventories')
    if (train['manifest_sha256']!=split.get('train_manifest_sha256') or
        val['manifest_sha256']!=split.get('val_manifest_sha256')):
        raise TrainingBlocked('generated manifest SHA does not match split report')
    return {'status':'PASS_SPLIT_SYNTHETIC_VAL_PREFLIGHT_NOT_TRAINING',
        'stage':'yolo26s_bbox_stage1','val_domain':'synthetic_solid',
        'train_domain':'synthetic_solid','model':str(model),
        'model_sha256':model_sha256.lower(),'model_trained':False,'optimizer_steps':0,
        'source_inventory_sha256':split.get('source_inventory_sha256'),
        'split_audit_sha256':sha256(split_root/'split_audit.json'),
        'train_unique_instances':len(inventory['train']),
        'val_unique_instances':len(inventory['val']),
        'embargoed_instances':len(inventory['embargo']),
        'train_generated_images':train['images'],'val_generated_images':val['images'],
        'source_sha_overlap':0,'train_manifest_sha256':train['manifest_sha256'],
        'val_manifest_sha256':val['manifest_sha256'],
        'temporal_group_isolation':'frame_index_embargo_not_independent_video',
        'independent_real_world_performance':'UNKNOWN',
        'split_root':str(split_root),'train_dir':str(train_dir),
        'val_dir':str(val_dir),'run_root':str(run_root)}

def make_dataset_yaml(split_root:Path,run_root:Path)->Path:
    if run_root.exists():
        raise TrainingBlocked('run root exists; refusing overwrite')
    import yaml
    run_root.mkdir(parents=True)
    content={'path':str(run_root.resolve()),
      'train':str((split_root/'train_synthetic'/'images').resolve()),
      'val':str((split_root/'val_synthetic'/'images').resolve()),
      'names':{0:'shuttlecock'},'nc':1}
    path=run_root/'dataset.yaml'
    path.write_text(yaml.safe_dump(content,sort_keys=False,allow_unicode=True),
                    encoding='utf-8')
    return path

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-root',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-sha256',required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--expected-original',type=int,default=13992)
    p.add_argument('--time-bins',type=int,default=10)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr0',type=float,default=1e-4)
    p.add_argument('--device',default='0')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    try:
        result=preflight(args.split_root,args.model,args.model_sha256,args.run_root,
            expected_original=args.expected_original,time_bins=args.time_bins)
        build_stage1_train_kwargs(Path('pending.yaml'),args.run_root,
            epochs=args.epochs,batch=args.batch,seed=args.seed,
            device=args.device,workers=args.workers,lr0=args.lr0)
        result.update(requested_epochs=args.epochs,batch=args.batch,
            seed=args.seed,lr0=args.lr0,optimizer='AdamW',
            warmup_bias_lr=0.0,execute_requested=args.execute)
        if not args.execute:
            result['status']='DRY_RUN_PASS_NO_TRAINING'
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            return 0
        if args.device!='0':
            raise TrainingBlocked('GPU device 0 is required by this protocol')
        import torch
        if not torch.cuda.is_available():
            raise TrainingBlocked('CUDA unavailable: refuse training')
        import ultralytics
        from ultralytics import YOLO
        trainer=YOLO(str(args.model))
        result.update(cuda_name=torch.cuda.get_device_name(0),
            ultralytics_version=ultralytics.__version__,
            loaded_architecture=inspect_loaded_yolo26s(trainer))
        data_yaml=make_dataset_yaml(args.split_root,args.run_root)
        (args.run_root/'preflight.json').write_text(
            json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'status':'TRAINING_LAUNCH','data_yaml':str(data_yaml),
            'preflight':str(args.run_root/'preflight.json')},ensure_ascii=False),flush=True)
        launch_train(trainer,data_yaml,args.run_root,epochs=args.epochs,
            batch=args.batch,seed=args.seed,device=args.device,
            workers=args.workers,lr0=args.lr0)
        return 0
    except (TrainingBlocked,OSError,ValueError,ImportError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)},ensure_ascii=False),flush=True)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
