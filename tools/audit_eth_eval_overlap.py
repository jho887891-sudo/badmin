#!/usr/bin/env python3
"""Read-only audit for training/evaluation SHA overlap and contradictory ETH labels.

This tool never edits the source manifests or produces a claimed-independent
evaluation set. It emits explicit overlap/conflict evidence for human decisions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path


class EvalAuditBlocked(RuntimeError):
    pass


def _read(path: str | Path) -> list[dict]:
    p=Path(path)
    if not p.is_file() or p.stat().st_size==0:
        raise EvalAuditBlocked('missing_or_empty_csv:'+str(p))
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:
            raise EvalAuditBlocked('csv_missing_header:'+str(p))
        return list(reader)


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _norm_sha(row:dict,keys:tuple[str,...])->str:
    for key in keys:
        value=str(row.get(key) or '').strip().lower()
        if value:
            return value
    return ''


def audit_eval_overlap(train_manifest:str|Path,frozen_eval_index:str|Path,
                       val_manifest:str|Path|None=None)->dict:
    train=_read(train_manifest);frozen=_read(frozen_eval_index)
    val=_read(val_manifest) if val_manifest else []
    required_train={'sha256','role','source','split','is_positive','is_negative','n_boxes','image'}
    required_eval={'sha256','role','origin'}
    missing_train=sorted(required_train-set(train[0]))
    missing_eval=sorted(required_eval-set(frozen[0]))
    if missing_train or missing_eval:
        raise EvalAuditBlocked('schema_missing:train='+','.join(missing_train)+';eval='+','.join(missing_eval))
    train_by_sha=defaultdict(list);eval_by_sha=defaultdict(list)
    for row in train:
        sha=_norm_sha(row,('sha256',))
        if sha:train_by_sha[sha].append(row)
    for row in frozen:
        sha=_norm_sha(row,('sha256',))
        if sha:eval_by_sha[sha].append(row)
    overlap=set(train_by_sha)&set(eval_by_sha)
    overlap_train_rows=[row for sha in sorted(overlap) for row in train_by_sha[sha]]
    overlap_eval_rows=[row for sha in sorted(overlap) for row in eval_by_sha[sha]]
    contradictions=[]
    for row in train:
        role=str(row.get('role') or '').strip()
        is_pos=str(row.get('is_positive') or '').strip().lower()
        is_neg=str(row.get('is_negative') or '').strip().lower()
        try:nbox=int(float(row.get('n_boxes') or 0))
        except ValueError:nbox=-1
        if role=='official_positives' and (is_pos not in {'true','1','yes'} or is_neg in {'true','1','yes'} or nbox<=0):
            contradictions.append({'image':row.get('image'),'sha256':row.get('sha256'),
                                   'role':role,'is_positive':row.get('is_positive'),
                                   'is_negative':row.get('is_negative'),'n_boxes':row.get('n_boxes'),
                                   'reason':'official_positive_role_conflicts_with_label_flags'})
    overlap_by_role=Counter(str(r.get('role') or 'UNKNOWN') for r in overlap_train_rows)
    overlap_by_source=Counter(str(r.get('source') or 'UNKNOWN') for r in overlap_train_rows)
    overlap_by_origin=Counter(str(r.get('origin') or 'UNKNOWN') for r in overlap_eval_rows)
    overlap_by_sha=Counter(_norm_sha(r,('sha256',)) for r in overlap_train_rows)
    repeated_sha={sha:count for sha,count in overlap_by_sha.items() if count>1}
    val_report={'provided':bool(val),'rows':len(val),'split_values':dict(Counter(str(r.get('split') or '') for r in val)),
                'in_official_training_true':sum(str(r.get('in_official_training') or '').strip().lower() in {'true','1','yes'} for r in val),
                'images_train_path_tokens':sum('/images/train/' in str(r.get('image') or '').replace(chr(92),'/').lower() for r in val),
                'locations':dict(Counter(str(r.get('location') or 'UNKNOWN') for r in val)),
                'train_manifest_location_intersection':sorted(
                    {str(r.get('location') or '') for r in val}&{str(r.get('location') or '') for r in train}
                )}
    return {
        'status':'AUDIT_ONLY_REVIEW_REQUIRED',
        'train_manifest':str(train_manifest),'frozen_eval_index':str(frozen_eval_index),
        'val_manifest':str(val_manifest) if val_manifest else None,
        'row_counts':{'train_manifest':len(train),'frozen_eval_index':len(frozen),'val_manifest':len(val)},
        'sha_overlap':{
            'unique_sha256':len(overlap),'training_rows':len(overlap_train_rows),'eval_rows':len(overlap_eval_rows),
            'train_rows_by_role':dict(overlap_by_role),'train_rows_by_source':dict(overlap_by_source),
            'eval_rows_by_origin':dict(overlap_by_origin),'repeated_training_sha256':repeated_sha,
        },
        'contradictory_official_positive_rows':contradictions,
        'val_semantics':val_report,
        'policy':{
            'source_manifests_modified':False,
            'new_independent_eval_set_created':False,
            'recall_or_false_positive_metrics_authorized':False,
            'note':'A clean evaluation index requires an explicitly reviewed origin allowlist and zero train/eval content overlap.'
        }
    }


def _write_csv(path:Path,rows:list[dict])->None:
    if path.exists():raise FileExistsError('refusing to overwrite '+str(path))
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    if not keys:keys=['status']
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def run(train_manifest,frozen_eval_index,output_dir,val_manifest=None)->dict:
    out=Path(output_dir)
    if out.exists():raise EvalAuditBlocked('output_directory_already_exists')
    report=audit_eval_overlap(train_manifest,frozen_eval_index,val_manifest)
    train=_read(train_manifest);frozen=_read(frozen_eval_index)
    frozen_sha={_norm_sha(r,('sha256',)) for r in frozen}
    overlap_rows=[r for r in train if _norm_sha(r,('sha256',)) in frozen_sha]
    contradictions=report['contradictory_official_positive_rows']
    out.mkdir(parents=True)
    _write_csv(out/'training_rows_overlapping_frozen_eval.csv',overlap_rows)
    _write_csv(out/'contradictory_official_positive_rows.csv',contradictions)
    report['input_sha256']={
        'train_manifest':_sha(Path(train_manifest)),
        'frozen_eval_index':_sha(Path(frozen_eval_index)),
        'val_manifest':_sha(Path(val_manifest)) if val_manifest else None,
    }
    report['outputs']={'overlap_rows':'training_rows_overlapping_frozen_eval.csv',
                       'label_conflicts':'contradictory_official_positive_rows.csv',
                       'report':'audit.json'}
    with (out/'audit.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write('\\n')
    return report


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest',required=True,type=Path)
    p.add_argument('--frozen-eval-index',required=True,type=Path)
    p.add_argument('--val-manifest',type=Path,default=None)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args(argv)
    try:report=run(args.train_manifest,args.frozen_eval_index,args.output_dir,args.val_manifest)
    except (EvalAuditBlocked,FileExistsError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)},indent=2));return 2
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
