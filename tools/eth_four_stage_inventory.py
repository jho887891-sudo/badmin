"""ETH all-box inventory adapter and executable preflight using the frozen V2 parser."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import prepare_eth_real_foregrounds as P


class InventoryBlocked(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def enumerate_eth_train_instances(v2_manifest: str | Path, *, label_root: str | Path | None = None,
                                  frozen_eval_manifests: tuple[str | Path, ...] = ()) -> dict:
    try:
        audit=P.audit_eth_train_instances(
            v2_manifest, min_short=1, min_long=1, label_root=label_root,
            frozen_eval_manifests=frozen_eval_manifests,
            include_all_sizes=True, require_image_files=True,
        )
    except P.EthForegroundBlocked as exc:
        raise InventoryBlocked(str(exc)) from exc

    instances=[]
    rejections=list(audit['rejections'])
    seen=set()
    duplicate_count=0
    for row in audit['instances']:
        instance_id=str(row['instance_id'])
        if not re.fullmatch(r'[0-9a-f]{64}#\d+',instance_id):
            raise InventoryBlocked('all-box inventory instance id must use actual source SHA256 plus bbox index')
        if instance_id in seen:
            duplicate_count+=1
            rejections.append({'image':row['image'],'instance_id':instance_id,
                               'reason':'duplicate_source_instance',
                               'detail':'same source bytes and bbox index appeared more than once'})
            continue
        seen.add(instance_id)
        label_path=Path(row['label'])
        instances.append({
            'instance_id':instance_id,
            'image':row['image'],
            'label':row['label'],
            'source_sha256':row['sha256'],
            'label_sha256':_sha256(label_path),
            'split':row['split'],
            'source':row['source'],
            'role':'eth_train',
            'location':row['location'],
            'set_dir':row['set_dir'],
            'bbox_xyxy':row['bbox_xyxy'],
            'bbox_wh':row['bbox_wh'],
            'bbox_short':row['bbox_short'],
            'bbox_long':row['bbox_long'],
            'status':'PENDING_REVIEW',
            'reason':'mask_not_reviewed',
        })

    invalid=int(audit['invalid_box_count'])+duplicate_count
    by_reason={}
    for item in rejections:
        reason=str(item.get('reason') or 'unknown')
        by_reason[reason]=by_reason.get(reason,0)+1
    excluded=sum(1 for row in rejections if row.get('reason') in {
        'eval_split','eval_set','eval_content','not_positive','non_eth_source','split_missing'
    })
    reconciles=(audit['eligible_train_label_boxes']==len(instances)+invalid)
    return {
        'status':'CANDIDATES_PENDING_REVIEW' if instances and reconciles else
                 ('NO_USABLE_INSTANCES' if not instances else 'BLOCKED_RECONCILIATION'),
        'manifest':str(v2_manifest),
        'rows':audit['rows'],
        'counts':{
            'total_manifest_rows':audit['rows'],
            'train_label_boxes_total':audit['eligible_train_label_boxes'],
            'usable_instances':len(instances),
            'unusable_instances':invalid,
            'excluded_non_train_rows':excluded,
        },
        'instances':instances,
        'rejections':rejections,
        'rejections_by_reason':by_reason,
        'invalid_box_count':invalid,
        'source_gate':'PASS' if reconciles else 'BLOCKED_RECONCILIATION',
        'legacy_min_short_long_defaults_unchanged':True,
    }


def write_inventory_csv(instances:list[dict],path:str|Path)->str:
    dest=Path(path)
    if dest.exists():raise FileExistsError('refusing to overwrite inventory: '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True)
    fields=['instance_id','image','label','source_sha256','label_sha256','split','source','role',
            'location','set_dir','bbox_xyxy','bbox_wh','bbox_short','bbox_long','status','reason']
    with dest.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in instances:
            item=dict(row)
            for key in ('bbox_xyxy','bbox_wh'):
                item[key]=json.dumps(item[key],separators=(',',':'))
            writer.writerow({k:item.get(k,'') for k in fields})
    return _sha256(dest)


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description='Inventory all ETH TRAIN positive boxes with the frozen V2 source firewall.')
    parser.add_argument('--manifest',type=Path,required=True,help='non-empty frozen V2 ETH train manifest')
    parser.add_argument('--label-root',type=Path,default=None,help='local mirror if manifest label paths are not directly mounted')
    parser.add_argument('--frozen-eval-manifest',type=Path,action='append',required=True,
                        help='frozen evaluation manifest; repeat for every frozen eval index')
    parser.add_argument('--out-inventory',type=Path,required=True)
    parser.add_argument('--out-audit',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out_inventory.resolve()==args.out_audit.resolve():
        print(json.dumps({'status':'BLOCKED_OUTPUT_PATH_COLLISION'}));return 2
    if args.out_inventory.exists() or args.out_audit.exists():
        print(json.dumps({'status':'BLOCKED_OUTPUT_EXISTS',
                          'out_inventory_exists':args.out_inventory.exists(),
                          'out_audit_exists':args.out_audit.exists()},indent=2));return 2
    try:
        report=enumerate_eth_train_instances(args.manifest,label_root=args.label_root,
                                             frozen_eval_manifests=tuple(args.frozen_eval_manifest))
    except InventoryBlocked as exc:
        print(json.dumps({'status':'BLOCKED_DATA_INVENTORY','error':str(exc)},indent=2));return 2
    if report['source_gate']!='PASS' or report['status']!='CANDIDATES_PENDING_REVIEW':
        print(json.dumps({k:v for k,v in report.items() if k!='instances'},indent=2));return 1
    try:
        inventory_sha=write_inventory_csv(report['instances'],args.out_inventory)
        summary={k:v for k,v in report.items() if k not in {'instances','rejections'}}
        summary['rejections']=report['rejections']
        summary['inventory_csv']=str(args.out_inventory)
        summary['inventory_csv_sha256']=inventory_sha
        summary['approved_masks']=0
        summary['next_gate']='HUMAN_MASK_REVIEW_REQUIRED'
        args.out_audit.parent.mkdir(parents=True,exist_ok=True)
        with args.out_audit.open('x',encoding='utf-8') as f:
            json.dump(summary,f,indent=2,ensure_ascii=False)
            f.write('\n')
    except FileExistsError as exc:
        print(json.dumps({'status':'BLOCKED_OUTPUT_EXISTS','error':str(exc)},indent=2));return 2
    print(json.dumps({'status':report['status'],'counts':report['counts'],
                      'source_gate':report['source_gate'],'inventory_csv':str(args.out_inventory),
                      'inventory_csv_sha256':inventory_sha,'audit_json':str(args.out_audit),
                      'next_gate':'HUMAN_MASK_REVIEW_REQUIRED'},indent=2,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
