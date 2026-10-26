"""ETH all-box inventory adapter reusing the repository's frozen V2 parser/firewall.

All valid-size positives from ETH train are recorded, regardless of pixel size.
The raw image must exist and match its SHA256/dimensions before it is accepted.
"""
from __future__ import annotations
from pathlib import Path
import re

import prepare_eth_real_foregrounds as P


class InventoryBlocked(RuntimeError):
    pass


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
        instances.append({
            'instance_id':instance_id,
            'image':row['image'],
            'label':row['label'],
            'source_sha256':row['sha256'],
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
    excluded=sum(1 for row in rejections if row.get('reason') in {
        'eval_split','eval_set','eval_content','not_positive','non_eth_source','split_missing'
    })
    return {
        'status':'CANDIDATES_PENDING_REVIEW' if instances else 'NO_USABLE_INSTANCES',
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
        'rejections_by_reason':audit['rejections_by_reason'],
        'invalid_box_count':invalid,
        'source_gate':'PASS' if audit['eligible_train_label_boxes']==len(instances)+invalid else 'BLOCKED_RECONCILIATION',
        'legacy_min_short_long_defaults_unchanged':True,
    }
