"""Enumerate every authorized ETH train bbox, including unsegmentable tiny boxes.

Does NOT determine mask quality. Frozen eval content is refused by hash/path.
"""
from __future__ import annotations
import csv
import hashlib
import math
from pathlib import Path

class InventoryBlocked(RuntimeError):
    pass


def _rows(path):
    p=Path(path)
    if not p.is_file():raise InventoryBlocked('manifest not found: '+str(p))
    with p.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _box_pixel(label, width, height):
    parts=label.split()
    if len(parts)!=5 or parts[0]!='0':
        raise InventoryBlocked('expected YOLO class 0 normalized cx cy w h')
    try: cx,cy,w,h=map(float,parts[1:])
    except ValueError as e:raise InventoryBlocked('invalid YOLO numeric bbox') from e
    if not all(math.isfinite(x) for x in (cx,cy,w,h)) or w<=0 or h<=0:
        raise InventoryBlocked('nonfinite or empty YOLO bbox')
    x1=round((cx-w/2)*width);y1=round((cy-h/2)*height)
    x2=round((cx+w/2)*width);y2=round((cy+h/2)*height)
    return (x1,y1,x2,y2)


def _label_path(row,label_root):
    raw=str(row.get('label') or row.get('abs_label_path') or '').strip()
    if raw and Path(raw).is_file():return Path(raw)
    if label_root and raw and (Path(label_root)/Path(raw).name).is_file():
        return Path(label_root)/Path(raw).name
    raise InventoryBlocked('label missing: '+raw)


def enumerate_eth_train_instances(v2_manifest: str | Path, *, label_root: str | Path | None = None,
                                   frozen_eval_manifests: tuple[str | Path, ...] = ()) -> dict:
    rows=_rows(v2_manifest)
    if not rows:raise InventoryBlocked('empty ETH manifest')
    frozen_hashes=set();frozen_paths=set();frozen_stems=set()
    for file in frozen_eval_manifests:
        for row in _rows(file):
            h=(row.get('sha256') or '').strip();p=(row.get('image') or row.get('path') or '').strip()
            if h:frozen_hashes.add(h)
            if p:
                frozen_paths.add(p);frozen_stems.add(Path(p).stem)
    instances=[];rejections=[]; seen=set(); train_count=0; excluded=0
    for row in rows:
        image=(row.get('image') or row.get('abs_path') or '').strip()
        split=(row.get('split') or '').strip().lower()
        source=(row.get('source') or '').strip().lower()
        digest=(row.get('sha256') or '').strip()
        if split!='train':
            excluded+=1;rejections.append({'image':image,'reason':'non_train_split'});continue
        if not source.startswith('eth'):
            rejections.append({'image':image,'reason':'non_eth_source'});continue
        if image in frozen_paths or Path(image).stem in frozen_stems or (digest and digest in frozen_hashes):
            rejections.append({'image':image,'reason':'frozen_eval_match'});continue
        if not image or not Path(image).is_file():
            raise InventoryBlocked('image not mounted: '+image)
        try:
            W=int(row['width']);H=int(row['height'])
        except (ValueError,KeyError,TypeError) as e:
            raise InventoryBlocked('missing/invalid image dimensions: '+image) from e
        if W<=0 or H<=0:raise InventoryBlocked('invalid dimensions: '+image)
        label_path=_label_path(row,label_root)
        box_lines=[s.strip() for s in label_path.read_text(encoding='utf-8').splitlines() if s.strip()]
        if str(row.get('n_boxes') or '').strip() and int(row['n_boxes'])!=len(box_lines):
            raise InventoryBlocked('n_boxes disagrees with label: '+str(label_path))
        for idx,s in enumerate(box_lines):
            train_count+=1
            bbox=_box_pixel(s,W,H)
            x1,y1,x2,y2=bbox
            inst_id=f'{digest or hashlib.sha256(image.encode()).hexdigest()}#{idx}'
            if inst_id in seen:
                rejections.append({'instance_id':inst_id,'reason':'duplicate_source_instance'});continue
            seen.add(inst_id)
            payload={'instance_id':inst_id,'image':image,'source_sha256':digest,
                     'label':str(label_path),'split':'train','source':source,
                     'location':row.get('location',''),'bbox_xyxy':list(bbox),
                     'bbox_wh':[x2-x1,y2-y1], 'bbox_short':min(x2-x1,y2-y1),
                     'bbox_long':max(x2-x1,y2-y1),'status':'PENDING_REVIEW',
                     'reason':'mask_not_reviewed'}
            if x2<=x1 or y2<=y1 or x1<0 or y1<0 or x2>W or y2>H:
                rejections.append({**payload,'reason':'quantized_empty_bbox' if x2<=x1 or y2<=y1 else 'bbox_out_of_bounds','status':'UNUSABLE'})
                continue
            instances.append(payload)
    invalid=sum(1 for r in rejections if r.get('status')=='UNUSABLE')
    dedup=sum(1 for r in rejections if r.get('reason')=='duplicate_source_instance')
    return {'instances':instances,'rejections':rejections,
            'counts':{'total_manifest_rows':len(rows),'train_label_boxes_total':train_count,
                      'usable_instances':len(instances),'unusable_instances':invalid+dedup,
                      'excluded_non_train_rows':excluded},
            'status':'CANDIDATES_PENDING_REVIEW' if instances else 'NO_USABLE_INSTANCES'}
