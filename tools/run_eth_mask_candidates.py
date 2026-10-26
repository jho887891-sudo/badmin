#!/usr/bin/env python3
"""Create source-verified, human-review-only Mask proposals for every inventory row.

No image is approved automatically. Tiny or unrecoverable instances remain in
review.csv with explicit reasons. Output directory must be new.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from propose_eth_instance_masks import propose_instance_mask


class MaskRunBlocked(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _resolve(raw: str, root: Path | None) -> Path:
    p=Path(raw)
    if p.is_absolute() or root is None:
        return p
    return root/p


def _bbox(raw: str, width: int, height: int) -> tuple[int,int,int,int]:
    try:
        b=json.loads(raw)
        if not isinstance(b,list) or len(b)!=4:
            raise ValueError
        x0,y0,x1,y1=map(int,b)
    except Exception as exc:
        raise MaskRunBlocked('invalid_bbox_xyxy') from exc
    if x0<0 or y0<0 or x1>width or y1>height or x1<=x0 or y1<=y0:
        raise MaskRunBlocked('bbox_outside_image_or_empty')
    return x0,y0,x1,y1


def _review_tile(image: np.ndarray, alpha: np.ndarray | None, bbox, instance_id: str,
                 status: str, reason: str, short: int, long_side: int,
                 tile_size=(220,132)) -> Image.Image:
    tw,th=tile_size
    tile=Image.new('RGB',(tw,th),(245,245,245))
    draw=ImageDraw.Draw(tile)
    draw.text((5,3),f'{instance_id[:14]} | {status}',fill=(15,15,15))
    draw.text((5,17),f'{short}x{long_side}px | {reason[:26]}',fill=(50,50,50))
    x0,y0,x1,y1=bbox
    pad=max(2,min(10,int(round(max(x1-x0,y1-y0)*.25))))
    cx0=max(0,x0-pad);cy0=max(0,y0-pad);cx1=min(image.shape[1],x1+pad);cy1=min(image.shape[0],y1+pad)
    crop=Image.fromarray(image[cy0:cy1,cx0:cx1])
    cw,ch=crop.size
    scale=min(66/max(cw,1),66/max(ch,1),12)
    nw=max(1,int(round(cw*scale)));nh=max(1,int(round(ch*scale)))
    crop=crop.resize((nw,nh),Image.Resampling.NEAREST)
    tile.paste(crop,(5,37))
    if alpha is None:
        mask_view=Image.new('RGB',(nw,nh),(80,80,80))
        ImageDraw.Draw(mask_view).text((2,2),'NO MASK',fill=(255,255,255))
    else:
        local=alpha[cy0:cy1,cx0:cx1]
        mask_view=Image.fromarray(local).resize((nw,nh),Image.Resampling.NEAREST).convert('L').convert('RGB')
        pix=np.asarray(mask_view).copy()
        pix[:,:,0]=np.maximum(pix[:,:,0],np.asarray(mask_view)[:,:,0])
        mask_view=Image.fromarray(pix)
    tile.paste(mask_view,(tw//2+5,37))
    draw.text((5,th-15),'SOURCE',fill=(20,20,20))
    draw.text((tw//2+5,th-15),'MASK',fill=(20,20,20))
    return tile


def run_mask_candidates(inventory_csv: str | Path, output_dir: str | Path, *,
                        image_root: str | Path | None = None, sheet_size: int = 64) -> dict:
    inv=Path(inventory_csv)
    out=Path(output_dir)
    if not inv.is_file() or inv.stat().st_size==0:
        raise MaskRunBlocked('inventory_missing_or_empty')
    if out.exists():
        raise MaskRunBlocked('output_directory_already_exists')
    if sheet_size<1 or sheet_size>256:
        raise MaskRunBlocked('sheet_size_out_of_range')
    with inv.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'instance_id','image','label','source_sha256','label_sha256','split','source','role','bbox_xyxy'}
        if not required.issubset(reader.fieldnames or []):
            raise MaskRunBlocked('inventory_schema_missing_fields')
        rows=list(reader)
    if not rows:
        raise MaskRunBlocked('inventory_has_no_rows')
    # Create outputs only after input contract checks.
    out.mkdir(parents=True)
    mask_dir=out/'masks';preview_dir=out/'review_sheets'
    mask_dir.mkdir();preview_dir.mkdir()
    root=Path(image_root) if image_root else None
    records=[];review_rows=[];tiles=[];failures=Counter();size_bands=Counter()
    processed_ids=set()
    integrity_failures=0
    for idx,row in enumerate(rows):
        iid=str(row.get('instance_id') or '')
        status='UNRECOVERABLE';reason='not_processed';mask_rel='';quality_flags=[]
        try:
            if not iid or iid in processed_ids:
                raise MaskRunBlocked('duplicate_or_missing_instance_id')
            processed_ids.add(iid)
            if row.get('role')!='eth_train' or row.get('split')!='train' or row.get('source')!='eth_main':
                raise MaskRunBlocked('invalid_train_source_role_or_split')
            image_path=_resolve(row['image'],root)
            label_path=_resolve(row['label'],root)
            if not image_path.is_file():
                raise MaskRunBlocked('source_image_missing')
            if not label_path.is_file():
                raise MaskRunBlocked('source_label_missing')
            source_sha=sha256_file(image_path)
            if source_sha.lower()!=str(row['source_sha256']).lower():
                raise MaskRunBlocked('source_image_sha256_mismatch')
            label_sha=sha256_file(label_path)
            if label_sha.lower()!=str(row['label_sha256']).lower():
                raise MaskRunBlocked('source_label_sha256_mismatch')
            try:
                with Image.open(image_path) as im:
                    im.verify()
                image=np.asarray(Image.open(image_path).convert('RGB'))
            except Exception as exc:
                raise MaskRunBlocked('source_image_unreadable:'+type(exc).__name__) from exc
            h,w=image.shape[:2]
            bbox=_bbox(row['bbox_xyxy'],w,h)
            x0,y0,x1,y1=bbox;bw=x1-x0;bh=y1-y0
            short=min(bw,bh);long_side=max(bw,bh)
            size_bands['short_lt_10']+=int(short<10)
            size_bands['long_lt_24']+=int(long_side<24)
            size_bands['bbox_area_lt_64']+=int(bw*bh<64)
            # Crop to a local region to avoid running GrabCut over a full video frame.
            margin=max(4,int(math.ceil(max(bw,bh)*0.5)))
            rx0=max(0,x0-margin);ry0=max(0,y0-margin)
            rx1=min(w,x1+margin);ry1=min(h,y1+margin)
            roi=image[ry0:ry1,rx0:rx1]
            local_bbox=(x0-rx0,y0-ry0,x1-rx0,y1-ry0)
            proposed=propose_instance_mask(roi,local_bbox,method='grabcut')
            if proposed['candidate_alpha'] is not None:
                local_alpha=np.asarray(proposed['candidate_alpha'],dtype=np.uint8)
                alpha=np.zeros((h,w),dtype=np.uint8)
                alpha[ry0:ry1,rx0:rx1]=local_alpha
                mask_name=iid.replace('#','_')+'.png'
                Image.fromarray(alpha).save(mask_dir/mask_name,format='PNG')
                mask_rel='masks/'+mask_name
                status='PENDING_REVIEW'
                reason='human_review_required'
                support=alpha>0
                in_box=support[y0:y1,x0:x1]
                if in_box.size and float(in_box.mean())>=.98:
                    quality_flags.append('rectangle_like_mask')
                if in_box.size and not in_box.any():
                    quality_flags.append('mask_has_no_support_inside_bbox')
                if (x0==0 or y0==0 or x1==w or y1==h):
                    quality_flags.append('bbox_touches_image_edge')
                tiles.append(_review_tile(image,alpha,bbox,iid,status,reason,short,long_side))
            else:
                status='UNRECOVERABLE'
                reason=str(proposed.get('reason') or 'mask_unrecoverable')
                alpha=None
                tiles.append(_review_tile(image,None,bbox,iid,status,reason,short,long_side))
        except MaskRunBlocked as exc:
            status='BLOCKED_SOURCE_INTEGRITY'
            reason=str(exc)
            integrity_failures+=1
            failures[reason]+=1
            alpha=None
            # Keep the row in the review ledger, but never draw/approve a mask from unverified bytes.
            try:
                image_path=_resolve(row.get('image',''),root)
                if image_path.is_file():
                    image=np.asarray(Image.open(image_path).convert('RGB'))
                    h,w=image.shape[:2]
                    bbox=_bbox(row.get('bbox_xyxy',''),w,h)
                    short=min(bbox[2]-bbox[0],bbox[3]-bbox[1])
                    long_side=max(bbox[2]-bbox[0],bbox[3]-bbox[1])
                    tiles.append(_review_tile(image,None,bbox,iid,status,reason,short,long_side))
                else:
                    tiles.append(Image.new('RGB',(220,132),(200,180,180)))
            except Exception:
                tiles.append(Image.new('RGB',(220,132),(200,180,180)))
        except Exception as exc:
            status='UNRECOVERABLE'
            reason='processing_error:'+type(exc).__name__+':'+str(exc)[:120]
            failures[reason]+=1
            alpha=None
            tiles.append(Image.new('RGB',(220,132),(200,180,180)))
        failures[status]+=1
        record={
            'instance_id':iid,'image':row.get('image'),'label':row.get('label'),
            'source_sha256':row.get('source_sha256'),'label_sha256':row.get('label_sha256'),
            'bbox_xyxy':row.get('bbox_xyxy'),'status':status,'reason':reason,
            'candidate_mask':mask_rel,'quality_flags':quality_flags,
        }
        records.append(record)
        review_rows.append({
            'instance_id':iid,'image':row.get('image'),'bbox_xyxy':row.get('bbox_xyxy'),
            'bbox_short':row.get('bbox_short',''),'bbox_long':row.get('bbox_long',''),
            'source_sha256':row.get('source_sha256'),'candidate_mask':mask_rel,
            'candidate_status':status,'candidate_reason':reason,
            'quality_flags':json.dumps(quality_flags,separators=(',',':')),
            'reviewed':'','reviewer_id':'','review_method':'','decision':'','notes':'',
        })
        if (idx+1)%sheet_size==0 or idx+1==len(rows):
            batch=tiles[-min(sheet_size,len(tiles)):]
            cols=8;tile_w,tile_h=220,132
            sheet=Image.new('RGB',(cols*tile_w,math.ceil(len(batch)/cols)*tile_h),(230,230,230))
            for j,tile in enumerate(batch):
                sheet.paste(tile,((j%cols)*tile_w,(j//cols)*tile_h))
            sheet.save(preview_dir/f'review_{(idx//sheet_size)+1:04d}.jpg',quality=88)
    with (out/'candidate_manifest.jsonl').open('x',encoding='utf-8') as f:
        for rec in records:f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\\n')
    review_fields=list(review_rows[0].keys())
    with (out/'review.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=review_fields);writer.writeheader();writer.writerows(review_rows)
    pending=sum(r['status']=='PENDING_REVIEW' for r in records)
    unrecoverable=sum(r['status']=='UNRECOVERABLE' for r in records)
    blocked=sum(r['status']=='BLOCKED_SOURCE_INTEGRITY' for r in records)
    report={
        'status':'BLOCKED_SOURCE_INTEGRITY' if integrity_failures else 'PENDING_HUMAN_REVIEW',
        'inventory_path':str(inv),'inventory_sha256':sha256_file(inv),
        'input_rows':len(rows),'unique_instance_ids':len(processed_ids),
        'candidate_masks':pending,'unrecoverable':unrecoverable,'source_integrity_failures':blocked,
        'approved_masks':0,'training_permitted':False,'optimizer_steps':0,
        'size_counts':dict(size_bands),'counts_by_status':dict(Counter(r['status'] for r in records)),
        'failure_reasons':dict(failures),'outputs':{
            'candidate_manifest':'candidate_manifest.jsonl','review_csv':'review.csv',
            'mask_dir':'masks/','review_sheets':'review_sheets/'},
        'notes':['All masks are proposals, never auto-approved.',
                 'Every inventory row is retained, including tiny and unrecoverable instances.',
                 'Source hash or label hash mismatch blocks training use.',
                 'Review CSV decisions are intentionally blank.']
    }
    with (out/'audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',required=True,type=Path)
    parser.add_argument('--image-root',type=Path,default=None)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--sheet-size',type=int,default=64)
    args=parser.parse_args(argv)
    try:
        report=run_mask_candidates(args.inventory,args.output_dir,image_root=args.image_root,sheet_size=args.sheet_size)
    except MaskRunBlocked as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)},indent=2));return 2
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 2 if report['status']=='BLOCKED_SOURCE_INTEGRITY' else 0


if __name__=='__main__':
    raise SystemExit(main())
