"""Stage 4: verified real-environment alpha Copy-Paste.

Requires a human-reviewed empty training ROI. Never generates foreground
from evaluation data, even if it is renamed or used only as a background.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


class EnvironmentRejected(ValueError):
    pass


def validate_real_background(record: dict, frozen:list[dict]) -> dict:
    failures=[]
    if record.get('source') not in {'eth_train_background','real_venue_train','approved_background'}:
        failures.append('invalid_background_source')
    if record.get('split')!='train':failures.append('non_train_split')
    if record.get('qa_status')!='APPROVED' or record.get('no_shuttle_reviewed') is not True or not record.get('reviewer_id'):
        failures.append('human_empty_roi_review_required')
    img=np.asarray(record.get('image'))
    roi=record.get('clean_roi')
    if img.ndim!=3 or img.shape[2]!=3 or not isinstance(roi,(tuple,list)) or len(roi)!=4:
        failures.append('invalid_background_or_roi')
    else:
        x0,y0,x1,y1=map(int,roi)
        if x0<0 or y0<0 or x1>img.shape[1] or y1>img.shape[0] or x1<=x0 or y1<=y0:
            failures.append('invalid_background_or_roi')
    sha=record.get('source_sha256')
    if not sha:failures.append('background_sha_missing')
    if any(sha and sha==row.get('source_sha256') or record.get('path') and record.get('path')==row.get('path') for row in frozen):
        failures.append('frozen_source_or_parent_sha')
    return {'status':'REJECTED' if failures else 'PASS','reasons':failures}


def make_stage4_sample(foreground:dict,background:dict,placement:dict,seed:int, *, frozen:list[dict] | None=None) -> dict:
    validation=validate_real_background(background,frozen or [])
    if validation['status']!='PASS':raise EnvironmentRejected('background rejected: '+','.join(validation['reasons']))
    if foreground.get('qa_status')!='APPROVED' or foreground.get('source')!='eth_train' or not foreground.get('source_sha256'):
        raise EnvironmentRejected('foreground requires approved ETH train source and hash')
    fg=np.asarray(foreground['rgb'],dtype=np.uint8)
    alpha=np.asarray(foreground['alpha'],dtype=np.float32)
    if fg.ndim!=3 or fg.shape[2]!=3 or alpha.shape!=fg.shape[:2] or alpha.min()<0 or alpha.max()>1:
        raise EnvironmentRejected('invalid foreground RGB/alpha')
    scale=float(placement.get('scale',1.0))
    if not 0<scale<=1.0:raise EnvironmentRejected('upscaling real foreground prohibited')
    if scale!=1.0:
        tile=np.dstack([fg,np.rint(alpha*255).astype(np.uint8)])
        w=max(1,round(fg.shape[1]*scale));h=max(1,round(fg.shape[0]*scale))
        tile=np.asarray(Image.fromarray(tile,'RGBA').resize((w,h),Image.Resampling.BICUBIC))
        fg=tile[:,:,:3];alpha=tile[:,:,3].astype(np.float32)/255
    out=np.asarray(background['image'],dtype=np.uint8).copy()
    H,W=out.shape[:2]
    rh,rw=fg.shape[:2]
    rx0,ry0,rx1,ry1=map(int,background['clean_roi'])
    cx=float(placement.get('center_x',(rx0+rx1)/2));cy=float(placement.get('center_y',(ry0+ry1)/2))
    left=round(cx-rw/2);top=round(cy-rh/2)
    x0,y0=max(left,rx0),max(top,ry0)
    x1,y1=min(left+rw,rx1),min(top+rh,ry1)
    visible=np.zeros((H,W),dtype=np.float32)
    if x1>x0 and y1>y0:
        fx0,fy0=x0-left,y0-top
        a=alpha[fy0:fy0+y1-y0,fx0:fx0+x1-x0]
        rgb=fg[fy0:fy0+y1-y0,fx0:fx0+x1-x0].astype(np.float32)
        orig=out[y0:y1,x0:x1].astype(np.float32)
        out[y0:y1,x0:x1]=np.clip(np.rint(orig*(1-a[:,:,None])+rgb*a[:,:,None]),0,255).astype(np.uint8)
        visible[y0:y1,x0:x1]=a
    yy,xx=np.nonzero(visible>=.25)
    if len(xx)<4:raise EnvironmentRejected('target invisible after ROI crop')
    bx=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
    w=bx[2]-bx[0];h=bx[3]-bx[1]
    if w*h<=0:raise EnvironmentRejected('empty bbox')
    label=f'0 {(bx[0]+bx[2])/(2*W):.6f} {(bx[1]+bx[3])/(2*H):.6f} {w/W:.6f} {h/H:.6f}\n'
    return {'image':out,'visible_alpha':visible,'label':label,
            'metadata':{'foreground_id':foreground.get('sample_id'),
                        'foreground_sha256':foreground['source_sha256'],
                        'background_id':background.get('background_id'),
                        'background_sha256':background['source_sha256'],
                        'environment_type':background.get('environment_type','unknown'),
                        'clean_roi':list(background['clean_roi']),
                        'bbox_xyxy':bx,'scale':scale,'seed':int(seed)}}
