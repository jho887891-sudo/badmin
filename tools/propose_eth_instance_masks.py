"""Box-guided, **unreviewed** foreground mask proposal.

A proposal is not a human-approved instance mask. Tiny/unrecoverable cases
stay traceable and may use their original ETH train frame during Stage 3.
"""
from __future__ import annotations
import numpy as np


def propose_instance_mask(image: np.ndarray, bbox_xyxy: tuple[int,int,int,int], *, method: str = 'grabcut') -> dict:
    im=np.asarray(image)
    x0,y0,x1,y1=map(int,bbox_xyxy)
    if im.ndim!=3 or im.shape[-1]!=3 or min(x0,y0)<0 or x1>im.shape[1] or y1>im.shape[0] or x1<=x0 or y1<=y0:
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'invalid_bbox_or_image','method':method}
    if min(x1-x0,y1-y0)<3 or (x1-x0)*(y1-y0)<16:
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'too_few_source_pixels','method':method}
    if method!='grabcut':
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'unsupported_proposal_method','method':method}
    try:
        import cv2
    except ImportError:
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'BLOCKED_MASK_BACKEND','method':method}
    # GrabCut needs background outside rectangle; when the bbox touches image edges
    # use an expanded local region if possible, else leave it for human processing.
    if x0<1 or y0<1 or x1>=im.shape[1] or y1>=im.shape[0]:
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'insufficient_background_border','method':method}
    mask=np.zeros(im.shape[:2],dtype=np.uint8)
    bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(im.copy(),mask,(x0,y0,x1-x0,y1-y0),bg,fg,5,cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'mask_backend_failed','method':method}
    alpha=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    if not np.any(alpha):
        return {'candidate_alpha':None,'status':'UNRECOVERABLE','reason':'mask_empty','method':method}
    return {'candidate_alpha':alpha,'status':'PENDING_REVIEW','reason':'human_review_required','method':method}


def validate_reviewed_candidate(record:dict,image:np.ndarray,mask:np.ndarray) -> dict:
    """Conservative preliminary QA; on the full repo chain this with
    prepare_eth_real_foregrounds.validate_foreground_mask for contract parity.
    """
    reasons=[]
    im=np.asarray(image); alpha=np.asarray(mask)
    if im.ndim!=3 or im.shape[-1]!=3 or alpha.ndim!=2 or alpha.shape!=im.shape[:2]:
        reasons.append('mask_dimensions_match_rgb')
    if record.get('reviewed') is not True or not record.get('reviewer_id') or not record.get('review_method'):
        reasons.append('human_review_required')
    box=record.get('bbox_xyxy')
    if not isinstance(box,(list,tuple)) or len(box)!=4:
        reasons.append('invalid_bbox')
    else:
        x0,y0,x1,y1=map(int,box)
        if x1<=x0 or y1<=y0 or x0<0 or y0<0 or x1>im.shape[1] or y1>im.shape[0]:
            reasons.append('invalid_bbox')
        elif alpha.shape==im.shape[:2]:
            support=alpha>0
            if not support.any():
                reasons.append('mask_empty')
            else:
                ys,xs=np.nonzero(support)
                sb=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
                if sb[0]<x0-1 or sb[1]<y0-1 or sb[2]>x1+1 or sb[3]>y1+1:
                    reasons.append('mask_outside_bbox')
                patch=support[y0:y1,x0:x1]
                if patch.mean()>=.98:
                    reasons.append('bbox_crop_not_mask')
    ss=record.get('source_size')
    if not ss or len(ss)!=2:
        reasons.append('source_size_unknown')
    elif box and len(box)==4 and max((box[2]-box[0])/float(ss[0]),(box[3]-box[1])/float(ss[1]))>1.0:
        reasons.append('upscaled_source_crop')
    return {'status':'REJECTED' if reasons else 'APPROVED','reasons':reasons,
            'instance_id':record.get('instance_id'),'reviewed':record.get('reviewed') is True}
