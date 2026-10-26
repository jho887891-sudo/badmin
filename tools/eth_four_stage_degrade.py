"""Stage 2: directional exposure blur and distinct multi-frame ghosting.

Do not claim these parameters match a specific real camera until measured.
All geometric image shifts and alpha support are jointly transformed.
"""
from __future__ import annotations
from io import BytesIO
import math
import numpy as np
from PIL import Image


class DegradeRejected(ValueError):
    pass


def _shift(arr, dx: int, dy: int):
    h,w=arr.shape[:2]
    result=np.zeros_like(arr)
    x0=max(0,dx);x1=min(w,w+dx)
    y0=max(0,dy);y1=min(h,h+dy)
    if x1>x0 and y1>y0:
        result[y0:y1,x0:x1]=arr[y0-dy:y1-dy,x0-dx:x1-dx]
    return result


def _normalize(rgb,alpha):
    r=np.asarray(rgb,dtype=np.float32)
    a=np.asarray(alpha,dtype=np.float32)
    if r.ndim!=3 or r.shape[2]!=3 or a.shape!=r.shape[:2]:
        raise DegradeRejected('RGB/alpha dimension mismatch')
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise DegradeRejected('invalid alpha')
    return r,a


def _unpremultiply(rp,ap):
    r=np.divide(rp,ap[:,:,None],out=np.zeros_like(rp),where=ap[:,:,None]>1e-7)
    return np.clip(np.rint(r),0,255).astype(np.uint8),np.clip(ap,0,1)


def apply_motion_blur_rgba(rgb: np.ndarray, alpha: np.ndarray, *,
                           length_px: float, angle_deg: float) -> tuple[np.ndarray,np.ndarray]:
    r,a=_normalize(rgb,alpha)
    if not math.isfinite(length_px) or length_px<0 or not math.isfinite(angle_deg):
        raise DegradeRejected('bad motion blur parameters')
    if length_px<.5:return np.asarray(rgb).copy(),a.copy()
    count=max(2,int(math.ceil(length_px))+1)
    theta=math.radians(angle_deg)
    xs=[int(round(t*math.cos(theta))) for t in np.linspace(-length_px/2,length_px/2,count)]
    ys=[int(round(t*math.sin(theta))) for t in np.linspace(-length_px/2,length_px/2,count)]
    pre=r*a[:,:,None]
    blurred=np.zeros_like(pre); ma=np.zeros_like(a)
    for dx,dy in zip(xs,ys):
        blurred+=_shift(pre,dx,dy)/count
        ma+=_shift(a,dx,dy)/count
    return _unpremultiply(blurred,ma)


def apply_ghosting_rgba(rgb: np.ndarray, alpha: np.ndarray, *,
                        displacement: tuple[float,float],
                        weights: tuple[float,float]) -> tuple[np.ndarray,np.ndarray]:
    r,a=_normalize(rgb,alpha)
    dx,dy=map(float,displacement)
    w0,w1=map(float,weights)
    if not all(map(math.isfinite,(dx,dy,w0,w1))) or w0<0 or w1<0 or abs(w0+w1-1)>.0001:
        raise DegradeRejected('invalid ghost displacement/weights')
    premul=r*a[:,:,None]
    xp=int(round(dx));yp=int(round(dy))
    aout=np.clip(a*w0+_shift(a,xp,yp)*w1,0,1)
    rout=premul*w0+_shift(premul,xp,yp)*w1
    return _unpremultiply(rout,aout)


def degrade_stage2_sample(sample: dict, params:dict,seed:int) -> dict:
    image=np.asarray(sample['image'],dtype=np.uint8)
    a=np.asarray(sample['visible_alpha'],dtype=np.float32)
    rgb,a=_normalize(image,a)
    bg=np.array(sample['metadata']['background_rgb'],dtype=np.float32)
    # Recover premultiplied foreground from the alpha composite before blur.
    src=np.divide(rgb-bg*(1-a[:,:,None]),a[:,:,None],out=np.zeros_like(rgb),where=a[:,:,None]>.001)
    src=np.clip(src,0,255).astype(np.uint8)
    length=float(params.get('motion_blur_px',0))
    angle=float(params.get('motion_angle_deg',0))
    if length:
        src,a=apply_motion_blur_rgba(src,a,length_px=length,angle_deg=angle)
    if params.get('ghost_weight',0):
        wt=float(params['ghost_weight'])
        src,a=apply_ghosting_rgba(src,a,displacement=(float(params.get('ghost_dx',0)),float(params.get('ghost_dy',0))),weights=(1-wt,wt))
    yy,xx=np.nonzero(a>=.25)
    if len(xx)<4:raise DegradeRejected('invisible target after degradation')
    h,w=a.shape
    x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    composite=np.clip(np.rint(bg*(1-a[:,:,None])+src.astype(np.float32)*a[:,:,None]),0,255)
    gamma=float(params.get('gamma',1))
    if not math.isfinite(gamma) or gamma<=0:raise DegradeRejected('invalid gamma')
    composite=255*(composite/255)**gamma
    sigma=float(params.get('noise_sigma',0))
    if not math.isfinite(sigma) or sigma<0:raise DegradeRejected('invalid noise')
    if sigma:composite+=np.random.default_rng(int(seed)).normal(0,sigma,composite.shape)
    out=np.clip(np.rint(composite),0,255).astype(np.uint8)
    if params.get('jpeg_quality'):
        quality=int(params['jpeg_quality'])
        if not 1<=quality<=100:raise DegradeRejected('invalid jpeg quality')
        f=BytesIO();Image.fromarray(out).save(f,'JPEG',quality=quality)
        f.seek(0);out=np.asarray(Image.open(f).convert('RGB')).copy()
    label=f'0 {(x0+x1)/(2*w):.6f} {(y0+y1)/(2*h):.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}\n'
    return {'image':out,'visible_alpha':a,'label':label,
            'metadata':{**sample['metadata'],'bbox_xyxy':[x0,y0,x1,y1],
                        'degradation':dict(params),'seed':int(seed)}}
