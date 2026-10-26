"""Stage 1: non-white solid canvas with audited alpha-derived labels."""
from __future__ import annotations
import random
import numpy as np
from PIL import Image


class SolidRejected(ValueError):
    pass


def _color(c):
    if len(c)!=3 or any(not isinstance(v,(int,np.integer)) or v<0 or v>255 for v in c) or min(c)>=235:
        raise SolidRejected('background must be a non-white RGB color')
    return tuple(int(v) for v in c)


def sample_solid_rgb(palette: list[tuple[int,int,int]], seed: int) -> tuple[int,int,int]:
    if len(set(map(_color,palette)))<2:raise SolidRejected('need >=2 different solid backgrounds')
    return random.Random(int(seed)).choice([_color(c) for c in palette])


def create_stage1_sample(fg_rgb: np.ndarray, alpha: np.ndarray, *, canvas_size: tuple[int,int],
                         color: tuple[int,int,int], center: tuple[float,float],seed:int,scale:float=1.0) -> dict:
    color=_color(color)
    rgb=np.asarray(fg_rgb,dtype=np.uint8)
    a=np.asarray(alpha,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or a.shape!=rgb.shape[:2]:
        raise SolidRejected('RGB and alpha dimensions disagree')
    if not 0<scale<=1.0:raise SolidRejected('raw crop enlargement prohibited')
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise SolidRejected('alpha outside [0,1]')
    W,H=map(int,canvas_size)
    if W<=0 or H<=0:raise SolidRejected('invalid canvas dimensions')
    if scale!=1.0:
        w=max(1,round(rgb.shape[1]*scale));h=max(1,round(rgb.shape[0]*scale))
        tile=np.dstack([rgb,np.rint(a*255).astype(np.uint8)])
        tile=np.asarray(Image.fromarray(tile,'RGBA').resize((w,h),Image.Resampling.BICUBIC))
        rgb=tile[:,:,:3];a=tile[:,:,3].astype(np.float32)/255.0
    iw,ih=rgb.shape[1],rgb.shape[0]
    x=round(center[0]-iw/2);y=round(center[1]-ih/2)
    bx0,by0=max(0,x),max(0,y)
    bx1,by1=min(W,x+iw),min(H,y+ih)
    out=np.full((H,W,3),color,dtype=np.uint8)
    visible=np.zeros((H,W),dtype=np.float32)
    if bx1>bx0 and by1>by0:
        tx0,ty0=bx0-x,by0-y
        sub=a[ty0:ty0+by1-by0,tx0:tx0+bx1-bx0]
        tile=rgb[ty0:ty0+by1-by0,tx0:tx0+bx1-bx0]
        region=out[by0:by1,bx0:bx1].astype(np.float32)
        out[by0:by1,bx0:bx1]=np.clip(np.rint(region*(1-sub[:,:,None])+tile*sub[:,:,None]),0,255).astype(np.uint8)
        visible[by0:by1,bx0:bx1]=sub
    yy,xx=np.nonzero(visible>=.25)
    if len(xx)<4:raise SolidRejected('invisible target')
    x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    if (x1-x0)*(y1-y0)<=0:raise SolidRejected('empty target bbox')
    cx,cy=(x0+x1)/2/W,(y0+y1)/2/H
    bw,bh=(x1-x0)/W,(y1-y0)/H
    label=f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n'
    return {'image':out,'visible_alpha':visible,'label':label,
            'metadata':{'seed':int(seed),'background_rgb':color,'scale':float(scale),
                        'bbox_xyxy':[x0,y0,x1,y1],'visible_pixels':len(xx)}}
