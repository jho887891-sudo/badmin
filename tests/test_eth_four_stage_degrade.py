import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from eth_four_stage_degrade import apply_motion_blur_rgba,apply_ghosting_rgba,degrade_stage2_sample,DegradeRejected
from eth_four_stage_solid import create_stage1_sample


def raw():
    rgb=np.full((20,20,3),245,dtype=np.uint8)
    a=np.zeros((20,20),dtype=np.float32); a[7:13,9:11]=1
    return rgb,a


def stage1():
    return create_stage1_sample(*raw(),canvas_size=(64,48),color=(20,40,80),center=(32,24),seed=7)


def test_zero_blur_is_identity():
    rgb,a=raw();o,b=apply_motion_blur_rgba(rgb,a,length_px=0,angle_deg=0)
    assert np.array_equal(o,rgb)
    assert np.array_equal(b,a)


def test_direction_changes_tail_axis():
    rgb,a=raw();_,h=apply_motion_blur_rgba(rgb,a,length_px=8,angle_deg=0)
    _,v=apply_motion_blur_rgba(rgb,a,length_px=8,angle_deg=90)
    def bbox(m):
        ys,xs=np.nonzero(m>.04)
        return xs.max()-xs.min(),ys.max()-ys.min()
    hw,hh=bbox(h);vw,vh=bbox(v)
    assert hw>vw and vh>hh


def test_ghosting_differs_from_single_exposure_blur():
    rgb,a=raw();_,blur=apply_motion_blur_rgba(rgb,a,length_px=7,angle_deg=0)
    _,ghost=apply_ghosting_rgba(rgb,a,displacement=(7,0),weights=(.75,.25))
    assert not np.allclose(blur,ghost)


def test_bbox_recomputed_from_final_visible_alpha():
    out=degrade_stage2_sample(stage1(),{'motion_blur_px':4,'motion_angle_deg':0},seed=3)
    yy,xx=np.nonzero(out['visible_alpha']>=.25)
    assert out['metadata']['bbox_xyxy']==[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]


def test_overblurred_invisible_target_has_no_positive_label():
    s=stage1();s['visible_alpha'][:]=0
    with pytest.raises(DegradeRejected):
        degrade_stage2_sample(s,{'motion_blur_px':0},seed=1)


def test_seed_replays_identically():
    a=degrade_stage2_sample(stage1(),{'noise_sigma':4,'motion_blur_px':3,'gamma':1.1},seed=31)
    b=degrade_stage2_sample(stage1(),{'noise_sigma':4,'motion_blur_px':3,'gamma':1.1},seed=31)
    assert np.array_equal(a['image'],b['image'])
    assert a['label']==b['label']
