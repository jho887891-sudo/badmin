import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from eth_four_stage_solid import sample_solid_rgb,create_stage1_sample,SolidRejected


def fg():
    rgb=np.full((20,20,3),240,dtype=np.uint8)
    mask=np.zeros((20,20),dtype=np.float32)
    mask[4:16,7:13]=1.0
    return rgb,mask


def test_palette_excludes_white_and_near_white():
    with pytest.raises(SolidRejected):
        sample_solid_rgb([(255,255,255),(30,30,30)],2)
    with pytest.raises(SolidRejected):
        sample_solid_rgb([(240,241,250),(30,30,30)],2)


def test_multiple_non_white_colors_appear_over_seed_set():
    pal=[(20,30,80),(20,80,30),(80,30,20)]
    assert len({sample_solid_rgb(pal,s) for s in range(15)})==3


def test_visible_bbox_matches_yolo_label():
    img=create_stage1_sample(*fg(),canvas_size=(64,48),color=(20,40,80),center=(32,24),seed=1)
    assert img['metadata']['bbox_xyxy']==[29,18,35,30]
    fields=img['label'].strip().split();assert fields[0]=='0'
    cx,cy,w,h=map(float,fields[1:])
    assert cx==pytest.approx(32/64,abs=1e-6)
    assert cy==pytest.approx(24/48,abs=1e-6)
    assert w==pytest.approx(6/64,abs=1e-6)
    assert h==pytest.approx(12/48,abs=1e-6)


def test_transparent_pixels_keep_exact_background():
    out=create_stage1_sample(*fg(),canvas_size=(64,48),color=(20,40,80),center=(32,24),seed=1)
    img=out['image']; m=out['visible_alpha']
    assert np.all(img[m==0]==np.array([20,40,80],dtype=np.uint8))


def test_invisible_positive_rejected():
    rgb,a=fg();a[:]=0
    with pytest.raises(SolidRejected):
        create_stage1_sample(rgb,a,canvas_size=(64,48),color=(20,40,80),center=(32,24),seed=1)


def test_real_photo_can_not_be_magnified():
    with pytest.raises(SolidRejected):
        create_stage1_sample(*fg(),canvas_size=(64,48),color=(20,40,80),center=(32,24),seed=1,scale=2)
