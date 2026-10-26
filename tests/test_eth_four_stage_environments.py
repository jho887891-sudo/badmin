import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from eth_four_stage_environments import validate_real_background,make_stage4_sample,EnvironmentRejected


def foreground():
    rgb=np.full((12,12,3),238,dtype=np.uint8)
    mask=np.zeros((12,12),dtype=np.float32); mask[2:10,4:8]=1
    return {'sample_id':'eth-1','rgb':rgb,'alpha':mask,'source_sha256':'F'*64,'source':'eth_train','qa_status':'APPROVED','reviewed':True,
            'reviewer_id':'human-1','review_method':'visual'}


def background():
    image=np.full((64,96,3),40,dtype=np.uint8)
    image[:10,:,0]=245 # overhead lamps / distractions
    return {'background_id':'gym-1','image':image,'source_sha256':'B'*64,
            'source':'real_venue_train','split':'train','qa_status':'APPROVED',
            'no_shuttle_reviewed':True,'clean_roi':(10,10,90,60),
            'reviewer_id':'person-1','environment_type':'lamp_fixtures'}


def test_eval_background_blocked_by_sha():
    bg=background()
    v=validate_real_background(bg,[{'source_sha256':'B'*64,'role':'frozen_eval'}])
    assert v['status']=='REJECTED'
    assert 'frozen_source_or_parent_sha' in v['reasons']


def test_unreviewed_roi_blocked():
    bg=background();bg['qa_status']='PENDING_REVIEW'
    assert validate_real_background(bg,[])['status']=='REJECTED'


def test_unlabeled_shuttle_in_background_blocked():
    bg=background();bg['no_shuttle_reviewed']=False
    assert validate_real_background(bg,[])['status']=='REJECTED'


def test_lamp_background_retained_with_valid_roi():
    assert validate_real_background(background(),[])['status']=='PASS'
    r=make_stage4_sample(foreground(),background(),{'center_x':40,'center_y':35,'scale':1.0},seed=21)
    assert r['image'].shape==(64,96,3)
    assert r['metadata']['environment_type']=='lamp_fixtures'
    assert r['metadata']['bbox_xyxy']==[38,31,42,39]


def test_source_provenance_and_final_bbox_saved():
    r=make_stage4_sample(foreground(),background(),{'center_x':40,'center_y':35,'scale':1.0},seed=21)
    assert r['metadata']['foreground_sha256']=='F'*64
    assert r['metadata']['background_sha256']=='B'*64
    assert r['label'].startswith('0 ')
    assert r['metadata']['seed']==21


def test_invalid_foreground_review_blocked():
    fg=foreground();fg['qa_status']='PENDING_REVIEW'
    with pytest.raises(EnvironmentRejected):make_stage4_sample(fg,background(),{'center_x':40,'center_y':35},1)


def test_unauthorized_roi_outside_frame_rejected():
    bg=background();bg['clean_roi']=(0,0,120,64)
    assert validate_real_background(bg,[])['status']=='REJECTED'


def test_background_pixels_outside_target_unchanged():
    bg=background();r=make_stage4_sample(foreground(),bg,{'center_x':40,'center_y':35},seed=21)
    assert np.array_equal(r['image'][0:10],bg['image'][0:10])

def test_approved_string_without_human_review_is_rejected():
    fg=foreground();fg['reviewed']=False
    assert fg['qa_status']=='APPROVED'
    with pytest.raises(EnvironmentRejected):
        make_stage4_sample(fg,background(),{'center_x':40,'center_y':35},seed=21)
