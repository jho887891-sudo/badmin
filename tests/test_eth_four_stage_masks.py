import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from propose_eth_instance_masks import propose_instance_mask,validate_reviewed_candidate


def frame():
    img=np.full((48,64,3),25,dtype=np.uint8)
    img[15:33,27:40]=220
    return img


def test_candidate_is_never_auto_approved():
    r=propose_instance_mask(frame(),(22,12,43,36))
    assert r['status'] in {'PENDING_REVIEW','UNRECOVERABLE'}
    assert r['status']!='APPROVED'
    assert r['candidate_alpha'] is None or r['candidate_alpha'].shape==(48,64)


def test_tiny_unsegmentable_instance_remains_recorded():
    r=propose_instance_mask(frame(),(5,5,6,6))
    assert r['status']=='UNRECOVERABLE'
    assert r['candidate_alpha'] is None
    assert r['reason']=='too_few_source_pixels'


def test_hard_rectangle_not_a_valid_mask():
    full=np.zeros((48,64),dtype=np.uint8)
    full[12:36,22:43]=255
    result=validate_reviewed_candidate({'instance_id':'s#0','bbox_xyxy':[22,12,43,36], 'source_size':[21,24],
                                        'reviewed':True,'reviewer_id':'human','review_method':'visual'},frame(),full)
    assert result['status']=='REJECTED'
    assert 'bbox_crop_not_mask' in result['reasons']


def test_approved_mask_requires_human_boolean_and_id():
    mask=np.zeros((48,64),dtype=np.uint8)
    yy,xx=np.ogrid[:48,:64]
    mask[((xx-32)/8)**2 + ((yy-23)/8)**2<1]=255
    row={'instance_id':'s#1','bbox_xyxy':[22,12,43,36],'source_size':[21,24],
         'reviewed':'True','reviewer_id':'human','review_method':'visual'}
    assert validate_reviewed_candidate(row,frame(),mask)['status']=='REJECTED'
    row['reviewed']=True
    assert validate_reviewed_candidate(row,frame(),mask)['status']=='APPROVED'
    row.pop('reviewer_id')
    assert validate_reviewed_candidate(row,frame(),mask)['status']=='REJECTED'


def test_mask_geometry_aligns_to_original_image():
    row={'instance_id':'s#1','bbox_xyxy':[22,12,43,36], 'source_size':[21,24],
         'reviewed':True,'reviewer_id':'human','review_method':'visual'}
    result=validate_reviewed_candidate(row,frame(),np.zeros((24,21),np.uint8))
    assert result['status']=='REJECTED'
    assert 'mask_dimensions_match_rgb' in result['reasons']
