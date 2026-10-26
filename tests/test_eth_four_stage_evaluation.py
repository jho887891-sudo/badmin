import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from eval_eth_four_stage import evaluate_from_predictions,compare_e3_e4


def sample(gt=True, predictions=True, blurred=False):
    return {'frame_id':'f1','gt':[{'box':[10,10,20,20],'blurred':blurred}] if gt else [],
            'dets':[{'box':[10,10,20,20],'conf':.8,'reviewed_fp_type':None}] if predictions else []}


def test_no_gt_yields_recall_unknown():
    row=sample();row.pop('gt')
    report=evaluate_from_predictions([row],{'iou':.5,'confidence':.25,'eval_manifest_sha':'SHA'})
    assert report['recall']=='UNKNOWN'
    assert report['precision']=='UNKNOWN'


def test_detection_count_not_confused_with_recall():
    row={'frame_id':'x','dets':[{'box':[5,5,8,8],'conf':.99}]}
    report=evaluate_from_predictions([row],{'iou':.5,'confidence':.25,'eval_manifest_sha':'SHA'})
    assert report['recall']=='UNKNOWN'


def test_identical_frozen_iou_center_logic():
    gt=sample()
    report=evaluate_from_predictions([gt],{'iou':.5,'confidence':.25,'eval_manifest_sha':'SHA'})
    assert report['TP']==1 and report['FP']==0 and report['FN']==0
    assert report['recall']==1.0


def test_lamp_fp_count_and_blur_recall_reported_separately():
    row=sample(blurred=True)
    row['dets'].append({'box':[30,30,35,35],'conf':.9,'reviewed_fp_type':'lamp'})
    report=evaluate_from_predictions([row],{'iou':.5,'confidence':.25,'eval_manifest_sha':'SHA'})
    assert report['TP']==1 and report['FP']==1
    assert report['lamp_false_positives']==1
    assert report['blurred_recall']==1.0


def test_e3_e4_mismatched_eval_manifest_refused():
    e3=evaluate_from_predictions([sample()],{'iou':.5,'confidence':.25,'eval_manifest_sha':'A'})
    e4=evaluate_from_predictions([sample()],{'iou':.5,'confidence':.25,'eval_manifest_sha':'B'})
    assert compare_e3_e4(e3,e4)['status']=='BLOCKED_EVAL_PARITY'


def test_e4_improvement_requires_real_data_and_fp_guardrail():
    e3=evaluate_from_predictions([sample(predictions=False)],{'iou':.5,'confidence':.25,'eval_manifest_sha':'A'})
    e4=evaluate_from_predictions([sample()],{'iou':.5,'confidence':.25,'eval_manifest_sha':'A'})
    assert compare_e3_e4(e3,e4)['status']=='IMPROVED_REAL_RECALL'
    e4['FP']=10
    assert compare_e3_e4(e3,e4)['status']=='TRADEOFF_REVIEW_REQUIRED'


def test_empty_ground_truth_is_valid_negative_frame():
    result=evaluate_from_predictions([sample(gt=False)],{'iou':.5,'confidence':.25,'eval_manifest_sha':'A'})
    assert result['GT']==0 and result['recall']=='UNKNOWN'
    assert result['FP']==1
