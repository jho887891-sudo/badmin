import json
import math
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.vision.evaluate_r11 import evaluate_detections, evaluate_predictions, load_ground_truth, EvaluationError


def label(frame, t, boxes=(), status="labeled"):
    return {"frame_id": frame, "timestamp_s": t, "status": status,
            "image_size": {"width_px": 100, "height_px": 100},
            "objects": [{"bbox_xyxy_px": list(b)} for b in boxes]}


def det(frame, box, confidence=.8):
    return {"frame_id":frame,"timestamp_s":frame/30,
            "image_size":{"width_px":100,"height_px":100},
            "bbox_xyxy_px":list(box),"confidence":confidence,
            "units":{"bbox_xyxy_px":"px"}}


def track(frame, t, x, y, horizon=.3, cov=None):
    return {"frame_id":frame,"timestamp_s":t,"units":"IMAGE_PLANE_2D",
            "horizon":[{"horizon_s":horizon,"centre_px":[x,y],
                        **({"position_covariance_px2":cov} if cov else {})}]}


def test_detection_exact_matches_and_size_buckets():
    labels=[label(0,0,[(20,20,25,25)]),label(1,1/30,[(30,30,44,44)]),label(2,2/30,[])]
    preds=[det(0,(20,20,25,25)),det(1,(30,30,44,44)),det(2,(5,5,9,9))]
    result=evaluate_detections(labels,preds,expected_frames=3)
    assert result['iou50']['tp']==2 and result['iou50']['fp']==1 and result['iou50']['fn']==0
    assert result['iou50']['precision']==pytest.approx(2/3)
    assert result['iou50']['recall']==1
    assert result['size_buckets']['4-6']['gt']==1
    assert result['size_buckets']['10-20']['gt']==1
    assert result['annotated_coverage']['labeled_frames']==3


def test_one_gt_does_not_match_two_overlapping_predictions():
    gt=[label(0,0,[(20,20,25,25)])]
    result=evaluate_detections(gt,[det(0,(20,20,25,25)),det(0,(20,20,25,25),.5)],expected_frames=1)
    assert (result['iou50']['tp'],result['iou50']['fp'],result['iou50']['fn'])==(1,1,0)


def test_unscorable_frame_is_not_false_negative_or_true_negative():
    gt=[label(0,0,[(20,20,25,25)]),label(1,1/30,status='unscorable')]
    result=evaluate_detections(gt,[det(1,(2,2,8,8))],expected_frames=2)
    assert (result['iou50']['tp'],result['iou50']['fp'],result['iou50']['fn'])==(0,0,1)
    assert result['annotated_coverage']['unscorable_frames']==1


def test_missing_frames_are_not_auto_negative():
    result=evaluate_detections([label(0,0,[])],[],expected_frames=336)
    assert result['annotated_coverage']['missing_frames']==335
    assert result['iou50']['precision'] is None
    assert result['iou50']['recall'] is None
    assert not result['generalization_claim_allowed']


def test_det_of_unannotated_frame_is_not_counted_and_is_reported():
    result=evaluate_detections([label(0,0,[])],[det(2,(1,1,5,5))],expected_frames=3)
    assert result['iou50']['fp']==0
    assert result['unscored_detections']==1


def test_wrong_frame_or_bad_box_fails_closed():
    with pytest.raises(EvaluationError):
        evaluate_detections([label(0,0,[(101,0,110,10)])],[],expected_frames=1)
    with pytest.raises(EvaluationError):
        evaluate_detections([label(0,0,[]), label(0,1,[])],[],expected_frames=2)
    with pytest.raises(EvaluationError):
        evaluate_detections([label(0,0,[])],[det(-1,(0,0,5,5))],expected_frames=1)


def test_centre_three_px_tolerance_is_independent_of_iou():
    labels=[label(0,0,[(20,20,24,24)])]
    boxes=[det(0,(23,20,27,24))]
    r=evaluate_detections(labels,boxes,expected_frames=1,centre_tolerance_px=3.1)
    assert r['iou50']['tp']==0
    assert r['center_distance']['tp']==1


def test_prediction_matches_future_timestamps_not_array_offsets():
    labels=[label(0,0,[(10,10,14,14)]),label(1,.3,[(19,19,23,23)])]
    tracks=[track(0,0,21,21)]
    r=evaluate_predictions(labels,tracks,tolerance_s=.01)
    assert r['horizons']['0.3']['n']==1
    assert r['horizons']['0.3']['median_error_px']==pytest.approx(0)
    assert r['horizons']['0.3']['coverage95'] is None


def test_prediction_sparse_future_is_skipped_no_interpolation():
    labels=[label(0,0,[(10,10,14,14)]),label(20,.7,[(20,20,24,24)])]
    r=evaluate_predictions(labels,[track(0,0,20,20)],tolerance_s=.01)
    assert r['horizons']['0.3']['n']==0
    assert r['horizons']['0.3']['skipped_no_future_gt']==1


def test_uncertainty_coverage_95_computed_from_xy_not_velocity():
    labels=[label(0,0,[(10,10,14,14)]),label(9,.3,[(19,19,23,23)])]
    r=evaluate_predictions(labels,[track(0,0,22,21,cov=[[1,0],[0,1]])],tolerance_s=.01)
    assert r['horizons']['0.3']['coverage95']==1
    r=evaluate_predictions(labels,[track(0,0,30,30,cov=[[1,0],[0,1]])],tolerance_s=.01)
    assert r['horizons']['0.3']['coverage95']==0


def test_legacy_covariance_cannot_fake_calibrated_coverage():
    labels=[label(0,0,[(10,10,14,14)]),label(9,.3,[(19,19,23,23)])]
    r=evaluate_predictions(labels,[track(0,0,21,21)],tolerance_s=.01)
    assert r['horizons']['0.3']['n']==1
    assert r['horizons']['0.3']['n_coverage_evaluable']==0


def test_prediction_rejects_duplicate_or_reversed_timestamps():
    with pytest.raises(EvaluationError):
        evaluate_predictions([label(0,0,[]),label(1,0,[])],[])
    with pytest.raises(EvaluationError):
        evaluate_predictions([label(0,0,[]),label(1,.3,[])],[track(0,-1,1,1)])


def test_prediction_ambiguous_multiple_gt_skipped():
    labels=[label(0,0,[(10,10,14,14)]),label(9,.3,[(19,19,23,23),(30,30,34,34)])]
    r=evaluate_predictions(labels,[track(0,0,21,21)],tolerance_s=.01)
    assert r['horizons']['0.3']['n']==0
    assert r['horizons']['0.3']['skipped_ambiguous_gt']==1


def test_real_jsonl_loader(tmp_path):
    p=tmp_path/'gt.jsonl'
    p.write_text(json.dumps(label(0,0,[]))+'\n')
    assert len(load_ground_truth(p))==1
    p.write_text('NOT JSON\n')
    with pytest.raises(EvaluationError):load_ground_truth(p)


def test_maximum_cardinality_matching_avoids_greedy_undercount():
    # G0 is near P0 and P1, G1 is only near P0. Greedy would match G0-P0
    # and miss G1; maximum bipartite matching must return two true positives.
    labels=[label(0,0,[(10,10,12,12),(14,10,16,12)])]
    guesses=[det(0,(11,10,13,12)),det(0,(8,10,10,12))]
    result=evaluate_detections(labels,guesses,expected_frames=1,centre_tolerance_px=3.1)
    assert result['center_distance']['tp']==2


def test_annotation_future_wrong_clock_rejected():
    labels=[label(0,0,[(10,10,14,14)]),label(9,.3,[(19,19,23,23)])]
    with pytest.raises(EvaluationError):
        evaluate_predictions(labels,[track(0,.1,21,21)],tolerance_s=.01)


def test_declared_heldout_is_not_independent_split_audit():
    r=evaluate_detections([label(0,0,[(10,10,14,14)])],
                          [det(0,(10,10,14,14))],expected_frames=1,
                          split_provenance='held_out_session',
                          annotation_provenance='human_verified')
    assert r['generalization_claim_allowed'] is False
    assert r['held_out_claim_candidate'] is True


def test_detection_same_frame_id_but_wrong_clock_rejected():
    gt=[label(0,0.0,[(20,20,25,25)])]
    candidate=det(0,(20,20,25,25))
    candidate['timestamp_s']=0.8
    with pytest.raises(EvaluationError):
        evaluate_detections(gt,[candidate],expected_frames=1)


def test_detection_outside_video_frame_count_rejected():
    gt=[label(0,0.0,[(20,20,25,25)])]
    with pytest.raises(EvaluationError):
        evaluate_detections(gt,[det(336,(20,20,25,25))],expected_frames=336)
