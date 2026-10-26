"""Real-domain metrics for four-stage ETH experiments.

Delegates GT matching to the repository's canonical eval_yolo26_v1 matcher.
No GT => no claim of Recall, even if detection count is high.
"""
from __future__ import annotations
from eval_yolo26_v1 import match_greedy


class EvaluationBlocked(ValueError):
    pass


def evaluate_from_predictions(samples:list[dict],config:dict)->dict:
    threshold=float(config.get('confidence',.25))
    iou=float(config.get('iou',.5))
    if not 0<=threshold<=1 or not 0<=iou<=1:
        raise EvaluationBlocked('invalid confidence or IoU')
    manifest=config.get('eval_manifest_sha')
    if not manifest:raise EvaluationBlocked('must freeze evaluation manifest sha256')
    complete=bool(samples) and all(isinstance(row.get('gt'),list) for row in samples)
    positives=0;tp=0;fp=0;fn=0;blurred_gt=0;blurred_hit=0
    lamp_fp=0;fp_unreviewed=0
    for row in samples:
        gt=row.get('gt')
        if not isinstance(gt,list):continue
        boxes=[g['box'] for g in gt]
        dets=[d for d in row.get('dets',[]) if float(d.get('conf',0))>=threshold]
        matches=match_greedy(boxes,[(float(d['conf']),d['box']) for d in dets],iou,metric='iou')
        positives+=len(gt)
        nh=sum(matches['gt_hit']);tp+=nh;fn+=len(gt)-nh
        fp+=len(dets)-nh
        for j,g in enumerate(gt):
            if g.get('blurred') is True:
                blurred_gt+=1
                if matches['gt_hit'][j]:blurred_hit+=1
        for i,d in enumerate(dets):
            if not matches['det_tp'][i]:
                category=d.get('reviewed_fp_type')
                if category=='lamp':lamp_fp+=1
                if category is None:fp_unreviewed+=1
    recall=tp/positives if complete and positives>0 else 'UNKNOWN'
    precision=tp/(tp+fp) if complete and tp+fp>0 else 'UNKNOWN'
    return {'status':'EVALUATED' if complete else 'GT_INCOMPLETE',
            'eval_manifest_sha':manifest,'iou':iou,'confidence':threshold,
            'frames':len(samples),'GT':positives if complete else 'UNKNOWN',
            'TP':tp if complete else 'UNKNOWN',
            'FP':fp if complete else 'UNKNOWN',
            'FN':fn if complete else 'UNKNOWN',
            'recall':recall,'precision':precision,
            'lamp_false_positives':lamp_fp if complete and fp_unreviewed==0 else 'UNKNOWN',
            'blurred_gt':blurred_gt if complete else 'UNKNOWN',
            'blurred_recall':blurred_hit/blurred_gt if complete and blurred_gt else 'UNKNOWN'}


def compare_e3_e4(e3_report:dict,e4_report:dict)->dict:
    fields=('eval_manifest_sha','confidence','iou','frames','GT')
    mismatch=[k for k in fields if e3_report.get(k)!=e4_report.get(k)]
    if mismatch:return {'status':'BLOCKED_EVAL_PARITY','mismatch':mismatch}
    a=e3_report.get('recall');b=e4_report.get('recall')
    if not isinstance(a,(float,int)) or not isinstance(b,(float,int)):
        return {'status':'UNKNOWN_GT','reason':'GT unavailable or no positives'}
    if b<=a:return {'status':'NO_RECALL_GAIN','recall_delta':b-a}
    if not isinstance(e3_report.get('FP'),int) or not isinstance(e4_report.get('FP'),int):
        return {'status':'UNKNOWN_FALSE_POSITIVES'}
    if e4_report['FP']>e3_report['FP']:
        return {'status':'TRADEOFF_REVIEW_REQUIRED','recall_delta':b-a,
                'fp_delta':e4_report['FP']-e3_report['FP']}
    return {'status':'IMPROVED_REAL_RECALL','recall_delta':b-a,
            'fp_delta':e4_report['FP']-e3_report['FP']}
