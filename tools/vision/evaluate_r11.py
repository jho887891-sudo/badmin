"""R11 post-hoc vision evaluation. Never feeds ground truth into inference.

Ground-truth JSONL, one reviewed frame per row::
  {"frame_id":0,"timestamp_s":0,"status":"labeled",
   "image_size":{"width_px":1280,"height_px":720},
   "objects":[{"bbox_xyxy_px":[200,100,205,105]}]}

Use status="unscorable" (objects=[]) for frames where the object cannot be
reliably judged; an absent GT row is *unlabeled*, not a negative frame.
Prediction JSONL matches the existing YOLO DetectionObservation schema and
R11 tracks.jsonl. All geometry is IMAGE_PLANE_2D, never metres.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

CHI2_2D_95 = 5.991464547107979
BUCKETS = ('<4', '4-6', '6-10', '10-20', '>=20')


class EvaluationError(ValueError):
    pass


def _finite(v, name):
    try:
        out = float(v)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f'{name} must be finite') from exc
    if not math.isfinite(out):
        raise EvaluationError(f'{name} must be finite')
    return out


def _integer(v, name):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise EvaluationError(f'{name} must be a nonnegative integer')
    return v


def _xyxy(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise EvaluationError(f'{name} needs four pixel coordinates')
    x1,y1,x2,y2 = [_finite(v,name) for v in value]
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise EvaluationError(f'{name} outside source image or degenerate')
    return (x1,y1,x2,y2)


def _size(record):
    sz = record.get('image_size')
    if not isinstance(sz, dict):
        raise EvaluationError('image_size missing')
    return _integer(sz.get('width_px'),'width_px'),_integer(sz.get('height_px'),'height_px')


def _validate_gt(rows, expected_frames=None):
    seen=set();last_t=-1;out={}
    if expected_frames is not None and (type(expected_frames) is not int or expected_frames <= 0):
        raise EvaluationError('expected_frames must be a positive integer')
    for r in rows:
        if not isinstance(r, dict):raise EvaluationError('GT row must be a mapping')
        fid=_integer(r.get('frame_id'),'GT frame_id')
        if fid in seen:raise EvaluationError(f'duplicate GT frame {fid}')
        if expected_frames is not None and fid>=expected_frames:
            raise EvaluationError('GT frame_id exceeds expected frame count')
        t=_finite(r.get('timestamp_s'),'GT timestamp_s')
        if t<0:raise EvaluationError('negative GT timestamp')
        status=r.get('status')
        if status not in ('labeled','unscorable'):
            raise EvaluationError('GT status must be labeled or unscorable')
        w,h=_size(r)
        if w<2 or h<2:raise EvaluationError('invalid image size')
        objects=r.get('objects')
        if not isinstance(objects,list):raise EvaluationError('GT objects must be a list')
        if status=='unscorable' and objects:
            raise EvaluationError('unscorable frames must contain zero objects')
        boxes=[]
        for obj in objects:
            if not isinstance(obj,dict):raise EvaluationError('GT object must be mapping')
            boxes.append(_xyxy(obj.get('bbox_xyxy_px'),(w,h),'GT bbox'))
        out[fid]={'time':t,'status':status,'boxes':boxes,'size':(w,h)}
        seen.add(fid)
    for fid in sorted(out):
        t=out[fid]['time']
        if t<=last_t:raise EvaluationError('GT timestamps must strictly increase by frame id')
        last_t=t
    return out


def _validate_pred(rows, gt):
    by_frame=defaultdict(list)
    for r in rows:
        if not isinstance(r,dict):raise EvaluationError('detection row must be a mapping')
        fid=_integer(r.get('frame_id'),'prediction frame_id')
        if fid in gt:
            sz=_size(r)
            if sz!=gt[fid]['size']:raise EvaluationError('prediction image size differs from GT')
        else:
            sz=_size(r)
        box=_xyxy(r.get('bbox_xyxy_px'),sz,'YOLO bbox')
        confidence=_finite(r.get('confidence'),'confidence')
        if not 0 <= confidence <= 1:raise EvaluationError('confidence not in [0,1]')
        units=r.get('units',{})
        if units and units.get('bbox_xyxy_px')!='px':
            raise EvaluationError('non-pixel detection')
        by_frame[fid].append((box,confidence))
    return by_frame


def _iou(a,b):
    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/(area_a+area_b-overlap) if area_a+area_b>overlap else 0.


def _distance(a,b):
    return math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)


def _bucket(b):
    d=math.sqrt((b[2]-b[0])*(b[3]-b[1]))
    return '<4' if d<4 else '4-6' if d<6 else '6-10' if d<10 else '10-20' if d<20 else '>=20'


def _matching(gtboxes,predboxes,criterion,threshold):
    """Global greedy one-to-one matching, ranked by spatial agreement.

    The metric is frozen before comparing full-frame and tiled model outputs.
    Each GT and prediction participates in at most one true positive.
    """
    pairs=[]
    for gi,g in enumerate(gtboxes):
        for pi,(p,_) in enumerate(predboxes):
            metric=_iou(g,p) if criterion=='iou' else _distance(g,p)
            ok=metric>=threshold if criterion=='iou' else metric<=threshold
            if ok:pairs.append((metric,gi,pi))
    pairs.sort(key=lambda v:(-v[0],v[1],v[2]) if criterion=='iou' else (v[0],v[1],v[2]))
    used_g=set();used_p=set()
    for _,gi,pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi);used_p.add(pi)
    return used_g,used_p


def _rate(num,den):
    return num/den if den else None


def evaluate_detections(gt_rows,pred_rows,*,expected_frames=None,centre_tolerance_px=3.0,
                        split_provenance='unknown',annotation_provenance='unknown'):
    """Human-reviewed TP/FP/FN. Missing/uncertain frames never become negatives."""
    if not math.isfinite(centre_tolerance_px) or centre_tolerance_px<=0:
        raise EvaluationError('centre_tolerance_px must be positive finite')
    gt=_validate_gt(gt_rows,expected_frames)
    preds=_validate_pred(pred_rows,gt)
    if split_provenance not in ('train','validation','held_out_session','unknown'):
        raise EvaluationError('invalid split provenance')
    if annotation_provenance not in ('human_verified','unknown'):
        raise EvaluationError('invalid annotation provenance')
    counts={'iou50':{'tp':0,'fp':0,'fn':0},'center_distance':{'tp':0,'fp':0,'fn':0}}
    buckets={key:{'gt':0,'tp_iou50':0,'tp_center':0} for key in BUCKETS}
    frame_details=[];unscored=0
    for fid in sorted(set(gt)|set(preds)):
        r=gt.get(fid)
        p=preds.get(fid,[])
        if r is None or r['status']=='unscorable':
            unscored+=len(p)
            continue
        boxes=r['boxes']
        for box in boxes:buckets[_bucket(box)]['gt']+=1
        entry={'frame_id':fid,'gt':len(boxes),'detections':len(p)}
        for name,criterion,threshold in (('iou50','iou',.5),
                                         ('center_distance','center',centre_tolerance_px)):
            gi,pi=_matching(boxes,p,criterion,threshold)
            tp=len(gi);fp=len(p)-len(pi);fn=len(boxes)-len(gi)
            for label,value in (('tp',tp),('fp',fp),('fn',fn)):
                counts[name][label]+=value
            entry[name]={'tp':tp,'fp':fp,'fn':fn}
            target='tp_iou50' if name=='iou50' else 'tp_center'
            for idx in gi:buckets[_bucket(boxes[idx])][target]+=1
        frame_details.append(entry)
    for c in counts.values():
        c['precision']=_rate(c['tp'],c['tp']+c['fp'])
        c['recall']=_rate(c['tp'],c['tp']+c['fn'])
    for b in buckets.values():
        b['recall_iou50']=_rate(b['tp_iou50'],b['gt'])
        b['recall_center']=_rate(b['tp_center'],b['gt'])
    labeled=sum(r['status']=='labeled' for r in gt.values())
    unscorable=len(gt)-labeled
    missing=expected_frames-len(gt) if expected_frames is not None else None
    return {'schema':'r11_detection_eval/v1','units':'IMAGE_PLANE_2D',
            'matching':{'iou50':0.5,'center_tolerance_px':centre_tolerance_px},
            'split_provenance':split_provenance,'annotation_provenance':annotation_provenance,
            'generalization_claim_allowed':(split_provenance=='held_out_session' and
                annotation_provenance=='human_verified' and expected_frames is not None and
                missing==0 and labeled>0),
            'annotated_coverage':{'expected_frames':expected_frames,'gt_rows':len(gt),
               'labeled_frames':labeled,'unscorable_frames':unscorable,
               'missing_frames':missing,'labeled_positive_frames':sum(
                   bool(r['boxes']) and r['status']=='labeled' for r in gt.values())},
            'iou50':counts['iou50'],'center_distance':counts['center_distance'],
            'size_buckets':buckets,'unscored_detections':unscored,
            'frame_details':frame_details,
            'warning':'Detection presence is not recall; scores cover only labeled frames.'}


def _percentile(values,percent):
    if not values:return None
    xs=sorted(values); k=(len(xs)-1)*percent/100.;a=int(math.floor(k));b=int(math.ceil(k))
    return xs[a]*(b-k)+xs[b]*(k-a) if a!=b else xs[a]


def _coverage95(diff,cov):
    if not isinstance(cov,(list,tuple)) or len(cov)!=2 or any(
            not isinstance(row,(list,tuple)) or len(row)!=2 for row in cov):
        return None
    a,b,c,d=(_finite(v,'position covariance') for row in cov for v in row)
    if abs(b-c)>1e-7*max(1,abs(b),abs(c)):
        return None
    bc=(b+c)/2;det=a*d-bc*bc
    if a<=0 or det<=1e-12:
        return None
    dx,dy=diff
    m=(d*dx*dx-2*bc*dx*dy+a*dy*dy)/det
    return m<=CHI2_2D_95


def evaluate_predictions(gt_rows,track_rows,*,tolerance_s=.02):
    """Post-hoc exact-clock horizon check; no interpolation or oracle prediction."""
    if not math.isfinite(tolerance_s) or tolerance_s<=0:
        raise EvaluationError('tolerance_s must be positive finite')
    gt=_validate_gt(gt_rows)
    tlist=sorted((r['time'],fid) for fid,r in gt.items())
    times=[t for t,_ in tlist]
    out={}
    for record in track_rows:
        fid=_integer(record.get('frame_id'),'track frame_id')
        t=_finite(record.get('timestamp_s'),'track timestamp')
        if t<0:raise EvaluationError('negative track timestamp')
        if record.get('units')!='IMAGE_PLANE_2D':
            raise EvaluationError('track must be IMAGE_PLANE_2D')
        if fid not in gt:continue
        if abs(t-gt[fid]['time'])>1e-6:
            raise EvaluationError('track and GT timestamps disagree for frame_id')
        if gt[fid]['status']!='labeled':continue
        horizons=record.get('horizon',[])
        if not isinstance(horizons,list):raise EvaluationError('horizon must be list')
        for h in horizons:
            horizon=_finite(h.get('horizon_s'),'horizon_s')
            if horizon<=0:raise EvaluationError('horizon_s must be positive')
            key=f'{horizon:g}'
            state=out.setdefault(key,{'errors':[],'coverage':[],'skipped_no_future_gt':0,
                                  'skipped_ambiguous_gt':0,'skipped_unscorable':0,
                                  'pairs':[]})
            point=h.get('centre_px')
            if not isinstance(point,(list,tuple)) or len(point)!=2:
                raise EvaluationError('prediction centre_px requires xy')
            px,py=[_finite(v,'prediction centre_px') for v in point]
            target=t+horizon
            i=bisect.bisect_left(times,target)
            nearby=[tlist[j] for j in (i-1,i) if 0<=j<len(tlist) and
                    tlist[j][0]>t and abs(tlist[j][0]-target)<=tolerance_s]
            if not nearby:
                state['skipped_no_future_gt']+=1
                continue
            _,future_fid=min(nearby,key=lambda item:(abs(item[0]-target),item[1]))
            future=gt[future_fid]
            if future['status']!='labeled':
                state['skipped_unscorable']+=1
                continue
            if len(future['boxes'])!=1:
                state['skipped_ambiguous_gt']+=1
                continue
            box=future['boxes'][0]
            gx=(box[0]+box[2])/2;gy=(box[1]+box[3])/2
            dx=px-gx;dy=py-gy
            state['errors'].append(math.hypot(dx,dy))
            cover=_coverage95((dx,dy),h.get('position_covariance_px2')) if h.get('position_covariance_px2') is not None else None
            if cover is not None:state['coverage'].append(cover)
            state['pairs'].append({'source_frame_id':fid,'future_frame_id':future_fid,
                'source_time_s':t,'expected_future_time_s':target,
                'actual_future_time_s':future['time'],
                'error_px':math.hypot(dx,dy),'coverage95':cover})
    summary={}
    for h,st in out.items():
        e=st['errors'];cov=st['coverage']
        summary[h]={'n':len(e),'median_error_px':statistics.median(e) if e else None,
            'p95_error_px':_percentile(e,95),'p99_error_px':_percentile(e,99),
            'n_coverage_evaluable':len(cov),
            'coverage95':sum(cov)/len(cov) if cov else None,
            'skipped_no_future_gt':st['skipped_no_future_gt'],
            'skipped_ambiguous_gt':st['skipped_ambiguous_gt'],
            'skipped_unscorable':st['skipped_unscorable'],
            'pairs':st['pairs']}
    return {'schema':'r11_prediction_eval/v1','units':'IMAGE_PLANE_2D',
        'future_match_tolerance_s':tolerance_s,'horizons':summary,
        'warning':'GT is for POST-HOC evaluation ONLY; no interpolation; XY coverage needs calibrated covariance.'}


def _read_jsonl(path):
    rows=[]
    try:
        with Path(path).open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                if line.strip():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise EvaluationError(f'{path}: invalid JSON at line {i}') from exc
    except OSError as exc:
        raise EvaluationError(f'cannot read {path}: {exc}') from exc
    return rows


def load_ground_truth(path):
    rows=_read_jsonl(path)
    _validate_gt(rows)
    return rows


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description='Offline R11 truth-aligned vision metrics; never calls YOLO/GT-driven inference')
    p.add_argument('--mode',required=True,choices=['detection','prediction'])
    p.add_argument('--ground-truth',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--expected-frames',type=int)
    p.add_argument('--centre-tolerance-px',type=float,default=3.)
    p.add_argument('--future-tolerance-s',type=float,default=.02)
    p.add_argument('--split-provenance',choices=['train','validation','held_out_session','unknown'],default='unknown')
    p.add_argument('--annotation-provenance',choices=['human_verified','unknown'],default='unknown')
    a=p.parse_args(argv)
    if a.output.exists():p.error('refusing to overwrite existing report')
    try:
        gt=load_ground_truth(a.ground_truth);data=_read_jsonl(a.input)
        if a.mode=='detection':
            report=evaluate_detections(gt,data,expected_frames=a.expected_frames,
                centre_tolerance_px=a.centre_tolerance_px,
                split_provenance=a.split_provenance,annotation_provenance=a.annotation_provenance)
        else:report=evaluate_predictions(gt,data,tolerance_s=a.future_tolerance_s)
        report['provenance']={'ground_truth_sha256':_hash(a.ground_truth),
                              'input_sha256':_hash(a.input)}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    except EvaluationError as exc:p.error(str(exc))
    print(json.dumps({'mode':a.mode,'report':str(a.output),'schema':report['schema']}))
    return 0


if __name__=='__main__':raise SystemExit(main())