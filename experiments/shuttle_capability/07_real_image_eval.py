#!/usr/bin/env python3
"""07_real_image_eval.py - union-proposal GT for real images + current-model evaluation.

GT provenance is recorded explicitly: gt_source='auto_threshold_union' and
verified_by_agent=False unless a sheet was visually checked.  Never presented as human GT.
"""
from __future__ import annotations

import argparse, csv, json, os
from pathlib import Path
import cv2, numpy as np

BUCKETS = [(0,4),(4,6),(6,8),(8,12),(12,16),(16,24),(24,32),(32,1e9)]


def bucket(size):
    for lo,hi in BUCKETS:
        if lo <= size < hi: return f'{lo}-{hi}px' if hi<1e9 else f'>{lo}px'
    return 'unknown'


def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/home/T7/ojh/robot_sim')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='0')
    a = ap.parse_args()
    repo = Path(a.repo)
    out = repo / 'outputs/shuttle_capability'
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments/shuttle_capability/_scratch/ultralytics_cfg')
    props = json.loads((out / 'annotations/proposals_real_images.json').read_text())
    from ultralytics import YOLO
    model = YOLO(str(repo / 'assets/external/_staging/F_yolo/weights/yolo26s.pt'))
    names = model.names
    sports = [k for k,v in names.items() if str(v) == 'sports ball']
    lab_dir = out / 'annotations/real_images'; lab_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cands in props.items():
        img = cv2.imread(str(out / 'real_images/raw' / name))
        if img is None: continue
        H, W = img.shape[:2]
        gts = []
        for (x,y,bw,bh,area) in cands:
            if area < 400:  # ignore tiny specks for the union proposal
                continue
            gts.append((x, y, x+bw, y+bh))
        if gts:
            x1 = min(g[0] for g in gts); y1 = min(g[1] for g in gts)
            x2 = max(g[2] for g in gts); y2 = max(g[3] for g in gts)
            ux1,uy1,ux2,uy2 = max(0,x1-4), max(0,y1-4), min(W,x2+4), min(H,y2+4)
            bw, bh = ux2-ux1, uy2-uy1
            eq = float(np.sqrt(bw*bh))
            (lab_dir / f'{Path(name).stem}.txt').write_text(
                f'0 {((ux1+ux2)/2)/W:.6f} {((uy1+uy2)/2)/H:.6f} {bw/W:.6f} {bh/H:.6f}\n')
            res = model.predict(img, imgsz=a.imgsz, conf=a.conf, iou=0.7, verbose=False, device=a.device)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
            scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,),int)
            best = max([iou((ux1,uy1,ux2,uy2), tuple(b)) for b in boxes], default=0.0)
            best_sb = max([iou((ux1,uy1,ux2,uy2), tuple(b)) for b,c in zip(boxes,cls) if int(c) in sports], default=0.0)
            rows.append({'file': name, 'bbox_w_px': bw, 'bbox_h_px': bh, 'equivalent_size_px': round(eq,3),
                         'bucket': bucket(eq), 'n_pred': len(boxes),
                         'best_iou_any': round(float(best),4), 'best_iou_sports_ball': round(float(best_sb),4),
                         'hit_any': int(best >= a.iou), 'hit_sports_ball': int(best_sb >= a.iou),
                         'max_conf': round(float(scores.max()),4) if len(scores) else 0.0,
                         'gt_source': 'auto_threshold_union', 'verified_by_agent': False})
    csv_path = out / 'metrics/real_image_eval_before.csv'
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'real images evaluated: {len(rows)}')
    buckets = {}
    for r in rows:
        buckets.setdefault(r['bucket'], []).append(r)
    print(f"{'bucket':>9s} {'n':>4s} {'recall_any':>10s} {'recall_sb':>10s}")
    for b in sorted(buckets, key=lambda s: float(s.split('-')[0].rstrip('px>') or 0)):
        sel = buckets[b]
        print(f"{b:>9s} {len(sel):4d} {sum(x['hit_any'] for x in sel)/len(sel):10.3f} {sum(x['hit_sports_ball'] for x in sel)/len(sel):10.3f}")
    tot = sum(r['hit_any'] for r in rows)/len(rows); totsb = sum(r['hit_sports_ball'] for r in rows)/len(rows)
    print(f'overall recall_any={tot:.3f}  recall_sports_ball={totsb:.3f}')
    print('wrote', csv_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
