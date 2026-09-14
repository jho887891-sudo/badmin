#!/usr/bin/env python3
"""09_p0_manifest.py - P0 deliverable: per-image GT verification manifest + verified-only metrics.

The verdicts below are the agent visual review of every one of the 59 downloaded images
(7 review sheets).  verified_by_agent is True ONLY where the drawn box was confirmed to span
the shuttle; images needing box repair are explicitly excluded from the official numbers.
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np

REPO = Path('/home/T7/ojh/robot_sim')
OUT = REPO / 'outputs/shuttle_capability'

ACCEPT = {
 'real_001.jpg': 'box spans cork+skirt',
 'real_002.jpg': 'box spans cork+skirt',
 'real_008.jpg': 'shuttle on grass; box covers object (slightly loose)',
 'real_014.jpg': 'box spans shuttle',
 'real_015.jpg': 'box spans shuttle',
 'real_017.jpg': 'two neon shuttles inside box',
 'real_020.jpg': 'box spans shuttle',
 'real_037.jpg': 'box spans shuttle',
 'real_038.jpg': 'box spans shuttle',
 'real_042.jpg': 'box spans shuttle',
}
NEGATIVE = {
 'real_004.jpg': 'person playing; no shuttle visible at this scale',
 'real_025.jpg': 'book cover with decorative feather, not a shuttle',
 'real_040.jpg': 'person playing; no shuttle visible',
 'real_056.jpg': 'bright lamps/fireworks - shuttle-like but no shuttle (hard negative)',
 'real_058.jpg': 'sailing - no shuttle (hard negative)',
 'real_059.jpg': 'matchbox-label wall - no shuttle (clutter negative)',
}
NEEDS_FIX = {
 'real_005.jpg':'box covers cork only', 'real_006.jpg':'box covers bottom strip only',
 'real_007.jpg':'no box proposed (shuttle visible)', 'real_009.jpg':'box includes display case',
 'real_010.jpg':'box includes display case', 'real_011.jpg':'box includes display case',
 'real_012.jpg':'box covers cork, skirt outside', 'real_013.jpg':'second shuttle unlabelled',
 'real_019.jpg':'box covers upper skirt only', 'real_021.jpg':'box covers whole racket',
 'real_022.jpg':'box includes display case', 'real_023.jpg':'box includes hand',
 'real_024.jpg':'box covers cork only', 'real_030.jpg':'box covers upper skirt only',
 'real_034.jpg':'box misses cork/base (installation photo)', 'real_041.jpg':'box covers upper skirt only',
 'real_043.jpg':'box covers bottom strip only', 'real_044.jpg':'no box proposed (low contrast)',
 'real_045.jpg':'box includes display case', 'real_046.jpg':'box includes display case',
 'real_047.jpg':'box includes display case', 'real_048.jpg':'box covers cork only',
 'real_049.jpg':'box on court region; shuttle too small to locate at view scale',
 'real_050.jpg':'box on players; shuttle not locatable at view scale',
 'real_051.jpg':'box on player region; shuttle not locatable',
 'real_052.jpg':'box on players; shuttle not locatable',
 'real_053.jpg':'box on players; shuttle not locatable',
 'real_054.jpg':'box on crowd/players; shuttle not locatable',
 'real_055.jpg':'box on court; shuttle not locatable at this resolution',
 'real_060.jpg':'box on player; shuttle not locatable at view scale',
}
EXCLUDE = {
 'real_003.png':'CG/graphic shuttle (not a real photo)', 'real_016.png':'CG/graphic shuttle',
 'real_018.png':'black frame', 'real_026.jpg':'historic engraving (illustration)',
 'real_027.jpg':'historic cartoon (illustration)', 'real_028.jpg':'archived document scan',
 'real_029.png':'parabola diagram', 'real_031.jpg':'engraving (illustration)',
 'real_032.jpg':'coloured illustration', 'real_033.jpg':'coloured illustration',
 'real_035.jpg':'engraving (illustration)', 'real_036.jpg':'engraving (illustration)',
 'real_039.png':'CG/graphic shuttle',
}

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def main() -> int:
    os.environ['YOLO_CONFIG_DIR'] = str(REPO / 'experiments/shuttle_capability/_scratch/ultralytics_cfg')
    props = json.loads((OUT / 'annotations/proposals_real_images.json').read_text())
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(REPO / 'assets/external/_staging/F_yolo/weights/yolo26s.pt'))
    names = model.names
    sports = [k for k,v in names.items() if str(v)=='sports ball']
    rows, metrics = [], []
    for name in sorted(props):
        if name in ACCEPT: verdict, reason, verified = 'POSITIVE_VERIFIED', ACCEPT[name], True
        elif name in NEGATIVE: verdict, reason, verified = 'NEGATIVE_VERIFIED', NEGATIVE[name], True
        elif name in NEEDS_FIX: verdict, reason, verified = 'NEEDS_BOX_FIX', NEEDS_FIX[name], False
        else: verdict, reason, verified = 'EXCLUDED_NONPHOTO', EXCLUDE.get(name,'-'), False
        img = cv2.imread(str(OUT / 'real_images/raw' / name))
        H, W = (img.shape[:2] if img is not None else (0,0))
        eq = bw = bh = None
        if name in ACCEPT:
            big = [c for c in props[name] if c[4] >= 400]
            if big:
                x1=min(c[0] for c in big); y1=min(c[1] for c in big)
                x2=max(c[0]+c[2] for c in big); y2=max(c[1]+c[3] for c in big)
                bw, bh = x2-x1, y2-y1; eq = float(np.sqrt(bw*bh))
        rows.append({'file':name,'verdict':verdict,'verified_by_agent':verified,'reason':reason,
                     'equiv_size_px': round(eq,1) if eq else ''})
        if verified and img is not None:
            res = model.predict(img, imgsz=1280, conf=0.05, iou=0.7, verbose=False, device='0')[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,),int)
            if verdict.startswith('POSITIVE'):
                gt = (x1, y1, x2, y2)
                best = max([iou(gt, tuple(b)) for b in boxes], default=0.0)
                best_sb = max([iou(gt, tuple(b)) for b,c in zip(boxes,cls) if int(c) in sports], default=0.0)
                metrics.append({'file':name,'kind':'positive','equiv_size_px':round(eq,1),
                                'best_iou_any':round(float(best),4),'best_iou_sports_ball':round(float(best_sb),4),
                                'hit_any':int(best>=0.5),'hit_sports_ball':int(best_sb>=0.5),'n_pred':len(boxes)})
            else:
                metrics.append({'file':name,'kind':'negative','equiv_size_px':'','best_iou_any':'',
                                'best_iou_sports_ball':'','hit_any':'','hit_sports_ball':'','n_pred':len(boxes)})
    with (OUT / 'annotations/real_images_gt_manifest.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT / 'metrics/real_image_verified_metrics.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics[0].keys())); w.writeheader(); w.writerows(metrics)

    pos = [m for m in metrics if m['kind']=='positive']; neg = [m for m in metrics if m['kind']=='negative']
    summary = {
        'reviewed_images': len(rows),
        'positive_verified': len(pos), 'negative_verified': len(neg),
        'needs_box_fix': sum(1 for r in rows if r['verdict']=='NEEDS_BOX_FIX'),
        'excluded_nonphoto': sum(1 for r in rows if r['verdict']=='EXCLUDED_NONPHOTO'),
        'verified_recall_any': round(sum(m['hit_any'] for m in pos)/len(pos),3) if pos else None,
        'verified_recall_sports_ball': round(sum(m['hit_sports_ball'] for m in pos)/len(pos),3) if pos else None,
        'negative_fp_frames': sum(1 for m in neg if m['n_pred']>0), 'negative_total': len(neg),
        'verified_size_range_px': [min(m['equiv_size_px'] for m in pos), max(m['equiv_size_px'] for m in pos)] if pos else None,
    }
    (OUT / 'metrics/real_image_verified_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('wrote manifest + verified metrics')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
