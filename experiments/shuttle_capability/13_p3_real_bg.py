#!/usr/bin/env python3
"""13_p3_real_bg.py - P3: controlled 3D shuttle composited on REAL backgrounds.

source_type = SYNTHETIC_ON_REAL_BG (never real image, never a substitute for real-video acceptance).
GT is exact (from the rasterised mask).  Also evaluates the current yolo26s.pt on this set.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, math, os
from pathlib import Path
import cv2, numpy as np

BUCKETS = [(0,4),(4,6),(6,8),(8,12),(12,16),(16,24),(24,32),(32,1e9)]


def load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


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
    ap.add_argument('--n', type=int, default=160)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval', action='store_true')
    a = ap.parse_args()
    repo = Path(a.repo)
    out = repo / 'outputs/shuttle_capability'
    gen = load_mod(repo / 'experiments/shuttle_capability/01_gen_synthetic.py', 'gen01')
    verts, faces = gen.load_shuttle_mesh(repo)
    center = verts.mean(axis=0)
    bg_dir = out / 'real_images/backgrounds'
    bgs = sorted([p for p in bg_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    rng = np.random.default_rng(a.seed)
    img_dir = out / 'synthetic_on_real_bg/images'; lab_dir = out / 'annotations/synthetic_on_real_bg'
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
    fx = 700.0; W = H = a.imgsz; K = (fx, fx, W/2.0, H/2.0)
    targets = np.clip(rng.lognormal(mean=math.log(9.0), sigma=0.55, size=a.n), 2.0, 60.0)
    rows = []
    for i, tgt in enumerate(targets):
        bgp = bgs[i % len(bgs)]
        bg = cv2.imread(str(bgp))
        if bg is None: continue
        if bg.shape[1] < W or bg.shape[0] < H:
            bg = cv2.resize(bg, (max(W, bg.shape[1]), max(H, bg.shape[0])))
        x0 = int(rng.integers(0, max(1, bg.shape[1]-W))); y0 = int(rng.integers(0, max(1, bg.shape[0]-H)))
        canvas = bg[y0:y0+H, x0:x0+W].copy()
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi, 3)
        R = gen.rot_ypr(yaw, pitch, roll)
        v_rel = (verts - center) @ R.T
        ext = float(np.max(v_rel.max(axis=0) - v_rel.min(axis=0)))
        dist = max(fx * ext / tgt, 0.5)
        far = rng.uniform(-0.35, 0.35)
        pos = np.array([rng.uniform(-0.45,0.45)*W, rng.uniform(-0.45,0.45)*H]) + np.array([W/2, H/2])
        v_cam = v_rel + np.array([ (pos[0]-W/2)*dist/fx, (pos[1]-H/2)*dist/fx, dist ])
        mask = gen.rasterize(v_cam, faces, K, W, H)
        if not mask.any(): continue
        ys, xs = np.nonzero(mask)
        bw = int(xs.max()-xs.min()+1); bh = int(ys.max()-ys.min()+1); eq = float(np.sqrt(bw*bh))
        col = np.array([244, 245, 238], np.uint8)
        canvas[mask] = (0.82*col + 0.18*canvas[mask]).astype(np.uint8)
        name = f'p3_{i:04d}'
        cv2.imwrite(str(img_dir / f'{name}.png'), canvas)
        (lab_dir / f'{name}.txt').write_text(
            f'0 {((xs.min()+xs.max())/2)/W:.6f} {((ys.min()+ys.max())/2)/H:.6f} {bw/W:.6f} {bh/H:.6f}\n')
        rows.append({'file': f'{name}.png', 'source_type': 'SYNTHETIC_ON_REAL_BG',
                     'background': bgp.name, 'distance_m': round(dist,3),
                     'yaw_deg': round(float(np.degrees(yaw)),1), 'pitch_deg': round(float(np.degrees(pitch)),1),
                     'roll_deg': round(float(np.degrees(roll)),1), 'bbox_w_px': bw, 'bbox_h_px': bh,
                     'equivalent_size_px': round(eq,3), 'bucket': bucket(eq),
                     'pos_x_px': int((xs.min()+xs.max())//2), 'pos_y_px': int((ys.min()+ys.max())//2),
                     'edge_distance_px': int(min(xs.min(), ys.min(), W-1-xs.max(), H-1-ys.max()))})
    man = out / 'metrics/p3_real_bg_manifest.csv'
    with man.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sizes = np.array([r['equivalent_size_px'] for r in rows])
    print(f'P3 generated {len(rows)} images | eq px min={sizes.min():.2f} med={np.median(sizes):.2f} max={sizes.max():.2f}')
    if not a.eval:
        return 0
    os.environ['YOLO_CONFIG_DIR'] = str(repo / 'experiments/shuttle_capability/_scratch/ultralytics_cfg')
    from ultralytics import YOLO
    model = YOLO(str(repo / 'assets/external/_staging/F_yolo/weights/yolo26s.pt'))
    names = model.names; sports = [k for k,v in names.items() if str(v)=='sports ball']
    ev = []
    for r in rows:
        img = cv2.imread(str(img_dir / r['file']))
        res = model.predict(img, imgsz=640, conf=0.05, iou=0.7, verbose=False, device='0')[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,),int)
        H2, W2 = img.shape[:2]
        t = (lab_dir / (r['file'].replace('.png','.txt'))).read_text().split()
        _,xc,yc,bwn,bhn = int(t[0]), *[float(v) for v in t[1:]]
        gt = ((xc-bwn/2)*W2, (yc-bhn/2)*H2, (xc+bwn/2)*W2, (yc+bhn/2)*H2)
        best = max([iou(gt, tuple(b)) for b in boxes], default=0.0)
        best_sb = max([iou(gt, tuple(b)) for b,c in zip(boxes,cls) if int(c) in sports], default=0.0)
        ev.append({**r, 'n_pred': len(boxes), 'best_iou_any': round(float(best),4),
                   'hit_any': int(best>=0.5), 'hit_sports_ball': int(best_sb>=0.5)})
    with (out / 'metrics/p3_real_bg_eval_before.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev[0].keys())); w.writeheader(); w.writerows(ev)
    print(f"{'bucket':>9s} {'n':>4s} {'recall_any':>10s} {'recall_sb':>9s}")
    for lo,hi in BUCKETS:
        lab = f'{lo}-{hi}px' if hi<1e9 else f'>{lo}px'
        sel = [e for e in ev if e['bucket']==lab]
        if not sel: continue
        print(f"{lab:>9s} {len(sel):4d} {sum(e['hit_any'] for e in sel)/len(sel):10.3f} {sum(e['hit_sports_ball'] for e in sel)/len(sel):9.3f}")
    n=len(ev)
    print(f'overall recall_any={sum(e["hit_any"] for e in ev)/n:.3f} recall_sports_ball={sum(e["hit_sports_ball"] for e in ev)/n:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
