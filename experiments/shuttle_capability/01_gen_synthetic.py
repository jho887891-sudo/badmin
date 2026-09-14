#!/usr/bin/env python3
"""01_gen_synthetic.py - controlled 3D capability test set from the archived shuttle mesh.

Software rasterizer (numpy z-buffer) over the real shuttle GLB triangles.  Because the
silhouette is rasterised, the GT bbox is exact and the resulting equivalent_size_px is a
measured quantity, not a guess.  Covers target pixel size, 3D pose (yaw/pitch/roll),
distance, image position and synthetic blur - all labelled source_type=SYNTHETIC_3D.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def find_repo(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / 'env_isaaclab').is_dir() and (c / 'src').is_dir():
            return c
    raise SystemExit('repo root not found')


def load_shuttle_mesh(repo: Path):
    """Return (vertices Nx3, faces Mx3) of the shuttlecock from the archived GLB."""
    sys.path.insert(0, str(repo / 'tools'))
    from usd_glb_common import iter_mesh_nodes, parse_glb, read_accessor
    glb_path = repo / 'assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb'
    glb = parse_glb(glb_path)
    verts, faces = [], []
    offset = 0
    for name, mesh_index in iter_mesh_nodes(glb, include=('Obj_Feather', 'Obj_Cork')):
        prim = glb.json['meshes'][mesh_index]['primitives'][0]
        pos = np.asarray(read_accessor(glb, prim['attributes']['POSITION']), dtype=np.float64)
        idx = [c[0] for c in read_accessor(glb, prim['indices'])]
        verts.append(pos)
        faces.append(np.asarray(idx, dtype=np.int64).reshape(-1, 3) + offset)
        offset += pos.shape[0]
    return np.vstack(verts), np.vstack(faces)


def rot_ypr(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def rasterize(verts_cam, faces, K, W, H):
    """Return a boolean mask of the rendered silhouette (z-buffered)."""
    fx, fy, cx, cy = K
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * verts_cam[:, 0] / verts_cam[:, 2] + cx
        v = fy * verts_cam[:, 1] / verts_cam[:, 2] + cy
    mask = np.zeros((H, W), dtype=bool)
    zbuf = np.full((H, W), np.inf)
    tri_u = u[faces]
    tri_v = v[faces]
    tri_z = verts_cam[faces][:, :, 2]
    for t in range(faces.shape[0]):
        us, vs, zs = tri_u[t], tri_v[t], tri_z[t]
        if np.any(zs <= 1e-6):
            continue
        x0, x1 = int(np.floor(us.min())), int(np.ceil(us.max()))
        y0, y1 = int(np.floor(vs.min())), int(np.ceil(vs.max()))
        x0, x1 = max(x0, 0), min(x1, W - 1)
        y0, y1 = max(y0, 0), min(y1, H - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        d = ((us[1] - us[0]) * (vs[2] - vs[0]) - (us[2] - us[0]) * (vs[1] - vs[0]))
        if abs(d) < 1e-12:
            continue
        w0 = ((us[1] - gx) * (vs[2] - gy) - (us[2] - gx) * (vs[1] - gy)) / d
        w1 = ((us[2] - gx) * (vs[0] - gy) - (us[0] - gx) * (vs[2] - gy)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (z < sub_z)
        sub_z[upd] = z[upd]
        mask[y0:y1 + 1, x0:x1 + 1] |= upd
    return mask


def synth_blur(img, sigma, angle_deg):
    import cv2
    if sigma <= 0:
        return img
    k = int(2 * round(2 * sigma) + 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-9)
    return cv2.filter2D(img, -1, kernel)


def background(W, H, style, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if style == 'flat':
        base = np.full((H, W), 120, np.float32)
    elif style == 'gradient':
        base = 60 + 120 * (yy / max(H - 1, 1))
    elif style == 'court_green':
        base = np.full((H, W), 70, np.float32) + 25 * (yy / max(H - 1, 1))
    else:
        base = 100 + 40 * rng.standard_normal((H, W)).astype(np.float32)
    img = np.stack([base, base, base], axis=-1)
    img *= (0.9 + 0.2 * rng.random())
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--n-pos', type=int, default=6)
    ap.add_argument('--n-dist', type=int, default=14)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--fx', type=float, default=700.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--blur', action='store_true', help='add the SYNTHETIC_BLUR_TEST subset')
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo(Path(__file__).resolve())
    out = Path(args.out) if args.out else repo / 'outputs' / 'shuttle_capability'
    img_dir = out / 'synthetic_3d' / 'images'
    lab_dir = out / 'annotations' / 'synthetic_3d'
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    rng = np.random.default_rng(args.seed)
    verts, faces = load_shuttle_mesh(repo)
    H = W = args.imgsz
    fx = fy = args.fx
    cx, cy = W / 2.0, H / 2.0
    K = (fx, fy, cx, cy)
    print(f'mesh: {verts.shape[0]} verts / {faces.shape[0]} faces | extent z=[{verts[:,2].min():.4f},{verts[:,2].max():.4f}] m')

    positions = [(0.0, 0.0), (-0.35, -0.30), (0.35, -0.30), (-0.35, 0.30), (0.35, 0.30), (0.0, -0.38)]
    distances = np.geomspace(0.8, 26.0, args.n_dist)
    styles = ['flat', 'gradient', 'court_green', 'noise']
    rows = []
    idx = 0
    for di, dist in enumerate(distances):
        for pi, (ox, oy) in enumerate(positions[:args.n_pos]):
            yaw, pitch, roll = rng.uniform(-np.pi, np.pi, 3)
            R = rot_ypr(yaw, pitch, roll)
            center = verts.mean(axis=0)
            v_rel = (verts - center) @ R.T
            v_cam = v_rel + np.array([ox * dist * 0.35, oy * dist * 0.35, dist])
            mask = rasterize(v_cam, faces, K, W, H)
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            eq = float(np.sqrt(bw * bh))
            if eq < 0.5:
                continue
            img = background(W, H, styles[(di + pi) % len(styles)], rng)
            img[mask] = np.array([245, 245, 235], dtype=np.uint8)
            blur_tag = 'none'
            if args.blur and eq >= 6.0 and (di + pi) % 3 == 0:
                sigma = float(np.clip(rng.uniform(1.0, 3.0), 1.0, 3.0))
                img = synth_blur(img, sigma, float(rng.uniform(0, 180)))
                blur_tag = f'synthetic_sigma{sigma:.1f}'
            name = f'syn_{idx:05d}'
            cv2.imwrite(str(img_dir / f'{name}.png'), img)
            (lab_dir / f'{name}.txt').write_text(
                f'0 {(xs.min()+bw/2)/W:.6f} {(ys.min()+bh/2)/H:.6f} {bw/W:.6f} {bh/H:.6f}\n')
            rows.append({'file': f'{name}.png', 'gt_label': f'{name}.txt', 'source_type': 'SYNTHETIC_3D',
                         'distance_m': round(float(dist), 4), 'yaw_deg': round(float(np.degrees(yaw)), 2),
                         'pitch_deg': round(float(np.degrees(pitch)), 2), 'roll_deg': round(float(np.degrees(roll)), 2),
                         'pos_x_px': int((xs.min() + xs.max()) // 2), 'pos_y_px': int((ys.min() + ys.max()) // 2),
                         'bbox_w_px': bw, 'bbox_h_px': bh, 'equivalent_size_px': round(eq, 3),
                         'blur': blur_tag, 'background': styles[(di + pi) % len(styles)], 'imgsz': W})
            idx += 1

    man = out / 'metrics' / 'synthetic_manifest.csv'
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sizes = np.array([r['equivalent_size_px'] for r in rows])
    print(f'rendered {len(rows)} images | equivalent_size_px min={sizes.min():.2f} median={np.median(sizes):.2f} max={sizes.max():.2f}')
    print('wrote', man)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
