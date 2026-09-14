#!/usr/bin/env python3
"""GLB-driven shuttlecock renderer for capability / training data generation.

Asset truth (measured 2026-09-14 on badminton_racket_and_shuttlecock_low_poly.glb):
  * shuttlecock = meshes Obj_Feather (3490 v / 1920 f) + Obj_Cork (3008 v / 1502 f)
  * both carry POSITION + NORMAL, and NO TEXCOORD_0; their material[0] 'White' has
    baseColorFactor=None and no baseColorTexture, so texture-mapped appearance is
    impossible from this asset.  The GLB's only texture belongs to material[1]
    'Strings' (the racket strings) and is irrelevant to the shuttlecock.
  * combined bbox extent 61.9 x 61.9 x 77.8 mm (measured, not assumed)

So the asset supplies geometry + authored normals, and appearance is modelled here:
two-sided wrapped diffuse, an ambient floor (a white shuttle must never render
black without a per-normal fudge), and feather backlight transmission.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLB = (ROOT / "assets" / "external" / "_staging" / "D_racket_shuttle"
               / "original" / "badminton_racket_and_shuttlecock_low_poly.glb")
SHUTTLE_MESH_NAMES = ("Obj_Feather", "Obj_Cork")

# Modelled albedo (there is no texture in the asset to sample).
FEATHER_ALBEDO = np.array([0.93, 0.94, 0.90], dtype=np.float64)
CORK_ALBEDO = np.array([0.86, 0.83, 0.74], dtype=np.float64)

# Floor and key are chosen so a white feather shuttlecock stays the bright object on court:
# at ambient 0.34 a fully back-facing feather face rendered at 80/255, i.e. darker than the
# mid-grey backgrounds it sits on, which showed up as dark smudges at 3-7 px.
AMBIENT = 0.56      # floor: nothing in the scene renders darker than this
KEY = 0.44          # wrapped-diffuse weight
WRAP = 0.5          # diffuse wrap width (feathers are thin, light wraps around them)
FEATHER_TRANSLUCENCY = 0.30   # backlight transmitted through the feather skirt
CORK_TRANSLUCENCY = 0.0

# Measured from the GLB: 77.8 mm is the LENGTH axis (foreshortened when the shuttle flies
# toward the camera), while the projected silhouette is driven by the 61.9 mm skirt.
SHUTTLE_LENGTH_M = 0.0778
MIN_DISTANCE_M = 0.3
# A ground-truth box is the object's FOOTPRINT: every pixel the shuttle touches belongs to
# it.  Requiring half coverage instead would erase the target entirely below ~4 px, which is
# precisely the regime this dataset exists to measure.
SUPPORT_COVERAGE = 1e-6


@dataclass(frozen=True)
class Part:
    """One GLB mesh of the shuttlecock: geometry, authored normals, modelled albedo."""
    name: str
    verts: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    albedo: np.ndarray


def _shuttle_glb_helpers():
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from usd_glb_common import iter_mesh_nodes, parse_glb, read_accessor  # noqa: E402
    return iter_mesh_nodes, parse_glb, read_accessor


def load_shuttle_parts(glb_path: Path = DEFAULT_GLB) -> List[Part]:
    """Decode the shuttlecock meshes (geometry + authored normals) from the GLB."""
    iter_mesh_nodes, parse_glb, read_accessor = _shuttle_glb_helpers()
    glb = parse_glb(Path(glb_path))
    parts: List[Part] = []
    for node_name, mesh_index in iter_mesh_nodes(glb, include=SHUTTLE_MESH_NAMES):
        part_name = next(n for n in SHUTTLE_MESH_NAMES if node_name.startswith(n))
        prim = glb.json["meshes"][mesh_index]["primitives"][0]
        attrs = prim["attributes"]
        verts = np.asarray(read_accessor(glb, attrs["POSITION"]), dtype=np.float64)
        normals = np.asarray(read_accessor(glb, attrs["NORMAL"]), dtype=np.float64)
        faces = np.asarray([c[0] for c in read_accessor(glb, prim["indices"])],
                           dtype=np.int64).reshape(-1, 3)
        albedo = FEATHER_ALBEDO if part_name.endswith("Feather") else CORK_ALBEDO
        parts.append(Part(part_name, verts, normals, faces, albedo))
    if not parts:
        raise ValueError("no shuttlecock meshes found in " + str(glb_path))
    return parts


def shade(normals: np.ndarray, light_dir: np.ndarray, albedo: np.ndarray, *,
          ambient: float = AMBIENT, key: float = KEY,
          translucency: float = 0.0) -> np.ndarray:
    """Shade N surface normals -> RGB in 0..255.

    Two-sided by construction: the diffuse term is wrapped rather than clamped, and a
    separate transmission term lifts faces that point away from the light, which is what
    a thin feather skirt does.  The ambient term is the floor, so no orientation - however
    unfortunate - can drive a white shuttlecock to black.
    """
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.asarray(light_dir, dtype=np.float64).ravel()
    light = light / max(float(np.linalg.norm(light)), 1e-12)
    ndl = n @ light
    wrapped = np.clip((ndl + WRAP) / (1.0 + WRAP), 0.0, 1.0)
    transmitted = np.clip(-ndl, 0.0, 1.0)
    luminance = ambient + key * wrapped + float(translucency) * transmitted
    alb = np.asarray(albedo, dtype=np.float64).reshape(1, 3)
    return np.clip(alb * luminance[:, None] * 255.0, 0.0, 255.0)


def _rasterize_part(part: Part, verts_cam: np.ndarray, K, width: int, height: int,
                    light_dir: np.ndarray, rgb: np.ndarray, coverage: np.ndarray,
                    zbuf: np.ndarray) -> None:
    fx, fy, cx, cy = K
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * verts_cam[:, 0] / verts_cam[:, 2] + cx
        v = fy * verts_cam[:, 1] / verts_cam[:, 2] + cy
    translucency = (FEATHER_TRANSLUCENCY if part.name.endswith("Feather")
                    else CORK_TRANSLUCENCY)
    for face in part.faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        p0, p1, p2 = verts_cam[i0], verts_cam[i1], verts_cam[i2]
        if p0[2] <= 1e-6 or p1[2] <= 1e-6 or p2[2] <= 1e-6:
            continue
        us = np.array([u[i0], u[i1], u[i2]])
        vs = np.array([v[i0], v[i1], v[i2]])
        zs = np.array([p0[2], p1[2], p2[2]])
        x0 = max(int(np.floor(us.min())), 0)
        x1 = min(int(np.ceil(us.max())), width - 1)
        y0 = max(int(np.floor(vs.min())), 0)
        y1 = min(int(np.ceil(vs.max())), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        denom = (us[1] - us[0]) * (vs[2] - vs[0]) - (us[2] - us[0]) * (vs[1] - vs[0])
        if abs(denom) < 1e-12:
            continue
        w0 = ((us[1] - gx) * (vs[2] - gy) - (us[2] - gx) * (vs[1] - gy)) / denom
        w1 = ((us[2] - gx) * (vs[0] - gy) - (us[0] - gx) * (vs[2] - gy)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
        update = inside & (z < sub_z)
        if not update.any():
            continue
        # Barycentric interpolation of the AUTHORED vertex normals (not a face normal).
        n = (w0[..., None] * part.normals[i0] + w1[..., None] * part.normals[i1]
             + w2[..., None] * part.normals[i2])
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        # Orient every normal toward the camera at that pixel; first order view direction.
        view = np.stack([(gx - cx) / fx, (gy - cy) / fy, np.ones_like(gx)], axis=-1)
        view = view / np.linalg.norm(view, axis=-1, keepdims=True)
        n = np.where((np.sum(n * view, axis=-1) > 0.0)[..., None], -n, n)
        sub_rgb = rgb[y0:y1 + 1, x0:x1 + 1]
        sub_rgb[update] = shade(n[update], light_dir, part.albedo,
                                translucency=translucency)
        coverage[y0:y1 + 1, x0:x1 + 1][update] = 1.0
        sub_z[update] = z[update]


def render_shuttle(parts: Sequence[Part], rotation: np.ndarray, distance_m: float,
                   pixel_xy: Tuple[float, float], K: Tuple[float, float, float, float],
                   width: int, height: int, *, light_dir: np.ndarray,
                   supersample: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Render the shuttle at a controlled pixel size.

    Rasterised at supersample-times resolution and box-filtered back down, so edges carry
    fractional coverage instead of a hard boolean silhouette, and shading is smooth across
    faces rather than constant per triangle.
    Returns (rgb float32 HxWx3 in 0..255, coverage float32 HxW in 0..1).
    """
    ss = int(supersample)
    if ss < 1:
        raise ValueError("supersample must be >= 1")
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    fx, fy, cx, cy = K
    big_w, big_h = width * ss, height * ss
    K_big = (fx * ss, fy * ss, cx * ss, cy * ss)
    rgb = np.zeros((big_h, big_w, 3), dtype=np.float32)
    coverage = np.zeros((big_h, big_w), dtype=np.float32)
    zbuf = np.full((big_h, big_w), np.inf, dtype=np.float32)
    center = np.mean([np.asarray(p.verts).mean(axis=0) for p in parts], axis=0)
    px, py = pixel_xy
    R = np.asarray(rotation, dtype=np.float64)
    for part in parts:
        v_rel = (np.asarray(part.verts, dtype=np.float64) - center) @ R.T
        v_cam = v_rel + np.array([(px - width / 2.0) * distance_m / fx,
                                  (py - height / 2.0) * distance_m / fx,
                                  distance_m])
        _rasterize_part(part, v_cam, K_big, big_w, big_h, light_dir, rgb, coverage, zbuf)
    rgb_down = rgb.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
    alpha = coverage.reshape(height, ss, width, ss).mean(axis=(1, 3))
    return rgb_down.astype(np.float32), alpha.astype(np.float32)


def mask_from_coverage(alpha: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Visible-object mask: pixels at least half covered by the shuttle."""
    return np.asarray(alpha) >= thresh


def yolo_bbox_from_mask(mask: np.ndarray, width: int, height: int
                        ) -> Optional[Tuple[float, float, float, float]]:
    """Normalised (cx, cy, w, h) of the tight box around the mask, or None if empty."""
    ys, xs = np.nonzero(np.asarray(mask))
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    return ((x0 + x1 + 1) / 2.0 / width, (y0 + y1 + 1) / 2.0 / height,
            box_w / float(width), box_h / float(height))


def split_backgrounds(paths: Sequence[Path], *, val_count: int, seed: int
                      ) -> Tuple[List[Path], List[Path]]:
    """Deterministically split background images into disjoint (train, val) pools."""
    pool = sorted(Path(p) for p in paths)
    if val_count > len(pool):
        raise ValueError("val_count=" + str(val_count) + " exceeds pool of " + str(len(pool)))
    order = np.random.default_rng(seed).permutation(len(pool))
    val = sorted(pool[i] for i in order[:val_count])
    train = sorted(pool[i] for i in order[val_count:])
    return train, val


def calibrate_distance(parts: Sequence[Part], rotation: np.ndarray, target_px: float,
                       pixel_xy: Tuple[float, float], K, width: int, height: int, *,
                       measure_supersample: int = 3, max_iterations: int = 8,
                       tolerance: float = 0.05) -> float:
    """Find the camera distance whose MEASURED ground-truth box is target_px.

    The naive fx * extent / target formula mis-sizes this asset twice over: the 77.8 mm
    extent is the length axis, which is foreshortened in projection, and a sparse feather
    skirt occupies much less area than its vertex hull.  So measure the rendered footprint
    and iterate on the ratio; the renderer is the ground truth, not the formula.

    measure_supersample MUST match the supersample the caller ships, otherwise calibration
    tunes a quantity nobody renders: measuring at 2x while shipping 3x put an 8 px target at
    10 px, i.e. worse than not calibrating at all.  The best distance seen is returned, which
    also absorbs the +-1 px quantisation that dominates at the small end.
    """
    if target_px <= 0:
        raise ValueError("target_px must be > 0")
    # Coverage is pure geometry, so calibration does not depend on the lighting.
    flat_light = np.array([0.0, 0.0, -1.0])
    distance = max(float(K[0]) * SHUTTLE_LENGTH_M / target_px, MIN_DISTANCE_M)
    best_distance, best_error = distance, float("inf")
    for _ in range(max_iterations):
        _rgb, alpha = render_shuttle(parts, rotation, distance, pixel_xy, K, width, height,
                                     light_dir=flat_light, supersample=measure_supersample)
        bbox = yolo_bbox_from_mask(mask_from_coverage(alpha, SUPPORT_COVERAGE),
                                   width, height)
        if bbox is None:
            distance = max(distance * 0.5, MIN_DISTANCE_M)
            continue
        achieved = math.sqrt(bbox[2] * width * bbox[3] * height)
        if achieved <= 0.0:
            distance = max(distance * 0.5, MIN_DISTANCE_M)
            continue
        error = abs(achieved - target_px) / target_px
        if error < best_error:
            best_error, best_distance = error, distance
        if error <= tolerance:
            return distance
        distance = max(distance * (achieved / target_px), MIN_DISTANCE_M)
    return best_distance


NOISE_SIGMA_RANGE = (1.0, 5.0)
SHADOW_GAIN_RANGE = (0.0, 0.35)
MOTION_PX_CHOICES = (0.0, 0.0, 1.0, 2.0, 4.0, 7.0)


@dataclass(frozen=True)
class Sample:
    """One generated image plus everything needed to reproduce it and audit its label."""
    image: np.ndarray
    label: str
    record: dict
    background: np.ndarray
    rgb: np.ndarray
    alpha: np.ndarray


def _cv2():
    import cv2
    return cv2


def rot_ypr(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Yaw-pitch-roll rotation matrix, matching the convention of the existing scripts."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


def imread_unicode(path):
    """Read an image, tolerating non-ASCII paths (this repository lives under one).

    cv2.imread returns None for such paths on Windows, so decode through a byte buffer.
    """
    cv2 = _cv2()
    p = Path(path)
    if not p.is_file():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image, *, quality: int = 95) -> bool:
    """Write an image, tolerating non-ASCII paths, for the same reason as imread_unicode."""
    cv2 = _cv2()
    p = Path(path)
    suffix = p.suffix.lower() or ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if suffix in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(suffix, np.asarray(image), params)
    if not ok:
        raise IOError("could not encode " + str(p))
    buf.tofile(str(p))
    return True


def find_blank_backgrounds(paths: Sequence[Path], *, min_std: float = 6.0,
                           min_mean: float = 12.0) -> List[Path]:
    """Return backgrounds carrying no scene: black, unreadable, or a single flat colour.

    A transparent PNG flattened to black looks like a valid file everywhere until you open
    it - tbg_018.jpg and tbg_005.png both entered the pool that way.
    """
    blank: List[Path] = []
    for path in paths:
        img = imread_unicode(path)
        if img is None:
            blank.append(Path(path))
            continue
        if float(img.std()) < min_std or float(img.mean()) < min_mean:
            blank.append(Path(path))
    return blank


def find_leaked_backgrounds(candidates: Sequence[Path], frozen: Sequence[Path], *,
                            threshold: float = 0.95) -> List[Tuple[Path, Path, float]]:
    """Return (candidate, frozen match, correlation) for backgrounds shared with a frozen set.

    Compared by downscaled luminance correlation rather than by filename, because the same
    source picture can be re-encoded under a different name and format - which is exactly how
    tbg_011.jpg and bg_021.jpg slipped past a name-based check (measured r = 1.0000).
    """
    frozen_paths = [Path(p) for p in frozen]
    if not frozen_paths:
        return []
    cv2 = _cv2()

    def signature(path):
        img = imread_unicode(path)
        if img is None:
            return None
        small = cv2.resize(img, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray -= gray.mean()
        norm = float(np.linalg.norm(gray))
        return gray / norm if norm > 0.0 else None

    frozen_sigs = []
    for path in frozen_paths:
        sig = signature(path)
        if sig is not None:
            frozen_sigs.append((path, sig))
    if not frozen_sigs:
        return []
    leaks: List[Tuple[Path, Path, float]] = []
    for candidate in candidates:
        sig = signature(candidate)
        if sig is None:
            continue
        best = max(((float((sig * other).sum()), path) for path, other in frozen_sigs),
                   key=lambda pair: pair[0])
        if best[0] >= threshold:
            leaks.append((Path(candidate), best[1], best[0]))
    return leaks


def prepare_background(background: np.ndarray, width: int, height: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Random crop of the requested size, upscaling first when the image is too small."""
    cv2 = _cv2()
    bg = np.asarray(background)
    if bg.ndim == 2:
        bg = np.stack([bg] * 3, axis=-1)
    h, w = bg.shape[:2]
    if w < width or h < height:
        bg = cv2.resize(bg, (max(width, w), max(height, h)), interpolation=cv2.INTER_LINEAR)
        h, w = bg.shape[:2]
    x0 = int(rng.integers(0, w - width + 1)) if w > width else 0
    y0 = int(rng.integers(0, h - height + 1)) if h > height else 0
    return np.ascontiguousarray(bg[y0:y0 + height, x0:x0 + width])


def _motion_kernel(motion_px: float, angle_deg: float) -> np.ndarray:
    cv2 = _cv2()
    k = int(2 * motion_px + 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    mat = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (k, k))
    return kernel / max(float(kernel.sum()), 1e-9)


def composite(background: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, *,
              shadow_gain: float = 0.0, motion_px: float = 0.0,
              motion_angle_deg: float = 0.0, noise_sigma: float = 0.0,
              noise_seed: int = 0) -> np.ndarray:
    """Alpha-composite the rendered shuttle onto a background in a fixed order.

    Deterministic for a given parameter set - the noise generator is seeded from the
    recorded noise_seed - so a manifest row can reproduce its own image exactly.
    """
    cv2 = _cv2()
    a = np.asarray(alpha, dtype=np.float32)[..., None]
    out = (np.asarray(background, dtype=np.float32) * (1.0 - a)
           + np.asarray(rgb, dtype=np.float32) * a)
    if shadow_gain > 0.0:
        # The shuttle casts the shadow onto the background, so scale it with the object and
        # never let it fall back on the object itself: a fixed (6, 8) offset with sigma 4
        # darkened a 7 px shuttle by 26% and swallowed it whole below that.
        extent = max(math.sqrt(float(np.count_nonzero(alpha))), 1.0)
        offset = int(np.clip(round(0.8 * extent), 1, 8))
        blur = float(np.clip(0.35 * extent, 0.6, 4.0))
        shadow = np.roll(np.asarray(alpha, dtype=np.float32), shift=(offset, offset),
                         axis=(0, 1))
        shadow = cv2.GaussianBlur(shadow, (0, 0), blur)
        shadow = shadow * (np.asarray(alpha) <= 0.0)
        out *= (1.0 - float(shadow_gain) * np.clip(shadow, 0.0, 1.0))[..., None]
    if motion_px >= 1.5:
        out = cv2.filter2D(out, -1, _motion_kernel(motion_px, motion_angle_deg))
    if noise_sigma > 0.0:
        noise_rng = np.random.default_rng(int(noise_seed))
        out = out + noise_rng.normal(0.0, float(noise_sigma), out.shape)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def render_sample(parts: Sequence[Part], background: np.ndarray, *, target_px: float,
                  seed: int, split: str, name: str, width: int = 960, height: int = 960,
                  focal_px: float = 700.0, supersample: int = 3,
                  background_name: str = "") -> Sample:
    """Render one labelled sample: calibrated size, random pose, composited appearance.

    Every drawn value is recorded, including the ones that were previously dropped from the
    manifest (noise_sigma, motion_angle, the noise seed), so the record reproduces the image.
    """
    rng = np.random.default_rng(int(seed))
    K = (float(focal_px), float(focal_px), width / 2.0, height / 2.0)
    bg = prepare_background(background, width, height, rng)
    yaw, pitch, roll = (float(v) for v in rng.uniform(-math.pi, math.pi, 3))
    R = rot_ypr(yaw, pitch, roll)
    px = float(rng.uniform(0.05, 0.95) * width)
    py = float(rng.uniform(0.05, 0.95) * height)
    distance = calibrate_distance(parts, R, target_px, (px, py), K, width, height,
                                  measure_supersample=supersample)
    azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
    light = np.array([math.cos(azimuth) * 0.45, -0.55, -0.70])
    light = light / np.linalg.norm(light)
    rgb, alpha = render_shuttle(parts, R, distance, (px, py), K, width, height,
                                light_dir=light, supersample=supersample)
    shadow_gain = float(rng.uniform(*SHADOW_GAIN_RANGE))
    motion_px = float(rng.choice(MOTION_PX_CHOICES))
    motion_angle = float(rng.uniform(0.0, 180.0))
    noise_sigma = float(rng.uniform(*NOISE_SIGMA_RANGE))
    noise_seed = int(rng.integers(0, 2 ** 31 - 1))
    image = composite(bg, rgb, alpha, shadow_gain=shadow_gain, motion_px=motion_px,
                      motion_angle_deg=motion_angle, noise_sigma=noise_sigma,
                      noise_seed=noise_seed)
    mask = mask_from_coverage(alpha, SUPPORT_COVERAGE)
    bbox = yolo_bbox_from_mask(mask, width, height)
    if bbox is None:
        raise ValueError("target_px=" + str(target_px) + " rendered nothing")
    cx, cy, bw, bh = bbox
    label = "0 " + f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" + "\n"
    ys, xs = np.nonzero(mask)
    record = {
        "file": name + ".jpg", "split": split, "source_type": "SYNTHETIC_HIFI_3D",
        "background": background_name, "target_px": round(float(target_px), 3),
        "equiv_size_px": round(math.sqrt(bw * width * bh * height), 3),
        "distance_m": round(float(distance), 4),
        "bbox_w_px": int(xs.max() - xs.min() + 1),
        "bbox_h_px": int(ys.max() - ys.min() + 1),
        "pos_x_px": int((xs.min() + xs.max()) // 2),
        "pos_y_px": int((ys.min() + ys.max()) // 2),
        "coverage_sum": round(float(np.asarray(alpha).sum()), 3),
        "solid_px": int((np.asarray(alpha) >= 0.5).sum()),
        "yaw_deg": round(math.degrees(yaw), 2), "pitch_deg": round(math.degrees(pitch), 2),
        "roll_deg": round(math.degrees(roll), 2),
        "light_azimuth_deg": round(math.degrees(azimuth), 2),
        # Exact, not rounded: these four replay through composite(), and rounding them
        # silently breaks the guarantee that a record reproduces its own image.
        "shadow_gain": shadow_gain, "motion_px": motion_px,
        "motion_angle_deg": motion_angle,
        "noise_sigma": noise_sigma, "noise_seed": noise_seed,
        "imgsz": width, "supersample": supersample,
    }
    return Sample(image=image, label=label, record=record, background=bg,
                  rgb=rgb, alpha=alpha)
