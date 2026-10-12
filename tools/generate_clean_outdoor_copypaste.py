#!/usr/bin/env python3
"""Offline alpha Copy-Paste generator and QA for CLEAN_OUTDOOR_CP_V1 (Task 6).

Only two ingredients are allowed: an ETH train-only real shuttlecock foreground with a human-approved instance
mask, and a COCO train2017 CLEAN outdoor ROI that a human approved. The foreground RGB and its alpha travel
through ONE RGBA pipeline so the mask can never drift away from the pixels it belongs to, the composite may
only touch pixels inside the approved ROI, and the YOLO box is derived from the alpha that is actually visible
after compositing - never from the intended placement.

A composition that loses the object, or whose mask is a hard-edged rectangle (a visible halo), is rejected and
logged rather than shipped with a positive label.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

FOREGROUND_SOURCE = "eth_train"
BACKGROUND_SOURCE = "coco_train2017"
APPROVED = "APPROVED"

MIN_VISIBLE_ALPHA = 0.25
MIN_VISIBLE_PIXELS = 4
MAX_RECTANGULAR_HALO_RATIO = 0.90
JPEG_QUALITY = 95

MANIFEST_COLUMNS = (
    "out_id", "image_path", "label_path", "image_sha256", "label_sha256", "width", "height",
    "bbox_xyxy", "bbox_wh", "visible_pixels", "foreground_id", "foreground_sha256", "foreground_source",
    "background_id", "background_sha256", "background_source", "coco_image_id", "clean_roi",
    "scale", "angle", "center_x", "center_y", "opacity", "seed", "label_nonempty",
)
REJECTION_COLUMNS = ("out_id", "reason", "detail")


class CopyPasteRejected(RuntimeError):
    """Raised for a single composition that must not be written (never fatal for the whole pool)."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _as_rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """One RGBA tile: RGB and alpha are transformed together, which is what keeps them in the same frame."""
    rgb = np.asarray(rgb)
    alpha = np.asarray(alpha, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CopyPasteRejected("foreground RGB must be HxWx3, got %s" % (rgb.shape,))
    if alpha.shape != rgb.shape[:2]:
        raise CopyPasteRejected("alpha %s does not match RGB %s" % (alpha.shape, rgb.shape[:2]))
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([np.asarray(rgb, dtype=np.uint8), alpha_u8]), "RGBA")


def rectangular_halo_ratio(alpha: np.ndarray, bbox, opaque_threshold: float = 0.95) -> float:
    """Share of the bbox that is fully opaque. A rectangle mask gives ~1.0; a shuttle silhouette gives much less."""
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    patch = np.asarray(alpha, dtype=np.float32)[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float((patch >= opaque_threshold).sum()) / float(patch.size)


def bbox_from_visible_alpha(alpha: np.ndarray, min_support: float = MIN_VISIBLE_ALPHA):
    """Bounding box of what is actually visible after compositing; None when nothing survives."""
    support = np.asarray(alpha, dtype=np.float32) >= float(min_support)
    if not support.any():
        return None
    rows = np.where(support.any(axis=1))[0]
    cols = np.where(support.any(axis=0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def composite_real_shuttle(fg_rgb, fg_alpha, bg_rgb, transform, roi, seed: int = 0):
    """Composite one real shuttlecock into one approved CLEAN ROI. Returns (image, visible_alpha, metadata)."""
    bg = np.asarray(bg_rgb, dtype=np.uint8)
    if bg.ndim != 3 or bg.shape[2] != 3:
        raise CopyPasteRejected("background RGB must be HxWx3, got %s" % (bg.shape,))
    height, width = bg.shape[:2]
    x0, y0, x1, y1 = (int(value) for value in roi)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise CopyPasteRejected("clean ROI %s is outside the %dx%d background" % (roi, width, height))

    scale = float(transform.get("scale", 1.0))
    if scale <= 0:
        raise CopyPasteRejected("scale must be positive, got %r" % scale)
    opacity = float(transform.get("opacity", 1.0))
    if not 0.0 < opacity <= 1.0:
        raise CopyPasteRejected("opacity must be in (0, 1], got %r" % opacity)

    tile = _as_rgba(fg_rgb, fg_alpha)
    tile = tile.resize((max(1, int(round(tile.width * scale))), max(1, int(round(tile.height * scale)))),
                       Image.BICUBIC)
    angle = float(transform.get("angle", 0.0))
    if angle:
        tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
    blur = float(transform.get("blur", 0.0))
    if blur > 0:
        tile = tile.filter(ImageFilter.GaussianBlur(blur))

    tile_array = np.asarray(tile, dtype=np.float32) / 255.0
    tile_rgb, tile_alpha = tile_array[..., :3], tile_array[..., 3] * opacity

    center_x = float(transform.get("center_x", (x0 + x1) / 2.0))
    center_y = float(transform.get("center_y", (y0 + y1) / 2.0))
    left = int(round(center_x - tile.width / 2.0))
    top = int(round(center_y - tile.height / 2.0))

    paste_x0, paste_y0 = max(left, x0), max(top, y0)
    paste_x1, paste_y1 = min(left + tile.width, x1), min(top + tile.height, y1)

    out = bg.astype(np.float32).copy()
    visible = np.zeros((height, width), dtype=np.float32)
    if paste_x1 > paste_x0 and paste_y1 > paste_y0:
        src_x0, src_y0 = paste_x0 - left, paste_y0 - top
        src_x1, src_y1 = src_x0 + (paste_x1 - paste_x0), src_y0 + (paste_y1 - paste_y0)
        sub_alpha = tile_alpha[src_y0:src_y1, src_x0:src_x1]
        sub_rgb = tile_rgb[src_y0:src_y1, src_x0:src_x1]
        region = out[paste_y0:paste_y1, paste_x0:paste_x1]
        out[paste_y0:paste_y1, paste_x0:paste_x1] = region * (1.0 - sub_alpha[..., None]) + sub_rgb * 255.0 * sub_alpha[..., None]
        visible[paste_y0:paste_y1, paste_x0:paste_x1] = sub_alpha

    rng = random.Random(int(seed))
    noise = float(transform.get("noise", 0.0))
    if noise > 0:
        patch = out[y0:y1, x0:x1]
        perturbation = np.array([[rng.gauss(0.0, noise) for _ in range(patch.shape[1])] for _ in range(patch.shape[0])],
                                dtype=np.float32)
        out[y0:y1, x0:x1] = np.clip(patch + perturbation[..., None], 0, 255)

    image = np.clip(out, 0, 255).astype(np.uint8)
    bbox = bbox_from_visible_alpha(visible)
    metadata = {
        "scale": scale,
        "angle": angle,
        "opacity": opacity,
        "seed": int(seed),
        "roi": [x0, y0, x1, y1],
        "bbox_xyxy": list(bbox) if bbox else None,
        "bbox_wh": [bbox[2] - bbox[0], bbox[3] - bbox[1]] if bbox else [0, 0],
        "visible_pixels": int((visible >= MIN_VISIBLE_ALPHA).sum()),
        "rectangular_halo_ratio": rectangular_halo_ratio(visible, bbox),
        "center_x": center_x,
        "center_y": center_y,
    }
    return image, visible, metadata


def write_yolo_label(bbox_xyxy, image_size, class_id: int = 0) -> str:
    """Single-class YOLO label with finite, in-range normalised coordinates."""
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        raise CopyPasteRejected("image size must be positive, got %s" % (image_size,))
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        raise CopyPasteRejected("degenerate bbox %s" % (bbox_xyxy,))
    values = ((x0 + x1) / 2.0 / width, (y0 + y1) / 2.0 / height, box_w / width, box_h / height)
    if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in values):
        raise CopyPasteRejected("normalised bbox out of range: %s" % (values,))
    return "%d %.6f %.6f %.6f %.6f\n" % (int(class_id), values[0], values[1], values[2], values[3])


def _check_provenance(foreground_row: dict, background_row: dict) -> None:
    if str(foreground_row.get("source", "")) != FOREGROUND_SOURCE:
        raise CopyPasteRejected("foreground must come from %s, got %r" % (FOREGROUND_SOURCE, foreground_row.get("source")))
    if str(background_row.get("source", "")) != BACKGROUND_SOURCE:
        raise CopyPasteRejected("background must come from %s, got %r" % (BACKGROUND_SOURCE, background_row.get("source")))
    if str(foreground_row.get("qa_status", "")).upper() != APPROVED:
        raise CopyPasteRejected("foreground %s is not human-approved" % foreground_row.get("sample_id"))
    if str(background_row.get("qa_status", "")).upper() != APPROVED:
        raise CopyPasteRejected("background %s is not human-approved" % background_row.get("coco_image_id"))
    if not str(background_row.get("clean_roi", "")).strip():
        raise CopyPasteRejected("background %s has no approved CLEAN ROI" % background_row.get("coco_image_id"))


def _parse_roi(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    parts = [int(float(part)) for part in str(value).replace(";", ",").split(",") if str(part).strip()]
    if len(parts) != 4:
        raise CopyPasteRejected("clean ROI must have four numbers, got %r" % (value,))
    return parts


def generate_copy_paste_pool(foregrounds: pd.DataFrame, backgrounds: pd.DataFrame, schedule: pd.DataFrame,
                             output_dir, save_images: bool = True) -> pd.DataFrame:
    """Write every accepted composite; return the manifest. Rejections are returned in the frame attrs."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    fg_index = {str(row["sample_id"]): row for row in foregrounds.to_dict("records")}
    bg_index = {str(row["coco_image_id"]): row for row in backgrounds.to_dict("records")}

    rows: list = []
    rejections: list = []
    for entry in schedule.to_dict("records"):
        out_id = str(entry.get("out_id"))
        try:
            foreground_row = fg_index[str(entry.get("foreground_id"))]
            background_row = bg_index[str(entry.get("background_id"))]
            _check_provenance(foreground_row, background_row)

            fg_rgb, fg_alpha = _load_foreground(foreground_row)
            bg_rgb = _load_background(background_row)
            roi = _parse_roi(background_row["clean_roi"])
            transform = {
                "scale": entry.get("scale", 1.0),
                "angle": entry.get("angle", 0.0),
                "opacity": entry.get("opacity", 1.0),
                "center_x": entry.get("center_x"),
                "center_y": entry.get("center_y"),
                "blur": entry.get("blur", 0.0),
                "noise": entry.get("noise", 0.0),
            }
            transform = {key: value for key, value in transform.items() if value is not None and value != ""}
            seed = int(entry.get("seed", 0))
            image, visible, meta = composite_real_shuttle(fg_rgb, fg_alpha, bg_rgb, transform, roi, seed)

            if meta["visible_pixels"] < MIN_VISIBLE_PIXELS or meta["bbox_xyxy"] is None:
                raise CopyPasteRejected("the shuttlecock is not visible after compositing")
            if meta["rectangular_halo_ratio"] > MAX_RECTANGULAR_HALO_RATIO:
                raise CopyPasteRejected(
                    "mask is a hard-edged rectangle (%.3f of the bbox fully opaque)" % meta["rectangular_halo_ratio"]
                )

            label = write_yolo_label(meta["bbox_xyxy"], (image.shape[1], image.shape[0]), class_id=0)
            image_path = images_dir / ("%s.jpg" % out_id)
            label_path = labels_dir / ("%s.txt" % out_id)
            image_bytes = b""
            if save_images:
                from io import BytesIO
                buffer = BytesIO()
                Image.fromarray(image).save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
                image_bytes = buffer.getvalue()
                image_path.write_bytes(image_bytes)
                label_path.write_text(label, encoding="utf-8")
            else:
                image_bytes = image.tobytes()

            rows.append({
                "out_id": out_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_sha256": sha256_bytes(image_bytes),
                "label_sha256": sha256_bytes(label.encode("utf-8")),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "bbox_xyxy": json.dumps(meta["bbox_xyxy"]),
                "bbox_wh": json.dumps(meta["bbox_wh"]),
                "visible_pixels": meta["visible_pixels"],
                "foreground_id": foreground_row["sample_id"],
                "foreground_sha256": foreground_row.get("sha256"),
                "foreground_source": foreground_row.get("source"),
                "background_id": background_row.get("coco_image_id"),
                "background_sha256": background_row.get("sha256"),
                "background_source": background_row.get("source"),
                "coco_image_id": background_row.get("coco_image_id"),
                "clean_roi": json.dumps(roi),
                "scale": meta["scale"],
                "angle": meta["angle"],
                "center_x": meta["center_x"],
                "center_y": meta["center_y"],
                "opacity": meta["opacity"],
                "seed": meta["seed"],
                "label_nonempty": bool(label.strip()),
            })
        except CopyPasteRejected as exc:
            rejections.append({"out_id": out_id, "reason": "composite_rejected", "detail": str(exc)})

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest.attrs["rejections"] = pd.DataFrame(rejections, columns=list(REJECTION_COLUMNS))
    return manifest


def _load_foreground(row: dict):
    image = Image.open(row["image_path"])
    alpha_path = row.get("alpha_path")
    if alpha_path:
        alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float32) / 255.0
    else:
        alpha = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return np.asarray(image.convert("RGB"), dtype=np.uint8), alpha


def _load_background(row: dict):
    return np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline alpha copy-paste generator (Task 6)")
    parser.add_argument("--foregrounds", type=Path, required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-rejections", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = generate_copy_paste_pool(
        pd.read_csv(args.foregrounds, keep_default_na=False, dtype=str),
        pd.read_csv(args.backgrounds, keep_default_na=False, dtype=str),
        pd.read_csv(args.schedule, keep_default_na=False, dtype=str),
        args.output_dir,
    )
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_manifest, index=False)
    manifest.attrs["rejections"].to_csv(args.out_rejections, index=False)
    print(json.dumps({"composites": int(len(manifest)),
                      "rejections": int(len(manifest.attrs["rejections"])),
                      "unique_coco_ids": int(manifest["coco_image_id"].nunique()) if len(manifest) else 0},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
