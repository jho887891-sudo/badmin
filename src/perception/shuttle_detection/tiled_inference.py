"""Deterministic full-frame tiling for tiny-object YOLO inference.

No ground-truth boxes, tracker predictions or human ROIs enter the tiling policy.
Each crop is inferred independently; candidate boxes are remapped to the
original image before class-aware cross-tile NMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math

import numpy as np


@dataclass(frozen=True)
class Tile:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class TileDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    end = length - tile_size
    starts = list(range(0, end + 1, stride))
    if starts[-1] != end:
        starts.append(end)
    return starts


def generate_tiles(width: int, height: int, tile_size: int,
                   overlap: float = .25) -> list[Tile]:
    """Cover image including right/bottom edges, in stable row-major order."""
    if any(type(v) is not int or v <= 0 for v in (width, height, tile_size)):
        raise ValueError("width, height, tile_size must be positive integers")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must lie in [0,1)")
    stride = max(1, int(round(tile_size * (1 - overlap))))
    return [Tile(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in _starts(height, tile_size, stride)
            for x in _starts(width, tile_size, stride)]


def map_and_filter(rows: Iterable[Sequence[float]], tile: Tile, *, width: int,
                   height: int) -> list[TileDetection]:
    """Map crop-local xyxy to full-image pixels; discard degenerate boxes."""
    kept = []
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(float(v)) for v in row):
            continue
        x1,y1,x2,y2,confidence,cls = [float(v) for v in row]
        if not (0 <= confidence <= 1 and cls >= 0 and cls.is_integer()):
            continue
        lx1 = min(tile.right, max(tile.left, x1 + tile.left))
        ly1 = min(tile.bottom, max(tile.top, y1 + tile.top))
        lx2 = min(tile.right, max(tile.left, x2 + tile.left))
        ly2 = min(tile.bottom, max(tile.top, y2 + tile.top))
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        kept.append(TileDetection((lx1,ly1,lx2,ly2),confidence,int(cls)))
    return kept


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    left, top = max(a[0],b[0]),max(a[1],b[1])
    right,bottom = min(a[2],b[2]),min(a[3],b[3])
    inter = max(0.,right-left)*max(0.,bottom-top)
    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter>0 else 0.


def class_aware_nms(detections: Iterable[TileDetection], *,
                    iou_threshold: float, max_det: int) -> list[TileDetection]:
    """Greedy score-sorted NMS on original-image coords, without class conflation."""
    if not 0 <= iou_threshold <= 1 or max_det < 1:
        raise ValueError("invalid global NMS configuration")
    ordered=sorted(detections,key=lambda det: -det.confidence)
    kept=[]
    for det in ordered:
        if any(det.class_id==old.class_id and
               _iou(det.xyxy,old.xyxy)>iou_threshold for old in kept):
            continue
        kept.append(det)
        if len(kept)>=max_det:
            break
    return kept


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value,"detach"):
        value=value.detach()
    if hasattr(value,"cpu"):
        value=value.cpu()
    return np.asarray(value)


def predict_tiled(model: Any, image: np.ndarray, *, tile_size: int,
                  overlap: float, imgsz: int, conf: float, iou: float,
                  device: str, max_det: int) -> tuple[list[TileDetection], dict]:
    """Run actual model.predict once per image tile, not GT-assisted crops."""
    if image.ndim!=3 or image.shape[0]<1 or image.shape[1]<1:
        raise ValueError("image must be HxWxC")
    height,width=image.shape[:2]
    tiles=generate_tiles(width,height,tile_size,overlap)
    candidates=[]
    for tile in tiles:
        crop=image[tile.top:tile.bottom,tile.left:tile.right]
        results=model.predict(source=crop,imgsz=imgsz,conf=conf,iou=iou,
                              device=device,max_det=max_det,verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes)==0:
            continue
        boxes=results[0].boxes
        xyxy=_as_numpy(boxes.xyxy).reshape(-1,4)
        scores=_as_numpy(boxes.conf).reshape(-1)
        classes=_as_numpy(boxes.cls).reshape(-1)
        if not (len(xyxy)==len(scores)==len(classes)):
            raise ValueError("YOLO output xyxy/conf/cls length mismatch")
        rows=np.column_stack((xyxy,scores,classes))
        candidates.extend(map_and_filter(rows,tile,width=width,height=height))
    kept=class_aware_nms(candidates,iou_threshold=iou,max_det=max_det)
    return kept,{"mode":"tiled","tile_size_px":tile_size,"overlap":overlap,
                 "tiles":len(tiles),"candidate_count":len(candidates),
                 "kept_count":len(kept),"uses_gt_roi":False}
