#!/usr/bin/env python3
"""Controlled single-variable sweeps for the shuttlecock capability test set.

Why this module exists: spec 03 section 7 says one experiment changes one variable,
and that is a property of the SAMPLE LIST, not of the renderer. render_sample()
randomises pose, position, background crop, blur and noise internally, so a sweep
built on it cannot be attributed to any single cause; and a sweep whose pinned
values are only *intended* to be pinned is indistinguishable from one that drifted.
The list of samples that will be rendered -- which variable each row moves, and
every value that is held -- is therefore data this module owns, pure and testable,
and the generator CLI (tools/generate_controlled_capability.py) only walks it.

The vocabulary written here is the one the evaluator reads
(src/perception/shuttle_detection/evaluate.py): when a manifest carries
size_bucket / pose_bucket / position_bucket / blur_bucket / occlusion_bucket /
background, those exact strings become the group labels of the capability curves,
so a wrong word here is a wrong axis there.

Ground truth, sizing and appearance all belong to tools/shuttle_render.py; this
module never re-implements them. It calls the renderer and checks the result.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_controlled_capability.py -v
"""
from __future__ import annotations

import math
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

# The pool every controlled row is drawn from. These are the audited real
# backgrounds; the training pools (train_data/bg_train, train_data/bg_val) must
# never appear here, because a test image sharing a training background measures
# the detector's memory of the scene instead of its capability on the target.
DATA_DIR = ROOT / "outputs" / "shuttle_capability" / "controlled_capability"
BACKGROUND_DIR = ROOT / "outputs" / "shuttle_capability" / "real_images" / "backgrounds"
TRAINING_BACKGROUND_DIRS = (
    ROOT / "outputs" / "shuttle_capability" / "train_data" / "bg_train",
    ROOT / "outputs" / "shuttle_capability" / "train_data" / "bg_val",
)

# Held-out vocabulary from src/perception/shuttle_detection/contracts.py. The
# training-path dataset builder refuses this split, which is the point: these
# images exist to measure the detector, never to fit it.
SPLIT = "fixed_core_test"
SOURCE_TYPE = "SYNTHETIC_3D"
CAMERA_ID = "synth_pinhole_f700"

IMAGE_SUBDIR = "images"
LABEL_SUBDIR = "labels"
MANIFEST_NAME = "manifest.csv"


def _renderer():
    """Import tools/shuttle_render.py without duplicating its math here."""
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import shuttle_render

    return shuttle_render


# --------------------------------------------------------------------------- #
# Spec 03 section 3: size buckets
# --------------------------------------------------------------------------- #

# Spec 03 section 3 in the spec's own order. metrics.SIZE_BUCKETS declares the same
# vocabulary for the evaluator; the tests assert the two are equal so the manifest,
# the audit and the capability curves cannot drift apart.
SIZE_BUCKETS: tuple[str, ...] = ("<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", ">32")

# Upper edges, exclusive: 4.0 belongs to "4-6", 8.0 to "8-12".
SIZE_BUCKET_EDGES: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)

# Candidate requests per bucket. The rendered ground-truth footprint is measured at
# whole-pixel resolution, so around 4-8 px the achieved size moves in steps of a
# pixel and a single request per bucket would leave holes at the edges: the
# generator walks these candidates and keeps the first whose MEASURED size still
# falls inside the bucket it was asked for. Every candidate lies inside its bucket,
# so no bucket edge is ever bent to make a target reachable.
SIZE_BUCKET_TARGETS_PX: dict[str, tuple[float, ...]] = {
    "<4": (3.0, 2.5, 3.5, 2.0),
    "4-6": (5.0, 4.5, 5.5, 4.0),
    "6-8": (7.0, 6.5, 7.5, 6.0),
    "8-12": (10.0, 9.0, 11.0, 8.0),
    "12-16": (14.0, 13.0, 15.0, 12.0),
    "16-24": (20.0, 18.0, 22.0, 16.0),
    "24-32": (28.0, 26.0, 30.0, 24.0),
    ">32": (40.0, 36.0, 44.0, 34.0),
}

BUCKET_SLUGS: dict[str, str] = {
    "<4": "lt4",
    "4-6": "4_6",
    "6-8": "6_8",
    "8-12": "8_12",
    "12-16": "12_16",
    "16-24": "16_24",
    "24-32": "24_32",
    ">32": "gt32",
}


def size_bucket(equivalent_size_px: float) -> str:
    """Bucket an equivalent size into the spec 03 section 3 vocabulary."""
    value = float(equivalent_size_px)
    if not math.isfinite(value):
        raise ValueError("equivalent_size_px must be finite, got " + repr(equivalent_size_px))
    for index, edge in enumerate(SIZE_BUCKET_EDGES):
        if value < edge:
            return SIZE_BUCKETS[index]
    return SIZE_BUCKETS[-1]


# --------------------------------------------------------------------------- #
# Spec 03 section 4.2: three-dimensional pose families
# --------------------------------------------------------------------------- #

# Every value is a true 3D orientation, not a 2D rotation of the image: the
# renderer receives rot_ypr(yaw, pitch, roll) and the mesh turns in space.
#
# Orientation convention, measured from the GLB and asserted by the tests rather
# than assumed: the shuttle's length axis is its local z, the cork sits at low z and
# the feather skirt at high z, and the camera looks down +z, so R = identity
# presents the CORK end and a 180 degree pitch about x presents the FEATHER end.
POSE_FAMILIES: dict[str, tuple[dict[str, float], ...]] = {
    "cork_end_on": ({"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},),
    "feather_end_on": ({"yaw_deg": 0.0, "pitch_deg": 180.0, "roll_deg": 0.0},),
    "side": (
        {"yaw_deg": 0.0, "pitch_deg": 90.0, "roll_deg": 0.0},
        {"yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        {"yaw_deg": 180.0, "pitch_deg": 90.0, "roll_deg": 0.0},
    ),
    "oblique": (
        {"yaw_deg": 35.0, "pitch_deg": 30.0, "roll_deg": 0.0},
        {"yaw_deg": -35.0, "pitch_deg": 30.0, "roll_deg": 0.0},
        {"yaw_deg": 45.0, "pitch_deg": 35.0, "roll_deg": 25.0},
        {"yaw_deg": -45.0, "pitch_deg": -35.0, "roll_deg": -25.0},
    ),
    "yaw_only": tuple(
        {"yaw_deg": angle, "pitch_deg": 0.0, "roll_deg": 0.0}
        for angle in (30.0, 60.0, 90.0, 120.0)
    ),
    "pitch_only": tuple(
        {"yaw_deg": 0.0, "pitch_deg": angle, "roll_deg": 0.0}
        for angle in (30.0, 60.0, 90.0, 120.0)
    ),
    # Roll about the shuttle's own axis is only visible when that axis is not
    # pointing at the camera, so the roll family is pinned to a true side view and
    # rolls from there. The other two angles never move.
    "roll_only": tuple(
        {"yaw_deg": 0.0, "pitch_deg": 90.0, "roll_deg": angle}
        for angle in (30.0, 60.0, 90.0, 120.0)
    ),
    # A tumbling shuttle mid-flight: the intermediate orientations between the named
    # ones, at a tilted axis so no row is axis-aligned.
    "flight_rotation": tuple(
        {"yaw_deg": 20.0, "pitch_deg": 20.0, "roll_deg": angle}
        for angle in (40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0)
    ),
}

POSE_BUCKETS: tuple[str, ...] = tuple(POSE_FAMILIES)


def long_axis_camera_space(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """The shuttle's length axis, in camera space, for one orientation."""
    rot_ypr = _renderer().rot_ypr
    rotation = rot_ypr(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    return rotation @ np.array([0.0, 0.0, 1.0])


def _shuttle_centre(parts: Sequence[Any]) -> np.ndarray:
    """The same origin render_shuttle recentres the mesh about, so both agree."""
    return np.mean([np.asarray(part.verts).mean(axis=0) for part in parts], axis=0)


def end_toward_camera(
    parts: Sequence[Any], yaw_deg: float, pitch_deg: float, roll_deg: float
) -> str:
    """Which end of the shuttle faces the camera, decided from the mesh.

    The camera looks down +z, so the smaller camera-space z is the nearer end. This
    is the check that makes "cork end toward the camera" a measurement rather than a
    claim about a rotation matrix.
    """
    rot_ypr = _renderer().rot_ypr
    rotation = rot_ypr(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    centre = _shuttle_centre(parts)
    depth: dict[str, float] = {}
    for part in parts:
        verts = np.asarray(part.verts, dtype=np.float64) - centre
        name = "feather" if str(part.name).endswith("Feather") else "cork"
        depth[name] = float((verts @ rotation.T)[:, 2].mean())
    if set(depth) != {"cork", "feather"}:
        raise ValueError(
            "parts must contain one feather and one cork mesh, got " + repr(sorted(depth))
        )
    return min(depth, key=lambda name: depth[name])


# --------------------------------------------------------------------------- #
# Spec 03 section 4.3: image position
# --------------------------------------------------------------------------- #

# The nine positions spec 03 section 4.3 names, as fractions of the frame. The
# fractions sit inside the outer third of each axis so a box of the largest pinned
# size still lies entirely inside the frame.
POSITION_TARGETS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.5),
    "left": (0.22, 0.5),
    "right": (0.78, 0.5),
    "top": (0.5, 0.22),
    "bottom": (0.5, 0.78),
    "top_left": (0.22, 0.22),
    "top_right": (0.78, 0.22),
    "bottom_left": (0.22, 0.78),
    "bottom_right": (0.78, 0.78),
}


def position_pixel_centre(name: str, width: int, height: int) -> tuple[float, float]:
    """Nominal pixel centre of one named position."""
    fraction_x, fraction_y = POSITION_TARGETS[name]
    return (fraction_x * float(width), fraction_y * float(height))


def position_stays_inside_frame(
    name: str, size_px: float, width: int, height: int, margin: float = 1.0
) -> bool:
    """Whether a square box of size_px at that position is wholly inside the frame.

    Spec 03 section 4.3 is about where the target sits, not about how much of it was
    cropped away, so a position that clips the object is not a position sample.
    """
    centre_x, centre_y = position_pixel_centre(name, width, height)
    half = float(size_px) / 2.0
    return bool(
        centre_x - half >= margin
        and centre_y - half >= margin
        and centre_x + half <= float(width) - 1.0 - margin
        and centre_y + half <= float(height) - 1.0 - margin
    )


# --------------------------------------------------------------------------- #
# Spec 03 sections 4.4 - 4.6: blur, occlusion, light
# --------------------------------------------------------------------------- #

# Motion blur length in pixels, as composite(motion_px=...) consumes it. 0 renders
# no blur at all (composite only convolves from 1.5 px up), so "none" is genuinely a
# clear image and not a very small amount of smear.
BLUR_LEVELS: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("light", 2.0),
    ("medium", 4.0),
    ("heavy", 7.0),
)

# Nominal occluded fraction of the object footprint per bucket. What the manifest
# records is the MEASURED fraction (see apply_occlusion), never this request.
OCCLUSION_LEVELS: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("light", 0.25),
    ("partial", 0.55),
)

# Several occluded fractions per bucket, so the curve inside a bucket is visible
# instead of being a single point.
OCCLUSION_TARGET_FRACTIONS: dict[str, tuple[float, ...]] = {
    "none": (0.0,),
    "light": (0.15, 0.25, 0.35),
    "partial": (0.45, 0.55, 0.70),
}

# The occluder enters from a fixed side in every row: which side a hand or racket
# crosses from is a second variable, and spec 03 section 7 allows exactly one.
OCCLUDER_SIDE = "left"

# Spec 03 section 4.5 asks for three levels; the MEASURED fraction decides which one
# a row belongs to, so a row whose bisection missed still lands in the bucket its
# image actually shows.
OCCLUSION_BUCKET_EDGES: tuple[float, ...] = (0.05, 0.40)


def occlusion_bucket(measured_fraction: float) -> str:
    """Bucket a MEASURED occluded fraction of the object footprint."""
    value = float(measured_fraction)
    if value < OCCLUSION_BUCKET_EDGES[0]:
        return OCCLUSION_LEVELS[0][0]
    if value < OCCLUSION_BUCKET_EDGES[1]:
        return OCCLUSION_LEVELS[1][0]
    return OCCLUSION_LEVELS[2][0]


def _cv2():
    import cv2

    return cv2


# The occluder is drawn in the image, not modelled in the 3D scene: the renderer has
# no support for it, and a foreground object -- a racket frame, a hand, the net --
# is exactly an opaque thing between the camera and the shuttle. What the manifest
# needs is not that the occluder is photoreal, but that the fraction of the object
# it hides is MEASURED from the pixels that were written.
OCCLUDER_TONE_BGR = (44.0, 47.0, 52.0)
OCCLUDER_FEATHER_PX = 0.8
OCCLUDER_SEARCH_STEPS = 32


@dataclass(frozen=True)
class OcclusionResult:
    """One occluded image plus what the occluder actually hid."""

    image: np.ndarray
    occluder_alpha: np.ndarray
    measured_fraction: float
    bucket: str
    reach: float


def _local_window(shape: tuple[int, int], bbox: tuple[int, int, int, int], side: str,
                  reach: float, feather_px: float) -> tuple[int, int, int, int, int, int]:
    """The rectangle the occluder bar occupies, plus a blur halo, as frame coords."""
    height, width = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = (int(value) for value in bbox)
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    if side in ("left", "right"):
        span = int(round(reach * box_w))
        pad = max(1, int(round(0.12 * box_h)))
        bar_y0, bar_y1 = y0 - pad, y1 + pad
        bar_x0, bar_x1 = (x0, x0 + span - 1) if side == "left" else (x1 - span + 1, x1)
    elif side in ("top", "bottom"):
        span = int(round(reach * box_h))
        pad = max(1, int(round(0.12 * box_w)))
        bar_x0, bar_x1 = x0 - pad, x1 + pad
        bar_y0, bar_y1 = (y0, y0 + span - 1) if side == "top" else (y1 - span + 1, y1)
    else:
        raise ValueError("side must be left, right, top or bottom, got " + repr(side))
    halo = max(2, int(math.ceil(4.0 * float(feather_px))))
    return (
        max(bar_x0, 0),
        max(bar_y0, 0),
        min(bar_x1, width - 1),
        min(bar_y1, height - 1),
        halo,
        span,
    )


def occluder_alpha(shape: tuple[int, int], bbox: tuple[int, int, int, int], *, reach: float,
                   side: str = OCCLUDER_SIDE, feather_px: float = OCCLUDER_FEATHER_PX
                   ) -> np.ndarray:
    """Coverage of a soft-edged bar that enters the object box from one side.

    reach is the fraction of the box extent the bar spans; 0 draws nothing at all.
    The blur only softens the bar's own edge -- the object coverage is measured on
    the thresholded alpha, so a soft halo is never counted as occlusion.
    """
    cv2 = _cv2()
    height, width = int(shape[0]), int(shape[1])
    alpha = np.zeros((height, width), dtype=np.float32)
    value = float(min(max(reach, 0.0), 1.0))
    if value <= 0.0:
        return alpha
    bar_x0, bar_y0, bar_x1, bar_y1, halo, span = _local_window(shape, bbox, side, value, feather_px)
    if span < 1 or bar_x1 < bar_x0 or bar_y1 < bar_y0:
        return alpha
    window_x0 = max(bar_x0 - halo, 0)
    window_y0 = max(bar_y0 - halo, 0)
    window_x1 = min(bar_x1 + halo, width - 1)
    window_y1 = min(bar_y1 + halo, height - 1)
    patch = np.zeros((window_y1 - window_y0 + 1, window_x1 - window_x0 + 1), dtype=np.float32)
    patch[bar_y0 - window_y0: bar_y1 - window_y0 + 1,
          bar_x0 - window_x0: bar_x1 - window_x0 + 1] = 1.0
    if feather_px > 0.0:
        patch = cv2.GaussianBlur(patch, (0, 0), float(feather_px))
    alpha[window_y0: window_y1 + 1, window_x0: window_x1 + 1] = np.clip(patch, 0.0, 1.0)
    return alpha


def occluded_fraction(footprint: np.ndarray, alpha: np.ndarray) -> float:
    """Fraction of the object footprint the drawn occluder hides."""
    mask = np.asarray(footprint).astype(bool)
    total = int(np.count_nonzero(mask))
    if total == 0:
        raise ValueError("the object footprint is empty, so no fraction is defined")
    hidden = np.count_nonzero((np.asarray(alpha) >= 0.5) & mask)
    return float(hidden) / float(total)


def _occluder_tone(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """A dark, faintly textured bar: flat black would be a synthetic giveaway."""
    rng = np.random.default_rng(int(seed) % (2 ** 31 - 1))
    base = np.array(OCCLUDER_TONE_BGR, dtype=np.float64).reshape(1, 1, 3)
    noise = rng.normal(0.0, 6.0, (int(shape[0]), int(shape[1]), 1))
    return np.clip(base + noise, 0.0, 255.0).astype(np.float32)


def apply_occlusion(image: np.ndarray, footprint: np.ndarray, *, target_fraction: float,
                    side: str = OCCLUDER_SIDE, feather_px: float = OCCLUDER_FEATHER_PX,
                    seed: int = 0) -> OcclusionResult:
    """Draw an occluder over the composited image and measure what it hid.

    The bar is bisected until it covers target_fraction of the object footprint, and
    the fraction that is RETURNED is measured from the alpha that was actually
    composited -- the object footprint is not a rectangle, so a bar covering half the
    box covers less than half the object, and recording the request instead of the
    measurement would label the row with a number no image shows.

    The ground-truth box is deliberately NOT shrunk to the visible part: the box
    describes the object, and an occluded object is still that object.
    """
    img = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(footprint).astype(bool)
    if img.ndim != 3:
        raise ValueError("image must be HxWx3")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("the object footprint is empty, so there is nothing to occlude")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    target = float(target_fraction)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_fraction must be in [0, 1], got " + repr(target_fraction))
    if target <= 0.0:
        return OcclusionResult(
            image=np.array(img, copy=True),
            occluder_alpha=np.zeros(mask.shape, dtype=np.float32),
            measured_fraction=0.0,
            bucket=occlusion_bucket(0.0),
            reach=0.0,
        )

    low, high = 0.0, 1.0
    best_error = float("inf")
    best_reach = 0.0
    best_alpha = np.zeros(mask.shape, dtype=np.float32)
    best_measured = 0.0
    for _ in range(OCCLUDER_SEARCH_STEPS):
        middle = 0.5 * (low + high)
        alpha = occluder_alpha(mask.shape, bbox, reach=middle, side=side, feather_px=feather_px)
        measured = occluded_fraction(mask, alpha)
        error = abs(measured - target)
        if error < best_error:
            best_error = error
            best_reach = middle
            best_alpha = alpha
            best_measured = measured
        if measured < target:
            low = middle
        else:
            high = middle

    coverage = best_alpha[..., None]
    tone = _occluder_tone(img.shape, seed)
    composited = np.clip(
        img.astype(np.float32) * (1.0 - coverage) + tone * coverage, 0.0, 255.0
    ).astype(np.uint8)
    return OcclusionResult(
        image=composited,
        occluder_alpha=best_alpha,
        measured_fraction=best_measured,
        bucket=occlusion_bucket(best_measured),
        reach=best_reach,
    )



# Light directions in the renderer's convention: +x right, +y down, and the light
# comes from the camera side (-z), as in the frozen synthetic set. Azimuth is in the
# image plane, 0 deg = from the right, 90 deg = from above.
LIGHT_AZIMUTHS_DEG: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
LIGHT_ELEVATION_DEG = 40.0


def light_vector(azimuth_deg: float, elevation_deg: float = LIGHT_ELEVATION_DEG) -> np.ndarray:
    """Unit vector pointing from the scene toward the light (renderer convention)."""
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    horizontal = math.cos(elevation)
    vector = np.array(
        [
            math.cos(azimuth) * horizontal,
            -math.sin(azimuth) * horizontal,
            -math.sin(elevation),
        ],
        dtype=np.float64,
    )
    return vector / float(np.linalg.norm(vector))


# --------------------------------------------------------------------------- #
# The manifest schema
# --------------------------------------------------------------------------- #

# The first block is read verbatim by src/perception/shuttle_detection/evaluate.py,
# and every string in it is therefore part of this module's contract. The second
# block is what makes each row reproducible and what lets the audit's own
# distribution report bucket by the MEASURED size and a real blur length:
# dataset_audit.bucket_size prefers "equiv_size_px" and bucket_blur reads
# "motion_px", so equiv_size_px repeats the measured equivalent_size_px exactly.
REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "file",
    "split",
    "source_type",
    "bbox_w_px",
    "bbox_h_px",
    "pos_x_px",
    "pos_y_px",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "pose_bucket",
    "position_bucket",
    "blur_bucket",
    "occlusion_bucket",
    "background",
    "light_azimuth_deg",
    "equivalent_size_px",
)

EXTRA_MANIFEST_COLUMNS: tuple[str, ...] = (
    "sweep",
    "seed",
    "target_px",
    "equiv_size_px",
    "distance_m",
    "motion_px",
    "motion_angle_deg",
    "occlusion_fraction",
    "occlusion_target_fraction",
    "noise_sigma",
    "noise_seed",
    "shadow_gain",
    "coverage_sum",
    "solid_px",
    "imgsz",
    "supersample",
    "focal_px",
    "camera_id",
    "is_negative",
)

MANIFEST_COLUMNS: tuple[str, ...] = REQUIRED_MANIFEST_COLUMNS + EXTRA_MANIFEST_COLUMNS

# Columns the generator fills in from the rendered image; the plan leaves them blank.
MEASURED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "bbox_w_px",
    "bbox_h_px",
    "pos_x_px",
    "pos_y_px",
    "equivalent_size_px",
    "equiv_size_px",
    "distance_m",
    "occlusion_fraction",
    "coverage_sum",
    "solid_px",
    "occlusion_bucket",
)


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #

SWEEP_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6a", "S6b")

# Rows per size bucket in the size sweep: one per candidate target, so the repeats
# inside a bucket are different SIZE requests, not the same request rendered twice.
SIZE_REPEATS_PER_BUCKET = 3


@dataclass(frozen=True)
class ControlSettings:
    """Every value the sweeps hold fixed, plus the pools they sweep over."""

    width: int = 960
    height: int = 960
    focal_px: float = 700.0
    supersample: int = 3
    background: str = "bg_001.jpg"
    backgrounds: tuple[str, ...] = ()
    light_azimuth_deg: float = 315.0
    light_azimuths_deg: tuple[float, ...] = LIGHT_AZIMUTHS_DEG
    noise_sigma: float = 2.0
    shadow_gain: float = 0.0
    motion_angle_deg: float = 30.0
    pose_deg: tuple[float, float, float] = (35.0, 30.0, 0.0)
    size_px: float = 16.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame size must be positive")
        if self.supersample < 1:
            raise ValueError("supersample must be >= 1")
        if self.size_px <= 0.0:
            raise ValueError("size_px must be > 0")
        if len(self.backgrounds) < 2:
            raise ValueError(
                "the background sweep needs at least two backgrounds to vary over"
            )
        if len(self.light_azimuths_deg) < 2:
            raise ValueError("the light sweep needs at least two azimuths to vary over")
        if self.background not in self.backgrounds:
            raise ValueError("the pinned background must be one of the swept backgrounds")


@dataclass(frozen=True)
class PlannedSample:
    """One image the generator will render, with every pinned value recorded.

    name / sweep / varied are the control bookkeeping: "varied" names the columns
    this sweep is allowed to move, and control_inputs() returns every other input so
    verify_single_variable() can prove they were held.
    """

    name: str
    sweep: str
    varied: tuple[str, ...]
    seed: int
    target_px: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    pose_bucket: str
    position_bucket: str
    background: str
    light_azimuth_deg: float
    motion_px: float
    motion_angle_deg: float
    blur_bucket: str
    occlusion_bucket: str
    occlusion_target_fraction: float
    noise_sigma: float
    noise_seed: int
    shadow_gain: float
    width: int
    height: int
    focal_px: float
    supersample: int

    @property
    def pixel_centre(self) -> tuple[float, float]:
        return position_pixel_centre(self.position_bucket, self.width, self.height)

    def control_inputs(self) -> dict[str, Any]:
        """Every input the sweep controls, keyed by manifest column name."""
        return {
            "target_px": self.target_px,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "pose_bucket": self.pose_bucket,
            "position_bucket": self.position_bucket,
            "background": self.background,
            "light_azimuth_deg": self.light_azimuth_deg,
            "motion_px": self.motion_px,
            "blur_bucket": self.blur_bucket,
            "occlusion_bucket": self.occlusion_bucket,
            "occlusion_target_fraction": self.occlusion_target_fraction,
        }

    @property
    def file_name(self) -> str:
        """Manifest "file" value, relative to the manifest's own directory.

        The audit resolves a row as <manifest_dir>/<file>, so the image subdirectory
        has to be part of the value; a bare file name would be IMAGE_MISSING.
        """
        return IMAGE_SUBDIR + "/" + self.name + ".jpg"

    def as_manifest_row(self) -> dict[str, Any]:
        """The planned half of a manifest row; measured columns are filled later."""
        row: dict[str, Any] = {column: "" for column in MANIFEST_COLUMNS}
        row.update(
            {
                "file": self.file_name,
                "split": SPLIT,
                "source_type": SOURCE_TYPE,
                "sweep": self.sweep,
                "seed": self.seed,
                "target_px": self.target_px,
                "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg,
                "roll_deg": self.roll_deg,
                "pose_bucket": self.pose_bucket,
                "position_bucket": self.position_bucket,
                "blur_bucket": self.blur_bucket,
                "occlusion_bucket": self.occlusion_bucket,
                "occlusion_target_fraction": self.occlusion_target_fraction,
                "background": self.background,
                "light_azimuth_deg": self.light_azimuth_deg,
                "motion_px": self.motion_px,
                "motion_angle_deg": self.motion_angle_deg,
                "noise_sigma": self.noise_sigma,
                "noise_seed": self.noise_seed,
                "shadow_gain": self.shadow_gain,
                "imgsz": self.width,
                "supersample": self.supersample,
                "focal_px": self.focal_px,
                "camera_id": CAMERA_ID,
                "is_negative": False,
            }
        )
        return row


def stable_seed(*parts: object) -> int:
    """A deterministic, reproducible seed for one planned sample.

    Not random: a rerun of the generator has to reproduce the same image, and the
    manifest records the value so a reader can re-render any single row.
    """
    text = "|".join(str(part) for part in parts)
    return int(zlib.crc32(text.encode("utf-8")) % (2 ** 31 - 1))


def _blur_bucket_of(motion_px: float) -> str:
    for name, value in BLUR_LEVELS:
        if abs(float(motion_px) - value) < 1e-9:
            return name
    raise ValueError("motion_px " + repr(motion_px) + " is not a declared blur level")


def _sample(
    settings: ControlSettings,
    *,
    name: str,
    sweep: str,
    varied: tuple[str, ...],
    pose: Mapping[str, float],
    pose_bucket: str,
    target_px: float,
    position_bucket: str,
    background: str,
    light_azimuth_deg: float,
    motion_px: float,
    occlusion_name: str,
    occlusion_target_fraction: float,
) -> PlannedSample:
    seed = stable_seed(sweep, name)
    return PlannedSample(
        name=name,
        sweep=sweep,
        varied=varied,
        seed=seed,
        target_px=float(target_px),
        yaw_deg=float(pose["yaw_deg"]),
        pitch_deg=float(pose["pitch_deg"]),
        roll_deg=float(pose["roll_deg"]),
        pose_bucket=pose_bucket,
        position_bucket=position_bucket,
        background=background,
        light_azimuth_deg=float(light_azimuth_deg),
        motion_px=float(motion_px),
        motion_angle_deg=float(settings.motion_angle_deg),
        blur_bucket=_blur_bucket_of(motion_px),
        occlusion_bucket=occlusion_name,
        occlusion_target_fraction=float(occlusion_target_fraction),
        noise_sigma=float(settings.noise_sigma),
        noise_seed=seed,
        shadow_gain=float(settings.shadow_gain),
        width=int(settings.width),
        height=int(settings.height),
        focal_px=float(settings.focal_px),
        supersample=int(settings.supersample),
    )


def build_plan(settings: ControlSettings) -> list[PlannedSample]:
    """The complete controlled sample list, one entry per image.

    Pinned for every sweep: the frame and camera, the noise level, the shadow, the
    occluder side. Pinned per sweep: everything control_inputs() returns that the
    sweep's "varied" tuple does not name.
    """
    pinned_pose = {
        "yaw_deg": settings.pose_deg[0],
        "pitch_deg": settings.pose_deg[1],
        "roll_deg": settings.pose_deg[2],
    }
    centre = "center"
    samples: list[PlannedSample] = []

    def add(**kwargs: Any) -> None:
        kwargs.setdefault("pose", pinned_pose)
        kwargs.setdefault("pose_bucket", "pinned_oblique")
        kwargs.setdefault("target_px", settings.size_px)
        kwargs.setdefault("position_bucket", centre)
        kwargs.setdefault("background", settings.background)
        kwargs.setdefault("light_azimuth_deg", settings.light_azimuth_deg)
        kwargs.setdefault("motion_px", 0.0)
        kwargs.setdefault("occlusion_name", "none")
        kwargs.setdefault("occlusion_target_fraction", 0.0)
        samples.append(_sample(settings, **kwargs))

    # S1 size: pose, background, position, light and blur all held.
    for bucket in SIZE_BUCKETS:
        for index, target in enumerate(
            SIZE_BUCKET_TARGETS_PX[bucket][:SIZE_REPEATS_PER_BUCKET]
        ):
            add(
                name="s1_size_" + BUCKET_SLUGS[bucket] + "_" + str(index),
                sweep="S1",
                varied=("target_px",),
                target_px=target,
            )

    # S2 pose: size, background, position, light and blur all held.
    for family in POSE_BUCKETS:
        for index, pose in enumerate(POSE_FAMILIES[family]):
            add(
                name="s2_pose_" + family + "_" + str(index),
                sweep="S2",
                varied=("yaw_deg", "pitch_deg", "roll_deg", "pose_bucket"),
                pose=pose,
                pose_bucket=family,
            )

    # S3 position: size, pose, background, light and blur all held.
    for position in POSITION_TARGETS:
        add(
            name="s3_position_" + position,
            sweep="S3",
            varied=("position_bucket",),
            position_bucket=position,
        )

    # S4 blur: size, pose, background, position and light all held.
    for blur_name, motion_px in BLUR_LEVELS:
        add(
            name="s4_blur_" + blur_name,
            sweep="S4",
            varied=("blur_bucket", "motion_px"),
            motion_px=motion_px,
        )

    # S5 occlusion: size, pose, background, position, light and blur all held.
    for occlusion_name, _representative in OCCLUSION_LEVELS:
        for index, fraction in enumerate(OCCLUSION_TARGET_FRACTIONS[occlusion_name]):
            add(
                name="s5_occlusion_" + occlusion_name + "_" + str(index),
                sweep="S5",
                varied=("occlusion_bucket", "occlusion_target_fraction"),
                occlusion_name=occlusion_name,
                occlusion_target_fraction=fraction,
            )

    # S6a background: size, pose, position, blur and the light all held.
    for index, background in enumerate(settings.backgrounds):
        add(
            name="s6a_background_" + str(index).zfill(2),
            sweep="S6a",
            varied=("background",),
            background=background,
        )

    # S6b light: size, pose, position, blur and the background all held.
    for azimuth in settings.light_azimuths_deg:
        add(
            name="s6b_light_" + str(int(round(azimuth))).zfill(3),
            sweep="S6b",
            varied=("light_azimuth_deg",),
            light_azimuth_deg=azimuth,
        )
    return samples


def _freeze(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def verify_single_variable(samples: Sequence[PlannedSample]) -> list[str]:
    """Report every sweep that moved something other than its own variable.

    Returns human-readable violations, empty when the plan is a set of controlled
    experiments. This is the check that makes "one variable at a time" a property of
    the data rather than an intention in a comment.
    """
    violations: list[str] = []
    order = list(SWEEP_IDS) + sorted({sample.sweep for sample in samples} - set(SWEEP_IDS))
    for sweep in order:
        rows = [sample for sample in samples if sample.sweep == sweep]
        if not rows:
            continue
        varied = tuple(rows[0].varied)
        for row in rows:
            if tuple(row.varied) != varied:
                violations.append(sweep + ": inconsistent varied columns: " + repr(row.varied))
                break
        fields = sorted({field for row in rows for field in row.control_inputs()})
        for field in fields:
            if field in varied:
                continue
            values = {_freeze(row.control_inputs()[field]) for row in rows}
            if len(values) > 1:
                violations.append(
                    sweep
                    + ": pinned input "
                    + field
                    + " takes "
                    + str(len(values))
                    + " values "
                    + repr(sorted(values, key=repr))
                )
        combos = {
            tuple(_freeze(row.control_inputs()[field]) for field in varied) for row in rows
        }
        if len(combos) < 2:
            violations.append(
                sweep + ": varied input(s) " + repr(list(varied)) + " take a single value"
            )
    return violations


# --------------------------------------------------------------------------- #
# Checking the rendered manifest, not the intention behind it
# --------------------------------------------------------------------------- #

# The recorded controls: every manifest column a sweep is supposed to hold, with the
# tolerance at which two rows still count as held. Anything not listed here is either
# the varied variable of some sweep, or a measured outcome of it (bbox extents, the
# target centre, the distance) and is deliberately not pinned.
PINNED_MANIFEST_COLUMNS: dict[str, float] = {
    "target_px": 0.0,
    "yaw_deg": 0.0,
    "pitch_deg": 0.0,
    "roll_deg": 0.0,
    "pose_bucket": 0.0,
    "position_bucket": 0.0,
    "background": 0.0,
    "light_azimuth_deg": 0.0,
    "motion_px": 0.0,
    "blur_bucket": 0.0,
    "occlusion_bucket": 0.0,
    "occlusion_target_fraction": 0.0,
    "occlusion_fraction": 0.0,
}

# The one variable each sweep is allowed to move, as manifest column names. Where a
# sweep moves a measured quantity it is named separately from the input that drives
# it, so "the occluder covered more of the object" is not confused with "the request
# asked for more occlusion".
VARIED_MANIFEST_COLUMNS: dict[str, tuple[str, ...]] = {
    "S1": ("target_px", "equivalent_size_px"),
    "S2": ("yaw_deg", "pitch_deg", "roll_deg", "pose_bucket"),
    "S3": ("position_bucket", "pos_x_px", "pos_y_px"),
    "S4": ("blur_bucket", "motion_px"),
    "S5": ("occlusion_bucket", "occlusion_target_fraction", "occlusion_fraction"),
    "S6a": ("background",),
    "S6b": ("light_azimuth_deg",),
}


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def verify_measured_pins(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Report pinned manifest columns that did not stay put inside a sweep.

    The plan is checked before rendering; this is the same question asked of the
    rows that were actually written, which is the only version of it a reader can
    verify. Continuous columns are compared against the sweep's own spread rather
    than against an exact equality, so a genuine drift is reported while the last
    decimal place of a float is not.
    """
    violations: list[str] = []
    for sweep in SWEEP_IDS:
        sweep_rows = [row for row in rows if str(row.get("sweep", "")) == sweep]
        if not sweep_rows:
            continue
        varied = set(VARIED_MANIFEST_COLUMNS.get(sweep, ()))
        for column, tolerance in PINNED_MANIFEST_COLUMNS.items():
            if column in varied:
                continue
            values = [row.get(column, "") for row in sweep_rows]
            if len(set(map(str, values))) <= 1:
                continue
            numbers = [_numeric(value) for value in values]
            if all(number is not None for number in numbers):
                spread = max(numbers) - min(numbers)
                if spread <= float(tolerance):
                    continue
                violations.append(
                    sweep
                    + ": pinned column "
                    + column
                    + " spread "
                    + f"{spread:.6g}"
                    + " over "
                    + str(len(sweep_rows))
                    + " rows"
                )
            else:
                violations.append(
                    sweep
                    + ": pinned column "
                    + column
                    + " takes "
                    + str(len(set(map(str, values))))
                    + " values "
                    + repr(sorted(set(map(str, values))))
                )
    return violations


def distribution(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    """Count rows by the value of one column, for the post-generation report."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def bbox_inside_frame(row: Mapping[str, Any], width: int, height: int) -> bool:
    """Whether a recorded ground-truth box lies wholly inside the frame."""
    centre_x = _numeric(row.get("pos_x_px"))
    centre_y = _numeric(row.get("pos_y_px"))
    bbox_w = _numeric(row.get("bbox_w_px"))
    bbox_h = _numeric(row.get("bbox_h_px"))
    if None in (centre_x, centre_y, bbox_w, bbox_h):
        return False
    half_w = (bbox_w - 1.0) / 2.0
    half_h = (bbox_h - 1.0) / 2.0
    return bool(
        centre_x - half_w >= 0.0
        and centre_y - half_h >= 0.0
        and centre_x + half_w <= float(width) - 1.0
        and centre_y + half_h <= float(height) - 1.0
    )


# --------------------------------------------------------------------------- #
# Ground truth from a rendered mask
# --------------------------------------------------------------------------- #


def equivalent_size_px(bbox_width_px: float, bbox_height_px: float) -> float:
    """Spec 03 section 2: sqrt(width * height), the single target-size scalar."""
    return math.sqrt(float(bbox_width_px) * float(bbox_height_px))


def footprint_measurement(alpha: np.ndarray, width: int, height: int) -> dict[str, Any]:
    """Ground truth from the rendered coverage: the object's whole footprint.

    SUPPORT_COVERAGE is the renderer's own threshold -- every pixel the shuttle
    touches belongs to its box -- and it is imported rather than re-chosen, because a
    ground truth that disagreed with the training set's would make this set
    incomparable with the baseline it exists to measure.
    """
    renderer = _renderer()
    mask = renderer.mask_from_coverage(alpha, renderer.SUPPORT_COVERAGE)
    bbox = renderer.yolo_bbox_from_mask(mask, width, height)
    if bbox is None:
        raise ValueError("the rendered footprint is empty: nothing was drawn")
    ys, xs = np.nonzero(mask)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    return {
        "bbox_w_px": bbox_w,
        "bbox_h_px": bbox_h,
        "pos_x_px": int((xs.min() + xs.max()) // 2),
        "pos_y_px": int((ys.min() + ys.max()) // 2),
        "equivalent_size_px": round(equivalent_size_px(bbox_w, bbox_h), 3),
        "coverage_sum": round(float(np.asarray(alpha).sum()), 3),
        "solid_px": int((np.asarray(alpha) >= 0.5).sum()),
        "label": "0 " + f"{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}" + "\n",
        "mask": mask,
    }
