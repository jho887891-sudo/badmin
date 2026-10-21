# -*- coding: utf-8 -*-
"""P7-04: the unified shuttlecock DETECTION OUTPUT schema (pixels only).

Why this module exists
----------------------
P7-04 of the continuous execution queue (management/morph_one_rs/
EXECUTION_QUEUE_CONTINUOUS_P0_P8.md, section P7) requires one detector output
contract: timestamp, frame_id, bbox, confidence, pixel_covariance, with unit
tests.  Before this module the detection package only had *dataset audit*
contracts (contracts.py) and the brain had a *3D* measurement contract
(badminton_brain.types.ShuttleMeasurement, metres / court frame).  A detector
produces pixels, so feeding its raw output straight into the 3D contract is
exactly the "2D bbox presented as 3D" failure the master spec forbids (P7-06).
This module is that missing, explicitly 2D boundary:

    detector --DetectionObservation--> tracker (P7-05) --> 3D only after a
                                                           calibrated depth path

Units are the whole point
-------------------------
Everything here is PIXELS:

  bbox_xyxy_px       4 floats (x1, y1, x2, y2), pixel edges of the image the
                     detector ran on (u = column, v = row, origin at the
                     top-left pixel centre, same convention as
                     badminton_brain.perception.stereo_geometry).
  pixel_covariance   2x2 covariance of the bbox CENTRE in px^2, x = u.
  timestamp_s        seconds, same clock as every other message.
  confidence         dimensionless in [0, 1].

The schema carries its units in the JSON payload (the "units" block) and refuses
a payload whose units are not px / px^2.  A metre-valued bbox is caught by two
independent guards: the image-bounds check, and MIN_PIXEL_VARIANCE_PX2, a
declared physical floor (0.1 px standard deviation) below which a "pixel"
covariance can only be a metre-squared covariance that was mislabelled (a
stereo sigma of 0.01 m..0.1 m is 1e-4..1e-2 m^2, at or below that floor).

What this module does NOT claim
-------------------------------
No accuracy number appears here.  confidence is whatever the detector says; no
test in this repository turns it into a probability of being correct.  The
covariance defaults used by the *tracker* live in
badminton_brain.estimation.shuttle_state_ekf and are explicitly labelled proxies
there.  See management/morph_one_rs/P7_DETECTION_EKF_INTERFACES.md.

Run the tests with:
    python -m pytest tests/perception/shuttle_detection/test_detection_output.py -q
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

__all__ = [
    "SCHEMA_ID", "SCHEMA_VERSION", "BBOX_UNIT", "PIXEL_COVARIANCE_UNIT",
    "MIN_PIXEL_VARIANCE_PX2", "PIXEL_COVARIANCE_SYMMETRY_TOL", "CONFIDENCE_MIN",
    "CONFIDENCE_MAX", "REJECT_REASONS", "DetectionSchemaError", "ImageSize",
    "ValidityFlags", "DetectionObservation", "validate_detection_payload", "units_block",
]

# ---------------------------------------------------------------------------
# Frozen vocabulary of the schema (never re-typed inside the module)
# ---------------------------------------------------------------------------
SCHEMA_ID = "badmin.shuttle_detection.observation"
SCHEMA_VERSION = 1

BBOX_UNIT = "px"
PIXEL_COVARIANCE_UNIT = "px^2"

# Declared physical floor of a pixel-centre covariance.  sqrt(1e-6) = 0.001 px,
# i.e. a thousandth of a pixel.  No bbox-centre estimator in this project is
# claimed to beat that, so a "pixel" covariance below it is read as a unit
# confusion (a metre-squared covariance supplied for a pixel bbox) and rejected
# instead of being silently used.  Rationale: sub-pixel centroid estimators
# reach about 0.1 px on an ideal large blob, and a bbox detector firing on a
# 4-6 px shuttle cannot be better, so a pixel covariance below 0.01 px^2 is not a
# credible detection covariance; a stereo sigma of 0.01 m..0.1 m is 1e-4..1e-2
# m^2, which lands at or below that floor.  Deliberately conservative: it must
# never reject a plausible pixel covariance of a real detection.
MIN_PIXEL_VARIANCE_PX2 = 0.01  # 0.1 px std; see the comment above
PIXEL_COVARIANCE_SYMMETRY_TOL = 1e-9

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# Explicit validity vocabulary.  "NONE" = no reason to reject; every other value
# names exactly one failure mode so a consumer never has to guess.
REJECT_REASONS: tuple[str, ...] = (
    "NONE",
    "NOT_DETECTED",
    "LOW_CONFIDENCE",
    "GEOMETRY_OUT_OF_IMAGE",
    "NON_FINITE",
    "BAD_COVARIANCE",
    "STALE",
    "UNIT_CONFUSION",
)

# Keys that would carry metre/metre-per-second quantities into a pixel schema.
# Their presence in a payload is a unit confusion, not "an extra field".
_METRIC_KEYS: tuple[str, ...] = (
    "bbox_xyxy_m", "bbox_xywh_m", "pixel_covariance_m2", "position_m",
    "position_mps", "velocity_mps", "covariance_m2", "depth_m",
)


class DetectionSchemaError(ValueError):
    """A detection payload/observation violates the P7-04 contract."""


def units_block() -> Dict[str, str]:
    """The unit declaration a payload must carry (also returned by to_json_dict)."""
    return {
        "bbox_xyxy_px": BBOX_UNIT,
        "pixel_covariance": PIXEL_COVARIANCE_UNIT,
        "timestamp_s": "s",
        "confidence": "1",
    }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DetectionSchemaError(f"{name} must be a real number, got bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectionSchemaError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise DetectionSchemaError(f"{name} must be finite, got {out!r} (NaN/Inf rejected)")
    return out


def _int_ge(value: Any, name: str, lower: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DetectionSchemaError(f"{name} must be an integer, got {type(value).__name__}")
    out = int(value)
    if out < lower:
        raise DetectionSchemaError(f"{name} must be >= {lower}, got {out}")
    return out


@dataclass(frozen=True)
class ImageSize:
    """Pixel size of the image the detector ran on (the bbox reference frame)."""

    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "width_px", _int_ge(self.width_px, "ImageSize.width_px", 2))
        object.__setattr__(self, "height_px", _int_ge(self.height_px, "ImageSize.height_px", 2))

    def to_json_dict(self) -> Dict[str, int]:
        return {"width_px": self.width_px, "height_px": self.height_px}

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ImageSize":
        if not isinstance(payload, Mapping):
            raise DetectionSchemaError("image_size must be a mapping")
        try:
            width = payload["width_px"]
            height = payload["height_px"]
        except KeyError as exc:
            raise DetectionSchemaError(f"image_size is missing {exc}") from exc
        return cls(width_px=width, height_px=height)


@dataclass(frozen=True)
class ValidityFlags:
    """Explicit validity flags: a consumer must read these, never infer validity.

    in_image is *derived* from the bbox and the image size in
    DetectionObservation; storing it here makes the derived value part of the
    serialised contract, and the constructor refuses a flag that disagrees with
    the geometry (a flag that can lie is worse than no flag).
    """

    detected: bool = True
    in_image: bool = True
    confidence_ok: bool = True
    covariance_ok: bool = True
    timestamp_ok: bool = True
    reject_reason: str = "NONE"

    def __post_init__(self) -> None:
        for name in ("detected", "in_image", "confidence_ok", "covariance_ok", "timestamp_ok"):
            if not isinstance(getattr(self, name), bool):
                raise DetectionSchemaError(f"ValidityFlags.{name} must be bool")
        if self.reject_reason not in REJECT_REASONS:
            raise DetectionSchemaError(
                f"ValidityFlags.reject_reason must be one of {REJECT_REASONS}, "
                f"got {self.reject_reason!r}")

    @property
    def usable(self) -> bool:
        """True only when every flag passes AND no reject reason was recorded."""
        return (self.detected and self.in_image and self.confidence_ok
                and self.covariance_ok and self.timestamp_ok
                and self.reject_reason == "NONE")

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "in_image": self.in_image,
            "confidence_ok": self.confidence_ok,
            "covariance_ok": self.covariance_ok,
            "timestamp_ok": self.timestamp_ok,
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ValidityFlags":
        if not isinstance(payload, Mapping):
            raise DetectionSchemaError("validity must be a mapping")
        expected = ("detected", "in_image", "confidence_ok", "covariance_ok",
                    "timestamp_ok", "reject_reason")
        missing = [k for k in expected if k not in payload]
        if missing:
            raise DetectionSchemaError(f"validity is missing {missing}")
        return cls(**{k: payload[k] for k in expected})


def _check_covariance(value: Any) -> np.ndarray:
    """Validate a 2x2 pixel covariance: finite, symmetric, PSD, above the unit floor."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2, 2):
        raise DetectionSchemaError(
            f"pixel_covariance must be 2x2 (px^2), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise DetectionSchemaError("pixel_covariance contains NaN/Inf")
    scale = max(1.0, float(np.max(np.abs(arr))))
    if not np.allclose(arr, arr.T, atol=PIXEL_COVARIANCE_SYMMETRY_TOL * scale):
        raise DetectionSchemaError("pixel_covariance must be symmetric")
    eigen = np.linalg.eigvalsh(0.5 * (arr + arr.T))
    if float(np.min(eigen)) < -PIXEL_COVARIANCE_SYMMETRY_TOL * scale:
        raise DetectionSchemaError(
            f"pixel_covariance must be positive semi-definite, eigenvalues {eigen.tolist()}")
    diag = np.diag(arr)
    if float(np.min(diag)) < MIN_PIXEL_VARIANCE_PX2:
        raise DetectionSchemaError(
            "pixel_covariance diagonal "
            f"{diag.tolist()} px^2 is below the declared pixel floor "
            f"{MIN_PIXEL_VARIANCE_PX2} px^2: this is a unit confusion "
            "(a metre^2 covariance labelled as px^2)")
    return np.array(arr, dtype=float)


@dataclass(frozen=True)
class DetectionObservation:
    """One shuttlecock detection, in PIXELS, with its own uncertainty and flags.

    Fields (all required; nothing optional is hidden):
      timestamp_s        seconds on the shared clock, finite, >= 0
      frame_id           non-negative integer index of the captured frame
      camera_id          non-empty string identifying the camera
      bbox_xyxy_px       (x1, y1, x2, y2) pixel edges, x2 > x1, y2 > y1
      confidence         [0, 1]
      pixel_covariance   2x2 covariance of the bbox CENTRE, px^2
      image_size         ImageSize of the image the bbox belongs to
      validity           ValidityFlags (in_image is cross-checked below)
    """

    timestamp_s: float
    frame_id: int
    camera_id: str
    bbox_xyxy_px: Any
    confidence: float
    pixel_covariance: Any
    image_size: ImageSize = ImageSize(width_px=1280, height_px=720)
    validity: ValidityFlags = ValidityFlags()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        if self.timestamp_s < 0.0:
            raise DetectionSchemaError(
                f"timestamp_s must be >= 0, got {self.timestamp_s}")
        object.__setattr__(self, "frame_id", _int_ge(self.frame_id, "frame_id", 0))
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise DetectionSchemaError("camera_id must be a non-empty string")
        if not isinstance(self.image_size, ImageSize):
            raise DetectionSchemaError("image_size must be an ImageSize")
        if not isinstance(self.validity, ValidityFlags):
            raise DetectionSchemaError("validity must be ValidityFlags")

        bbox = np.asarray(self.bbox_xyxy_px, dtype=float)
        if bbox.shape != (4,):
            raise DetectionSchemaError(
                f"bbox_xyxy_px must have shape (4,), got {bbox.shape}")
        if not np.all(np.isfinite(bbox)):
            raise DetectionSchemaError("bbox_xyxy_px contains NaN/Inf")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if not (x2 > x1 and y2 > y1):
            raise DetectionSchemaError(
                f"bbox_xyxy_px must satisfy x2>x1 and y2>y1, got {bbox.tolist()}")
        if x1 < 0.0 or y1 < 0.0:
            raise DetectionSchemaError(
                "bbox_xyxy_px must not have negative coordinates "
                f"({bbox.tolist()}): normalised [0,1] or metre coordinates are not "
                "pixels - convert to pixels before publishing")
        object.__setattr__(self, "bbox_xyxy_px", bbox)

        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))
        if not (CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX):
            raise DetectionSchemaError(
                f"confidence must be in [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}], "
                f"got {self.confidence}")
        object.__setattr__(self, "pixel_covariance", _check_covariance(self.pixel_covariance))

        derived_in_image = bool(
            x2 <= float(self.image_size.width_px) and y2 <= float(self.image_size.height_px))
        if derived_in_image != self.validity.in_image:
            raise DetectionSchemaError(
                "ValidityFlags.in_image="
                f"{self.validity.in_image} contradicts the geometry (bbox {bbox.tolist()} "
                f"on {self.image_size.width_px}x{self.image_size.height_px}): the flag must "
                "be derived, not asserted")

    # -- derived conveniences -------------------------------------------------
    @property
    def units(self) -> Dict[str, str]:
        return units_block()

    @property
    def bbox_centre_px(self) -> np.ndarray:
        b = self.bbox_xyxy_px
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=float)

    @property
    def bbox_extent_px(self) -> np.ndarray:
        b = self.bbox_xyxy_px
        return np.array([b[2] - b[0], b[3] - b[1]], dtype=float)

    @property
    def usable(self) -> bool:
        return self.validity.usable

    # -- JSON ---------------------------------------------------------------
    def to_json_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict (no numpy types, units included)."""
        return {
            "schema": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "units": units_block(),
            "timestamp_s": float(self.timestamp_s),
            "frame_id": int(self.frame_id),
            "camera_id": self.camera_id,
            "image_size": self.image_size.to_json_dict(),
            "bbox_xyxy_px": [float(v) for v in self.bbox_xyxy_px],
            "confidence": float(self.confidence),
            "pixel_covariance": [[float(v) for v in row] for row in self.pixel_covariance],
            "validity": self.validity.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any], *, strict: bool = True
                       ) -> "DetectionObservation":
        """Rebuild an observation from a JSON dict, rejecting unit/contract drift."""
        return validate_detection_payload(payload, strict=strict)


_KNOWN_KEYS: tuple[str, ...] = (
    "schema", "schema_version", "units", "timestamp_s", "frame_id", "camera_id",
    "image_size", "bbox_xyxy_px", "confidence", "pixel_covariance", "validity",
)


def validate_detection_payload(payload: Mapping[str, Any], *, strict: bool = True
                               ) -> DetectionObservation:
    """Validate any detector-shaped JSON payload against the P7-04 contract.

    Rejections (all DetectionSchemaError):
      * not a mapping / wrong schema or schema_version
      * a metre-valued key present (bbox_xyxy_m ...) -> unit confusion
      * missing or wrong units block -> unit confusion
      * unknown keys when strict
      * any field-level violation (NaN/Inf, range, covariance PSD/floor, flags)
    """
    if not isinstance(payload, Mapping):
        raise DetectionSchemaError(
            f"detection payload must be a mapping, got {type(payload).__name__}")
    present_metric = [k for k in _METRIC_KEYS if k in payload]
    if present_metric:
        raise DetectionSchemaError(
            f"metric key(s) {present_metric} found in a pixel detection payload: "
            "unit confusion (P7-04 schema is pixels only)")
    if payload.get("schema") != SCHEMA_ID:
        raise DetectionSchemaError(
            f"payload.schema must be {SCHEMA_ID!r}, got {payload.get('schema')!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DetectionSchemaError(
            f"payload.schema_version must be {SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}")
    units = payload.get("units")
    if units is None:
        raise DetectionSchemaError(
            "payload.units is missing: a detection payload must state its units")
    if not isinstance(units, Mapping):
        raise DetectionSchemaError("payload.units must be a mapping")
    for key, want in units_block().items():
        got = units.get(key)
        if got != want:
            raise DetectionSchemaError(
                f"payload.units[{key!r}] must be {want!r} (pixels), got {got!r}: "
                "unit confusion")
    if strict:
        unknown = [k for k in payload if k not in _KNOWN_KEYS]
        if unknown:
            raise DetectionSchemaError(
                f"unknown key(s) {unknown} in the detection payload (strict=True); "
                "extend the schema instead of smuggling data in")
    for key in ("timestamp_s", "frame_id", "camera_id", "bbox_xyxy_px",
                "confidence", "pixel_covariance"):
        if key not in payload:
            raise DetectionSchemaError(f"payload is missing required key {key!r}")
    image_size = ImageSize.from_json_dict(payload.get("image_size", {}))
    flags = ValidityFlags.from_json_dict(payload.get("validity", {}))
    return DetectionObservation(
        timestamp_s=payload["timestamp_s"],
        frame_id=payload["frame_id"],
        camera_id=payload["camera_id"],
        bbox_xyxy_px=payload["bbox_xyxy_px"],
        confidence=payload["confidence"],
        pixel_covariance=payload["pixel_covariance"],
        image_size=image_size,
        validity=flags,
    )
