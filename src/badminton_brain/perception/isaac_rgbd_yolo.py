# -*- coding: utf-8 -*-
"""Isaac RGB-D camera -> existing YOLO detector -> timestamped shuttle measurement.

This module is deliberately importable without Isaac Sim or Ultralytics installed.
The Isaac camera is supplied by dependency injection, and the learned detector is
loaded only when UltralyticsShuttleDetector is constructed.

Authenticity boundaries:
- It consumes rendered RGB and distance-to-image-plane buffers, never shuttle GT.
- It only supports one environment until env-origin to court-frame handling is verified.
- The configured camera rig and pixel/depth uncertainty are simulation engineering
  parameters, not measured hardware calibration.
- A valid depth projection is a measurement candidate, not proof of 3D accuracy.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..interfaces import PerceptionModule
from ..types import BrainBoundaryError, RobotSensorState, ShuttleMeasurement


def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy, Torch, or Isaac Lab ProxyArray data to a CPU NumPy array."""
    if value is None:
        raise BrainBoundaryError("camera returned a missing data buffer")
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray) and hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    return np.asarray(value)


def _first_view(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim >= 1 and array.shape[0] > 1:
        raise BrainBoundaryError(
            name + " contains more than one view; this adapter is N=1 only")
    return array


def _rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the local-to-world rotation matrix for an (x, y, z, w) quaternion."""
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise BrainBoundaryError("camera quaternion must be four finite xyzw values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise BrainBoundaryError("camera quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def read_camera_frame(camera: Any) -> dict[str, np.ndarray]:
    """Read one initialized Isaac Lab camera view and its runtime calibration.

    RGB is normalized to uint8 RGB. Depth remains in metres as returned by
    distance_to_image_plane. The pose quaternion uses Isaac Lab's OpenGL camera
    convention, where the optical axis is -Z and image up is +Y.
    """
    data = getattr(camera, "data", None)
    if data is None:
        raise BrainBoundaryError("camera object has no initialized data")
    output = getattr(data, "output", None)
    if not isinstance(output, Mapping):
        raise BrainBoundaryError("camera data.output is not available yet")
    if "rgb" not in output:
        raise BrainBoundaryError("camera output is missing rgb")
    if "distance_to_image_plane" not in output:
        raise BrainBoundaryError("camera output is missing distance_to_image_plane")

    rgb = _first_view(output["rgb"], "rgb")
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise BrainBoundaryError("rgb must have shape HxWxC, got " + repr(rgb.shape))
    rgb = rgb[:, :, :3]
    if not np.issubdtype(rgb.dtype, np.integer):
        finite = rgb[np.isfinite(rgb)]
        maximum = float(finite.max()) if finite.size else 0.0
        if maximum <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)

    depth = _first_view(output["distance_to_image_plane"], "distance_to_image_plane")
    if depth.ndim == 3:
        if depth.shape[2] < 1:
            raise BrainBoundaryError("depth image has no channels")
        depth = depth[:, :, 0]
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise BrainBoundaryError(
            "depth shape must match rgb height/width, got " + repr(depth.shape))
    depth = np.asarray(depth, dtype=np.float32)

    if not all(hasattr(data, field) for field in
               ("intrinsic_matrices", "pos_w", "quat_w_opengl")):
        raise BrainBoundaryError(
            "camera must expose intrinsic_matrices, pos_w, and quat_w_opengl")

    intrinsic = _first_view(data.intrinsic_matrices, "intrinsic_matrices")
    position = _first_view(data.pos_w, "pos_w").reshape(-1)
    quaternion = _first_view(data.quat_w_opengl, "quat_w_opengl").reshape(-1)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise BrainBoundaryError("camera intrinsic matrix must be finite 3x3")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise BrainBoundaryError("camera world position must be three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise BrainBoundaryError("camera OpenGL quaternion must be four finite values")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise BrainBoundaryError("camera focal lengths must be positive")

    return {
        "rgb": rgb,
        "depth_m": depth,
        "intrinsic_matrix_px": np.asarray(intrinsic, dtype=float),
        "position_world_m": np.asarray(position, dtype=float),
        "quaternion_world_opengl_xyzw": np.asarray(quaternion, dtype=float),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class UltralyticsShuttleDetector:
    """Adapter for an already-trained Ultralytics-compatible checkpoint.

    No training or checkpoint conversion occurs here. Some legacy ETH YOLOv8
    state-dict checkpoints are not directly loadable by Ultralytics YOLO(); use a
    compatible existing checkpoint or a separately audited architecture adapter.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        imgsz: int = 1024,
        conf: float = 0.25,
        device: str = "cuda:0",
        class_id: Optional[int] = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("checkpoint does not exist: " + str(self.checkpoint_path))
        if int(imgsz) < 32:
            raise ValueError("imgsz must be at least 32")
        if not (0.0 <= float(conf) <= 1.0):
            raise ValueError("conf must be in [0, 1]")
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = str(device)
        self.class_id = None if class_id is None else int(class_id)
        self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is unavailable in this Isaac environment; no fallback detector is used"
            ) from exc
        try:
            self.model = YOLO(str(self.checkpoint_path))
        except Exception as exc:
            raise RuntimeError(
                "checkpoint is not directly loadable by Ultralytics YOLO(); "
                "select an existing compatible checkpoint or implement an explicit adapter"
            ) from exc
        self.model_names = getattr(self.model, "names", None)

    def detect(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Run the existing model. Ultralytics NumPy input is BGR, so convert explicitly."""
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("detector input must be an HxWx3 RGB image")
        source_bgr = np.ascontiguousarray(image[:, :, ::-1])
        kwargs: dict[str, Any] = {
            "source": source_bgr,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "device": self.device,
            "verbose": False,
        }
        if self.class_id is not None:
            kwargs["classes"] = [self.class_id]
        results = self.model.predict(**kwargs)
        detections: list[dict[str, Any]] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_numpy(boxes.xyxy)
            confidence = _to_numpy(boxes.conf).reshape(-1)
            classes = _to_numpy(boxes.cls).reshape(-1)
            for box, score, cls in zip(xyxy, confidence, classes):
                detections.append({
                    "xyxy": [float(v) for v in np.asarray(box).reshape(4)],
                    "confidence": float(score),
                    "class_id": int(cls),
                })
        return detections


class IsaacRgbdPerceptionModule(PerceptionModule):
    """Single-view RGB-D perception module for the existing brain message contract.

    The returned point is back-projected from the highest-confidence target-class
    bounding-box centre using the depth pixels in a central box patch. A second
    camera can be evaluated independently and compared by the integration probe.
    This is intentionally not labelled as stereo triangulation.
    """

    name = "isaac_rgbd_yolo_perception"
    is_implemented = True

    def __init__(
        self,
        camera: Any,
        detector: Any,
        *,
        num_envs: int = 1,
        target_class_id: Optional[int] = 0,
        min_depth_m: float = 0.05,
        max_depth_m: float = 20.0,
        depth_patch_fraction: float = 0.30,
        pixel_sigma_px: float = 1.5,
        depth_sigma_floor_m: float = 0.005,
    ) -> None:
        if int(num_envs) != 1:
            raise ValueError("IsaacRgbdPerceptionModule is N=1 only; env-origin conversion is unverified")
        if not (0.0 < float(min_depth_m) < float(max_depth_m)):
            raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
        if not (0.0 < float(depth_patch_fraction) <= 1.0):
            raise ValueError("depth_patch_fraction must be in (0, 1]")
        if float(pixel_sigma_px) <= 0.0 or float(depth_sigma_floor_m) <= 0.0:
            raise ValueError("uncertainty proxy values must be positive")
        self.camera = camera
        self.detector = detector
        self.target_class_id = None if target_class_id is None else int(target_class_id)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_patch_fraction = float(depth_patch_fraction)
        self.pixel_sigma_px = float(pixel_sigma_px)
        self.depth_sigma_floor_m = float(depth_sigma_floor_m)
        self._previous_position: Optional[np.ndarray] = None
        self._previous_timestamp: Optional[float] = None
        self.last_diagnostic: dict[str, Any] = {}

    def reset(self, env_ids) -> None:
        self._previous_position = None
        self._previous_timestamp = None
        self.last_diagnostic = {}

    @staticmethod
    def _parse_detections(raw: Any, target_class_id: Optional[int]) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                continue
            xyxy_value = item.get("xyxy")
            if xyxy_value is None:
                xyxy_value = [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
            try:
                box = np.asarray(xyxy_value, dtype=float).reshape(4)
                confidence = float(item.get("confidence", item.get("conf", 0.0)))
                class_value = item.get("class_id", item.get("class"))
                class_id = None if class_value is None else int(class_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(box).all() or not math.isfinite(confidence):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            if target_class_id is not None and class_id is not None and class_id != target_class_id:
                continue
            parsed.append({
                "xyxy": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
            })
        return sorted(parsed, key=lambda row: row["confidence"], reverse=True)

    def _invalid_measurement(
        self, timestamp: float, reason: str, diagnostic: dict[str, Any], started: float
    ) -> ShuttleMeasurement:
        self.last_diagnostic = dict(diagnostic)
        self.last_diagnostic.update({
            "valid": False,
            "reason": reason,
            "position_court_m": [0.0, 0.0, 0.0],
            "velocity_court_mps": [0.0, 0.0, 0.0],
            "covariance_status": "TEMP_PARAMETERIZED_PROXY",
            "processing_ms": (time.perf_counter() - started) * 1000.0,
        })
        return ShuttleMeasurement(
            position=np.zeros((1, 3), dtype=float),
            velocity=np.zeros((1, 3), dtype=float),
            covariance=(np.eye(3, dtype=float) * 1e6)[None, :, :],
            timestamp=timestamp,
            valid_mask=np.array([0.0]),
        )

    def process(self, sensors: RobotSensorState) -> ShuttleMeasurement:
        if not isinstance(sensors, RobotSensorState):
            raise BrainBoundaryError("IsaacRgbdPerceptionModule expects RobotSensorState")
        timestamp = float(sensors.timestamp)
        if not math.isfinite(timestamp):
            raise BrainBoundaryError("sensor timestamp must be finite")
        started = time.perf_counter()
        frame = read_camera_frame(self.camera)
        rgb = frame["rgb"]
        depth = frame["depth_m"]
        intrinsic = frame["intrinsic_matrix_px"]
        camera_position = frame["position_world_m"]
        camera_quaternion = frame["quaternion_world_opengl_xyzw"]
        rotation = _rotation_matrix_xyzw(camera_quaternion)
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        height, width = depth.shape

        base_diagnostic: dict[str, Any] = {
            "timestamp_s": timestamp,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "intrinsic_matrix_px": intrinsic.tolist(),
            "camera_position_world_m": camera_position.tolist(),
            "camera_quaternion_world_opengl_xyzw": camera_quaternion.tolist(),
            "depth_source": "distance_to_image_plane",
            "backprojection_convention": "OpenGL camera frame: +x right, +y up, -z forward",
        }
        raw = self.detector.detect(rgb)
        detections = self._parse_detections(raw, self.target_class_id)
        base_diagnostic["raw_detection_count"] = len(raw or [])
        base_diagnostic["target_detection_count"] = len(detections)
        if not detections:
            return self._invalid_measurement(
                timestamp, "no_target_detection", base_diagnostic, started)

        detection = detections[0]
        x1, y1, x2, y2 = detection["xyxy"]
        x1 = max(0.0, min(float(width - 1), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height - 1), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            return self._invalid_measurement(
                timestamp, "box_outside_image", base_diagnostic, started)

        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        half_w = max(0, int(math.floor((x2 - x1) * self.depth_patch_fraction / 2.0)))
        half_h = max(0, int(math.floor((y2 - y1) * self.depth_patch_fraction / 2.0)))
        center_u = int(round(u))
        center_v = int(round(v))
        patch_x0 = max(0, center_u - half_w)
        patch_x1 = min(width, center_u + half_w + 1)
        patch_y0 = max(0, center_v - half_h)
        patch_y1 = min(height, center_v + half_h + 1)
        patch = depth[patch_y0:patch_y1, patch_x0:patch_x1]
        valid_depth = patch[
            np.isfinite(patch) & (patch > self.min_depth_m) & (patch < self.max_depth_m)
        ]
        base_diagnostic.update({
            "detection": detection,
            "bbox_xyxy_px": [x1, y1, x2, y2],
            "bbox_center_uv_px": [u, v],
            "depth_patch_xyxy_px": [patch_x0, patch_y0, patch_x1, patch_y1],
            "depth_valid_pixel_count": int(valid_depth.size),
        })
        if valid_depth.size == 0:
            return self._invalid_measurement(
                timestamp, "no_valid_depth_in_center_patch", base_diagnostic, started)

        z = float(np.median(valid_depth))
        mad = float(np.median(np.abs(valid_depth - z)))
        sigma_z = max(self.depth_sigma_floor_m, 1.4826 * mad)
        # Pixel u increases right and v increases down. OpenGL camera coordinates
        # use +X right, +Y up and -Z forward; depth is positive metres.
        point_opengl = np.array([
            (u - cx) * z / fx,
            -(v - cy) * z / fy,
            -z,
        ], dtype=float)
        position_world = camera_position + rotation @ point_opengl

        sigma_u = max(0.5, self.pixel_sigma_px)
        sigma_v = max(0.5, self.pixel_sigma_px)
        sigma_camera = np.array([
            z * sigma_u / fx,
            z * sigma_v / fy,
            sigma_z,
        ], dtype=float)
        covariance_world = rotation @ np.diag(sigma_camera ** 2) @ rotation.T
        covariance_world = 0.5 * (covariance_world + covariance_world.T)

        velocity_status = "BOOTSTRAP_ZERO_FIRST_VALID_FRAME"
        velocity = np.zeros(3, dtype=float)
        if self._previous_position is not None and self._previous_timestamp is not None:
            dt = timestamp - self._previous_timestamp
            if dt <= 0.0:
                raise BrainBoundaryError("valid RGB-D measurements require strictly increasing timestamps")
            velocity = (position_world - self._previous_position) / dt
            velocity_status = "FINITE_DIFFERENCE_BETWEEN_VALID_RGBD_MEASUREMENTS"
        self._previous_position = position_world.copy()
        self._previous_timestamp = timestamp

        self.last_diagnostic = dict(base_diagnostic)
        self.last_diagnostic.update({
            "valid": True,
            "reason": "detection_and_depth_valid",
            "confidence": float(detection["confidence"]),
            "class_id": detection["class_id"],
            "depth_m": z,
            "depth_mad_m": mad,
            "position_court_m": position_world.tolist(),
            "velocity_court_mps": velocity.tolist(),
            "velocity_status": velocity_status,
            "covariance_court_m2": covariance_world.tolist(),
            "covariance_status": "TEMP_PARAMETERIZED_PROXY",
            "covariance_note": (
                "pixel_sigma_px and depth_sigma_floor_m are engineering proxies, not calibrated noise"
            ),
            "processing_ms": (time.perf_counter() - started) * 1000.0,
        })
        return ShuttleMeasurement(
            position=position_world[None, :],
            velocity=velocity[None, :],
            covariance=covariance_world[None, :, :],
            timestamp=timestamp,
            valid_mask=np.array([1.0]),
        )


__all__ = [
    "IsaacRgbdPerceptionModule",
    "UltralyticsShuttleDetector",
    "read_camera_frame",
]
