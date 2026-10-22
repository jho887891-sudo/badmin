# -*- coding: utf-8 -*-
"""R11 M1: real YOLO -> association -> 2D EKF -> image-plane horizon prediction -> overlay video.

This is the first REAL vision path in this repository: it opens an actual video file, runs an
actual YOLO checkpoint forward pass on every frame, associates the boxes, feeds them through
the P7-04/P7-05 contracts, and writes an overlay video plus per-frame JSONL.

What is real and what is not
----------------------------
REAL: the video pixels, the checkpoint (sha256 recorded), the forward pass, the box
coordinates, the timings, the EKF states and the predictions.

NOT CLAIMED: no metre-valued 3D position is produced anywhere.  Every number in tracks.jsonl is
px, px/s or px^2 and labelled IMAGE_PLANE_2D.  A 2D image has no depth, so the horizon
prediction is an image-plane projection.  Detection quality on a given clip is whatever the
checkpoint does; nothing here turns it into an accuracy claim.

Why the horizon prediction lives here
-------------------------------------
badminton_brain.prediction.horizon_prediction.PolynomialHorizonPredictor is METRIC
(final_position_m, error bounds in metres, gravity in m/s^2).  Using it on pixels would be a
unit lie, so the image-plane horizon is produced by propagating the EKF's OWN constant-velocity
transition and process noise forward: mean = F(dt) x, covariance = F P F^T + Q(dt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _extra in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from perception.shuttle_detection.detection_output import (  # noqa: E402
    DetectionObservation,
    ImageSize,
    ValidityFlags,
)
from badminton_brain.estimation.shuttle_state_ekf import (  # noqa: E402
    MODE_IMAGE_2D,
    ShuttleEKFError,
    ShuttleStateEKF,
)
from badminton_brain.prediction.position_uncertainty import (  # noqa: E402
    position_uncertainty,
)

PX = "IMAGE_PLANE_2D"
HORIZONS_S = (0.3, 0.5, 0.8)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_input_source(source: Any, *, max_frames: int = 0) -> Dict[str, Any]:
    """Hash actual bytes of frame inputs, not only a .txt list of their paths.

    A changed image must change the fingerprint even if the .txt manifest stays
    identical.  Video files are hashed by complete bytes.  Image directories
    are hashed in stable source order, including the unique ordinal.
    """
    if isinstance(source, ImageSequenceFrameSource):
        paths = source.images[:max_frames] if max_frames else source.images
        digest = hashlib.sha256()
        for index, image_path in enumerate(paths):
            item_sha = sha256_file(image_path)
            digest.update(("%d:%s\n" % (index, item_sha)).encode("ascii"))
        return {"kind": "REAL_RGB_FRAMES", "sha256": digest.hexdigest(),
                "hashed_frame_count": len(paths),
                "list_sha256": sha256_file(source.path)
                if os.path.isfile(source.path) else None}
    return {"kind": "REAL_RGB_VIDEO", "sha256": sha256_file(source.path),
            "hashed_frame_count": None}


def git_head(repo_root: str) -> str:
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


@dataclass
class Frame:
    frame_id: int
    timestamp_s: float
    camera_id: str
    image: Any
    timestamp_source: str


class VideoFrameSource:
    source_class = "REAL_RGB_VIDEO"

    """Reads a video file and reports the REAL presentation timestamps when they exist.

    CAP_PROP_POS_MSEC after a successful read is the decoder presentation time, a genuine
    capture-time estimate for a constant-rate recording.  When it comes back as 0 for every
    frame, the timestamps are recomputed from the declared FPS and labelled ASSUMED_FPS so
    nobody mistakes them for hardware sync.
    """

    def __init__(self, path: str, camera_id: str = "video0") -> None:
        import cv2
        self.path = os.path.abspath(path)
        self.camera_id = camera_id
        self._cv2 = cv2
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise RuntimeError("cannot open video: %s" % self.path)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 0.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.timestamp_source = "UNKNOWN"

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        pts_seen = 0
        pts_available = True
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            pts_ms = float(self.capture.get(self._cv2.CAP_PROP_POS_MSEC))
            if pts_ms > 0.0:
                timestamp = pts_ms / 1000.0
                pts_seen += 1
                source = "PTS_VIDEO"
            else:
                pts_available = False
                timestamp = index / self.fps if self.fps > 0 else float(index)
                source = "ASSUMED_FPS" if self.fps > 0 else "INDEX_ONLY"
            yield Frame(index, timestamp, self.camera_id, image, source)
            index += 1
        self.timestamp_source = ("PTS_VIDEO" if pts_available and pts_seen > 0
                                 else ("ASSUMED_FPS" if self.fps > 0 else "INDEX_ONLY"))

    def release(self) -> None:
        self.capture.release()


class ImageSequenceFrameSource:
    """A directory of images or a text file listing image paths, in the same Frame contract.

    Timestamps are ASSERTED, never measured: an image sequence carries no capture clock, so the
    source is labelled ASSUMED_FPS and the declared --source-fps is recorded in the manifest.
    This exists because the in-repository real videos have zero detections with the current
    checkpoint (a domain-gap finding, see the report), while real annotated camera frames are
    available; the pipeline must be able to consume both through one contract.
    """

    source_class = "REAL_RGB_FRAMES"

    def __init__(self, path: str, camera_id: str = "seq0", fps: float = 30.0,
                 limit: int = 0) -> None:
        self.path = os.path.abspath(path)
        self.camera_id = camera_id
        self.fps = float(fps)
        self.timestamp_source = "ASSUMED_FPS"
        if os.path.isdir(self.path):
            names = sorted(
                os.path.join(self.path, name) for name in os.listdir(self.path)
                if os.path.splitext(name)[1].lower() in
                (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        elif self.path.lower().endswith(".txt"):
            with open(self.path, encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
        else:
            raise RuntimeError("image sequence source must be a directory or a .txt list")
        if limit:
            names = names[:limit]
        if not names:
            raise RuntimeError("no images found for %s" % self.path)
        self.images = names
        self.frame_count = len(names)
        # CRITICAL: a frame list is NOT necessarily consecutive.  If every name carries a
        # trailing integer, timestamp from that integer (so a gap becomes a real dt gap);
        # otherwise fall back to the index and SAY that the spacing is assumed.  Without this,
        # a sampled split (ticino_2_00159, 00188, 00430 ...) is fed to the EKF as 1/30 s
        # steps and the horizon prediction is measured against a wrong clock: the ethval run
        # produced a 266 px median error that is a clock artefact, not a model error.
        self.indices = []
        for name in names:
            stem = os.path.splitext(os.path.basename(name))[0]
            trailing = ""
            for char in reversed(stem):
                if char.isdigit():
                    trailing = char + trailing
                else:
                    break
            self.indices.append(int(trailing) if trailing else None)
        if all(value is not None for value in self.indices) and \
                len(set(self.indices)) == len(self.indices):
            self.timestamp_source = "FILENAME_INDEX_ASSUMED_FPS"
            self.spacing_regular = (self.indices == list(range(self.indices[0],
                                                               self.indices[0]
                                                               + len(self.indices))))
        else:
            self.indices = list(range(len(names)))
            self.timestamp_source = "ASSUMED_FPS"
            self.spacing_regular = True
        import cv2
        probe = cv2.imread(names[0])
        if probe is None:
            raise RuntimeError("cannot read first image: %s" % names[0])
        self.height, self.width = probe.shape[:2]

    def __iter__(self) -> Iterator[Frame]:
        import cv2
        for position, name in enumerate(self.images):
            image = cv2.imread(name)
            if image is None:
                continue
            clock = self.indices[position]
            yield Frame(position, clock / self.fps if self.fps > 0 else float(clock),
                        self.camera_id, image, self.timestamp_source)

    def release(self) -> None:
        return None


class YoloDetector:
    """A thin, honest wrapper around an ultralytics checkpoint."""

    def __init__(self, weights: str, *, imgsz: int, conf: float, iou: float,
                 device: str, max_det: int = 64) -> None:
        from ultralytics import YOLO
        self.weights = os.path.abspath(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = str(device)
        self.max_det = int(max_det)
        self.model = YOLO(self.weights)
        self.names = dict(getattr(self.model, "names", {}) or {})

    def describe(self) -> Dict[str, Any]:
        import torch
        import ultralytics
        try:
            params = sum(p.numel() for p in self.model.model.parameters())
        except Exception:
            params = None
        return {
            "weights_path": self.weights,
            "weights_sha256": sha256_file(self.weights),
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": self.device,
            "parameter_count": params,
            "class_names": self.names,
            "imgsz": self.imgsz, "conf": self.conf, "iou": self.iou,
            "max_det": self.max_det,
        }

    def detect(self, image: Any, frame_id: int, timestamp_s: float,
               camera_id: str) -> Tuple[List[DetectionObservation], float]:
        started = time.perf_counter()
        results = self.model.predict(
            source=image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, max_det=self.max_det, verbose=False,
        )
        elapsed_s = time.perf_counter() - started
        height, width = image.shape[:2]
        size = ImageSize(width_px=int(width), height_px=int(height))
        observations: List[DetectionObservation] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                for bbox, confidence in zip(xyxy, confidences):
                    extent = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
                    sigma = max(0.25 * extent, 1.0)
                    observations.append(DetectionObservation(
                        timestamp_s=float(timestamp_s),
                        frame_id=int(frame_id),
                        camera_id=camera_id,
                        bbox_xyxy_px=[float(v) for v in bbox],
                        confidence=float(confidence),
                        pixel_covariance=[[sigma * sigma, 0.0], [0.0, sigma * sigma]],
                        image_size=size,
                        validity=ValidityFlags(detected=True, in_image=True,
                                               confidence_ok=True, covariance_ok=True,
                                               timestamp_ok=True),
                    ))
        return observations, elapsed_s


class ImagePlaneTracker:
    """Single-target association with a distance gate, then the P7-05 constant-velocity EKF."""

    def __init__(self, *, gate_px: float, dt_s: float, camera_id: str,
                 process_accel_std: "Optional[float]" = None,
                 measurement_std: "Optional[float]" = None) -> None:
        self.gate_px = float(gate_px)
        self.dt_s = float(dt_s)
        self.camera_id = camera_id
        self.ekf = ShuttleStateEKF(mode=MODE_IMAGE_2D, dt_s=self.dt_s,
                                   process_accel_std=process_accel_std,
                                   measurement_std=measurement_std)
        self.track_id: Optional[int] = None
        self.history: List[Tuple[float, float, float]] = []
        self.rejections = 0
        self.associations = 0
        self.reinitialisations = 0
        self.last_result = "IDLE"

    def _has_state(self) -> bool:
        """The EKF raises before initialisation, so "has state" must be probed, not compared."""
        try:
            self.ekf.state
        except ShuttleEKFError:
            return False
        return True

    @property
    def alive(self) -> bool:
        if not self._has_state():
            return False
        try:
            return not self.ekf.is_lost
        except ShuttleEKFError:
            return False

    def _centre(self, observation: DetectionObservation) -> np.ndarray:
        return np.asarray(observation.bbox_centre_px, dtype=float)

    def step(self, observations: Sequence[DetectionObservation],
             timestamp_s: float) -> Dict[str, Any]:
        predicted = (np.asarray(self.ekf.position, dtype=float)
                     if self._has_state() else None)
        chosen: Optional[DetectionObservation] = None
        distance: Optional[float] = None
        if self.alive and predicted is not None and observations:
            best = None
            for observation in observations:
                delta = self._centre(observation) - predicted
                d = float(np.hypot(delta[0], delta[1]))
                if best is None or d < best[1]:
                    best = (observation, d)
            if best is not None and best[1] <= self.gate_px:
                chosen, distance = best
            elif best is not None:
                distance = best[1]
                self.rejections += 1
                self.last_result = "REJECTED_GATE"
        elif observations:
            chosen = max(observations, key=lambda o: o.confidence)
            self.last_result = "REINITIALISED"

        if chosen is None:
            if self._has_state() and self.alive:
                self.last_result = "COASTING"
                try:
                    self.ekf.predict()
                except ShuttleEKFError:
                    pass
            return {
                "track_id": self.track_id, "result": self.last_result,
                "association_distance_px": distance,
                "detections_in_frame": len(observations),
                "ekf": self._ekf_snapshot(0.0),
            }

        if not self._has_state() or not self.alive:
            centre_measured = self._centre(chosen)
            measurement_cov = np.asarray(chosen.pixel_covariance, dtype=float)
            initial_cov = np.zeros((4, 4))
            initial_cov[:2, :2] = measurement_cov
            initial_cov[2:, 2:] = np.diag([self.ekf._initial_vel_std ** 2] * 2)
            self.ekf.initialize(centre_measured, timestamp_s, covariance=initial_cov)
            self.track_id = 0 if self.track_id is None else self.track_id + 1
            if self.last_result == "REINITIALISED":
                self.reinitialisations += 1
            self.last_result = "INITIALISED"
        else:
            try:
                centre_measured = self._centre(chosen)
                update = self.ekf.step(
                    centre_measured, timestamp_s,
                    measurement_covariance=np.asarray(chosen.pixel_covariance, dtype=float))
            except ShuttleEKFError as exc:
                self.last_result = "EKF_ERROR:%s" % type(exc).__name__
                return {"track_id": self.track_id, "result": self.last_result,
                        "association_distance_px": distance,
                        "detections_in_frame": len(observations),
                        "ekf": self._ekf_snapshot(0.0)}
            if update.accepted:
                self.associations += 1
                self.last_result = "UPDATED"
            else:
                self.rejections += 1
                self.last_result = "EKF_GATED:%s" % update.reason

        centre = np.asarray(self.ekf.position, dtype=float)
        self.history.append((float(timestamp_s), float(centre[0]), float(centre[1])))
        if len(self.history) > 240:
            self.history = self.history[-240:]
        return {
            "track_id": self.track_id, "result": self.last_result,
            "association_distance_px": distance,
            "detections_in_frame": len(observations),
            "ekf": self._ekf_snapshot(timestamp_s),
        }

    def _ekf_snapshot(self, timestamp_s: float) -> Dict[str, Any]:
        if not self._has_state():
            return {"mode": PX, "valid": False}
        covariance = np.asarray(self.ekf.covariance, dtype=float)
        return {
            "mode": PX, "valid": True,
            "timestamp_s": round(float(timestamp_s), 6),
            "centre_px": [round(float(v), 4) for v in np.asarray(self.ekf.position)],
            "velocity_px_s": [round(float(v), 4) for v in np.asarray(self.ekf.velocity)],
            "state_covariance_mixed_units": [[round(float(v), 6) for v in row] for row in covariance],
            "position_covariance_px2": [[round(float(v), 6) for v in row]
                                        for row in covariance[:2, :2]],
            "velocity_covariance_px2_s2": [[round(float(v), 6) for v in row]
                                           for row in covariance[2:, 2:]],
            "steps_without_measurement": int(self.ekf.steps_without_measurement),
            "status": self.ekf.status,
        }

    def horizon(self) -> List[Dict[str, Any]]:
        if not self._has_state() or not self.alive:
            return []
        x = np.asarray(self.ekf.state, dtype=float)
        P = np.asarray(self.ekf.covariance, dtype=float)
        out = []
        for horizon_s in HORIZONS_S:
            F = self.ekf.transition_matrix(horizon_s)
            Q = self.ekf.process_noise(horizon_s)
            mean = F @ x
            cov = F @ P @ F.T + Q
            uncertainty = position_uncertainty(cov)
            out.append({
                "mode": PX,
                "horizon_s": horizon_s,
                "centre_px": [round(float(mean[0]), 4), round(float(mean[1]), 4)],
                "velocity_px_s": [round(float(mean[2]), 4), round(float(mean[3]), 4)],
                "state_covariance_mixed_units": [[round(float(v), 6) for v in row] for row in cov],
                "position_covariance_px2": uncertainty["position_covariance_px2"],
                "velocity_covariance_px2_s2": uncertainty["velocity_covariance_px2_s2"],
                "sigma_major_px": round(uncertainty["position_sigma_major_px"], 4),
                "sigma_minor_px": round(uncertainty["position_sigma_minor_px"], 4),
                "ellipse95": uncertainty["ellipse95"],
                "units": {"position": "px", "velocity": "px/s",
                          "position_covariance": "px^2", "velocity_covariance": "px^2/s^2"},
                "usable_for_metric_control": False,
                "reject_reason": "IMAGE_PLANE_2D_ONLY: no depth, no calibration",
            })
        return out


def draw_overlay(image: Any, *, detections: Sequence[DetectionObservation],
                 history: Sequence[Tuple[float, float, float]],
                 horizons: Sequence[Dict[str, Any]], status_text: str) -> Any:
    import cv2
    canvas = image.copy()
    for observation in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in observation.bbox_xyxy_px]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, "%.2f" % observation.confidence, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for index in range(1, len(history)):
        p0 = (int(round(history[index - 1][1])), int(round(history[index - 1][2])))
        p1 = (int(round(history[index][1])), int(round(history[index][2])))
        cv2.line(canvas, p0, p1, (255, 160, 0), 2)
    if history:
        centre = (int(round(history[-1][1])), int(round(history[-1][2])))
        cv2.circle(canvas, centre, 4, (255, 160, 0), -1)
    colours = {0.3: (0, 0, 255), 0.5: (0, 128, 255), 0.8: (255, 0, 255)}
    for item in horizons:
        point = (int(round(item["centre_px"][0])), int(round(item["centre_px"][1])))
        colour = colours[item["horizon_s"]]
        cv2.circle(canvas, point, 5, colour, 2)
        ellipse = item["ellipse95"]
        axes = (max(1, int(round(ellipse["major_radius_px"]))),
                max(1, int(round(ellipse["minor_radius_px"]))))
        cv2.ellipse(canvas, point, axes, ellipse["angle_deg_image_xy"],
                    0.0, 360.0, colour, 1)
        cv2.putText(canvas, "%.1fs / xy 95%%" % item["horizon_s"], (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "IMAGE_PLANE_2D (px only, no 3D claim)", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas


def open_frame_source(path: str, camera_id: str, fps: float, limit: int = 0):
    """One contract, three real inputs: video file / image directory / .txt image list."""
    if os.path.isdir(path) or path.lower().endswith(".txt"):
        return ImageSequenceFrameSource(path, camera_id=camera_id, fps=fps, limit=limit)
    return VideoFrameSource(path, camera_id=camera_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R11 M1: real video -> real YOLO -> 2D EKF -> image-plane horizon -> overlay")
    parser.add_argument("--source", required=True, help="video file (real capture)")
    parser.add_argument("--weights", required=True, help="YOLO checkpoint (.pt)")
    parser.add_argument("--out-dir", required=True, help="new run directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--gate-px", type=float, default=120.0)
    parser.add_argument("--process-accel-std", type=float, default=None,
                        help="EKF process accel std in px/s^2 (default: the inherited P7-05 "
                             "value). Overriding it is an IN-SAMPLE calibration and is "
                             "recorded in the manifest as such.")
    parser.add_argument("--measurement-std", type=float, default=None,
                        help="EKF measurement std in px (default: the inherited P7-05 value)")
    parser.add_argument("--camera-id", default="video0")
    parser.add_argument("--no-overlay", action="store_true", help="JSONL only")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="FPS to ASSERT for an image sequence (it carries no clock)")
    parser.add_argument("--limit", type=int, default=0,
                        help="max images to take from an image-sequence source")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name in ("detections.jsonl", "tracks.jsonl", "metrics.json", "manifest.json",
                 "overlay.mp4"):
        if os.path.exists(os.path.join(out_dir, name)):
            print("REFUSING: %s already exists in %s (a run directory is append-only)"
                  % (name, out_dir), file=sys.stderr)
            return 2

    source = open_frame_source(args.source, args.camera_id, args.source_fps, args.limit)
    detector = YoloDetector(args.weights, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device)
    fps = source.fps if source.fps > 0 else 30.0
    tracker = ImagePlaneTracker(gate_px=args.gate_px, dt_s=1.0 / fps,
                                camera_id=args.camera_id,
                                process_accel_std=args.process_accel_std,
                                measurement_std=args.measurement_std)

    writer = None
    frames = 0
    detected_frames = 0
    detection_count = 0
    detect_ms: List[float] = []
    track_ms: List[float] = []
    miss_streak = 0
    longest_miss = 0
    timestamp_source = "UNKNOWN"

    det_path = os.path.join(out_dir, "detections.jsonl")
    trk_path = os.path.join(out_dir, "tracks.jsonl")
    with open(det_path, "w", encoding="utf-8") as det_file, \
            open(trk_path, "w", encoding="utf-8") as trk_file:
        for frame in source:
            if args.max_frames and frames >= args.max_frames:
                break
            timestamp_source = frame.timestamp_source
            observations, detect_s = detector.detect(frame.image, frame.frame_id,
                                                     frame.timestamp_s, frame.camera_id)
            detect_ms.append(detect_s * 1000.0)
            detection_count += len(observations)
            if observations:
                detected_frames += 1
                miss_streak = 0
            else:
                miss_streak += 1
                longest_miss = max(longest_miss, miss_streak)

            for observation in observations:
                det_file.write(json.dumps(observation.to_json_dict(),
                                          ensure_ascii=False) + "\n")

            track_started = time.perf_counter()
            track = tracker.step(observations, frame.timestamp_s)
            horizons = tracker.horizon()
            track_ms.append((time.perf_counter() - track_started) * 1000.0)

            trk_file.write(json.dumps({
                "frame_id": frame.frame_id,
                "timestamp_s": round(frame.timestamp_s, 6),
                "timestamp_source": frame.timestamp_source,
                "camera_id": frame.camera_id,
                "input_source_class": source.source_class,
                "measurement_source": "YOLO_BOX",
                "units": PX,
                "association": track,
                "horizon": horizons,
                "miss_streak": miss_streak,
            }, ensure_ascii=False) + "\n")

            if not args.no_overlay:
                status = ("frame %d t=%.3fs det=%d track=%s %s"
                          % (frame.frame_id, frame.timestamp_s, len(observations),
                             track.get("track_id"), track.get("result")))
                canvas = draw_overlay(frame.image, detections=observations,
                                      history=tracker.history, horizons=horizons,
                                      status_text=status)
                if writer is None:
                    height, width = canvas.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(out_dir, "overlay.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (width, height))
                writer.write(canvas)
            frames += 1
            if os.environ.get("R11_PROGRESS"):
                print("frame %d dets=%d cum=%d" % (frame.frame_id, len(observations),
                                                   detection_count),
                      file=sys.stderr, flush=True)
    if writer is not None:
        writer.release()
    source.release()

    def percentiles(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=float)
        return {"p50": round(float(np.percentile(array, 50)), 3),
                "p95": round(float(np.percentile(array, 95)), 3),
                "p99": round(float(np.percentile(array, 99)), 3),
                "max": round(float(array.max()), 3)}

    metrics = {
        "schema": "r11_vision_m1_metrics/v2",
        "source_class": getattr(source, "source_class", "REAL_RGB_VIDEO"),
        "measurement_source": "YOLO_BOX",
        "state_units": PX,
        "frames_processed": frames,
        "frames_with_detection": detected_frames,
        "detection_rate": round(detected_frames / frames, 6) if frames else None,
        "detection_rate_definition": "FRAMES_WITH_ANY_BOX_DIVIDED_BY_FRAMES; NOT GT_RECALL",
        "detections_total": detection_count,
        "detections_per_frame": round(detection_count / frames, 6) if frames else None,
        "longest_miss_streak_frames": longest_miss,
        "association": {
            "accepted": tracker.associations,
            "rejected": tracker.rejections,
            "reinitialisations": tracker.reinitialisations,
            "gate_px": tracker.gate_px,
        },
        "latency_ms": {"detect": percentiles(detect_ms),
                       "track_predict": percentiles(track_ms)},
        "detect_wall_s": round(sum(detect_ms) / 1000.0, 3),
        "timestamp_source": timestamp_source,
        "sequence_spacing_regular": getattr(source, "spacing_regular", None),
        "sequence_frame_index_span": (getattr(source, "indices", None) or [None])[-1]
        if getattr(source, "indices", None) else None,
        "three_d_claimed": False,
        "hit_rate": None,
        "notes": [
            "pixel_covariance is a PROXY derived from the bbox size (0.25 * extent, 1 px floor)",
            "horizon prediction propagates the EKF own CV transition and process noise",
            "no metric 3D position is produced or claimed anywhere in this run",
        ],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=1)

    source_fingerprint = hash_input_source(source, max_frames=args.max_frames)
    manifest = {
        "schema": "r11_vision_m1_manifest/v2",
        "run_id": os.path.basename(out_dir),
        "git_head": git_head(_REPO_ROOT),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "position_uncertainty_sha256": sha256_file(
            os.path.join(_REPO_ROOT, "src", "badminton_brain", "prediction",
                         "position_uncertainty.py")),
        "python": platform.python_version(),
        "host": platform.node(),
        "command": " ".join(sys.argv),
        "source_video": {
            "path": source.path,
            "sha256": source_fingerprint["sha256"],
            "content_fingerprint": source_fingerprint,
            "fps": source.fps, "frame_count": source.frame_count,
            "width": source.width, "height": source.height,
        },
        "detector": detector.describe(),
        "tracker": {"mode": PX, "gate_px": tracker.gate_px, "ekf_dt_s": tracker.dt_s,
                    "process_accel_std_override": args.process_accel_std,
                    "measurement_std_override": args.measurement_std,
                    "noise_calibration": ("IN_SAMPLE_TUNED_ON_THIS_CLIP" if
                                          (args.process_accel_std is not None or
                                           args.measurement_std is not None)
                                          else "INHERITED_P7_05_DEFAULT")},
        "horizons_s": list(HORIZONS_S),
        "outputs": ["overlay.mp4", "detections.jsonl", "tracks.jsonl", "metrics.json",
                    "manifest.json"],
        "input_source_class": getattr(source, "source_class", "REAL_RGB_VIDEO"),
        "not_claimed": ["metric 3D position", "hit rate", "detection accuracy",
                        "hardware-synchronised timestamps"],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)

    print(json.dumps({"run_id": manifest["run_id"], "frames": frames,
                      "detections": detection_count,
                      "detection_rate": metrics["detection_rate"],
                      "weights_sha256": manifest["detector"]["weights_sha256"][:16],
                      "out_dir": out_dir}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
