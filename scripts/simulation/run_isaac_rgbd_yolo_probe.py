#!/usr/bin/env python3
"""Render real Isaac Lab RGB-D frames and run an existing YOLO checkpoint.

This is an N=1 camera/detector/metric-projection probe, not a closed-loop robot run.
The shuttle USD root pose is a visual test fixture; it is never passed to perception.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(PARSER)
PARSER.add_argument("--checkpoint", required=True, help="existing Ultralytics-compatible trained checkpoint")
PARSER.add_argument("--expected-checkpoint-sha256", default=None,
                    help="optional frozen expected SHA256; mismatch aborts before model loading")
PARSER.add_argument("--out", default=None, help="new output directory; existing paths are refused")
PARSER.add_argument("--shuttle-usd", default=None, help="visual shuttle USD; defaults to repository asset")
PARSER.add_argument("--frames", type=int, default=3)
PARSER.add_argument("--imgsz", type=int, default=1024)
PARSER.add_argument("--conf", type=float, default=0.25)
PARSER.add_argument("--class-id", type=int, default=0, help="target class ID; use -1 to disable class filtering")
PARSER.add_argument("--model-device", default="cuda:0", help="device passed to Ultralytics, separate from Isaac AppLauncher")
PARSER.add_argument("--target-pos", type=float, nargs=3, default=[-0.5, 0.0, 1.25],
                    help="visual test-fixture root position in world-frame metres")
ARGS = PARSER.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def main() -> int:
    checkpoint = Path(ARGS.checkpoint).expanduser().resolve()
    shuttle_usd = Path(ARGS.shuttle_usd).expanduser().resolve() if ARGS.shuttle_usd else (
        REPO_ROOT / "assets" / "third_party" / "shuttlecock_visual.usd"
    )
    if not checkpoint.is_file():
        PARSER.error("checkpoint does not exist: " + str(checkpoint))
    if not shuttle_usd.is_file():
        PARSER.error("shuttle visual USD does not exist: " + str(shuttle_usd))
    if ARGS.frames < 1:
        PARSER.error("--frames must be >= 1")
    if ARGS.imgsz < 32:
        PARSER.error("--imgsz must be >= 32")
    if not 0.0 <= ARGS.conf <= 1.0:
        PARSER.error("--conf must be in [0, 1]")

    if ARGS.out is None:
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = REPO_ROOT / "outputs" / "vision" / ("vis_c2_isaac_rgbd_" + run_stamp)
    else:
        out = Path(ARGS.out).expanduser()
        if not out.is_absolute():
            out = REPO_ROOT / out
    if out.exists():
        PARSER.error("refusing to overwrite existing output directory: " + str(out))
    out.mkdir(parents=True, exist_ok=False)

    ARGS.headless = True
    ARGS.enable_cameras = True
    app = None
    try:
        launcher = AppLauncher(ARGS)
        app = launcher.app

        import omni.usd
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom

        from simulation.badminton_scene.scene_cfg import create_scene
        from simulation.badminton_scene.stereo_camera_cfg import build_stereo_camera_sensor_specs
        from badminton_brain.perception.isaac_rgbd_yolo import (
            IsaacRgbdPerceptionModule,
            UltralyticsShuttleDetector,
            read_camera_frame,
        )
        from badminton_brain.types import RobotSensorState

        class_id = None if ARGS.class_id < 0 else int(ARGS.class_id)
        detector = UltralyticsShuttleDetector(
            checkpoint, imgsz=ARGS.imgsz, conf=ARGS.conf,
            device=ARGS.model_device, class_id=class_id,
            expected_sha256=ARGS.expected_checkpoint_sha256,
        )
        sim, scene, _robot, _shuttle, _contact, extra = create_scene(
            num_envs=1,
            extra_sensors=build_stereo_camera_sensor_specs(width=1280, height=720),
            statics_mode="first",
        )
        stage = omni.usd.get_context().get_stage()
        visual_path = "/World/envs/env_0/ShuttleVisual"
        visual = UsdGeom.Xform.Define(stage, visual_path)
        visual.GetPrim().GetReferences().AddReference(str(shuttle_usd))
        visual.AddTranslateOp().Set(Gf.Vec3d(
            float(ARGS.target_pos[0]), float(ARGS.target_pos[1]), float(ARGS.target_pos[2])
        ))
        # Reinitialize after adding the static visual fixture. The pose is used only
        # to place the rendered asset; it is never passed into the detector/estimator.
        sim.reset()
        scene.reset()

        cameras = {
            "left": extra["camera_left"],
            "right": extra["camera_right"],
        }
        modules = {
            name: IsaacRgbdPerceptionModule(camera, detector, num_envs=1, target_class_id=class_id)
            for name, camera in cameras.items()
        }
        for _ in range(6):
            sim.render()
            for camera in cameras.values():
                camera.update(1.0 / 60.0)

        (out / "images").mkdir()
        (out / "depth").mkdir()
        records: list[dict[str, Any]] = []
        camera_shapes: dict[str, Any] = {}
        pair_disagreement: list[float] = []
        latency_by_camera: dict[str, list[float]] = {"left": [], "right": []}
        detections_by_camera: dict[str, int] = {"left": 0, "right": 0}
        valid3d_by_camera: dict[str, int] = {"left": 0, "right": 0}

        for frame_index in range(int(ARGS.frames)):
            timestamp = float(frame_index) / 60.0
            sim.render()
            for camera in cameras.values():
                camera.update(1.0 / 60.0)
            sensors = RobotSensorState(
                base_pose=np.array([[-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                joint_pos=np.zeros((1, 6)),
                joint_vel=np.zeros((1, 6)),
                timestamp=timestamp,
            )
            frame_measurements = {}
            for name, camera in cameras.items():
                measurement = modules[name].process(sensors)
                diagnostic = dict(modules[name].last_diagnostic)
                frame = read_camera_frame(camera)
                camera_shapes[name] = {
                    "rgb_shape": list(frame["rgb"].shape),
                    "depth_shape": list(frame["depth_m"].shape),
                    "intrinsic_matrix_px": frame["intrinsic_matrix_px"].tolist(),
                    "position_world_m": frame["position_world_m"].tolist(),
                    "quaternion_world_opengl_xyzw": frame["quaternion_world_opengl_xyzw"].tolist(),
                }
                stem = name + "_frame_" + str(frame_index).zfill(6)
                rgb_path = out / "images" / (stem + ".png")
                depth_path = out / "depth" / (stem + ".npy")
                overlay_path = out / "images" / (stem + "_overlay.png")
                Image.fromarray(frame["rgb"]).save(rgb_path)
                np.save(depth_path, frame["depth_m"])
                overlay = Image.fromarray(frame["rgb"])
                draw = ImageDraw.Draw(overlay)
                detection = diagnostic.get("detection")
                if detection:
                    box = diagnostic["bbox_xyxy_px"]
                    draw.rectangle(
                        (box[0], box[1], box[2], box[3]),
                        outline=(255, 0, 0), width=3,
                    )
                    draw.text((box[0], max(0, box[1] - 16)),
                              "shuttle conf=" + str(round(float(detection["confidence"]), 3)),
                              fill=(255, 0, 0))
                    detections_by_camera[name] += 1
                if bool(diagnostic.get("valid")):
                    valid3d_by_camera[name] += 1
                latency_by_camera[name].append(float(diagnostic.get("processing_ms", 0.0)))
                record = {
                    "frame_index": frame_index,
                    "timestamp_s": timestamp,
                    "timestamp_source": "render_frame_index_at_60Hz",
                    "camera": name,
                    "measurement_valid": bool(diagnostic.get("valid")),
                    "measurement_reason": diagnostic.get("reason"),
                    "detection": native(detection),
                    "bbox_xyxy_px": native(diagnostic.get("bbox_xyxy_px")),
                    "depth_m": diagnostic.get("depth_m"),
                    "depth_valid_pixel_count": int(diagnostic.get("depth_valid_pixel_count", 0)),
                    "position_world_m": native(measurement.position[0]),
                    "velocity_world_mps": native(measurement.velocity[0]),
                    "covariance_world_m2": native(measurement.covariance[0]),
                    "covariance_status": diagnostic.get("covariance_status"),
                    "velocity_status": diagnostic.get("velocity_status"),
                    "processing_ms": float(diagnostic.get("processing_ms", 0.0)),
                    "rgb_path": str(rgb_path.relative_to(out)).replace("\\", "/"),
                    "depth_path": str(depth_path.relative_to(out)).replace("\\", "/"),
                    "overlay_path": str(overlay_path.relative_to(out)).replace("\\", "/"),
                    "checkpoint_sha256": detector.checkpoint_sha256,
                    "camera_calibration_status": "SIM_CONFIGURED_TEMP_RIG_NOT_HARDWARE_CALIBRATED",
                }
                records.append(record)
                overlay.save(overlay_path)
                frame_measurements[name] = measurement

            left = frame_measurements["left"]
            right = frame_measurements["right"]
            left_valid = bool(np.asarray(left.valid_mask).reshape(-1)[0] > 0.5)
            right_valid = bool(np.asarray(right.valid_mask).reshape(-1)[0] > 0.5)
            if left_valid and right_valid:
                pair_disagreement.append(float(np.linalg.norm(left.position[0] - right.position[0])))

        jsonl_path = out / "detections.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

        detection_status = "PASS" if all(detections_by_camera[name] > 0 for name in cameras) else "BLOCKED_DETECTION"
        rgbd_status = "PASS" if all(valid3d_by_camera[name] > 0 for name in cameras) else "BLOCKED_DEPTH_OR_DETECTION"
        overall = "PASS_VIS_C2_SENSOR_YOLO_RGBD_PROJECTION" if (
            detection_status == "PASS" and rgbd_status == "PASS"
        ) else "BLOCKED"
        metrics = {
            "status": overall,
            "camera_sensor_status": "PASS",
            "detector_execution_status": "PASS",
            "detection_status": detection_status,
            "rgbd_measurement_status": rgbd_status,
            "three_d_accuracy_status": "UNKNOWN_NO_INDEPENDENT_SHUTTLE_CENTRE_GT",
            "hardware_calibration_status": "UNKNOWN_SIM_ONLY_TEMP_RIG",
            "frames_requested": int(ARGS.frames),
            "records_written": len(records),
            "detection_frames_by_camera": detections_by_camera,
            "valid_rgbd_measurements_by_camera": valid3d_by_camera,
            "processing_latency_ms_by_camera": {
                name: {
                    "p50": float(np.percentile(values, 50)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                    "max": float(max(values)) if values else None,
                } for name, values in latency_by_camera.items()
            },
            "left_right_position_disagreement_m": {
                "count": len(pair_disagreement),
                "median": float(np.median(pair_disagreement)) if pair_disagreement else None,
                "max": float(max(pair_disagreement)) if pair_disagreement else None,
                "interpretation": "diagnostic only; no calibrated acceptance threshold is claimed",
            },
            "camera_shapes_and_runtime_calibration": camera_shapes,
            "notes": [
                "The visual shuttle root pose is only a render fixture and is not fed to perception.",
                "RGB-D back-projection is not stereo triangulation.",
                "A valid 3D projection is not a validated 3D accuracy result.",
                "No robot commands, closed-loop control, training, or deployment are performed.",
            ],
        }
        write_json(out / "metrics.json", metrics)

        artifacts = {}
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts[str(path.relative_to(out)).replace("\\", "/")] = sha256_file(path)
        manifest = {
            "schema_version": "vis-c2-isaac-rgbd-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": detector.checkpoint_sha256,
            "expected_checkpoint_sha256": detector.expected_checkpoint_sha256,
            "checkpoint_model_names": str(detector.model_names),
            "checkpoint_format": "Ultralytics YOLO API loaded directly; no conversion or training",
            "shuttle_visual_usd": str(shuttle_usd),
            "shuttle_visual_usd_sha256": sha256_file(shuttle_usd),
            "visual_fixture_root_world_m": [float(v) for v in ARGS.target_pos],
            "camera_config_source": "simulation/badminton_scene/scene_layout.py CAMERA (TEMP engineering values)",
            "camera_resolution_wh": [1280, 720],
            "camera_data_types": ["rgb", "distance_to_image_plane"],
            "model_imgsz": int(ARGS.imgsz),
            "confidence_threshold": float(ARGS.conf),
            "target_class_id": class_id,
            "command": [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
            "artifacts_sha256": artifacts,
        }
        write_json(out / "manifest.json", manifest)
        print(json.dumps({
            "status": overall,
            "output": str(out),
            "checkpoint_sha256": detector.checkpoint_sha256,
            "detections_by_camera": detections_by_camera,
            "valid_rgbd_by_camera": valid3d_by_camera,
        }, indent=2), flush=True)
        return 0 if overall.startswith("PASS_") else 3

    except Exception as exc:
        failure = {
            "status": "FAIL_RUNTIME",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "git_head": git_head(),
            "checkpoint_path": str(checkpoint),
        }
        write_json(out / "failure.json", failure)
        if not (out / "metrics.json").exists():
            write_json(out / "metrics.json", {
                "status": "FAIL_RUNTIME",
                "camera_sensor_status": "UNKNOWN",
                "detector_execution_status": "UNKNOWN",
                "three_d_accuracy_status": "UNKNOWN",
                "failure": str(exc),
            })
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
