# VIS-C2: Isaac Sim RGB-D + existing YOLO probe

Status at authoring time: implementation submitted for review; **Isaac Sim runtime is not yet verified**.

## Purpose

Connect the existing single-environment court scene to two actual Isaac Lab camera sensors, read RGB plus distance-to-image-plane depth, run a caller-selected already-trained Ultralytics-compatible YOLO checkpoint, and produce timestamped 3D measurement candidates.

This is a sensor/inference integration gate only. It does not run the robot, send commands, train YOLO, or claim calibrated 3D accuracy.

## Preconditions

- Run on the Linux host that has the project Isaac Lab environment and court USD asset.
- Supply an existing Ultralytics-compatible checkpoint explicitly. The script never chooses or converts weights silently.
- Recommended first probe candidate from the repository's known release record: `baseline_os` / `shuttle-detector-v2-2026-09-15`, YOLO26s, class 0 `shuttlecock`, SHA256 `61c491206904bba66ee866a8d5b7a45b4b10668c87c07804e2b58e147d74c963` (20,317,957 bytes). The freeze handoff labels it `FROZEN_CANDIDATE`, not deployable; use it only as the existing simulation probe candidate. Locate the actual local `weights/best.pt`, verify that SHA, and stop if the file is absent or mismatched.
- The older ETH YOLOv8 state-dict checkpoint recorded as non-Ultralytics format may not load through the Ultralytics API. If that is the only checkpoint available, stop and add an explicit architecture adapter rather than changing the weights.
- The camera rig values come from scene_layout.CAMERA and are TEMP engineering geometry, not measured hardware extrinsics.
- The output directory must not already exist.

## Run

From the repository root, choose the frozen checkpoint path from the local checkpoint registry and execute:

    scripts/simulation/run_isaac_rgbd_yolo_probe.sh --checkpoint /absolute/path/to/baseline_os/weights/best.pt --expected-checkpoint-sha256 61c491206904bba66ee866a8d5b7a45b4b10668c87c07804e2b58e147d74c963 --out outputs/vision/vis_c2_isaac_rgbd_run_001 --frames 3 --imgsz 1024 --conf 0.25 --class-id 0 --model-device cuda:0

The default visual test fixture is the existing shuttlecock_visual.usd placed at (-0.5, 0.0, 1.25) m. It is rendered into the scene only; its known pose is not passed to the detector or measurement module. This near-side placement avoids using a hidden target behind the net as a false-negative test.

The wrapper refuses to install packages, update environments, kill other processes, overwrite output, train a model, or launch closed-loop robot control.

## Output contract

- images/left_frame_*.png and images/right_frame_*.png: raw camera RGB.
- images/*_overlay.png: detector boxes on the same rendered frame.
- depth/*.npy: raw distance-to-image-plane depth in metres.
- detections.jsonl: per-frame, per-camera boxes, confidence, timestamp, depth patch statistics, 3D point, velocity, covariance, latency, and checkpoint SHA256.
- metrics.json: camera, detector, RGB-D projection status, per-camera counts, latency, and left/right consistency diagnostics.
- manifest.json: code HEAD, checkpoint and visual-asset SHA256, parameters, and SHA256 of output artifacts.
- failure.json: exception details if the runtime fails after output creation.

## Acceptance

Preliminary VIS-C2 PASS requires both cameras to return correctly shaped RGB and depth, finite runtime intrinsics/pose, at least one target detection from each camera, and at least one valid depth-derived 3D measurement from each camera.

A detector miss is not a camera sensor failure. The script records camera status separately and returns a non-zero status if either camera has no valid detection/measurement.

Still UNKNOWN after this gate: 3D point accuracy against independent ground truth, real hardware camera calibration, measured detector noise, UKF prediction quality, contact, hit rate, and closed-loop performance. The depth-derived covariance is explicitly TEMP and must not be presented as calibrated.
