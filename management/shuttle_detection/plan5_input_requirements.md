# Plan 5 input requirements — what the detector hard requirements need, and what is missing

Spec: `docs/superpowers/specs/05_SYSTEM_REQUIREMENT_MAPPING_SPEC.md`
Status: **REQUIREMENT_PARTIAL** — not blocked by implementation, blocked by facts only the
human partner can supply. Spec 05 forbids inventing any of these numbers, so they cannot be
filled in by the agent and marked as derived.

## 1. What Plan 5 must output (spec 05 section 6)

| Requirement | Derived from | Have the inputs? |
|---|---|---|
| `minimum_required_target_px` | shuttle size + camera intrinsics + resolution + max working distance + a justified safety margin | **NO** |
| `required_recall` | downstream tolerable measurement gap | **NO** |
| `max_allowed_miss_streak_frames` / `_ms` | gap budget / frame period | **NO** |
| `max_reacquisition_time_ms` | downstream tolerable gap | **NO** |
| `required_precision` | downstream filtering capability | **NO** |
| `confidence_operating_point` | miss/false-positive cost trade-off | **NO** |
| `top_k` | downstream filtering + compute budget | **NO** |
| `max_detection_latency_ms` | whole-chain latency budget | **NO** |
| `max_vram_mb` | deployment GPU budget | **NO** |

## 2. Input inventory — what exists and what does not

### Available (measured or configured, not guessed)

| Input | Where | Value |
|---|---|---|
| Shuttlecock geometry | `configs/shuttlecock.yaml` | feather length 0.066 m, cork dia 0.0265 m, skirt tip dia 0.0618 m |
| Court dimensions | `configs/court.yaml` | 13.40 m x 6.10 m (singles 5.18 m) |
| Detection latency measured so far | Plan 3 evaluation | inference 51.7 ms/image at imgsz 640 on the A6000 (single-image, warm) |
| Peak training VRAM | `run_metadata.json` | 4595 MB at batch 16 / imgsz 640 |

### MISSING — required, and only the human partner can provide them

| Input | Why Plan 5 needs it | Current state |
|---|---|---|
| **Camera intrinsics** (fx, fy, cx, cy, distortion) | `minimum_required_target_px` is a pure pinhole projection; without fx it cannot be computed at all | `CameraIntrinsics` exists in `src/badminton_brain/perception/stereo_geometry.py` but its parameters are marked `REQUIRES_CALIBRATION` (TEMP) |
| **Image resolution** | pixel size depends on the delivered frame size, which the detector may also resize to | `docs/hardware_interface.md` says 帧率 / 分辨率 = 待填 |
| **Stereo baseline** | the left/right split of the acceptance criteria needs it; also feeds the working-distance geometry | `docs/hardware_interface.md` stereo section = 待填 |
| **Max effective working distance** | the far end of the workspace sets the smallest apparent target size | not defined anywhere |
| **Whole-chain latency budget** | `max_detection_latency_ms` is a slice of it, not a standalone number | not defined |
| **Deployment GPU** | `max_vram_mb` | training ran on an A6000 that also hosts another user's vLLM; the deployment target is unspecified |
| **Downstream tolerable measurement gap** | drives recall, miss-streak and reacquisition thresholds | not defined |

## 3. What the agent CAN do meanwhile, and did

Spec 05 section 7 says incomplete parameters give `REQUIREMENT_PARTIAL`, and the master spec
section 3.15 says the model stays at `BASELINE_MEASURED` until this mapping is done. So the correct
behaviour is to keep measuring capability and NOT to invent the requirements. Completed so far:

- frozen-set capability by size bucket, with before/after (Plan 3)
- real-image capability, which found the large-target gap (Plan 3/4 boundary)
- controlled single-variable sweeps in progress, which will add pose/position/blur/occlusion curves

## 4. The one number that can be partially derived today, and why it is not

`minimum_required_target_px` needs `fx`. The repository holds no calibrated `fx`; the only camera
numbers in the tree are TEMP placeholders whose own status enum is `REQUIRES_CALIBRATION`. Filling
a plausible-looking fx in would produce a requirement that LOOKS derived and is not, which is
exactly what spec 05 section 6 forbids. The value therefore stays explicit as
`REQUIREMENT_DERIVED_LATER` until real intrinsics exist.
