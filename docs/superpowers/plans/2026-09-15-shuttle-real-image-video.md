# Shuttle Real Image & Video Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Global Constraints:**
- Detection task is `nc=1`, class `shuttlecock`.
- Do not add P2, temporal detection, or a new detector architecture before the Level-3 gate in `07_THREE_LEVEL_ADJUSTMENT_SPEC.md`.
- `FIXED_CORE_TEST` and `CHALLENGE_TEST` samples never enter training directly.
- Left and right cameras share weights by default and are evaluated separately.
- Report standard detection metrics and project task metrics together.
- On remote `jxxy`, do not modify Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / driver, do not stop the existing vLLM process, and use `/home/T7/dgut/robot_sim/` for project data/caches.
- Every experiment records code commit, data version, config, seed, model weight identity, and evaluation outputs.
- A change is kept only after the same regression protocol shows benefit; otherwise revert it.

---
**Goal:** Measure real-image, left/right camera, and real-video performance including temporal miss streaks and reacquisition.

**Architecture:** Extend the common evaluator with frame-order-aware aggregation. Keep frame-level matching identical to the controlled evaluator so synthetic and real results remain comparable.

**Tech Stack:** Python, pandas, NumPy, OpenCV/ffmpeg only for frame extraction if needed, pytest.

**Spec:** `docs/superpowers/specs/04_REAL_IMAGE_VIDEO_SPEC.md`

### Task 1: Implement temporal metrics

**Files:**
- Create: `src/perception/shuttle_detection/temporal_metrics.py`
- Test: `tests/perception/shuttle_detection/test_temporal_metrics.py`

**Interfaces:**
- Produces: `compute_temporal_metrics(detected: list[bool], fps: float) -> dict`.

- [ ] **Step 1: Write failing test**

```python
def test_longest_miss_streak_and_reacquisition():
    from src.perception.shuttle_detection.temporal_metrics import compute_temporal_metrics
    m = compute_temporal_metrics([True, False, False, True, True], fps=100.0)
    assert m["longest_miss_streak_frames"] == 2
    assert m["longest_miss_streak_ms"] == 20.0
```

- [ ] **Step 2: Run**

Expected: FAIL.

- [ ] **Step 3: Implement deterministic streak analysis**

Also output:
`mean_miss_streak, reacquisition_frames_mean, reacquisition_ms_mean, frame_recall`.

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_temporal_metrics.py -v
git add src/perception/shuttle_detection/temporal_metrics.py tests/perception/shuttle_detection/test_temporal_metrics.py
git commit -m "feat: add shuttle video continuity metrics"
```

### Task 2: Add real-image grouping

**Files:**
- Create: `scripts/shuttle_detection/evaluate_real_images.py`
- Modify: `src/perception/shuttle_detection/evaluate.py`
- Test: `tests/perception/shuttle_detection/test_real_image_grouping.py`

- [ ] **Step 1: Write test**

```python
def test_left_right_are_reported_separately():
    import pandas as pd
    from src.perception.shuttle_detection.evaluate import summarize_by_camera
    df = pd.DataFrame({"camera_id": ["left", "right"], "hit": [1, 0]})
    out = summarize_by_camera(df)
    assert set(out["camera_id"]) == {"left", "right"}
```

- [ ] **Step 2: Implement `summarize_by_camera` and CLI**

- [ ] **Step 3: Run**

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/shuttle_detection/evaluate_real_images.py src/perception/shuttle_detection/evaluate.py tests/perception/shuttle_detection/test_real_image_grouping.py
git commit -m "feat: evaluate shuttle detector on real images"
```

### Task 3: Add ordered real-video evaluator

**Files:**
- Create: `scripts/shuttle_detection/evaluate_real_video.py`
- Test: `tests/perception/shuttle_detection/test_video_evaluator.py`

**Interfaces:**
- Manifest includes `video_id,frame_index,timestamp_s,camera_id,fps`.

- [ ] **Step 1: Write a test rejecting duplicate frame indices per video/camera**

- [ ] **Step 2: Implement sorting, duplicate rejection, frame matching, and temporal aggregation**

- [ ] **Step 3: Run**

Run: `pytest tests/perception/shuttle_detection/test_video_evaluator.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/shuttle_detection/evaluate_real_video.py tests/perception/shuttle_detection/test_video_evaluator.py
git commit -m "feat: evaluate shuttle detector on real video"
```

### Task 4: Run real-domain evaluation

**Files:**
- Output: `outputs/shuttle_detection/capability/real_images/`
- Output: `outputs/shuttle_detection/capability/real_video/`

- [ ] **Step 1: Run real images**

```bash
python scripts/shuttle_detection/evaluate_real_images.py \
  --weights outputs/shuttle_detection/training/baseline_synthetic_real/weights/best.pt \
  --manifest manifest_fixed_core_real_images.csv \
  --out outputs/shuttle_detection/capability/real_images
```

- [ ] **Step 2: Run real video**

```bash
python scripts/shuttle_detection/evaluate_real_video.py \
  --weights outputs/shuttle_detection/training/baseline_synthetic_real/weights/best.pt \
  --manifest manifest_fixed_core_real_video.csv \
  --out outputs/shuttle_detection/capability/real_video
```

- [ ] **Step 3: Write `synthetic_vs_real.csv` using matched size buckets**

- [ ] **Step 4: Write `BASELINE_MEASURED.json`**

- [ ] **Step 5: Commit reports**

```bash
git add outputs/shuttle_detection/capability
git commit -m "exp: measure real shuttle detection capability"
```
