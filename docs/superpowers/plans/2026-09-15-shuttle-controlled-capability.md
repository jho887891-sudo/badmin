# Shuttle Controlled Capability Evaluation Implementation Plan

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
**Goal:** Measure the baseline detector's capability as target size, 3D pose, image position, blur, occlusion, and background change under controlled conditions.

**Architecture:** Centralize matching and metric computation in `metrics.py`, then run controlled manifests through one evaluator. `equivalent_size_px = sqrt(width_px * height_px)` is the primary size scalar; standard detection metrics and task-specific Top-K/coverage/center-error metrics are both produced.

**Tech Stack:** Python, NumPy, pandas, Ultralytics inference, pytest.

**Spec:** `docs/superpowers/specs/03_CONTROLLED_CAPABILITY_SPEC.md`

### Task 1: Implement target-size metrics

**Files:**
- Create: `src/perception/shuttle_detection/metrics.py`
- Test: `tests/perception/shuttle_detection/test_metrics.py`

**Interfaces:**
- Produces: `equivalent_size_px(w, h)`, `size_bucket(eq)`.

- [ ] **Step 1: Write failing tests**

```python
from src.perception.shuttle_detection.metrics import equivalent_size_px, size_bucket

def test_equivalent_size():
    assert equivalent_size_px(4, 9) == 6.0

def test_size_bucket_boundary():
    assert size_bucket(5.0) == "4-6"
    assert size_bucket(8.0) == "8-12"
```

- [ ] **Step 2: Run**

Run: `pytest tests/perception/shuttle_detection/test_metrics.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
import math

def equivalent_size_px(width_px: float, height_px: float) -> float:
    return math.sqrt(width_px * height_px)

def size_bucket(eq: float) -> str:
    if eq < 4: return "<4"
    if eq < 6: return "4-6"
    if eq < 8: return "6-8"
    if eq < 12: return "8-12"
    if eq < 16: return "12-16"
    if eq < 24: return "16-24"
    if eq < 32: return "24-32"
    return ">32"
```

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_metrics.py -v
git add src/perception/shuttle_detection/metrics.py tests/perception/shuttle_detection/test_metrics.py
git commit -m "feat: add shuttle capability size metrics"
```

### Task 2: Implement Top-K task metrics

**Files:**
- Modify: `src/perception/shuttle_detection/metrics.py`
- Test: `tests/perception/shuttle_detection/test_candidate_metrics.py`

**Interfaces:**
- Produces: `match_top_k(gt_box, candidate_boxes, k) -> dict`.

- [ ] **Step 1: Write failing test**

```python
def test_top_k_preserves_true_candidate():
    from src.perception.shuttle_detection.metrics import match_top_k
    gt = (10, 10, 14, 14)
    candidates = [
        {"bbox": (30, 30, 35, 35), "confidence": 0.9},
        {"bbox": (9, 9, 15, 15), "confidence": 0.7},
    ]
    m = match_top_k(gt, candidates, k=2)
    assert m["top1_hit"] is False
    assert m["topk_hit"] is True
```

- [ ] **Step 2: Run**

Expected: FAIL.

- [ ] **Step 3: Implement**

Return:
`top1_hit, topk_hit, gt_covered, center_error_px, best_iou`.

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_candidate_metrics.py -v
git add src/perception/shuttle_detection/metrics.py tests/perception/shuttle_detection/test_candidate_metrics.py
git commit -m "feat: add shuttle top-k task metrics"
```

### Task 3: Add controlled evaluator

**Files:**
- Create: `src/perception/shuttle_detection/evaluate.py`
- Create: `scripts/shuttle_detection/evaluate_controlled.py`
- Test: `tests/perception/shuttle_detection/test_controlled_evaluator.py`

**Interfaces:**
- CLI consumes model weight + controlled manifest.
- Produces per-sample predictions and grouped metrics.

- [ ] **Step 1: Write fake-detector test**

```python
def fake_detector(_image):
    return [{"bbox": (9, 9, 15, 15), "confidence": 0.8}]
```

Assert evaluator outputs `size_bucket`, `topk_hit`, and `center_error_px`.

- [ ] **Step 2: Implement detector injection so unit tests do not require GPU**

- [ ] **Step 3: Keep Ultralytics import inside the CLI adapter**

- [ ] **Step 4: Run**

Run: `pytest tests/perception/shuttle_detection/test_controlled_evaluator.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perception/shuttle_detection/evaluate.py scripts/shuttle_detection/evaluate_controlled.py tests/perception/shuttle_detection/test_controlled_evaluator.py
git commit -m "feat: add controlled shuttle capability evaluator"
```

### Task 4: Run the controlled capability matrix

**Files:**
- Output: `outputs/shuttle_detection/capability/controlled/`

- [ ] **Step 1: Validate manifest fields for size, pose, position, blur, occlusion, background, lighting**

- [ ] **Step 2: Run**

```bash
python scripts/shuttle_detection/evaluate_controlled.py \
  --weights outputs/shuttle_detection/training/baseline_synthetic_real/weights/best.pt \
  --manifest manifest_fixed_core_controlled.csv \
  --out outputs/shuttle_detection/capability/controlled
```

- [ ] **Step 3: Require outputs**

`predictions.csv`, `size_bucket_metrics.csv`, `pose_metrics.csv`, `position_metrics.csv`, `blur_metrics.csv`, `occlusion_metrics.csv`, `summary.json`.

- [ ] **Step 4: Write `CONTROLLED_CAPABILITY_MEASURED.json`**

- [ ] **Step 5: Commit reports**

```bash
git add outputs/shuttle_detection/capability
git commit -m "exp: measure controlled shuttle detection capability"
```
