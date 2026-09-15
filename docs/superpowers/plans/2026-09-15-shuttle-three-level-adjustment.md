# Shuttle Three-Level Adjustment Implementation Plan

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
**Goal:** Make local, mid-level, and architecture-level detector adjustments reproducible, gated, and directly comparable to the frozen baseline.

**Architecture:** Implement an experiment registry and comparison harness first. Level 1 and 2 experiments reuse the same detector architecture; Level 3 permits P2/new detector/two-stage/temporal experiments only when evidence files prove previous gates failed.

**Tech Stack:** Python, YAML, Ultralytics, pandas, pytest.

**Spec:** `docs/superpowers/specs/07_THREE_LEVEL_ADJUSTMENT_SPEC.md`

### Task 1: Add adjustment gate validator

**Files:**
- Create: `src/perception/shuttle_detection/adjustment.py`
- Test: `tests/perception/shuttle_detection/test_adjustment_gates.py`

- [ ] **Step 1: Write failing gate test**

```python
import pytest

def test_cannot_enter_level3_without_level2_failure():
    from src.perception.shuttle_detection.adjustment import validate_level_transition
    with pytest.raises(ValueError):
        validate_level_transition(
            3,
            {"level1_failed": True, "level2_failed": False, "structural_evidence": True},
        )
```

- [ ] **Step 2: Implement strict Level-2 and Level-3 gates**

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_adjustment_gates.py -v
git add src/perception/shuttle_detection/adjustment.py tests/perception/shuttle_detection/test_adjustment_gates.py
git commit -m "feat: enforce shuttle adjustment levels"
```

### Task 2: Implement experiment registry

**Files:**
- Create: `scripts/shuttle_detection/register_adjustment_experiment.py`
- Test: `tests/perception/shuttle_detection/test_experiment_registry.py`

- [ ] **Step 1: Test registry captures baseline identity, changed variables, seed, data version, model config, and retest scope**

- [ ] **Step 2: Implement canonical JSON registry records**

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_experiment_registry.py -v
git add scripts/shuttle_detection/register_adjustment_experiment.py tests/perception/shuttle_detection/test_experiment_registry.py
git commit -m "feat: register shuttle adjustment experiments"
```

### Task 3: Implement Level-1 experiment runner

**Files:**
- Create: `configs/shuttle_detection/adjustments/level1.yaml`
- Create: `scripts/shuttle_detection/run_level1_adjustment.py`

- [ ] **Step 1: Allow only local knobs**

Allowed:
`source_sampling, size_bucket_sampling, pose_sampling, hard_negative_sampling, learning_rate, batch, epochs, augmentation, confidence_threshold, iou_threshold, top_k`.

- [ ] **Step 2: Reject architecture keys**

Reject `p2`, `temporal`, `detector_family`.

- [ ] **Step 3: Train from a real Failure Record and automatically rerun fixed-core evaluation**

- [ ] **Step 4: Keep only if fixed-core results improve without violating hard requirements**

- [ ] **Step 5: Commit evidence**

```bash
git add configs/shuttle_detection/adjustments/level1.yaml outputs/shuttle_detection/adjustments
git commit -m "exp: run level1 shuttle adjustment"
```

### Task 4: Implement Level-2 experiment runner

**Files:**
- Create: `configs/shuttle_detection/adjustments/level2.yaml`
- Create: `scripts/shuttle_detection/run_level2_adjustment.py`

- [ ] **Step 1: Support input-resolution comparison at 640, 960, 1280**

- [ ] **Step 2: Support ROI only when an evidence file proves a reliable ROI source exists**

- [ ] **Step 3: Support weighted sampling, curriculum ordering, and synthetic/balanced/real-heavy stages**

- [ ] **Step 4: Require capability + temporal + latency + VRAM retest**

- [ ] **Step 5: Commit evidence**

```bash
git add configs/shuttle_detection/adjustments/level2.yaml outputs/shuttle_detection/adjustments
git commit -m "exp: run level2 shuttle adjustment"
```

### Task 5: Implement Level-3 P2 experiment

**Files:**
- Create: `configs/shuttle_detection/adjustments/level3_p2.yaml`
- Create: `scripts/shuttle_detection/run_level3_p2.py`
- Test: `tests/perception/shuttle_detection/test_p2_fairness.py`

- [ ] **Step 1: Test P2 is rejected without Level-3 evidence**

- [ ] **Step 2: Test baseline and P2 use identical data version, seed, epochs, input resolution, optimizer, and evaluator version**

- [ ] **Step 3: Run the corresponding Ultralytics P2 architecture only after the gate passes**

Record rationale exactly:

```text
增加 P2 检测分支
→ 检测头新增 stride=4 检测尺度
→ 更高分辨率特征直接进入检测头
→ 是否提高极小羽毛球 Recall 由固定测试实测
```

- [ ] **Step 4: Compare Recall, temporal continuity, Precision, latency, and VRAM**

- [ ] **Step 5: Keep or revert using all hard requirements**

### Task 6: Add common Level-3 detector protocol

**Files:**
- Create: `src/perception/shuttle_detection/level3_protocol.py`
- Test: `tests/perception/shuttle_detection/test_level3_protocol.py`

- [ ] **Step 1: Define protocol**

```python
from typing import Protocol

class DetectorProtocol(Protocol):
    def predict(self, frame, timestamp: float) -> list[dict]:
        ...
```

- [ ] **Step 2: Test single-frame and future temporal adapters share the same output contract**

- [ ] **Step 3: Do not implement temporal fusion unless failure type is `TEMPORAL_INSTABILITY` and Level-3 gate is open**

- [ ] **Step 4: Commit**

```bash
git add src/perception/shuttle_detection/level3_protocol.py tests/perception/shuttle_detection/test_level3_protocol.py
git commit -m "feat: define shuttle level3 detector protocol"
```
