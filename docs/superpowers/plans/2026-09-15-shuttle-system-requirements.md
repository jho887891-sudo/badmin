# Shuttle Detection System Requirement Mapping Implementation Plan

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
**Goal:** Derive formal detector acceptance requirements from camera geometry, downstream measurement-gap tolerance, false-candidate tolerance, and end-to-end latency budget.

**Architecture:** Add a pure calculation module that consumes explicit system parameters and emits a versioned requirement JSON. Missing source parameters produce `REQUIREMENT_PARTIAL`; no arbitrary numeric defaults are inserted.

**Tech Stack:** Python, NumPy, dataclasses, YAML/JSON, pytest.

**Spec:** `docs/superpowers/specs/05_SYSTEM_REQUIREMENT_MAPPING_SPEC.md`

### Task 1: Implement geometric minimum-pixel calculation

**Files:**
- Create: `src/perception/shuttle_detection/requirements.py`
- Test: `tests/perception/shuttle_detection/test_requirements_geometry.py`

- [ ] **Step 1: Write failing geometry test**

```python
def test_projected_size_uses_pinhole_relation():
    from src.perception.shuttle_detection.requirements import projected_size_px
    assert projected_size_px(0.07925, 1000.0, 5.0) == 15.85
```

- [ ] **Step 2: Run**

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def projected_size_px(real_size_m: float, focal_length_px: float, distance_m: float) -> float:
    if real_size_m <= 0 or focal_length_px <= 0 or distance_m <= 0:
        raise ValueError("all geometric inputs must be positive")
    return real_size_m * focal_length_px / distance_m
```

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_requirements_geometry.py -v
git add src/perception/shuttle_detection/requirements.py tests/perception/shuttle_detection/test_requirements_geometry.py
git commit -m "feat: derive shuttle projected size requirement"
```

### Task 2: Implement miss-streak and latency-budget derivation

**Files:**
- Modify: `src/perception/shuttle_detection/requirements.py`
- Test: `tests/perception/shuttle_detection/test_requirements_timing.py`

- [ ] **Step 1: Write tests**

```python
def test_max_miss_frames():
    from src.perception.shuttle_detection.requirements import max_miss_frames
    assert max_miss_frames(30.0, fps=100.0) == 3
```

- [ ] **Step 2: Implement exact frame conversion and nonnegative budget checks**

- [ ] **Step 3: Run**

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/perception/shuttle_detection/requirements.py tests/perception/shuttle_detection/test_requirements_timing.py
git commit -m "feat: derive shuttle temporal and latency requirements"
```

### Task 3: Add versioned requirement generator

**Files:**
- Create: `configs/shuttle_detection/system_requirements.yaml`
- Create: `scripts/shuttle_detection/derive_requirements.py`
- Test: `tests/perception/shuttle_detection/test_requirement_status.py`

- [ ] **Step 1: Create config with explicit unknown source fields**

```yaml
camera:
  focal_length_px: null
  max_effective_distance_m: null
shuttle:
  reference_size_m: 0.07925
downstream:
  max_measurement_gap_ms: null
  max_false_candidates_per_frame: null
latency:
  total_closed_loop_budget_ms: null
  reserved_non_detection_ms: null
deployment:
  max_vram_mb: null
```

`null` means unknown and the script may not invent a value.

- [ ] **Step 2: Write test**

Incomplete inputs must produce `"status": "REQUIREMENT_PARTIAL"`.

- [ ] **Step 3: Implement generator**

When all required inputs exist, output `SYSTEM_REQUIREMENTS_MAPPED` and the derived hard requirements.

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_requirement_status.py -v
git add configs/shuttle_detection/system_requirements.yaml scripts/shuttle_detection/derive_requirements.py tests/perception/shuttle_detection/test_requirement_status.py
git commit -m "feat: generate shuttle detector system requirements"
```

### Task 4: Compare measured capability with requirements

**Files:**
- Create: `scripts/shuttle_detection/check_acceptance_gap.py`
- Test: `tests/perception/shuttle_detection/test_acceptance_gap.py`

- [ ] **Step 1: Write a test where one hard metric fails**

Overall status must be `NEEDS_ADJUSTMENT`.

- [ ] **Step 2: Implement strict AND gate**

No averaging across hard requirements.

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_acceptance_gap.py -v
git add scripts/shuttle_detection/check_acceptance_gap.py tests/perception/shuttle_detection/test_acceptance_gap.py
git commit -m "feat: compare shuttle capability with system requirements"
```
