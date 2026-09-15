# Shuttle Failure Diagnosis & Data Feedback Implementation Plan

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
**Goal:** Turn every failed acceptance condition into a structured failure record, evidence-based root-cause hypothesis, and safe data-feedback action.

**Architecture:** Failures are stored as immutable records. Fixed/challenge test samples are references only; the feedback module emits collection/generation requests for new training data instead of copying test samples into training.

**Tech Stack:** Python, pandas, JSON, pytest.

**Spec:** `docs/superpowers/specs/06_FAILURE_DIAGNOSIS_FEEDBACK_SPEC.md`

### Task 1: Define failure taxonomy

**Files:**
- Create: `src/perception/shuttle_detection/failure_feedback.py`
- Test: `tests/perception/shuttle_detection/test_failure_feedback.py`

- [ ] **Step 1: Write failing enum test**

```python
def test_failure_taxonomy_contains_required_classes():
    from src.perception.shuttle_detection.failure_feedback import FailureType
    assert FailureType.SMALL_TARGET.value == "SMALL_TARGET"
    assert FailureType.TEMPORAL_INSTABILITY.value == "TEMPORAL_INSTABILITY"
    assert FailureType.COMPUTE_LIMIT.value == "COMPUTE_LIMIT"
```

- [ ] **Step 2: Implement enum and immutable `FailureRecord`**

Include all failure classes from the Spec plus `UNKNOWN`.

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_failure_feedback.py -v
git add src/perception/shuttle_detection/failure_feedback.py tests/perception/shuttle_detection/test_failure_feedback.py
git commit -m "feat: add shuttle failure taxonomy"
```

### Task 2: Implement evidence-based escalation recommendation

**Files:**
- Modify: `src/perception/shuttle_detection/failure_feedback.py`
- Test: `tests/perception/shuttle_detection/test_failure_diagnosis.py`

- [ ] **Step 1: Write test**

```python
def test_sparse_small_target_data_does_not_jump_to_level3():
    from src.perception.shuttle_detection.failure_feedback import recommend_level
    assert recommend_level(
        data_coverage_ok=False,
        level1_retest_failed=False,
        level2_retest_failed=False,
        structural_evidence=False,
    ) == 1
```

- [ ] **Step 2: Implement gate logic**

Level 3 is allowed only when Level-1 and Level-2 retests failed and structural evidence exists.

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_failure_diagnosis.py -v
git add src/perception/shuttle_detection/failure_feedback.py tests/perception/shuttle_detection/test_failure_diagnosis.py
git commit -m "feat: enforce shuttle adjustment escalation evidence"
```

### Task 3: Generate safe data-feedback requests

**Files:**
- Create: `scripts/shuttle_detection/build_failure_feedback.py`
- Test: `tests/perception/shuttle_detection/test_feedback_requests.py`

- [ ] **Step 1: Test fixed-test samples are referenced but never copied into training requests**

- [ ] **Step 2: Implement request fields**

`failure_type,target_size_range,pose,blur,background,source_preference,requested_count,reason`.

- [ ] **Step 3: Write `failure_records.csv` and `data_collection_requests.csv`**

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_feedback_requests.py -v
git add scripts/shuttle_detection/build_failure_feedback.py tests/perception/shuttle_detection/test_feedback_requests.py
git commit -m "feat: generate safe shuttle data feedback requests"
```

### Task 4: Add hard-negative mining report

**Files:**
- Create: `scripts/shuttle_detection/mine_hard_negatives.py`
- Test: `tests/perception/shuttle_detection/test_hard_negative_mining.py`

- [ ] **Step 1: Test ranking by false-positive confidence**

- [ ] **Step 2: Implement report-only mining**

Do not mutate training manifests automatically.

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_hard_negative_mining.py -v
git add scripts/shuttle_detection/mine_hard_negatives.py tests/perception/shuttle_detection/test_hard_negative_mining.py
git commit -m "feat: report shuttle hard negatives"
```
