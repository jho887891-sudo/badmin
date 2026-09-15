# Shuttle Detector Freeze & Handoff Implementation Plan

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
**Goal:** Freeze the accepted detector as a reproducible release including weights, inference settings, dataset identities, capability report, regression commands, and downstream interface contract.

**Architecture:** Build a release packager that refuses to operate unless final evaluation is `FINAL_ACCEPTED`. The package contains hashes and immutable metadata so future versions can be compared or rolled back.

**Tech Stack:** Python, JSON, hashlib, archive packaging, pytest.

**Spec:** `docs/superpowers/specs/09_FREEZE_HANDOFF_SPEC.md`

### Task 1: Define downstream output contract

**Files:**
- Create: `src/perception/shuttle_detection/interface.py`
- Test: `tests/perception/shuttle_detection/test_interface.py`

- [ ] **Step 1: Write test**

```python
def test_empty_detection_is_not_zero_bbox():
    from src.perception.shuttle_detection.interface import DetectionFrameResult
    r = DetectionFrameResult(timestamp=1.0, camera_id="left", candidates=[], valid=True)
    assert r.candidates == []
```

- [ ] **Step 2: Implement immutable dataclasses**

Candidate:
`candidate_rank, bbox_xyxy, confidence`.

Frame:
`timestamp, camera_id, candidates, valid`.

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_interface.py -v
git add src/perception/shuttle_detection/interface.py tests/perception/shuttle_detection/test_interface.py
git commit -m "feat: freeze shuttle detector output contract"
```

### Task 2: Implement release identity and packager

**Files:**
- Create: `src/perception/shuttle_detection/release.py`
- Create: `scripts/shuttle_detection/freeze_release.py`
- Test: `tests/perception/shuttle_detection/test_release_packager.py`

- [ ] **Step 1: Test deterministic release ID**

- [ ] **Step 2: Implement canonical composition**

`code commit + weight hash + model config hash + inference config hash + dataset version + evaluation report hash`.

- [ ] **Step 3: Refuse packaging unless final status is `FINAL_ACCEPTED`**

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_release_packager.py -v
git add src/perception/shuttle_detection/release.py scripts/shuttle_detection/freeze_release.py tests/perception/shuttle_detection/test_release_packager.py
git commit -m "feat: package accepted shuttle detector release"
```

### Task 3: Generate capability/limitations report

**Files:**
- Create: `scripts/shuttle_detection/build_capability_report.py`
- Output: `outputs/shuttle_detection/release/<release_id>/CAPABILITY.md`

- [ ] **Step 1: Generate report directly from measured outputs**

Required sections:
stable size range, degradation range, unsupported range, hardest poses, hardest video conditions, left/right results, longest miss streak, end-to-end latency, peak VRAM, known limitations.

- [ ] **Step 2: Reject missing source metrics instead of inventing values**

- [ ] **Step 3: Commit generator**

```bash
git add scripts/shuttle_detection/build_capability_report.py
git commit -m "feat: generate shuttle detector capability contract"
```

### Task 4: Freeze regression command and tolerances

**Files:**
- Create: `configs/shuttle_detection/regression.yaml`
- Create: `scripts/shuttle_detection/run_regression.py`
- Test: `tests/perception/shuttle_detection/test_regression_gate.py`

- [ ] **Step 1: Encode accepted metrics and allowed tolerances from final evaluation**

- [ ] **Step 2: Implement regression runner using the same evaluator versions**

- [ ] **Step 3: Test an out-of-tolerance metric fails**

- [ ] **Step 4: Commit**

```bash
pytest tests/perception/shuttle_detection/test_regression_gate.py -v
git add configs/shuttle_detection/regression.yaml scripts/shuttle_detection/run_regression.py tests/perception/shuttle_detection/test_regression_gate.py
git commit -m "test: freeze shuttle detector regression gate"
```

### Task 5: Produce release package and handoff

**Files:**
- Output: `outputs/shuttle_detection/release/<release_id>/`

- [ ] **Step 1: Confirm final evaluation still says `FINAL_ACCEPTED`**

- [ ] **Step 2: Run**

```bash
python scripts/shuttle_detection/freeze_release.py \
  --final-evaluation outputs/shuttle_detection/final_evaluation/final_evaluation.json \
  --out outputs/shuttle_detection/release
```

- [ ] **Step 3: Verify package contents**

Weight/hash, architecture/config, inference config, dataset identities, final evaluation, capability report, interface version, regression config, code commit, release ID.

- [ ] **Step 4: Run regression from the package**

- [ ] **Step 5: Mark `FROZEN` and commit release metadata**

```bash
git add outputs/shuttle_detection/release
git commit -m "release: freeze accepted shuttle detector"
```
