# Shuttle Final Evaluation Implementation Plan

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
**Goal:** Run an immutable, independent acceptance evaluation of the final candidate across controlled, real-image, real-video, left/right camera, and deployment-performance requirements.

**Architecture:** Create a release-candidate manifest that hashes every model/config/data input before evaluation. The evaluator runs all required suites without allowing parameter mutation and produces a strict AND-gate acceptance report.

**Tech Stack:** Python, hashlib, JSON, pandas, pytest.

**Spec:** `docs/superpowers/specs/08_FINAL_EVALUATION_SPEC.md`

### Task 1: Create release-candidate manifest

**Files:**
- Create: `src/perception/shuttle_detection/release_candidate.py`
- Test: `tests/perception/shuttle_detection/test_release_candidate.py`

- [ ] **Step 1: Write hash-stability test**

- [ ] **Step 2: Implement SHA-256 file hashing and canonical JSON config hashing**

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_release_candidate.py -v
git add src/perception/shuttle_detection/release_candidate.py tests/perception/shuttle_detection/test_release_candidate.py
git commit -m "feat: freeze shuttle release candidate inputs"
```

### Task 2: Add end-to-end latency benchmark

**Files:**
- Create: `scripts/shuttle_detection/benchmark_detector.py`
- Test: `tests/perception/shuttle_detection/test_latency_summary.py`

- [ ] **Step 1: Write percentile test using deterministic timings**

- [ ] **Step 2: Implement warm-up plus full-path timing**

Measure:
`image_received -> preprocessing -> inference -> postprocess -> result_ready`.

- [ ] **Step 3: Produce median, P95, P99, throughput, peak VRAM**

- [ ] **Step 4: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_latency_summary.py -v
git add scripts/shuttle_detection/benchmark_detector.py tests/perception/shuttle_detection/test_latency_summary.py
git commit -m "feat: benchmark end-to-end shuttle detection latency"
```

### Task 3: Implement strict final gate

**Files:**
- Create: `src/perception/shuttle_detection/final_gate.py`
- Test: `tests/perception/shuttle_detection/test_final_gate.py`

- [ ] **Step 1: Write strict-AND test**

```python
def test_any_hard_failure_rejects_candidate():
    from src.perception.shuttle_detection.final_gate import final_status
    checks = {"recall": True, "real_video": False, "latency": True}
    assert final_status(checks) == "FINAL_REJECTED"
```

- [ ] **Step 2: Implement `all()` semantics plus explicit failed-check list**

- [ ] **Step 3: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_final_gate.py -v
git add src/perception/shuttle_detection/final_gate.py tests/perception/shuttle_detection/test_final_gate.py
git commit -m "feat: add strict shuttle final acceptance gate"
```

### Task 4: Add final evaluation orchestrator

**Files:**
- Create: `scripts/shuttle_detection/final_evaluate.py`

- [ ] **Step 1: Validate candidate hashes before testing**

- [ ] **Step 2: Run FIXED_CORE_TEST controlled, real-image, and real-video suites without changing parameters**

- [ ] **Step 3: Run current CHALLENGE_TEST snapshot**

- [ ] **Step 4: Run deployment latency benchmark**

- [ ] **Step 5: Write baseline-vs-final table with both improvements and regressions**

- [ ] **Step 6: Run strict final gate**

Rejected candidates generate a Failure Record seed for Spec 06.

- [ ] **Step 7: Commit report**

```bash
git add outputs/shuttle_detection/final_evaluation
git commit -m "test: run shuttle final detector acceptance"
```
