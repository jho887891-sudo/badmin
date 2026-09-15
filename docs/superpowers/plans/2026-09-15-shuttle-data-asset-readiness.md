# Shuttle Detection Data & Asset Readiness Implementation Plan

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
**Goal:** Make shuttlecock assets, labels, manifests, background pool, and dataset splits auditable and safe for training.

**Architecture:** Add a focused audit module under `src/perception/shuttle_detection/` and a CLI under `scripts/shuttle_detection/`. The audit reads manifests rather than embedding dataset paths, produces machine-readable reports, and fails before training if asset consistency, label validity, or split isolation is violated.

**Tech Stack:** Python 3.12, pandas, Pillow/OpenCV if already present, pytest.

**Spec:** `docs/superpowers/specs/01_DATA_ASSET_READINESS_SPEC.md`

### Task 1: Define audit data contracts

**Files:**
- Create: `src/perception/shuttle_detection/__init__.py`
- Create: `src/perception/shuttle_detection/contracts.py`
- Test: `tests/perception/shuttle_detection/test_contracts.py`

**Interfaces:**
- Produces: `DatasetRecord`, `AuditIssue`, `AuditResult`.

- [ ] **Step 1: Write the failing contract test**

```python
from src.perception.shuttle_detection.contracts import DatasetRecord

def test_dataset_record_keeps_source_and_split():
    r = DatasetRecord(
        sample_id="train_00001",
        image_path="images/train_00001.png",
        label_path="labels/train_00001.txt",
        split="train",
        source_type="SYNTHETIC_3D",
        camera_id=None,
    )
    assert r.split == "train"
    assert r.source_type == "SYNTHETIC_3D"
```

- [ ] **Step 2: Run it**

Run: `pytest tests/perception/shuttle_detection/test_contracts.py -v`

Expected: FAIL because `contracts.py` does not exist.

- [ ] **Step 3: Implement the contracts**

```python
from dataclasses import dataclass
from typing import Literal

Split = Literal["train", "val", "fixed_core_test", "challenge_test"]
SourceType = Literal["SYNTHETIC_3D", "REAL_IMAGE", "REAL_VIDEO", "NEGATIVE"]

@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    image_path: str
    label_path: str | None
    split: Split
    source_type: SourceType
    camera_id: str | None

@dataclass(frozen=True)
class AuditIssue:
    sample_id: str
    code: str
    severity: Literal["ERROR", "WARNING"]
    detail: str

@dataclass(frozen=True)
class AuditResult:
    passed: bool
    issue_count: int
    error_count: int
```

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/perception/shuttle_detection/test_contracts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perception/shuttle_detection tests/perception/shuttle_detection/test_contracts.py
git commit -m "feat: add shuttle dataset audit contracts"
```

### Task 2: Implement manifest and label validation

**Files:**
- Create: `src/perception/shuttle_detection/dataset_audit.py`
- Test: `tests/perception/shuttle_detection/test_dataset_audit.py`

**Interfaces:**
- Produces: `validate_yolo_label(sample_id: str, path: Path, expected_class: int = 0) -> list[AuditIssue]`.

- [ ] **Step 1: Write failing tests for invalid YOLO boxes**

```python
from pathlib import Path
from src.perception.shuttle_detection.dataset_audit import validate_yolo_label

def test_rejects_out_of_range_bbox(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")
    issues = validate_yolo_label("s1", p, expected_class=0)
    assert any(i.code == "BBOX_OUT_OF_RANGE" for i in issues)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/perception/shuttle_detection/test_dataset_audit.py -v`

Expected: FAIL because the function is missing.

- [ ] **Step 3: Implement strict label checks**

```python
from pathlib import Path
from .contracts import AuditIssue

def validate_yolo_label(sample_id: str, path: Path, expected_class: int = 0) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    if not path.exists():
        return [AuditIssue(sample_id, "LABEL_MISSING", "ERROR", str(path))]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            issues.append(AuditIssue(sample_id, "LABEL_FORMAT", "ERROR", f"line={line_no}"))
            continue
        cls, xc, yc, w, h = map(float, parts)
        if int(cls) != expected_class:
            issues.append(AuditIssue(sample_id, "WRONG_CLASS", "ERROR", f"class={cls}"))
        if not all(0.0 <= x <= 1.0 for x in (xc, yc, w, h)) or w <= 0 or h <= 0:
            issues.append(AuditIssue(sample_id, "BBOX_OUT_OF_RANGE", "ERROR", f"line={line_no}"))
    return issues
```

- [ ] **Step 4: Add manifest checks for missing image, duplicate `sample_id`, and illegal split/source values, then run tests**

Run: `pytest tests/perception/shuttle_detection/test_dataset_audit.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perception/shuttle_detection/dataset_audit.py tests/perception/shuttle_detection/test_dataset_audit.py
git commit -m "feat: validate shuttle manifests and labels"
```

### Task 3: Implement split leakage detection and distribution reports

**Files:**
- Modify: `src/perception/shuttle_detection/dataset_audit.py`
- Test: `tests/perception/shuttle_detection/test_dataset_leakage.py`

**Interfaces:**
- Produces: exact-duplicate hash checks and source-group leakage checks.

- [ ] **Step 1: Write a failing cross-split duplicate test**

```python
def test_same_content_cannot_cross_train_and_fixed_test():
    from src.perception.shuttle_detection.dataset_audit import find_cross_split_duplicates
    rows = [
        ("a", "train", "abc123"),
        ("b", "fixed_core_test", "abc123"),
    ]
    issues = find_cross_split_duplicates(rows)
    assert len(issues) == 1
    assert issues[0].code == "CROSS_SPLIT_DUPLICATE"
```

- [ ] **Step 2: Run it**

Run: `pytest tests/perception/shuttle_detection/test_dataset_leakage.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement duplicate/leakage detection and distribution aggregation**

Create:

```python
def find_cross_split_duplicates(rows: list[tuple[str, str, str]]) -> list[AuditIssue]:
    ...

def summarize_distribution(df):
    ...
```

Distribution output must include counts by:
`split, source_type, camera_id, size_bucket, blur_bucket, pose_bucket, is_negative`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/perception/shuttle_detection/test_dataset_leakage.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perception/shuttle_detection/dataset_audit.py tests/perception/shuttle_detection/test_dataset_leakage.py
git commit -m "feat: detect shuttle dataset leakage"
```

### Task 4: Add audit CLI and reports

**Files:**
- Create: `scripts/shuttle_detection/audit_dataset.py`
- Test: `tests/perception/shuttle_detection/test_audit_cli.py`

**Interfaces:**
- CLI: `python scripts/shuttle_detection/audit_dataset.py --manifest <csv> --dataset-root <dir> --out <dir>`.

- [ ] **Step 1: Write CLI test**

```python
def test_cli_returns_nonzero_on_error(tmp_path):
    from scripts.shuttle_detection.audit_dataset import main
    rc = main([
        "--manifest", str(tmp_path / "missing.csv"),
        "--dataset-root", str(tmp_path),
        "--out", str(tmp_path / "out"),
    ])
    assert rc != 0
```

- [ ] **Step 2: Run it**

Run: `pytest tests/perception/shuttle_detection/test_audit_cli.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement CLI**

The CLI writes:
- `dataset_inventory.csv`
- `dataset_cleaning_report.csv`
- `dataset_distribution_report.csv`
- `dataset_leakage_report.csv`
- `audit_summary.json`

Return `0` only when error count is zero.

- [ ] **Step 4: Run audit tests**

Run: `pytest tests/perception/shuttle_detection/test_*audit*.py tests/perception/shuttle_detection/test_contracts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/shuttle_detection/audit_dataset.py tests/perception/shuttle_detection/test_audit_cli.py
git commit -m "feat: add shuttle dataset audit cli"
```

### Task 5: Finish current background review and remote sync gate

**Files:**
- Create: `management/shuttle_detection/background_review.csv`
- Create: `management/shuttle_detection/asset_readiness_report.md`

- [ ] **Step 1: Review the remaining 31 full-resolution backgrounds**

Required columns:

```text
filename,status,reason,keep_remove,notes
```

Allowed `status`: `PASS`, `FAIL`.

- [ ] **Step 2: Run the audit CLI on the full current manifest**

```bash
python scripts/shuttle_detection/audit_dataset.py \
  --manifest manifest_train.csv \
  --dataset-root "$(git rev-parse --show-toplevel)" \
  --out outputs/shuttle_detection/dataset_audit
```

Expected: exit code 0.

- [ ] **Step 3: Record local commit**

Run: `git rev-parse HEAD`

Write the hash into `asset_readiness_report.md`.

- [ ] **Step 4: Sync to jxxy without touching system environments**

```bash
rsync -a \
  --exclude '.git/' \
  ./ dgut@172.31.68.251:/home/T7/dgut/robot_sim/
```

Do not use cleanup flags that could delete unrelated remote data.

- [ ] **Step 5: Commit readiness evidence**

```bash
git add management/shuttle_detection outputs/shuttle_detection/dataset_audit
git commit -m "chore: record shuttle asset readiness"
```
