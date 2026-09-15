# Shuttle Detection Baseline Training Implementation Plan

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
**Goal:** Train reproducible `nc=1` shuttlecock baselines from pretrained weights, first synthetic-only and then synthetic+real.

**Architecture:** Use a thin training wrapper that converts audited manifests into a runtime Ultralytics dataset configuration, records every training parameter, and stores immutable run metadata. The wrapper does not introduce P2 or temporal logic.

**Tech Stack:** Python 3.12, Ultralytics YOLO26, PyTorch, YAML, pytest.

**Spec:** `docs/superpowers/specs/02_BASELINE_TRAINING_SPEC.md`

### Task 1: Add baseline configuration schema

**Files:**
- Create: `configs/shuttle_detection/baseline.yaml`
- Create: `src/perception/shuttle_detection/training_config.py`
- Test: `tests/perception/shuttle_detection/test_training_config.py`

**Interfaces:**
- Produces: `BaselineConfig.load(path)`.

- [ ] **Step 1: Write the failing schema test**

```python
from src.perception.shuttle_detection.training_config import BaselineConfig

def test_baseline_config_requires_single_class(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "model: yolo26s.pt\nnc: 1\nname: shuttlecock\nimgsz: 640\nseed: 17\n",
        encoding="utf-8",
    )
    c = BaselineConfig.load(p)
    assert c.nc == 1
    assert c.name == "shuttlecock"
```

- [ ] **Step 2: Run it**

Run: `pytest tests/perception/shuttle_detection/test_training_config.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement immutable config parsing**

Use a frozen dataclass containing:
`model, nc, name, imgsz, seed, epochs, batch, optimizer, lr0`.

Reject `nc != 1`.

- [ ] **Step 4: Add checked-in baseline config**

```yaml
model: yolo26s.pt
nc: 1
name: shuttlecock
imgsz: 640
seed: 17
epochs: 100
batch: 16
optimizer: auto
lr0: 0.01
```

These are baseline experiment settings, not final acceptance values.

- [ ] **Step 5: Run and commit**

```bash
pytest tests/perception/shuttle_detection/test_training_config.py -v
git add configs/shuttle_detection/baseline.yaml src/perception/shuttle_detection/training_config.py tests/perception/shuttle_detection/test_training_config.py
git commit -m "feat: define shuttle baseline training config"
```

### Task 2: Build runtime dataset configuration from manifests

**Files:**
- Create: `src/perception/shuttle_detection/ultralytics_dataset.py`
- Test: `tests/perception/shuttle_detection/test_ultralytics_dataset.py`

**Interfaces:**
- Produces: `build_dataset_yaml(train_manifest, val_manifest, output_dir) -> Path`.

- [ ] **Step 1: Write failing test**

```python
def test_dataset_yaml_is_single_class(tmp_path):
    from src.perception.shuttle_detection.ultralytics_dataset import write_dataset_yaml
    p = write_dataset_yaml(tmp_path, tmp_path / "train.txt", tmp_path / "val.txt")
    text = p.read_text()
    assert "nc: 1" in text
    assert "shuttlecock" in text
```

- [ ] **Step 2: Run**

Expected: FAIL.

- [ ] **Step 3: Implement manifest-to-image-list conversion and YAML writing**

Fail if any `fixed_core_test` or `challenge_test` row is included.

- [ ] **Step 4: Re-run**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perception/shuttle_detection/ultralytics_dataset.py tests/perception/shuttle_detection/test_ultralytics_dataset.py
git commit -m "feat: build single-class shuttle dataset config"
```

### Task 3: Add reproducible training wrapper

**Files:**
- Create: `scripts/shuttle_detection/train_baseline.py`
- Create: `src/perception/shuttle_detection/run_metadata.py`
- Test: `tests/perception/shuttle_detection/test_run_metadata.py`

**Interfaces:**
- CLI supports `--config`, `--train-manifest`, `--val-manifest`, `--run-name`, `--data-root`.
- Produces run metadata, resolved config, weights, and training logs.

- [ ] **Step 1: Write metadata test**

```python
def test_metadata_contains_commit_and_seed():
    from src.perception.shuttle_detection.run_metadata import make_metadata
    m = make_metadata(seed=17, data_version="v1", model="yolo26s.pt")
    assert m["seed"] == 17
    assert "code_commit" in m
    assert m["model"] == "yolo26s.pt"
```

- [ ] **Step 2: Run**

Expected: FAIL.

- [ ] **Step 3: Implement metadata and training call**

Core call:

```python
from ultralytics import YOLO

model = YOLO(cfg.model)
model.train(
    data=str(dataset_yaml),
    imgsz=cfg.imgsz,
    epochs=cfg.epochs,
    batch=cfg.batch,
    seed=cfg.seed,
    optimizer=cfg.optimizer,
    lr0=cfg.lr0,
    project=str(run_root),
    name=run_name,
)
```

Record GPU name, start/end time, commit, config hash, manifest hashes, and peak VRAM when available.

- [ ] **Step 4: Run tests**

Run: `pytest tests/perception/shuttle_detection/test_run_metadata.py tests/perception/shuttle_detection/test_training_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/shuttle_detection/train_baseline.py src/perception/shuttle_detection/run_metadata.py tests/perception/shuttle_detection/test_run_metadata.py
git commit -m "feat: add reproducible shuttle baseline trainer"
```

### Task 4: Train synthetic-only baseline

**Files:**
- Output: `outputs/shuttle_detection/training/baseline_synthetic_only/`

- [ ] **Step 1: Build an audited synthetic-only train/val manifest pair**

- [ ] **Step 2: Launch**

```bash
python scripts/shuttle_detection/train_baseline.py \
  --config configs/shuttle_detection/baseline.yaml \
  --train-manifest manifest_train_synthetic.csv \
  --val-manifest manifest_val_synthetic.csv \
  --data-root "$(git rev-parse --show-toplevel)" \
  --run-name baseline_synthetic_only
```

- [ ] **Step 3: Verify best/last weights, resolved config, metadata, and metrics exist**

- [ ] **Step 4: Record the run identity**

- [ ] **Step 5: Commit lightweight evidence**

```bash
git add outputs/shuttle_detection/training/baseline_synthetic_only/run_metadata.json
git commit -m "exp: record synthetic-only shuttle baseline"
```

### Task 5: Train synthetic+real baseline

**Files:**
- Output: `outputs/shuttle_detection/training/baseline_synthetic_real/`

- [ ] **Step 1: Build audited mixed manifests and write `source_mix.json`**

- [ ] **Step 2: Launch**

```bash
python scripts/shuttle_detection/train_baseline.py \
  --config configs/shuttle_detection/baseline.yaml \
  --train-manifest manifest_train_mixed.csv \
  --val-manifest manifest_val_mixed.csv \
  --data-root "$(git rev-parse --show-toplevel)" \
  --run-name baseline_synthetic_real
```

- [ ] **Step 3: Compare Precision, Recall, mAP50, and mAP50-95 with synthetic-only**

- [ ] **Step 4: Write `outputs/shuttle_detection/training/BASELINE_TRAINED.json`**

It contains both run identities and the selected baseline candidate.

- [ ] **Step 5: Commit evidence**

```bash
git add outputs/shuttle_detection/training/BASELINE_TRAINED.json
git commit -m "exp: establish shuttle detection baseline"
```
