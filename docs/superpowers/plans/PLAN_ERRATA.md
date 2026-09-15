# Plan set errata — defects found while executing the 2026-09-15 shuttle detection plans

These notes do NOT modify the shipped plan files. Those files are governing documents delivered
with their own `PLAN_MANIFEST.json` sha256 hashes, and editing them in place would destroy the
integrity check that makes them trustworthy. This file records what was wrong and what was done
instead, so a future reader does not trip over the same lines.

## E1 — `2026-09-15-shuttle-data-asset-readiness.md`, Task 4 Step 4: a glob that does not expand

The step says:

```
pytest tests/perception/shuttle_detection/test_*audit*.py tests/perception/shuttle_detection/test_contracts.py -v
```

Neither PowerShell nor pytest 9 expands that glob. Observed:
`ERROR: file or directory not found: tests/perception/shuttle_detection/test_*audit*.py`.

**Instead:** pass the explicit file list. The controller and the implementer both did, and the
run reported `70 passed`.

```
pytest tests/perception/shuttle_detection/test_audit_cli.py \
       tests/perception/shuttle_detection/test_dataset_audit.py \
       tests/perception/shuttle_detection/test_contracts.py -v
```

## E2 — `2026-09-15-shuttle-baseline-training.md`, Tasks 4 and 5: `$(git rev-parse --show-toplevel)`

Both tasks invoke the trainer with:

```
--data-root "$(git rev-parse --show-toplevel)"
```

That cannot work on the remote training host. Per these same plans and the user constraint, the
project lives at `/home/T7/dgut/robot_sim/` and the sync deliberately EXCLUDES `.git`, so it is not
a git checkout and `git rev-parse` fails there.

**Instead:** pass the data root explicitly. `--data-root` is now a required argument of
`scripts/shuttle_detection/train_baseline.py`, and the trainer never infers it from git (no
`git rev-parse` call exists in it). The command that actually ran the baseline was:

```
python scripts/shuttle_detection/train_baseline.py \
  --config configs/shuttle_detection/baseline.yaml \
  --train-manifest outputs/shuttle_capability/train_data/manifest_train_synthetic.csv \
  --val-manifest   outputs/shuttle_capability/train_data/manifest_val_synthetic.csv \
  --data-root /home/T7/dgut/robot_sim \
  --run-name baseline_synthetic_only
```

## E3 — `2026-09-15-shuttle-baseline-training.md`, Task 5: a baseline that cannot be built

Task 5 trains a `synthetic+real` baseline. Every real image this project holds is frozen
capability-test data (the 10 verified positives and 6 negatives are part of the REAL_IMAGE
evaluation domain, and the frozen sets are recorded as never returning to training). The count of
real TRAINING positives available is therefore zero, and forcing it would violate the isolation
rule the same plan states as a global constraint.

**Instead:** Task 5 is recorded as BLOCKED pending new real training images with verified boxes.
Stage A alone satisfies the spec's baseline pass condition. See the Plan 2 ledger.

## E4 — `2026-09-15-shuttle-controlled-capability.md`, Task 4 Step 2: weights that do not exist

The step points at `outputs/shuttle_detection/training/baseline_synthetic_real/weights/best.pt`.
That directory cannot exist while E3 is unresolved.

**Instead:** the controlled capability matrix is measured on
`outputs/shuttle_detection/training/baseline_synthetic_only/weights/best.pt`, and the report must
state plainly that this is the synthetic-only baseline.

## E5 — tech stack lists that name unavailable packages

`2026-09-15-shuttle-data-asset-readiness.md`, `...-baseline-training.md` and
`...-controlled-capability.md` list `pandas` in their tech stack. pandas is installed neither
locally nor on the remote, and the project has avoided adding dependencies.

**Instead:** the audit, the dataset builder and the metrics modules use the standard library
(`csv`, `math`, `json`, `dataclasses`, `hashlib`). A negative side effect to remember:
`ultralytics` itself DOES need `polars` at weights-save time, which is a different package and is
installed into the isolated ultralytics target directory on the host, not into `env_isaaclab`.

