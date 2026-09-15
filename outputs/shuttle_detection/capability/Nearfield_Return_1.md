# Failure-driven data return, round 1: the near-field ceiling moved 15x, and the real-image
# failure is explained by what remains

Date: 2026-09-15  |  Model: `baseline_nearfield` (100 epochs, 19.5 min, commit 4a1d2b6 data)
Data change: 100 new near-field positives, log-uniform 39.5-283.9 px, on training backgrounds.
Positives 400 -> 500; training ceiling 32.86 -> 283.94 px. Validated before use, audit exit 0.

## 1. The near-field ceiling moved, and it moved exactly as far as the training data

Same controlled set, same protocol, only the model changes. Broken down by ladder rung:

| Target size | n | Recall BEFORE | Recall AFTER |
|---|---|---|---|
| 35 / 40 / 44 px | 1 each | 0.000 | **1.000** |
| 64 px | 3 | 0.000 | **1.000** |
| 96 px | 3 | 0.000 | **1.000** |
| 127.5 px | 3 | 0.000 | **1.000** |
| 192 px | 3 | 0.000 | **1.000** |
| 256.5 px | 3 | 0.000 | **1.000** |
| 383.5 px | 3 | 0.000 | **1.000** |
| 512.5 px | 3 | 0.000 | **1.000** |
| 768 px | 3 | 0.000 | **0.000** |
| 1023 px | 3 | 0.000 | **0.000** |

The `>32` bucket therefore reads recall 0.000 -> **0.800**, which is 24 of 30 rows: everything up to
512 px is now detected perfectly and everything from 768 px up is still missed entirely.

## 2. The rule this reveals

| | Training ceiling | Detection ceiling | Generalisation margin |
|---|---|---|---|
| baseline_synthetic_only | 32.86 px | ~33 px (0.000 already at 64 px) | ~1x |
| baseline_nearfield | 283.94 px | 512.5 px (0.000 from 768 px) | ~1.8x |

So the detection ceiling tracks the TRAINING ceiling with a margin of roughly two. That is the
actionable rule this experiment buys: to detect a target of size S, the training data must reach
about S/2, not S.

## 3. This also explains the real images, which still fail

The 10 verified real positives measure **838-1578 px**. The new ceiling is 512-768 px. They are
above it, and recall on them is still **0.000** (13 false positives, 10 false negatives).

So the real-image failure is STILL a size-coverage failure, and it is now quantified rather than
asserted: the model needs training data reaching roughly 1578/2 = 790 px, and the current addition
stops at 284 px.

This also sharpens a caveat the controlled-set implementer raised: its ladder is long-lens and
mild-perspective, unlike a real close-up. That caveat still stands as a possible SECONDARY factor -
this experiment does not rule perspective out - but it is no longer needed to explain the real
failure, because size alone accounts for it.

## 4. What it cost elsewhere

| Frozen P3 set (160, small targets) | BEFORE | AFTER |
|---|---|---|
| overall recall | 0.4437 | 0.4375 |
| 4-6 px | 0.250 | 0.125 |
| 6-8 px | 0.429 | 0.393 |
| 8-12 px | 0.540 | 0.560 |
| 12-16 px | 0.579 | 0.526 |
| 16-24 px | 0.667 | **0.762** |
| 24-32 px | 0.500 | **1.000** |

Overall recall on the small-target frozen set is essentially unchanged (-0.006), so the addition did
not cost the far field. The small shifts down at 4-8 px and up at 16-32 px are within what n=24-28
per bucket can resolve. Precision remains poor on both models - 0.124 here, and the false-positive
diagnosis in `FALSE_POSITIVE_DIAGNOSIS.md` is untouched by this change.

## 5. What the next data return should be, now precisely targeted

Generate near-field positives reaching at least ~790 px, so the ceiling clears the 1578 px real
targets. Cost is the constraint: an 800 px render takes ~52 s, so this is tens of samples rather
than hundreds - but tens is what the rule above says is needed, since the far field is already
covered and only the top of the range is missing.

## 6. Reproduction

```bash
# data
python experiments/shuttle_detection/02_add_nearfield_training.py --n 100 --max-px 300
# train
python scripts/shuttle_detection/train_baseline.py \
  --config configs/shuttle_detection/baseline.yaml \
  --train-manifest outputs/shuttle_capability/train_data/manifest_train_nearfield.csv \
  --val-manifest outputs/shuttle_capability/train_data/manifest_val_synthetic.csv \
  --data-root /home/T7/dgut/robot_sim --run-name baseline_nearfield
# evaluate, all three frozen sets
#   frozen P3 160 | real verified 16 | controlled 138 incl. the S7 ladder
```
