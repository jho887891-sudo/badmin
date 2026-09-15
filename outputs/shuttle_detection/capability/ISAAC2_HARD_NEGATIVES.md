# Hard-negative batch 2: false positives halved while the real domain held

Date: 2026-09-15  |  Model: `baseline_isaac2`  |  Change since `baseline_isaac`: 53 new hard negatives
targeted at what the new model fires on, selected for the same measured character and leak-checked first.

## 1. The result

| Metric | round3 | isaac | **isaac2** |
|---|---|---|---|
| real photographs: recall | 0.000 | 0.600 | **0.600** |
| real photographs: precision | 0.000 | 0.400 | **0.500** |
| real photographs: mAP50 | 0.000 | 0.485 | **0.505** |
| false positives, 30 shuttle-free scenes | 46 | 82 | **43** |
| scenes carrying a FP above 0.5 | 8/30 | 11/30 | **5/30** |
| worst false-positive confidence | 0.874 | 0.816 | **0.669** |
| frozen P3 mAP50 | 0.140 | 0.224 | **0.253** |
| frozen P3 precision | 0.124 | 0.171 | **0.196** |
| controlled matrix recall | 0.504 | 0.527 | 0.426 |
| controlled matrix mAP50 | 0.350 | 0.411 | 0.351 |

**The precision problem is now the best it has ever been on every one of its three measures**, and the
real domain was not sacrificed: recall stayed at 0.600, precision rose to 0.500. Fifty-three images
selected by measured model response did what the earlier thirty-six did for the previous model.

## 2. The one regression, and why it may not be one

Controlled-matrix recall fell 0.527 -> 0.426 and its mAP50 0.411 -> 0.351. That is a real drop on that
set, and it has a plausible structural explanation that matters for how the whole capability matrix is
read from here on:

**The controlled set is rendered by the numpy rasteriser.** Every one of its 2070 rows composites a
numpy-rendered shuttle onto a real background. As the model is trained toward photographic (Isaac Sim)
appearance, its score on a numpy-rendered test set falls - because that test set no longer represents the
domain the model is being pointed at.

This does NOT excuse the drop, and it does not mean the capability is unchanged. It means the controlled
matrix now measures something narrower than it did: it measures capability on THIS renderer's appearance,
and it can no longer be treated as a proxy for deployment capability. The honest options are to re-render
the controlled set with Isaac Sim, or to report its numbers alongside that caveat. The caveat is cheaper
and is recorded here.

## 3. What this makes the project state

| Axis | Best model | Value |
|---|---|---|
| real photographs | isaac2 | recall 0.600, precision 0.500 |
| false positives | isaac2 | 43 boxes, 5/30 scenes above 0.5, worst 0.669 |
| frozen P3 | isaac2 | mAP50 0.253, precision 0.196 |
| controlled matrix | isaac | recall 0.527, mAP50 0.411 |
| near-field ceiling | round3/isaac | 1.000 from 192 to 1023 px |

`baseline_isaac2` is the best model on three of the five axes and the candidate for final evaluation. The
one thing it does not win is the nested set whose renderer it has moved away from.

## 4. Reproduction

```bash
python experiments/shuttle_detection/06_acquire_hard_negatives_2.py     # on the workstation: no internet on the host
python experiments/shuttle_detection/07_add_hard_negatives_2.py \
  --base-manifest manifest_train_isaac.csv --out-manifest manifest_train_isaac2.csv
bash experiments/shuttle_detection/05_retrain_and_retest.sh \
  baseline_isaac2 configs/shuttle_detection/baseline_cached.yaml manifest_train_isaac2.csv
```
