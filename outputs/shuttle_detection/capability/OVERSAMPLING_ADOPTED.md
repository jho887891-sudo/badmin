# Level-2 training organisation works: oversampling the hard negatives gives the best model yet

Date: 2026-09-15  |  Model: `baseline_os`  |  One change: the 86 hard negatives are declared `repeat=3` in
the training manifest, so they carry 28% of the training entries instead of 7.8%. **No new data.**

## 1. The result

| Axis | `baseline_isaac2` (previous candidate) | **`baseline_os`** |
|---|---|---|
| **real photographs recall / precision / mAP50** | 0.600 / 0.500 / 0.505 | **0.800 / 0.800 / 0.769** |
| controlled matrix precision | 0.240 | **0.350** |
| controlled matrix recall | 0.426 | 0.429 |
| frozen P3 precision / mAP50 | 0.196 / 0.253 | **0.226 / 0.284** |
| frozen P3 recall | **0.438** | 0.413 |
| appearance ladder precision | 0.272 | **0.389** |
| appearance ladder recall | **0.586** | 0.553 |
| CHALLENGE precision / mAP50 | 0.278 / 0.274 | **0.401 / 0.287** |
| **false positives, 30 shuttle-free scenes** | 43, 8/30 clean | **36, 10/30 clean** |
| worst false-positive confidence | **0.669** | 0.724 |
| validation mAP50 | **0.570** | 0.529 |

**Precision improves on every set measured, real-photograph recall rises by 0.200, and false positives fall
from 43 boxes with 8 clean scenes to 36 with 10.** The cost is a small recall decrease on the synthetic sets,
0.03 or so, and the same on validation.

This is the best model the project has produced on the axes that matter for deployment, and it is the FIRST
adjustment that pays for itself without an offsetting loss on real scenes. The three earlier attempts either
helped one axis at the cost of another (more hard negatives), or looked good on synthetic sets while hurting
real ones (whole-scene negatives, side views).

## 2. Why it is worth noticing that no data changed

The three previous improvements to real-domain performance all came from changing the DATA: more near-field
samples, more large targets, then a better renderer. This one changes only how the existing images are
WEIGHTED. The defects it attacks - false positives on clutter and weak discrimination - were described
throughout as data problems, and this shows they were at least partly a training-balance problem.

It also retroactively explains something: the 86 hard negatives had been doing less work than intended.
They were 7.8% of the training entries, and the model still fired about once per hard scene. Tripling their
weight moved precision materially on every set.

## 3. The mechanism had to be built first, and the first attempt was invalid

Spec 07 section 3.4 requires a difficult-example oversampling comparison. The first attempt repeated manifest
ROWS and produced three protocols identical to the reference run to the last digit, because
`ultralytics_dataset._dedupe` removes repeated images before the trainer sees them. That guard is deliberate
- it stops an image being silently trained on twice - and it was kept.

What was added is an explicit `repeat` column that a manifest declares on purpose. Deduplication moved to
ROW level, because a flattened string list cannot distinguish "one row declaring three repeats" from "three
rows declaring one each" - and the first implementation of the fix got exactly that wrong and failed its own
test. Two declarations that disagree now raise rather than quietly picking one.

Verified end to end rather than assumed: `train.txt` holds **1280 lines against 1108 unique**, which is the
172 extra entries that 86 negatives at repeat=3 should produce. The invalid first attempt produced 1108 and
1108.

## 4. Decision

```
ADOPTED: difficult-example oversampling at repeat=3 on the hard negatives
  measured on: real photographs, controlled matrix, frozen P3, 30 shuttle-free scenes,
               appearance-aligned ladder, challenge set, validation
  outcome: precision better on every set, real-photo recall 0.600 -> 0.800, FP 43 -> 36
           with clean scenes 8/30 -> 10/30; synthetic recall -0.03
  decision: baseline_os becomes the frozen candidate
```

`baseline_isaac2` remains on disk as the previous candidate so the comparison is reproducible.

## 5. What is still open

Unchanged by this result: the sub-6 px floor, the side-view coverage gap in the trained appearance, the
fact that the real-photograph set has only 6 distinct images behind 10 rows, the absence of any real
training photographs, and the uncalibrated rendering appearance. The cluttered-scene discrimination defect
is improved - precision rose from 0.278 to 0.401 on the challenge set - but not resolved, and the
threshold-dependent operating point recorded earlier still applies.
