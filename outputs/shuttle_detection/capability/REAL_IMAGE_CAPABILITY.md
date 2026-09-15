# Real-image capability of the nc=1 baseline — and why it is zero

Date: 2026-09-15  |  Data: the 16 human-verified REAL_IMAGE samples (10 positives, 6 negatives)
from the frozen P0 real-image pool. Manifest: `outputs/shuttle_capability/metrics/real_image_verified_manifest.csv`.
Protocol identical to the frozen-P3 comparison: `imgsz=640`, `conf>=0.05`, match `IoU>=0.5`, Top-K K=5.

## 1. The result

| Model | Recall (10 positives) | Precision | Top-1 | FP on the 6 real negatives |
|---|---|---|---|---|
| before: untrained COCO yolo26s (`--expected-class any`) | **0.900** | 0.191 | 0.800 | 19 |
| after: `baseline_synthetic_only` | **0.000** | 0.000 | 0.000 | 13 |

The trained baseline detects **none** of the 10 real shuttlecocks. Per-sample: it emits 0 or 1
candidate on almost every positive, with `best_iou` at or near 0, while still emitting up to 5
candidates on shuttle-free photographs.

## 2. Why — measured, not guessed

| Target size distribution | min | median | p90 | max |
|---|---|---|---|---|
| **Training set** (400 positives, `manifest_train_synthetic.csv`) | 2.45 px | 8.94 px | 14.49 px | **32.86 px** |
| **Real verified positives** | 837.8 px | — | — | 1578.8 px |

Samples in the training set above 100 px: **0**. Above 200 px: **0**. Above 500 px: **0**.

The two ranges do not overlap at all: the largest target the model ever saw is 32.86 px, and the
smallest real positive is 837.8 px - a **25x to 48x** gap. The model has never been shown a
shuttlecock that occupies a large fraction of the frame, so it has no reason to fire on one.

This is a DATA DISTRIBUTION GAP, not an architecture problem and not an under-training problem.
More epochs on the same data cannot fix it: the training data simply does not contain the regime.

## 3. What this changes

1. **The synthetic capability result stands on its own terms.** On small targets the baseline is a
   real improvement over the untrained model (overall recall 0.119 -> 0.444 on the frozen P3 set,
   and 0.540 at 8-12 px). That measurement was about far-field small targets and it is unaffected.
2. **The baseline cannot be used for the real deployment as it stands.** A robot also sees the
   shuttle large, when it is close, and this model is blind there.
3. **The training pool was built around one end of the problem.** The P4-A renderer targeted
   2-32 px because that was the documented failure regime; large targets were never in scope, and
   the consequence only became visible when the model met real photographs.
4. **This is the first concrete, failure-driven data requirement.** The fix is a training pool that
   covers large targets as well - near-field renders and, ideally, real photographs of shuttlecocks
   at large apparent size. It is a data-return item, not a Level-3 architecture item.

## 4. Honest limits

- **n=10 positives and n=6 negatives.** These are the only human-verified real images the project
  has. A recall of 0.000 on 10 samples is unambiguous as a failure signal, but the precision figure
  and the false-positive count (13) are small-sample numbers and should not be over-read.
- **The "before" model is not a fair competitor.** COCO has no shuttlecock class, so
  `--expected-class any` credits it for any of its 80 classes that happens to overlap the shuttle.
  That is deliberately generous and it is why 0.900 should be read as "these images are detectable
  in principle", not as a baseline to beat.
- **These 10 images are all large targets.** They cannot answer anything about real-domain
  SMALL-target performance. The real domain remains unmeasured in that regime.
- The computed `equivalent_size_px` values here (837.8-1578.8) differ by about 1% from the values
  recorded during P0 (829.8-1572.8), because they are recomputed from the YOLO labels times the
  stored image dimensions rather than copied from the P0 record. The difference is noted, not hidden.
