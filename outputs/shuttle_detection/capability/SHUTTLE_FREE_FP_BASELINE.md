# False positives on shuttle-free scenes, and the full ledger of the first data return

Date: 2026-09-15  |  Measurement: 30 frozen real backgrounds containing NO shuttlecock at all, fed
to each model with no target present. Every detection is therefore a false positive by construction,
with no confound from a real target in frame.

## 1. The clean baseline

| Scene | Detections | Max confidence | Detections at conf >= 0.5 |
|---|---|---|---|
| bg_009 | **7** | 0.774 | 1 |
| bg_007 | 4 | **0.931** | 2 |
| bg_003 | 4 | 0.435 | 0 |
| bg_001 | 4 | 0.322 | 0 |
| bg_004 | 3 | **0.912** | 3 |
| bg_006 | 3 | **0.906** | 2 |
| bg_016 | 3 | 0.713 | 1 |
| bg_025 | 3 | 0.527 | 1 |

Across all 30 scenes with the `baseline_nearfield` model: **65 detections, 7 scenes perfectly clean,
9 scenes carrying at least one detection above 0.5 confidence, and a maximum of 0.931.**

This is a better metric than the false-positive count taken from the frozen P3 set, because there the
shuttle IS present and a misplaced box is ambiguous between a false positive and a localisation
failure. Here there is nothing to find, so every box is unambiguously wrong.

## 2. What the first data return actually bought and cost

The same model that produced the near-field gain was measured on every axis available:

| Axis | baseline_synthetic_only | baseline_nearfield | Verdict |
|---|---|---|---|
| >32 px recall (S7 ladder, n=360) | 0.000 | **0.772** | big gain |
| overall recall, 1750 rows | 0.170 | **0.467** | gain |
| mAP50 | 0.040 | **0.250** | gain |
| 24-32 px recall | 0.250 | **0.850** | gain |
| 16-24 px recall (S1, n=40) | 0.275 | **0.500** | gain |
| 6-8 px recall (S1, n=40) | 0.225 | 0.025 | regression, disputed (see below) |
| 8-12 px recall (S1, n=40) | 0.300 | 0.125 | regression, disputed |
| total FP on 30 shuttle-free scenes | 63 | 65 | neutral |
| scenes with a conf >= 0.5 FP | **5 of 30** | **9 of 30** | worse |
| worst FP confidence | 0.833 | **0.931** | worse |

So the honest summary is: **a large near-field gain, a disputed small-target cost, and a small but
real worsening of false-positive confidence.** The last of these was not visible in any earlier
measurement and only appeared once the shuttle-free protocol was used - the P3-derived FP count was
in fact slightly better (608 -> lower) and would have hidden it.

## 3. The disputes and unknowns, stated rather than resolved

1. **The 6-12 px regression is contradicted.** The controlled S1 curve (one background, n=40) shows
   -0.200 and -0.175; the frozen P3 set (30 backgrounds) shows -0.036 and +0.020. Different sign at
   8-12 px. A size-by-background block design has been dispatched to settle it; until it lands,
   neither figure should be quoted as the answer.
2. **Why false-positive confidence worsened is not established.** Adding 100 large synthetic targets
   plausibly made the model more willing to fire on large light-coloured blobs, but that is a
   hypothesis. Nothing here measures it.
3. **Precision remains the dominant defect regardless.** 0.186 overall, 0.098 for the original model,
   and 65 boxes on scenes that contain nothing. The `FALSE_POSITIVE_DIAGNOSIS.md` characterisation
   (two scene types: halls with distant players in white, and group photographs of people in light
   shirts) still stands unchallenged, and the hard-negative pool remains unbuilt.

## 4. Reproduction

```bash
# both models, 30 shuttle-free backgrounds, conf 0.05, imgsz 640, device 0
#   outputs/shuttle_capability/real_images/backgrounds/*.jpg
#   baseline_synthetic_only/weights/best.pt  and  baseline_nearfield/weights/best.pt
```
