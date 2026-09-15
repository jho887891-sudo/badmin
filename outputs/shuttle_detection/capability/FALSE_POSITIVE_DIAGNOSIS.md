# False-positive diagnosis on the frozen P3 set (Spec 06 inputs)

Model: `baseline_synthetic_only`. Data: the frozen 160-image `SYNTHETIC_ON_REAL_BG` set.
Protocol: imgsz 640, conf >= 0.05, match IoU >= 0.5. Every image in this set is a positive, so
any predicted box that does not match its ground truth is a false positive.

## 1. Headline

| Quantity | Value |
|---|---|
| Predictions / TP / FP | 681 / 73 / **608** |
| FP confidence: median / **max** | 0.167 / **0.902** |
| FP with confidence >= 0.25 | **201** |
| FP with confidence >= 0.5 | **77** |

**77 false positives sit above 0.5 confidence and one reaches 0.902.** Thresholding cannot fix
this: a confidently wrong candidate is exactly what survives a downstream gate and corrupts state
estimation. This is a PRECISION problem, not a recall problem.

## 2. The false positives are concentrated, not diffuse

| Background | Samples | FP | TP | Precision |
|---|---|---|---|---|
| bg_009.jpg | 6 | **89** | 4 | 0.043 |
| bg_007.jpg | 6 | **82** | 1 | 0.012 |
| bg_006.jpg | 6 | **57** | 1 | 0.017 |
| bg_024.jpg | 5 | **55** | 3 | 0.052 |
| bg_004.jpg | 6 | **45** | 0 | 0.000 |
| bg_015.jpg | 5 | **45** | 1 | 0.022 |

Those six backgrounds carry **373 of 608 false positives (61%)**. The rest of the 30-background
pool is comparatively clean. So this is not a uniform tendency to hallucinate; it is a small number
of background textures the model has learned to fire on.

Inspecting the crops at full resolution shows why: the worst backgrounds are dense in white
specks and blobs of roughly the same apparent scale as a small shuttlecock. The model has learned
"small white blob" and those textures supply hundreds of them per frame.

## 3. What the highest-confidence false positives actually are

A contact sheet of the 24 highest-confidence FPs (`fp_top24.jpg`) shows two distinct populations:

1. **Localisation failures on the real target.** Most tiles contain the rendered shuttlecock, and
   the prediction sits on it but the box does not reach IoU 0.5. These are counted as one FP plus
   one FN, and they are a box-quality problem rather than a hallucination.
2. **Genuine responses to background structure.** At least one tile shows a wooden floor with a
   shoe and court lines and no shuttlecock at all - a pure false positive on real image texture.

## 4. A caveat that weakens one earlier conclusion, stated plainly

Full-resolution inspection of the `<4 px` bucket found that its smallest members are not
meaningful detection targets at all. The 1.0 px sample is a single pixel on a photograph of a wall,
a bench and a person: nothing is visible at the annotated position. The 8.9 px sample sits on a
background containing hundreds of similar white specks, so the target is not distinguishable from
its surroundings even in principle.

Therefore the earlier statement "<4 px is a hard floor, recall 0.000 in both models" should be read
as **"below about 4 px this test set does not contain a detectable target"** rather than "the model
fails there". The model still fails at 4-8 px, where targets are real but small, and that part of
the conclusion is unaffected. This is a limitation of the test set as built by the older renderer,
not an excuse for the model, and it is one more reason the calibrated controlled set is being built.

## 5. What Spec 06 asks for next, with the evidence now in hand

| Spec 06 section 8 output | Content from this diagnosis |
|---|---|
| `failure_summary` | 608 false positives on 160 images; 77 above 0.5 confidence |
| `failure_distribution` | concentrated: six backgrounds carry 61% |
| `root_cause_hypothesis` | the model responds to small white blobs; several frozen backgrounds are dense with them at shuttle-like scale |
| `evidence` | per-background FP counts above; full-resolution crops; confidence distribution |
| `recommended_adjustment_level` | Level 1/2 (data), NOT Level 3: this is a data-coverage failure, not an architecture limit |
| `required_new_data` | a HARD_NEGATIVE_POOL of new images with the same CHARACTER (white speckle, court lines, bright small highlights) - NOT the frozen images themselves |
| `retest_scope` | re-run this exact protocol on the same frozen set after retraining |

## 6. Reproduction

The FP counts, confidences and per-background breakdown come from `frozen_p3_after_baseline/`,
specifically `predictions.csv`, `size_bucket_metrics.csv` and `background_metrics.csv`. The crop
sheet is `fp_top24.jpg`; the full-resolution check is `p3_fullres_check.jpg`.
