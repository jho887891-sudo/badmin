# Controlled capability matrix — first run, and what it does and does not establish

Date: 2026-09-15  |  Model: `baseline_synthetic_only`  |  Set: the 138-image controlled set
Protocol: imgsz 640, conf >= 0.05, match IoU >= 0.5, Top-K K=5, GPU inference.

## 1. The one curve this run establishes

The `>32 px` bucket is now populated by the near-field ladder, and it is the result that matters:

| Bucket | n | TP | Recall | Precision |
|---|---|---|---|---|
| **>32 px** | **30** | **0** | **0.000** | 0.000 |

That bucket mixes 3 rows at 35-44 px (from the size sweep, where spec 03's fixed edges put them)
with **27 rows at 64-1023 px** from the near-field ladder. Read per SWEEP rather than per bucket, the
ladder alone is **0 of 27 detected**, at measured sizes 64.0, 96.0, 127.5, 192.0, 256.5, 383.5,
512.5, 768.0 and 1023.0 px.

This is the independent confirmation of the scratch probe reported in `NEAR_FIELD_GAP.md`, and it is
the stronger statement because the ladder was rendered with a DIFFERENT camera (fx=11200, 1280 px
frame, supersample 1) by a different implementation, on the frozen controlled set rather than on
controller scaffolding. Two independent renderings agree: recall is zero across the whole near field.

It also shows the failure is not "no output": the model emitted candidates on 27 of the 30 rows
(2 or 3 boxes each, 57 false positives in total) and not one matched. That matches the mechanism
already recorded - small boxes at its trained scale, landing at IoU ~0 against a large target.

## 2. What this run does NOT establish

The other conditioned curves are NOT results and must not be quoted as such. Their group sizes are
still 1-3 per condition, and at n=3 a binomial proportion carries a 95% half-width of about 57%:

| Dimension | Group sizes | Status |
|---|---|---|
| size (buckets other than >32) | n=3 each | not reportable |
| pose | cork_end_on 1, side 3, oblique 4, yaw_only 4, pitch_only 4, roll_only 4, flight 8 | not reportable |
| position | 1 per position except center | not reportable |
| blur | 1 per level except none | not reportable |
| occlusion | 3 per level | not reportable |

One bucket, `16-24 px n=90 recall 0.489 precision 0.310`, IS statistically usable - but n=90 there
is an artefact: every sweep other than the size sweep pins its size at 18-16 px, so those rows pile
into one bucket. It is a real measurement of that size at one pose, not a size curve.

The replicate rebuild (n>=40 for size/pose/position, n>=20 elsewhere) is still running. Until it
lands, this file reports ONE curve and says so.

## 3. Files

`controlled_baseline/{predictions.csv, size_bucket_metrics.csv, pose_metrics.csv,
position_metrics.csv, blur_metrics.csv, occlusion_metrics.csv, background_metrics.csv, summary.json}`.

## 4. Consequence

The data return for the near-field gap is already generated (`manifest_train_nearfield.csv`, 100 new
samples, 39.5-283.9 px, positives 400 -> 500, ceiling extended 8.6x). Per Ruling 9 the retrain waits
for the replicate rebuild so that a single retrain addresses every gap the matrix finds.
