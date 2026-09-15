# Controlled capability matrix, 1750 rows with published error bars

Date: 2026-09-15  |  Two models on the SAME 1750-row controlled set, same protocol
(imgsz 640, conf >= 0.05, match IoU >= 0.5, Top-K K=5).
  orig = baseline_synthetic_only   (trained on 2.45-32.86 px)
  nf   = baseline_nearfield        (trained on 2.45-283.94 px)
Every conditioned group now has n>=20; size/pose/position groups have n>=40, where the 95%
half-width is 15.5%. condition_counts.csv publishes the count and half-width per condition, and
two tests hold the committed manifest to them.

## 1. Overall

| | orig | nf |
|---|---|---|
| Recall | 0.1703 | **0.4669** |
| mAP50 | 0.0401 | **0.2500** |
| Top-K hit rate | 0.1680 | 0.4669 |
| Precision | 0.0975 | 0.1857 |

## 2. Size curve (n=40 per bucket, except 16-24 at 1110 and >32 at 400)

| bucket | n | orig | nf | delta |
|---|---|---|---|---|
| <4 px | 40 | 0.000 | 0.000 | 0.000 |
| 4-6 px | 40 | 0.000 | 0.000 | 0.000 |
| 6-8 px | 40 | 0.225 | **0.025** | **-0.200** |
| 8-12 px | 40 | 0.300 | **0.125** | **-0.175** |
| 12-16 px | 40 | 0.275 | 0.350 | +0.075 |
| 16-24 px | 1110 | 0.228 | 0.404 | +0.176 |
| 24-32 px | 40 | 0.250 | **0.850** | +0.600 |
| >32 px | 400 | 0.008 | **0.788** | **+0.780** |

The near field is transformed: 0.008 -> 0.788. Below 6 px nothing changed, and nothing could - the
set contains nothing detectable there, as the full-resolution check recorded earlier.

**The 6-12 px column is a REGRESSION, and it is the honest cost of this data return.**

## 3. And it is contradicted by the frozen P3 set, which must be stated rather than resolved

| bucket | frozen P3 set (30 backgrounds) | controlled set (ONE background, bg_001) |
|---|---|---|
| 6-8 px | 0.429 -> 0.393 (-0.036) | 0.225 -> 0.025 (-0.200) |
| 8-12 px | 0.540 -> 0.560 (+0.020) | 0.300 -> 0.125 (-0.175) |
| 12-16 px | 0.579 -> 0.526 (-0.053) | 0.275 -> 0.350 (+0.075) |

The two disagree in both sign and magnitude on 6-8 and 8-12 px. The most likely reason is structural,
not noise: the controlled size sweep pins ONE background and ONE pose, so its curve is the curve of
that single scene - the generator author flagged exactly this. The frozen P3 buckets, by contrast,
span 30 backgrounds.

So the small-target cost is REAL on the single controlled scene and NOT VISIBLE on the 30-background
frozen set. Which one governs the deployment decision depends on how representative bg_001 is, and
that is not settled by either measurement. Recording the disagreement instead of quoting the
convenient half, and flagging it as the first item the next controlled revision should address:
a multi-background size sweep.

## 4. Pose curve (n=40 per family)

| pose | orig | nf |
|---|---|---|
| flight_rotation | 0.200 | **0.600** |
| feather_end_on | 0.125 | **0.550** |
| pitch_only | 0.275 | 0.525 |
| cork_end_on | 0.200 | 0.475 |
| oblique | 0.100 | 0.400 |
| yaw_only | 0.025 | 0.225 |
| side | 0.075 | 0.150 |
| roll_only | 0.125 | 0.100 |

side is the hardest orientation, and roll_only did not move at all - which is expected rather than an
omission: the shuttle is near-axisymmetric about its length axis, so a roll-only family differs in
feather layout and shading but barely in silhouette. The controlled-set implementer predicted a flat
roll curve for exactly that reason, and the measurement agrees.

## 5. Position curve (n=40 per position)

| position | orig | nf |
|---|---|---|
| bottom | 0.550 | **0.825** |
| bottom_right | 0.400 | 0.625 |
| bottom_left | 0.350 | 0.550 |
| right | 0.300 | 0.450 |
| left | 0.025 | 0.375 |
| top_left | 0.125 | 0.200 |
| top | 0.050 | 0.150 |
| top_right | 0.100 | 0.100 |

Bottom positions are consistently easier than top ones in both models. A plausible reading is the
background content behind each third of the frame rather than geometry, but this measurement does not
separate those and does not claim to.

## 6. Blur and occlusion (n=20 per level, half-width 21.9%)

| blur | orig | nf | occlusion | orig | nf |
|---|---|---|---|---|---|
| none | 0.172 | 0.478 | none | 0.171 | 0.477 |
| light | 0.200 | 0.200 | light | 0.250 | 0.100 |
| medium | 0.050 | 0.150 | partial | 0.050 | 0.000 |
| heavy | 0.100 | 0.150 | | | |

No blur effect is resolvable at n=20 with a 21.9% half-width. Occlusion does show a drop, including
0.000 at partial occlusion, but at n=20 that is two of twenty samples either way, so it is a direction
to test rather than a result.

## 7. Reproduction

```bash
python scripts/shuttle_detection/evaluate_controlled.py \
  --weights outputs/shuttle_detection/training/<run>/weights/best.pt \
  --manifest outputs/shuttle_capability/controlled_capability/manifest.csv \
  --images   outputs/shuttle_capability/controlled_capability \
  --out outputs/shuttle_detection/capability/cc1750_<tag> \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0
```
