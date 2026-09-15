# The 14 closure questions — interim answers, with the evidence behind each

Status: **INTERIM**. The final retrain (round-2 large-target data + the hard-negative pool) and the
freeze have not happened yet, so every answer here is current-as-of the measurements listed. Anything
that would change is marked. Questions 5 and 14 are the two that genuinely cannot be closed yet, and
the reason is stated in each.

Models referred to below:
  `baseline_synthetic_only` - trained on 400 positives spanning 2.45-32.86 px (the original baseline)
  `baseline_nearfield`      - + 100 near-field positives spanning 39.5-283.9 px

---

## 1. Baseline 成功了吗

**Yes as a baseline, no as a deployable detector.** Trained and validated: 100 epochs, recall 0.458 /
mAP50 0.550 on its own validation set. On the frozen 160-image set it reaches recall 0.444, and on the
1750-2070-row controlled matrix 0.206 (orig) rising to 0.480 (nearfield). What fails is the real
domain: recall 0.000 on the 10 verified real photographs. So the pipeline, protocol and evaluation
chain all work; the model does not yet transfer.

## 2. 稳定像素区间

| range | measured recall (`baseline_nearfield`, marginalised) |
|---|---|
| below 4 px | **0.000** - and the set contains nothing detectable there |
| 4-8 px | 0.25-0.40 |
| 8-12 px | 0.425 |
| 12-16 px | 0.625 |
| 16-24 px | 0.825 |
| **24-512 px** | **0.900-1.000** |
| 768 px and above | **0.000** (until the next retrain) |

So the STABLE band is roughly **24-512 px**, and 16-24 px is usable at 0.825.

## 3. 退化点在哪

Two of them, and they are not symmetric:

- **Below about 8 px** recall falls off a cliff: 0.40 at 4-8 px, 0.425 at 8-12, and 0.000 below 4 px.
  The floor is partly a property of the test set: a full-resolution check found the smallest annotated
  targets are single pixels on real photographs, so nothing could be detected there by anyone.
- **Above 512 px** recall is 0.000, because the training data stops at 283.9 px. This is a
  TRAINING-DATA ceiling, not an optical or architectural one, and it is the one the current data
  return is attacking.

## 4. 小于多少像素不可靠

**Below 16 px is unreliable** (0.825 at 16-24 px, 0.625 at 12-16, 0.425 at 8-12, 0.40 at 4-8).
**Below 4 px it is not merely unreliable but meaningless** in this test set, and that was verified by
looking at the images rather than inferred from the score.

## 5. 三维数据结果

**Measured, and it closes.** The frozen `SYNTHETIC_3D` set is 84 images on FLAT synthetic backgrounds
in 1280 px frames, sizes 4-72 px, rendered directly from the 3D asset rather than composited on real
photographs. Both models, same protocol:

| bucket | n | orig recall | nf recall |
|---|---|---|---|
| `<4 px` | 21 | 0.000 | 0.000 |
| `4-6 px` | 7 | 0.000 | 0.000 |
| `6-8 px` | 8 | 0.625 | 0.500 |
| `8-12 px` | 7 | 0.714 | **0.857** |
| `12-16 px` | 7 | 1.000 | 1.000 |
| `16-24 px` | 10 | 1.000 | 1.000 |
| `24-32 px` | 6 | 1.000 | 1.000 |
| `>32 px` | 18 | 0.611 | **1.000** |

| overall | orig | nf |
|---|---|---|
| recall | 0.5238 | **0.6071** |
| mAP50 | 0.4817 | **0.5971** |
| precision | 0.4231 | **0.5368** |

Three things follow. First, the near-field data return improved this set too, taking the `>32 px`
bucket from 0.611 to 1.000 - an INDEPENDENT confirmation of the near-field finding on a different
renderer path and a different background family. Second, the floor agrees exactly: `<4 px` and `4-6 px`
are 0.000 for both models, the same floor seen everywhere else. Third, a plain 3D-rendered view of the
asset is considerably EASIER than the same asset composited on a real photograph - 0.607 here against
0.480 on the 2070-row real-background set - which is the domain gap showing up as a number rather than
as an opinion.

The one regression is `6-8 px` (0.625 -> 0.500) on n=8, which is inside the noise for that group.

## 6. 真实图片结果

**Recall 0.000 on the 10 verified real positives, with 13 false positives across the 6 verified real
negatives.** The cause is now QUANTIFIED rather than hypothesised: the real positives measure
**838-1578 px** while the training data reaches 284 px, and the measured rule is that the detection
ceiling sits at roughly twice the training ceiling.

Every other candidate cause was checked first, as spec 06 section 4 requires: the training pool had
ZERO targets above 33 px, so this was never an architecture question. The prescribed fix is more
large-target training data, and round 2 of that data is generating now.

## 7. 真实视频结果

**No recall can be claimed, and none is.** The 150 frames have no per-frame ground truth, so spec 04
permits only a diagnostic continuity test. That test: the model emits a detection on 31-67% of frames
at under 1.1 candidates per frame and a maximum confidence of 0.745.

Re-verified on the actual frames rather than inherited: the only match-like clip is `ball_badminton`,
a BALL sport, and zooming its net region at 2x across four frames shows rackets but **no projectile**.
So the footage does not reliably contain an identifiable shuttlecock at all. Whether any of those
detections is a shuttle is UNKNOWN, and the prudent reading - the model was measured to fire on small
white blobs and these are outdoor courts full of players in white - is recorded as a hypothesis, not a
result.

## 8. 主要姿态失败原因

| pose (n=40 each, `baseline_nearfield`) | recall |
|---|---|
| flight_rotation | 0.600 |
| feather_end_on | 0.550 |
| pitch_only | 0.525 |
| cork_end_on | 0.475 |
| oblique | 0.400 |
| yaw_only | 0.225 |
| side | 0.150 |
| roll_only | 0.100 |

**`side` (the true side view) is the hardest real orientation.** `roll_only` did not move at all, and
that is EXPECTED rather than a failure: the shuttle is near-axisymmetric about its length axis, so a
roll-only family changes feather layout and shading but barely the silhouette. The set implementer
predicted a flat roll curve before it was measured, and the measurement agrees.

## 9. 运动模糊影响

**Not resolvable at the sample size available.** The blur groups are n=20 each with a 21.9% confidence
half-width, and the readings are 0.478 (none), 0.200 (light), 0.150 (medium), 0.150 (heavy). The
pattern is in the expected direction but the light and heavy levels are within noise of each other.
Reporting "blur costs X" from this would be over-reading it. A blur sweep at n>=40 is what would
settle it.

## 10. 误检主要来自什么

Two scene types, established by looking at the six worst backgrounds and then CONFIRMED on independent
new data:

1. badminton halls and arenas with **distant players in white** (bg_009 89 FP, bg_024 55, bg_015 45);
2. **group photographs of people in light-coloured shirts** (bg_007 82 FP, bg_006 57, bg_004 45).

The confirmation: 36 new images collected from Wikimedia Commons AFTER the diagnosis, chosen for that
character, with no reuse of any earlier image - and **31 of 36 fire**, producing 70 boxes, 6 of them
above 0.5 confidence. A prediction made before the data existed, and it held.

On 30 shuttle-free real scenes the model produces **65 boxes with a maximum confidence of 0.931**, and
9 of the 30 scenes carry at least one box above 0.5. This is a precision failure that thresholding
cannot fix, because the false positives are confidently wrong.

## 11. 是否需要回流数据

**Yes, and it has already worked once.** 100 new large-target samples took near-field recall from
0.000 to 1.000 up to 512 px and moved the detection ceiling from ~33 px to ~512 px. A second round is
generating now to reach the 1578 px real targets, and a 36-image hard-negative pool is ready for the
false-positive side.

What is NOT needed, and this is as important: no architecture change has been justified. Every failure
measured so far traces to data coverage, which spec 06 section 4 requires be excluded before any
structural conclusion.

## 12. 边界移动了多少

| | training ceiling | detection ceiling |
|---|---|---|
| original baseline | 32.86 px | ~33 px |
| after round 1 | 283.94 px | **512.5 px** |

A 15x move on the measured ceiling, with the small-target side unchanged (marginalised over 5 scenes:
+0.050 at 6-8 px, -0.025 at 8-12 px, both inside noise). The real targets at 838-1578 px are still
outside it, which is what round 2 addresses.

## 13. 普通结构是否够用

**Yes, so far, on the evidence available.** A plain single-stage, single-class YOLO detector reached
1.000 recall across 24-512 px once its training data covered that range, and every failure found has
been traced to data rather than structure. Nothing measured yet requires a two-stage, temporal, or
otherwise more elaborate design.

The honest caveat: this is a statement about the failures MEASURED. The real domain still scores 0.000
and the false-positive rate is still bad, so "sufficient" here means "no structural change has been
justified yet", not "validated for deployment".

## 14. 是否真的需要 P2 — CANNOT BE CLOSED YET, and saying so is the answer

The Level-3 gate is explicit: no P2 work before the Level-1 and Level-2 adjustments have been tried
and shown insufficient. Round 1 of Level-2 (data) produced a 15x ceiling move, and round 2 is running.
**Until those land and are measured, P2 is not needed and must not be started** — that is not an
evasion, it is the gate working as designed.

The measurements that WOULD justify P2, stated in advance so the decision is not made post hoc:
- a real-domain failure that survives training data covering the real size range AND the real
  appearance character;
- a false-positive rate that hard negatives fail to reduce;
- a temporal requirement (miss streaks, reacquisition latency) that a per-frame detector cannot meet
  even with adequate data.

---

## What is still unfinished, stated plainly

1. Round 2 large-target data is generating; the retrain and the three-protocol retest have not run.
2. The hard-negative pool is built but not yet trained on, so the false-positive answer is a diagnosis
   and a plan, not yet a result.
3. ~~The `SYNTHETIC_3D` frozen set has not been re-evaluated.~~ CLOSED: measured, recall 0.607
   (nf) against 0.524 (orig), with the `>32 px` bucket moved 0.611 -> 1.000.
4. `bg_001` is anomalously hard for a reason none of these measurements explains.
5. Plan 5 needs camera intrinsics, image resolution, stereo baseline, the required working distance,
   the latency budget and the deployment GPU from the human partner.
6. Real video cannot be scored without per-frame ground truth on footage that actually contains a
   shuttlecock.
7. `AMBIENT`/`KEY` in the renderer have never been calibrated against a real camera, so the synthetic
   appearance is plausible rather than validated.