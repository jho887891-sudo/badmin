# Input ROI at 2x doubles small-target recall, and that is the first quantified case for Level 3

Date: 2026-09-15  |  Level-2 item 3.3  |  Model: `baseline_os` (v2)  |  Set: the small-target ladder, 8 rungs
x 40, appearance-aligned and holdout.

## 1. The measurement

Spec 07 section 3.3 asks whether changing the INPUT helps. A region of interest taken from the ground truth
is an ORACLE, so these are upper bounds: in deployment a ROI would need a coarse pass or a motion cue first.

| target | full frame | **ROI at 2x** | delta | ROI at 4x | delta |
|---|---|---|---|---|---|
| 6 px | 0.325 | **0.450** | **+0.125** | 0.225 | -0.100 |
| 8 px | 0.275 | **0.550** | **+0.275** | 0.200 | -0.075 |
| 12 px | 0.250 | **0.400** | **+0.150** | 0.125 | -0.125 |
| 16 px | 0.475 | **0.500** | +0.025 | 0.175 | -0.300 |
| 24 px | 0.525 | 0.400 | -0.125 | 0.200 | -0.325 |
| 32 px | 0.625 | 0.575 | -0.050 | 0.225 | -0.400 |
| 48 px | 0.825 | 0.650 | -0.175 | 0.150 | -0.675 |
| 64 px | 0.850 | 0.775 | -0.075 | 0.250 | -0.600 |
| **overall** | 0.519 | **0.537** | **+0.019** | 0.194 | -0.325 |

## 2. What the shape says, which the overall number hides

**At 2x the ROI helps exactly where the problem is: +0.275 at 8 px, +0.150 at 12 px, +0.125 at 6 px.** It
hurts at the large end, where recall was already 0.85. The overall figure is a wash because the two effects
cancel, and reading only that number would have missed the result entirely.

**At 4x it is uniformly worse, at every size.** So the trade is not "more zoom is better": there is an
optimum, it depends on the target size, and 4x is past it. The reason is out-of-distribution appearance -
the model was trained on 1280 px composites where a shuttle looks a certain way, and magnifying it enough
turns it into a blur the model has never seen.

## 3. Why this is the first real case for Level 3

Spec 07 gates Level 3 on Level 1 and Level 2 being exhausted with a STRUCTURAL bottleneck remaining. Every
measurement until now pointed the other way: failures were data failures, and the one structural remedy
considered - a P2 branch for small objects - addresses spatial detail lost in the stride, which is not what
an 8 px recognition failure looks like.

**This changes the picture, and only partly.** A 2x ROI recovers +0.275 recall at 8 px, the largest single
improvement measured at the small end anywhere in this project. But it requires knowing WHERE to crop, and
obtaining that from a single-stage detector is circular: you need a detection to choose the ROI that would
let you detect. Getting it from a motion cue is not available in still images, and getting it from a coarse
first pass IS the two-stage architecture that Level 3 gates.

So the honest position is:

| Question | Answer |
|---|---|
| Does input ROI help? | **Yes, at 2x, at the small end, by up to +0.275** |
| Does it help through the whole range? | No - it costs large-target recall, roughly a wash overall |
| Can it be used as it stands? | **No - it needs an oracle to choose the ROI** |
| Does that justify a two-stage structure? | **It is the first measurement that argues for one, with a number attached** |

That is materially different from the earlier "P2 is not needed" answer, which rested on there being no
structural bottleneck. There is now one, quantified, at 8-16 px, and it is reachable only by choosing where
to look before looking.

## 4. What is NOT claimed

- The oracle sees the answer. A real ROI would be wrong sometimes, and a wrong ROI is worse than none -
  that cost is not in these numbers.
- One zoom is tested per size. The optimum might be size-adaptive: 2x below 16 px, 1x above. That is a
  follow-up, not a result.
- The set is synthetic renders in the trained appearance, 40 rows per rung, +/-0.155 half-width. The +0.275
  at 8 px is the only delta beyond that, and it is the one the conclusion rests on.

## 5. A note on this experiment's first run

The first version returned 0.000 at every size for both conditions, which is impossible because the
full-frame column had to reproduce the earlier 0.325-0.850. The cause was double-scaling: Ultralytics returns
boxes in the coordinates of the array it was GIVEN, and the script divided them by a 640/patch-width factor
a second time. **The all-zero column was the tell.** A result identical to zero across every condition of a
measurement that has non-zero values elsewhere is a bug report, not a finding.
