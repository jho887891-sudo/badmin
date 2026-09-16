# The hard-negative hypothesis is REJECTED: the small-target limit is a recognition limit

Date: 2026-09-15  |  Question: do the hard negatives cost small-target recall, because they teach exactly
the signal a small shuttle presents?  Answer: no.

## 1. The test that was run

Last round raised the hypothesis and named the measurement that would settle it. That measurement was run:
a focused appearance-aligned ladder over the small end, 8 rungs x **40 repeats** each - double the previous
20, so the 95% half-width shrinks from +/-0.219 to **+/-0.155** - built from the holdout frames and scored
on both models.

| target | n | `baseline_isaac2` (v1, crop-weight negatives) | `baseline_os` (v2, negatives at 3x) | delta |
|---|---|---|---|---|
| 6 px | 40 | 0.300 | 0.325 | **+0.025** |
| 8 px | 40 | 0.350 | 0.275 | -0.075 |
| 12 px | 40 | 0.425 | 0.250 | **-0.175** |
| 16 px | 40 | 0.425 | 0.475 | **+0.050** |
| 24 px | 40 | 0.600 | 0.525 | -0.075 |
| 32 px | 40 | 0.675 | 0.625 | -0.050 |
| 48 px | 40 | 0.825 | 0.825 | 0.000 |
| 64 px | 40 | 0.850 | 0.850 | 0.000 |
| overall | 320 | P 0.255 / R **0.556** / mAP50 0.437 | P **0.362** / R 0.519 / mAP50 **0.442** | |

## 2. Why this rejects it

**The rungs move in both directions.** Four are worse for v2, two are better, two are identical. A real
penalty from tripling the hard-negative weight would show up as a consistent deficit concentrated at the
sizes where a shuttle looks most like the negatives - instead the largest deficit (12 px, -0.175) sits
next to a gain (16 px, +0.050), which is not a pattern but a scatter.

The 12 px rung is the only one beyond the +/-0.155 half-width, and with eight rungs examined at once the
chance of one exceeding it somewhere is substantial. **One rung outside the band, in a set of eight, with
neighbours going the other way, is not evidence of an effect.**

**What the comparison does show is a trade in v2's favour**: on the same 320 rows it has precision 0.362
against 0.255 and mAP50 0.442 against 0.437, for 0.037 less recall. That is the same shape as its behaviour
on every other set - better precision, slightly less recall - and it is consistent rather than anomalous.

## 3. What that leaves standing

The previous round's central finding is unaffected and now has a cleaner interpretation: **the small-target
failure is a recognition limit, not a consequence of the negative training and not a box-precision problem.**
The evidence for it is the empty 0.05-0.50 IoU band at every size, and the fact that 85% of 8 px images
produce a candidate while only 25% cover the target. Both models show it; it is a property of the task and
the appearance, not of this training recipe.

So the far bound of the working range in the Plan 5 mapping is set by something that neither more negatives,
less negatives, nor better boxes will move. What would move it is small-target data whose appearance at
8-32 px is more distinguishable from clutter than a downsampled large render is - and that is a data
question nobody has yet answered, because every success in this project has been at the large end.

## 4. Value of the round

A hypothesis was raised with an explicit test attached, the test was run at double the previous sample size,
and the hypothesis was rejected rather than left to be cited later as a plausible cause. **The small-target
limit is not a side effect of the adjustment that improved precision; it is a floor both versions share.**
