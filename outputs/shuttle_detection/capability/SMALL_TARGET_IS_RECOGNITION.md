# The small-target failure is all-or-nothing, so it is a recognition problem and not a box problem

Date: 2026-09-15  |  Model: `baseline_os` (v2)  |  Set: the appearance-aligned holdout ladder, 237 rows,
12 sizes at n=20. This characterises the constraint that now sets the working distance.

## 1. The distribution that settles what kind of problem it is

For every row, the best IoU achieved by any candidate, binned:

| target | IoU < 0.05 (looking elsewhere) | 0.05-0.20 | 0.20-0.50 (nearly right) | >= 0.50 (matched) |
|---|---|---|---|---|
| 4 px | 0.85 | **0.00** | 0.10 | 0.05 |
| 6 px | 0.75 | **0.00** | 0.00 | 0.25 |
| 8 px | 0.75 | **0.00** | 0.00 | 0.25 |
| 12 px | 0.55 | **0.00** | 0.00 | 0.45 |
| 16 px | 0.70 | **0.00** | 0.00 | 0.30 |
| 24 px | 0.55 | **0.00** | 0.05 | 0.40 |
| 32 px | 0.50 | **0.00** | 0.05 | 0.45 |
| 64 px | 0.15 | **0.00** | 0.00 | 0.85 |
| 128 px | 0.10 | **0.00** | 0.00 | 0.90 |
| 1024 px | 0.00 | **0.00** | 0.00 | 1.00 |

**The middle band is empty at every size.** There is no population of "close but not precise enough"
predictions - not at 8 px, not at 1024 px. When the model fails on a small target its best box is
essentially DISJOINT from the ground truth, and when it succeeds the box is already good enough to match.

## 2. What that rules out, and what it points at

**It rules out a localisation-precision explanation.** If small-target failure were about a few pixels of
box error, the 0.20-0.50 band would be populated at small sizes and would thin out as size grew. It is
empty everywhere.

**It points at recognition.** The model either identifies the shuttle - and then localises it well - or it
fails to identify it and reports something else. Confirmed independently by the candidate counts: at 8 px,
**85% of images produce at least one candidate while only 25% cover the target**. The model is firing, just
not at the shuttle.

That makes this the same defect as the cluttered-scene one, at a different scale: a discrimination failure,
not a resolution or regression failure. It also means remedies aimed at box quality - better anchors, box
loss changes, refinement heads - would not touch it.

## 3. A consequence for how the working distance should be read

The Plan 5 mapping says the far bound of the working range is set by recall at 8-32 px, which is 0.25-0.45.
This result says that figure is a recognition limit: at those sizes the model cannot reliably tell the
shuttle from its surroundings, and a bigger or better-boxed prediction would not change that.

## 4. One hypothesis this raises, and it is NOT established

The hard negatives exist precisely to teach "small light blob is not a shuttlecock". That is the same
signal a small shuttle presents. It is therefore plausible that they cost small-target recall, and `v2`,
which tripled their weight, is slightly worse than `v1` at 16 and 32 px (0.300 against 0.500, and 0.450
against 0.600).

**It is not established.** n=20 per rung gives a 95% half-width of +/-0.219, so both differences are a few
rows and inside the noise, and v2 is better at 6 px while worse at 16 and 32. Recording it as a hypothesis
with the measurement that would test it is the honest position: a small-target ladder run on both v1 and v2
with enough repeats to separate them, which is maybe forty rows per rung rather than twenty.

## 5. What this does not change

Nothing about v2 as the frozen candidate. Its real-photograph recall of 0.800 and its false-positive
improvement are unaffected by this analysis, and the small-target figures are essentially unchanged from
every model this project trained.
