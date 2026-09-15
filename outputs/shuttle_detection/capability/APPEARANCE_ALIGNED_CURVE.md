# The appearance-aligned capability curve, and why the numpy-rendered sets understated the release

Date: 2026-09-15  |  Set: `isaac_ladder`, 236 rows, 12 sizes x 20 repeats (1024 px has 16), built from the
386 usable Isaac Sim frames composited onto the 30 FROZEN test backgrounds. It is an ADDITIONAL set; the
frozen controlled matrix is untouched so cross-version comparisons stay valid.

## 1. The curve

| target size | `baseline_isaac2` | `baseline_round3` |
|---|---|---|
| 4 px | **0.000** | 0.000 |
| 6 px | 0.150 | 0.150 |
| 8 px | 0.200 | 0.150 |
| 12 px | 0.400 | 0.350 |
| 16 px | 0.450 | 0.200 |
| 24 px | **0.500** | 0.000 |
| 32 px | **0.550** | 0.050 |
| 64 px | **0.800** | 0.200 |
| 128 px | **0.950** | 0.100 |
| 256 px | **0.900** | 0.100 |
| 512 px | **0.950** | 0.050 |
| 1024 px | **1.000** | 0.062 |
| overall | P 0.269 / R 0.564 / mAP50 0.508 | P 0.060 / R 0.119 / mAP50 0.022 |

## 2. Two things this establishes

**First, the capability curve in the appearance the release is aimed at.** Stable detection (>= 0.90)
begins at **128 px**. It is 0.80 at 64 px, 0.50-0.55 across 24-32 px, 0.40-0.45 at 12-16 px, and falls to
0.00-0.20 below 8 px. This is the number a downstream module should plan against, because it is measured
on the same appearance the detector was trained for.

**Second, and this is the methodological point made concrete:** the numpy-rendered controlled matrix and
challenge set UNDERSTATED this release, exactly as the freeze package warned. `baseline_round3`, which was
trained on numpy renders, scores 0.05-0.20 across 24-1024 px on this ladder - it cannot see the
photographic appearance at all - while `baseline_isaac2` scores 0.50-1.00 across the same range.

So the earlier reading of the controlled matrix, where isaac2 appeared WORSE than round3, was measuring
renderer alignment rather than capability. On appearance the model was trained for, the gap is enormous
and in the opposite direction.

## 3. What this changes in the frozen release

The freeze package records the controlled-matrix number alongside a caveat. That caveat can now be
quantified rather than gestured at, and the capability section should be read with this curve as the
primary figure for deployment planning:

| Question | Numpy-rendered sets said | Appearance-aligned ladder says |
|---|---|---|
| stable range | ~24 px | **~128 px** for 0.90+, 64 px for 0.80 |
| 24-32 px | 0.775 | 0.500-0.550 |
| 128-1024 px | not separately reported | **0.90-1.00** |

The stable-range figure is WORSE aligned (128 px rather than 24 px) and the large-target behaviour is
BETTER. Both are more credible than the numpy numbers because they were measured on the right appearance.

## 4. Limits of this measurement

- **n=20 per rung** (16 at 1024 px), so the 95% half-width is about 21.9%. The shape of the curve is
  clear; individual rungs are not tight.
- The ladder composites RECOGNISABLE Isaac frames onto frozen backgrounds, so a repeat is a different
  crop and placement of an existing render rather than a fresh render. Pose diversity therefore comes
  from the 386 source frames and is finite.
- It is a new set, so it has no cross-version history. It can only be compared backwards by re-running
  older weights on it, which is what the two columns above do.
- One source frame at 1024 px was too large to place inside the 1280 px frame, which is why that rung has
  16 rows rather than 20.
