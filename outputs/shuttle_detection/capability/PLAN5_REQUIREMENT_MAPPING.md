# Plan 5 requirement mapping, rebuilt on the deployment-aligned curve (v2)

Supersedes the first version of `PLAN5_REQUIREMENT_MAPPING.md`, and the change is the point.

## 1. The useful working distance bands, v2

| fx | recall >= 0.85 | recall >= 0.50 |
|---|---|---|
| 500 | 0.48 m | 0.85 m |
| 800 | 0.77 m | 1.37 m |
| 1000 | 0.96 m | **1.71 m** |
| 1500 | 1.44 m | 2.57 m |
| 2000 | 1.93 m | 3.43 m |
| 3000 | 2.89 m | 5.15 m |

(The far bound is the number; the near bound is an artefact - see section 3.)

## 2. Why the bands are much narrower than before, and why that is the correct direction

The first version of this mapping was built from the numpy-rendered controlled matrix. That set measures
**renderer alignment, not capability**: it reports 0.90 recall at 32 px, where the deployment-aligned
appearance gives 0.45. For fx=1000 it promised a working range of 0.12 to 5.37 m at recall >= 0.5; the
rebuilt mapping says **1.71 m**.

The narrower figure is the one measured in the appearance the release was trained for, on 237 rows of
Isaac-rendered holdout frames at fixed sizes. **A requirement mapping that had been left on the numpy curve
would have told a reader the robot could work at three times the distance it actually can.**

## 3. The near bound is an artefact and should not be quoted

It prints as 0.02 m because the scan starts there and the curve is flat-topped at 1.000 above its last
measured rung. It has no physical meaning for a racket robot, which never approaches within centimetres.
**The mapping constrains how FAR the camera may be**, and the far bound is where recall falls off: the
appearance-aligned ladder reaches only 0.45 at 32 px and 0.25 at 8 px.

That is also a change from the earlier analysis. In earlier rounds the binding constraint at the near end
was the 512 px training ceiling, which the second data return moved to 1023 px. With v2 detecting 1024 px at
1.000 and the real positives at 0.800, **the near end is no longer binding at any realistic working
distance**. The constraint is now the small end, which nothing has yet improved.

## 4. What would move it

**Small-target training data.** The far bound is set by recall at 8-32 px, which is 0.25-0.45 and unchanged
across every model trained in this project. Every successful data return so far moved the LARGE end; none has
moved the small end, and the one adjustment that touched it - extending the near-field range - made it
slightly worse.

## 5. What is still missing for a single number

Camera intrinsics (fx, fy, cx, cy, distortion); image resolution; stereo baseline; the maximum effective
working distance the robot actually needs; the whole-chain latency budget; the deployment GPU; and the
downstream tolerable measurement gap. Status remains `REQUIREMENT_PARTIAL`: a mapping, not a requirement.

## 6. Caveats on the curve itself

n=20 per rung gives a 95% half-width of +/-0.219, so the SHAPE is solid and individual rungs are not tight.
And the curve is measured on synthetic renders in the trained appearance rather than on real photographs, of
which the release has six distinct images behind ten rows.
