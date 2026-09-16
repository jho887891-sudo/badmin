# The pose curve is an appearance artefact too, and the Isaac pool has no true side views

Date: 2026-09-15  |  Model: `baseline_isaac2`  |  Set: `isaac_pose`, 60 rows, three pose families x 20
repeats at a FIXED 48 px apparent size, built from the 260 HOLDOUT Isaac frames so nothing was seen in
training.

## 1. The comparison

| pose family | **Isaac appearance** (48 px, n=20) | numpy appearance (controlled matrix, n=40) |
|---|---|---|
| `cork_end_on` | **0.800** +/- 0.219 | 0.400 |
| `feather_end_on` | **0.900** +/- 0.219 | 0.375 |
| `oblique` | **0.900** +/- 0.219 | 0.275 |
| `side` | **not available - see section 3** | **0.100** |
| `roll_only` | not available | 0.025 |
| overall | **P 0.406 / R 0.867 / mAP50 0.835** | - |

**In the appearance the release was trained for, pose barely matters: 0.80-0.90 across every family
measured.** The same model on the same pose families rendered by the numpy rasteriser scores 0.275-0.400.

This is the same effect the size curve showed, now confirmed on a second axis: **the numpy-rendered
controlled matrix measures renderer alignment, not capability**, and its pose curve in particular - which
reported `side` at 0.100 and `roll_only` at 0.025, and which I earlier summarised as "side is the hardest
pose" - is largely a statement about the appearance gap rather than about orientation.

## 2. What survives from the earlier pose conclusion

The one part that was reasoned rather than measured still stands: `roll_only` is near-flat because the
shuttle is near-axisymmetric about its length axis, so a roll-only family changes feather layout and
shading but barely the silhouette. That is a property of the object and the set implementer predicted it
before measuring it.

What does NOT survive is "side is the hardest". On the deployment-aligned appearance it was never measured,
because the pool cannot render it.

## 3. A real gap in the Isaac pool, found while trying to measure it

The pose sweep was built by classifying the 260 holdout frames by the pitch of the shuttle relative to the
camera. The families that emerged were `cork_end_on` (62 frames), `feather_end_on` (46) and `oblique` (152).
**There is no `side` family at all**, because the renderer samples pitch from `uniform(-60, +60)` degrees and
a true side view needs near 90.

Two consequences, both recorded rather than papered over:

1. **The 382 Isaac training samples contain no true side views either.** They are 41% of the training
   positives, so the release has seen side views only in numpy appearance.
2. **Whether side views are easy or hard in Isaac appearance is therefore UNKNOWN.** I cannot claim the
   pose curve is uniformly good, only that the three families I could build are 0.80-0.90.

For contrast, the NUMPY training pool covers side views abundantly: 62.4% of its positives have
|pitch| >= 65 degrees. So the coverage gap is specific to the Isaac pool and was introduced by the pitch
range in my renderer, not by the data design.

## 4. What this implies

- The controlled matrix pose and size curves should be read as renderer-alignment measurements and the
  appearance-aligned sets as capability measurements. The freeze package already says this for the size
  curve; it now applies to pose as well and both are in the artefacts.
- A v2 pool should sample pitch to at least +/-90 degrees so side views exist in the trained appearance,
  and then the pose question can finally be answered rather than inferred.
- Nothing here justifies a structural change; it is one more instance of the same data-side explanation.
