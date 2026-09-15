# Near-field capability: the baseline fails above its trained size range, and the mechanism is
# box-scale anchoring, not appearance

Date: 2026-09-15  |  Model: `baseline_synthetic_only`  |  Probe: 48 images, controller scratch run

## 1. Why this probe exists

The real-image measurement found recall 0.000 on the 10 verified real positives, whose sizes are
838-1578 px, while the training set contains NO target larger than 32.86 px. Two explanations were
possible and they imply different remedies:

- **appearance/domain gap** - synthetic shuttles look unlike real ones at large size, and size is
  coincidental. Remedy: close the appearance gap.
- **size-coverage gap** - the model has never learned large targets, synthetic or real. Remedy: add
  large-target training samples.

Only large targets rendered with SYNTHETIC appearance can separate them, so a probe was rendered on
FROZEN test backgrounds with pose, position, blur and light pinned - varying size alone.

## 2. A prerequisite defect had to be fixed first (my own renderer)

The first probe produced invalid rows: requested 256/512/800 px all measured 166.00 px. Cause:
`calibrate_distance` clamped the camera distance to a flat `MIN_DISTANCE_M = 0.3`, which overrode
the naive starting guess and then clamped every iteration, so no target above ~166 px could be
rendered at fx=700. The failure was silent.

That mattered because the entire near-field regime - the capability measurement AND the
large-target data-return path - was unrenderable, and the real positives are 838-1578 px. Fixed in
commit 57700cc: the floor is now computed per pose from the rotated mesh forward half-extent, which
is the physically correct bound, lifting the reachable size at fx=700 to roughly 900 px. A residual
bound remains and is documented: beyond ~900 px the focal length must rise.

## 3. The result, with the fix in place

| Requested | Measured | n | Recall | Max confidence |
|---|---|---|---|---|
| 16 px | 16.0 | 8 | 0.250 | 0.527 |
| **64 px** | 64.0 | 8 | **0.000** | 0.742 |
| **128 px** | 127.5 | 8 | **0.000** | 0.846 |
| **256 px** | 257.0 | 8 | **0.000** | 0.654 |
| **512 px** | 516.5 | 8 | **0.000** | 0.752 |
| **800 px** | 788.5 | 8 | **0.000** | 0.900 |

Measured sizes now track the requested ones, so the large rows are valid renders, not artefacts.

## 4. The mechanism, which matters more than the zero

Inspecting the predicted boxes against the ground truth at 128-800 px shows the model does not
"see a big shuttle and box it badly". It either emits NO box at all, or emits a **tiny box of
6-25 px** - squarely inside its trained range - landing with IoU essentially 0.000 against a
128-800 px target.

So the classifier does not recognise a large shuttlecock as a shuttlecock at all, and the box
regressor is anchored to the small scale it was trained on. That is exactly the failure mode you
expect from a detector whose entire training distribution is 2.45-32.86 px.

## 5. Answer to the question the probe was built for

**This is a SIZE-COVERAGE GAP, not an appearance/domain gap.** The model fails on large targets
rendered with its OWN synthetic appearance, so appearance cannot be the cause. The real-image
failure at 838-1578 px is the same deficit seen through a different camera.

Consequences:
1. The remedy is large-target TRAINING data, which is a Level 1/2 (data) adjustment, not Level 3
   (architecture). Spec 06 section 4 requires exactly this check before escalating, and it now has
   its evidence.
2. The large-target data can legitimately be SYNTHETIC - the probe shows the deficit is scale, not
   realism, so synthetic large targets attack the cause directly.
3. Appearance may STILL have a residual effect on the real domain; this probe does not measure that
   and does not claim to. It only rules appearance out as the dominant cause of the large-target
   failure.

## 6. Honest limits

- n=8 per size. Enough to establish "zero recall across five sizes from 64 to 800 px", not enough to
  put a number on anything between 32 and 64 px.
- The probe used 4 frozen backgrounds and one pinned pose, so it isolates SIZE but says nothing
  about pose-by-size interaction.
- It is a controller scratch probe, not the frozen controlled set; the controlled set being rebuilt
  with replicates remains the authoritative artefact.
- `max_conf` of 0.900 at 800 px is a confident box that is not the target - consistent with the
  false-positive characterisation already recorded (small light blobs), not with "the model found
  the shuttle".
