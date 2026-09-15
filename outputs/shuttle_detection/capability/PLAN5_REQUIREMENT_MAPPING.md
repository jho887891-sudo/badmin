# Plan 5, the derivable half: the requirement mapping as a function of the unknown camera parameters

Date: 2026-09-15  |  Status: **REQUIREMENT_PARTIAL, but actionable**

`minimum_required_target_px` is a pinhole projection once fx and the working distance are known. The
repository holds no calibrated fx and no image resolution - `CameraIntrinsics` is marked
`REQUIRES_CALIBRATION` and `docs/hardware_interface.md` is a placeholder - so spec 05 section 6 forbids
stating a number. What CAN be stated is the mapping itself, computed against the measured recall
curve, so the number falls out the moment the hardware is known.

## 1. Apparent target size

`px = fx * extent / distance`, using the shuttlecock skirt diameter 0.0618 m from
`configs/shuttlecock.yaml` (overall length 0.0779 m gives a longer equivalent).

| fx | 0.5 m | 1 m | 2 m | 3 m | 5 m | 8 m | 12 m |
|---|---|---|---|---|---|---|---|
| 500 | 61.8 | 30.9 | 15.5 | 10.3 | 6.2 | 3.9 | 2.6 |
| 800 | 98.9 | 49.4 | 24.7 | 16.5 | 9.9 | 6.2 | 4.1 |
| 1000 | 123.6 | 61.8 | 30.9 | 20.6 | 12.4 | 7.7 | 5.2 |
| 1500 | 185.4 | 92.7 | 46.4 | 30.9 | 18.5 | 11.6 | 7.7 |
| 2000 | 247.2 | 123.6 | 61.8 | 41.2 | 24.7 | 15.5 | 10.3 |
| 3000 | 370.8 | 185.4 | 92.7 | 61.8 | 37.1 | 23.2 | 15.5 |

## 2. Measured recall at that size

Interpolated from the measured marginalised curve: S8 over 5 scenes (n=40 per bucket) below 40 px, and
the S7 near-field ladder (n=40 per rung) above 64 px. Below 4 px the set contains nothing detectable -
a full-resolution check found the smallest targets are single pixels on real photographs - and above
512 px the model has not been trained. Both ends are measured, not assumed.

| fx | 0.5 m | 1 m | 2 m | 3 m | 5 m | 8 m | 12 m |
|---|---|---|---|---|---|---|---|
| 500 | 0.99 | 0.90 | 0.67 | 0.44 | 0.40 | 0.00 | 0.00 |
| 800 | 1.00 | 0.94 | 0.87 | 0.71 | 0.42 | 0.40 | 0.40 |
| 1000 | 1.00 | 0.99 | 0.90 | 0.83 | 0.54 | 0.41 | 0.40 |
| 1500 | 1.00 | 1.00 | 0.93 | 0.90 | 0.78 | 0.50 | 0.41 |
| 2000 | 1.00 | 1.00 | 0.99 | 0.91 | 0.87 | 0.67 | 0.44 |
| 3000 | 1.00 | 1.00 | 1.00 | 0.99 | 0.90 | 0.85 | 0.67 |

## 3. The requirement, read as a band

The constraint is not a far limit but a BAND. Too far and the target falls below the size the model
can see; **too close and it exceeds the 512 px the model was trained on, which is a property of the
training data rather than of the camera.**

| fx | recall >= 0.9 band | recall >= 0.5 band | >= 0.5 band width |
|---|---|---|---|
| 500 | 0.06 - 1.11 m | 0.06 - 2.69 m | 2.63 m |
| 800 | 0.10 - 1.77 m | 0.10 - 4.31 m | 4.21 m |
| 1000 | 0.12 - 2.21 m | 0.12 - 5.37 m | 5.25 m |
| 1500 | 0.18 - 3.32 m | 0.18 - 8.07 m | 7.89 m |
| 2000 | 0.24 - 4.42 m | 0.24 - 10.75 m | 10.51 m |
| 3000 | 0.36 - 6.62 m | 0.36 - 16.12 m | 15.76 m |

## 4. What this changes about the blocker

Before this, Plan 5 read as "blocked on camera intrinsics". It is still blocked from producing a
SINGLE number, but the mapping is now complete on the model side, so:

- the human partner supplies fx and the working-distance range, and `minimum_required_target_px` is a
  lookup rather than an experiment;
- the near-end bound is not a hardware fact at all - it is the 512 px training ceiling, and it moves
  outwards exactly as the training data grows. That is the same rule the near-field data return
  produced: the detection ceiling tracks the training ceiling with a margin of about two;
- the far-end bound is the one that depends on optics: a longer lens buys working range roughly
  linearly, which is what the table shows.

## 5. Still missing, and only the human partner can supply it

Camera intrinsics (fx, fy, cx, cy, distortion); image resolution; stereo baseline; the maximum
effective working distance the robot actually needs; the whole-chain latency budget; the deployment
GPU; and the downstream tolerable measurement gap. Until those exist the honest status is
`REQUIREMENT_PARTIAL` - a mapping, not a requirement.

## 6. Reproduction

```bash
python tools/plan5_requirement_mapping.py
```
