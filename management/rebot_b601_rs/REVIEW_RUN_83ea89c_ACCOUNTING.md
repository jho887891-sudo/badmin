# Velocity-accounting verification run - commit 83ea89c

Executor record only: code, assertions, tolerances, gains, limits and assets are ChatGPT's and were run
unchanged. Nothing on the audit branch was modified or committed.

## 1. Provenance

| Item | Value |
| --- | --- |
| Target commit | 83ea89c05af57e31dcc67b50f9ff14aa9fa66033 |
| Commit tree | 3534b4217d62fcc6abbe14bb51739815c5c5fb0d |
| Executed tree (git write-tree on host) | 3534b421...b0d - MATCH |
| Per-file blob check | 2244 / 2244 match (0 blobdiff, 0 missing) |
| Transport | git archive, sha256 716bb98de564dcc36216269e37423751913aa659c41ac27f0318b31f4b62845a on both sides |
| Diff vs previous target 173966b | new velocity_accounting.py + test_velocity_accounting.py + audit doc, modified probe and rest-pose harness |
| Host checkout | /home/T7/dgut/runs/review-83ea89c |

Module provenance printed before any simulation: badminton_scene, **velocity_accounting**, rs_physics and
scene_layout all resolve inside that checkout; gains 300 / 20 untouched.

## 2. Commands and exit codes

| # | Command | exit | s |
| --- | --- | --- | --- |
| T1 | pytest tests/simulation/rebot_b601_rs/test_velocity_accounting.py -q | 0 | 8 |
| T2 | probe --mode step --dt 1/240 --settle-steps 240 --sample-steps 480 --solver tgs --external-forces-every-iteration on | 0 | 84 |
| T3 | probe --mode step --dt 1/480 --settle-steps 480 --sample-steps 960 --solver tgs --external-forces-every-iteration on | 0 | 35 |
| T4 | check_arm_rest_pose_hold.py --mode scene (5 s) | 0 | 54 |

T1 result: **8 passed** (first step included in integral and displacement; outbound/return windows tile
without overlap; gravity ghost velocity not accepted as correct; non-finite and invalid windows rejected).

As instructed, no 128-environment climb was run this round.

## 3. Accounting verification (the requested identities)

| Check | 240 Hz (T2) | 480 Hz (T3) |
| --- | --- | --- |
| samples kept (unrounded) | 480 | 960 |
| initial_sample step | 0 | 0 |
| whole-window measured (q(N) - q(0)) | -0.000000954 | +0.000000000 |
| whole-window integrated velocity | -0.000195928 | -0.000104100 |
| whole-window integral - position | -0.000194974 | -0.000104100 |
| sum of segment actual deltas | -0.000000954 | +0.000000000 |
| **error: sum(segments) vs whole window (position)** | **0.000e+00** | **0.000e+00** |
| sum of segment integrals | -0.000195928 | -0.000104100 |
| **error: sum(segments) vs whole window (integral)** | **0.000e+00** | **6.451e-18** |
| accounting.net_matches_segment_sum | **true** | **true** |
| accounting.integral_matches_segment_sum | **true** | **true** |

Per-segment detail (identical bounds rule: outbound 1..switch_step-1, return switch_step..N):

| run | phase | steps | actual dq | integrated v | integral - dq | peak |dq| per step |
| --- | --- | --- | --- | --- | --- | --- |
| 240 Hz | outbound | 0-239 | +0.103009403 | +0.102700553 | -0.000308849 | 0.004165828 |
| 240 Hz | return | 239-480 | -0.103010356 | -0.102896481 | +0.000113875 | 0.004180282 |
| 480 Hz | outbound | 0-479 | +0.102982610 | +0.102942184 | -0.000040426 | 0.002088577 |
| 480 Hz | return | 479-960 | -0.102982610 | -0.103046284 | -0.000063673 | 0.002096027 |

## 4. What changed relative to the 173966b run (physics unchanged, accounting corrected)

| Quantity | 173966b run | this run | comment |
| --- | --- | --- | --- |
| 240 Hz whole-window integral - position | -0.001052 | **-0.000195** | the previous figure carried the first-step omission / window overlap |
| 480 Hz whole-window integral - position | -0.000371 | **-0.000104** | same cause |
| 240 Hz outbound residual | -0.000309 | -0.000309 | identical to 1e-6 |
| 240 Hz return residual | +0.000114 | +0.000114 | identical to 1e-6 |
| 480 Hz outbound residual | -0.000040 | -0.000040 | identical to 1e-6 |
| 480 Hz return residual | -0.000064 | -0.000064 | identical to 1e-6 |

So the segment-level physics reproduces exactly, while the net-window bookkeeping moved by roughly a
factor of five. That is the intended effect of a statistical-accounting fix and it is worth stating
plainly: the earlier whole-window numbers understated the agreement, not the other way round. Dynamic
velocity still tracks position to 0.04 % .. 0.3 % of the ~0.103 rad travel in both directions.

## 5. Single-env rest pose and the new metadata (T4)

The 5 s scene run now records:

`{"solver_type": 1, "enable_external_forces_every_iteration": true, "configured_by": "badminton_scene.rs_physics.rebot_rs_simulation_cfg", "verification": "configured in code; not an independent runtime PhysX solver readback"}`

Verdict unchanged and healthy: held_5s true, worst hold error 0.013724 rad, joint2/joint3 inside the
official limits, wrist 0.493378 m above the mount, drop 0.003447 m, nothing below the mount plane.
The `verification` string is accurate - the run records what was configured, not an independent PhysX
solver readback - and I did not attempt to upgrade that claim.

## 6. Notes

* GPU shared with another session again (24.8 GiB in use at start); wall-clock values are not
  performance data.
* No API error, no traceback, no failed command.
* Raw unrounded samples are kept in full: T2 carries 480 and T3 carries 960 per-step records (lab q/qd,
  PhysX q/qd, command, target) plus the step-0 initial sample.

## 7. Files

| Path | Contents |
| --- | --- |
| T2_step_tgs_exton_240.json | step probe 240 Hz: summary, summary.segments, accounting flags, 480 unrounded samples |
| T3_step_tgs_exton_480.json | step probe 480 Hz: same, 960 samples |
| T4_rest_scene_5s.json, T4_curve_scene.csv | 5 s scene rest pose with the new physics_configuration metadata |
| logs/ | one log per command plus the runner summary |
| MANIFEST.sha256 | sha256 and size of every file here |
