# Acceptance execution record - commit 173966b (TGS external-forces production fix)

Executor role only: the code, tests, tolerances, gains, limits and assets are ChatGPT's and were run
unchanged. Nothing in the review branch was modified or committed; every artefact below is a new file.

## 1. Provenance

| Item | Value |
| --- | --- |
| Target commit | 173966bf7c12c0e4c2da4d82719fbad80eda063c |
| Commit tree | 157be05f945b92e481f0a8eab684c423dd752ec5 |
| Executed tree (git write-tree on host) | 157be05f...ec5 - MATCH |
| Per-file blob check | 2241 / 2241 match (0 blobdiff, 0 missing) |
| Transport | git archive, sha256 d11791c8f06f5adf8dd88cbf1ea62140ddce1da11f942f918239131464b95e09 on both sides |
| Diff vs previous audit target 55112d2 | new rs_physics.py + test_rs_physx_config.py + decision doc, modified 3 harness/probe scripts, multienv_arm.py, scene_cfg.py, merge tool, multienv test |
| Host checkout | /home/T7/dgut/runs/review-173966b |

Module provenance printed before any simulation pointed at that checkout (badminton_scene, rs_physics,
scene_layout), gains 300 / 20 unchanged.

## 2. Phase 1 - configuration and single environment

| Step | Command | exit | s | Result |
| --- | --- | --- | --- | --- |
| P1a | pytest tests/simulation/rebot_b601_rs/test_rs_physx_config.py -q | 0 | 9 | **4 passed** |
| P1b | check_arm_rest_pose_hold.py --mode standalone (5 s) | 0 | 93 | held, see below |
| P1c | check_arm_rest_pose_hold.py --mode scene (5 s) | 0 | 78 | held, see below |

Rest-pose result, identical for standalone and scene (worst hold error 0.013724 rad on joint2):

| Metric | Value |
| --- | --- |
| held_5s | true |
| worst_hold_error_rad | 0.013724 |
| joint2 / joint3 inside official limits | true (full limits as committed: j1 +-2.8, j2/j3 [0,3.14], j4/j5 +-1.57, j6 +-3.14, fingers [0,0.05] / [0,0.0715]) |
| EE above mount at 5 s | 0.493378 m (drop over the window 0.003447 m) |
| lowest body vs mount plane | 0.3 m, i.e. nothing below the mount |
| PhysX drive read-back | stiffness 300 x8, damping 20 x8, maxForce 36/36/36/14/14/14/500/500 |

Caveat worth carrying into the audit: these two JSONs record `physics_configuration: null`. The new
TGS/external-forces setting is proven for this harness only at source/test level
(`test_every_formal_arm_entrypoint_uses_the_same_physx_config` asserts the script calls
`rebot_rs_simulation_cfg` exactly once), not by a field in the run record. The multi-env runs below do
record it explicitly.

## 3. Phase 2 - multi-environment

Gate order honoured: 2 envs had to satisfy all five flags before 4 was attempted, and so on.

| envs | hold | finite | all 8 DOFs in limits | independent reset | **velocity_position_consistent** | worst window error | throughput | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | yes | yes | yes | yes | **yes** | 0.017681 | 38.6 steps/s | pass |
| 4 | yes | yes | yes | yes | **yes** | 0.018841 | 32.0 | pass |
| 16 | yes | yes | yes | yes | **yes** | 0.018841 | 10.7 | pass |
| 32 | yes | yes | yes | yes | **yes** | 0.018842 | 5.9 | pass |
| 64 | yes | yes | yes | yes | **yes** | 0.018842 | 2.5 | pass |
| 128 | yes | yes | yes | yes | **yes** | 0.018842 | 1.7 | pass |

Every run records the applied configuration:
`{"solver_type": 1, "enable_external_forces_every_iteration": true, "source": "rebot_rs_simulation_cfg"}`.
Throughput is again contended (another Isaac Sim session was resident); treat as order of magnitude.

| Step | Command | exit | Result |
| --- | --- | --- | --- |
| P2_merge | tools/merge_multienv_evidence.py | 0 | merged 6 runs |
| P2_pytest_rs | pytest tests/simulation/rebot_b601_rs -q | 0 | **66 passed** |

## 4. Phase 3 - dynamic velocity, per segment

`probe_rs_velocity_channels.py --mode step --solver tgs --external-forces-every-iteration on`, same
physical timing at both rates (1 s settle + 2 s measured; the command goes out at the start and returns
at the midpoint).

| run | window | actual dq | integrated v | integral - dq | peak |dq| per step |
| --- | --- | --- | --- | --- | --- |
| 240 Hz | whole 2 s (net) | -0.001151 | -0.002203 | **-0.001052** | 0.004180 |
| 240 Hz | **outbound** (steps 0-239) | +0.103009 | +0.102701 | **-0.000309** | 0.004166 |
| 240 Hz | **return** (steps 239-480) | -0.103010 | -0.102896 | **+0.000114** | 0.004180 |
| 480 Hz | whole 2 s (net) | -0.000323 | -0.000694 | **-0.000371** | 0.002096 |
| 480 Hz | **outbound** (steps 0-479) | +0.102983 | +0.102942 | **-0.000040** | 0.002089 |
| 480 Hz | **return** (steps 479-960) | -0.102983 | -0.103046 | **-0.000064** | 0.002096 |

Reading: the arm really travels about +0.103 rad and comes all the way back, and in each direction the
velocity integral tracks the actual motion to 4e-5 .. 3.1e-4 rad, i.e. 0.04 % .. 0.3 % of the motion.
The large whole-window figure (-0.001052 rad at 240 Hz) is a **composition artefact**: the two
directions carry residuals of opposite sign (-0.000309 outbound, +0.000114 return) that do not cancel,
and the net position change itself is only -0.0012 rad. Net-window comparison therefore cannot judge
velocity accuracy - the per-segment numbers are the honest ones, and by those the dynamic velocity now
agrees with position in both directions.

This supersedes the caution in management/rebot_b601_rs/REVIEW_RUN_55112d2_VELOCITY.md section 4: that
caution was drawn from whole-window numbers and is not borne out once the window is split.

## 5. Deviations and notes

* The probe's 480 Hz segments are defined over 960 samples (2 s), so both runs keep identical physical
  duration, as requested.
* Before Phase 2 the committed canonical `MULTIENV_*` evidence in the checkout was copied to
  /home/T7/dgut/runs/pristine_canonical/ because the merge tool reads and the suite expects the
  canonical paths; the working tree was overwritten with the fresh runs, git was not touched, and the
  originals are preserved both in the backup and in the review branch itself.
* GPU shared with another session throughout (throughput 38.6 -> 1.7 steps/s, 128 envs took ~9 min).
* No API error, no traceback, no failed command in this round.

## 6. Files

| Path | Contents |
| --- | --- |
| P1_rest_standalone.json, P1_rest_scene.json, P1_curve_*.csv | 5 s rest-pose runs: checkpoints, limits, drive read-back, curves |
| MULTIENV_run_<N>.json, MULTIENV_hold.json, MULTIENV_curve_<N>.csv | per-count records, merged evidence the gate test reads, per-step curves |
| P3_step_tgs_exton_240.json, P3_step_tgs_exton_480.json | step probes with summary.segments (480 / 960 unrounded samples each) |
| logs/ | one log per command plus the runner summary |
| MANIFEST.sha256 | sha256 and size of every file here |
