# Execution evidence for review commit 454197e (run on the GPU host)

Role split this round: ChatGPT authored and committed the code; this document is the execution record
produced by running that commit unchanged. No source, assertion, tolerance or test condition was
modified, and no older evidence file was overwritten.

## 1. What actually executed (provenance)

| Item | Value |
| --- | --- |
| Target commit | 454197ebb5615df2f3b5fe58567a6c813b9daf7b |
| Commit tree | ab9daae5af4893051c07095c59357b34dd9df9ca |
| Executed tree (recomputed on the host with git write-tree) | ab9daae5af4893051c07095c59357b34dd9df9ca (MATCH) |
| Per-file blob check | 2238 of 2238 files match their committed blob hash (0 blobdiff, 0 missing) |
| Transport | git archive of that commit, sha256 56a82fafc1548c4789e0c602769534a78e02e64bdf6015090f09ab9f7f086e06 on both sides |
| Local worktree | E:/具身智能/badmin_wt_454197e (git worktree, detached at the target commit) |
| Host checkout | /home/T7/dgut/runs/review-454197e |

The GPU host has no GitHub egress (proxy 127.0.0.1:10080 refused), so the commit tree was shipped as an
archive and then verified against the git object hashes - the code that ran is byte-identical to the
commit, not merely similar.

Module provenance was printed by the runner before any simulation:

```
badminton_scene : /home/T7/dgut/runs/review-454197e/simulation/badminton_scene/__init__.py
multienv_arm    : /home/T7/dgut/runs/review-454197e/simulation/badminton_scene/multienv_arm.py
assembly_route  : /home/T7/dgut/runs/review-454197e/simulation/badminton_scene/assembly_route.py
scene_layout    : /home/T7/dgut/runs/review-454197e/simulation/badminton_scene/scene_layout.py
gains           : 300.0 20.0
```

Environment: RTX A6000 (driver 580.178.04, 49140 MiB), python 3.12.13, pytest 9.1.1, git 2.34.1,
Isaac Lab from /home/T7/ojh/robot_sim/IsaacLab (installed, unchanged). The GPU was already carrying
another session's Isaac Sim instance (~42.9 GiB in use at start), so throughput numbers below are
contended, not clean benchmarks.

## 2. Commands and exit codes

All commands ran from /home/T7/dgut/runs/review-454197e with PYTHONPATH pointing at the installed
Isaac Lab source. Full stdout/stderr per command is in logs/.

| # | Command | exit | seconds |
| --- | --- | --- | --- |
| 10_climb_2 | check_multienv_hold.py --num-envs 2 --hold-seconds 2.0 | 0 | 62 |
| 10_climb_4 | ... --num-envs 4 | 0 | 83 |
| 10_climb_16 | ... --num-envs 16 | 0 | 79 |
| 10_climb_32 | ... --num-envs 32 | 0 | 152 |
| 10_climb_64 | ... --num-envs 64 | 0 | 245 |
| 10_climb_128 | ... --num-envs 128 | 0 | 462 |
| 20_probe_hold_240 | probe_rs_velocity_channels.py --mode hold --dt 1/240 | 0 | 58 |
| 21_probe_step_240 | ... --mode step --dt 1/240 | 0 | 54 |
| 22_probe_hold_240_zerog | ... --mode hold --gravity-off --dt 1/240 | 0 | 51 |
| 23_probe_step_240_zerog | ... --mode step --gravity-off --dt 1/240 | 0 | 57 |
| 30_probe_step_120 | ... --mode step --dt 1/120 | 0 | 50 |
| 31_probe_step_480 | ... --mode step --dt 1/480 | 0 | 53 |
| 32_probe_hold_480 | ... --mode hold --dt 1/480 | 0 | 48 |
| 40_merge | tools/merge_multienv_evidence.py | 0 | 1 |
| 50_pytest_rs | pytest tests/simulation/rebot_b601_rs -q | 0 | 15 |

Climb policy was respected: each count had to pass (all_envs_hold true) before the next was attempted.
All six passed, so no level was skipped and no failure was hidden.

## 3. Multi-env results (2 s window per count)

| envs | all hold | finite | all 8 DOFs in limits | independent reset | worst window error | min wrist above mount | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | yes | yes | yes | yes | 0.017627 rad | 0.435134 m | 31.4 steps/s |
| 4 | yes | yes | yes | yes | 0.018787 rad | 0.435134 m | 44.8 steps/s |
| 16 | yes | yes | yes | yes | 0.018787 rad | 0.435134 m | 30.1 steps/s |
| 32 | yes | yes | yes | yes | 0.018788 rad | 0.435134 m | 6.1 steps/s |
| 64 | yes | yes | yes | yes | 0.018790 rad | 0.435133 m | 3.4 steps/s |
| 128 | yes | yes | yes | yes | 0.018790 rad | 0.435133 m | 1.7 steps/s |

**The suite passes, but note the flag that does not:** every run records
`velocity_position_consistent: false`. That is the same state-consistency defect as before, reproduced
under the new code. It is tracked as a known issue, so the tests pass while the flag stays red - see
section 4 and the recommendation at the end.

Test suite: 61 passed on the host, and the same 61 passed locally against the same fresh evidence
(logs/50_pytest_rs.log and logs/60_pytest_rs_local.txt).

## 4. Velocity channel experiment (the raw data)

joint2 of the probe arm; the script records unrounded lab and PhysX channels plus the numeric position
derivative. lab-physx difference is 0.0 in BOTH q and qd for every run, so the two APIs agree exactly.

| run | mode | gravity | dt | mean PhysX qd | integral of qd | measured dq | integral - dq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hold_240 | hold | on | 1/240 | 0.067574 | 0.134867 | 0.000000 | **0.134867** |
| step_240 | step | on | 1/240 | 0.065353 | 0.128416 | -0.001151 | **0.129567** |
| hold_240_zero_g | hold | **off** | 1/240 | **0.000000** | 0.000000 | 0.000000 | **0.000000** |
| step_240_zero_g | step | off | 1/240 | -0.000092 | -0.002180 | -0.001141 | -0.001039 |
| step_120 | step | on | 1/120 | 0.110949 | 0.436764 | -0.003772 | 0.440537 |
| step_480 | step | on | 1/480 | 0.035572 | 0.034902 | -0.000308 | 0.035210 |
| hold_480 | hold | on | 1/480 | 0.036709 | 0.036630 | -0.000003 | 0.036633 |

What the numbers say:

1. **The phantom velocity is entirely gravity-driven.** With gravity off and a hold command the
   reported velocity is exactly 0.0 and the integral is 0.0. It is not a sampling artefact of the
   script, because the same script at the same dt reports 0.0676 rad/s when gravity is on.
2. **It scales with the physics step.** The mean reported velocity is 0.0676 at 1/240 and 0.0367 at
   1/480 (roughly halving when dt halves), and the phantom *integral* falls by ~4x (0.1349 to 0.0366),
   i.e. the accumulated phantom displacement behaves like dt squared. A real rate would not scale this
   way; a per-step solver residual would.
3. **The position channel is unaffected.** max |dq per step| over the hold window is 0.0, and the
   recorded position deltas match the drive targets within 2e-5 rad. Under a step command the arm does
   move (max |dq per step| = 0.0042 rad), so the drive itself is alive - only the steady-state velocity
   reading is not a derivative.
4. **No wrapper desync.** Isaac Lab and PhysX report identical numbers in every sample of every run.

Interpretation offered for the authors (not a fix, and not verified by me): a gravity-loaded implicit
P drive sits at equilibrium where the solver keeps producing a velocity-level correction that the
position constraint cancels within the same step. Under zero gravity that correction has nothing to
fight, hence exactly zero. If that is right, the value is the solver's correction velocity rather than
dq/dt, which matches both the dt scaling and the total absence of position drift.

## 5. Problems encountered (none blocked the run)

| Observation | Detail |
| --- | --- |
| No GitHub access on the host | proxy 127.0.0.1:10080 refused; solved with git archive + object-hash verification |
| GPU shared with another Isaac Sim session | ~42.9 GiB in use; throughput varies non-monotonically (2 envs 31.4 vs 4 envs 44.8 steps/s) - do not read these as clean scaling |
| First archive attempt had CRLF line endings | 1267 text files hashed differently; re-exported with core.autocrlf=false, then 2238/2238 matched |
| No test failures | every command exited 0; there is no traceback to report this round |

## 6. Why the suite is green while the consistency flag is red

The gate test treats the mismatch as a recorded known issue, so `velocity_position_consistent: false`
does not fail the run. I did not change that: it is the authors' call. Flagging it because the acceptance
summary can read as "state consistency solved" when the measurement still says it is not.

## 7. Open items for the authors

1. Decide whether `velocity_position_consistent` should gate (currently it does not) - as long as it is
   false, anything that differentiates or thresholds joint velocity (safety velocity proxy, damping)
   would see ~0.068 rad/s of motion that is not happening.
2. The gravity-off result is the strongest lever found so far: the residual exists only under gravity,
   so the probe should probably also cover contact/limit-free poses and different drive types before
   pointing at the solver.
3. Throughput needs a clean machine before it is quoted: 128 envs at 1.7 steps/s under contention says
   nothing about the assembly's cost.

## 8. File map

| Path (this directory) | Contents |
| --- | --- |
| MULTIENV_run_<N>.json | per-count run record: per-env targets, positions, velocities, limit trace, reset probe, performance |
| MULTIENV_hold.json | merged evidence produced by the repo's own merge tool (the file the gate test reads) |
| MULTIENV_curve_<N>.csv | per-step joint positions for every env at that count |
| review_probe_*.json | velocity-channel probes: 480 unrounded samples each (lab and PhysX q/qd, command, target) |
| logs/ | one stdout/stderr log per command, plus the pytest runs and the runner summary |
| MANIFEST.sha256 | sha256 and size of every file in this directory |
