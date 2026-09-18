# B601-DM badminton capability: result, and the honest state of each gate

Date: 2026-09-17  |  Candidate arm: reBot B601-DM  |  Status: **2 of 3 gates pass, the third is UNEVALUATED**

Per `management/rebot_b601dm/ARM_IDENTITY_DECISION.md`, the B601-DM is a **test bench / candidate arm**.
PiPER keeps the project identity; `docs/simulation/BADMINTON_ROBOT.md` is unchanged. Nothing here changes the
project state, and a passing result would only be evidence for a later decision.

---

## The three gates from the plan

| Gate | Requirement | Result |
|---|---|---|
| **Asset honesty** | the loaded USD carries the real velocity limits, verified by re-reading it after load rather than by trusting the patch step | **PASS** |
| **Provenance completeness** | no value used is SIM_ONLY or UNKNOWN without being named as such | **PASS** |
| **Physical plausibility** | a measured racket-head speed exists and is below the bound from the real motors | **UNEVALUATED - no measured speed exists** |

**The overall verdict is therefore NOT ACCEPTED, and the reason is a missing measurement rather than a failed
one.** Recording it this way matters: an unevaluated gate that is quietly reported as passed would be the
single most damaging thing this document could do.

---

## Gate 1: asset honesty - PASS, with the evidence

The shipped asset enforced `physxJoint:maxJointVelocity` of 2864.789 and 11459.156 deg/s, which are the real
motors rpm figures used as rad/s: the DM4340 runs at 52.5 rpm (315 deg/s) and the DM4310 at 200 rpm
(1200 deg/s). The asset therefore permitted **9.09x and 9.55x** the real no-load speed.

| Check | Evidence |
|---|---|
| the patch changed only velocity lines | unified diff over both files, every changed line contains "velocity"; geometry, materials and inertias hash-identical |
| the upstream tree was never modified | whole-tree hash before and after |
| the values are the real ones | 52.5 and 200 rpm convert to exactly 315.0 and 1200.0 deg/s |
| **the loaded asset carries them** | Isaac Sim started against the patched tree and logged `[step 4/4] Isaac Sim started; ground and robot asset loaded` |
| **re-read from the host, not from the patch step** | `3 x maxJointVelocity = 1200.0`, `3 x 315.0`, `2 x 859.4367` (the gripper, deliberately unchanged) |

**This is the gate the work was actually about.** Every speed figure this project might produce from this arm
would have been roughly ten times too high without it.

## Gate 2: provenance completeness - PASS

`management/rebot_b601dm/TEMP_README.md` is generated from the code and lists every non-real value, together
with what would be needed to fix each. A staleness test fails if it drifts from the generator.

| Value | Status | Why |
|---|---|---|
| Joint angle limits, J1-J6 | `UNKNOWN` | no hardware document in either repository states a range; the URDF has values but the URDF is simulation. Enforced in code - `limits_for(joint, require_real=True)` raises |
| Motor masses | `TEMP_PARAMETERIZED_PROXY` | motor3/4/5/7 modelled at 99.03 g against 362 g and 300 g; motor1 and motor6 absent |
| Gripper force and speed | `UNKNOWN` / `SIM_ONLY` | no hardware source; left at the shipped values, never substituted |
| Racket mass and inertia | `UNKNOWN` | the asset is a mesh with no inertial data |
| Motor speed and torque | partly real | real per the official table, which documents `-2EC` variants while the BOM specifies V4 |

## Gate 3: physical plausibility - UNEVALUATED

**No racket-head speed has been measured.** The harness exists, is tested, and has not produced a number.

### Why

The measurement needs Isaac Sim, which this host takes **23.6 minutes** to start when idle. At the time of the
attempt the machine was saturated by work belonging to other users, and the project constraints forbid
disturbing it:

| Observation | Value |
|---|---|
| load average | 10.21 / 8.81 / 6.10 |
| heaviest process, not mine | `pt_main_thread` at 585% CPU |
| GPU | 88% utilised, 20.7 GB held by another job |
| my Isaac Sim after 1,506 s | 3 extensions started, plus a `carb.tasking is likely stuck` warning |
| the same app on an idle host earlier | dozens of extensions by 500 s |

Competing would have slowed another user's training to make this number arrive sooner. My process was stopped
cleanly and their jobs were verified untouched.

### What the bound says anyway, because it is derivable without simulation

| Case | Bound |
|---|---|
| the shipped asset permitted | **153.4 m/s** |
| real motors, 100% of no-load | **16.06 m/s** |
| official guidance, 70% of maximum | **11.24 m/s** |

`omega * reach`, at the specified 767 mm reach. This is an **upper bound**, not a prediction: it ignores joint
coupling, which lowers it, and it stops at the flange, so a racket beyond it moves faster still.

### Three biases that would apply even once it runs

Both sides of that comparison are generous, and all three biases point the same way:

1. **The arm is modelled at 61% of its real mass** (2.7445 kg against 4.5 kg). Lighter distal links accelerate
   faster, so a measured speed is **optimistic**. The motor mass error alone explains 93% of the deficit.
2. **The racket contributes no inertia at all**, because its mass and inertia are `UNKNOWN`.
3. **The measurement is the flange speed, not the racket head.** The racket adds a moment arm beyond the
   flange; racket length is a parameter, not an assumption.

**So even a completed measurement could only support one claim: whether the result stays UNDER the bound.** A
speed above it cannot come from these motors and means the asset is still wrong - which is why
`experiment_record` flags that case as an asset fault rather than reporting it as a fast arm.

---

## What was delivered

| Task | Deliverable | State |
|---|---|---|
| 0 | `ARM_IDENTITY_DECISION.md` | done |
| 1 | joint limits with per-value provenance, plus a generated YAML mirror | done, tested |
| 2 | two patched USD variants carrying real velocity limits | done, verified by re-reading |
| 3 | the torque convention, rated by default | done, tested |
| 4 | the motor mass error quantified, and substitution **refused** with the reason recorded | done, tested |
| 5 | the receiver launched headless on this host without modifying either upstream repository | done, proven by a real start |
| 6 | one racket selected from an asset that also held a second racket, a shuttlecock and a lamp | done, tested |
| 7 | the speed bound, the duty-cycle guard, and a torque-limited measurement harness | done, tested; **no measurement** |
| 8 | this document | done |

**120 tests pass.** The two upstream repositories are unmodified at `1958511342dab181b94a2b7c068c77e54eb85d3a`
and `f01a1dc189ecb74b5435a336a68b287c146c1b60`.

---

## What would complete gate 3

One contiguous window of roughly 30 minutes with the host quiet - load near 1 and the GPU free. The harness
needs no changes:

```bash
scripts/simulation/run_isaacsim_receiver.sh --dry-run          # resolves host, asset, headless
/home/T7/ojh/robot_sim/env_isaaclab/bin/python -u scripts/simulation/measure_racket_speed.py \
  --asset outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda \
  --variant real --torque-convention rated --target-joint joint6 --seconds 6 \
  --out outputs/simulation/rebot_b601dm/swing_record.json
```

The measurement itself takes 6 seconds. It is entirely the app launch that is contended.

## What this document does NOT claim

- It does not claim the arm can reach any particular racket-head speed. No such measurement exists.
- It does not claim the arm is suitable for badminton. Two of three gates passing is not a capability result.
- It does not claim the model is physically faithful. It is known to be 39% light and to carry a racket with no
  inertia at all.
- It does not change the project state. PiPER remains the project arm; this remains a candidate.
