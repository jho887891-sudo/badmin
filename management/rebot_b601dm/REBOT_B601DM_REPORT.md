# reBot Arm B601-DM simulation resources: download and inspection report

Date: 2026-09-16  |  Scope: download + inspect only. No environment changes, no simulation run yet.

## A. Download locations

| Item | Path |
|---|---|
| Isaac Sim repository | `/home/T7/dgut/robot_sim/reBot-Isaacsim` |
| Hardware repository | `/home/T7/dgut/robot_sim/reBot-DevArm` |

Both were cloned **on the workstation and transferred**, because **the host has no outbound network**: `git
ls-remote` fails against a dead proxy at `127.0.0.1:10080` and `curl` fails likewise. Nothing was written to
the root filesystem; `/home/T7` remains at 1.4 T free.

## B. Commit hashes

| Repository | HEAD |
|---|---|
| `reBot-Isaacsim` | `1958511342dab181b94a2b7c068c77e54eb85d3a` |
| `reBot-DevArm` | `f01a1dc189ecb74b5435a336a68b287c146c1b60` |

Verified identical on the workstation and on the host after transfer. Neither directory existed beforehand,
so nothing was overwritten.

## C. B601-DM model files

| Layer | Path | Size |
|---|---|---|
| Top-level stage | `reBot-Isaacsim/usd/reBot_B601_DM/reBot_B601_DM.usda` | 9,875 B |
| Robot payload | `.../payloads/robot.usda` | 5,198 B |
| Physics (masses, limits) | `.../payloads/Physics/physics.usda` | 16,208 B |
| PhysX drives | `.../payloads/Physics/physx.usda` | 4,973 B |
| MuJoCo layer | `.../payloads/Physics/mujoco.usda` | 9,757 B |
| Geometry (binary) | `.../payloads/geometries.usd` | 14,120,676 B |
| Base / instances / materials | `.../payloads/{base,instances,materials}.usda` | 29,815 / 30,750 / 11,818 B |

A second, unrelated asset exists at `usd/RS-rebot-dev-arm/` - that is the **RS** variant on RobStride motors,
not the DM arm. Its `scripts/` directory contains useful validation code (see K) but targets different hardware.

## D. URDF paths

| Copy | Path | Size |
|---|---|---|
| Isaac Sim repo | `reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf` | 16,656 B |
| Isaac Sim repo, vendored | `reBot-Isaacsim/third_party/reBotArm_control_py/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf` | 16,656 B (identical) |
| Hardware repo | `reBot-DevArm/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf` | 16,720 B |
| Simulation export | `reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.csv` | 38 lines, CAD inertials |

Joint names are listed at `.../config/joint_names_reBot_B601_DM.yaml`.

## E. CAD / STEP paths

All under `reBot-DevArm/hardware/reBot_B601_DM/` - **93 STEP + 10 STP files**:

| Item | Path |
|---|---|
| Full assembly v1.1 | `reBot_B601_DM_v1.1_20260425.step` (also a `.stp` variant in the tree) |
| 3D-printed parts | `3D_Printed_Parts/*.step` (base plate, links, cable restraints, fingers) |
| Metal parts | `Metal_Parts/*.step` |
| Purchased parts | `Purchased_Parts/` |

## F. Joint names, J1-J6

`joint1, joint2, joint3, joint4, joint5, joint6` - then `gripper_joint` (fixed), `gripper_joint1` and
`gripper_joint2` (prismatic). Confirmed in the URDF and in `joint_names_reBot_B601_DM.yaml`, which lists
`['', 'joint1', ..., 'joint6', 'gripper_joint1', 'gripper_joint2']` - note the leading empty string.

## G. J1-J6 limits, with the source of each

### Angles (radians)

| Joint | Lower | Upper | Source | Real-hardware status |
|---|---|---|---|---|
| joint1 | -2.8 | 2.8 | URDF; USD `physics:lowerLimit/upperLimit` = -160.428 / 160.428 degrees | **UNKNOWN** |
| joint2 | -3.14 | 0 | URDF; USD -179.909 / 0 degrees | **UNKNOWN** |
| joint3 | -3.14 | 0 | URDF; USD -179.909 / 0 degrees | **UNKNOWN** |
| joint4 | -1.87 | 1.57 | URDF | **UNKNOWN** |
| joint5 | -1.57 | 1.57 | URDF | **UNKNOWN** |
| joint6 | -3.14 | 3.14 | URDF | **UNKNOWN** |

### Velocity - and this is the finding of the inspection

| Joint | Sim value | Sim source | Real no-load | Real rated | Verdict |
|---|---|---|---|---|---|
| joint1-3 | **50 rad/s** | URDF `limit velocity`; USD `physxJoint:maxJointVelocity = 2864.789` deg/s | **5.50 rad/s** (52.5 rpm) | 3.77 rad/s (36 rpm) | **SIM_ONLY, 9.09x too high** |
| joint4-6 | **200 rad/s** | URDF; USD `= 11459.156` deg/s | **20.94 rad/s** (200 rpm) | 12.57 rad/s (120 rpm) | **SIM_ONLY, 9.55x too high** |

The sim numbers are the real motors' **rpm figures used as rad/s**: 52.5 -> 50, 200 -> 200. Converting
properly gives 5.50 and 20.94 rad/s. Real parameters are from the official SeeedStudio Damiao table
(`https://wiki.seeedstudio.com/damiao_series/`), rows `J4340-2EC` and `J4310-2EC V1.1`.

### Torque

| Joint | Sim value | Real rated | Real peak | Verdict |
|---|---|---|---|---|
| joint1-3 | **27 N·m** | **9 N·m** | **27 N·m** | **CONFIRMED_REAL as PEAK only** |
| joint4-6 | **7 N·m** | **3 N·m** | **7 N·m** | **CONFIRMED_REAL as PEAK only** |

Both sim values equal the peak torque exactly, so they are real numbers - but a controller that drives to
27 N·m continuously is operating at three times the rated torque.

Caveat on the comparison: the official table documents the `-2EC` variants, while the BOM specifies
`DM4340P(V4)` and `DM4310(V4)` (`reBot-DevArm/hardware/reBot_B601_DM/readme.md:141-142`). The agreement at
27 and 7 N·m says they are the same family, but **V4-specific numbers were not found and are UNKNOWN**.

## H. Link masses and inertias

**Complete.** Every one of the 10 links carries a mass and a full 6-component inertia tensor in both the
URDF and `physics.usda`, and `reBot_B601_DM.csv` adds centre-of-mass for 26 sub-components (motors 3-7, CNC
and printed parts). Spot-check values agree across all three sources.

One anomaly worth flagging: **`motor3`, `motor4`, `motor5` and `motor7` all carry mass 0.0990319 kg** -
identical to seven digits across two different motor models. That reads as a placeholder rather than CAD
mass, and it matters for a swing test because J4-J6 inertia is dominated by those motors.

## I. DM4340P / DM4310 real parameters

**Not present in either repository.** A repository-wide search for rated/peak torque, gear ratio or
reduction found **zero** hits; the BOM lists only purchase links. The numbers above come from the SeeedStudio
wiki, fetched separately.

| Parameter | DM4340 | DM4310 |
|---|---|---|
| Rated torque | 9 N·m | 3 N·m |
| Peak torque | 27 N·m | 7 N·m |
| No-load speed | 52.5 rpm | 200 rpm |
| Rated speed | 36 rpm | 120 rpm |
| Reduction ratio | **40:1** | **10:1** |
| Mass | ~362 g | ~300 g |
| Supply | 24 V (15-32 V) | 24 V (15-32 V) |

## J. Official thermal testing

**Found, and it is the strongest hardware document in either repository**:
`reBot-DevArm/hardware/reBot_B601_DM/performance_testing/Performance_Testing_zh.md`.

| Test | Result | Termination |
|---|---|---|
| 5-70% reach reciprocating, 1.5 kg | > 2 h | motor 2 reached 90 C, stopped by operator |
| 5-70% reach reciprocating, 2.5 kg | 40 min | motor 2 thermal protection |
| 5-100% reach reciprocating, 1.5 kg | 45 min | motor 2 thermal protection |
| Hover at 70% reach, 1.5 kg | 18 min | motor 2 thermal protection |
| Hover at 100% reach, 1.5 kg | 3 min | motor 2 thermal protection |

Official recommendations (lines 85-88): payload < 1.5 kg; **working radius < 70% of reach (450 mm)**;
**motion speed < 70% of maximum**; ambient 15-35 C; and above 35 C ambient or 75 C motor temperature, reduce
both load and speed. The document states the tests used **V4 motors** and warns that **V3 and earlier
differ**.

**This is the single most important number for a swing test**: the write-up says the structure held up and
the terminating cause was always motor-2 overheating, at 70% reach and 1.5 kg.

## K. Hardware-free joint testing in Isaac Sim

**Yes.** `reBot-Isaacsim/reBotArm_Isaacsim/isaacsim_joint_test_sender.py` states in its own header
"does not depend on a real arm", sends 6-DoF joint angles by UDP at 60 Hz to
`isaacsim_joint_receiver.py`, and interpolates between preset poses. Related senders: `isaacsim_ik_sender.py`,
`isaacsim_traj_sender.py` (MIN_JERK profiles), `gravity_joint_sender.py`.

Also present but for the **RS** arm rather than DM: `usd/RS-rebot-dev-arm/scripts/` with 14 validation
scripts (`sim_hold_test.py`, `validate_dynamics_physics.py`, `gravity_droop_analysis.py`).

**Launch procedure** (README_EN.md lines 142-162): the receiver must run under the official `python.sh`:
`"<isaacsim>"/python.sh "<workspace>"/reBot-Isaacsim/reBotArm_Isaacsim/isaacsim_joint_receiver.py`. The
sender runs under plain `python3` or `uv`.

**Compatibility: satisfied.** The repository requires Isaac Sim 6.0.0; the host has **6.0.1.0** installed as
a pip package in `env_isaaclab` (`site-packages/isaacsim`, dist-infos `isaacsim-6.0.1.0`,
`isaacsim_app-6.0.1.0`). IsaacLab is 3.0.0. **There is no standalone Isaac Sim tree and no `python.sh`**, so
the documented launch command has to be adapted to the env python - that is the one integration detail that
needs work before anything runs.

## L. Parameters still missing

| Missing | Why it matters |
|---|---|
| **V4-specific motor specs** | the official performance and spec documents are for `-2EC`; the BOM is V4 |
| **Real joint velocity limit** | no hardware document states one; the only numbers are the sim's, which are wrong by ~9x |
| **Continuous (rated) torque per joint** | only peak is confirmed; rated is 9 and 3 N·m from the family table |
| **Joint angle limit provenance** | the URDF values have no hardware source; UNKNOWN whether they are the real limits |
| **Motor thermal model** | no thermal resistance/capacity given, so overheating cannot be simulated |
| **Torque-speed curves at 24 V** | only a 12 N·m-family load-curve image, not a per-motor curve |
| **Racket mass and inertia** | not applicable yet - no racket model exists |
| **motor3/4/5/7 true mass** | all 0.0990319 kg, apparently a placeholder |

## M. Readiness for "add a racket and measure maximum racket-head speed"

**Not ready, and the reason is specific rather than general.**

The assets and the test harness are sufficient: the USD loads, the URDF is complete with masses and
inertias, the gripper attachment point is well defined (`link6` -> `gripper_joint` fixed -> `gripper_link`,
URDF lines 251-254), the joint-test sender runs without hardware, and the requirement of Isaac Sim 6.0.0 is
met by the installed 6.0.1.0.

**But a racket-speed measurement taken today would be meaningless, because the simulator enforces a joint
speed limit about nine times the real one** (`physxJoint:maxJointVelocity` 2864.789 and 11459.156 deg/s = 50
and 200 rad/s, against the real 5.50 and 20.94 rad/s). At the 767 mm reach this bounds the tip at about
**153 m/s in sim against 16 m/s real at no-load, 9.6 m/s at rated, and 11.2 m/s at the officially
recommended 70% of maximum speed**. A number measured inside that gap would be a property of the asset, not
of the arm.

**What must happen first**, in order:

1. Set the six joint velocity limits to the real values (5.50 / 20.94 rad/s), or to 70% of them per the
   official recommendation, and record that change as a deliberate deviation from the shipped asset.
2. Decide the torque convention - peak (27 / 7) for a brief swing, rated (9 / 3) for anything sustained -
   and record which. The thermal table says the arm cannot sustain high load, so a swing test needs a
   stated duty cycle.
3. Resolve the `motor3/4/5/7` mass placeholder, because wrist inertia drives racket-head speed.
4. Obtain the V4 motor data, or record explicitly that `-2EC` figures were used as a proxy.

Then a racket can be attached and a swing measured - and the measurement will still need the duty-cycle and
temperature caveats attached, because the official testing shows motor 2 protecting itself within 3 minutes
at full reach with only 1.5 kg.
