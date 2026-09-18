# reBot B601-DM Badminton Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Global Constraints:**
- On remote `jxxy`: do not modify Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / driver, do not stop the existing vLLM or any other user process, and keep project data under `/home/T7/dgut/robot_sim/`.
- Never assert a simulation limit as a real-hardware capability. Every joint limit carries one of `CONFIRMED_REAL`, `SIM_ONLY`, or `UNKNOWN` (`BADMINTON_ROBOT.md:216-217`), and `SIM_ONLY` values may never be used to justify a capability claim.
- Temporary physical values use `TEMP_PARAMETERIZED_PROXY` and are listed in `TEMP_README.md` (`BADMINTON_ROBOT.md:509-552`).
- A change is kept only after the same measurement protocol shows benefit; otherwise revert it and record the attempt.
- Every experiment records code commit, asset hash, config, seed, and output paths.
- The two upstream repositories stay unmodified on disk; this work adds a patched asset beside them, never inside them.

---

**Goal:** Make the reBot B601-DM arm in Isaac Sim physically answerable for the question "what racket-head speed can this arm actually produce", by replacing the shipped joint limits with real motor limits, attaching a badminton racket, and measuring the swing under the official thermal duty cycle.

**Architecture:** The shipped USD enforces `physxJoint:maxJointVelocity` of 2864.789 and 11459.156 deg/s, which are the real motors' rpm figures used as rad/s and are **9.09x and 9.55x the real no-load speeds**. Measurement before this is corrected would describe the asset, not the arm. A single source-of-truth limits file, carrying a provenance tag per value, generates the patched asset; the racket attaches at `link6 -> gripper_link` reusing the project's existing `racket_tcp` / `racket_contact_frame` conventions; and the swing is measured against a bound derived from the real limits rather than reported as a bare number.

**Tech Stack:** Python, Isaac Sim 6.0.1 (`env_isaaclab`), Isaac Lab, USD, pytest, numpy.

**Spec:** `docs/simulation/BADMINTON_ROBOT.md` (Robot module design, racket frames at lines 167-184 and 602-644) and `management/rebot_b601dm/REBOT_B601DM_REPORT.md` (inspection evidence for every number below).

---

### Task 0: Resolve the arm-identity decision before any asset work

The project design spec assumes a **PiPER** arm (`BADMINTON_ROBOT.md:169`, "球拍通过 FixedJoint 固定在 `PiPER link6`"), while this work targets the **reBot B601-DM**. Nothing below is valid until this is settled.

**Files:**
- Create: `management/rebot_b601dm/ARM_IDENTITY_DECISION.md`

- [ ] **Step 1: Record the three options and their consequences**

| Option | Consequence |
|---|---|
| B601-DM replaces PiPER | `BADMINTON_ROBOT.md` racket section must be re-pointed at the B601-DM link tree, and the Morph One whole-body contract re-checked |
| B601-DM is a second platform | two Robot modules, shared interfaces; both must satisfy the same acceptance |
| B601-DM is a test bench only | no change to `BADMINTON_ROBOT.md`; results are advisory |

- [ ] **Step 2: State which one is in force, with the date and who decided it**

- [ ] **Step 3: Commit**

```bash
git add management/rebot_b601dm/ARM_IDENTITY_DECISION.md
git commit -m "docs: record the arm identity decision for the badminton capability work"
```

---

### Task 1: Write the authoritative joint-limit table with provenance

Every number here is already established by inspection; this task makes it machine-readable and tagged so no downstream code can silently use a `SIM_ONLY` value as real.

**Files:**
- Create: `src/simulation/rebot_b601dm/joint_limits.py`
- Create: `configs/simulation/rebot_b601dm_joint_limits.yaml`
- Test: `tests/simulation/rebot_b601dm/test_joint_limits.py`

- [ ] **Step 1: Write the failing provenance test**

```python
def test_every_limit_carries_a_provenance_tag():
    from src.simulation.rebot_b601dm.joint_limits import LIMITS
    for name, spec in LIMITS.items():
        assert spec["provenance"] in {"CONFIRMED_REAL", "SIM_ONLY", "UNKNOWN"}
        assert spec["source"], f"{name} has no source"
```

- [ ] **Step 2: Populate the table from the inspection evidence**

| Joint | Angle (rad) | Velocity sim | Velocity real | Torque sim | Torque real | Provenance |
|---|---|---|---|---|---|---|
| joint1-3 | -2.8..2.8, -3.14..0, -3.14..0 | 50 rad/s | **5.50 rad/s** (DM4340, 52.5 rpm) | 27 N·m | peak 27 / rated 9 N·m | angles `UNKNOWN`, velocity `CONFIRMED_REAL`, torque `CONFIRMED_REAL` (peak) |
| joint4-6 | -1.87..1.57, ±1.57, ±3.14 | 200 rad/s | **20.94 rad/s** (DM4310, 200 rpm) | 7 N·m | peak 7 / rated 3 N·m | as above |

Sources to record verbatim: `reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf` (URDF limits), `reBot-Isaacsim/usd/reBot_B601_DM/payloads/Physics/physx.usda` (enforced `maxJointVelocity`), `reBot-DevArm/hardware/reBot_B601_DM/readme.md:141-142` (motor assignment), and the official SeeedStudio Damiao table rows `J4340-2EC` / `J4310-2EC V1.1`.

- [ ] **Step 3: Assert the angle limits are UNKNOWN, not real**

No hardware document in either repository states a joint angle range. The test must fail if someone later tags them `CONFIRMED_REAL` without a source.

- [ ] **Step 4: Run and commit**

```bash
pytest tests/simulation/rebot_b601dm/test_joint_limits.py -v
git add src/simulation/rebot_b601dm/joint_limits.py configs/simulation/rebot_b601dm_joint_limits.yaml tests/simulation/rebot_b601dm/test_joint_limits.py
git commit -m "feat: record B601-DM joint limits with per-value provenance"
```

---

### Task 2: Build the patched USD with real velocity limits

This is the task that makes every later measurement meaningful.

**Files:**
- Create: `src/simulation/rebot_b601dm/patch_asset.py`
- Create: `outputs/simulation/rebot_b601dm/reBot_B601_DM_real_limits.usda` (generated)
- Test: `tests/simulation/rebot_b601dm/test_patch_asset.py`

- [ ] **Step 1: Write the failing patch test**

```python
def test_patched_asset_uses_real_velocity_limits(tmp_path):
    out = patch_asset(SOURCE_USD, tmp_path, limits=LIMITS)
    text = out.read_text(encoding="utf-8")
    assert "physxJoint:maxJointVelocity = 315.0" in text   # 5.50 rad/s in deg/s
    assert "2864.789" not in text
```

- [ ] **Step 2: Patch the three velocity values, and nothing else**

| Joint | Shipped | Patched |
|---|---|---|
| joint1-3 | 2864.789 deg/s | **315.0 deg/s** (5.50 rad/s, DM4340 no-load) |
| joint4-6 | 11459.156 deg/s | **1200.0 deg/s** (20.94 rad/s, DM4310 no-load) |
| gripper | 859.4367 deg/s | leave as shipped, tag `SIM_ONLY` |

Also write the 70%-of-maximum variant demanded by the official recommendation (`Performance_Testing_zh.md:87`): 220.5 and 840.0 deg/s. Which one a given experiment uses must be recorded in that experiment.

- [ ] **Step 3: Write a diff test that fails on any change beyond the intended lines**

The upstream asset must remain reproducible from its own hash; an over-broad patch would silently alter geometry or inertias.

- [ ] **Step 4: Run and commit**

```bash
pytest tests/simulation/rebot_b601dm/test_patch_asset.py -v
git add src/simulation/rebot_b601dm/patch_asset.py tests/simulation/rebot_b601dm/test_patch_asset.py
git commit -m "feat: generate a B601-DM USD carrying real joint velocity limits"
```

---

### Task 3: Declare the torque convention explicitly

27 and 7 N·m are the **peak** torques; rated are 9 and 3 N·m. A brief swing may use peak, but the official thermal testing says the arm cannot sustain load, so the convention must be stated per experiment rather than inherited from the asset.

**Files:**
- Modify: `configs/simulation/rebot_b601dm_joint_limits.yaml`
- Test: `tests/simulation/rebot_b601dm/test_torque_convention.py`

- [ ] **Step 1: Write a test that refuses an unlabelled torque run**

```python
def test_torque_convention_is_required():
    with pytest.raises(ValueError):
        resolve_torque_limit("joint1", convention=None)
```

- [ ] **Step 2: Implement `peak` and `rated` resolution, with `rated` as the default**

Defaulting to rated is the conservative choice and matches the official advice to stay under 1.5 kg at under 70% reach.

- [ ] **Step 3: Run and commit**

---

### Task 4: Replace the motor mass placeholder, or mark it TEMP

`motor3`, `motor4`, `motor5` and `motor7` all carry mass `0.0990319 kg` — identical to seven digits across two different motor models, which reads as a placeholder. Wrist inertia dominates racket-head speed, so this cannot be left unexamined.

**Files:**
- Create: `management/rebot_b601dm/TEMP_README.md`
- Test: `tests/simulation/rebot_b601dm/test_motor_mass.py`

- [ ] **Step 1: Look for a real source**

The official table gives **~362 g** for DM4340 and **~300 g** for DM4310. Compare against the CAD-derived values in `reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.csv`.

- [ ] **Step 2: Either substitute sourced masses, or record the placeholder as `TEMP_PARAMETERIZED_PROXY`**

If no per-link source is found, do not invent one: record `UNKNOWN` with the official total-motor mass as the only available anchor, and list it in `TEMP_README.md`.

- [ ] **Step 3: Run and commit**

---

### Task 5: Make the Isaac Sim receiver launchable on this host

The README requires the official `python.sh` (`README_EN.md:144-147`), but the host has Isaac Sim as a **pip package** in `env_isaaclab` (`site-packages/isaacsim`, version **6.0.1.0**) with **no standalone tree and no `python.sh`**.

**Files:**
- Create: `scripts/simulation/run_isaacsim_receiver.sh`
- Test: `tests/simulation/rebot_b601dm/test_launcher_contract.py`

- [ ] **Step 1: Write a test that the launcher exports the required PYTHONPATH**

Isaac Lab is importable only with its `source/*` directories on `PYTHONPATH`; the launcher must set that, and must not modify the interpreter, PyTorch, CUDA or the driver.

- [ ] **Step 2: Implement the launcher against `env_isaaclab/bin/python`**

- [ ] **Step 3: Prove the receiver starts headless and loads the patched asset**

Record app-launch time; earlier work on this host measured Isaac Sim startup at 103-200 s, so the acceptance must not use a short timeout.

- [ ] **Step 4: Run and commit**

---

### Task 6: Attach the racket and define its frames

Reuse the project's existing racket conventions rather than inventing new ones: `racket_tcp` and `racket_contact_frame`, with `+X_racket` the face normal and `+Z_racket` handle-to-head (`BADMINTON_ROBOT.md:167-184`).

**Files:**
- Create: `src/simulation/rebot_b601dm/racket.py`
- Test: `tests/simulation/rebot_b601dm/test_racket_attachment.py`

- [ ] **Step 1: Write the failing attachment test**

```python
def test_racket_attaches_to_the_flange():
    # B601-DM flange chain, from the URDF: link6 -> gripper_joint (fixed) -> gripper_link  (lines 251-254)
    prim = attach_racket(robot, racket_usd, parent="gripper_link")
    assert parent_of(prim) == "gripper_link"
```

- [ ] **Step 2: Attach as a FixedJoint to `gripper_link`**

The gripper fingers (`gripper_joint1/2`, prismatic, 0..0.0715 m) must be excluded or locked; a swinging racket on a free slider is not a fixed tool.

- [ ] **Step 3: Record the racket mass properties, tagged `UNKNOWN` until measured**

A racket asset exists in the repository at `assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb`. Its mass and inertia are not established; the plan must not assume them.

- [ ] **Step 4: Run and commit**

---

### Task 7: Measure racket-head speed inside the real bound

**Files:**
- Create: `scripts/simulation/measure_racket_speed.py`
- Test: `tests/simulation/rebot_b601dm/test_speed_bound.py`

- [ ] **Step 1: Write the reference bound as a test, not as prose**

```python
def test_reference_bound_matches_the_real_motors():
    # omega * reach, upper bound; reach 0.767 m from README_zh.md:176
    assert approx(tip_speed_bound(5.50, 0.767), 4.22, rel=0.02)    # DM4340 no-load
    assert approx(tip_speed_bound(20.94, 0.767), 16.06, rel=0.02)  # DM4310 no-load
    assert approx(tip_speed_bound(0.70 * 20.94, 0.767), 11.24, rel=0.02)  # official 70% guidance
```

- [ ] **Step 2: Drive a commanded swing and record peak racket-head speed**

Report the sim number **together with** the bound it must not exceed. A speed above the bound means the asset is still wrong, not that the arm is fast.

- [ ] **Step 3: Enforce the official duty cycle**

The official testing terminated every run on motor-2 overheating: 2.5 kg at 40 min, 1.5 kg at full reach in 3 min hover, and 1.5 kg at 70% reach in 18 min hover (`Performance_Testing_zh.md:57-68`). A swing test must state its duty cycle and either stay inside 70% reach / under 1.5 kg or record a thermal guard.

- [ ] **Step 4: Record the experiment identity and commit**

Commit, asset hash, config, seed, limits variant (100% or 70%), torque convention, duty cycle, measured peak, and the bound.

---

### Task 8: Acceptance for this phase

- [ ] **Step 1: The three gates, all required**

| Gate | Passes when |
|---|---|
| Asset honesty | the loaded USD carries the real velocity limits, verified by re-reading it after load, not by trusting the patch step |
| Provenance completeness | no value used in the measurement is tagged `SIM_ONLY` without being named as such in the report |
| Physical plausibility | measured racket-head speed is below the bound from the real motors |

- [ ] **Step 2: Write the report**

`management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md`, containing the measured speed, the bound, the limits variant, the torque convention, the duty cycle, and an explicit list of what remains `UNKNOWN`.

- [ ] **Step 3: Commit**

```bash
git add management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md outputs/simulation/rebot_b601dm
git commit -m "test: measure B601-DM racket-head speed under real joint limits"
```

---

## What this plan deliberately does not do

- It does not attempt a badminton **rally** or a hitting task. That needs trajectory prediction, hit feasibility and planning, which `BADMINTON_ROBOT.md:47-62` assigns to other modules.
- It does not trust the official thermal numbers as a simulation model. They are a duty-cycle constraint to respect, not a thermal model to simulate; no thermal resistance or capacity was found in either repository.
- It does not resolve the V4-versus-`-2EC` motor question. The shipped BOM specifies V4 and the only official tables are `-2EC`; the 27 / 7 N·m agreement says same family, but until V4 data is obtained every real figure carries that caveat.
- It does not touch the two upstream repositories on disk. The patched asset is generated beside them.
