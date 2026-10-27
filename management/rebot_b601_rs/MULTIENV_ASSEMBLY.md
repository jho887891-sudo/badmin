# Multi-environment B601-RS assembly (2026-10)

Question this round: why does the arm get no usable drive authority inside InteractiveScene, and can
the project run many environments reliably anyway? Answer: the InteractiveScene defect is
**characterised but not explained upstream**; a direct-Articulation route is implemented, verified from 2 to 128
environments, and recommended for now. Nothing here relaxes a joint limit or a safety value.

---

## 1. Where the drive is lost inside InteractiveScene (what was ruled out)

Every row is a real run on the GPU host, 1 env, 120 steps after commanding joint1 = 0.6 rad (yaw, no
gravity load, so any motion proves a live drive).

| Probe | Direct Articulation | InteractiveScene |
| --- | --- | --- |
| actuator groups | 1 ImplicitActuator over all 8 joints | identical |
| _has_implicit_actuators | True | True |
| data.joint_pos_target after set_joint_position_target | [0.6, 0.3, 1.0, ...] | identical |
| _joint_pos_target_sim after write_data_to_sim | [0.6, 0.3, 1.0, ...] | **identical** |
| joint1 after 120 steps | **0.596 (tracks)** | 0.423 / then 0.008 in the longer probe |
| USD articulation roots on stage | 1 (.../RebotB601RS/Geometry) | identical |
| PhysX articulation schemas | Geometry :: PhysicsArticulationRoot + PhysxArticulationAPI | same (+Newton root when articulation_props is applied) |

So the staging buffers are correct and the view is bound to the single articulation root. What was
explicitly excluded by A/B runs: spawn property overrides (rigid_props / articulation_props on and off),
replicate_physics True/False, scene.reset(), a flattened 103 MB self-contained USD, three different
write/update orderings, gravity on/off, self-collision on/off, and stiffness 100/200/400/800 (outcome
identical to five decimals in all of them). Isac Lab source inspection confirms ImplicitActuator has no
write_data_to_sim of its own: Articulation.write_data_to_sim() pushes _joint_pos_target_sim through
root_view.set_dof_position_targets(...), which is exactly the path the direct assembly uses successfully.

**Status: not explained.** The evidence says the difference lives inside the InteractiveScene/cloner
construction (something PhysX keeps from that path), not in the target buffers, the actuator config, the
gains or the scene properties. It is worked around, not papered over: the workaround is a different,
verified assembly, and the single-env scene still holds (see the F1 gate).

---

## 2. Gains: derived, not turned up

The position drive is a P controller, so a loaded joint settles at tau_g / k. Three runs at identical
targets gave:

| stiffness / damping | worst hold error | implied tau_g = k x err |
| --- | --- | --- |
| 100 / 1 | 0.053228 rad | 5.32 N m |
| 300 / 20 | 0.017627 rad | 5.29 N m |
| 1000 / 60 | 0.005675 rad | 5.68 N m |

Three independent measurements agree on a ~5.3 N m load, and the error scales as 1/k, which is what
distinguishes steady-state lag from a broken drive. With a 0.02 rad error budget: k >= 5.3 / 0.02 = 265,
so k = 300 and damping 20 (at 100 / 1 the joints were still moving at 0.17 rad/s after 2.5 s; at
300 / 20 the residual is under 0.08 rad/s). This is documented in scene_layout.py, and the F1 gate
re-measured at 0.0137 rad (was 0.0423 rad at 100 / 1).

---

## 3. The multi-environment route that works

simulation/badminton_scene/multienv_arm.py: one plain Articulation per env, spawned at
/World/envs/env_<i>/Robot/RebotB601RS with its own mount translation, driven and refreshed by the class.
Each env has its own target tensor row, its own joint state and its own reset: reset(env_ids) writes only
the listed envs.

Per-env targets in the test are deliberately different (joint1 = -0.10 / 0.00 / +0.10 rad cycling,
joint3 = 0.775 .. 1.225 rad), so a passing run proves independent targeting rather than a shared pose.

### Measured results (2 s hold at 240 Hz, after a 0.5 s settle)

| envs | all hold | worst error | limits | independent reset | steps/s | sim s / wall s | GPU delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | yes | 0.017627 | ok | ok | 117.4 | 0.49 | +2206 MiB |
| 4 | yes | 0.018784 | ok | ok | 34.4 | 0.14 | +2112 MiB |
| 16 | yes | 0.018785 | ok | ok | 10.3 | 0.04 | +2108 MiB |
| 32 | yes | 0.018786 | ok | ok | 7.0 | 0.03 | +2112 MiB |
| 64 | yes | 0.018787 | ok | ok | 3.3 | 0.01 | +2112 MiB |
| 128 | yes | 0.018790 | ok | ok | 3.6 | 0.02 | unreliable (see note) |

Notes that belong with these numbers:

* The GPU was shared with another Isaac Sim session during the whole climb, which is why GPU deltas are
  noisy and the 128-env run looks faster than 64; the memory column for 128 even went negative because
  the other session released memory mid-run. Treat throughput as an order of magnitude, not a benchmark.
* Throughput falls roughly as 1/N because every env is a separate articulation driven from Python:
  each step costs 3 Python calls per env (write, update plus the loop itself). This is CPU-bound, not
  GPU-bound - the GPU delta barely moves from 2 to 128 envs.
* The epilogue of every run shows joint3 still creeping at ~0.07 rad/s. That is the underdamped tail of
  the P drive, not an instability: the pose is inside limits and the error is stable at 0.0188 rad.

---

## 4. Recommended assembly route

**For now: direct Articulation per env (MultiEnvArm).** It is the only route that has been verified end
to end, it gives genuinely independent per-env targets and resets, and it holds the official limits at
every count from 2 to 128.

**Before anything time-critical (PPO, curriculum): fix or replace the InteractiveScene path.** At 128
envs the Python-per-env loop gives 3.6 steps/s, which is far too slow for training. The options, in the
order worth trying:

1. Reproduce the same 2-env test through InteractiveScene while capturing the PhysX-side DOF state per
   articulation (omni.physx tensors) to see whether PhysX even created a drive for the cloned joints.
2. If PhysX does not keep the drive, author the drive properties into the USD root layer (not the
   payload) and re-test - that is a non-invasive fix that stays inside the asset.
3. If Isaac Lab itself drops the configuration during cloning, produce a minimal repro against
   upstream and hand it over; do not patch the installed library in place.

Until one of those lands, do not let a scene-registered arm carry any capability claim.
---

## 5. Joint velocity channel is NOT usable yet (found by the 70843e7 review)

The review asked whether the logged joint velocities agree with the almost-flat position curves. They
do not, and the reason is not a stale buffer:

| Measurement (2 envs, gains 300 / 20, rest target, after 240 settle steps) | Value |
| --- | --- |
| IsaacLab data.joint_vel, joint2 | 0.06757433 rad/s |
| PhysX root_view.get_dof_velocities, joint2 | 0.06757433 rad/s (identical - buffer is live) |
| Numeric dq/dt from the position series, joint2 | 0.0 |
| Position change over 40 steps (1/240 s), joint2 | 0.0 rad |

So PhysX itself reports a steady velocity for a DOF whose position does not change at all. Measured
across step sizes the reported value scales with the step:

| physics rate | reported joint2 velocity | numeric dq/dt | position delta |
| --- | --- | --- | --- |
| 480 Hz | 0.03670 rad/s | -0.000029 | -0.000003 rad |
| 240 Hz | 0.06757 rad/s | 0.0 | 0.0 rad |
| 120 Hz | 0.11512 rad/s | 0.0 | 0.0 rad |
| 60 Hz | 0.16724 rad/s | 0.0 | 0.0 rad |

A rate would not scale with dt; a per-step residual does. Concrete consequences for the project:

* positions are authoritative and every gate here uses positions (hold, limits, reset, wrists);
* **nothing may differentiate, integrate or threshold joint velocity until this is resolved** - the
  safety shield's velocity proxy and any damping term based on measured velocity would see phantom
  motion of ~0.07 rad/s at steady state;
* the evidence carries this as a known issue and the test suite has a canary that FAILS if the defect
  quietly disappears (so the record has to be updated deliberately, not silently).

Evidence: outputs/simulation/rebot_b601_rs/STATE_SYNC_dt_scaling.json (four step sizes, 2 envs each)
plus MULTIENV_curve_2_probe_240hz.csv (480 rows of raw 240 Hz position AND velocity).

Not yet explained: whether this is a PhysX TGS artifact (the run log warns about
enable_external_forces_every_iteration=False), a drive-type interaction, or the reported velocity being
the pre-constraint value. That needs a solver-level probe and is out of scope for this round.

---

## 6. What the formal scene does and does not support (70843e7 review item 7)

| Path | envs | status |
| --- | --- | --- |
| create_scene(include_robot=True), B601-RS | **1 only** | verified; any other count raises NotImplementedError |
| create_scene(include_robot=True), PiPER legacy | unchanged | register-through-InteractiveScene, as before |
| create_scene(include_robot=False) | any | court/statics only, no arm |
| MultiEnvArm (this document) | 2..128 verified | separate research assembly, NOT part of create_scene |

The rule lives in simulation/badminton_scene/assembly_route.py as a pure function so it can be unit
tested without Isaac Sim (tests/simulation/rebot_b601_rs/test_scene_arm_route.py), and create_scene
calls it - the test fails if an inline copy of the rule reappears.

Running 128 environments in a test script therefore does NOT mean the formal robot scene supports 128
environments. It does not.
