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
