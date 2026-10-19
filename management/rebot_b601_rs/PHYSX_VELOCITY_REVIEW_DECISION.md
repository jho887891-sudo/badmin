# B601-RS PhysX velocity channel: fourth-round code decision

Status: **CONFIGURATION FIX COMMITTED, GPU REGRESSION PENDING**.

## Reproducible evidence

Source execution branch: `evidence/b601-rs-physx-velocity-55112d2`,
commit `01a9b27455c742297f370606c2e9c4f4cdf04a6f`.
Report: `management/rebot_b601_rs/REVIEW_RUN_55112d2_VELOCITY.md`.

With identical USD, joint targets and gains 300/20:

| TGS setting | Gravity | 240Hz 2s hold: ∫qdot dt - Δq (joint2) |
| --- | --- | --- |
| external forces per iteration = false | on | +0.134867 rad |
| external forces per iteration = true | on | -0.000143 rad |
| external forces per iteration = false | off | 0 rad |

At 480Hz with the **same 2-second duration**, off/on errors were
+0.073341 / -0.000311 rad. PGS at 240Hz yielded about -0.000004 rad.
Increasing minimum velocity iterations from 0/default to 2 was
indistinguishable from TGS-on.

**Conclusion:** This is strong evidence for a gravity × TGS force-update
configuration interaction. It is not evidence that the entire PhysX solver
returns correct qdot under every trajectory or contact state.

## Code change

Central helper: `simulation/badminton_scene/rs_physics.py`.

- TGS solver_type=1
- enable_external_forces_every_iteration=True

This is selected only in the following B601-RS paths:
`create_scene(include_robot=True)` when B601-RS is the formal arm,
`MultiEnvArm.spawn()` and the standalone B601-RS hold harness.
Legacy PiPER and court-only paths are not modified.

The multi-env hold gate now accepts **velocity-position consistency** over
2 seconds (6 revolute joints, residual tolerance 0.002 rad) instead of
requiring the previously documented ghost-velocity defect to exist. This
is a validation threshold, not a claim of real-robot sensor accuracy.
The prior test evidence becomes stale and *should fail* until the new
harness is executed on the GPU host and its output merged.

## Important limitation: moving-joint velocity remains unverified

The 240Hz position-step A/B probe with the switch ON improved the net
2-second discrepancy from +0.129567 to -0.001052 rad. However,
the net displacement over an outbound-and-return command was only
-0.001151 rad; a small net error alone is a misleading metric when
the trajectory contains opposing motion.

The updated `probe_rs_velocity_channels.py` records `initial_sample`
and independent `outbound` / `return` segment-wise integrals and
actual displacement. **Do not label moving-arm velocity as certified until
the segment data pass an explicitly justified accuracy criterion.**

Do not start PPO, velocity feedback safety logic or real-hardware motion
based on these results.

## Next GPU validation by DeepSeek, code authored by ChatGPT

1. Isolated checkout at the exact review SHA. Verify import provenance.
2. Run a single-environment scene F1 5s hold and the 2-env hold on the new
   TGS configuration. Record effective settings, failures, q/qdot and
   full-window limit/position errors.
3. If both pass, climb 4/16/32/64/128 envs, merge evidence and run
   the B601-RS suite. Do not overwrite historical evidence; isolate
   current output in a new evidence branch.
4. Run `probe_rs_velocity_channels.py --mode step --solver tgs
   --external-forces-every-iteration on` at 240Hz and 480Hz with
   matched physical time. Report each segment's measured Δq, ∫qdot dt,
   residual and maximum individual step displacement.
5. If dynamic fidelity still fails, report raw evidence without modifying
   source or relaxing tests.

The formal `create_scene` route still supports only one B601-RS environment;
MultiEnvArm is a separate research assembly with poor parallel throughput.
