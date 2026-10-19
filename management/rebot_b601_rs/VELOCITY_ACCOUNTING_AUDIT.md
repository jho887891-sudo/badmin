# Velocity accounting review after GPU acceptance of 173966b

Status: **STATIC PHYSICS CONFIG ACCEPTED; DYNAMIC PROBE SUMMARY BUG FIXED IN CODE, RERUN PENDING**.

## Source of truth and scope

- Production PhysX implementation: `review/b601-rs-physx-production-20261019`
  commit `173966bf7c12c0e4c2da4d82719fbad80eda063c`.
- Unmodified execution evidence: `evidence/b601-rs-acceptance-173966b`
  commit `972819366fba3dcb841c3dcb1909455d1e40ed72`.
- Raw trajectory: `outputs/simulation/rebot_b601_rs/production_review_173966b/P3_step_tgs_exton_240.json`.

Single-env static hold (5 seconds), 2/4/16/32/64/128-env hold (2 seconds),
position limits, reset and static velocity-integral check all passed in the
execution record. This supports the **specific** B601-RS TGS configuration
`enable_external_forces_every_iteration=True`; it is not a blanket guarantee
of true-robot dynamics, contact stability or PPO throughput.

## Off-by-one in the old dynamic probe summary

Old `summary.integrated_physx_velocity` summed only v(2)..v(N) while
`summary.measured_delta` used q(N)-q(1). Both omitted the first timestep.
The `segments` field correctly used q(0)..q(N) and v(1)..v(N).

Thus the old `summary.integral_minus_position = -0.001051786 rad` at 240 Hz
was **not on the same time interval** as the segmented result. It should not
be interpreted as an aggregate physics residual.

Recomputed directly from unrounded 240 Hz evidence:

| Same interval | Value |
| --- | ---: |
| q(0) | 0.2862764596939087 rad |
| q(480) | 0.2862755060195923 rad |
| measured q(480)-q(0) | -0.0000009536743164 rad |
| sum(v(1)..v(480))*dt | -0.0001959276961543 rad |
| correct integral minus position | -0.0001949740218379 rad |

The net displacement is tiny because an outbound step is followed by a
return step. The useful dynamic tests are the independent, non-overlapping
segments, on which the previous GPU evidence showed:
- outbound error = -0.0003088493678888 rad on ~+0.1030 rad travel
- return error = +0.0001138753460509 rad on ~-0.1030 rad travel.

These results support **good integration consistency on joint2 in this
specific unloaded step trajectory**, but do not certify all six joints,
high-speed hitting, contact forces, or angular velocity for closed-loop
safety use.

## Author-side code correction

- `badminton_scene.velocity_accounting.summarize_velocity_window` is
  pure Python; q(0) is explicit, each v(1)..v(N) is integrated once, and
  outbound/return segments tile the same window without overlap.
- `probe_rs_velocity_channels.py` now delegates to it instead of
  independently computing conflicting net and segment statistics.
- CPU regressions reproduce the first-step omission and test the segment
  boundaries. The harness fails if the segment sums do not equal net stats.
- `check_arm_rest_pose_hold.py` records the intended PhysX config, marked
  explicitly as **configuration metadata, not solver readback**.

## Next runner action

DeepSeek must execute this review SHA without changing source, run CPU
`test_velocity_accounting.py` and two step probes (240 / 480 Hz with
1s settle + 2s sample). Return raw JSON/logs in a new evidence branch,
including `summary.accounting`, `summary.segments`, and full-window
integral mismatch. Run the 5-second single-env scene hold if available
to verify the newly recorded config provenance.

No need to rerun 128-env performance climb for this statistics-only change.
Do not start PPO or run the real robot.
