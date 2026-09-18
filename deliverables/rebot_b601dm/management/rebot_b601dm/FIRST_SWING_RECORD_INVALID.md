# The first swing record is INVALID, and this file says why

Date: 2026-09-17  |  Artifact: `outputs/simulation/rebot_b601dm/swing_record.json` (run 1)

## The record that was produced

```json
{ "measured_peak_m_s": 0.0659, "bound_m_s": 16.06, "inside_bound": true, "verdict": "inside the bound",
  "measurement": { "commanded_rate_rad_s": 20.94, "peak_joint_rate_rad_s": 0.086, "samples": 409 } }
```

**It reports a pass, and it is worthless.** It is kept on disk so the failure is auditable, and it must not be
cited as a capability figure.

## Why it is worthless

The joint was commanded at 20.94 rad/s and reached 0.086 rad/s - **244 times slower**. A joint does not fail to
move like that; the measurement was reading back something it had itself just zeroed:

```python
robot.set_joint_positions(positions)
robot.set_joint_velocities([0.0] * len(dof_names))   # zeroed the quantity about to be measured
world.step(render=False)
vel = robot.get_joint_velocities()                    # read the zero back
```

Peak flange speed 0.066 m/s is what a stationary arm reads. It passed `inside_bound` **because it was near
zero**, which is the most dangerous kind of passing result: the check ran, the assertion held, and the number
means nothing.

## What was changed

| Before | After |
|---|---|
| joint position teleported each step | joint given a **velocity target** plus a torque ceiling, and the simulation integrates what happens |
| velocities zeroed, then read | velocities never touched; the achieved rate is read as a consequence of the drive |
| any number written to the record | a result under 5% of the commanded rate **raises and writes nothing** |

That last row is the substantive fix. The old version would have written a near-zero record every time; the new
one refuses to, on the grounds that a joint which does not move means the drive was not driven rather than that
the arm is slow. Three tests pin the behaviour, including one that asserts the string which zeroed the
velocity is gone.

## What this says about the harness

Two of the three bugs found in this harness so far were of the same shape: **a result that looked like a
result.** The first was a record that was never written because it was built after `SimulationApp.close()`;
this one is a record that was written and passed a gate while measuring nothing. Both were caught by reading
the numbers rather than the exit code - `rc=0` in both cases.

## Status

The measurement has **not** been made. A corrected run needs another quiet window of about 30 minutes, and the
host was back at load 12 with the GPU at 100% when this was written. `BADMINTON_CAPABILITY_RESULT.md` still
records gate 3 as UNEVALUATED, which remains accurate.
