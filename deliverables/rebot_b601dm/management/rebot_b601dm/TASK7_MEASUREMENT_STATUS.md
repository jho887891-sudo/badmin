# Task 7 status: the bound and the harness are delivered, the physical measurement is not

Date: 2026-09-17  |  Candidate arm: reBot B601-DM (test bench, see ARM_IDENTITY_DECISION.md)

## What is done and tested

| Item | State |
|---|---|
| The speed bound, as executable assertions | **done** - `src/simulation/rebot_b601dm/speed_bound.py`, 21 tests |
| The duty-cycle guard | **done** - a constructor-validated object that refuses an unsustainable cycle |
| The experiment record | **done** - carries variant, convention, duty cycle, bound, and every known bias |
| The measurement harness | **done** - `scripts/simulation/measure_racket_speed.py`, 6 tests |
| **The physical measurement** | **NOT COMPLETED** - see below |

The bound, from the real motors and the specified 767 mm reach:

| Case | Bound |
|---|---|
| Shipped asset permitted | **153.4 m/s** |
| Real, 100% of no-load | **16.06 m/s** |
| Official 70% guidance | **11.24 m/s** |

## The finding that shapes this task

Reading the upstream receiver showed it applies incoming commands with `set_joint_positions` (line 371), which
**teleports** the joints. In that mode the arm tracks whatever it is told regardless of torque, so the speed it
achieves is a property of the command, not of the motors. It is the right tool for replaying a real arm and the
wrong tool for asking what the arm can do. The harness therefore drives the PhysX drives instead.

## Why the measurement did not complete, and why I stopped

The host was saturated by work belonging to other users, and the constraints for this project forbid
disturbing it. Measured at the time of the attempt:

| Observation | Value |
|---|---|
| load average | **10.21 / 8.81 / 6.10** |
| heaviest process, not mine | `pt_main_thread` at **585% CPU** |
| GPU utilisation | **88%**, 20.7 GB in use by another job |
| my Isaac Sim after 1,506 s | **3 extensions started**, one `carb.tasking is likely stuck` warning |
| the same app on an idle host earlier | dozens of extensions by 500 s |

Isaac Sim startup on this host measured **23.6 minutes** when the machine was free. Under this load it had made
a fraction of that progress after twenty-five minutes. **Competing for the GPU would have slowed another user's
training to make my own number arrive sooner, which is not a trade this work is allowed to make.**

My process was stopped cleanly: no measure or kit processes remain, and the other users' `pt_main_thread` jobs
are untouched at their original CPU usage.

## What the measurement would still not tell you, even when it runs

Recording this now so the number is not over-read later. Three biases, all in the same direction:

1. **The arm is modelled at 61% of its real mass** (Task 4), so lighter distal links accelerate faster.
2. **The racket contributes no inertia at all**, because its mass and inertia are UNKNOWN (Task 6).
3. **The measurement is the flange speed, not the racket head.** The racket adds a moment arm beyond the
   flange, so the head moves faster; racket length is a parameter, not an assumption.

And the bound it is compared against is `omega * reach`, which ignores joint coupling and is therefore itself
an over-estimate.

**So both sides of the comparison are generous, and the only claim the measurement can support is whether the
result stays UNDER the bound.** A speed above it cannot come from these motors and means the asset is still
wrong - which is why `experiment_record` flags that case as an asset fault rather than reporting it as a fast
arm.

## What would complete it

One of:

- **a quieter window on the host**, when the load average is near 1 and the GPU is free; the harness is ready
  and needs no changes, only the ~24 minutes of basically uncontended app launch; or
- **a decision to accept the contention**, run it anyway, and treat the longer wall time as expected. The
  measurement itself takes 6 seconds once the app is up; it is entirely the launch that is contended.

Either way the record it writes goes through `experiment_record`, so the number arrives with its variant,
torque convention, duty cycle, bound and biases attached.
