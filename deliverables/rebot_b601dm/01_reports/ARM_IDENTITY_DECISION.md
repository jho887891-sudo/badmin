# Arm identity decision: reBot B601-DM is a test bench / candidate arm

Date: 2026-09-17  |  Decided by: project owner  |  Status: **IN FORCE**

## Decision

**The reBot B601-DM does not take over the PiPER arm's formal project identity.** It is evaluated as an
independent **test bench / candidate arm**. The badminton swing and dynamic interception capability of the
B601-DM is verified on its own, and only if that verification passes does the question of formally replacing
PiPER get decided.

## What that constrains, concretely

| Area | Consequence |
|---|---|
| `docs/simulation/BADMINTON_ROBOT.md` | **Unchanged.** Its racket attachment at `PiPER link6` (line 169) and its whole-body contract stay as they are. |
| This work's outputs | Live under `rebot_b601dm/` namespaces only; nothing is written into the PiPER module paths |
| Robot State / Command interfaces | Not redefined. If B601-DM later replaces PiPER, that contract is re-checked as a separate piece of work |
| Results | Advisory. A passing swing test is evidence for a future decision, not a change of project state |

## Why this ordering is the right one

The inspection found the shipped simulation asset enforces a joint speed limit **9.09x and 9.55x the real
motor speeds** (`physiosJoint:maxJointVelocity` 2864.789 and 11459.156 deg/s against DM4340 52.5 rpm and
DM4310 200 rpm). Until that is corrected, no swing measurement means anything - for either arm. Evaluating
the candidate first, on corrected limits, avoids committing the project's formal architecture to numbers that
have not yet been shown to be real.

## Promotion criteria, stated in advance

The B601-DM is considered for promotion to the formal project arm only when all of the following hold:

1. The loaded USD carries real joint limits, verified by re-reading the asset after load.
2. A measured racket-head speed exists that is **inside** the bound derived from the real motors, with the
   torque convention, limits variant (100% or 70%) and duty cycle all recorded.
3. The official thermal constraint is respected: the published testing terminated every run on motor-2
   overheating, worst case a 1.5 kg hover at full reach lasting 3 minutes.
4. The remaining `UNKNOWN` values are listed, and none of them is load-bearing for the capability claim.

Criteria 1 to 4 are the Task 8 gates in `docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md`.
They are stated here, before the work, so that promotion is decided against a target rather than a result.
