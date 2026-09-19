# Gate 3 is blocked: the host is up but its SSH daemon is not answering

Date: 2026-09-17  |  Status: **measurement not run**; gate 3 stays UNEVALUATED

## The blocker

| Check | Result |
|---|---|
| ping 172.31.68.251 | **3 of 3 replies, 4 ms** - the machine is alive on the network |
| TCP port 22 | **refused** |
| ssh, retried | connection timed out |

Earlier in the same session a command was terminated with "Connection to 172.31.68.251 closed by remote host",
and every attempt since has failed. The host is running; its sshd is not accepting connections.

**Nothing was done about it.** Restarting another machine's services is not this work's to do, and the project
constraints forbid interfering with other users.

## What was about to run, and why it had to be checked first

The corrected measurement was ready, but the previous attempt had already failed for a reason that had nothing
to do with the host:

```
AttributeError: SingleArticulation object has no attribute set_joint_velocity_targets
```

That method was written from memory. It does not exist. It was the second time in this task that an API was
assumed rather than checked - the first was `cv2.imwrite` silently failing on a non-ASCII path, and before that
a `Path.resolve()` diagnosis that measurement disproved.

**So the next step is deliberately NOT another guess.** scripts/simulation/probe_articulation_api.sh prints:

1. the public methods of `SingleArticulation` whose names mention velocity, effort, target or action, read
   from the installed source file rather than recalled;
2. the same for the multi-articulation class, which is where velocity targets usually live;
3. whether `ArticulationAction` and `apply_action` exist, the documented route for physics-driven control;
4. **what the upstream receiver itself calls** - `set_joint_positions`, `set_joint_velocities`,
   `get_joint_positions`, `get_joint_velocities` - which is known to work because it ran.

The last of those is the strongest lead: the receiver drives a real robot with this class, so whatever it uses
is a working route, and the measurement should follow it rather than invent a third one.

## What is not in doubt

The torque bug fixed before this outage stands on its own and is independent of the API question: the script
applied a hardcoded 27 N m to a DM4310 whose rated torque is 3 and whose peak is 7, while recording
"torque_convention": "rated". Whatever the correct method turns out to be, the torque it is given must come
from the declared convention, and five tests now enforce that.

## Current state

| | |
|---|---|
| gate 1, asset honesty | PASS |
| gate 2, provenance | PASS |
| gate 3, physical plausibility | **UNEVALUATED** - blocked, and previously also buggy |
| the record on disk | still the invalid one from 15:53; FIRST_SWING_RECORD_INVALID.md says so |
| delivery package | regenerated with the torque fix; packaged suite 72 passed, 62 skipped |

## To resume

```bash
ssh <host> bash scripts/simulation/probe_articulation_api.sh   # once ssh answers
# fix the drive call against what it prints, add a test, rebuild the package
bash scripts/simulation/wait_for_quiet_window.sh              # then wait for the GPU
```
