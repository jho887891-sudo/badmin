"""A torque-limited swing, measured at the flange.

WHY THIS IS NOT THE UPSTREAM RECEIVER. Reading isaacsim_joint_receiver.py showed that it applies commands
with set_joint_positions (line 371), which TELEPORTS the joints. In that mode the arm tracks whatever it is
told regardless of torque, so the speed it achieves is a property of the command, not of the motors. It is the
right tool for replaying a real arm and the wrong tool for asking what the arm can do.

This script therefore drives the joints through their PHYSX DRIVES with a torque limit, letting the arm
accelerate only as fast as the motors allow, and measures the resulting flange speed.

WHAT IT MEASURES, AND WHAT IT DOES NOT.
  - The speed is the flange (gripper_link) speed, not the racket head. The racket adds a moment arm beyond the
    flange, so the racket head moves FASTER; the racket length is a parameter, not a hidden assumption.
  - The result is an OVERESTIMATE of the real arm, for two reasons already recorded: the modelled arm is 61%
    of the real mass (motor_mass.py), so it accelerates faster; and the racket mass and inertia are UNKNOWN
    (racket.py), so the racket contributes no inertia at all.
  - The bound it is compared against is omega * reach from the real motors, which is itself an over-estimate
    because it ignores joint coupling.

Both the number and the bound are therefore generous, and the comparison that matters is whether the measured
speed stays UNDER the bound. A speed above it cannot come from these motors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.rebot_b601dm.speed_bound import (  # noqa: E402
    DEFAULT_DUTY_CYCLE,
    bound_at_fraction,
    experiment_record,
    limit_variant_factor,
)

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# A generous ceiling used only to make the ramp visible: the point is to observe how fast the joint can reach
# its velocity target under a torque limit, not to stress the drive model. Recorded with the result.
ABSOLUTE_TORQUE_LIMIT_NM = 27.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Torque-limited swing measurement")
    p.add_argument("--asset", required=True, help="patched USD tree entry point")
    p.add_argument("--variant", default="real", choices=["real", "recommended_70"])
    p.add_argument("--torque-convention", default="rated", choices=["peak", "rated"])
    p.add_argument("--target-joint", default="joint6", help="the joint driven to its limit")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--hz", type=float, default=120.0)
    p.add_argument("--out", default=None, help="where to write the JSON record")
    return p.parse_args(argv)


def drive_and_measure(args) -> dict:
    """Open Isaac Sim, drive one joint to its torque limit, and sample the flange speed.

    Isaac Sim is imported INSIDE this function so that the module can be imported, and its record logic tested,
    without paying the 20-minute app launch this host takes.
    """
    import numpy as np
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import UsdPhysics

    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(str(args.asset), "/World/reBot")
    world.scene.add_default_ground_plane()

    robot = SingleArticulation(prim_path="/World/reBot", name="rebot")
    world.scene.add(robot)
    world.reset()

    dof_names = list(robot.dof_names)
    indices = [dof_names.index(name) for name in ARM_JOINTS if name in dof_names]

    factor = limit_variant_factor(args.variant)
    from src.simulation.rebot_b601dm.joint_limits import JOINT_LIMITS

    # Drive the chosen joint at the speed the patched asset enforces, and let the torque limits decide how
    # quickly it can get there.
    target_index = dof_names.index(args.target_joint)
    target_rate = JOINT_LIMITS[args.target_joint].velocity.value * factor

    stage = get_current_stage()
    drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/reBot"), "angular")

    # DRIVE THE JOINT, DO NOT TELEPORT IT.
    #
    # The first version set the joint positions directly and then zeroed the velocities before stepping, and
    # read the velocities back afterwards. That measured a joint it had just told to stop: the commanded rate
    # was 20.94 rad/s and the "achieved" rate came back at 0.086 rad/s, 244 times smaller, with a peak flange
    # speed of 0.066 m/s. The record was written and it passed its own bound trivially, because the number was
    # near zero. It was not a measurement of anything.
    #
    # What is measured here instead: the joint is given a VELOCITY TARGET and a torque ceiling, and the
    # simulation integrates what actually happens. The achieved velocity is then a consequence of the motors
    # against the arm's own inertia, which is the question worth asking.
    torque = [0.0] * len(dof_names)
    torque[target_index] = ABSOLUTE_TORQUE_LIMIT_NM
    samples = []
    period = 1.0 / args.hz
    deadline = time.time() + args.seconds
    t0 = time.time()
    while time.time() < deadline:
        targets = [0.0] * len(dof_names)
        targets[target_index] = target_rate
        robot.set_joint_velocity_targets(targets)
        robot.set_joint_efforts(torque)
        world.step(render=False)

        vel = robot.get_joint_velocities()
        reached = float(vel[target_index])
        tip = reached * 0.767  # reach; see speed_bound.REACH_M
        samples.append({"t": time.time() - t0, "joint_rate": reached, "flange_m_s": tip})
        time.sleep(period)

    peak_joint = max(abs(s["joint_rate"]) for s in samples) if samples else 0.0
    peak_flange = max(abs(s["flange_m_s"]) for s in samples) if samples else 0.0
    # A near-zero result means the drive was not actually driven, which is what the broken first version
    # produced. Refusing to write a record in that case is better than writing one that passes its bound for
    # the wrong reason.
    if peak_joint < 0.05 * abs(target_rate):
        raise RuntimeError(
            "the joint reached only {:.4f} rad/s against a target of {:.2f} rad/s, under 5 percent. That is "
            "not a measurement of the arm; it means the drive was not driven. No record written.".format(
                peak_joint, target_rate
            )
        )

    measured = {
        "samples": len(samples),
        "peak_joint_rate_rad_s": peak_joint,
        "peak_flange_m_s": peak_flange,
        "commanded_rate_rad_s": target_rate,
        "target_joint": args.target_joint,
        "torque_ceiling_nm": ABSOLUTE_TORQUE_LIMIT_NM,
        "joints_seen": dof_names,
    }

    # EVERYTHING THAT MUST SURVIVE IS FINISHED BEFORE close().
    #
    # SimulationApp.close() shuts down Kit and the process ends with it, so code placed after it never runs.
    # The first real run of this script started the app successfully, ran the swing, closed, exited rc=0, and
    # wrote no record at all for exactly that reason - the record was built after the close.
    record = experiment_record(
        limit_variant=args.variant,
        torque_convention=args.torque_convention,
        duty_cycle=DEFAULT_DUTY_CYCLE,
        measured_peak_m_s=peak_flange,
    )
    record["measurement"] = measured
    record["note"] = (
        "flange speed, not racket head. The racket adds a moment arm beyond the flange, so the racket head "
        "moves faster; racket length and mass are parameters, not assumptions."
    )
    text = json.dumps(record, indent=2)
    print(text[:2500], flush=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("[swing] record written to " + str(args.out), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    app.close()
    return record


def main(argv=None) -> int:
    args = parse_args(argv)
    print("[swing] asset   : " + str(args.asset), flush=True)
    print("[swing] variant : " + args.variant, flush=True)
    print("[swing] duty    : " + str(DEFAULT_DUTY_CYCLE), flush=True)
    print("[swing] starting Isaac Sim; this host has taken ~24 minutes", flush=True)
    drive_and_measure(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
