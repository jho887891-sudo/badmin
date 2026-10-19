# -*- coding: utf-8 -*-
"""Read unrounded joint position and velocity from both Isaac Lab and PhysX.

This is a DIAGNOSTIC, not a test that assumes the source of the discrepancy.
It records the SAME robot in hold/step modes, with an optional zero-gravity scene.
Run in the installed Isaac Sim/Isaac Lab environment. No real robot is commanded.

Examples:
  python scripts/simulation/probe_rs_velocity_channels.py \
      --mode hold --dt 0.004166666666666667 --sample-steps 480 \
      --out outputs/simulation/rebot_b601_rs/review_probe_hold_240.json
  python scripts/simulation/probe_rs_velocity_channels.py \
      --mode step --gravity-off --dt 0.004166666666666667 \
      --out outputs/simulation/rebot_b601_rs/review_probe_step_zero_g_240.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "simulation"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("hold", "step"), default="hold")
parser.add_argument("--dt", type=float, default=1.0 / 240.0)
parser.add_argument("--gravity-off", action="store_true")
parser.add_argument("--solver", choices=("default", "tgs", "pgs"), default="default",
                    help="PhysX solver; default preserves the old script's configuration")
parser.add_argument("--external-forces-every-iteration", choices=("default", "on", "off"),
                    default="default",
                    help="TGS force application per position iteration; test on vs off at fixed dt")
parser.add_argument("--min-velocity-iterations", type=int, choices=(0, 1, 2, 4),
                    default=None, help="Optional PhysX minimum velocity iterations")
parser.add_argument("--settle-steps", type=int, default=240)
parser.add_argument("--sample-steps", type=int, default=480)
parser.add_argument("--step-amplitude", type=float, default=0.1)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (math.isfinite(args.dt) and 0 < args.dt < 0.1):
    parser.error("--dt must be finite and in (0, 0.1)")
if args.sample_steps < 4 or args.settle_steps < 0:
    parser.error("sample-steps >= 4 and settle-steps >= 0 are required")
if args.step_amplitude <= 0 or args.step_amplitude > 0.5:
    parser.error("step amplitude must be within (0, 0.5] rad")
if args.solver == "pgs" and args.external_forces_every_iteration != "default":
    parser.error("external-forces flag is ignored for PGS; use default when testing PGS")

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from badminton_scene.robot_cfg import (REBOT_RS_PRIM_PATH, rebot_rs_actuators,
                                        rebot_rs_spawn_cfg)
from badminton_scene.scene_layout import REBOT_RS
from badminton_scene.velocity_accounting import summarize_velocity_window


def flatten(values):
    """Do not round measured floats; support torch, warp.numpy and numpy buffers."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values).reshape(-1).astype(float).tolist()


asset = Path(REBOT_RS["usd"])
if not asset.is_file():
    raise FileNotFoundError("B601-RS USD not found: " + str(asset))
kwargs = {"dt": args.dt, "device": "cuda:0"}
if args.gravity_off:
    kwargs["gravity"] = (0.0, 0.0, 0.0)

# Isaac Lab 3.x: scene-level PhysX configuration lives in isaaclab_physx.physics.
# Override only the requested attributes. The `default` case preserves the
# original code path for a byte-for-byte baseline comparison.
solver_config = {
    "solver": args.solver,
    "external_forces_every_iteration": args.external_forces_every_iteration,
    "min_velocity_iterations": args.min_velocity_iterations,
}
if (args.solver != "default" or args.external_forces_every_iteration != "default"
        or args.min_velocity_iterations is not None):
    from isaaclab_physx.physics import PhysxCfg
    physics_kwargs = {}
    if args.solver != "default":
        physics_kwargs["solver_type"] = 1 if args.solver == "tgs" else 0
    if args.external_forces_every_iteration != "default":
        physics_kwargs["enable_external_forces_every_iteration"] = (
            args.external_forces_every_iteration == "on")
    if args.min_velocity_iterations is not None:
        physics_kwargs["min_velocity_iteration_count"] = args.min_velocity_iterations
        physics_kwargs["max_velocity_iteration_count"] = max(4, args.min_velocity_iterations)
    kwargs["physics"] = PhysxCfg(**physics_kwargs)
    solver_config["physics_kwargs"] = physics_kwargs
else:
    solver_config["physics_kwargs"] = None

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(**kwargs))
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/ground", ground)
spawn = rebot_rs_spawn_cfg()
spawn.func(REBOT_RS_PRIM_PATH, spawn, translation=tuple(REBOT_RS["mount_world"]))
robot = Articulation(ArticulationCfg(
    prim_path=REBOT_RS_PRIM_PATH, spawn=None, actuators=rebot_rs_actuators()))
sim.reset()
names = list(robot.joint_names)
assert names.count("joint2") == 1
joint2 = names.index("joint2")
rest = torch.tensor([[REBOT_RS["rest_joint_pos"].get(n, 0.0) for n in names]],
                    dtype=torch.float32, device=robot.device)
robot.write_joint_state_to_sim(rest, torch.zeros_like(rest))
robot.set_joint_position_target(rest)
robot.write_data_to_sim()


def step_once():
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())


def capture(step: int, target: torch.Tensor) -> dict:
    view = robot.root_physx_view
    row = {
        "step": step,
        "time_s": step * args.dt,
        "command": flatten(target),
        "lab_q": flatten(robot.data.joint_pos[0]),
        "lab_qd": flatten(robot.data.joint_vel[0]),
        "physx_q": flatten(view.get_dof_positions()[0]),
        "physx_qd": flatten(view.get_dof_velocities()[0]),
        "physx_target": flatten(view.get_dof_position_targets()[0]),
    }
    for key in ("lab_q", "lab_qd", "physx_q", "physx_qd"):
        if not all(math.isfinite(x) for x in row[key]):
            raise RuntimeError("Non-finite " + key + " at sample " + str(step))
    return row


for _ in range(args.settle_steps):
    step_once()

# Capture before applying the step; otherwise q[1]-q[0] is omitted from
# the integral comparison, biasing the moving-arm acceptance metric.
baseline = capture(0, rest)
samples = []
active = rest.clone()
for step in range(1, args.sample_steps + 1):
    if args.mode == "step":
        # Preserve a fixed root and official joint2 limits: all targets 0.3 or 0.4.
        if step == 1:
            active = rest.clone()
            active[0, joint2] += args.step_amplitude
            robot.set_joint_position_target(active)
        elif step == args.sample_steps // 2:
            active = rest.clone()
            robot.set_joint_position_target(active)
    step_once()
    samples.append(capture(step, active))

j = joint2
q = [r["physx_q"][j] for r in samples]
v = [r["physx_qd"][j] for r in samples]
# q0 is captured before the first command. All N velocity samples
# correspond to steps 1..N and must be included in the same interval.
switch_step = args.sample_steps // 2 if args.mode == "step" else None
summary = {
    "joint": "joint2",
    **summarize_velocity_window(
        q0=baseline["physx_q"][j],
        positions=q,
        velocities=v,
        dt=args.dt,
        switch_step=switch_step,
    ),
    "mean_physx_velocity": math.fsum(v) / len(v),
    "min_physx_velocity": min(v),
    "max_physx_velocity": max(v),
    "max_abs_lab_physx_q_difference": max(
        abs(r["lab_q"][j] - r["physx_q"][j]) for r in samples),
    "max_abs_lab_physx_qd_difference": max(
        abs(r["lab_qd"][j] - r["physx_qd"][j]) for r in samples),
}

report = {
    "script": "scripts/simulation/probe_rs_velocity_channels.py",
    "mode": args.mode,
    "gravity_off": args.gravity_off,
    "dt": args.dt,
    "solver_config": solver_config,
    "settle_steps": args.settle_steps,
    "sample_steps": args.sample_steps,
    "joint_names": names,
    "asset": str(asset),
    "summary": summary,
    "initial_sample": baseline,
    "samples": samples,
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print("REVIEW_VELOCITY_PROBE " + json.dumps({
    "out": str(out), "mode": args.mode, "gravity_off": args.gravity_off,
    "dt": args.dt, "solver_config": solver_config, "summary": summary,
}, allow_nan=False))
simulation_app.close()
