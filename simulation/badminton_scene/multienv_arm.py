# -*- coding: utf-8 -*-
"""Multi-environment B601-RS assembly built on plain Articulations (the verified F1 path).

WHY THIS EXISTS
    Registering the converted B601-RS through InteractiveScene does not give the joints usable drive
    authority: the staged position targets are correct (inspection shows _joint_pos_target_sim
    matching the command) and the targets reach the PhysX view, yet the arm does not follow them,
    while the same asset bound as a plain Articulation tracks to a few milliradians. See
    management/rebot_b601_rs/ARM_MIGRATION_RECORD.md (F1 and the multi-env section).

    Until that is understood upstream, environments are assembled the way that is verified to work:
    one plain Articulation per env, all of them driven and refreshed by this class.

Constraint: nothing here relaxes a joint limit or a safety value; the limits come from the asset and
are only read.
"""
from __future__ import annotations

from typing import Iterable, Sequence
from .assembly_route import resolve_scene_arm_route  # noqa: F401  (re-exported for callers)
from .rs_physics import rebot_rs_simulation_cfg


def _torch():
    """Lazy torch import: this module must stay importable where torch is absent (pure unit tests)."""
    import torch

    return torch


class MultiEnvArm:
    """N independently targetable B601-RS arms, one plain Articulation per environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        env_spacing: float = 15.0,
        mount_world: Sequence[float] = (-1.60, 0.0, 0.30),
        rest_joint_pos: dict | None = None,
        stiffness: float = 100.0,
        damping: float = 1.0,
        physics_dt: float = 1.0 / 240.0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.env_spacing = float(env_spacing)
        self.mount_world = tuple(float(v) for v in mount_world)
        self.rest_joint_pos = dict(rest_joint_pos or {})
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.physics_dt = float(physics_dt)
        self.robots: list = []
        self.joint_names: list[str] = []
        self.device = None

    # -- construction -------------------------------------------------------
    @staticmethod
    def prim_path(env_id: int) -> str:
        return "/World/envs/env_%d/Robot/RebotB601RS" % env_id

    def env_origin(self, env_id: int) -> tuple[float, float, float]:
        return (env_id * self.env_spacing + self.mount_world[0],
                self.mount_world[1], self.mount_world[2])

    def spawn(self) -> None:
        """Create the simulation ground and one articulation per env. Call before sim.reset()."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from .robot_cfg import rebot_rs_spawn_cfg

        # The gains are OWNED HERE, not taken from the scene defaults: a sweep that silently reused
        # the scene values would prove nothing (that bug was caught by three identical sweep runs).
        actuators = {"all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=self.stiffness, damping=self.damping)}

        # Matches the verified single-arm TGS fix. Do not silently use Isaac Sim defaults.
        self.sim = sim_utils.SimulationContext(
            rebot_rs_simulation_cfg(dt=self.physics_dt))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        light = sim_utils.DistantLightCfg(intensity=700.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light, translation=(0.0, 0.0, 10.0))
        for env_id in range(self.num_envs):
            spawn = rebot_rs_spawn_cfg()
            spawn.func(self.prim_path(env_id), spawn, translation=self.env_origin(env_id))
            self.robots.append(Articulation(ArticulationCfg(
                prim_path=self.prim_path(env_id), spawn=None, actuators=actuators)))
        self.sim.reset()
        self.joint_names = list(self.robots[0].joint_names)
        self.device = self.robots[0].device
        # every env starts at the rest pose, but each one is commanded on its own afterwards
        self.reset(range(self.num_envs))

    # -- control ------------------------------------------------------------
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def _rest_tensor(self, env_ids: Iterable[int]) -> torch.Tensor:
        rows = [[self.rest_joint_pos.get(nm, 0.0) for nm in self.joint_names] for _ in env_ids]
        torch = _torch()
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def reset(self, env_ids: Iterable[int]) -> None:
        """Write the rest pose into exactly these envs; nothing else is touched."""
        env_ids = list(env_ids)
        if not env_ids:
            return
        for env_id in env_ids:
            robot = self.robots[env_id]
            target = self._rest_tensor([env_id])
            robot.write_joint_state_to_sim(target, _torch().zeros_like(target))
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()

    def set_targets(self, targets: torch.Tensor) -> None:
        """Set a per-env target. Shape [num_envs, num_joints]."""
        if tuple(targets.shape) != (self.num_envs, self.num_joints):
            raise ValueError("targets must be [%d, %d], got %s"
                             % (self.num_envs, self.num_joints, tuple(targets.shape)))
        self._targets = targets.to(self.device)
        for env_id, robot in enumerate(self.robots):
            robot.set_joint_position_target(self._targets[env_id:env_id + 1])

    def step(self) -> None:
        for robot in self.robots:
            robot.write_data_to_sim()
        self.sim.step()
        dt = self.sim.get_physics_dt()
        for robot in self.robots:
            robot.update(dt)

    # -- observation --------------------------------------------------------
    def joint_pos(self) -> torch.Tensor:
        return _torch().cat([r.data.joint_pos for r in self.robots], dim=0)

    def joint_vel(self) -> torch.Tensor:
        return _torch().cat([r.data.joint_vel for r in self.robots], dim=0)

    def joint_limits(self) -> list[list[float]]:
        return [list(map(float, lim)) for lim in self.robots[0].data.joint_limits[0].cpu().tolist()]

    def ee_world_z(self, body: str = "link6") -> list[float]:
        out = []
        for robot in self.robots:
            idx = list(robot.body_names).index(body)
            out.append(float(robot.data.body_pos_w[0, idx, 2]))
        return out



__all__ = ["MultiEnvArm", "resolve_scene_arm_route"]
