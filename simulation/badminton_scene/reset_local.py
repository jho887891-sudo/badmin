# -*- coding: utf-8 -*-
"""Per-env reset helpers (spec 28) - IsaacLab v3 API.

write_root_*_to_sim take WORLD coordinates: local pose must be offset by the
environment origin (scene.env_origins). Velocity is translation-invariant.
"""
import torch

CANONICAL_SPAWN = (1.20, 0.00, 1.80)
REST_JOINT = {"joint1": 0.0, "joint2": 1.5, "joint3": -1.5, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}


def reset_idx(env_ids, robot, shuttle, env_origins=None, states=None, joint_pos=None):
    """Reset only requested envs (robot joints + shuttle full state, world frame)."""
    n = env_ids.numel()
    if n == 0:
        return
    dev = shuttle.device
    if joint_pos is None:
        jp = torch.zeros((n, robot.num_joints), device=dev)
        for k, name in enumerate(robot.joint_names):
            jp[:, k] = REST_JOINT.get(name, 0.0)
    else:
        jp = joint_pos
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    if states is None:
        local = torch.tensor([CANONICAL_SPAWN], device=dev).repeat(n, 1).float()
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(n, 1)
        vel = torch.zeros((n, 3), device=dev)
        ang = torch.zeros((n, 3), device=dev)
    else:
        local, quat, vel, ang = states
    pos = local.clone()
    if env_origins is not None:
        pos = pos + env_origins[env_ids]
    shuttle.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    shuttle.write_root_velocity_to_sim(torch.cat([vel, ang], dim=-1), env_ids=env_ids)

