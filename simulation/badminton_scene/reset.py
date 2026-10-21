# -*- coding: utf-8 -*-
"""Per-env reset helpers (spec 28) - IsaacLab v3 API.

write_root_*_to_sim take WORLD coordinates: local pose must be offset by the
environment origin (scene.env_origins). Velocity is translation-invariant.

MODEL-AWARE REST POSE (P5B)
    reset_idx() used to build its default joint target from ONE table, copied from the PiPER arm,
    and write it whatever articulation the scene had loaded. On the formal reBot B601-RS that table
    is not reachable (its joint3 sits below the arm's lower stop), so every reset that did not pass
    joint_pos wrote a limit-violating state - and the in-tree callers never pass it. See
    management/morph_one_rs/P5_ARM_STATIC_CONSISTENCY.md (F1) and P5B_RESET_POSE.md.

    The pose now comes from badminton_scene.rest_pose: one table per robot, addressed by joint NAME
    (the USD traversal order is not the DOF order), and validated against that robot's limits. The
    loaded arm is identified from robot.joint_names; robot_key= overrides it when the caller already
    knows, and a contradiction raises like any other mismatch.

FAILURE MODE: RAISE, DO NOT RETURN
    reset_idx() returns None on the success path, so an error return would be ignorable and the
    caller (scripts/t06_reset_local.py, t07_local.py, t08_local.py) would keep stepping a sim whose
    arm was never reset. Everything below raises badminton_scene.rest_pose.ResetPoseError instead:
      * UnknownRobotModel         robot.joint_names match no known arm, or robot_key is unknown
      * UnknownJoint              the model table cannot cover the runtime DOF names
      * RestPoseLimitViolation    an explicit joint_pos falls outside that arm's limits
      * ResetPoseError            robot.num_joints disagrees with len(robot.joint_names), or the
                                  joint_pos column count disagrees with the runtime DOF count
    Nothing is written to the sim before the checks pass.
"""
import torch

from .rest_pose import (
    ResetPoseError,
    RestPoseLimitViolation,
    limits_for,
    resolve_model,
    rest_joint_pos,
)

CANONICAL_SPAWN = (1.20, 0.00, 1.80)


def reset_idx(env_ids, robot, shuttle, env_origins=None, states=None, joint_pos=None,
              robot_key=None):
    """Reset only requested envs (robot joints + shuttle full state, world frame).

    joint_pos=None (the only mode the in-tree scripts use) writes the loaded arm's own rest pose.
    An explicit joint_pos is written verbatim after being checked against that arm's limits, so a
    stale constant from another arm fails here instead of reaching PhysX.
    """
    n = env_ids.numel()
    if n == 0:
        return
    dev = shuttle.device
    joint_names = tuple(str(name) for name in robot.joint_names)
    model = resolve_model(joint_names, robot_key=robot_key)
    num_joints = getattr(robot, "num_joints", len(joint_names))
    if int(num_joints) != len(joint_names):
        raise ResetPoseError(
            "robot.num_joints=%d disagrees with the %d joint names %s: refusing to guess the "
            "layout of the joint state" % (int(num_joints), len(joint_names), list(joint_names)))

    if joint_pos is None:
        target = rest_joint_pos(model, joint_names)
        jp = torch.zeros((n, len(joint_names)), device=dev)
        for column, name in enumerate(joint_names):
            jp[:, column] = target[name]
    else:
        jp = joint_pos
        width = int(jp.shape[-1])
        if width != len(joint_names):
            raise ResetPoseError(
                "joint_pos has %d columns but the %s robot has %d joints %s"
                % (width, model, len(joint_names), list(joint_names)))
        _reject_out_of_limits(jp, joint_names, model, dev)

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


def _reject_out_of_limits(jp, joint_names, model, dev):
    """Raise if any column of an explicit joint_pos leaves its joint's limits.

    Inclusive: a caller may command the exact stop (that is a deliberate request, not a rest pose).
    The comparison runs on the device, so the CPU sync happens once per reset at most, and the
    per-element message is only built when a violation was actually found.
    """
    limits = limits_for(model)
    lower = [limits[name].lower for name in joint_names]
    upper = [limits[name].upper for name in joint_names]
    # NaN is the one value a range comparison cannot see: NaN < lo and NaN > hi are both False, so a
    # NaN target would be written and PhysX would propagate it through the whole articulation state.
    # The reflexive test (NaN != NaN) is the only portable way to catch it in one tensor op.
    not_a_number = jp != jp
    outside = ((jp < torch.tensor(lower, device=dev))
               | (jp > torch.tensor(upper, device=dev))
               | not_a_number)
    if not bool(outside.any()):
        return

    rows = jp.tolist()
    offenders = []
    for column, name in enumerate(joint_names):
        for row, values in enumerate(rows):
            value = float(values[column])
            if value != value:
                offenders.append(
                    "env %d %s = nan (not a finite number; limits [%r, %r] %s)"
                    % (row, name, lower[column], upper[column], limits[name].unit))
            elif value < lower[column] or value > upper[column]:
                offenders.append(
                    "env %d %s = %r outside [%r, %r] %s"
                    % (row, name, value, lower[column], upper[column], limits[name].unit))
    shown = offenders[:8]
    if len(offenders) > len(shown):
        shown.append("(+%d more)" % (len(offenders) - len(shown)))
    raise RestPoseLimitViolation(
        "reset_idx refuses to write a %s joint_pos outside its limits: %s"
        % (model, "; ".join(shown)))
