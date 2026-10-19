# -*- coding: utf-8 -*-
"""Assembly-route policy for the badminton scene. Pure stdlib: importable without Isaac Sim or torch.

MultiEnvArm supports 128 environments, but that is a separate research assembly. create_scene is the
formal robot scene and still assembles the B601-RS with num_envs == 1 only (the InteractiveScene drive
defect is worked around, not solved there). Asking create_scene for more must raise, not degrade.
"""
from __future__ import annotations

ARMS = ("rebot_b601_rs", "piper")


def resolve_scene_arm_route(*, include_robot: bool, arm_model: str, num_envs: int) -> str:
    """Return "direct" (plain Articulation, verified) or "scene_registered" (InteractiveScene)."""
    direct = bool(include_robot) and str(arm_model) == "rebot_b601_rs"
    if direct and int(num_envs) != 1:
        raise NotImplementedError(
            "create_scene supports the B601-RS with num_envs == 1 only (see MULTIENV_ASSEMBLY.md); "
            "multi-env research goes through MultiEnvArm, not the formal scene. Got num_envs=%d"
            % int(num_envs))
    return "direct" if direct else "scene_registered"


__all__ = ["resolve_scene_arm_route", "ARMS"]
