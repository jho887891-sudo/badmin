# -*- coding: utf-8 -*-
"""Robot assembly config - reBot B601-RS mounted on the Morph One placeholder.

ENGINEERING_V0_1:
  robot_base world = (-1.60, 0, 0); heading +X (yaw 0)
  Morph placeholder static visual; the arm keeps a fixed base on the chassis top plate.

ARM IDENTITY (2026-10 migration)
  Formal arm: reBot B601-RS, converted from the official Seeed URDF by Isaac Lab
  (see ASSET in src/simulation/rebot_b601_rs/joint_limits.py). The PiPER assembly is still reachable
  with BADMINTON_ARM_MODEL=piper for historical reproduction only; it is no longer the default and
  its numbers must never be reused as B601-RS values.
"""
import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .scene_layout import MORPH_PLACEHOLDER, PIPER, REBOT_RS

_PIPER_SCRIPTS = "/home/T7/ojh/robot_sim/simulation/piper_stage0"
_PIPER_USD = "/home/T7/ojh/robot_sim/assets/piper_stage0/assets/piper_with_racket.usd"


def _rebot_rs_articulation_cfg() -> ArticulationCfg:
    """Formal arm: 6 revolute + 2 prismatic fingers, base fixed to the chassis mount."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RebotB601RS",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(REBOT_RS["usd"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                # The Isaac Lab URDF conversion already authored a fixed root_joint (--fix-base).
                # Asking the spawner to fix it again fails: it looks for the first rigid body under
                # the articulation root and finds the /Geometry container instead.
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(REBOT_RS["mount_world"]),
            rot=(1.0, 0.0, 0.0, 0.0),  # yaw 0 facing +X
            joint_pos=dict(REBOT_RS["rest_joint_pos"]),
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=float(REBOT_RS["stiffness"]),
                damping=float(REBOT_RS["damping"]),
            )
        },
    )


def _piper_articulation_cfg() -> ArticulationCfg:
    """HISTORICAL: frozen Stage-0 PiPER assembly, kept for pre-migration reproduction."""
    if _PIPER_SCRIPTS not in sys.path:
        sys.path.insert(0, _PIPER_SCRIPTS)
    from piper_cfg import PIPER_CFG  # noqa: E402

    cfg = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/PiPER")
    p = PIPER["mount_world"]
    cfg.init_state.pos = (_p[0], _p[1], _p[2])
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # yaw 0 facing +X
    cfg.init_state.joint_pos = dict(PIPER["rest_joint_pos"])
    cfg.spawn.usd_path = _PIPER_USD
    return cfg


ARM_MODEL = os.environ.get("BADMINTON_ARM_MODEL", "rebot_b601_rs").strip().lower()
if ARM_MODEL not in ("rebot_b601_rs", "piper"):
    raise ValueError(
        f"BADMINTON_ARM_MODEL={ARM_MODEL!r} is not an arm this project has models for "
        "(expected 'rebot_b601_rs' or 'piper')"
    )

ROBOT_ARTICULATION_CFG: ArticulationCfg = (
    _piper_articulation_cfg() if ARM_MODEL == "piper" else _rebot_rs_articulation_cfg()
)

MORPH = MORPH_PLACEHOLDER
