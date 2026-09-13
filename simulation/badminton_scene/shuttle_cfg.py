# -*- coding: utf-8 -*-
"""SimpleShuttle v0.1 rigid object config - no aerodynamics."""
from .scene_layout import SHUTTLE
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

SHUTTLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shuttle",
    spawn=sim_utils.SphereCfg(
        radius=SHUTTLE["physics_collision_radius"],
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=5.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=SHUTTLE["mass"]),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.3)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30)),
)
