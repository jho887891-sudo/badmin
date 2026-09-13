# -*- coding: utf-8 -*-
"""Camera rig placeholder geometry (no sensors/render) - ENGINEERING_V0_1."""
from .scene_layout import CAMERA

RIG = dict(
    mast_height=CAMERA["rig_height"],
    mast_radius=0.02,
    arm_y=CAMERA["baseline"] / 2.0 + 0.03,
    cam_box=(0.05, 0.05, 0.05),
)
LEFT = dict(y=+CAMERA["left_y"])
RIGHT = dict(y=-CAMERA["right_y"])
