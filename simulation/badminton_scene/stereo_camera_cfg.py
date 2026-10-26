# -*- coding: utf-8 -*-
"""Single-environment Isaac Lab RGB-D stereo camera configuration.

All pose values come from scene_layout.CAMERA and remain engineering placeholders.
They are not hardware-measured extrinsics.
"""
from __future__ import annotations

import math
from typing import Any

from .scene_layout import CAMERA


def build_stereo_camera_sensor_specs(
    *, width: int = 1280, height: int = 720, update_period: float = 1.0 / 60.0
) -> list[tuple[str, Any, type]]:
    """Build extra_sensors entries for scene_cfg.create_scene (N=1 only).

    Both sensors produce RGB and distance-to-image-plane depth. The prims attach
    to env_0, whose origin is the court origin for the single-environment route.
    """
    if int(width) < 32 or int(height) < 32:
        raise ValueError("camera width and height must each be at least 32")
    if float(update_period) < 0.0:
        raise ValueError("update_period must be non-negative")

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    rig_world = tuple(float(v) for v in CAMERA["rig_world"])
    pitch_rad = -math.radians(float(CAMERA["pitch_deg"]))
    rotation_xyzw = (
        0.0,
        math.sin(pitch_rad / 2.0),
        0.0,
        math.cos(pitch_rad / 2.0),
    )
    horizontal_aperture = 20.955
    vertical_aperture = horizontal_aperture * float(height) / float(width)
    focal_length = 24.0

    def make_cfg(name: str, y: float):
        return CameraCfg(
            prim_path="/World/envs/env_0/" + name,
            update_period=float(update_period),
            height=int(height),
            width=int(width),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture,
                clipping_range=(0.05, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(rig_world[0], float(y), rig_world[2]),
                rot=rotation_xyzw,
                convention="world",
            ),
        )

    # Use the signed layout values directly. Do not negate right_y again.
    left_cfg = make_cfg("CameraLeft", float(CAMERA["left_y"]))
    right_cfg = make_cfg("CameraRight", float(CAMERA["right_y"]))
    return [
        ("camera_left", left_cfg, Camera),
        ("camera_right", right_cfg, Camera),
    ]


__all__ = ["build_stereo_camera_sensor_specs"]
