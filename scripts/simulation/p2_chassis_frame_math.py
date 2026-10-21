"""Explicit parent-child frame arithmetic for Morph One chassis collision proxies.

In nested USD transforms a wheel child position is relative to Chassis, not world.
Lift must be applied exactly once at the Chassis parent. Geometry sizes are metres.
"""
from __future__ import annotations

import math


def wheel_bottom_world_z(lift_m: float, wheel_local_z_m: float = -0.244,
                         wheel_half_height_m: float = 0.090) -> float:
    values = (lift_m, wheel_local_z_m, wheel_half_height_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite frame geometry")
    if wheel_half_height_m <= 0:
        raise ValueError("wheel half-height must be positive")
    return lift_m + wheel_local_z_m - wheel_half_height_m


def historical_nested_wheel_bottom_world_z(
    lift_m: float,
    chassis_bbox_center_z_m: float = -0.0404,
    wheel_local_z_m: float = -0.244,
    wheel_half_height_m: float = 0.090,
) -> float:
    """Reconstruct the now-superseded RC authoring from commit 459cb12.

    Chassis xform=(chassis_bbox_center_z+lift) and child wheel xform=
    (wheel_local_z+lift), causing two lifts in the composed world pose.
    This is diagnostic only and must never be used as an authoring function.
    """
    if not all(math.isfinite(v) for v in (
        lift_m, chassis_bbox_center_z_m, wheel_local_z_m, wheel_half_height_m
    )):
        raise ValueError("nonfinite frame geometry")
    return (chassis_bbox_center_z_m + lift_m +
            wheel_local_z_m + lift_m - wheel_half_height_m)
