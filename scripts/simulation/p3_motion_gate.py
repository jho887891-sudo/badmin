"""Three-dimensional, signed displacement checks for P3 diagnostic mount profiles.

A non-zero +Z command must not be accepted just because chassis X/Y moved;
the original P3 harness checked only hypot(command.x, command.y).
The implementation is dependency-free so it can be negative-tested without Isaac.
"""
from __future__ import annotations

import math


def directional_displacement_gate(command_xyz, start_xyz, end_xyz, ratio=0.9):
    """Check signed achieved displacement along the 3D commanded direction.

    Returns None for a zero command, because static profiles must be evaluated
    by their separate static pose/hold gates. Values are in metres.
    """
    if len(command_xyz) != 3 or len(start_xyz) != 3 or len(end_xyz) != 3:
        raise ValueError("all vectors must contain exactly three XYZ components")
    vals = [float(v) for vec in (command_xyz, start_xyz, end_xyz) for v in vec]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("nonfinite displacement geometry")
    if not math.isfinite(float(ratio)) or not (0.0 < ratio <= 1.0):
        raise ValueError("invalid motion ratio")
    cmd = vals[:3]
    start = vals[3:6]
    end = vals[6:9]
    commanded_m = math.sqrt(sum(v * v for v in cmd))
    if commanded_m <= 1e-12:
        return None
    actual = [a - b for a, b in zip(end, start)]
    achieved_along_m = sum(c * v for c, v in zip(cmd, actual)) / commanded_m
    displacement_norm = math.sqrt(sum(v * v for v in actual))
    cross_track_m = math.sqrt(max(0.0, displacement_norm**2 - achieved_along_m**2))
    return {
        "commanded_displacement_m": commanded_m,
        "achieved_along_command_m": achieved_along_m,
        "achieved_ratio": achieved_along_m / commanded_m,
        "cross_track_m": cross_track_m,
        "pass": bool(achieved_along_m >= ratio * commanded_m),
    }
