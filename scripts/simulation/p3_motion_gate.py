"""Three-dimensional, signed displacement checks for P3 diagnostic mount profiles.

A non-zero +Z command must not be accepted just because chassis X/Y moved; the original
P3 harness checked only hypot(command.x, command.y).

Ported from ChatGPT PR #31 (7056844 / 4a26828) and extended: that port compared only the
FIRST and LAST frame, so a profile that reaches the target mid-run and drifts back would be
scored on its final pose. `directed_displacement_track` projects EVERY frame onto the
commanded direction and takes the maximum — the directional equivalent of the previous
`realized_translation_max_m` diagnostic.

Dependency-free so the negative cases can be tested without Isaac.
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


def directed_displacement_track(command_xyz, positions, ratio=0.9):
    """Same gate scored on the MAX projection over every recorded frame.

    `positions` is the per-step chassis position list in metres. Using every frame keeps
    the "did the motion happen" semantics of the old norm-based diagnostic while still
    requiring the motion to be along the COMMANDED direction.
    """
    if len(command_xyz) != 3:
        raise ValueError("command must contain exactly three XYZ components")
    # numpy arrays must not be truth-tested (ambiguous); length is the only safe check.
    if len(positions) == 0:
        raise ValueError("no recorded positions")
    cmd = [float(v) for v in command_xyz]
    if not all(math.isfinite(v) for v in cmd):
        raise ValueError("nonfinite command")
    if not math.isfinite(float(ratio)) or not (0.0 < ratio <= 1.0):
        raise ValueError("invalid motion ratio")
    commanded_m = math.sqrt(sum(v * v for v in cmd))
    if commanded_m <= 1e-12:
        return None
    start = [float(v) for v in positions[0][:3]]
    if not all(math.isfinite(v) for v in start):
        raise ValueError("nonfinite start position")
    best = -math.inf
    best_frame = None
    for index, position in enumerate(positions):
        pos = [float(v) for v in position[:3]]
        if not all(math.isfinite(v) for v in pos):
            raise ValueError("nonfinite position at frame %d" % index)
        actual = [a - b for a, b in zip(pos, start)]
        projected = sum(c * v for c, v in zip(cmd, actual)) / commanded_m
        if projected > best:
            best = projected
            best_frame = index
    end = [float(v) for v in positions[-1][:3]]
    final_actual = [a - b for a, b in zip(end, start)]
    final_projected = sum(c * v for c, v in zip(cmd, final_actual)) / commanded_m
    displacement_norm = math.sqrt(sum(v * v for v in final_actual))
    cross_track_m = math.sqrt(max(0.0, displacement_norm**2 - final_projected**2))
    return {
        "commanded_displacement_m": commanded_m,
        "achieved_along_command_max_m": best,
        "achieved_along_command_final_m": final_projected,
        "best_frame": best_frame,
        "achieved_ratio": best / commanded_m,
        "final_ratio": final_projected / commanded_m,
        "cross_track_m": cross_track_m,
        "pass": bool(best >= ratio * commanded_m),
    }
