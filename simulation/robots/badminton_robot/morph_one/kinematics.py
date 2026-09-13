# -*- coding: utf-8 -*-
"""Morph One four-steer / four-drive kinematics (Phase 3).

Spec: BADMINTON_ROBOT.md S10 (inverse kinematics), S11 (steer optimisation).
Topology is frozen (four steer + four drive); only numeric values may be TEMP.

Model: each wheel module i sits at r_i = (x_i, y_i) in robot_base.  For a commanded
chassis twist xi = [vx, vy, wz] the wheel-centre plane velocity is
        v_i = [vx - wz*y_i, vy + wz*x_i]
so the target steer angle is theta_i = atan2(v_iy, v_ix), the tangential speed is
        s_i = |v_i| and the wheel rate is omega_i = s_i / r_w.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from ..badminton_robot_cfg import WheelId

WHEEL_ORDER = (WheelId.FL, WheelId.FR, WheelId.RL, WheelId.RR)


def normalise_angle(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a <= -math.pi + 1e-15 else a


@dataclass(frozen=True)
class WheelTargets:
    """Per-wheel command for STEER_DRIVE_WHEEL_MODEL."""
    steer_angle_rad: float
    wheel_speed_rad_s: float
    tangential_speed_mps: float = 0.0


def _validated(wheel_positions: Mapping[WheelId, Sequence[float]], wheel_radius_m: float, ) -> None:
    if not math.isfinite(float(wheel_radius_m)) or float(wheel_radius_m) <= 0.0:
        raise ValueError("wheel_radius_m must be finite and > 0")
    for wid in WHEEL_ORDER:
        if wid not in wheel_positions:
            raise KeyError(f"wheel_positions is missing {wid.value}")
        pos = wheel_positions[wid]
        if len(pos) != 2:
            raise ValueError(f"wheel_positions[{wid.value}] must be (x, y)")


def body_twist_to_wheel_targets(
    twist: Sequence[float],
    wheel_positions: Mapping[WheelId, Sequence[float]],
    wheel_radius_m: float,
    *,
    current_steer_rad: Optional[Mapping[WheelId, float]] = None,
) -> Dict[WheelId, WheelTargets]:
    """S10 inverse kinematics with the S11 minimum-steering solution."""
    _validated(wheel_positions, wheel_radius_m)
    vx, vy, wz = [float(v) for v in twist]
    if not all(math.isfinite(v) for v in (vx, vy, wz)):
        raise ValueError("twist must be finite")
    rw = float(wheel_radius_m)

    out: Dict[WheelId, WheelTargets] = {}
    for wid in WHEEL_ORDER:
        xi, yi = [float(v) for v in wheel_positions[wid]]
        vix = vx - wz * yi
        viy = vy + wz * xi
        speed_mps = math.hypot(vix, viy)
        current = None if current_steer_rad is None else current_steer_rad.get(wid)

        if speed_mps < 1e-12:
            # No wheel motion requested: hold the current steering angle (no fake rotation).
            theta = 0.0 if current is None else normalise_angle(float(current))
            out[wid] = WheelTargets(theta, 0.0, 0.0)
            continue

        theta = math.atan2(viy, vix)
        omega = speed_mps / rw

        if current is not None:
            delta = normalise_angle(theta - float(current))
            if abs(delta) > math.pi / 2.0:
                theta = normalise_angle(theta + math.pi)
                omega = -omega
        out[wid] = WheelTargets(normalise_angle(theta), omega, speed_mps)
    return out


def wheel_targets_to_body_twist(
    targets: Mapping[WheelId, WheelTargets],
    wheel_positions: Mapping[WheelId, Sequence[float]],
    wheel_radius_m: float,
) -> np.ndarray:
    """Least-squares recovery of the chassis twist from wheel targets (S51 mixed-motion check)."""
    _validated(wheel_positions, wheel_radius_m)
    rw = float(wheel_radius_m)
    rows, rhs = [], []
    for wid in WHEEL_ORDER:
        if wid not in targets:
            raise KeyError(f"targets is missing {wid.value}")
        xi, yi = [float(v) for v in wheel_positions[wid]]
        t = targets[wid]
        # physics definition: the wheel tangential velocity carries the sign of the
        # wheel rate, so (theta, w) and (theta + pi, -w) reconstruct identically.
        vix = t.wheel_speed_rad_s * rw * math.cos(t.steer_angle_rad)
        viy = t.wheel_speed_rad_s * rw * math.sin(t.steer_angle_rad)
        rows.append([1.0, 0.0, -yi])
        rhs.append(vix)
        rows.append([0.0, 1.0, xi])
        rhs.append(viy)
    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


__all__ = ["WheelTargets", "WHEEL_ORDER", "normalise_angle",
           "body_twist_to_wheel_targets", "wheel_targets_to_body_twist"]
