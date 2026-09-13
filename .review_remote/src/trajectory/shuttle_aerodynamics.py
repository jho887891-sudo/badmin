#!/usr/bin/env python3
"""Quadratic-drag badminton shuttlecock flight model.

Robot Brain v0.1 task model:

    p_dot = v
    v_dot = g - k * ||v - w|| * (v - w)

The default project parameter is k = 1 / L with the literature reference
feather-shuttle aerodynamic length L = 6.5 m.  This module intentionally does
not add orientation-dependent lift or restoring torque; those require a separate
higher-fidelity model and measured coefficients.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np

ArrayLike3 = Iterable[float] | np.ndarray


def _vec3(value: ArrayLike3, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def k_from_aerodynamic_length(length_m: float) -> float:
    """Return quadratic-drag acceleration coefficient k [1/m] from L [m]."""
    length_m = float(length_m)
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("aerodynamic length must be finite and > 0")
    return 1.0 / length_m


def terminal_speed(k_per_m: float, gravity_magnitude_mps2: float = 9.80665) -> float:
    """Terminal speed for vertical fall under ``a = g - k |v| v``."""
    k_per_m = float(k_per_m)
    g = float(gravity_magnitude_mps2)
    if not math.isfinite(k_per_m) or k_per_m <= 0.0:
        raise ValueError("k_per_m must be finite and > 0")
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError("gravity magnitude must be finite and > 0")
    return math.sqrt(g / k_per_m)


def acceleration(
    position: ArrayLike3,
    velocity: ArrayLike3,
    *,
    k_per_m: float,
    gravity: ArrayLike3 = (0.0, 0.0, -9.80665),
    wind: ArrayLike3 = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Evaluate translational acceleration in Court Frame.

    ``position`` is accepted for a stable ODE interface although v0.1 drag has
    no explicit position dependence.
    """
    _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    g = _vec3(gravity, "gravity")
    w = _vec3(wind, "wind")
    k = float(k_per_m)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("k_per_m must be finite and >= 0")

    relative = v - w
    speed = float(np.linalg.norm(relative))
    if speed == 0.0 or k == 0.0:
        return g.copy()
    return g - k * speed * relative


def _derivative(
    state: np.ndarray,
    *,
    k_per_m: float,
    gravity: np.ndarray,
    wind: np.ndarray,
) -> np.ndarray:
    p = state[:3]
    v = state[3:]
    return np.concatenate((v, acceleration(p, v, k_per_m=k_per_m, gravity=gravity, wind=wind)))


def rk4_step(
    position: ArrayLike3,
    velocity: ArrayLike3,
    dt_s: float,
    *,
    k_per_m: float,
    gravity: ArrayLike3 = (0.0, 0.0, -9.80665),
    wind: ArrayLike3 = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance one fixed RK4 step."""
    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    g = _vec3(gravity, "gravity")
    w = _vec3(wind, "wind")
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and > 0")
    k = float(k_per_m)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("k_per_m must be finite and >= 0")

    y = np.concatenate((p, v))
    k1 = _derivative(y, k_per_m=k, gravity=g, wind=w)
    k2 = _derivative(y + 0.5 * dt * k1, k_per_m=k, gravity=g, wind=w)
    k3 = _derivative(y + 0.5 * dt * k2, k_per_m=k, gravity=g, wind=w)
    k4 = _derivative(y + dt * k3, k_per_m=k, gravity=g, wind=w)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if not np.isfinite(y_next).all():
        raise FloatingPointError("non-finite shuttle state produced by RK4")
    return y_next[:3], y_next[3:]


def rollout(
    position: ArrayLike3,
    velocity: ArrayLike3,
    *,
    duration_s: float,
    dt_s: float,
    k_per_m: float,
    gravity: ArrayLike3 = (0.0, 0.0, -9.80665),
    wind: ArrayLike3 = (0.0, 0.0, 0.0),
) -> Dict[str, np.ndarray]:
    """Roll out a deterministic trajectory including t=0 and the final time.

    The final integration step is shortened when ``duration_s`` is not an exact
    multiple of ``dt_s`` so time never overshoots the requested horizon.
    """
    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    g = _vec3(gravity, "gravity")
    w = _vec3(wind, "wind")
    duration = float(duration_s)
    dt = float(dt_s)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and >= 0")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and > 0")

    times = [0.0]
    positions = [p.copy()]
    velocities = [v.copy()]
    t = 0.0
    eps = max(1e-15, duration * 1e-15)
    while t + eps < duration:
        h = min(dt, duration - t)
        p, v = rk4_step(p, v, h, k_per_m=k_per_m, gravity=g, wind=w)
        t += h
        times.append(t)
        positions.append(p.copy())
        velocities.append(v.copy())

    return {
        "time": np.asarray(times, dtype=float),
        "position": np.asarray(positions, dtype=float),
        "velocity": np.asarray(velocities, dtype=float),
    }
