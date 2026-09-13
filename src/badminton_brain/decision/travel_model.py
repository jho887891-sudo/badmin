# -*- coding: utf-8 -*-
"""Base minimum-travel-time model - single source of truth for the decision layer.

Model: docs/architecture/04_HIT_DECISION.md S62-S64 (1-D minimum time under |v| <= v_max and
|a| <= a_max): triangle profile while d <= v_max^2/a_max, trapezoid profile beyond it.

Review ruling C3 (same-layer duplicate truth): this model used to exist twice - once in
feasibility.py (T5 gate) and once in intercept_search.py (T6 search), with nobody importing the
other.  It now lives here, and both modules import it:
  * feasibility.py imports min_travel_time_s (as base_min_travel_time_s) and min_travel_times_s;
  * intercept_search.py imports min_travel_time_s.
tests/badminton_brain/test_feasibility.py asserts the identity of all three names, so the copy
cannot come back.

Input contract (review finding C1): a non-finite distance must never be silently read as
"no travel needed", so NaN/Inf - and a non-finite or non-positive speed/acceleration limit -
raise ValueError instead of returning a plausible-looking time.  The sign of the distance is
ignored: a travel distance has no direction (S63/S64 use |d|).
"""
from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np


def _limits(v_max_mps: float, a_max_mps2: float) -> Tuple[float, float]:
    v_max = float(v_max_mps)
    a_max = float(a_max_mps2)
    if not (math.isfinite(v_max) and math.isfinite(a_max)):
        raise ValueError('v_max_mps and a_max_mps2 must be finite')
    if v_max <= 0.0 or a_max <= 0.0:
        raise ValueError('v_max_mps and a_max_mps2 must be > 0')
    return v_max, a_max


def min_travel_time_s(distance_m: float, v_max_mps: float, a_max_mps2: float) -> float:
    """Minimum travel time for one distance, triangle/trapezoid profile (S62-S64).

    Lower bound only: no jerk limit, no steering time, no path curvature - it is a gate and a
    search bound, never a motion plan.
    """
    distance = float(distance_m)
    v_max, a_max = _limits(v_max_mps, a_max_mps2)
    if not math.isfinite(distance):
        raise ValueError('distance_m must be finite, got %r' % (distance_m,))
    distance = abs(distance)
    if distance <= 0.0:
        return 0.0
    cruise_distance = v_max * v_max / a_max
    if distance <= cruise_distance:
        return 2.0 * math.sqrt(distance / a_max)
    return distance / v_max + v_max / a_max


def min_travel_times_s(distance_m: Any, v_max_mps: float, a_max_mps2: float) -> np.ndarray:
    """Vectorised min_travel_time_s for an array of distances (new array; input untouched)."""
    v_max, a_max = _limits(v_max_mps, a_max_mps2)
    distances = np.abs(np.asarray(distance_m, dtype=float))
    if not np.all(np.isfinite(distances)):
        raise ValueError('distance_m must be finite')
    cruise_distance = v_max * v_max / a_max
    return np.where(distances <= cruise_distance,
                    2.0 * np.sqrt(distances / a_max),
                    distances / v_max + v_max / a_max)


__all__ = ["min_travel_time_s", "min_travel_times_s"]
