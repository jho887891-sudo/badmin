# -*- coding: utf-8 -*-
"""Physics ODE trajectory prediction (ROBOT_BRAIN.md 4.3 / S11 / S12, plan task T4).

Responsibility (and nothing else): roll the estimated shuttle state forward and publish a
:class:`~badminton_brain.types.PredictedTrajectory`.  This module never decides whether to
hit, never plans motion and never touches Safety (ROBOT_BRAIN.md S12 "Prediction != Decision").

Model and provenance
--------------------
The physics is NOT implemented here.  Every sample comes from
``src/trajectory/shuttle_aerodynamics.py`` -- the frozen quadratic-drag model
``p_dot = v`, ``v_dot = g - k ||v - w|| (v - w)`` integrated with fixed-step RK4.  Calling that
``rollout()`` (instead of copying the ODE) is what keeps Simulation and Robot Brain on one
model and what makes the 1e-9 agreement acceptance criterion meaningful.

The quadratic-drag coefficient is ``k = 1 / L`` with the literature feather-shuttle
aerodynamic length ``L = 6.5 m``; ``L`` and ``k`` are exposed as :class:`~common.status.Param`
with an explicit source (BADMINTON_ROBOT.md S4 authenticity levels / S12 TEMP policy).
Home-hall airflow is not measured yet, so wind is a declared TEMP proxy rather than a
silently assumed zero.

Landing point
-------------
``landing_point`` / ``arrival_time`` are the first downward crossing of the court ground plane
``z = ground_z_m`` (Court Frame, ROBOT_BRAIN.md S3), obtained by linear interpolation between
the two rollout samples that straddle the plane.  If the shuttle is airborne for the whole
horizon the crossing does not exist yet: the module then reports the last sample projected
onto the ground plus the horizon time and sets the additive diagnostic flag
``landed_within_horizon[env] = False``, so a caller can never mistake a horizon-clipped
extrapolation for a real bounce.  (The type contract has no field for that flag and the
frozen contracts must not change, hence an extra attribute.)

Usage
-----
```python
predictor = PhysicsTrajectoryPredictor(horizon_s=0.6, dt_s=0.005)   # configurable horizon
trajectory = predictor.process(unified_state)                        # (N, T, 3) + landing
```
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # the frozen aerodynamics model lives outside the brain package
    from trajectory.shuttle_aerodynamics import k_from_aerodynamic_length, rollout
except ImportError:  # pragma: no cover - only when 'src' is not already on sys.path
    _SRC_DIR = Path(__file__).resolve().parents[2]
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from trajectory.shuttle_aerodynamics import k_from_aerodynamic_length, rollout

from ..interfaces import PredictionModule
from ..status import AssetStatus, Param
from ..types import BrainBoundaryError, Layer, PredictedTrajectory, UnifiedState

# ---------------------------------------------------------------------------
# Declared model parameters (BADMINTON_ROBOT.md S4: no value may hide its origin)
# ---------------------------------------------------------------------------

AERODYNAMIC_LENGTH_M = Param(
    value=6.5,
    status=AssetStatus.TRACEABLE_REFERENCE,
    source=("Feather-shuttle aerodynamic length L = 6.5 m from Darbois-Texier, Cohen, "
            "Le Bozec, Guerin & Clanet, 'Flight of the feathered shuttlecock', "
            "New Journal of Physics 14 (2012) 115004 (k = 1/L law); the same literature value "
            "is already the project model default in src/trajectory/shuttle_aerodynamics.py. "
            "It is a published reference, not a measurement of our own shuttle batch: replace "
            "with a bench-measured L/k when those data exist (see adaptation module)."),
)

DRAG_K_PER_M = Param(
    value=k_from_aerodynamic_length(6.5),
    status=AssetStatus.TRACEABLE_REFERENCE,
    source=("k = 1 / L derived by src/trajectory/shuttle_aerodynamics.k_from_aerodynamic_length "
            "from the literature L = 6.5 m [Darbois-Texier et al. 2012] -- identical to the "
            "project default coefficient, not an independently measured drag constant."),
)

GRAVITY_MPS2 = Param(
    value=(0.0, 0.0, -9.80665),
    status=AssetStatus.TRACEABLE_REFERENCE,
    source=("Standard gravity 9.80665 m/s^2, the default gravity vector of "
            "src/trajectory/shuttle_aerodynamics.py; shared model constant, not a bench value."),
)

WIND_MPS = Param(
    value=(0.0, 0.0, 0.0),
    status=AssetStatus.TEMP_PARAMETERIZED_PROXY,
    source=("Home-hall airflow REQUIRES_MEASUREMENT (no anemometry yet); zero wind is the "
            "declared TEMP proxy because it is also the aerodynamics module default. "
            "Replace with the measured hall wind / the UKF wind state."),
)

# ROBOT_BRAIN.md 4.3: the predictor rolls the state forward "about 0.3-0.8 s".
DEFAULT_HORIZON_S = 0.6
DEFAULT_DT_S = 0.005


def _finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _batched3(value: Any, name: str) -> np.ndarray:
    """Accept (3,) or (N, 3) and always return (N, 3)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise BrainBoundaryError(f"{name} must be (3,) or (N, 3), got {arr.shape}")
    if arr.shape[0] < 1:
        raise BrainBoundaryError(f"{name} must have at least one batch element")
    if not np.isfinite(arr).all():
        raise BrainBoundaryError(f"{name} contains NaN/Inf")
    return arr


def _as_param(value: Any, default: Param, name: str) -> Param:
    if value is None:
        return default
    if isinstance(value, Param):
        return value
    return Param(value=value, status=AssetStatus.TEMP_PARAMETERIZED_PROXY,
                 source=f"caller-supplied {name} override (provenance is the caller's, "
                        "e.g. the Shuttle UKF drag estimate)")


def _ground_crossing(times: np.ndarray, positions: np.ndarray, ground_z_m: float
                     ) -> Tuple[np.ndarray, float, bool]:
    """First downward crossing of the ground plane, by linear interpolation between samples."""
    z = positions[:, 2]
    downward = np.nonzero((z[1:] <= ground_z_m) & (z[:-1] > ground_z_m))[0]
    if downward.size:
        i = int(downward[0]) + 1
        span = z[i] - z[i - 1]
        frac = 0.0 if span == 0.0 else float((ground_z_m - z[i - 1]) / span)
        frac = min(max(frac, 0.0), 1.0)
        point = positions[i - 1] + frac * (positions[i] - positions[i - 1])
        arrival = float(times[i - 1] + frac * (times[i] - times[i - 1]))
        return point, arrival, True

    # Still airborne at the end of the horizon: report a horizon-clipped projection and say so.
    point = positions[-1].copy()
    point[2] = ground_z_m
    return point, float(times[-1]), False


class PhysicsTrajectoryPredictor(PredictionModule):
    """UnifiedState -> PredictedTrajectory via the frozen shuttle aerodynamics rollout."""

    layer = Layer.PREDICTION
    name = 'physics_predictor'
    is_implemented = True

    def __init__(self, *, horizon_s: float = DEFAULT_HORIZON_S, dt_s: float = DEFAULT_DT_S,
                 aerodynamic_length_m: Any = None, k_per_m: Any = None,
                 gravity_mps2: Any = None, wind_mps: Any = None,
                 ground_z_m: float = 0.0) -> None:
        self.horizon_s = _finite_float(horizon_s, 'horizon_s')
        if self.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be > 0, got {self.horizon_s}")
        self.dt_s = _finite_float(dt_s, 'dt_s')
        if self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")
        if self.dt_s > self.horizon_s:
            raise ValueError(
                f"dt_s ({self.dt_s}) must not exceed horizon_s ({self.horizon_s}): "
                "at least one integration step must fit the horizon")
        self.ground_z_m = _finite_float(ground_z_m, 'ground_z_m')

        if k_per_m is not None and aerodynamic_length_m is not None:
            raise ValueError("give either k_per_m or aerodynamic_length_m, not both")

        self.aerodynamic_length_m = _as_param(aerodynamic_length_m, AERODYNAMIC_LENGTH_M,
                                              'aerodynamic_length_m')
        if k_per_m is not None:
            self.k_per_m = _as_param(k_per_m, DRAG_K_PER_M, 'k_per_m')
        else:
            length = _finite_float(self.aerodynamic_length_m.value, 'aerodynamic_length_m.value')
            if length <= 0.0:
                raise ValueError(f"aerodynamic_length_m must be > 0, got {length}")
            # k is derived by the frozen aerodynamics module, never recomputed here.
            self.k_per_m = Param(value=k_from_aerodynamic_length(length),
                                 status=self.aerodynamic_length_m.status,
                                 source=f"k = 1 / L with L = {length} m; "
                                        f"{self.aerodynamic_length_m.source}")
        if _finite_float(self.k_per_m.value, 'k_per_m.value') < 0.0:
            raise ValueError(f"k_per_m must be >= 0, got {self.k_per_m.value}")

        self.gravity_mps2 = _as_param(gravity_mps2, GRAVITY_MPS2, 'gravity_mps2')
        self.wind_mps = _as_param(wind_mps, WIND_MPS, 'wind_mps')
        _vec3(self.gravity_mps2.value, 'gravity_mps2.value')
        _vec3(self.wind_mps.value, 'wind_mps.value')

    # -- provenance ---------------------------------------------------------
    def params(self) -> Dict[str, Param]:
        """Every declared model parameter, with its status and source (S4/S12)."""
        return {
            'aerodynamic_length_m': self.aerodynamic_length_m,
            'k_per_m': self.k_per_m,
            'gravity_mps2': self.gravity_mps2,
            'wind_mps': self.wind_mps,
        }

    # -- prediction ---------------------------------------------------------
    def process(self, state: UnifiedState) -> PredictedTrajectory:
        """Prediction-layer entry point: one UnifiedState -> one PredictedTrajectory."""
        if not isinstance(state, UnifiedState):
            raise BrainBoundaryError(
                f"{type(self).__name__}.process expects a UnifiedState, got {type(state).__name__}")
        if state.shuttle_position is None or state.shuttle_velocity is None:
            raise BrainBoundaryError(
                'physics_predictor requires shuttle_position and shuttle_velocity on the '
                'UnifiedState (estimate the shuttle before predicting it)')
        return self.predict(state.shuttle_position, state.shuttle_velocity,
                            timestamp=state.timestamp)

    def predict(self, position: Any, velocity: Any, *, k_per_m: Optional[float] = None,
                horizon_s: Optional[float] = None, dt_s: Optional[float] = None,
                timestamp: float = 0.0) -> PredictedTrajectory:
        """Roll out (3,) or (N, 3) shuttle states; identical maths to rollout() per sample."""
        pos = _batched3(position, 'position')
        vel = _batched3(velocity, 'velocity')
        if pos.shape[0] != vel.shape[0]:
            raise BrainBoundaryError(
                f"position and velocity batch sizes differ: {pos.shape[0]} vs {vel.shape[0]}")

        horizon = self.horizon_s if horizon_s is None else _finite_float(horizon_s, 'horizon_s')
        if horizon <= 0.0:
            raise ValueError(f"horizon_s must be > 0, got {horizon}")
        step = self.dt_s if dt_s is None else _finite_float(dt_s, 'dt_s')
        if step <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {step}")
        k = float(self.k_per_m.value) if k_per_m is None else _finite_float(k_per_m, 'k_per_m')
        gravity = _vec3(self.gravity_mps2.value, 'gravity_mps2.value')
        wind = _vec3(self.wind_mps.value, 'wind_mps.value')

        times: Optional[np.ndarray] = None
        positions, velocities, landings, arrivals, landed_flags = [], [], [], [], []
        for index in range(pos.shape[0]):
            # The single source of truth for shuttle flight: the frozen aerodynamics rollout.
            sample = rollout(pos[index], vel[index], duration_s=horizon, dt_s=step,
                             k_per_m=k, gravity=gravity, wind=wind)
            sample_times = np.asarray(sample['time'], dtype=float)
            if times is None:
                times = sample_times
            elif sample_times.shape != times.shape or not np.array_equal(sample_times, times):
                raise BrainBoundaryError(
                    'shuttle_aerodynamics.rollout returned inconsistent time grids across the '
                    'batch; PredictedTrajectory.times is shared by every environment')
            sample_position = np.asarray(sample['position'], dtype=float)
            point, arrival, landed = _ground_crossing(sample_times, sample_position,
                                                      self.ground_z_m)
            positions.append(sample_position)
            velocities.append(np.asarray(sample['velocity'], dtype=float))
            landings.append(point)
            arrivals.append(arrival)
            landed_flags.append(landed)

        trajectory = PredictedTrajectory(
            times=times,
            position=np.stack(positions, axis=0),
            velocity=np.stack(velocities, axis=0),
            landing_point=np.stack(landings, axis=0),
            arrival_time=np.asarray(arrivals, dtype=float),
            timestamp=float(timestamp),
        )
        # Additive diagnostic (the frozen contract has no field for it): False means the
        # shuttle never reached the ground inside this horizon and landing_point/arrival_time
        # are horizon-clipped projections, not a predicted bounce.
        trajectory.landed_within_horizon = np.asarray(landed_flags, dtype=bool)
        return trajectory


__all__ = ["PhysicsTrajectoryPredictor", "AERODYNAMIC_LENGTH_M", "DRAG_K_PER_M",
           "GRAVITY_MPS2", "WIND_MPS", "DEFAULT_HORIZON_S", "DEFAULT_DT_S"]
