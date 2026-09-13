# -*- coding: utf-8 -*-
"""T6 - intercept point search: PredictedTrajectory (+UnifiedState) -> BestIntercept.

Layer: decision (ROBOT_BRAIN.md S11/S12 - "feasibility gate + intercept search, never motion").
Spec: docs/architecture/04_HIT_DECISION.md
  * S26-S39  geometric candidate intervals + candidate interpolation (cubic Hermite uses p *and* v)
  * S54-S75  cheap whole-body reachability: workspace box + base triangle/trapezoid minimum time
  * S99-S115 soft score; the full component breakdown is recorded on every candidate
  * S41-S48  strike pose from the racket_contact convention (+X = face normal, +Z = grip -> head)
  * S27      the search walks the *whole* future trajectory - never a fixed x strike plane
Plan row: docs/superpowers/plans/2026-09-13-brain-modules.md T6 - "earliest feasible intercept on
the predicted trajectory; deterministic ordering; no magic offset".

Algorithm (one pass, all of it deterministic)
--------------------------------------------
1. candidate times: a uniform arithmetic grid over ``PredictedTrajectory.times`` (S33-S34);
2. candidate state: cubic Hermite interpolation of position *and* velocity (S36-S37), so a
   candidate may fall between two prediction samples;
3. hard gates, evaluated in this fixed order (the first failure is the recorded reason):
      t > arrival_time            -> OUT_OF_BOUNDS      (the shuttle has already landed)
      z outside the racket window -> NO_GEOMETRIC_WINDOW
      T_available <= 0            -> NO_TIME_MARGIN
      T_required >= T_available   -> UNREACHABLE        (base travel + arm slew, S70-S75)
      no legal racket pose        -> NO_ORIENTATION_SOLUTION
   with T_available = t_hit - now - decision_latency - safety_margin (S70-S73) and
        T_required  = max(T_base_min, T_arm_slew_min)              (S74: base and arm run in
   parallel, so they are *not* added);
4. soft score (S99): score = w_t S_time + w_h S_height + w_v S_speed + w_m S_margin
                              - w_b C_base, every term in [0, 1], all components kept on the
   candidate as :class:`ScoreTerms` - the score is a pure function of that breakdown;
5. selection: maximum score, ties broken by the earliest time and then by the lexicographically
   smallest position (S115 without hysteresis, which is stateful and out of scope for T6).

Determinism (acceptance criterion)
----------------------------------
Pure function of its inputs: no RNG, no wall clock, no unordered iteration, no iterative solver.
The candidate grid is arithmetic, the racket roll is fixed by the documented up-projection rule,
and every quaternion is canonicalised to w >= 0, so two identical calls return bit-identical
arrays (tests/badminton_brain/test_intercept_search.py asserts np.array_equal).

Time base
----------
04_HIT_DECISION.md S6 requires ``trajectory_points[].absolute_time``.  A prediction module that
publishes a grid starting at 0 (the frozen aerodynamics rollout does) is detected by comparing
the grid origin with ``PredictedTrajectory.timestamp`` and then measured from the prediction
instant instead - see :func:`_resolve_now`.  Passing ``now`` explicitly always wins, and a base
that lies after the whole grid is refused loudly rather than silently rejecting every candidate.

What is deliberately NOT here (honest scope, S46-S59 / S78-S82)
-------------------------------------------------------------
* no PiPER IK, no WorkspaceMap, no manipulability: reachability is the cheap box + base
  travel-time bound of S54/S61-S66, not a reachability oracle (T7 does IK in planning);
* no racket roll search (S50-S52) - a roll set only means something once IK exists;
* no impact-model normal seed (S46-S49): the face normal is the flat-block normal n = -v_hat,
  which is exactly the ideal mirror normal of S48 for a shuttle returned straight back - no
  restitution or deflection constant is invented;
* no effective contact offset: the racket contact point is the shuttle position (S43).  The real
  offset is carried as a REQUIRES_MEASUREMENT Param instead of being guessed;
* the achievable face-normal cone of the real PiPER + racket mount is not measured yet, so it is
  carried as REQUIRES_MEASUREMENT as well (measurement_requirements()).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..status import UNRESOLVED_STATUSES, AssetStatus, Param
from ..types import BestIntercept, BrainBoundaryError, PredictedTrajectory, UnifiedState

_TEMP = AssetStatus.TEMP_PARAMETERIZED_PROXY
_MEASURE = AssetStatus.REQUIRES_MEASUREMENT

# Numerical guards only - none of them is a physical threshold (S4: no invented constant).
_SPEED_EPS_MPS = 1e-12
_DEGENERATE_EPS = 1e-12
_TIME_EPS_S = 1e-9
_UP = np.array([0.0, 0.0, 1.0])
_COURT_X = np.array([1.0, 0.0, 0.0])


class InterceptReason(str, Enum):
    """Reason vocabulary of 04_HIT_DECISION.md S14 (decision layer codes) + NOT_IMPLEMENTED.

    NOT_IMPLEMENTED is not produced by this module: it is the code a DecisionModule reports when no
    search is run at all (disabled / policy slot), so the vocabulary stays complete in one place.
    """
    BEST_INTERCEPT_FOUND = 'BEST_INTERCEPT_FOUND'
    INVALID_STATE = 'INVALID_STATE'
    INVALID_PREDICTION = 'INVALID_PREDICTION'
    NO_GEOMETRIC_WINDOW = 'NO_GEOMETRIC_WINDOW'
    OUT_OF_BOUNDS = 'OUT_OF_BOUNDS'
    UNREACHABLE = 'UNREACHABLE'
    NO_TIME_MARGIN = 'NO_TIME_MARGIN'
    NO_ORIENTATION_SOLUTION = 'NO_ORIENTATION_SOLUTION'
    NOT_IMPLEMENTED = 'NOT_IMPLEMENTED'


@dataclass(frozen=True)
class ScoreWeights:
    """Soft-score weights (S99).  Plain floats: the Param wrappers live in the config."""
    time: float = 0.0
    height: float = 0.0
    speed: float = 0.0
    margin: float = 0.0
    base_cost: float = 0.0


@dataclass(frozen=True)
class ScoreTerms:
    """Full score breakdown of one candidate (S51: record every component).

    time   S_time   = sigmoid((T_available - T_required) / tau_time)                    (S101)
    height S_height = exp(-(z - z_pref)^2 / (2 sigma_z^2))                             (S102)
    speed  S_speed  = exp(-(||v|| - v_pref)^2 / (2 sigma_v^2))                         (S103)
    margin S_margin = min over the workspace-box axes of the residual margin, in [0, 1] (S100)
    base   C_base   = min(1, ||required base displacement|| / d_ref)                   (S107)
    """
    time: float = 0.0
    height: float = 0.0
    speed: float = 0.0
    margin: float = 0.0
    base_cost: float = 0.0

    def total(self, weights: ScoreWeights) -> float:
        """The documented soft score - the only place a candidate score is computed."""
        return (weights.time * self.time + weights.height * self.height
                + weights.speed * self.speed + weights.margin * self.margin
                - weights.base_cost * self.base_cost)

    def as_pairs(self) -> Tuple[Tuple[str, float], ...]:
        return (('time', self.time), ('height', self.height), ('speed', self.speed),
                ('margin', self.margin), ('base_cost', self.base_cost))


@dataclass(frozen=True)
class InterceptSearchConfig:
    """Every number the search uses, with its authenticity status (S4/S12).

    The values that describe the robot mirror the proxies already declared in
    ``decision/feasibility.py`` (T5) so the two decision modules cannot drift apart; they stay
    TEMP_PARAMETERIZED_PROXY until the real Morph One / PiPER measurements exist.
    """
    base_v_max_mps: Param = field(default_factory=lambda: _temp(
        1.0, 'TEMP proxy (same as decision/feasibility.py): Morph One linear speed limit'))
    base_a_max_mps2: Param = field(default_factory=lambda: _temp(
        1.5, 'TEMP proxy (same as decision/feasibility.py): Morph One acceleration limit'))
    arm_reach_x_m: Param = field(default_factory=lambda: _temp(
        0.60, 'TEMP proxy (same as decision/feasibility.py): racket box half extent in X'))
    arm_reach_y_m: Param = field(default_factory=lambda: _temp(
        0.60, 'TEMP proxy (same as decision/feasibility.py): racket box half extent in Y'))
    arm_z_min_m: Param = field(default_factory=lambda: _temp(
        0.15, 'TEMP proxy (same as decision/feasibility.py): lowest racket contact height'))
    arm_z_max_m: Param = field(default_factory=lambda: _temp(
        1.40, 'TEMP proxy (same as decision/feasibility.py): highest racket contact height'))
    arm_slew_time_s: Param = field(default_factory=lambda: _temp(
        0.15, 'TEMP proxy (same as decision/feasibility.py): PiPER slew-to-strike lower bound'))
    decision_latency_s: Param = field(default_factory=lambda: _temp(
        0.02, 'TEMP proxy (same as decision/feasibility.py): decision -> command budget'))
    safety_time_margin_s: Param = field(default_factory=lambda: _temp(
        0.05, 'TEMP proxy (same as decision/feasibility.py): robustness margin'))
    candidate_dt_s: Param = field(default_factory=lambda: _temp(
        0.01, 'TEMP proxy: coarse candidate step of the intercept grid (S34); replace with the '
              'measured control period once the planning/control rate is fixed'))
    z_pref_m: Param = field(default_factory=lambda: _temp(
        1.00, 'TEMP proxy: comfortable racket contact height for the low-speed rig (S102)'))
    z_sigma_m: Param = field(default_factory=lambda: _temp(
        0.30, 'TEMP proxy: width of the comfortable contact-height band (S102)'))
    speed_pref_mps: Param = field(default_factory=lambda: _temp(
        5.0, 'TEMP proxy: preferred shuttle speed at contact; the project favours controllable '
             'low-speed exchanges (S103)'))
    speed_sigma_mps: Param = field(default_factory=lambda: _temp(
        3.0, 'TEMP proxy: tolerance around the preferred contact speed (S103)'))
    base_travel_ref_m: Param = field(default_factory=lambda: _temp(
        1.0, 'TEMP proxy: normalisation distance of the base-travel cost (S107)'))
    time_margin_tau_s: Param = field(default_factory=lambda: _temp(
        0.10, 'TEMP proxy: time scale of the margin sigmoid (S101)'))
    min_face_alignment: Param = field(default_factory=lambda: _temp(
        0.5, 'TEMP proxy: an edge-on racket face (|d_in . n| below this = 60 deg off the '
             'incoming velocity) cannot strike the shuttle; the real usable normal window of '
             'the racket needs measurement'))
    weight_time: Param = field(default_factory=lambda: _temp(0.30, 'TEMP proxy score weight (S99)'))
    weight_height: Param = field(default_factory=lambda: _temp(0.25, 'TEMP proxy score weight (S99)'))
    weight_speed: Param = field(default_factory=lambda: _temp(0.15, 'TEMP proxy score weight (S99)'))
    weight_margin: Param = field(default_factory=lambda: _temp(0.15, 'TEMP proxy score weight (S99)'))
    weight_base_cost: Param = field(default_factory=lambda: _temp(0.15, 'TEMP proxy penalty weight (S99)'))
    achievable_face_normal_cone_rad: Param = field(default_factory=lambda: Param(
        None, _MEASURE,
        'measure the achievable racket face-normal cone of the real PiPER + racket mount; until '
        'then the search only checks the edge-on guard (min_face_alignment) and does not pretend '
        'to know the orientation workspace (S57/S59)'))
    effective_contact_offset_m: Param = field(default_factory=lambda: Param(
        None, _MEASURE,
        'measure the racket sweet-spot offset from racket_contact on the real rig (S43); until '
        'then the search uses the shuttle position itself and applies no offset'))

    def param_limits(self) -> Dict[str, Param]:
        """Every numeric limit of this module, as Param objects."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if isinstance(getattr(self, f.name), Param)}

    def unresolved_limits(self) -> Tuple[Tuple[str, Param], ...]:
        """The declared-but-unmeasured quantities, sorted by name."""
        return tuple((name, param) for name, param in sorted(self.param_limits().items())
                     if param.status in UNRESOLVED_STATUSES or param.value is None)

    def measurement_requirements(self) -> Dict[str, Param]:
        """What must be measured before this search can be trusted on the real robot."""
        return {name: Param(None, _MEASURE,
                            'measure ' + name + ' of the real rig: ' + param.source)
                for name, param in self.unresolved_limits()}

    def weights(self) -> ScoreWeights:
        return ScoreWeights(time=_param_float(self.weight_time, 'weight_time'),
                            height=_param_float(self.weight_height, 'weight_height'),
                            speed=_param_float(self.weight_speed, 'weight_speed'),
                            margin=_param_float(self.weight_margin, 'weight_margin'),
                            base_cost=_param_float(self.weight_base_cost, 'weight_base_cost'))


@dataclass(frozen=True)
class SearchCandidate:
    """One evaluated candidate time (kept even when it is rejected, so a decision is auditable)."""
    env_id: int
    time_s: float
    position: np.ndarray
    velocity: np.ndarray
    feasible: bool
    reason: InterceptReason
    racket_pose: Optional[np.ndarray] = None
    score: float = float('-inf')
    score_terms: ScoreTerms = ScoreTerms()
    time_available_s: float = 0.0
    time_required_s: float = 0.0
    base_travel_m: float = 0.0
    workspace_margin: float = 0.0


@dataclass(frozen=True)
class InterceptSearchResult:
    """Search output for the selected environments (rows follow ``env_ids`` order).

    ``best`` is the frozen BestIntercept message and contains one row per *feasible* environment,
    in the order of ``env_ids`` - infeasible environments are simply absent (they are listed in
    ``best_env_ids`` and flagged in ``feasible``), so a BestIntercept never carries a fabricated
    position.  When no environment has a feasible intercept, ``best`` is None.
    """
    feasible: np.ndarray
    reason: Tuple[InterceptReason, ...]
    score: np.ndarray
    best: Optional[BestIntercept]
    best_env_ids: np.ndarray
    candidate_times: np.ndarray
    candidates: Tuple[Tuple[SearchCandidate, ...], ...]
    env_ids: Tuple[int, ...]
    now_s: float

    def _index(self, env_id: int) -> int:
        try:
            return self.env_ids.index(int(env_id))
        except ValueError:
            raise BrainBoundaryError(f'env {env_id} is not part of this search result')

    def best_for_env(self, env_id: int) -> Optional[SearchCandidate]:
        """Highest-scoring feasible candidate of one environment (None if it has none)."""
        return select_best_candidate(self.candidates[self._index(env_id)])

    def earliest_for_env(self, env_id: int) -> Optional[SearchCandidate]:
        """Earliest feasible candidate of one environment (None if it has none)."""
        for candidate in self.candidates[self._index(env_id)]:
            if candidate.feasible:
                return candidate
        return None


# --------------------------------------------------------------------------- helpers
def _temp(value: float, source: str) -> Param:
    return Param(value, _TEMP, source)


def _param_float(param: Any, name: str) -> float:
    if not isinstance(param, Param):
        raise BrainBoundaryError(f'{name} must be a Param (S4/S12), got {type(param).__name__}')
    if param.value is None:
        raise BrainBoundaryError(
            f'{name} is unresolved ({param.status.value}) and the search needs a number: '
            f'{param.source}')
    value = float(param.value)
    if not math.isfinite(value):
        raise BrainBoundaryError(f'{name} must be finite, got {param.value!r}')
    return value


def _sigmoid(value: float) -> float:
    """Overflow-free logistic function."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def min_travel_time_s(distance_m: float, v_max_mps: float, a_max_mps2: float) -> float:
    """Minimum 1-D travel time under |v| <= v_max and |a| <= a_max (S62-S64).

    Triangle profile while ``d <= v_max^2 / a_max`` (S63), trapezoid profile beyond it (S64).
    """
    distance = abs(float(distance_m))
    v_max = float(v_max_mps)
    a_max = float(a_max_mps2)
    if not math.isfinite(v_max) or not math.isfinite(a_max):
        raise BrainBoundaryError('v_max_mps and a_max_mps2 must be finite')
    if v_max <= 0.0 or a_max <= 0.0:
        raise BrainBoundaryError('v_max_mps and a_max_mps2 must be > 0')
    if not math.isfinite(distance) or distance <= 0.0:
        return 0.0
    cruise_distance = v_max * v_max / a_max
    if distance <= cruise_distance:
        return 2.0 * math.sqrt(distance / a_max)
    return 2.0 * v_max / a_max + (distance - cruise_distance) / v_max


def quaternion_from_rotation(rotation: Any) -> np.ndarray:
    """(x, y, z, w) unit quaternion of a rotation matrix, canonicalised to w >= 0 (S2.4 order)."""
    m = np.asarray(rotation, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(m)):
        raise BrainBoundaryError('rotation matrix must be finite')
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm <= _DEGENERATE_EPS:
        raise BrainBoundaryError('rotation matrix is not a rotation (degenerate quaternion)')
    quaternion = quaternion / norm
    if quaternion[3] < 0.0:          # q and -q are the same rotation: keep the canonical sign
        quaternion = -quaternion
    return quaternion


def rotation_from_quaternion(quaternion: Any) -> np.ndarray:
    """Rotation matrix of an (x, y, z, w) unit quaternion (columns are the body axes)."""
    q = np.asarray(quaternion, dtype=float).reshape(4)
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def racket_pose_axes(pose: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(+X face normal, +Y, +Z grip -> head) of a [x, y, z, qx, qy, qz, qw] racket pose."""
    rotation = rotation_from_quaternion(np.asarray(pose, dtype=float).reshape(7)[3:7])
    return rotation[:, 0], rotation[:, 1], rotation[:, 2]


def racket_pose_from_contact(position: Any, incoming_velocity: Any) -> np.ndarray:
    """Build the racket_contact pose for a flat block (COORDINATE_SYSTEM.md S6.4/S6.5).

    +X_racket (the face normal) is set opposite to the incoming shuttle velocity.  That is the
    ideal mirror normal of 04_HIT_DECISION.md S48, i.e. the shuttle leaves along -d_in; no
    deflection or restitution constant is invented.  +Z_racket (grip -> head) is the component of
    the court up axis orthogonal to the normal - the head is held "as upright as the normal
    allows", which fixes the roll without an arbitrary offset.  When the normal is parallel to up
    (vertical drop) the documented fallback reference is the court +X axis.

    Returns [x, y, z, qx, qy, qz, qw] with the contact point equal to ``position``.
    """
    contact = np.asarray(position, dtype=float).reshape(3)
    velocity = np.asarray(incoming_velocity, dtype=float).reshape(3)
    if not np.all(np.isfinite(contact)) or not np.all(np.isfinite(velocity)):
        raise ValueError('racket_pose_from_contact requires finite inputs')
    speed = float(np.linalg.norm(velocity))
    if speed <= _SPEED_EPS_MPS:
        raise ValueError(
            'racket_pose_from_contact needs the incoming shuttle velocity to define the face '
            'normal; the shuttle must not be at rest')
    normal = -velocity / speed
    head = _UP - float(np.dot(_UP, normal)) * normal
    if float(np.linalg.norm(head)) <= _DEGENERATE_EPS:
        head = _COURT_X - float(np.dot(_COURT_X, normal)) * normal
    if float(np.linalg.norm(head)) <= _DEGENERATE_EPS:
        raise ValueError('degenerate face normal: no racket head axis can be constructed')
    z_axis = head / np.linalg.norm(head)
    x_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return np.concatenate((contact, quaternion_from_rotation(rotation)))


def racket_pose_is_feasible(pose: Any, incoming_velocity: Any,
                            config: Optional[InterceptSearchConfig] = None) -> bool:
    """Is this racket_contact pose a legal strike pose for the incoming shuttle?

    Checks what v0.1 can honestly check: the pose is a finite, unit-quaternion frame, the shuttle
    is not at rest, and the face normal is not edge-on to the incoming velocity
    (``|d_in . n| >= min_face_alignment``).  Orientation reachability (S57/S59) needs a
    WorkspaceMap or IK and is therefore not claimed here.
    """
    cfg = config if config is not None else InterceptSearchConfig()
    try:
        contact_pose = np.asarray(pose, dtype=float).reshape(7)
    except (TypeError, ValueError):
        return False
    if not np.all(np.isfinite(contact_pose)):
        return False
    quaternion = contact_pose[3:7]
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-9:
        return False
    velocity = np.asarray(incoming_velocity, dtype=float).reshape(3)
    speed = float(np.linalg.norm(velocity))
    if not math.isfinite(speed) or speed <= _SPEED_EPS_MPS:
        return False
    normal = rotation_from_quaternion(quaternion)[:, 0]
    alignment = abs(float(np.dot(velocity / speed, normal)))
    return alignment >= _param_float(cfg.min_face_alignment, 'min_face_alignment')


def select_best_candidate(candidates: Sequence[SearchCandidate]) -> Optional[SearchCandidate]:
    """Maximum score; ties go to the earliest time, then to the smallest position (S115).

    Implemented as a max over the key (score, -time, -x, -y, -z) so the result does not depend on
    the order in which the candidates are supplied.
    """
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        return None
    return max(feasible, key=lambda c: (c.score, -c.time_s, *(-np.asarray(c.position,
                                                                 dtype=float).reshape(3))))


def _hermite(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray,
             time_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation of position and velocity between two samples (S36-S37).

    p(s) = h00 p_i + h10 dt v_i + h01 p_{i+1} + h11 dt v_{i+1} with s = (t - t_i) / dt;
    the derivative v(s) uses the same basis so the interpolant is C1 and exact for a parabola.
    """
    index = int(np.searchsorted(times, time_s, side='right')) - 1
    index = min(max(index, 0), times.shape[0] - 2)
    t0 = float(times[index])
    dt = float(times[index + 1]) - t0
    s = (float(time_s) - t0) / dt
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    position = (h00 * positions[index] + h10 * dt * velocities[index]
                + h01 * positions[index + 1] + h11 * dt * velocities[index + 1])
    dh00 = 6.0 * s2 - 6.0 * s
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = -6.0 * s2 + 6.0 * s
    dh11 = 3.0 * s2 - 2.0 * s
    velocity = ((dh00 * positions[index] + dh01 * positions[index + 1]) / dt
                + dh10 * velocities[index] + dh11 * velocities[index + 1])
    return position, velocity


def _yaw_from_quaternion(quaternion: Any) -> float:
    """Yaw of a pose quaternion (x, y, z, w); the base is planar, so only yaw matters here."""
    x, y, z, w = (float(v) for v in np.asarray(quaternion, dtype=float).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _resolve_env_ids(env_ids: Optional[Sequence[int]], batch: int) -> Tuple[int, ...]:
    if batch < 1:
        raise BrainBoundaryError('the search needs at least one environment')
    if env_ids is None:
        return tuple(range(batch))
    selected = tuple(int(env) for env in env_ids)
    if len(set(selected)) != len(selected):
        raise BrainBoundaryError(f'env_ids must be unique, got {selected}')
    for env in selected:
        if env < 0 or env >= batch:
            raise BrainBoundaryError(f'env {env} is out of range for a batch of {batch}')
    return selected


def _resolve_now(state: UnifiedState, trajectory: PredictedTrajectory, times: np.ndarray,
                 now: Optional[float]) -> float:
    """Reference instant for T_available (S70-S73).

    ``now`` wins when supplied.  Otherwise ``state.timestamp`` is used, except when the prediction
    grid starts *after* its own creation timestamp - that can only mean the grid is relative to
    the prediction instant (S6 asks for absolute times, the frozen aerodynamics rollout returns
    times from 0), and then the elapsed time since the prediction is the honest base.
    """
    if now is not None:
        value = float(now)
        if not math.isfinite(value):
            raise BrainBoundaryError(f'now must be finite, got {now!r}')
        return value
    reference = float(state.timestamp)
    if not math.isfinite(reference):
        raise BrainBoundaryError('UnifiedState.timestamp must be finite')
    predicted_at = float(trajectory.timestamp)
    if not math.isfinite(predicted_at):
        raise BrainBoundaryError('PredictedTrajectory.timestamp must be finite')
    if float(times[0]) < predicted_at - _TIME_EPS_S:
        return max(0.0, reference - predicted_at)
    return reference


class InterceptSearcher:
    """Stateless intercept search (decision layer): one PredictedTrajectory -> BestIntercept.

    Holds only configuration, so repeated calls on the same input are bit-identical and the
    module can be shared by the DecisionModule wrapper of T5/T11.
    """

    def __init__(self, config: Optional[InterceptSearchConfig] = None) -> None:
        self.config = config if config is not None else InterceptSearchConfig()

    def param_limits(self) -> Dict[str, Param]:
        return self.config.param_limits()

    def unresolved_limits(self) -> Tuple[Tuple[str, Param], ...]:
        return self.config.unresolved_limits()

    def measurement_requirements(self) -> Dict[str, Param]:
        return self.config.measurement_requirements()

    def search(self, state: UnifiedState, trajectory: PredictedTrajectory, *,
               now: Optional[float] = None,
               env_ids: Optional[Sequence[int]] = None) -> InterceptSearchResult:
        return search_intercepts(state, trajectory, config=self.config, now=now, env_ids=env_ids)


def _empty_result(env_ids: Tuple[int, ...], times: np.ndarray, now_s: float,
                  reason: InterceptReason) -> InterceptSearchResult:
    return InterceptSearchResult(
        feasible=np.zeros(len(env_ids), dtype=bool),
        reason=tuple(reason for _ in env_ids),
        score=np.full(len(env_ids), -np.inf, dtype=float),
        best=None,
        best_env_ids=np.zeros(0, dtype=int),
        candidate_times=np.asarray(times, dtype=float),
        candidates=tuple(() for _ in env_ids),
        env_ids=env_ids,
        now_s=now_s)


def search_intercepts(state: UnifiedState, trajectory: PredictedTrajectory, *,
                      config: Optional[InterceptSearchConfig] = None,
                      now: Optional[float] = None,
                      env_ids: Optional[Sequence[int]] = None) -> InterceptSearchResult:
    """Search the whole predicted trajectory for the best (and the earliest) feasible intercept.

    Returns an :class:`InterceptSearchResult`; ``result.best`` is a frozen ``BestIntercept`` with
    one row per feasible environment (None when no environment has a feasible intercept) and
    ``result.candidates`` keeps every evaluated candidate with its reason and score breakdown.
    """
    if not isinstance(state, UnifiedState):
        raise BrainBoundaryError(
            f'search_intercepts expects a UnifiedState, got {type(state).__name__}')
    if not isinstance(trajectory, PredictedTrajectory):
        raise BrainBoundaryError(
            f'search_intercepts expects a PredictedTrajectory, got {type(trajectory).__name__}')
    cfg = config if config is not None else InterceptSearchConfig()
    times = np.asarray(trajectory.times, dtype=float)
    batch = int(np.asarray(state.base_pose).shape[0])
    predicted_batch = int(np.asarray(trajectory.position).shape[0])
    if predicted_batch != batch:
        raise BrainBoundaryError(
            f'UnifiedState batch ({batch}) and PredictedTrajectory batch ({predicted_batch}) differ')
    selected = _resolve_env_ids(env_ids, batch)
    now_s = _resolve_now(state, trajectory, times, now)

    if times.shape[0] < 2:
        return _empty_result(selected, times, now_s, InterceptReason.INVALID_PREDICTION)
    if np.any(np.diff(times) <= 0.0):
        raise BrainBoundaryError(
            'PredictedTrajectory.times must be strictly increasing for interpolation')
    if now_s > float(times[-1]) + _TIME_EPS_S:
        raise BrainBoundaryError(
            f'search time base now={now_s} lies after the whole prediction grid '
            f'[{float(times[0])}, {float(times[-1])}]; pass now explicitly (a prediction grid may '
            f'be relative to its own start instant)')

    limits = {name: _param_float(param, name) for name, param in cfg.param_limits().items()
              if param.value is not None}
    reach_x = limits['arm_reach_x_m']
    reach_y = limits['arm_reach_y_m']
    z_min = limits['arm_z_min_m']
    z_max = limits['arm_z_max_m']
    if not z_min < z_max:
        raise BrainBoundaryError('arm_z_min_m must be below arm_z_max_m')
    overhead_s = limits['decision_latency_s'] + limits['safety_time_margin_s']
    arm_slew_s = limits['arm_slew_time_s']
    weights = cfg.weights()

    step = limits['candidate_dt_s']
    span = float(times[-1] - times[0])
    count = int(math.floor(span / step + 1e-9)) + 1
    candidate_times = float(times[0]) + step * np.arange(count, dtype=float)

    per_env: list = []
    feasible_flags = []
    scores = []
    reasons = []
    best_rows = []
    best_ids = []

    for env in selected:
        positions = np.asarray(trajectory.position[env], dtype=float)
        velocities = np.asarray(trajectory.velocity[env], dtype=float)
        arrival_s = float(np.asarray(trajectory.arrival_time[env], dtype=float))
        base_pose = np.asarray(state.base_pose[env], dtype=float).reshape(7)
        base_x = float(base_pose[0])
        base_y = float(base_pose[1])
        yaw = _yaw_from_quaternion(base_pose[3:7])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        candidates = []
        for time_s in candidate_times:
            time_s = float(time_s)
            position, velocity = _hermite(times, positions, velocities, time_s)
            reason = None
            travel = 0.0
            required = 0.0
            available = 0.0
            margin = 0.0
            terms = ScoreTerms()
            pose = None

            if time_s > arrival_s:
                reason = InterceptReason.OUT_OF_BOUNDS
            elif not (z_min <= float(position[2]) <= z_max):
                reason = InterceptReason.NO_GEOMETRIC_WINDOW
            else:
                # Base-frame offset of the contact point: the workspace box is carried by the base.
                offset_x = float(position[0]) - base_x
                offset_y = float(position[1]) - base_y
                local_x = cos_yaw * offset_x + sin_yaw * offset_y
                local_y = -sin_yaw * offset_x + cos_yaw * offset_y
                excess_x = max(0.0, abs(local_x) - reach_x)
                excess_y = max(0.0, abs(local_y) - reach_y)
                travel = math.hypot(excess_x, excess_y)
                required = max(min_travel_time_s(travel, limits['base_v_max_mps'],
                                                 limits['base_a_max_mps2']), arm_slew_s)
                available = time_s - now_s - overhead_s
                if available <= 0.0:
                    reason = InterceptReason.NO_TIME_MARGIN
                elif required >= available:
                    reason = InterceptReason.UNREACHABLE
                else:
                    residual_x = max(abs(local_x) - excess_x, 0.0)
                    residual_y = max(abs(local_y) - excess_y, 0.0)
                    margin = min((reach_x - residual_x) / reach_x,
                                 (reach_y - residual_y) / reach_y,
                                 min(float(position[2]) - z_min, z_max - float(position[2]))
                                 / (z_max - z_min))
                    margin = min(max(margin, 0.0), 1.0)
                    try:
                        pose = racket_pose_from_contact(position, velocity)
                    except ValueError:
                        pose = None
                    if pose is None or not racket_pose_is_feasible(pose, velocity, cfg):
                        reason = InterceptReason.NO_ORIENTATION_SOLUTION
                        pose = None
                    else:
                        reason = InterceptReason.BEST_INTERCEPT_FOUND

            score = float('-inf')
            if reason is InterceptReason.BEST_INTERCEPT_FOUND:
                speed = float(np.linalg.norm(velocity))
                terms = ScoreTerms(
                    time=_sigmoid((available - required) / limits['time_margin_tau_s']),
                    height=math.exp(-(float(position[2]) - limits['z_pref_m']) ** 2
                                     / (2.0 * limits['z_sigma_m'] ** 2)),
                    speed=math.exp(-(speed - limits['speed_pref_mps']) ** 2
                                    / (2.0 * limits['speed_sigma_mps'] ** 2)),
                    margin=margin,
                    base_cost=min(1.0, travel / limits['base_travel_ref_m']))
                score = terms.total(weights)

            candidates.append(SearchCandidate(
                env_id=int(env),
                time_s=time_s,
                position=position,
                velocity=velocity,
                feasible=reason is InterceptReason.BEST_INTERCEPT_FOUND,
                reason=reason,
                racket_pose=pose,
                score=score,
                score_terms=terms,
                time_available_s=available,
                time_required_s=required,
                base_travel_m=travel,
                workspace_margin=margin))

        per_env.append(tuple(candidates))
        best = select_best_candidate(candidates)
        feasible_flags.append(best is not None)
        scores.append(-np.inf if best is None else best.score)
        reasons.append(InterceptReason.BEST_INTERCEPT_FOUND if best is not None
                       else (candidates[0].reason if candidates
                             else InterceptReason.INVALID_PREDICTION))
        if best is not None:
            best_rows.append(best)
            best_ids.append(int(env))

    best_message = None
    if best_rows:
        best_message = BestIntercept(
            position=np.stack([row.position for row in best_rows], axis=0),
            time_s=np.asarray([row.time_s for row in best_rows], dtype=float),
            racket_pose=np.stack([row.racket_pose for row in best_rows], axis=0),
            score=np.asarray([row.score for row in best_rows], dtype=float),
            timestamp=float(state.timestamp))

    return InterceptSearchResult(
        feasible=np.asarray(feasible_flags, dtype=bool),
        reason=tuple(reasons),
        score=np.asarray(scores, dtype=float),
        best=best_message,
        best_env_ids=np.asarray(best_ids, dtype=int),
        candidate_times=candidate_times,
        candidates=tuple(per_env),
        env_ids=selected,
        now_s=now_s)


__all__ = ["InterceptReason", "InterceptSearchConfig", "InterceptSearchResult", "InterceptSearcher",
           "ScoreTerms", "ScoreWeights", "SearchCandidate", "min_travel_time_s",
           "quaternion_from_rotation", "racket_pose_axes", "racket_pose_from_contact",
           "racket_pose_is_feasible", "rotation_from_quaternion", "search_intercepts",
           "select_best_candidate"]
