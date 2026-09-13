# -*- coding: utf-8 -*-
"""T7 expert planner: intercept -> WholeBodyTarget (base stationing + racket timing).

Layer: PLANNING (ROBOT_BRAIN.md S11/S12/S15).  This module emits a *target* only: it never
produces an executable command and never bypasses Safety.

Frames (coordinator verdict, recorded in the plan ledger):
  * Every Robot Brain message is court frame EXCEPT the base twist fields.
  * WholeBodyTarget.base_twist and SafeCommand.base_twist are robot_base (body) frame,
    [vx_body, vy_body, wz].  Execution only ever receives SafeCommand - it has no pose - so it
    cannot perform the court -> body rotation itself; the planner holds UnifiedState (which
    carries the base pose, hence the yaw) and does the rotation here.
  * Where the conversion happens: process() derives the stationing demand in the court frame,
    then rotates the translational part into the body frame with the base rotation
    (v_body = R_base^T v_court) and passes that body twist to the Morph One kinematics for
    wheel-limit clipping.  The yaw rate wz is unchanged by a rotation about the court +Z axis,
    so it is the same scalar in both frames.

Honesty contract - what is exact and what is an approximation:
  * EXACT (base side).  The chassis command is validated by mapping it through the frozen
    Phase-3 Morph One four-steer/four-drive kinematics
    (simulation/robots/badminton_robot/morph_one/kinematics.py), whose input is a robot_base
    twist.  That map is linear in the twist, so one uniform scale is the minimal correction
    that brings every wheel rate inside max_wheel_speed_rad_s.  Steer angles do not depend on
    the twist magnitude, so they are checked directly on the final command; because
    (theta, omega) and (theta +- pi, -omega) are the same physical command, the check asks
    whether the smallest equivalent representation fits the limit.
  * APPROXIMATE (arm side).  There is no PiPER IK and no whole-body NMPC in this repository,
    and none is invented here.  The joint target is a bounded first-order (Moore-Penrose) step
    on a proxy Jacobian of the racket contact point (TEMP_CONTACT_JACOBIAN unless the caller
    supplies a measured one), taken in the robot-base frame and clipped by a TEMP per-joint
    offset and by the TEMP joint-rate limit over the horizon.  It is explicitly NOT an
    inverse-kinematics solution: a collision-aware whole-body solver must replace it.
  * INDEPENDENCE.  The base term and the arm term are computed independently (no whole-body
    coordination, no collision check, no racket-orientation control).  The target is re-planned
    every control cycle, so the contact residual shrinks cycle by cycle.

Known gaps, reported rather than hidden (see measurement_requirements()):
  * WholeBodyTarget.horizon_s is a scalar while intercept deadlines are per-environment.  The
    planner plans every environment against its own deadline and reports the batch minimum
    (earliest = most conservative).  Changing the contract is a coordinator decision.
  * Steer-rate feasibility cannot be checked here: UnifiedState carries no steer state, and a
    steer-rate limit constrains how fast the wheel *direction* may change, not the twist
    magnitude.  A deployment with max_steer_angle_rad below the required steer angle cannot be
    honoured by twist scaling alone; the result is reported in
    last_diagnostics['steer_limit_exceeded'] / ['steer_equivalent_max_rad'] and the final
    enforcement belongs to Safety (T8).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..interfaces import PlanningModule
from ..status import AssetStatus, Param, UNRESOLVED_STATUSES
from ..types import BrainBoundaryError, WholeBodyTarget

# --- TEMP engineering baseline (never a measured claim) -------------------------------------

TEMP_CONTACT_JACOBIAN = np.array([
    # joint1   joint2   joint3   joint4   joint5   joint6      [m/rad], rows = base x, y, z
    [0.00,     0.30,   -0.10,    0.00,    0.05,    0.00],
    [0.35,     0.00,    0.00,    0.10,    0.00,    0.08],
    [0.00,     0.05,    0.30,    0.00,    0.15,    0.00],
], dtype=float)
"""First-order proxy for d p_contact_base / d q, column j = PiPER joint j+1.

TEMP_PARAMETERIZED_PROXY: no PiPER link geometry, joint axes or racket mount transform are
measured in this repository (RacketCfg.t_link6_tcp and .t_tcp_contact are REQUIRES_MEASUREMENT),
so these columns are an engineering-shaped placeholder with the correct qualitative structure -
joint1 (base yaw) moves the contact laterally, joints 2/3/5 (shoulder, elbow, wrist pitch) move
it forward/up.  Pass a measured 3x6 matrix to the constructor to replace it; nothing outside
this module may treat the TEMP matrix as truth.
"""

TEMP_WHEEL_POSITIONS = {
    'FL': (0.25, 0.20),
    'FR': (0.25, -0.20),
    'RL': (-0.25, 0.20),
    'RR': (-0.25, -0.20),
}
"""Wheel centres in robot_base, m.  TEMP, identical to the Phase-3 kinematics baseline in
tests/simulation/robots/test_morph_one_kinematics.py; MorphOneGeometry.wheel_positions_robot is
REQUIRES_MEASUREMENT and must override these once measured."""

_SRC_TEMP_STANDOFF = ("TEMP engineering baseline standoff for a 0.70x0.55 m chassis with the "
                      "PiPER mount at (0,0,0.30); REQUIRES_MEASUREMENT from the PiPER reach study")
_SRC_TEMP_HORIZON = "TEMP engineering baseline replan window; the real stage rate is REQUIRES_MEASUREMENT"
_SRC_TEMP_JOINT = ("TEMP engineering baseline PiPER joint limit proxy; PiPER joint limits are not "
                   "measured in this repository")
_SRC_TEMP_WHEEL = ("TEMP engineering baseline, same value as MorphOneCfg.max_wheel_speed_rad_s "
                   "(REQUIRES_MEASUREMENT on the real base)")
_SRC_TEMP_STEER = ("TEMP engineering baseline, same value as MorphOneCfg.max_steer_angle_rad: "
                   "continuous steering assumed")
_SRC_TEMP_RADIUS = ("TEMP engineering baseline, same value as MorphOneGeometry.wheel_radius_m; "
                    "REQUIRES_MEASUREMENT on the real wheel")

_SRC_NEED_WHEEL_POSITIONS = ("measure the four Morph One wheel centres in robot_base "
                             "(MorphOneGeometry.wheel_positions_robot); until then the planner "
                             "uses TEMP_WHEEL_POSITIONS mirrored from the Phase-3 kinematics test")
_SRC_NEED_CONTACT_JACOBIAN = ("measure the PiPER racket-contact Jacobian d p_contact / d q on the "
                              "real rig (link lengths, joint axes, RacketCfg.t_link6_tcp and "
                              "t_tcp_contact are all unmeasured); until then TEMP_CONTACT_JACOBIAN "
                              "is an engineering-shaped proxy, not kinematics")
_SRC_NEED_JOINT_LIMITS = ("measure the PiPER joint position and velocity limits (no joint limit "
                          "table exists in this repository)")


@dataclass
class PlannerLimits:
    """Every number the planner needs, each carrying its own authenticity status (S4/S12)."""
    standoff_m: Param = field(default_factory=lambda: Param(
        0.45, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_STANDOFF))
    min_horizon_s: Param = field(default_factory=lambda: Param(
        0.02, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_HORIZON))
    max_horizon_s: Param = field(default_factory=lambda: Param(
        1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_HORIZON))
    max_joint_offset_rad: Param = field(default_factory=lambda: Param(
        0.6, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_JOINT))
    max_joint_rate_rad_s: Param = field(default_factory=lambda: Param(
        2.0, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_JOINT))
    max_wheel_speed_rad_s: Param = field(default_factory=lambda: Param(
        40.0, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_WHEEL))
    max_steer_angle_rad: Param = field(default_factory=lambda: Param(
        math.pi, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_STEER))
    wheel_radius_m: Param = field(default_factory=lambda: Param(
        0.06, AssetStatus.TEMP_PARAMETERIZED_PROXY, _SRC_TEMP_RADIUS))

    def param_limits(self) -> Dict[str, Param]:
        """Every limit of this module as a Param, by name (same API as T5/T6)."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__
                if isinstance(getattr(self, name), Param)}

    def unresolved_limits(self) -> Tuple[Tuple[str, Param], ...]:
        """Declared-but-unmeasured limits, sorted by name: not backed by a real measurement."""
        return tuple((name, param) for name, param in sorted(self.param_limits().items())
                     if param.status in UNRESOLVED_STATUSES or param.value is None)

    def measurement_requirements(self) -> Dict[str, Param]:
        """One Param(None, REQUIRES_MEASUREMENT, source) per limit that is still a proxy."""
        return {name: Param(None, AssetStatus.REQUIRES_MEASUREMENT,
                            'measure ' + name + ': ' + param.source)
                for name, param in self.unresolved_limits()}


def _morph_one():
    """Import the frozen Phase-3 Morph One machinery (pure numpy, no Isaac/Kit)."""
    simulation_root = Path(__file__).resolve().parents[3] / 'simulation'
    if str(simulation_root) not in sys.path:
        sys.path.insert(0, str(simulation_root))
    from robots.badminton_robot.badminton_robot_cfg import WheelId
    from robots.badminton_robot.frames.robot_frames import quat_to_matrix
    from robots.badminton_robot.morph_one.kinematics import body_twist_to_wheel_targets
    return WheelId, quat_to_matrix, body_twist_to_wheel_targets


_WHEEL_GEOMETRY_SOURCE = ("TEMP wheel centres and radius (TEMP_WHEEL_POSITIONS / "
                          "PlannerLimits.wheel_radius_m); override with the measured values")


class ExpertPlanner(PlanningModule):
    """Deterministic baseline planner: station the base, then step the racket towards contact.

    Constructor arguments are all optional.  With no arguments the planner runs on the TEMP
    engineering baseline above and every TEMP value stays labelled as such in the limits object
    (introspect planner.limits / planner.measurement_requirements()).  Passing a
    BadmintonRobotCfg takes over any Morph One value that is actually resolved in the config;
    unresolved (REQUIRES_MEASUREMENT) fields keep the labelled TEMP fallback instead of silently
    inventing a number.

    Output frame: WholeBodyTarget.base_twist is robot_base (body) frame - see the module
    docstring, which is where the court -> body rotation is documented.
    """

    name = 'expert_planner'
    is_implemented = True

    def __init__(self, num_envs: Optional[int] = None, *, cfg: Any = None,
                 limits: Optional[PlannerLimits] = None,
                 wheel_positions: Optional[Mapping[Any, Any]] = None,
                 wheel_radius_m: Optional[float] = None,
                 contact_jacobian: Optional[Any] = None) -> None:
        # The frozen kinematics and frame helpers are resolved once, at construction: the
        # control cycle must not import modules or touch sys.path.
        wheel_id, self._quat_to_matrix, self._twist_to_wheels = _morph_one()
        self.num_envs = num_envs
        self.limits = limits or PlannerLimits()

        if cfg is not None:
            self._absorb_cfg(cfg)

        if wheel_radius_m is not None:
            self.limits = replace(self.limits, wheel_radius_m=Param(
                float(wheel_radius_m), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "caller-supplied wheel radius; still TEMP until measured and re-labelled"))

        raw = wheel_positions
        if raw is None and cfg is not None:
            geometry = getattr(getattr(cfg, 'morph_one', None), 'geometry', None)
            raw = getattr(getattr(geometry, 'wheel_positions_robot', None), 'value', None)
        self.wheel_positions: Dict[Any, Any] = self._normalise_wheel_positions(raw, wheel_id)
        self.wheel_geometry_source = _WHEEL_GEOMETRY_SOURCE
        self._wheel_positions_measured = raw is not None

        self._contact_jacobian, self._contact_jacobian_is_temp = self._resolve_contact_jacobian(
            contact_jacobian)
        self._contact_jacobian_pinv = np.linalg.pinv(self._contact_jacobian)
        self.last_diagnostics: Dict[str, Any] = self._empty_diagnostics()

    # -- configuration helpers ---------------------------------------------------------------
    def _absorb_cfg(self, cfg: Any) -> None:
        """Adopt the Morph One numbers that the robot config really resolved."""
        morph_one = getattr(cfg, 'morph_one', None)
        geometry = getattr(morph_one, 'geometry', None)
        candidates = (
            ('wheel_radius_m', getattr(geometry, 'wheel_radius_m', None)),
            ('max_wheel_speed_rad_s', getattr(morph_one, 'max_wheel_speed_rad_s', None)),
            ('max_steer_angle_rad', getattr(morph_one, 'max_steer_angle_rad', None)),
        )
        overrides = {name: param for name, param in candidates
                     if isinstance(param, Param) and param.value is not None}
        if overrides:
            self.limits = replace(self.limits, **overrides)

    @staticmethod
    def _normalise_wheel_positions(raw: Any, wheel_id) -> Dict[Any, Any]:
        if raw is None:
            return {wheel_id[name]: tuple(pos) for name, pos in TEMP_WHEEL_POSITIONS.items()}
        return {key if isinstance(key, wheel_id) else wheel_id[str(key)]:
                tuple(float(value) for value in position) for key, position in raw.items()}

    @staticmethod
    def _resolve_contact_jacobian(contact_jacobian: Optional[Any]) -> Tuple[np.ndarray, bool]:
        """Return (jacobian, is_temp).  Handing the TEMP proxy back in stays TEMP."""
        if contact_jacobian is None:
            return TEMP_CONTACT_JACOBIAN, True
        jacobian = np.asarray(contact_jacobian, dtype=float)
        if jacobian.shape != (3, 6):
            raise ValueError("contact_jacobian must be (3, 6) [m/rad], got %s" % (jacobian.shape,))
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("contact_jacobian contains NaN/Inf")
        return jacobian, bool(np.array_equal(jacobian, TEMP_CONTACT_JACOBIAN))

    @staticmethod
    def _empty_diagnostics() -> Dict[str, Any]:
        return {'hold': False, 'clip_scale': np.ones(0), 'horizon_s': np.ones(0),
                'steer_limit_exceeded': np.zeros(0, dtype=bool),
                'steer_equivalent_max_rad': np.zeros(0)}

    # -- contract accessors ------------------------------------------------------------------
    @property
    def wheel_radius_m(self) -> float:
        return float(self.limits.wheel_radius_m.value)

    @property
    def max_wheel_speed_rad_s(self) -> float:
        return float(self.limits.max_wheel_speed_rad_s.value)

    @property
    def max_steer_angle_rad(self) -> float:
        return float(self.limits.max_steer_angle_rad.value)

    def measurement_requirements(self) -> Dict[str, Param]:
        """Everything that must be measured before this planner can be trusted (S4/S12).

        Same API as the T5/T6 configs: {name: Param(None, REQUIRES_MEASUREMENT, source)}.
        """
        requirements = dict(self.limits.measurement_requirements())
        if not self._wheel_positions_measured:
            requirements['wheel_positions_robot'] = Param(
                None, AssetStatus.REQUIRES_MEASUREMENT, _SRC_NEED_WHEEL_POSITIONS)
        if self._contact_jacobian_is_temp:
            requirements['contact_jacobian'] = Param(
                None, AssetStatus.REQUIRES_MEASUREMENT, _SRC_NEED_CONTACT_JACOBIAN)
        requirements['piper_joint_limits'] = Param(
            None, AssetStatus.REQUIRES_MEASUREMENT, _SRC_NEED_JOINT_LIMITS)
        return requirements

    def reset(self, env_ids) -> None:
        self.last_diagnostics = self._empty_diagnostics()

    # -- the planning step -------------------------------------------------------------------
    def process(self, state, decision, intercept, trajectory) -> WholeBodyTarget:
        base_pose = np.asarray(state.base_pose, dtype=float)
        joint_pos = np.asarray(state.joint_pos, dtype=float)
        n = int(base_pose.shape[0])
        timestamp = float(state.timestamp)

        nothing_to_play = (not bool(getattr(decision, 'feasible', False))
                           or intercept is None
                           or getattr(intercept, 'position', None) is None)
        if nothing_to_play:
            # Nothing should be played: hold the measured posture and command no motion.
            self.last_diagnostics = {'hold': True, 'clip_scale': np.ones(n),
                                     'horizon_s': np.zeros(n),
                                     'steer_limit_exceeded': np.zeros(n, dtype=bool),
                                     'steer_equivalent_max_rad': np.zeros(n)}
            return WholeBodyTarget(base_twist=np.zeros((n, 3)),
                                   joint_position_target=np.array(joint_pos, dtype=float, copy=True),
                                   horizon_s=0.0, timestamp=timestamp)

        hit = self._intercept_position(intercept, n)
        time_s = self._intercept_times(intercept, n)
        horizon = np.clip(time_s, self.limits.min_horizon_s.value, self.limits.max_horizon_s.value)

        rotation = np.stack([self._quat_to_matrix(q) for q in base_pose[:, 3:7]])
        yaw = self._yaw(rotation)
        base_xy = base_pose[:, :2]
        offset = hit[:, :2] - base_xy
        distance = np.linalg.norm(offset, axis=1)
        approaching = distance > 1e-9
        direction = np.where(approaching[:, None],
                             offset / np.where(approaching, distance, 1.0)[:, None],
                             np.stack([np.cos(yaw), np.sin(yaw)], axis=1))

        # Stationing demand is derived in the court frame ...
        station_xy = hit[:, :2] - float(self.limits.standoff_m.value) * direction
        velocity_court = (station_xy - base_xy) / horizon[:, None]
        yaw_desired = np.where(approaching, np.arctan2(offset[:, 1], offset[:, 0]), yaw)
        # ... and the translational part is rotated into robot_base before any wheel reasoning
        # (the verdict: base_twist is a body-frame field because Execution has no pose).
        # A rotation about the court +Z axis leaves the yaw rate unchanged.
        velocity_body = np.einsum('nji,nj->ni', rotation[:, :2, :2], velocity_court)
        yaw_rate = self._wrap_to_pi(yaw_desired - yaw) / horizon
        desired_twist = np.concatenate([velocity_body, yaw_rate[:, None]], axis=1)

        twist, clip_scale, steer_exceeded, steer_max = self._apply_wheel_limits(desired_twist)

        desired_contact = self._desired_contact(intercept, hit, n)
        current_contact = np.asarray(state.racket_contact_pose, dtype=float)[:, :3]
        residual_base = np.einsum('nji,nj->ni', rotation, desired_contact - current_contact)
        joint_delta = (self._contact_jacobian_pinv @ residual_base.T).T
        rate_bound = self.limits.max_joint_rate_rad_s.value * horizon[:, None]
        joint_delta = np.clip(np.clip(joint_delta, -rate_bound, rate_bound),
                              -self.limits.max_joint_offset_rad.value,
                              self.limits.max_joint_offset_rad.value)
        joint_target = joint_pos + joint_delta

        self.last_diagnostics = {'hold': False, 'clip_scale': clip_scale, 'horizon_s': horizon,
                                 'steer_limit_exceeded': steer_exceeded,
                                 'steer_equivalent_max_rad': steer_max,
                                 'joint_delta_rad': joint_delta,
                                 'twist_desired_body': desired_twist,
                                 'velocity_court': velocity_court}
        return WholeBodyTarget(base_twist=twist, joint_position_target=joint_target,
                               horizon_s=float(np.min(horizon)), timestamp=timestamp)

    # -- pieces ------------------------------------------------------------------------------
    @staticmethod
    def _yaw(rotation: np.ndarray) -> np.ndarray:
        """Base yaw about the court +Z axis, read off the frozen quat_to_matrix output."""
        return np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])

    @staticmethod
    def _steer_equivalent(steer_rad: float) -> float:
        """Smallest |steer| among the equivalent representations of one wheel direction.

        (theta, omega) and (theta +- pi, -omega) command the same physical motion - the frozen
        kinematics round-trip test relies on exactly that - so the meaningful question is
        whether any representation fits the limit, not whether the first one does.
        """
        magnitude = abs(float(steer_rad))
        return min(magnitude, abs(math.pi - magnitude))

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _intercept_position(intercept, n: int) -> np.ndarray:
        hit = np.asarray(intercept.position, dtype=float)
        if hit.shape != (n, 3):
            raise BrainBoundaryError("BestIntercept.position must be (%d, 3), got %s" % (n, hit.shape))
        if not np.all(np.isfinite(hit)):
            raise BrainBoundaryError("BestIntercept.position contains NaN/Inf")
        return hit

    @staticmethod
    def _intercept_times(intercept, n: int) -> np.ndarray:
        if getattr(intercept, 'time_s', None) is None:
            raise BrainBoundaryError("BestIntercept.time_s is required for planning")
        time_s = np.asarray(intercept.time_s, dtype=float).reshape(-1)
        if time_s.shape != (n,):
            raise BrainBoundaryError("BestIntercept.time_s must be (%d,), got %s" % (n, time_s.shape))
        if not np.all(np.isfinite(time_s)):
            raise BrainBoundaryError("BestIntercept.time_s contains NaN/Inf")
        if np.any(time_s < 0.0):
            raise BrainBoundaryError("BestIntercept.time_s must be >= 0 (simulation time)")
        return time_s

    @staticmethod
    def _desired_contact(intercept, hit: np.ndarray, n: int) -> np.ndarray:
        pose = getattr(intercept, 'racket_pose', None)
        if pose is None:
            return hit
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (n, 7):
            raise BrainBoundaryError("BestIntercept.racket_pose must be (%d, 7), got %s" % (n, pose.shape))
        return pose[:, :3]

    def _apply_wheel_limits(self, desired_twist: np.ndarray):
        """Validate a robot_base twist against the frozen kinematics limits, per environment.

        The drive limit is enforced exactly: the Morph One inverse kinematics is linear in the
        twist, so one uniform scale multiplies every wheel rate by the same factor and leaves
        every steer angle unchanged - the smallest correction is also the complete one for the
        speed limit.  The steer limit is checked on the smallest equivalent representation (see
        _steer_equivalent); it cannot be repaired by scaling, so it is reported instead of
        silently ignored (Safety owns final enforcement).
        """
        limit = self.max_wheel_speed_rad_s
        steer_limit = self.max_steer_angle_rad
        n = desired_twist.shape[0]
        twist = np.array(desired_twist, dtype=float, copy=True)
        clip_scale = np.ones(n)
        steer_exceeded = np.zeros(n, dtype=bool)
        steer_max = np.zeros(n)
        for i in range(n):
            targets = self._twist_to_wheels(twist[i], self.wheel_positions, self.wheel_radius_m)
            peak = max(abs(target.wheel_speed_rad_s) for target in targets.values())
            if peak > limit:
                clip_scale[i] = limit / peak
                twist[i] = twist[i] * clip_scale[i]
                targets = self._twist_to_wheels(twist[i], self.wheel_positions, self.wheel_radius_m)
            steer_max[i] = max(self._steer_equivalent(target.steer_angle_rad)
                               for target in targets.values())
            steer_exceeded[i] = bool(steer_max[i] > steer_limit + 1e-9)
        return twist, clip_scale, steer_exceeded, steer_max


__all__ = ["ExpertPlanner", "PlannerLimits", "TEMP_CONTACT_JACOBIAN", "TEMP_WHEEL_POSITIONS"]
