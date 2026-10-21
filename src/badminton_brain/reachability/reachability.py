# -*- coding: utf-8 -*-
"""P7-09: REACHABLE / NOT_REACHABLE / UNKNOWN, with the reason that produced it.

WHAT THIS MODULE DECIDES, AND WHAT IT REFUSES TO DECIDE
------------------------------------------------------
Input: a target racket-face pose (or a predicted intercept point plus an approach
direction) and the robot's kinematic limits.
Output: one of three decisions, a typed reason, and the evidence that produced it.

  REACHABLE      an IK solver CONVERGED to the documented tolerance, the joint
                 solution keeps the joint-limit margin, the face centre is inside
                 the workspace box and outside the self-collision proxy, and the
                 available time is enough for the required joint slew.
  NOT_REACHABLE  one of those checks failed.  The reason names WHICH one; the
                 verdict is a refusal, never a "close enough" pose.
  UNKNOWN        the question cannot be answered with what is available: no IK
                 solver, or no verified calibration.  UNKNOWN is a first-class
                 result because inventing REACHABLE would be a safety claim with
                 no evidence behind it.

Check order is fixed and documented so a verdict is reproducible:
    calibration -> finiteness/dimension -> time_to_target -> workspace box ->
    self-collision proxy -> time-sufficency -> IK -> joint-limit margin
The FIRST failing check decides the reason, so a test can pin one cause at a time.

The self-collision proxy is a CYLINDER around the base column plus a face sphere.
It is a proxy, not a collision check: the real arm's link geometry and the real
racket mesh are not modelled.  It only catches "the planner asked the racket to
sit inside the base".

The time-sufficency check is a LOWER BOUND: required slew time = max over joints
of |dq_i| / v_cap_i, compared against the time left to the intercept.  It ignores
the base (Morph One) travel, which the P6 intercept search owns.  v_cap is the
P5-05 POLICY CAP (2.0 rad/s revolute), not a datasheet number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .ik import (IkNotConverged, IkNotImplemented, IkResult, IkSolver, IkTarget,
                 UnavailableIk, DampedLeastSquaresIk)
from .kinematics import (DOF, JOINT_LOWER_RAD, JOINT_NAMES, JOINT_UPPER_RAD,
                         MAX_FACE_REACH_M, PROXY_SOURCE)

__all__ = [
    "ReachabilityDecision", "ReachabilityReason", "DetectionDimensionalityError",
    "Calibration", "KinematicLimits", "ReachabilityVerdict", "P5_VELOCITY_CAP_RAD_S",
    "P5_ACCELERATION_CAP_RAD_S2", "DEFAULT_JOINT_MARGIN_RAD",
    "target_from_intercept", "assess_reachability", "require_world_3d",
]

#: P5-05 policy caps (simulation/badminton_scene/safety.py): revolute defaults.
#: POLICY CAP - unverified; the URDF carries no velocity limit for these joints.
P5_VELOCITY_CAP_RAD_S = 2.0
P5_ACCELERATION_CAP_RAD_S2 = 20.0

DEFAULT_JOINT_MARGIN_RAD = 0.05


class DetectionDimensionalityError(AssertionError):
    """A 2-D detection was about to be promoted to a world 3-D point.

    Subclasses AssertionError because this is an invariant violation, not a
    recoverable input error: there is no calibrated depth behind a 2-D box, and
    any 3-D point derived from it is fiction.
    """


class ReachabilityDecision(str, Enum):
    REACHABLE = "REACHABLE"
    NOT_REACHABLE = "NOT_REACHABLE"
    UNKNOWN = "UNKNOWN"


class ReachabilityReason(str, Enum):
    IK_CONVERGED = "IK_CONVERGED"
    JOINT_LIMIT_MARGIN = "JOINT_LIMIT_MARGIN"
    WORKSPACE_BOUND = "WORKSPACE_BOUND"
    TIME_TO_TARGET = "TIME_TO_TARGET"
    SELF_COLLISION_PROXY = "SELF_COLLISION_PROXY"
    IK_NOT_CONVERGED = "IK_NOT_CONVERGED"
    IK_NOT_IMPLEMENTED = "IK_NOT_IMPLEMENTED"
    CALIBRATION_MISSING = "CALIBRATION_MISSING"
    NON_FINITE_TARGET = "NON_FINITE_TARGET"
    COVARIANCE_ABOVE_THRESHOLD = "COVARIANCE_ABOVE_THRESHOLD"


@dataclass(frozen=True)
class Calibration:
    """The extrinsic that makes a court-frame target mean something.

    verified=False is the honest state of this repository: there is no D455
    calibration and no measured base placement, so extrinsics are unknown.
    """

    court_from_base: Optional[Sequence[Sequence[float]]] = None
    verified: bool = False
    source: str = "UNVERIFIED"

    def usable(self) -> bool:
        return bool(self.verified) and self.court_from_base is not None


@dataclass(frozen=True)
class KinematicLimits:
    """Joint + workspace limits the reachability decision is made against."""

    lower_rad: Tuple[float, ...] = tuple(float(v) for v in JOINT_LOWER_RAD)
    upper_rad: Tuple[float, ...] = tuple(float(v) for v in JOINT_UPPER_RAD)
    joint_margin_rad: float = DEFAULT_JOINT_MARGIN_RAD
    workspace_min_m: Tuple[float, float, float] = (0.10, -0.85, 0.05)
    workspace_max_m: Tuple[float, float, float] = (1.25, 0.85, 1.90)
    velocity_cap_rad_s: Tuple[float, ...] = (P5_VELOCITY_CAP_RAD_S,) * DOF
    base_column_radius_m: float = 0.18
    face_sphere_radius_m: float = 0.06
    max_position_sigma_m: float = 0.05

    def __post_init__(self) -> None:
        if len(self.lower_rad) != DOF or len(self.upper_rad) != DOF:
            raise ValueError(f"limits must carry {DOF} entries per side")
        for lo, hi in zip(self.lower_rad, self.upper_rad):
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
                raise ValueError(f"bad joint interval [{lo}, {hi}]")
        if self.joint_margin_rad < 0.0:
            raise ValueError("joint_margin_rad must be >= 0")
        if len(self.velocity_cap_rad_s) != DOF or min(self.velocity_cap_rad_s) <= 0.0:
            raise ValueError("velocity_cap_rad_s must be > 0 for every joint")
        mins = np.asarray(self.workspace_min_m, dtype=float)
        maxs = np.asarray(self.workspace_max_m, dtype=float)
        if mins.shape != (3,) or maxs.shape != (3,):
            raise ValueError("workspace bounds must be (3,)")
        if np.any(mins >= maxs):
            raise ValueError("workspace_min_m must be below workspace_max_m on every axis")


@dataclass(frozen=True)
class ReachabilityVerdict:
    """The decision plus everything needed to re-derive it."""

    decision: ReachabilityDecision
    reason: ReachabilityReason
    detail: str
    target_position_m: Optional[Tuple[float, float, float]] = None
    time_to_target_s: Optional[float] = None
    joint_margin_rad: Optional[float] = None
    required_slew_s: Optional[float] = None
    ik: Optional[IkResult] = None
    evidence_class: str = "OFFLINE-SYNTHETIC"

    @property
    def reachable(self) -> bool:
        return self.decision is ReachabilityDecision.REACHABLE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "target_position_m": None if self.target_position_m is None
            else [float(v) for v in self.target_position_m],
            "time_to_target_s": self.time_to_target_s,
            "joint_margin_rad": self.joint_margin_rad,
            "required_slew_s": self.required_slew_s,
            "ik": None if self.ik is None else self.ik.as_dict(),
            "evidence_class": self.evidence_class,
        }


def require_world_3d(point: object, *, source: str) -> np.ndarray:
    """Return a (3,) point, or raise DetectionDimensionalityError.

    The single gate that stops a 2-D detection from being promoted to world 3-D.
    A 2-D point (shape (2,)), a box, or None is refused BY NAME - there is no
    fallback depth, no "assume z = ball height".
    """
    if point is None:
        raise DetectionDimensionalityError(
            f"{source}: 2-D-only detection has NO world 3-D point; refusing to promote it")
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise DetectionDimensionalityError(
            f"{source}: expected a world 3-D point of shape (3,), got {arr.shape}; a 2-D "
            "detection must never be promoted to world 3-D")
    if not np.all(np.isfinite(arr)):
        raise DetectionDimensionalityError(f"{source}: 3-D point contains NaN/Inf")
    return arr


def target_from_intercept(intercept_point_m: Sequence[float],
                          approach_direction: Sequence[float],
                          *, up_hint: Sequence[float] = (0.0, 0.0, 1.0)) -> IkTarget:
    """Build a face-pose target from an intercept point + approach direction.

    Racket-face convention (badminton_scene/racket_cfg.py): +X is the face
    NORMAL, +Z runs handle -> head.  The approach direction becomes +X; +Z is the
    component of up_hint orthogonal to it, so the face is "upright" in the sense
    the racket asset defines.  Degenerate inputs raise InvalidIkTarget through
    IkTarget / this function's own checks.
    """
    point = require_world_3d(intercept_point_m, source="target_from_intercept")
    x_axis = np.asarray(approach_direction, dtype=float).reshape(-1)
    if x_axis.shape != (3,) or not np.all(np.isfinite(x_axis)):
        raise ValueError("approach_direction must be a finite 3-vector")
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-9:
        raise ValueError("approach_direction must be non-zero")
    x_axis = x_axis / norm
    up = np.asarray(up_hint, dtype=float).reshape(-1)
    if up.shape != (3,) or not np.all(np.isfinite(up)):
        raise ValueError("up_hint must be a finite 3-vector")
    z_axis = up - float(np.dot(up, x_axis)) * x_axis
    if float(np.linalg.norm(z_axis)) < 1e-6:
        # The hint is parallel to the approach (a straight-down or straight-up swing).
        # Fall back DETERMINISTICALLY to the world axis least aligned with the
        # approach, so the same inputs always produce the same face frame.  Which
        # roll is actually correct needs a measured TCP - that is an UNKNOWN.
        alt = np.zeros(3)
        alt[int(np.argmin(np.abs(x_axis)))] = 1.0
        z_axis = alt - float(np.dot(alt, x_axis)) * x_axis
    z_axis = z_axis / float(np.linalg.norm(z_axis))
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return IkTarget(position_m=point, rotation=rotation)


def _joint_margin(q: np.ndarray, limits: KinematicLimits) -> Tuple[float, Optional[int]]:
    lower = np.asarray(limits.lower_rad, dtype=float)
    upper = np.asarray(limits.upper_rad, dtype=float)
    margins_low = q - lower
    margins_high = upper - q
    margins = np.minimum(margins_low, margins_high)
    index = int(np.argmin(margins))
    return float(margins[index]), index


def assess_reachability(
    *,
    target: Optional[IkTarget] = None,
    intercept_point_m: Optional[Sequence[float]] = None,
    approach_direction: Optional[Sequence[float]] = None,
    limits: Optional[KinematicLimits] = None,
    time_to_target_s: Optional[float] = None,
    time_margin_s: float = 0.02,
    seed_rad: Optional[Sequence[float]] = None,
    solver: Optional[IkSolver] = None,
    calibration: Optional[Calibration] = None,
    position_sigma_m: Optional[float] = None,
    detection_dim: int = 3,
) -> ReachabilityVerdict:
    """Decide REACHABLE / NOT_REACHABLE / UNKNOWN for one racket-face target."""
    limits = limits or KinematicLimits()
    solver = solver or DampedLeastSquaresIk()
    calibration = calibration or Calibration()

    def _verdict(decision: ReachabilityDecision, reason: ReachabilityReason, detail: str,
                 **kwargs: Any) -> ReachabilityVerdict:
        return ReachabilityVerdict(decision=decision, reason=reason, detail=detail,
                                   target_position_m=position, **kwargs)

    position: Optional[Tuple[float, float, float]] = None

    # 1. calibration - without it nothing downstream means anything
    if not calibration.usable():
        return ReachabilityVerdict(
            decision=ReachabilityDecision.UNKNOWN,
            reason=ReachabilityReason.CALIBRATION_MISSING,
            detail="no verified court_from_base extrinsic: a target pose cannot be tied to "
                   "the court, so reachability is UNKNOWN rather than guessed",
        )

    # 2. dimensionality - a 2-D detection is never promoted
    if detection_dim != 3:
        raise DetectionDimensionalityError(
            f"detection_dim={detection_dim}: only a calibrated 3-D point may be tested for "
            "reachability")

    # 3. build the target
    if target is None:
        if intercept_point_m is None or approach_direction is None:
            return ReachabilityVerdict(
                decision=ReachabilityDecision.UNKNOWN,
                reason=ReachabilityReason.NON_FINITE_TARGET,
                detail="neither a target pose nor an intercept point + approach direction "
                       "was supplied",
            )
        try:
            target = target_from_intercept(intercept_point_m, approach_direction)
        except DetectionDimensionalityError:
            raise
        except (ValueError, TypeError) as exc:
            return ReachabilityVerdict(
                decision=ReachabilityDecision.UNKNOWN,
                reason=ReachabilityReason.NON_FINITE_TARGET,
                detail=f"target could not be built: {exc}",
            )
    if not isinstance(target, IkTarget):
        return ReachabilityVerdict(
            decision=ReachabilityDecision.UNKNOWN,
            reason=ReachabilityReason.NON_FINITE_TARGET,
            detail=f"target must be an IkTarget, got {type(target).__name__}",
        )
    position = (float(target.position_m[0]), float(target.position_m[1]),
                float(target.position_m[2]))

    # 4. time to target must be positive AND large enough
    if time_to_target_s is None:
        return _verdict(ReachabilityDecision.UNKNOWN, ReachabilityReason.TIME_TO_TARGET,
                        "time_to_target_s is required; reachability in time cannot be "
                        "assessed without it")
    t = float(time_to_target_s)
    if not math.isfinite(t) or t <= 0.0:
        return _verdict(ReachabilityDecision.NOT_REACHABLE, ReachabilityReason.TIME_TO_TARGET,
                        f"time_to_target_s={time_to_target_s!r} is zero, negative or "
                        "non-finite: a target that is already due cannot be reached",
                        time_to_target_s=(t if math.isfinite(t) else None))

    # 5. workspace box
    mins = np.asarray(limits.workspace_min_m, dtype=float)
    maxs = np.asarray(limits.workspace_max_m, dtype=float)
    p = np.asarray(target.position_m, dtype=float)
    outside = (p < mins) | (p > maxs)
    if bool(np.any(outside)):
        bad = ["xyz"[i] for i in range(3) if outside[i]]
        return _verdict(
            ReachabilityDecision.NOT_REACHABLE, ReachabilityReason.WORKSPACE_BOUND,
            f"face centre {position} is outside the workspace box on {bad}", )

    # 6. self-collision proxy: the base column cylinder
    radial = float(math.hypot(float(p[0]), float(p[1])))
    if radial < limits.base_column_radius_m + limits.face_sphere_radius_m:
        return _verdict(
            ReachabilityDecision.NOT_REACHABLE, ReachabilityReason.SELF_COLLISION_PROXY,
            f"face centre is {radial:.3f} m from the base column axis, inside the "
            f"{limits.base_column_radius_m + limits.face_sphere_radius_m:.3f} m proxy "
            "(racket would sit inside the base); PROXY, not a collision check")

    # 7. uncertainty gate - a target that is only known to +-sigma is not reachable-on-purpose
    if position_sigma_m is not None:
        sigma = float(position_sigma_m)
        if not math.isfinite(sigma) or sigma < 0.0:
            return _verdict(ReachabilityDecision.NOT_REACHABLE,
                            ReachabilityReason.NON_FINITE_TARGET,
                            f"position_sigma_m={position_sigma_m!r} is not a usable sigma")
        if sigma > limits.max_position_sigma_m:
            return _verdict(
                ReachabilityDecision.NOT_REACHABLE,
                ReachabilityReason.COVARIANCE_ABOVE_THRESHOLD,
                f"position sigma {sigma:.3f} m exceeds the {limits.max_position_sigma_m:.3f} m "
                "decision threshold: the intercept is not localised well enough to commit")

    # 8. IK - the only place a joint vector may come from
    seed = np.zeros(DOF) if seed_rad is None else np.asarray(seed_rad, dtype=float).reshape(-1)
    if seed.shape != (DOF,):
        return _verdict(ReachabilityDecision.UNKNOWN, ReachabilityReason.NON_FINITE_TARGET,
                        f"seed_rad must have {DOF} entries, got {seed.shape}")
    try:
        result = solver.solve(target, seed)
    except IkNotImplemented as exc:
        return _verdict(ReachabilityDecision.UNKNOWN, ReachabilityReason.IK_NOT_IMPLEMENTED,
                        f"{exc} -- geometry may allow this target, but no solver can say so")
    except IkNotConverged as exc:
        return _verdict(
            ReachabilityDecision.NOT_REACHABLE, ReachabilityReason.IK_NOT_CONVERGED,
            f"IK did not converge: residual {exc.residual_position_m:.4f} m / "
            f"{exc.residual_rotation_rad:.4f} rad (tol {exc.tol_position_m:.1e} m / "
            f"{exc.tol_rotation_rad:.1e} rad); no approximate pose is emitted",
            time_to_target_s=t)

    q = np.asarray(result.joints_rad, dtype=float)
    # 9. joint-limit margin on the CONVERGED solution
    margin, index = _joint_margin(q, limits)
    if margin < limits.joint_margin_rad:
        return _verdict(
            ReachabilityDecision.NOT_REACHABLE, ReachabilityReason.JOINT_LIMIT_MARGIN,
            f"{JOINT_NAMES[index]} sits {margin:.4f} rad from its stop, inside the "
            f"{limits.joint_margin_rad:.4f} rad margin required by the P5-05 envelope",
            time_to_target_s=t, joint_margin_rad=margin, ik=result)

    # 10. time sufficency for the required slew
    required = float(np.max(np.abs(q - np.clip(seed, limits.lower_rad, limits.upper_rad))
                            / np.asarray(limits.velocity_cap_rad_s, dtype=float)))
    if required > t - float(time_margin_s):
        return _verdict(
            ReachabilityDecision.NOT_REACHABLE, ReachabilityReason.TIME_TO_TARGET,
            f"required slew {required:.4f} s exceeds the {t - float(time_margin_s):.4f} s "
            f"available (t={t:.4f} s, margin {float(time_margin_s):.4f} s) at the P5-05 "
            "velocity cap; lower bound only, base travel excluded",
            time_to_target_s=t, joint_margin_rad=margin, required_slew_s=required, ik=result)

    return _verdict(
        ReachabilityDecision.REACHABLE, ReachabilityReason.IK_CONVERGED,
        f"IK converged (residual {result.residual_position_m:.2e} m / "
        f"{result.residual_rotation_rad:.2e} rad) with joint margin {margin:.4f} rad and "
        f"{required:.4f} s of slew inside a {t:.4f} s target",
        time_to_target_s=t, joint_margin_rad=margin, required_slew_s=required, ik=result)
