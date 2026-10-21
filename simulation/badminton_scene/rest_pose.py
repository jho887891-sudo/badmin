# -*- coding: utf-8 -*-
"""Per-robot arm rest pose, keyed by JOINT NAME, valid only inside that robot's joint limits.

WHY THIS MODULE EXISTS
    simulation/badminton_scene/reset.py used to carry one REST_JOINT dict - the PiPER Stage-0 pose,
    joint2 = 1.5 / joint3 = -1.5 - and write it on whatever articulation the scene had loaded. The
    formal arm is the reBot B601-RS, whose joint3 range is [0, 3.14] rad: -1.5 rad is outside it, so
    every reset that did not pass joint_pos wrote a limit-violating state. P5 found and pinned it
    (management/morph_one_rs/P5_ARM_STATIC_CONSISTENCY.md, F1); P5B fixes it here.

    The pose is therefore data, per robot, addressed by joint NAME. It is never addressed by DOF
    index: on this asset the USD stage traversal order (joint6, gripper_joint1, gripper_joint2,
    joint5..joint1, root_joint) is neither the URDF order nor the DOF order (P5 F2).

WHAT "VALID" MEANS HERE
    Every rest value is STRICTLY inside its joint's limits, by at least 5% of that joint's travel
    from the nearer stop:

        required_margin = 0.05 * travel          (5% of range)

    in the joint's own unit (rad for revolute, m for prismatic). The rule is relative, not a fixed
    0.05 rad, because the two finger joints travel 0.05 m and 0.0715 m: 5% of 0.05 m is 0.0025 m,
    whereas "0.05 rad" would demand a margin larger than a finger's whole stroke. A value sitting
    exactly on a limit is a violation, not a pass: it is what put the URDF zero pose against the
    joint stops in the first place (see simulation/piper_stage0/piper_cfg.py comments) and it
    leaves PhysX no room to hold the joint without pushing.

FAILURE MODE (the caller cannot ignore it)
    resolve_model() / rest_joint_pos() / validate_rest_pose() RAISE; they never return a sentinel.
    reset_idx() has no return value today, so an error return would be droppable and the caller
    (scripts/t06_reset_local.py, t07_local.py, t08_local.py) would keep stepping the sim. The
    exception types are ResetPoseError and its two specialisations, UnknownRobotModel and
    UnknownJoint, plus RestPoseLimitViolation.

WHERE THE NUMBERS COME FROM
    PiPER       rest = the Stage-0 verified pose (simulation/piper_stage0/piper_cfg.py
                PIPER_REST_JOINT_POS); limits = the AgileX URDF values recorded in that same Stage-0
                tree (simulation/piper_stage0/t02_articulation_checks.py:41-49).
    B601-RS     rest = the pose the formal scene spawns the arm at (joint2 = 0.3, joint3 = 1.0; the
                PiPER pair is unreachable on this arm, P5 section 5.2) with the two gripper fingers
                at mid-travel; limits = the official URDF range, cross-checked against the converted
                USD to 8.3e-8 rad by P5 and re-checked read-only by
                tests/simulation/arm_pose/test_rs_rest_pose_against_usda.py.
    Both limit sets are OFFICIAL_URDF: model values, not measurements of a physical unit. That
    caveat travels with src/simulation/rebot_b601_rs/joint_limits.py (usable_for_capability=False)
    and is not weakened here.

NO RUNTIME DEPENDENCIES
    No torch, no numpy, no pxr. This module is deliberately importable on any machine, because the
    contract it encodes is exactly what a plain CI box has to be able to check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

MODEL_PIPER = "piper"
MODEL_REBOT_RS = "rebot_b601_rs"

#: The rule: every rest value sits at least this fraction of the joint's travel from either stop.
MARGIN_FRACTION = 0.05
#: Only used for a joint with zero travel, where a fraction of the travel is 0 and would accept a
#: value sitting exactly on the stop. No joint in either table is degenerate; this keeps the rule
#: total instead of special-casing the caller.
MARGIN_ABS = 0.05


class ResetPoseError(RuntimeError):
    """Base class: reset_idx() refuses to write a pose it cannot justify."""


class UnknownRobotModel(ResetPoseError):
    """The runtime DOF names match no model this table knows, or an explicit key is unknown."""


class UnknownJoint(ResetPoseError):
    """A joint named at runtime is missing from the model table (or the table has an extra one)."""


class RestPoseLimitViolation(ResetPoseError):
    """A rest value (or an explicitly requested pose) is outside its joint limits / too close to them."""


@dataclass(frozen=True)
class JointLimit:
    """One joint's travel, in the unit the sim uses for that joint."""

    name: str
    kind: str          # "revolute" | "prismatic"
    lower: float
    upper: float
    unit: str          # "rad" | "m"
    source: str        # where the number came from, so a reader can re-check it

    @property
    def travel(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class RestPoseSpec:
    """One robot: its DOF name set, its rest pose and the limits that pose must respect."""

    key: str
    label: str
    aliases: Tuple[str, ...]
    joint_names: Tuple[str, ...]
    rest: Mapping[str, float]
    limits: Mapping[str, JointLimit]
    rest_source: str


# --------------------------------------------------------------------------- limits
_URDF_SRC_PIPER = "AgileX piper_no_gripper_description.urdf"
_URDF_SRC_RS = (
    "reBot-DevArm/Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf <limit lower/upper>; "
    "cross-checked against the converted USD (P5: max 8.3e-8 rad)"
)


def _revolute(name: str, lower: float, upper: float, source: str) -> JointLimit:
    return JointLimit(name=name, kind="revolute", lower=float(lower), upper=float(upper),
                      unit="rad", source=source)


def _prismatic(name: str, lower: float, upper: float, source: str) -> JointLimit:
    return JointLimit(name=name, kind="prismatic", lower=float(lower), upper=float(upper),
                      unit="m", source=source)


_PIPER_LIMITS: Dict[str, JointLimit] = {
    # Recorded from the AgileX URDF in simulation/piper_stage0/t02_articulation_checks.py:41-49.
    # joint1 is asymmetric on this arm; do not "symmetrise" it.
    "joint1": _revolute("joint1", -2.618, 2.168, _URDF_SRC_PIPER),
    "joint2": _revolute("joint2", 0.0, 3.14, _URDF_SRC_PIPER),
    "joint3": _revolute("joint3", -2.967, 0.0, _URDF_SRC_PIPER),
    "joint4": _revolute("joint4", -1.745, 1.745, _URDF_SRC_PIPER),
    "joint5": _revolute("joint5", -1.22, 1.22, _URDF_SRC_PIPER),
    "joint6": _revolute("joint6", -2.0944, 2.0944, _URDF_SRC_PIPER),
}

_RS_LIMITS: Dict[str, JointLimit] = {
    # Official URDF ranges, in radians; the USD authors the same numbers in degrees
    # (joint1 +-160.4281768798828 deg, joint2/joint3 [0, 179.90875244140625] deg, ...).
    "joint1": _revolute("joint1", -2.8, 2.8, _URDF_SRC_RS),
    "joint2": _revolute("joint2", 0.0, 3.14, _URDF_SRC_RS),
    "joint3": _revolute("joint3", 0.0, 3.14, _URDF_SRC_RS),
    "joint4": _revolute("joint4", -1.57, 1.57, _URDF_SRC_RS),
    "joint5": _revolute("joint5", -1.57, 1.57, _URDF_SRC_RS),
    "joint6": _revolute("joint6", -3.14, 3.14, _URDF_SRC_RS),
    "gripper_joint1": _prismatic("gripper_joint1", 0.0, 0.05, _URDF_SRC_RS),
    "gripper_joint2": _prismatic("gripper_joint2", 0.0, 0.0715, _URDF_SRC_RS),
}

# --------------------------------------------------------------------------- rest poses
#: PiPER Stage-0 pose, BIT FOR BIT (plain float literals, no arithmetic applied to them). Existing
#: PiPER runs must not move because the formal arm changed.
PIPER_REST_JOINT: Dict[str, float] = {
    "joint1": 0.0,
    "joint2": 1.5,
    "joint3": -1.5,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
}

#: B601-RS pose. joint2/joint3 are the formal scene's values (the PiPER pair is unreachable: joint3
#: is [0, 3.14]). The fingers sit at mid-travel rather than at 0.0, because 0.0 IS the lower limit
#: of both fingers and a rest pose on a hard stop is not a rest pose. Mid-travel also leaves the
#: gripper visibly open in a reset, which is what a HOLD fallback wants.
REBOT_RS_REST_JOINT: Dict[str, float] = {
    "joint1": 0.0,
    "joint2": 0.3,
    "joint3": 1.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "gripper_joint1": 0.025,    # mid of [0, 0.05]
    "gripper_joint2": 0.03575,  # mid of [0, 0.0715]
}

REST_POSES: Dict[str, RestPoseSpec] = {
    MODEL_PIPER: RestPoseSpec(
        key=MODEL_PIPER,
        label="AgileX PiPER (Stage 0, no gripper)",
        aliases=("piper", "piper_stage0", "agilex_piper", "agilex"),
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        rest=PIPER_REST_JOINT,
        limits=_PIPER_LIMITS,
        rest_source=(
            "simulation/piper_stage0/piper_cfg.py PIPER_REST_JOINT_POS (Stage-0 verified); "
            "mid-range on joint2/joint3 so the URDF zero pose does not rest on a limit"
        ),
    ),
    MODEL_REBOT_RS: RestPoseSpec(
        key=MODEL_REBOT_RS,
        label="reBot B601-RS (formal arm)",
        aliases=("rebot_b601_rs", "rebot_b601rs", "rebot_rs", "rebot_arm_rs", "rebot", "rs"),
        joint_names=(
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            "gripper_joint1", "gripper_joint2",
        ),
        rest=REBOT_RS_REST_JOINT,
        limits=_RS_LIMITS,
        rest_source=(
            "formal scene REBOT_RS['rest_joint_pos'] (joint2 0.3, joint3 1.0) - the PiPER (1.5, -1.5) "
            "pair is unreachable on this arm; fingers at mid-travel (P5B_RESET_POSE.md section 2)"
        ),
    ),
}


# --------------------------------------------------------------------------- helpers
def normalise_key(key: object) -> str:
    """' ReBot-B601-RS ' -> 'rebot_b601_rs'. Keys are matched loosely; joint names never are."""
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _spec(model: object) -> RestPoseSpec:
    if isinstance(model, RestPoseSpec):
        return model
    key = normalise_key(model)
    if key in REST_POSES:
        return REST_POSES[key]
    for spec in REST_POSES.values():
        if key in {normalise_key(alias) for alias in spec.aliases}:
            return spec
    raise UnknownRobotModel(
        "unknown robot model %r; known models: %s"
        % (model, ", ".join(sorted(REST_POSES))))


def resolve_model(joint_names: Optional[Sequence[str]], robot_key: object = None) -> str:
    """Return the model key for this runtime DOF name set, or raise.

    The match is an EXACT set match on names. Anything else - an extra DOF, a missing one, a
    differently named arm - raises UnknownRobotModel instead of falling back to 0.0 per joint: a
    silent 0.0 is how a reset ends up commanding a joint it has never seen.
    """
    names = tuple(str(name) for name in (joint_names or ()))
    if not names:
        raise UnknownRobotModel("the robot reports no joint names; cannot pick a rest pose")

    if robot_key is not None:
        spec = _spec(robot_key)
        if len(names) != len(spec.joint_names) or set(names) != set(spec.joint_names):
            raise ResetPoseError(
                "robot_key %r (%s) expects joints %s, but the robot reports %s"
                % (robot_key, spec.key, sorted(spec.joint_names), sorted(names)))
        return spec.key

    matches = [
        spec for spec in REST_POSES.values()
        if len(names) == len(spec.joint_names) and set(names) == set(spec.joint_names)
    ]
    if len(matches) == 1:
        return matches[0].key
    if not matches:
        raise UnknownRobotModel(
            "no rest pose is defined for joint names %s; known models are %s"
            % (sorted(names),
               "; ".join("%s=%s" % (spec.key, sorted(spec.joint_names))
                         for spec in REST_POSES.values())))
    raise ResetPoseError(
        "joint names %s match more than one model (%s)"
        % (sorted(names), sorted(spec.key for spec in matches)))


def rest_joint_pos(model: object, joint_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """The rest pose of one robot as {joint name: value}, ordered like joint_names when given."""
    spec = _spec(model)
    if joint_names is None:
        return {name: spec.rest[name] for name in spec.joint_names}

    names = tuple(str(name) for name in joint_names)
    missing = [name for name in names if name not in spec.rest]
    extra = [name for name in spec.joint_names if name not in names]
    if missing or extra:
        raise UnknownJoint(
            "the %s rest pose cannot cover this robot: unknown %s, missing %s"
            % (spec.key, missing or "[]", extra or "[]"))
    return {name: spec.rest[name] for name in names}


def limits_for(model: object) -> Dict[str, JointLimit]:
    return dict(_spec(model).limits)


def required_margin(limit: JointLimit) -> float:
    """>= 5% of the joint's travel, in the joint's own unit (rad / m).

    Relative rather than absolute so that a 0.05 m finger stroke and a 3.14 rad wrist get a
    comparable "off the stop" guarantee instead of one rule that is absurd for one of them.
    """
    travel = abs(limit.travel)
    if travel == 0.0:
        return MARGIN_ABS
    return MARGIN_FRACTION * travel


def margin(limit: JointLimit, value: float) -> float:
    """Signed distance to the nearer limit; <= 0 means at, on or beyond a stop."""
    return min(float(value) - limit.lower, limit.upper - float(value))


def check_rest_pose(model: object, values: Mapping[str, float],
                    require_margin: bool = True) -> list:
    """Every problem with this pose, as human-readable lines. Empty list means valid.

    Non-raising companion of validate_rest_pose(), for diagnostics and for callers that want to
    report several offenders at once.
    """
    spec = _spec(model)
    problems = []
    names = tuple(values)
    for name in names:
        if name not in spec.rest:
            problems.append("unknown joint %r for model %s" % (name, spec.key))
    for name in spec.joint_names:
        if name not in values:
            problems.append("model %s joint %r missing from the pose" % (spec.key, name))

    for name in spec.joint_names:
        if name not in values:
            continue
        limit = spec.limits[name]
        value = float(values[name])
        if value != value or value in (float("inf"), float("-inf")):
            problems.append("%s = %r is not a finite number" % (name, value))
            continue
        if not (limit.lower <= value <= limit.upper):
            problems.append(
                "%s = %r outside the %s limit [%r, %r] %s"
                % (name, value, spec.key, limit.lower, limit.upper, limit.unit))
            continue
        if require_margin:
            got = margin(limit, value)
            need = required_margin(limit)
            if got < need:
                problems.append(
                    "%s = %r is only %.6g %s from a limit of the %s range [%r, %r]; "
                    "need >= %.6g (a rest pose must not sit on a stop)"
                    % (name, value, got, limit.unit, spec.key, limit.lower, limit.upper, need))
    return problems


def validate_rest_pose(model: object, values: Mapping[str, float],
                       require_margin: bool = True) -> None:
    """Raise unless every value is inside its joint's limits (and, by default, off the stops)."""
    spec = _spec(model)
    problems = check_rest_pose(spec.key, values, require_margin=require_margin)
    if not problems:
        return
    unknown = [problem for problem in problems if problem.startswith("unknown joint")]
    missing = [problem for problem in problems if "missing from the pose" in problem]
    if unknown or missing:
        raise UnknownJoint("cannot validate a pose for %s: %s" % (spec.key, "; ".join(problems)))
    raise RestPoseLimitViolation(
        "the %s rest pose is not reachable: %s" % (spec.key, "; ".join(problems)))


__all__ = [
    "MODEL_PIPER", "MODEL_REBOT_RS", "MARGIN_ABS", "MARGIN_FRACTION",
    "ResetPoseError", "UnknownRobotModel", "UnknownJoint", "RestPoseLimitViolation",
    "JointLimit", "RestPoseSpec", "REST_POSES",
    "PIPER_REST_JOINT", "REBOT_RS_REST_JOINT",
    "normalise_key", "resolve_model", "rest_joint_pos", "limits_for",
    "required_margin", "margin", "check_rest_pose", "validate_rest_pose",
]
