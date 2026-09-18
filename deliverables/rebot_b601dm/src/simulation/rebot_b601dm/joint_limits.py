"""Per-joint limits for the reBot B601-DM, each carrying where it came from.

WHY PROVENANCE IS A FIELD AND NOT A COMMENT.

The shipped simulation asset enforces `physxJoint:maxJointVelocity` of 2864.789 and 11459.156 deg/s, which
are the real motors' rpm figures used as rad/s: the DM4340 runs at 52.5 rpm and the DM4310 at 200 rpm, so the
asset permits 9.09x and 9.55x the real no-load speed. That is a mistake of exactly the kind a comment does
not prevent. Here every limit is a `Limit` carrying a `Provenance`, and `limits_for(..., require_real=True)`
refuses to hand back a value that is not `CONFIRMED_REAL`.

Sources, all verified by inspection (`management/rebot_b601dm/REBOT_B601DM_REPORT.md`):
  URDF      reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf
  USD       reBot-Isaacsim/usd/reBot_B601_DM/payloads/Physics/{physx,physics}.usda
  BOM       reBot-DevArm/hardware/reBot_B601_DM/readme.md:141-142
  motors    SeeedStudio official table, rows J4340-2EC and J4310-2EC V1.1
            (https://wiki.seeedstudio.com/damiao_series/)

Two caveats that the values below do NOT hide:
  1. The official motor table documents the `-2EC` variants while the BOM specifies `DM4340P(V4)` and
     `DM4310(V4)`. Peak torques agree at 27 and 7 N.m, which says same family, but V4-specific numbers were
     not found. Everything real here carries that caveat.
  2. No hardware document in either repository states a joint ANGLE range. Those stay UNKNOWN even though
     the URDF has values, because the URDF is simulation, not hardware.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Provenance(str, Enum):
    """Where a number came from. SIM_ONLY may never support a capability claim."""

    CONFIRMED_REAL = "CONFIRMED_REAL"
    SIM_ONLY = "SIM_ONLY"
    UNKNOWN = "UNKNOWN"


def rad_s_from_rpm(rpm: float) -> float:
    """Convert a motor output speed in rpm to rad/s."""
    return rpm * 2.0 * math.pi / 60.0


@dataclass(frozen=True)
class Limit:
    """One limit value together with where it came from.

    `value` is the number to USE. For velocity it is the real motor speed; `sim_value` keeps what the shipped
    asset declares, so the discrepancy stays visible instead of being overwritten.
    """

    value: float | None
    provenance: Provenance
    source: str = ""
    sim_value: float | None = None
    sim_overshoot: float | None = None
    lower: float | None = None
    upper: float | None = None
    rated: float | None = None
    is_peak: bool = False


@dataclass(frozen=True)
class JointSpec:
    """A single joint: its limits, its motor, and its transmission."""

    name: str
    kind: str
    motor: str
    reduction_ratio: int
    angle: Limit
    velocity: Limit
    torque: Limit


URDF_SRC = "reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf"
USD_SRC = "reBot-Isaacsim/usd/reBot_B601_DM/payloads/Physics/physx.usda"
BOM_SRC = "reBot-DevArm/hardware/reBot_B601_DM/readme.md:141-142"
MOTOR_SRC = "SeeedStudio damiao_series table (J4340-2EC / J4310-2EC V1.1)"

# Official motor figures, verbatim from the SeeedStudio table.
DM4340 = {"rpm_no_load": 52.5, "rpm_rated": 36.0, "peak_nm": 27.0, "rated_nm": 9.0, "ratio": 40}
DM4310 = {"rpm_no_load": 200.0, "rpm_rated": 120.0, "peak_nm": 7.0, "rated_nm": 3.0, "ratio": 10}

MOTOR_ASSIGNMENT: Mapping[str, str] = {
    "joint1": "DM4340P", "joint2": "DM4340P", "joint3": "DM4340P",
    "joint4": "DM4310", "joint5": "DM4310", "joint6": "DM4310",
    # the seventh motor drives the gripper; the BOM lists 3x DM4340P + 4x DM4310
    "gripper_joint1": "DM4310", "gripper_joint2": "DM4310",
}


def _arm_joint(name: str, motor: str, spec: dict, angle_lower: float, angle_upper: float,
               sim_velocity: float) -> JointSpec:
    real_velocity = rad_s_from_rpm(spec["rpm_no_load"])
    return JointSpec(
        name=name,
        kind="revolute",
        motor=motor,
        reduction_ratio=spec["ratio"],
        angle=Limit(value=None, provenance=Provenance.UNKNOWN, source="",
                    lower=angle_lower, upper=angle_upper),
        velocity=Limit(
            value=real_velocity,
            provenance=Provenance.CONFIRMED_REAL,
            source=MOTOR_SRC + " no-load " + repr(spec["rpm_no_load"]) + " rpm",
            sim_value=sim_velocity,
            sim_overshoot=sim_velocity / real_velocity,
        ),
        torque=Limit(
            value=spec["peak_nm"],
            provenance=Provenance.CONFIRMED_REAL,
            source=MOTOR_SRC + " peak",
            sim_value=spec["peak_nm"],
            rated=spec["rated_nm"],
            is_peak=True,
        ),
    )


JOINT_LIMITS: Mapping[str, JointSpec] = {
    # angle values are the URDF's, kept for reference but tagged UNKNOWN: they are simulation, not hardware
    "joint1": _arm_joint("joint1", "DM4340P", DM4340, -2.8, 2.8, 50.0),
    "joint2": _arm_joint("joint2", "DM4340P", DM4340, -3.14, 0.0, 50.0),
    "joint3": _arm_joint("joint3", "DM4340P", DM4340, -3.14, 0.0, 50.0),
    "joint4": _arm_joint("joint4", "DM4310", DM4310, -1.87, 1.57, 200.0),
    "joint5": _arm_joint("joint5", "DM4310", DM4310, -1.57, 1.57, 200.0),
    "joint6": _arm_joint("joint6", "DM4310", DM4310, -3.14, 3.14, 200.0),
    # the gripper: geometry from the URDF, but no hardware source for its force or speed
    # A SIM_ONLY value still needs a source, saying where the simulation number came from. "It is only a sim
    # value" is not a reason to leave it untraceable - the point of the tag is that a reader can go and look.
    "gripper_joint1": JointSpec(
        name="gripper_joint1", kind="prismatic", motor="DM4310", reduction_ratio=DM4310["ratio"],
        angle=Limit(value=None, provenance=Provenance.UNKNOWN, source="", lower=0.0, upper=0.0715),
        velocity=Limit(value=None, provenance=Provenance.SIM_ONLY,
                       source=URDF_SRC + " limit velocity 15 (simulation only, no hardware source)",
                       sim_value=15.0),
        torque=Limit(value=None, provenance=Provenance.UNKNOWN, source="",
                     sim_value=100.0, rated=None),
    ),
    "gripper_joint2": JointSpec(
        name="gripper_joint2", kind="prismatic", motor="DM4310", reduction_ratio=DM4310["ratio"],
        angle=Limit(value=None, provenance=Provenance.UNKNOWN, source="", lower=0.0, upper=0.0715),
        velocity=Limit(value=None, provenance=Provenance.SIM_ONLY,
                       source=URDF_SRC + " limit velocity 15 (simulation only, no hardware source)",
                       sim_value=15.0),
        torque=Limit(value=None, provenance=Provenance.UNKNOWN, source="",
                     sim_value=100.0, rated=None),
    ),
}


def limits_for(name: str, *, require_real: bool = False) -> JointSpec:
    """Return the spec for a joint.

    With `require_real=True` this refuses to return a joint any of whose limits is not CONFIRMED_REAL. It is
    deliberately strict across all three kinds rather than only the one the caller has in mind: a joint whose
    ANGLE range is unknown is not fully real either, and silently returning it would be the same failure mode
    as trusting the shipped velocities.
    """
    try:
        spec = JOINT_LIMITS[name]
    except KeyError:
        raise KeyError("unknown joint: " + repr(name)) from None
    if not require_real:
        return spec
    not_real = [
        kind for kind in ("angle", "velocity", "torque")
        if getattr(spec, kind).provenance != Provenance.CONFIRMED_REAL
    ]
    if not_real:
        raise ValueError(
            "joint " + name + " is not fully real; not CONFIRMED_REAL: " + ", ".join(not_real)
        )
    return spec