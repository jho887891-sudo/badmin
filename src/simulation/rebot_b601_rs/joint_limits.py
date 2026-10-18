# -*- coding: utf-8 -*-
"""Per-joint limits for the formal arm: reBot B601-RS, each value carrying where it came from.

WHY THIS FILE EXISTS
    The project arm changes from PiPER to reBot B601-RS. Renaming a PiPER file would carry PiPER
    numbers across the identity boundary, so this table is built from the B601-RS asset itself and
    from the Seeed product pages, never from the PiPER or B601-DM records.

SOURCES (all re-checkable from the repository and the public URLs below)
    URDF    reBot-DevArm/Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf
            https://github.com/Seeed-Projects/reBot-DevArm  @ f01a1dc189ecb74b5435a336a68b287c146c1b60
    BOM     reBot-DevArm/hardware/reBot_B601_RS/README.md: "RobStride RS00 x4, RS06 x3"
    SPEC    reBot-DevArm/README.md hardware table, B601-RS column (payload / reach / weight / DOF)
    MOTOR   https://www.seeedstudio.com/Robostride-00-Actuator-p-6664.html  (RS00: 14 N.m, 10:1)
            https://www.seeedstudio.com/Robostride-06-Actuator-p-6668.html  (RS06: 36 N.m, 9:1, 621 g)

CROSS-CHECK THAT MAKES THE MOTOR ASSIGNMENT REAL, NOT GUESSED
    The URDF effort column is 36 for joint1..joint3 and 14 for joint4..joint6. Those are exactly the
    RS06 and RS00 peak torques. Together with the BOM (4x RS00 + 3x RS06 = 7 motors for 6 arm joints
    + 1 gripper drive) the assignment is consistent from two independent directions.

THREE THINGS THIS TABLE REFUSES TO CLAIM
    1. Angle ranges are the official MODEL's values (OFFICIAL_URDF). No hardware document in the
       repository states a joint angle range, so they may not back a real-robot capability claim.
    2. Joint speed 50/40 in the URDF is NOT a verified motor output speed. The B601-DM line already
       burned the project once by shipping rpm figures as rad/s (9.09x / 9.55x overshoot); the same
       signature is present here, so velocity is OFFICIAL_URDF and gated out of capability claims.
    3. The gripper URDF has TWO independent prismatic fingers and NO mimic tag, while the BOM has ONE
       gripper drive; the fingers are hardware-coupled and the coupling direction is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping

MODEL = "reBot B601-RS"
ROBOT_NAME = "ReBot_Arm_RS"
IDENTITY = "formal_arm"
#: The two fingers of the shipped gripper are driven by one motor (BOM: 4x RS00 covers joints 4-6
#: plus the gripper). The URDF does not say so with a mimic tag.
GRIPPER_FINGERS_HARDWARE_COUPLED = True


class Provenance(str, Enum):
    """Where a number came from. Only CONFIRMED_REAL may support a real-robot claim."""

    CONFIRMED_REAL = "CONFIRMED_REAL"
    OFFICIAL_URDF = "OFFICIAL_URDF"
    UNKNOWN = "UNKNOWN"


class LimitNotReal(RuntimeError):
    """Raised when a caller demands a real value that the project does not actually have."""


@dataclass(frozen=True)
class MotorSpec:
    model: str
    peak_torque_nm: float
    reduction_ratio: int
    source: str
    rated_torque_nm: float | None = None
    mass_g: float | None = None
    provenance: Provenance = Provenance.CONFIRMED_REAL


@dataclass(frozen=True)
class Limit:
    """One limit together with how it was obtained.

    value is the number to USE when it exists. usable_for_capability is False when the value is real
    only as a model value; a capability claim must then find another source.
    """

    value: float | None
    provenance: Provenance
    source: str
    usable_for_capability: bool = True
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    motor: str
    angle: Limit
    velocity: Limit
    torque: Limit
    urdf_velocity: float
    urdf_effort: float


URDF_SRC = "reBot-DevArm/Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf"
BOM_SRC = "reBot-DevArm/hardware/reBot_B601_RS/README.md (RobStride RS00 x4 + RS06 x3)"
SPEC_SRC = "reBot-DevArm/README.md hardware table, reBot Arm B601-RS column"

#: Official Seeed product-page figures for the two motors the BOM names.
MOTORS: Mapping[str, MotorSpec] = {
    "RS00": MotorSpec(
        model="RobStride RS00",
        peak_torque_nm=14.0,
        reduction_ratio=10,
        mass_g=None,
        source="https://www.seeedstudio.com/Robostride-00-Actuator-p-6664.html (14 N.m, 10:1)",
    ),
    "RS06": MotorSpec(
        model="RobStride RS06",
        peak_torque_nm=36.0,
        reduction_ratio=9,
        mass_g=621.0,
        source="https://www.seeedstudio.com/Robostride-06-Actuator-p-6668.html (36 N.m, 9:1, 621 g)",
    ),
}

#: Which motor drives which joint. 7 motors: 3x RS06 (joint1-3) + 4x RS00 (joint4-6 + gripper).
MOTOR_ASSIGNMENT: Mapping[str, str] = {
    "joint1": "RS06",
    "joint2": "RS06",
    "joint3": "RS06",
    "joint4": "RS00",
    "joint5": "RS00",
    "joint6": "RS00",
    "gripper_joint1": "RS00",
    "gripper_joint2": "COUPLED",
}

#: URDF link masses (kg). The exporter CSV table lists roughly half of each of these for the CAD
#: parts alone; the difference is consistent with motor mass being added by the exporter. Model
#: total 6.0085 kg sits 0.69 kg under the 6.7 kg product weight (harness/electronics are not links).
LINK_MASSES_KG: Mapping[str, float] = {
    "base_link": 1.1774,
    "link1": 0.4655,
    "link2": 1.552,
    "link3": 1.252,
    "link4": 0.46,
    "link5": 0.2012,
    "link6": 0.1,
    "gripper_end": 0.65,
    "gripper_left": 0.0752,
    "gripper_right": 0.0752,
}

#: Product-sheet figures (B601-RS column). Series numbers, not this unit's measurements.
SPEC: Mapping[str, float] = {
    "dof": 6,
    "payload_kg": 2.5,
    "reach_m": 0.754,
    "weight_kg": 6.7,
    "repeatability_mm": 0.2,
    "supply_voltage_dc": 48.0,
}

ASSET: Mapping[str, str] = {
    "repo": "https://github.com/Seeed-Projects/reBot-DevArm",
    "commit": "f01a1dc189ecb74b5435a336a68b287c146c1b60",
    "urdf": URDF_SRC,
    "urdf_sha256": "3838ca8f9d586be22a89b33c7cbdd4abd43d4a92a2de377b294eef4c5bd7654d",
    "csv_sha256": "5b921c678f5c003869a9b410ad866fcc172f67435efc905fd32bd16e9434d959",
    "license": "Seeed reBot-DevArm repository LICENSE",
    "remote_dir": "/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs",
}

_URDF_ANGLE = {
    "joint1": (-2.8, 2.8),
    "joint2": (0.0, 3.14),
    "joint3": (0.0, 3.14),
    "joint4": (-1.57, 1.57),
    "joint5": (-1.57, 1.57),
    "joint6": (-3.14, 3.14),
}
_URDF_VELOCITY = {"joint1": 50.0, "joint2": 50.0, "joint3": 50.0,
                  "joint4": 40.0, "joint5": 40.0, "joint6": 40.0}
_URDF_EFFORT = {"joint1": 36.0, "joint2": 36.0, "joint3": 36.0,
                "joint4": 14.0, "joint5": 14.0, "joint6": 14.0}


def _arm_joint(name: str) -> JointSpec:
    motor = MOTOR_ASSIGNMENT[name]
    lower, upper = _URDF_ANGLE[name]
    return JointSpec(
        name=name,
        kind="revolute",
        motor=motor,
        angle=Limit(
            value=None,
            provenance=Provenance.OFFICIAL_URDF,
            source=URDF_SRC + " <limit lower/upper>",
            usable_for_capability=False,
            lower=lower,
            upper=upper,
        ),
        velocity=Limit(
            value=_URDF_VELOCITY[name],
            provenance=Provenance.OFFICIAL_URDF,
            source=URDF_SRC + " <limit velocity>; rpm-vs-rad/s not resolved",
            usable_for_capability=False,
        ),
        torque=Limit(
            value=MOTORS[motor].peak_torque_nm,
            provenance=Provenance.CONFIRMED_REAL,
            source=MOTORS[motor].source + "; URDF effort agrees",
        ),
        urdf_velocity=_URDF_VELOCITY[name],
        urdf_effort=_URDF_EFFORT[name],
    )


def _finger_joint(name: str, upper: float) -> JointSpec:
    src = URDF_SRC + " <limit lower/upper>"
    return JointSpec(
        name=name,
        kind="prismatic",
        motor=MOTOR_ASSIGNMENT[name],
        angle=Limit(value=None, provenance=Provenance.OFFICIAL_URDF, source=src,
                    usable_for_capability=False, lower=0.0, upper=upper),
        velocity=Limit(value=10.0, provenance=Provenance.OFFICIAL_URDF, source=src,
                       usable_for_capability=False),
        # 500 in the URDF effort column is a finger FORCE in newtons, not a motor torque.
        torque=Limit(value=500.0, provenance=Provenance.UNKNOWN,
                     source=src + "; effort is newtons of finger force, no motor torque stated",
                     usable_for_capability=False),
        urdf_velocity=10.0,
        urdf_effort=500.0,
    )


JOINT_LIMITS: Mapping[str, JointSpec] = {
    "joint1": _arm_joint("joint1"),
    "joint2": _arm_joint("joint2"),
    "joint3": _arm_joint("joint3"),
    "joint4": _arm_joint("joint4"),
    "joint5": _arm_joint("joint5"),
    "joint6": _arm_joint("joint6"),
    "gripper_joint1": _finger_joint("gripper_joint1", 0.05),
    "gripper_joint2": _finger_joint("gripper_joint2", 0.0715),
}

#: Joint order as it appears in the URDF / the converted USD. Never reorder silently.
JOINT_ORDER = tuple(JOINT_LIMITS)

ARM_JOINTS = tuple(n for n in JOINT_LIMITS if n.startswith("joint"))


def angle_range(name: str, *, require_real: bool = False) -> tuple[float, float]:
    spec = JOINT_LIMITS[name]
    if require_real and not spec.angle.usable_for_capability:
        raise LimitNotReal(
            name + " angle range is " + spec.angle.provenance.value + " (" + spec.angle.source + "); "
            "measure the unit or find a hardware document before claiming a real range"
        )
    if spec.angle.lower is None or spec.angle.upper is None:
        raise LimitNotReal(name + " angle range is unknown")
    return spec.angle.lower, spec.angle.upper


def joint_velocity(name: str, *, require_real: bool = False) -> float:
    spec = JOINT_LIMITS[name]
    if require_real and not spec.velocity.usable_for_capability:
        raise LimitNotReal(
            name + " velocity " + repr(spec.velocity.value) + " is " + spec.velocity.provenance.value
            + "; it may not be used as a real joint speed limit"
        )
    assert spec.velocity.value is not None
    return spec.velocity.value


def joint_torque(name: str, *, require_real: bool = False) -> float:
    spec = JOINT_LIMITS[name]
    if require_real and spec.torque.provenance is not Provenance.CONFIRMED_REAL:
        raise LimitNotReal(name + " torque is " + spec.torque.provenance.value + ": " + spec.torque.source)
    assert spec.torque.value is not None
    return spec.torque.value


def total_model_mass() -> float:
    return sum(LINK_MASSES_KG.values())


__all__ = [
    "MODEL", "ROBOT_NAME", "IDENTITY", "GRIPPER_FINGERS_HARDWARE_COUPLED",
    "Provenance", "LimitNotReal", "MotorSpec", "Limit", "JointSpec",
    "MOTORS", "MOTOR_ASSIGNMENT", "LINK_MASSES_KG", "SPEC", "ASSET",
    "JOINT_LIMITS", "JOINT_ORDER", "ARM_JOINTS",
    "angle_range", "joint_velocity", "joint_torque", "total_model_mass",
]
