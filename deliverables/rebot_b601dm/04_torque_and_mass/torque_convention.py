"""The torque convention: peak or rated, and never silently either.

27 and 7 N m are the PEAK torques of the DM4340 and DM4310. The rated figures are 9 and 3 N m. A swing that
quietly assumed peak would report roughly three times the torque the motor can sustain, and the official
thermal testing (Performance_Testing_zh.md) terminated EVERY run on motor-2 overheating - so "sustained" is
not hypothetical for this arm.

TWO PLACES, HANDLED DIFFERENTLY ON PURPOSE:

  resolve_torque_limit   refuses to guess. convention=None is an error, not a default, because the caller is
                         the one who knows whether a brief swing or a sustained motion is being modelled.

  ExperimentTorque       carries a convention that defaults to RATED, and that convention is part of the
                         experiment record. An experiment therefore always has one, and a reader can always
                         see which produced a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .joint_limits import JOINT_LIMITS, Provenance


class TorqueConvention(str, Enum):
    """Which torque figure a run is allowed to use."""

    PEAK = "peak"
    RATED = "rated"


# Conservative by default. The official advice is payload under 1.5 kg and speed under 70% of maximum; an arm
# that trips thermal protection after 3 minutes of hovering at 1.5 kg is not a peak-torque machine.
DEFAULT_TORQUE_CONVENTION = TorqueConvention.RATED


def _normalise(convention) -> TorqueConvention:
    if convention is None:
        raise ValueError(
            "a torque convention is required; pass TorqueConvention.PEAK or TorqueConvention.RATED. It is "
            "not defaulted here because the caller knows whether the motion is brief or sustained."
        )
    if isinstance(convention, TorqueConvention):
        return convention
    try:
        return TorqueConvention(str(convention).strip().lower())
    except ValueError:
        raise ValueError(
            "unknown torque convention " + repr(convention) + "; expected peak or rated"
        ) from None


def resolve_torque_limit(joint_name: str, *, convention) -> float:
    """Return the torque limit in N m for one joint under the stated convention.

    Raises ValueError when the convention is missing or unknown, or when the joint has no hardware-sourced
    torque at all - the gripper, whose force would otherwise be invented.
    """
    conv = _normalise(convention)
    try:
        spec = JOINT_LIMITS[joint_name]
    except KeyError:
        raise KeyError("unknown joint: " + repr(joint_name)) from None
    limit = spec.torque
    if limit.provenance is not Provenance.CONFIRMED_REAL:
        raise ValueError(
            "joint " + joint_name + " has no hardware-sourced torque (" + limit.provenance.value
            + "); refusing to invent one"
        )
    if conv is TorqueConvention.PEAK:
        if limit.value is None:
            raise ValueError("joint " + joint_name + " has no peak torque recorded")
        return float(limit.value)
    if limit.rated is None:
        raise ValueError("joint " + joint_name + " has no rated torque recorded")
    return float(limit.rated)


@dataclass
class ExperimentTorque:
    """The torque convention an experiment runs under, and the record of that choice."""

    convention: object = DEFAULT_TORQUE_CONVENTION
    joints: tuple = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    _normalised: TorqueConvention = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._normalised = _normalise(self.convention)
        self.convention = self._normalised

    def limit_for(self, joint_name: str) -> float:
        return resolve_torque_limit(joint_name, convention=self._normalised)

    def limits(self) -> dict:
        return {j: self.limit_for(j) for j in self.joints}

    def unresolved(self) -> list:
        """Joints whose torque has no hardware source. Listed, never omitted, so the gap stays visible."""
        return [
            name for name, spec in JOINT_LIMITS.items()
            if spec.torque.provenance is not Provenance.CONFIRMED_REAL
        ]

    def as_record(self) -> dict:
        """The part of an experiment identity that makes a torque number interpretable."""
        peak = resolve_torque_limit("joint1", convention=TorqueConvention.PEAK)
        rated = resolve_torque_limit("joint1", convention=TorqueConvention.RATED)
        return {
            "torque_convention": self._normalised.value,
            "torque_nm": self.limits(),
            "unresolved": self.unresolved(),
            "note": (
                "the shipped asset and the URDF carry the PEAK torques, "
                + str(peak)
                + " and 7 N m for the two motor models; the rated figures are "
                + str(rated)
                + " and 3 N m. This experiment ran under the "
                + self._normalised.value
                + " convention."
            ),
        }
