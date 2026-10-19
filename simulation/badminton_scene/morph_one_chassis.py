# -*- coding: utf-8 -*-
"""Morph One four-steer/four-drive joint topology and parameter discipline.

WHAT THIS MODULE IS, AND IS NOT
    This is the *interface* layer: the eight semantic joints, their fixed order, the mapping to the
    project's wheel-module vocabulary, and an explicit status for every physical parameter.
    It is NOT a model of the real Morph One geometry. Phase A could not classify the official CAD
    (see management/morph_one_rs/PHASE_A_CAD_ASSETIZATION.md), so no proxy geometry is fabricated
    here: claiming a movable chassis without the real geometry is exactly what the task forbids.

    Every physical number below is REQUIRES_MEASUREMENT or TEMP_PARAMETERIZED_PROXY. None is
    VERIFIED_MEASURED, and nothing here may be promoted without a measurement record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

#: Wheel module vocabulary, in the order the project already uses (FL, FR, RL, RR).
WHEEL_MODULES: Tuple[str, ...] = ("fl", "fr", "rl", "rr")

#: One steering joint and one drive joint per wheel module.
JOINT_KINDS: Tuple[str, ...] = ("steer", "drive")

#: The eight semantic joints, in canonical order: fl_steer, fl_drive, fr_steer, fr_drive, ...
SEMANTIC_JOINTS: Tuple[str, ...] = tuple(
    module + "_" + kind for module in WHEEL_MODULES for kind in JOINT_KINDS
)


class ChassisParameterStatus:
    """Only these three labels may be used; anything else is a defect."""

    REQUIRES_MEASUREMENT = "REQUIRES_MEASUREMENT"
    TEMP_PARAMETERIZED_PROXY = "TEMP_PARAMETERIZED_PROXY"
    VERIFIED_MEASURED = "VERIFIED_MEASURED"


@dataclass(frozen=True)
class ChassisParameter:
    name: str
    value: float | None
    unit: str
    status: str
    source: str

    def __post_init__(self) -> None:
        allowed = {
            ChassisParameterStatus.REQUIRES_MEASUREMENT,
            ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY,
            ChassisParameterStatus.VERIFIED_MEASURED,
        }
        if self.status not in allowed:
            raise ValueError("unknown status for " + self.name + ": " + str(self.status))
        if self.status == ChassisParameterStatus.REQUIRES_MEASUREMENT and self.value is not None:
            raise ValueError(self.name + " carries a value but is REQUIRES_MEASUREMENT")
        if self.status == ChassisParameterStatus.VERIFIED_MEASURED and not self.source:
            raise ValueError(self.name + " claims VERIFIED_MEASURED without a source")
        if not str(self.source).strip():
            raise ValueError(self.name + " needs a source string")


#: Nothing in this table is measured on hardware. The CAD file that would provide the geometry could
#: not be classified (no CAD kernel on the host), so the honest status for every geometric quantity
#: is REQUIRES_MEASUREMENT. The two TEMP entries exist only so that software that needs *some* number
#: can run in development, and they are marked as such.
CHASSIS_PARAMETERS: Dict[str, ChassisParameter] = {
    "wheel_radius_m": ChassisParameter(
        "wheel_radius_m", None, "m", ChassisParameterStatus.REQUIRES_MEASUREMENT,
        "no wheel geometry identified from M1-4WD4S_model2601.step; requires caliper measurement"),
    "wheel_width_m": ChassisParameter(
        "wheel_width_m", None, "m", ChassisParameterStatus.REQUIRES_MEASUREMENT,
        "no wheel geometry identified from the official CAD; requires measurement"),
    "wheel_base_m": ChassisParameter(
        "wheel_base_m", None, "m", ChassisParameterStatus.REQUIRES_MEASUREMENT,
        "front/rear axle spacing not derivable without part classification; requires measurement"),
    "track_width_m": ChassisParameter(
        "track_width_m", None, "m", ChassisParameterStatus.REQUIRES_MEASUREMENT,
        "left/right wheel spacing not derivable without part classification; requires measurement"),
    "base_length_m": ChassisParameter(
        "base_length_m", 0.70, "m", ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY,
        "project engineering baseline (docs/simulation/BADMINTON_ROBOT.md), not a Morph One value"),
    "base_width_m": ChassisParameter(
        "base_width_m", 0.55, "m", ChassisParameterStatus.TEMP_PARAMETERIZED_PROXY,
        "project engineering baseline (docs/simulation/BADMINTON_ROBOT.md), not a Morph One value"),
    "total_mass_kg": ChassisParameter(
        "total_mass_kg", None, "kg", ChassisParameterStatus.REQUIRES_MEASUREMENT,
        "no mass in the CAD text; chassis mass and inertia must be measured"),
    "steer_axis_height_m": ChassisParameter(
        "steer_axis_height_m", None, "m", ChassisParameterStatus.REQUIRES_MEASUREMENT,
        "kingpin geometry not identified; requires measurement or the real assembly"),
}

#: Frames this project must be able to name once the geometry exists. Declared now so that code can
#: fail loudly instead of inventing a transform.
REQUIRED_FRAMES: Tuple[str, ...] = (
    "chassis_base",
    "fl_steer_axis", "fr_steer_axis", "rl_steer_axis", "rr_steer_axis",
    "fl_wheel_center", "fr_wheel_center", "rl_wheel_center", "rr_wheel_center",
    "arm_mount",
)


def joint_index_map() -> Dict[str, int]:
    """Semantic joint name -> index in the canonical order used by every test and config."""
    return {name: i for i, name in enumerate(SEMANTIC_JOINTS)}


def validate_topology() -> None:
    """Fail loudly if the topology is not exactly four modules x (steer, drive)."""
    if len(WHEEL_MODULES) != 4:
        raise AssertionError("Morph One has four wheel modules")
    if len(set(WHEEL_MODULES)) != len(WHEEL_MODULES):
        raise AssertionError("duplicate wheel module id")
    if len(SEMANTIC_JOINTS) != 8:
        raise AssertionError("expected 8 semantic joints, got %d" % len(SEMANTIC_JOINTS))
    if len(set(SEMANTIC_JOINTS)) != len(SEMANTIC_JOINTS):
        raise AssertionError("duplicate semantic joint name")
    for name in SEMANTIC_JOINTS:
        module, _, kind = name.rpartition("_")
        if module not in WHEEL_MODULES or kind not in JOINT_KINDS:
            raise AssertionError("malformed joint name: " + name)


def unresolved_parameters() -> Tuple[str, ...]:
    return tuple(name for name, p in CHASSIS_PARAMETERS.items()
                 if p.status != ChassisParameterStatus.VERIFIED_MEASURED)


validate_topology()

__all__ = [
    "WHEEL_MODULES", "JOINT_KINDS", "SEMANTIC_JOINTS", "ChassisParameterStatus",
    "ChassisParameter", "CHASSIS_PARAMETERS", "REQUIRED_FRAMES",
    "joint_index_map", "validate_topology", "unresolved_parameters",
]
