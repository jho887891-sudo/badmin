# -*- coding: utf-8 -*-
"""Shared authenticity vocabulary (single source for both Robot Brain and simulation).

Spec: docs/simulation/BADMINTON_ROBOT.md S4 (asset authenticity levels) + S12 (TEMP policy).
Re-exported by simulation/robots/badminton_robot/badminton_robot_cfg.py so that the two
layers can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AssetStatus(str, Enum):
    """S4 authenticity levels."""
    VERIFIED_OFFICIAL = "VERIFIED_OFFICIAL"
    VERIFIED_MEASURED = "VERIFIED_MEASURED"
    DERIVED_FROM_MEASUREMENT = "DERIVED_FROM_MEASUREMENT"
    TRACEABLE_REFERENCE = "TRACEABLE_REFERENCE"
    TEMP_PARAMETERIZED_PROXY = "TEMP_PARAMETERIZED_PROXY"
    UNKNOWN = "UNKNOWN"
    REQUIRES_MEASUREMENT = "REQUIRES_MEASUREMENT"
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION"


UNRESOLVED_STATUSES = frozenset({
    AssetStatus.TEMP_PARAMETERIZED_PROXY,
    AssetStatus.UNKNOWN,
    AssetStatus.REQUIRES_MEASUREMENT,
    AssetStatus.REQUIRES_CALIBRATION,
})


@dataclass
class Param:
    """A value that can never hide how it was obtained (S4 / S12)."""
    value: Any = None
    status: AssetStatus = AssetStatus.REQUIRES_MEASUREMENT
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, AssetStatus):
            raise ValueError(f"status must be an AssetStatus, got {self.status!r}")
        if not str(self.source).strip():
            raise ValueError("Param.source is required: state where the value comes from")
        unresolved = {AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.UNKNOWN, AssetStatus.REQUIRES_CALIBRATION}
        if self.value is not None and self.status in unresolved:
            raise ValueError(
                f"value provided while status={self.status.value}; "
                "either mark it TEMP_PARAMETERIZED_PROXY or leave value=None"
            )

    @property
    def is_resolved(self) -> bool:
        return self.status not in UNRESOLVED_STATUSES


__all__ = ["AssetStatus", "Param", "UNRESOLVED_STATUSES"]
