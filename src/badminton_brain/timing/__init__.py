# -*- coding: utf-8 -*-
"""P7-08 timing: end-to-end latency budget and stale-frame rejection."""
from .latency_budget import (  # noqa: F401
    END_TO_END_BUDGET_S,
    MAX_SOURCE_FRAME_AGE_S,
    REJECT_AGE_EXCEEDED,
    REJECT_CLOCK_REGRESSION,
    REJECT_FRAME_REUSE,
    REJECT_FUTURE_FRAME,
    RESERVE_S,
    STAGE_BUDGET_S,
    STAGES,
    Admission,
    LatencyError,
    StalenessGuard,
    StageTimestamps,
    StaleFrameRejected,
    latency_report,
)

__all__ = [
    "Admission", "LatencyError", "StalenessGuard", "StageTimestamps",
    "StaleFrameRejected", "latency_report", "STAGES", "STAGE_BUDGET_S",
    "END_TO_END_BUDGET_S", "RESERVE_S", "MAX_SOURCE_FRAME_AGE_S",
    "REJECT_AGE_EXCEEDED", "REJECT_FRAME_REUSE", "REJECT_FUTURE_FRAME",
    "REJECT_CLOCK_REGRESSION",
]
