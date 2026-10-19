# -*- coding: utf-8 -*-
"""Pure-Python velocity/position accounting for joint trajectory probes.

The initial position is q(0); samples contain q(1)..q(N) and v(1)..v(N).
The integral must therefore include *all* N velocity samples, while the
measured displacement is q(N)-q(0). Splitting a step-out/step-back command
cannot silently change that time interval.
"""
from __future__ import annotations

import math
from typing import Sequence


def summarize_velocity_window(
    q0: float,
    positions: Sequence[float],
    velocities: Sequence[float],
    dt: float,
    *,
    switch_step: int | None = None,
) -> dict:
    """Compute net and per-phase displacement/velocity integrals on identical bounds.

    If switch_step is given, the commanded target changes at that physics
    step: outbound uses samples 1..switch_step-1 and return uses
    switch_step..N. Array index i (zero-based) corresponds to step i+1.
    """
    n = len(positions)
    if n < 2 or len(velocities) != n:
        raise ValueError("Need at least two positions and one velocity per position")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not all(math.isfinite(float(x)) for x in [q0, *positions, *velocities]):
        raise ValueError("Non-finite position or velocity sample")
    if switch_step is not None and not (2 <= switch_step < n):
        raise ValueError("switch_step must leave nonempty outbound and return segments")

    bounds = ([("outbound", 0, switch_step - 1),
               ("return", switch_step - 1, n)]
              if switch_step is not None else [("hold", 0, n)])
    segments = []
    for phase, start, stop in bounds:
        begin = q0 if start == 0 else float(positions[start - 1])
        finish = float(positions[stop - 1])
        delta = finish - begin
        integral = math.fsum(float(v) for v in velocities[start:stop]) * dt
        prior = [begin, *(float(q) for q in positions[start:stop - 1])]
        peak = max(abs(float(x) - y) for x, y
                   in zip(positions[start:stop], prior))
        segments.append({
            "phase": phase,
            "start_step": start,
            "end_step": stop,
            "actual_delta_rad": delta,
            "integrated_physx_velocity_rad": integral,
            "integral_minus_position_rad": integral - delta,
            "peak_abs_position_change_per_step_rad": peak,
        })

    measured = float(positions[-1]) - float(q0)
    integrated = math.fsum(float(v) for v in velocities) * dt
    if not math.isclose(sum(s["actual_delta_rad"] for s in segments),
                        measured, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("Segment position windows do not cover all samples")
    if not math.isclose(sum(s["integrated_physx_velocity_rad"] for s in segments),
                        integrated, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("Segment velocity windows do not cover all samples")
    return {
        "q_start": float(q0),
        "q_end": float(positions[-1]),
        "measured_delta": measured,
        "integrated_physx_velocity": integrated,
        "integral_minus_position": integrated - measured,
        "max_abs_position_step": max(
            s["peak_abs_position_change_per_step_rad"] for s in segments),
        "segments": segments,
        "accounting": {
            "window": "samples step=1..N vs initial_sample step=0",
            "integration_rule": "sum(physx_qd[step=1..N]) * dt",
            "step_count": n,
            "net_matches_segment_sum": True,
            "integral_matches_segment_sum": True,
        },
    }
