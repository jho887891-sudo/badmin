# -*- coding: utf-8 -*-
"""Decision layer adapter: feasibility gate (T5) + intercept search (T6) -> DecisionModule.

Coordinator ruling (2026-09-13): the frozen interface returns ONE 2-tuple per step, so the
decision is presented at batch level - aggregate ``feasible`` (all environments) plus the
batched ``BestIntercept`` that T6 already produces - while the per-environment ``HitDecision``
objects stay available through :attr:`last_decisions`.  Conservative by construction: if any
environment must not play, nothing is planned.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import DecisionModule
from ..types import BrainBoundaryError, HitDecision, PredictedTrajectory, UnifiedState
from .feasibility import HitFeasibilityGate, FeasibilityLimits
from .intercept_search import InterceptSearchConfig, InterceptSearchResult, InterceptSearcher


class FeasibilityDecisionModule(DecisionModule):
    """Runs the gate, then the intercept search, and reports one batch-level decision."""

    name = 'feasibility_decision'
    is_implemented = True

    def __init__(self, *, limits: Optional[FeasibilityLimits] = None,
                 search_config: Optional[InterceptSearchConfig] = None, num_envs: int = 1,
                 now_provider: Optional[Callable[[], float]] = None) -> None:
        self.num_envs = int(num_envs)
        self.now_provider = now_provider
        self._pushed_now: Optional[float] = None
        self.last_now: Optional[float] = None
        self.gate = HitFeasibilityGate(limits)
        self.searcher = InterceptSearcher(search_config)
        self.last_decisions: Tuple[HitDecision, ...] = ()
        self.last_search: Optional[InterceptSearchResult] = None

    def process(self, state: UnifiedState, trajectory: PredictedTrajectory):
        if not isinstance(state, UnifiedState) or not isinstance(trajectory, PredictedTrajectory):
            raise BrainBoundaryError(
                'FeasibilityDecisionModule.process expects (UnifiedState, PredictedTrajectory)')

        now = self._resolve_now(state, trajectory)
        self.last_now = now
        decisions = tuple(self.gate.evaluate_batch(state, trajectory, now=now))
        self.last_decisions = decisions
        search = self.searcher.search(state, trajectory)
        self.last_search = search

        gate_ok = all(bool(d.feasible) for d in decisions)
        search_feasible = np.asarray(search.feasible, dtype=bool)
        search_ok = bool(search_feasible.size > 0 and np.all(search_feasible))
        feasible = bool(gate_ok and search_ok)

        reason = self._reason(decisions, search, feasible)
        timestamp = float(state.timestamp)
        decision = HitDecision(feasible=feasible, reason=reason, timestamp=timestamp)
        intercept = search.best if (feasible and search.best is not None) else None
        return decision, intercept

    def set_now(self, now: float) -> None:
        """Push the runtime clock (T5 review C2): without it the gate cannot see a stale state."""
        self._pushed_now = float(now)

    def _resolve_now(self, state: UnifiedState, trajectory: PredictedTrajectory) -> float:
        """Runtime clock if available, else the newest message timestamp (documented fallback)."""
        if self.now_provider is not None:
            return float(self.now_provider())
        if self._pushed_now is not None:
            return float(self._pushed_now)
        return max(float(state.timestamp), float(trajectory.timestamp))

    @staticmethod
    def _reason(decisions: Sequence[HitDecision], search: InterceptSearchResult,
                feasible: bool) -> str:
        if feasible:
            return 'FEASIBLE'
        for decision in decisions:
            if not bool(decision.feasible):
                return str(decision.reason)
        reasons = getattr(search, 'reason', ())
        for reason in reasons:
            name = getattr(reason, 'name', None) or str(reason)
            if name not in ('FEASIBLE',):
                return str(name)
        return 'NOT_FEASIBLE'

    def reset(self, env_ids: Sequence[int]) -> None:
        self.last_decisions = ()
        self.last_search = None
        for target in (self.gate, self.searcher):
            reset = getattr(target, 'reset', None)
            if callable(reset):
                reset(env_ids)


__all__ = ["FeasibilityDecisionModule"]
