# -*- coding: utf-8 -*-
"""Decision layer: feasibility gate + intercept search + shared travel model + adapter.

Coordinator merge (2026-09-13): every symbol the layer owns is exported here so that no
single implementer can break another one imports.
"""
from .feasibility import FeasibilityLimits, HitFeasibilityGate, HitReason, base_min_travel_time_s
from .travel_model import min_travel_time_s, min_travel_times_s
from .intercept_search import (InterceptReason, InterceptSearchConfig, InterceptSearchResult,
                               InterceptSearcher, search_intercepts)
from .decision_module import FeasibilityDecisionModule

__all__ = ["FeasibilityLimits", "HitFeasibilityGate", "HitReason", "base_min_travel_time_s",
           "min_travel_time_s", "min_travel_times_s",
           "InterceptReason", "InterceptSearchConfig", "InterceptSearchResult",
           "InterceptSearcher", "search_intercepts", "FeasibilityDecisionModule"]
