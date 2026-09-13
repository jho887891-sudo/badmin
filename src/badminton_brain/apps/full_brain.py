# -*- coding: utf-8 -*-
"""T11 integration: wire all eight layers into the BrainPipeline.

Design note (coordinator ruling): the modules were implemented in parallel by independent
agents, so this file discovers the single implementation of each layer interface inside the
layer module instead of hard-coding class names.  Missing or ambiguous modules raise a clear
BrainBoundaryError that names the offender.

Ruling: discovery over hard-coded names - why: parallel implementers may name classes freely -
cost if wrong: a module could expose two implementers and be rejected loudly (safe failure).
"""
from __future__ import annotations

import importlib
import inspect
from typing import Dict, List, Optional, Tuple

from ..interfaces import (
    AdaptationModule, DecisionModule, EstimationModule, ExecutionModule, PerceptionModule,
    PlanningModule, PredictionModule, SafetyModule,
)
from ..pipeline import BrainPipeline, PipelineConfig
from ..registry import ModuleRegistry
from ..types import BrainBoundaryError, Layer

# Candidate module paths per layer (first that yields exactly one implementer wins).
LAYER_MODULE_PATHS: Dict[Layer, Tuple[type, Tuple[str, ...]]] = {
    Layer.PERCEPTION: (PerceptionModule, ('badminton_brain.perception.stereo_geometry',
                                          'badminton_brain.perception.perception_module')),
    Layer.ESTIMATION: (EstimationModule, ('badminton_brain.estimation.estimator',
                                          'badminton_brain.estimation.estimation_module')),
    Layer.PREDICTION: (PredictionModule, ('badminton_brain.prediction.physics_predictor',
                                          'badminton_brain.prediction.prediction_module')),
    Layer.DECISION: (DecisionModule, ('badminton_brain.decision.decision_module',
                                      'badminton_brain.decision.feasibility')),
    Layer.PLANNING: (PlanningModule, ('badminton_brain.planning.planner',
                                      'badminton_brain.planning.planning_module',
                                      'badminton_brain.planning.expert_planner')),
    Layer.SAFETY: (SafetyModule, ('badminton_brain.safety.safety_shield',)),
    Layer.EXECUTION: (ExecutionModule, ('badminton_brain.execution.execution_module',
                                        'badminton_brain.execution.sim_adapter')),
    Layer.ADAPTATION: (AdaptationModule, ('badminton_brain.adaptation.adaptation_module',
                                         'badminton_brain.adaptation.online_adaptation')),
}


def _implementers(module, interface) -> List[type]:
    found = []
    for _, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if obj is interface or not issubclass(obj, interface):
            continue
        if not inspect.isabstract(obj) or getattr(obj, 'layer', None) is not None:
            found.append(obj)
    return found


def _instantiate(cls, num_envs: int, truth_provider=None):
    last_error: Optional[Exception] = None
    attempts = [{'num_envs': num_envs, 'truth_provider': truth_provider},
                {'num_envs': num_envs}, {}, {'n': num_envs}] if truth_provider is not None else [
                {'num_envs': num_envs}, {}, {'n': num_envs}]
    for kwargs in attempts:
        try:
            return cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise BrainBoundaryError(
        f"could not instantiate {cls.__name__} with (num_envs) or no arguments: {last_error}")


def build_full_brain(*, num_envs: int = 1, config: Optional[PipelineConfig] = None,
                     truth_provider=None):
    """Return (registry, pipeline) with all eight layers wired, or raise a clear error."""
    registry = ModuleRegistry()
    missing: List[str] = []
    problems: List[str] = []

    for layer, (interface, paths) in LAYER_MODULE_PATHS.items():
        chosen = None
        for path in paths:
            try:
                module = importlib.import_module(path)
            except ModuleNotFoundError:
                continue
            candidates = _implementers(module, interface)
            if len(candidates) == 1:
                chosen = _instantiate(candidates[0], num_envs, truth_provider)
                break
            if len(candidates) > 1:
                problems.append(
                    f"layer '{layer.value}': {path} exposes {len(candidates)} implementers "
                    f"({[c.__name__ for c in candidates]}); exactly one is required")
                break
        if chosen is None:
            missing.append(layer.value)
        else:
            registry.register(chosen)

    if problems:
        raise BrainBoundaryError('; '.join(problems))
    if missing:
        raise BrainBoundaryError(
            'full brain needs every layer implemented; missing: ' + ', '.join(missing))

    pipeline = BrainPipeline(registry, config or PipelineConfig())
    return registry, pipeline




class FullBrainRuntime:
    """Runtime loop: BrainPipeline plus the push channels the modules need (DEC-016).

    The frozen pipeline only calls process(), so the application layer pushes the per-step
    prediction into the adaptation module (slow-loop residual) and may push a safety context
    (clock / e-stop) before each step.
    """

    def __init__(self, registry, pipeline):
        self.registry = registry
        self.pipeline = pipeline

    def step(self, sensors):
        result = self.pipeline.step(sensors)
        adaptation = self.registry.get(Layer.ADAPTATION)
        if adaptation is not None and result.trajectory is not None:
            push = getattr(adaptation, 'set_prediction', None)
            if callable(push):
                push(result.trajectory)
        return result

    def push_safety_context(self, context) -> None:
        safety = self.registry.get(Layer.SAFETY)
        push = getattr(safety, 'set_context', None)
        if not callable(push):
            raise BrainBoundaryError('the registered safety module has no set_context API')
        push(context)

    def push_clock(self, now: float) -> None:
        decision = self.registry.get(Layer.DECISION)
        push = getattr(decision, 'set_now', None)
        if callable(push):
            push(now)

    def reset(self, env_ids) -> None:
        self.pipeline.reset(env_ids)


def build_full_runtime(*, num_envs: int = 1, config: Optional[PipelineConfig] = None,
                       truth_provider=None) -> FullBrainRuntime:
    """Same wiring as build_full_brain, wrapped in the runtime loop used by sim and real."""
    registry, pipeline = build_full_brain(num_envs=num_envs, config=config,
                                          truth_provider=truth_provider)
    return FullBrainRuntime(registry, pipeline)

__all__ = ["build_full_brain", "build_full_runtime", "FullBrainRuntime", "LAYER_MODULE_PATHS"]
