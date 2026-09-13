# -*- coding: utf-8 -*-
"""Brain pipeline: wires the layers in the architecture baseline order (S15).

Guarantees enforced here:
  * every baseline layer is present - Safety can never be bypassed;
  * Execution only ever consumes SafeCommand;
  * a stage may only emit its own layer message type (no silent role mixing);
  * per-stage timings are recorded for later frequency/real-time analysis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .registry import ModuleRegistry
from .status import AssetStatus, Param
from .types import (
    BASELINE_ORDER, BrainBoundaryError, Feedback, HitDecision, Layer, PredictedTrajectory,
    RobotSensorState, SafeCommand, ShuttleMeasurement, UnifiedState, WholeBodyTarget,
)


@dataclass
class StageTiming:
    layer: str = ''
    seconds: float = 0.0


@dataclass
class PipelineConfig:
    """Stage rates are configuration, never hardcoded constants (ROBOT_BRAIN.md)."""
    stage_rate_hz: Dict[str, Param] = field(default_factory=lambda: {
        layer.value: Param(None, AssetStatus.REQUIRES_MEASUREMENT,
                           "stage rate set by interface/real-time tests, not hardcoded")
        for layer in BASELINE_ORDER
    })


@dataclass
class BrainStepResult:
    sensor_state: Optional[RobotSensorState] = None
    shuttle_measurement: Optional[ShuttleMeasurement] = None
    unified_state: Optional[UnifiedState] = None
    trajectory: Optional[PredictedTrajectory] = None
    decision: Optional[HitDecision] = None
    intercept: Any = None
    target: Optional[WholeBodyTarget] = None
    safe_command: Optional[SafeCommand] = None
    feedback: Optional[Feedback] = None
    adaptation: Any = None
    timings: List[StageTiming] = field(default_factory=list)


EXPECTED_OUTPUT = {
    Layer.PERCEPTION: ShuttleMeasurement,
    Layer.ESTIMATION: UnifiedState,
    Layer.PREDICTION: PredictedTrajectory,
    Layer.DECISION: tuple,
    Layer.PLANNING: WholeBodyTarget,
    Layer.SAFETY: SafeCommand,
    Layer.EXECUTION: Feedback,
}


class BrainPipeline:
    """Runs Perception -> ... -> Execution every step, with Adaptation afterwards."""

    def __init__(self, registry: ModuleRegistry, config: Optional[PipelineConfig] = None) -> None:
        self.registry = registry
        self.config = config or PipelineConfig()
        self.timings: List[StageTiming] = []
        missing = registry.missing()
        if missing:
            names = [l.value for l in missing]
            extra = " (safety must not be bypassed)" if Layer.SAFETY in missing else ""
            raise BrainBoundaryError(
                "architecture baseline requires every layer; missing: " + ", ".join(names) + extra)

    def _run(self, layer: Layer, fn) -> Any:
        start = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - start
        self.timings.append(StageTiming(layer.value, elapsed))
        expected = EXPECTED_OUTPUT.get(layer)
        if expected is not None and not isinstance(out, expected):
            raise BrainBoundaryError(
                f"layer '{layer.value}' returned {type(out).__name__}, expected {expected.__name__}; "
                "a module must not take over another layer duty")
        return out

    def step(self, sensors: RobotSensorState) -> BrainStepResult:
        if not isinstance(sensors, RobotSensorState):
            raise BrainBoundaryError('BrainPipeline.step expects a RobotSensorState')
        self.timings = []
        result = BrainStepResult(sensor_state=sensors)

        perception = self.registry.get(Layer.PERCEPTION)
        result.shuttle_measurement = self._run(Layer.PERCEPTION, lambda: perception.process(sensors))

        estimation = self.registry.get(Layer.ESTIMATION)
        result.unified_state = self._run(
            Layer.ESTIMATION, lambda: estimation.process(result.shuttle_measurement, sensors))

        prediction = self.registry.get(Layer.PREDICTION)
        result.trajectory = self._run(Layer.PREDICTION, lambda: prediction.process(result.unified_state))

        decision = self.registry.get(Layer.DECISION)
        pair = self._run(Layer.DECISION,
                         lambda: decision.process(result.unified_state, result.trajectory))
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise BrainBoundaryError('decision layer must return (HitDecision, BestIntercept|None)')
        result.decision, result.intercept = pair

        planning = self.registry.get(Layer.PLANNING)
        result.target = self._run(Layer.PLANNING, lambda: planning.process(
            result.unified_state, result.decision, result.intercept, result.trajectory))

        safety = self.registry.get(Layer.SAFETY)
        result.safe_command = self._run(Layer.SAFETY, lambda: safety.process(result.target))

        execution = self.registry.get(Layer.EXECUTION)
        result.feedback = self._run(Layer.EXECUTION, lambda: execution.process(result.safe_command))

        adaptation = self.registry.get(Layer.ADAPTATION)
        if adaptation is not None:
            result.adaptation = self._run(
                Layer.ADAPTATION, lambda: adaptation.process(result.feedback, result.unified_state))

        result.timings = list(self.timings)
        return result

    def reset(self, env_ids) -> None:
        """Reset only the selected environments across every registered module."""
        for module in self.registry.modules():
            module.reset(env_ids)


__all__ = ["BrainPipeline", "BrainStepResult", "PipelineConfig", "StageTiming", "EXPECTED_OUTPUT"]
