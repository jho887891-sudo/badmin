# -*- coding: utf-8 -*-
"""Layer interfaces of the modular architecture (ROBOT_BRAIN.md S11/S12/S15).

Rule enforced here: a module belongs to exactly one layer and may only return that
layer message type.  Nothing may silently absorb another layer duty
(Perception != Prediction != Decision != Planning != Safety != Execution).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from .types import (
    BestIntercept, BrainBoundaryError, Feedback, HitDecision, Layer, PredictedTrajectory,
    RobotSensorState, SafeCommand, ShuttleMeasurement, UnifiedState, WholeBodyTarget,
)


class BrainModule:
    """Base class of every brain module (sim/real shared, ROBOT_BRAIN.md S13)."""
    layer: Layer = None  # type: ignore[assignment]
    name: str = 'unnamed'
    is_implemented: bool = False

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset only the selected environments (stateful modules must override)."""
        return None


class PerceptionModule(BrainModule):
    """Sensors -> ShuttleMeasurement (measures; never predicts trajectories)."""
    layer = Layer.PERCEPTION
    output_type = ShuttleMeasurement

    def process(self, sensors: RobotSensorState) -> ShuttleMeasurement:
        raise NotImplementedError


class EstimationModule(BrainModule):
    """Measurements -> UnifiedState (Robot EKF + Shuttle UKF live here)."""
    layer = Layer.ESTIMATION
    output_type = UnifiedState

    def process(self, perception: ShuttleMeasurement, sensors: RobotSensorState) -> UnifiedState:
        raise NotImplementedError


class PredictionModule(BrainModule):
    """UnifiedState -> PredictedTrajectory (physics ODE; never decides to hit)."""
    layer = Layer.PREDICTION
    output_type = PredictedTrajectory

    def process(self, state: UnifiedState) -> PredictedTrajectory:
        raise NotImplementedError


class DecisionModule(BrainModule):
    """Feasibility gate + intercept search (never generates motion commands)."""
    layer = Layer.DECISION
    output_type = tuple

    def process(self, state: UnifiedState, trajectory: PredictedTrajectory):
        raise NotImplementedError


class PlanningModule(BrainModule):
    """Intercept -> WholeBodyTarget (planning / PPO live here; never bypasses Safety)."""
    layer = Layer.PLANNING
    output_type = WholeBodyTarget

    def process(self, state: UnifiedState, decision: HitDecision, intercept: Optional[BestIntercept],
                trajectory: PredictedTrajectory) -> WholeBodyTarget:
        raise NotImplementedError


class SafetyModule(BrainModule):
    """WholeBodyTarget -> SafeCommand (the only producer of executable commands)."""
    layer = Layer.SAFETY
    output_type = SafeCommand

    def process(self, target: WholeBodyTarget) -> SafeCommand:
        raise NotImplementedError


class ExecutionModule(BrainModule):
    """SafeCommand -> Feedback (Morph One / PiPER drivers live here)."""
    layer = Layer.EXECUTION
    output_type = Feedback

    def process(self, command: SafeCommand) -> Feedback:
        raise NotImplementedError


class AdaptationModule(BrainModule):
    """Feedback -> parameter/state correction (runs after the baseline chain)."""
    layer = Layer.ADAPTATION
    output_type = dict

    def process(self, feedback: Feedback, state: UnifiedState) -> Any:
        raise NotImplementedError


LAYER_INTERFACES = {
    Layer.PERCEPTION: PerceptionModule,
    Layer.ESTIMATION: EstimationModule,
    Layer.PREDICTION: PredictionModule,
    Layer.DECISION: DecisionModule,
    Layer.PLANNING: PlanningModule,
    Layer.SAFETY: SafetyModule,
    Layer.EXECUTION: ExecutionModule,
    Layer.ADAPTATION: AdaptationModule,
}


def interface_layer_of(module: BrainModule) -> Layer:
    """Return the layer declared by the module interface class."""
    for cls in type(module).__mro__:
        for layer, iface in LAYER_INTERFACES.items():
            if cls is iface:
                return layer
    raise BrainBoundaryError(
        f"{type(module).__name__} does not implement any brain layer interface")


__all__ = ["BrainModule", "PerceptionModule", "EstimationModule", "PredictionModule",
           "DecisionModule", "PlanningModule", "SafetyModule", "ExecutionModule",
           "AdaptationModule", "LAYER_INTERFACES", "interface_layer_of", "BrainBoundaryError"]
