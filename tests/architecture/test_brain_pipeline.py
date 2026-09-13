# -*- coding: utf-8 -*-
"""Modular architecture wiring: registry, pipeline order, boundaries, validation.

Spec: docs/architecture/ROBOT_BRAIN.md S11 (boundaries), S12 (no silent role mixing),
      S15 (architecture baseline), plus S43 timestamps and S28 court-frame discipline.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.interfaces import (  # noqa: E402
    AdaptationModule, BrainBoundaryError, DecisionModule, EstimationModule,
    ExecutionModule, PerceptionModule, PlanningModule, PredictionModule, SafetyModule,
)
from badminton_brain.pipeline import BrainPipeline, PipelineConfig, StageTiming  # noqa: E402
from badminton_brain.registry import ModuleRegistry  # noqa: E402
from badminton_brain.types import (  # noqa: E402
    BASELINE_ORDER, BestIntercept, Feedback, HitDecision, Layer, PredictedTrajectory,
    RobotSensorState, SafeCommand, ShuttleMeasurement, UnifiedState, WholeBodyTarget,
)
from badminton_brain.validation import validate_architecture  # noqa: E402

N = 3
CALLS = []


def _sensors(n=N, t=0.0):
    return RobotSensorState(base_pose=np.zeros((n, 7)), joint_pos=np.zeros((n, 6)),
                            joint_vel=np.zeros((n, 6)), timestamp=t)


class FakePerception(PerceptionModule):
    name = 'fake_perception'
    def process(self, sensors):
        CALLS.append(self.layer)
        return ShuttleMeasurement(position=np.zeros((N, 3)), velocity=np.zeros((N, 3)),
                                  covariance=np.zeros((N, 3, 3)), timestamp=sensors.timestamp)


class FakeEstimation(EstimationModule):
    name = 'fake_estimation'
    def process(self, perception, sensors):
        CALLS.append(self.layer)
        return UnifiedState(base_pose=np.zeros((N, 7)), base_twist=np.zeros((N, 6)),
                            joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)),
                            racket_contact_pose=np.zeros((N, 7)), racket_contact_twist=np.zeros((N, 6)),
                            shuttle_position=np.zeros((N, 3)), shuttle_velocity=np.zeros((N, 3)),
                            timestamp=sensors.timestamp)


class FakePrediction(PredictionModule):
    name = 'fake_prediction'
    def process(self, state):
        CALLS.append(self.layer)
        T = 4
        return PredictedTrajectory(times=np.linspace(0, 0.3, T), position=np.zeros((N, T, 3)),
                                   velocity=np.zeros((N, T, 3)), landing_point=np.zeros((N, 3)),
                                   arrival_time=np.zeros((N,)), timestamp=state.timestamp)


class FakeDecision(DecisionModule):
    name = 'fake_decision'
    def process(self, state, trajectory):
        CALLS.append(self.layer)
        decision = HitDecision(feasible=True, reason='reachable', timestamp=state.timestamp)
        intercept = BestIntercept(position=np.zeros((N, 3)), time_s=np.zeros((N,)),
                                  racket_pose=np.zeros((N, 7)), score=np.zeros((N,)),
                                  timestamp=state.timestamp)
        return decision, intercept


class FakePlanning(PlanningModule):
    name = 'fake_planning'
    def process(self, state, decision, intercept, trajectory):
        CALLS.append(self.layer)
        return WholeBodyTarget(base_twist=np.zeros((N, 3)), joint_position_target=np.zeros((N, 6)),
                               horizon_s=0.3, timestamp=state.timestamp)


class FakeSafety(SafetyModule):
    name = 'fake_safety'
    def process(self, target):
        CALLS.append(self.layer)
        return SafeCommand(base_twist=target.base_twist, joint_position_target=target.joint_position_target,
                           limited=False, timestamp=target.timestamp)


class FakeExecution(ExecutionModule):
    name = 'fake_execution'
    def process(self, command):
        CALLS.append(self.layer)
        # DEC-015: execution reports its tracking residual; the prediction residual belongs to
        # the estimation/prediction side and must not be fabricated here.
        return Feedback(timestamp=command.timestamp, prediction_error=None,
                        tracking_residual=np.zeros((N,)),
                        contact_detected=np.zeros((N,), dtype=bool))


class FakeAdaptation(AdaptationModule):
    name = 'fake_adaptation'
    def process(self, feedback, state):
        CALLS.append(self.layer)
        return {'drag_delta': np.zeros((N,))}


class MismatchedPerception(PerceptionModule):
    """Perception that wrongly returns a prediction-layer message."""
    name = 'bad_perception'
    def process(self, sensors):
        return PredictedTrajectory(times=np.zeros(2), position=np.zeros((N, 2, 3)),
                                   velocity=np.zeros((N, 2, 3)), landing_point=np.zeros((N, 3)),
                                   arrival_time=np.zeros((N,)), timestamp=sensors.timestamp)


def build_registry(*, with_safety=True, with_adaptation=False, perception=None):
    reg = ModuleRegistry()
    reg.register(perception or FakePerception())
    reg.register(FakeEstimation())
    reg.register(FakePrediction())
    reg.register(FakeDecision())
    reg.register(FakePlanning())
    if with_safety:
        reg.register(FakeSafety())
    reg.register(FakeExecution())
    if with_adaptation:
        reg.register(FakeAdaptation())
    return reg


class RegistryTests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()

    def test_register_and_lookup_by_layer(self) -> None:
        reg = build_registry()
        self.assertIsInstance(reg.get(Layer.SAFETY), FakeSafety)
        self.assertIn(Layer.PERCEPTION, reg.layers())

    def test_duplicate_layer_is_rejected(self) -> None:
        reg = build_registry()
        with self.assertRaises(BrainBoundaryError):
            reg.register(FakeSafety())

    def test_module_layer_must_match_its_interface(self) -> None:
        class WrongLayer(PerceptionModule):
            layer = Layer.SAFETY
            name = 'wrong'
            def process(self, sensors):
                return None
        with self.assertRaises(BrainBoundaryError):
            ModuleRegistry().register(WrongLayer())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()

    def test_baseline_order_is_enforced(self) -> None:
        pipe = BrainPipeline(build_registry())
        pipe.step(_sensors())
        self.assertEqual(tuple(CALLS[:7]), BASELINE_ORDER)

    def test_step_returns_all_layer_outputs(self) -> None:
        pipe = BrainPipeline(build_registry())
        result = pipe.step(_sensors())
        self.assertIsInstance(result.shuttle_measurement, ShuttleMeasurement)
        self.assertIsInstance(result.unified_state, UnifiedState)
        self.assertIsInstance(result.trajectory, PredictedTrajectory)
        self.assertIsInstance(result.safe_command, SafeCommand)
        self.assertIsInstance(result.feedback, Feedback)

    def test_safety_cannot_be_bypassed(self) -> None:
        with self.assertRaises(BrainBoundaryError) as ctx:
            BrainPipeline(build_registry(with_safety=False))
        self.assertIn('safety', str(ctx.exception).lower())

    def test_execution_only_receives_safe_command(self) -> None:
        seen = {}

        class SpyExecution(FakeExecution):
            def process(self, command):
                seen['type'] = type(command).__name__
                return super().process(command)

        reg = build_registry()
        reg.replace(SpyExecution())
        BrainPipeline(reg).step(_sensors())
        self.assertEqual(seen['type'], 'SafeCommand')

    def test_layer_output_type_mismatch_is_a_boundary_error(self) -> None:
        pipe = BrainPipeline(build_registry(perception=MismatchedPerception()))
        with self.assertRaises(BrainBoundaryError):
            pipe.step(_sensors())

    def test_stage_timings_are_recorded(self) -> None:
        pipe = BrainPipeline(build_registry())
        pipe.step(_sensors())
        self.assertEqual(len(pipe.timings), 7)
        self.assertTrue(all(isinstance(t, StageTiming) for t in pipe.timings))

    def test_module_swap_keeps_pipeline_working(self) -> None:
        class FasterPrediction(FakePrediction):
            name = 'faster_prediction'
        reg = build_registry()
        reg.replace(FasterPrediction())
        pipe = BrainPipeline(reg)
        result = pipe.step(_sensors())
        self.assertEqual(result.timings[-1].layer, Layer.EXECUTION.value)

    def test_adaptation_runs_after_baseline_when_present(self) -> None:
        pipe = BrainPipeline(build_registry(with_adaptation=True))
        pipe.step(_sensors())
        self.assertEqual(CALLS[-1], Layer.ADAPTATION)

    def test_reset_reaches_every_module_with_env_ids_only(self) -> None:
        seen = []

        class SpySafety(FakeSafety):
            def reset(self, env_ids):
                seen.append((self.layer, list(env_ids)))
        reg = build_registry()
        reg.replace(SpySafety())
        BrainPipeline(reg).reset([1, 3, 6])
        self.assertEqual(seen, [(Layer.SAFETY, [1, 3, 6])])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()

    def test_development_mode_ok_with_warnings(self) -> None:
        report = validate_architecture(build_registry(), PipelineConfig(), mode='development')
        self.assertTrue(report.ok)
        self.assertTrue(report.warnings)

    def test_final_mode_rejects_unresolved_stage_rates(self) -> None:
        report = validate_architecture(build_registry(), PipelineConfig(), mode='final')
        self.assertFalse(report.ok)
        self.assertTrue(any('final' in e.lower() for e in report.errors))

    def test_final_mode_rejects_not_implemented_algorithms(self) -> None:
        class NotImplementedPerception(FakePerception):
            is_implemented = False
        report = validate_architecture(build_registry(perception=NotImplementedPerception()),
                                       PipelineConfig(), mode='final')
        self.assertFalse(report.ok)
        self.assertTrue(any('not implemented' in e.lower() or 'perception' in e.lower()
                            for e in report.errors))


if __name__ == '__main__':
    unittest.main()
