# -*- coding: utf-8 -*-
"""T11 integration (coordinator): all eight layers through the BrainPipeline.

Scenario: canonical incoming shuttle -> full chain -> SafeCommand -> Feedback -> Adaptation.
Spec: docs/superpowers/plans/2026-09-13-brain-modules.md (T11 row).
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.apps.full_brain import build_full_brain  # noqa: E402
from badminton_brain.interfaces import PlanningModule  # noqa: E402
from badminton_brain.types import WholeBodyTarget  # noqa: E402
from badminton_brain.types import Layer, RobotSensorState  # noqa: E402
from badminton_brain.validation import validate_architecture  # noqa: E402

N = 2


# Physics-consistent ground truth: the same frozen aerodynamics the predictor integrates.
# A straight-line truth would leave a constant model mismatch that the slow loop would then
# push into its parameters until they saturate (T10 review, 2026-09-13).
_TRUTH_START = np.array([5.20, 0.30, 2.10])
_TRUTH_VELOCITY = np.array([-8.0, -0.10, 1.20])
_TRUTH_DT = 1e-3
_TRUTH_TABLE = None


def _truth_table():
    global _TRUTH_TABLE
    if _TRUTH_TABLE is None:
        from trajectory.shuttle_aerodynamics import k_from_aerodynamic_length, rollout
        result = rollout(_TRUTH_START, _TRUTH_VELOCITY, duration_s=1.5, dt_s=_TRUTH_DT,
                         k_per_m=k_from_aerodynamic_length(6.5))
        times = np.asarray(result['time'], dtype=float)
        positions = np.asarray(result['position'], dtype=float)
        _TRUTH_TABLE = (times, positions)
    return _TRUTH_TABLE


def canonical_truth(num_envs: int, timestamp: float) -> np.ndarray:
    """Simulated ground-truth shuttle position at an absolute time, from the frozen physics."""
    times, positions = _truth_table()
    index = int(np.clip(np.searchsorted(times, float(timestamp)), 0, times.shape[0] - 1))
    point = np.asarray(positions[index], dtype=float).reshape(3)
    return np.stack([point for _ in range(num_envs)])


def sensors(t: float = 0.0) -> RobotSensorState:
    return RobotSensorState(base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (N, 1)),
                            joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)), timestamp=t,
                            odom_twist=np.zeros((N, 3)), imu_yaw_rate=np.zeros((N,)))




class AlwaysMovePlanner(PlanningModule):
    """Stub planner that always commands motion (the canonical scenario may command none),
    so an e-stop comparison has a non-zero baseline to differ from."""

    name = 'always_move'
    is_implemented = True

    def process(self, state, decision, intercept, trajectory):
        return WholeBodyTarget(base_twist=np.full((N, 3), 0.4),
                               joint_position_target=np.zeros((N, 6)),
                               horizon_s=0.2, timestamp=state.timestamp)

class FullBrainIntegrationTests(unittest.TestCase):
    def test_all_eight_layers_are_registered(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        self.assertEqual(len(registry.modules()), 8)

    def test_canonical_step_produces_safe_command_and_feedback(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        result = pipeline.step(sensors(0.0))
        self.assertIsNotNone(result.safe_command)
        self.assertIsNotNone(result.feedback)
        self.assertIsNotNone(result.unified_state)
        self.assertIsNotNone(result.trajectory)
        self.assertEqual(np.asarray(result.safe_command.base_twist).shape, (N, 3))
        self.assertEqual(np.asarray(result.safe_command.joint_position_target).shape, (N, 6))

    def test_chain_runs_several_steps_without_diverging(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        for step in range(5):
            result = pipeline.step(sensors(0.05 * step))
            self.assertTrue(np.all(np.isfinite(np.asarray(result.safe_command.base_twist))))
            self.assertEqual(len(pipeline.timings), 8)

    def test_final_mode_refuses_unresolved_stage_rates(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        report = validate_architecture(registry, pipeline.config, mode='final')
        self.assertFalse(report.ok)
        self.assertTrue(any('stage rate' in e.lower() for e in report.errors), report.errors)

    def test_final_mode_refuses_the_ppo_placeholder(self) -> None:
        from badminton_brain.planning.ppo_policy_stub import PpoPolicyStub
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        registry.replace(PpoPolicyStub())
        report = validate_architecture(registry, pipeline.config, mode='final')
        self.assertFalse(report.ok)
        self.assertTrue(any('not implemented' in e.lower() for e in report.errors), report.errors)

    def test_development_mode_ok_with_warnings(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        report = validate_architecture(registry, pipeline.config, mode='development')
        self.assertTrue(report.ok, report.errors)
        self.assertTrue(report.warnings)


    def test_estop_context_reaches_the_command_end_to_end(self) -> None:
        from badminton_brain.safety.safety_shield import SafetyContext
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        registry.replace(AlwaysMovePlanner())   # guarantee a non-zero baseline command
        reference = pipeline.step(sensors(0.0))
        reference_twist = np.asarray(reference.safe_command.base_twist)
        assert np.max(np.abs(reference_twist)) > 0.0, 'the baseline must actually command motion'

        registry_b, pipeline_b = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        registry_b.replace(AlwaysMovePlanner())
        safety = registry_b.get(Layer.SAFETY)
        safety.set_context(SafetyContext(now=0.0, estop=np.array([True, False])))
        result = pipeline_b.step(sensors(0.0))
        twist = np.asarray(result.safe_command.base_twist)
        np.testing.assert_allclose(twist[0], np.zeros(3), atol=1e-12, err_msg='e-stopped env must not move')
        np.testing.assert_allclose(twist[1], reference_twist[1], atol=1e-12,
                                   err_msg='the non-e-stopped env must keep its command')
        self.assertTrue(result.safe_command.limited, 'an e-stop must be reported in the command')

    def test_context_free_run_is_flagged_and_not_silently_normal(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        safety = registry.get(Layer.SAFETY)
        self.assertTrue(hasattr(safety, 'context_snapshot'),
                        'the safety module must expose its context API')
        snapshot = safety.context_snapshot()
        self.assertIn(str(snapshot.get('source', '')), ('none', 'pushed', 'argument'))
        result = pipeline.step(sensors(0.0))
        violations = ' '.join(getattr(result.safe_command, 'violations', ()) or ())
        if snapshot.get('source') == 'none':
            self.assertIn('no_context', violations,
                          'running without any clock/e-stop information must be visible')

    def test_runtime_pushes_the_prediction_into_the_adaptation_layer(self) -> None:
        from badminton_brain.apps.full_brain import build_full_runtime
        runtime = build_full_runtime(num_envs=N, truth_provider=canonical_truth)
        for step in range(3):
            runtime.step(sensors(0.05 * step))
        adaptation = runtime.registry.get(Layer.ADAPTATION)
        diagnostics = adaptation.diagnostics()
        self.assertTrue(all(np.asarray(diagnostics['prediction_available'])),
                        'the runtime must feed the prediction to the slow loop (DEC-016)')

    def test_runtime_exposes_the_safety_context_push(self) -> None:
        from badminton_brain.apps.full_brain import build_full_runtime
        from badminton_brain.safety.safety_shield import SafetyContext
        runtime = build_full_runtime(num_envs=N, truth_provider=canonical_truth)
        runtime.step(sensors(0.0))
        runtime.push_safety_context(SafetyContext(now=0.05, estop=np.array([False, False])))
        runtime.push_clock(0.05)
        result = runtime.step(sensors(0.05))
        self.assertIsNotNone(result.safe_command)
    def test_reset_is_propagated_to_every_layer(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        seen = []
        for module in registry.modules():
            original = module.reset

            def spy(env_ids, _module=module, _original=original):
                seen.append((_module.layer.value, list(env_ids)))
                return _original(env_ids)

            module.reset = spy
        pipeline.reset([1])
        self.assertEqual(len(seen), len(registry.modules()),
                         'reset must reach every registered layer')
        self.assertEqual(sorted(layer for layer, _ in seen),
                         sorted(m.layer.value for m in registry.modules()))
        self.assertTrue(all(ids == [1] for _, ids in seen),
                        'every layer must receive exactly the requested env ids')


if __name__ == '__main__':
    unittest.main()
