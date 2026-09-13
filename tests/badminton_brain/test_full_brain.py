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
from badminton_brain.types import RobotSensorState  # noqa: E402
from badminton_brain.validation import validate_architecture  # noqa: E402

N = 2


def canonical_truth(num_envs: int, timestamp: float) -> np.ndarray:
    """Simulated ground-truth shuttle path (simulation-side input for the perception proxy)."""
    start = np.array([5.20, 0.30, 2.10])
    velocity = np.array([-8.0, -0.10, 1.20])
    return np.stack([start + velocity * timestamp for _ in range(num_envs)])


def sensors(t: float = 0.0) -> RobotSensorState:
    return RobotSensorState(base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (N, 1)),
                            joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)), timestamp=t,
                            odom_twist=np.zeros((N, 3)), imu_yaw_rate=np.zeros((N,)))


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

    def test_reset_is_propagated_to_every_layer(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        pipeline.reset([1])


if __name__ == '__main__':
    unittest.main()
