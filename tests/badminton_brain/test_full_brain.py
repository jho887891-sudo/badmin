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

N = 2


class FullBrainIntegrationTests(unittest.TestCase):
    def test_all_eight_layers_are_registered(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N)
        self.assertEqual(len(registry.modules()), 8)

    def test_canonical_step_produces_safe_command_and_feedback(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N)
        sensors = RobotSensorState(base_pose=np.tile(np.array([-1.6, 0, 0, 0, 0, 0, 1.0]), (N, 1)),
                                   joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)), timestamp=0.0)
        result = pipeline.step(sensors)
        self.assertIsNotNone(result.safe_command)
        self.assertIsNotNone(result.feedback)
        self.assertEqual(result.safe_command.base_twist.shape, (N, 3))

    def test_final_mode_refuses_the_declared_not_implemented_layers(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N)
        from badminton_brain.validation import validate_architecture
        report = validate_architecture(registry, pipeline.config, mode='final')
        self.assertFalse(report.ok)
        self.assertTrue(any('not implemented' in e.lower() for e in report.errors))

    def test_development_mode_still_reports_warnings(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N)
        from badminton_brain.validation import validate_architecture
        report = validate_architecture(registry, pipeline.config, mode='development')
        self.assertTrue(report.ok)
        self.assertTrue(report.warnings)


if __name__ == '__main__':
    unittest.main()
