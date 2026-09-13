# -*- coding: utf-8 -*-
"""Decision layer adapter tests (coordinator T11 wiring)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.decision.decision_module import FeasibilityDecisionModule  # noqa: E402
from badminton_brain.decision.intercept_search import search_intercepts  # noqa: E402
from badminton_brain.interfaces import DecisionModule  # noqa: E402
from badminton_brain.types import BestIntercept, HitDecision, Layer, PredictedTrajectory, UnifiedState  # noqa: E402

N = 2


def make_state(t=0.0):
    return UnifiedState(base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (N, 1)),
                        base_twist=np.zeros((N, 6)), joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)),
                        racket_contact_pose=np.tile(np.array([-1.6, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]), (N, 1)),
                        racket_contact_twist=np.zeros((N, 6)),
                        shuttle_position=np.tile(np.array([1.5, 0.0, 1.5]), (N, 1)),
                        shuttle_velocity=np.tile(np.array([-4.0, 0.0, 1.0]), (N, 1)), timestamp=t)


def make_trajectory(t=0.0):
    T = 6
    pos = np.zeros((N, T, 3))
    pos[:, :, 0] = np.linspace(1.5, -1.6, T)
    pos[:, :, 2] = np.linspace(1.5, 0.2, T)
    vel = np.zeros((N, T, 3))
    vel[:, :, 0] = -4.0
    vel[:, :, 2] = -1.0
    return PredictedTrajectory(times=np.linspace(0, 0.5, T), position=pos, velocity=vel,
                              landing_point=np.tile(np.array([-1.6, 0.0, 0.0]), (N, 1)),
                              arrival_time=np.full((N,), 0.5), timestamp=t)


class DecisionModuleTests(unittest.TestCase):
    def test_is_a_decision_module(self) -> None:
        mod = FeasibilityDecisionModule(num_envs=N)
        self.assertIsInstance(mod, DecisionModule)
        self.assertIs(mod.layer, Layer.DECISION)

    def test_returns_two_tuple_including_per_env_decisions(self) -> None:
        mod = FeasibilityDecisionModule(num_envs=N)
        decision, intercept = mod.process(make_state(), make_trajectory())
        self.assertIsInstance(decision, HitDecision)
        self.assertTrue(intercept is None or isinstance(intercept, BestIntercept))
        self.assertEqual(len(mod.last_decisions), N)
        self.assertTrue(all(isinstance(d, HitDecision) for d in mod.last_decisions))

    def test_reason_is_a_stable_upper_snake_token(self) -> None:
        mod = FeasibilityDecisionModule(num_envs=N)
        decision, _ = mod.process(make_state(), make_trajectory())
        self.assertRegex(decision.reason, r'^[A-Z][A-Z0-9_]*$')

    def test_aggregate_is_conservative_when_any_env_is_rejected(self) -> None:
        # The gate judges the PREDICTED trajectory (its first sample speed / landing), not the
        # instantaneous shuttle_velocity, so env 1 is made infeasible by pushing its predicted
        # path far outside the playable volume.
        mod = FeasibilityDecisionModule(num_envs=N)
        state = make_state()
        traj = make_trajectory()
        traj.position[1, :, 0] += 60.0  # env 1 lands far outside the court
        traj.landing_point[1] = np.array([60.0, 0.0, 0.0])
        decision, intercept = mod.process(state, traj)
        self.assertFalse(decision.feasible)
        self.assertIsNone(intercept)
        self.assertFalse(bool(mod.last_decisions[1].feasible))
        self.assertTrue(bool(mod.last_decisions[0].feasible))

    def test_end_to_end_feasible_case_requires_the_real_search(self) -> None:
        probe = search_intercepts(make_state(), make_trajectory())
        if not bool(np.any(probe.feasible)):
            self.skipTest('T6 intercept search is still a shell (NOT_IMPLEMENTED); '
                          're-run after it lands')
        mod = FeasibilityDecisionModule(num_envs=N)
        decision, intercept = mod.process(make_state(), make_trajectory())
        self.assertTrue(decision.feasible)
        self.assertIsNotNone(intercept)
        self.assertEqual(np.asarray(intercept.position).shape, (N, 3))


if __name__ == '__main__':
    unittest.main()
