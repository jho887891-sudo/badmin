# -*- coding: utf-8 -*-
"""T4 physics trajectory prediction (docs/superpowers/plans/2026-09-13-brain-modules.md, T4 row).

Acceptance under test:
  * every returned sample matches src/trajectory/shuttle_aerodynamics.rollout within 1e-9
    (the predictor must CALL that rollout, never re-implement the physics);
  * landing_point / arrival_time are the z = 0 (court ground) crossing of that same rollout,
    cross-checked against an independent fine-step numerical solution;
  * the returned object passes the frozen PredictedTrajectory contract;
  * L / k / wind provenance is declared as Param with a source (S4/S12 policy).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from common.status import AssetStatus, Param  # noqa: E402
from trajectory.shuttle_aerodynamics import k_from_aerodynamic_length, rollout  # noqa: E402
from badminton_brain.interfaces import PredictionModule, interface_layer_of  # noqa: E402
from badminton_brain.types import (  # noqa: E402
    BrainBoundaryError, Layer, PredictedTrajectory, UnifiedState,
)
from badminton_brain.prediction import PhysicsTrajectoryPredictor  # noqa: E402

GROUND_Z = 0.0
GRAVITY = np.array([0.0, 0.0, -9.80665])
HORIZON_S = 2.0          # long enough for the canonical shuttle to reach the court
DT_S = 0.005
K_REF = 1.0 / 6.5

POS = np.array([0.0, 0.0, 3.0])
VEL = np.array([4.0, 0.5, 0.5])

BATCH_POS = np.array([
    [0.0, 0.0, 3.0],
    [1.5, -0.4, 2.2],
    [-2.0, 0.7, 3.6],
    [0.5, 1.2, 2.8],
])
BATCH_VEL = np.array([
    [4.0, 0.5, 0.5],
    [-3.0, 0.2, -0.4],
    [2.5, -1.0, 1.0],
    [1.0, 2.0, -0.2],
])


def make_state(position, velocity, timestamp: float = 0.0) -> UnifiedState:
    pos = np.atleast_2d(np.asarray(position, dtype=float))
    vel = np.atleast_2d(np.asarray(velocity, dtype=float))
    n = pos.shape[0]
    return UnifiedState(
        base_pose=np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)),
        base_twist=np.zeros((n, 6)),
        joint_pos=np.zeros((n, 6)),
        joint_vel=np.zeros((n, 6)),
        racket_contact_pose=np.zeros((n, 7)),
        racket_contact_twist=np.zeros((n, 6)),
        shuttle_position=pos,
        shuttle_velocity=vel,
        timestamp=timestamp,
    )


def reference_rollout(position, velocity, *, horizon_s=HORIZON_S, dt_s=DT_S, k_per_m=K_REF):
    return rollout(np.asarray(position, dtype=float), np.asarray(velocity, dtype=float),
                   duration_s=horizon_s, dt_s=dt_s, k_per_m=k_per_m, gravity=GRAVITY)


def ground_crossing(times, positions, ground_z=GROUND_Z):
    """Independent (numpy) linear-interpolation ground crossing of a sampled path."""
    z = positions[:, 2]
    below = np.nonzero(z <= ground_z)[0]
    if below.size == 0 or below[0] == 0:
        raise AssertionError('no ground crossing in the supplied samples')
    i = int(below[0])
    frac = (ground_z - z[i - 1]) / (z[i] - z[i - 1])
    point = positions[i - 1] + frac * (positions[i] - positions[i - 1])
    t = times[i - 1] + frac * (times[i] - times[i - 1])
    return point, t


class DragParameterProvenanceTests(unittest.TestCase):
    def test_aerodynamic_length_and_k_are_sourced_params(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        length = predictor.aerodynamic_length_m
        k_param = predictor.k_per_m
        self.assertIsInstance(length, Param)
        self.assertIsInstance(k_param, Param)
        self.assertAlmostEqual(float(length.value), 6.5, places=12)
        self.assertAlmostEqual(float(k_param.value), K_REF, places=15)
        self.assertNotIn(length.status, (AssetStatus.UNKNOWN, AssetStatus.REQUIRES_MEASUREMENT,
                                         AssetStatus.REQUIRES_CALIBRATION))
        self.assertTrue('Darbois' in length.source or 'literature' in length.source.lower(),
                        msg=f'L source must cite the reference, got {length.source!r}')
        self.assertTrue(k_param.source.strip(), msg='k source is required')
        # k must be derived by the frozen aerodynamics module, not recomputed here
        self.assertEqual(float(k_param.value), k_from_aerodynamic_length(float(length.value)))

    def test_wind_is_a_declared_temp_proxy(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        self.assertIsInstance(predictor.wind_mps, Param)
        self.assertEqual(predictor.wind_mps.status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertTrue(predictor.wind_mps.source.strip())
        self.assertTrue(np.allclose(np.asarray(predictor.wind_mps.value, dtype=float), 0.0))


class RolloutAgreementTests(unittest.TestCase):
    def test_single_trajectory_matches_rollout_within_1e_9(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(POS, VEL))
        ref = reference_rollout(POS, VEL)

        self.assertIsInstance(traj, PredictedTrajectory)
        self.assertEqual(traj.position.shape, (1, ref['position'].shape[0], 3))
        self.assertEqual(traj.times.shape, (ref['time'].shape[0],))
        self.assertLessEqual(float(np.max(np.abs(traj.times - ref['time']))), 1e-9)
        self.assertLessEqual(float(np.max(np.abs(traj.position[0] - ref['position']))), 1e-9)
        self.assertLessEqual(float(np.max(np.abs(traj.velocity[0] - ref['velocity']))), 1e-9)

    def test_batched_trajectories_each_match_their_own_rollout_within_1e_9(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(BATCH_POS, BATCH_VEL))
        n = BATCH_POS.shape[0]
        self.assertEqual(traj.position.shape[0], n)
        self.assertEqual(traj.velocity.shape[0], n)
        for i in range(n):
            ref = reference_rollout(BATCH_POS[i], BATCH_VEL[i])
            self.assertLessEqual(float(np.max(np.abs(traj.position[i] - ref['position']))), 1e-9,
                                 msg=f'batch element {i} diverged from rollout')
            self.assertLessEqual(float(np.max(np.abs(traj.velocity[i] - ref['velocity']))), 1e-9,
                                 msg=f'batch element {i} velocity diverged from rollout')

    def test_step_size_follows_the_configured_dt(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=1.0, dt_s=0.02)
        traj = predictor.process(make_state(POS, VEL))
        self.assertAlmostEqual(float(traj.times[0]), 0.0, places=15)
        self.assertAlmostEqual(float(traj.times[-1]), 1.0, places=12)
        self.assertLessEqual(float(np.max(np.diff(traj.times)) - 0.02), 1e-12)


class LandingPointTests(unittest.TestCase):
    def test_landing_point_and_arrival_time_match_the_ground_crossing(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(BATCH_POS, BATCH_VEL))
        for i in range(BATCH_POS.shape[0]):
            ref = reference_rollout(BATCH_POS[i], BATCH_VEL[i])
            point, t = ground_crossing(ref['time'], ref['position'])
            self.assertLessEqual(float(np.max(np.abs(traj.landing_point[i] - point))), 1e-12,
                                 msg=f'landing point mismatch for batch element {i}')
            self.assertLessEqual(abs(float(traj.arrival_time[i]) - t), 1e-12)
            self.assertAlmostEqual(float(traj.landing_point[i][2]), GROUND_Z, places=12)

    def test_landing_point_matches_an_independent_fine_step_solution(self) -> None:
        """Same physics, 50x finer RK4 steps: catches interpolation errors, not just re-derivation."""
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(POS, VEL))
        fine = reference_rollout(POS, VEL, dt_s=1e-4)
        point, t = ground_crossing(fine['time'], fine['position'])
        self.assertLessEqual(float(np.max(np.abs(traj.landing_point[0] - point))), 1e-4)
        self.assertLessEqual(abs(float(traj.arrival_time[0]) - t), 1e-4)
        self.assertLess(float(traj.arrival_time[0]), HORIZON_S)

    def test_landing_point_lies_on_the_returned_discretised_trajectory(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(POS, VEL))
        t = float(traj.arrival_time[0])
        pos = np.asarray(traj.position[0], dtype=float)
        interpolated = np.array([np.interp(t, traj.times, pos[:, axis]) for axis in range(3)])
        self.assertLessEqual(float(np.max(np.abs(interpolated - traj.landing_point[0]))), 1e-9)

    def test_no_ground_crossing_within_horizon_is_flagged_and_finite(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=0.05, dt_s=DT_S)
        traj = predictor.process(make_state(POS, VEL))
        self.assertFalse(bool(traj.landed_within_horizon[0]))
        self.assertTrue(np.all(np.isfinite(traj.landing_point)))
        self.assertAlmostEqual(float(traj.landing_point[0][2]), GROUND_Z, places=12)
        self.assertAlmostEqual(float(traj.arrival_time[0]), 0.05, places=12)

    def test_ground_crossing_flag_is_true_for_the_canonical_incoming_shuttle(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(POS, VEL))
        self.assertTrue(bool(traj.landed_within_horizon[0]))
        self.assertAlmostEqual(float(traj.landing_point[0][2]), GROUND_Z, places=12)


class ContractAndConfigTests(unittest.TestCase):
    def test_returned_object_satisfies_the_frozen_contract(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        traj = predictor.process(make_state(BATCH_POS, BATCH_VEL, timestamp=1.25))
        n = BATCH_POS.shape[0]
        self.assertEqual(traj.frame, 'court')
        self.assertAlmostEqual(traj.timestamp, 1.25, places=15)
        self.assertEqual(traj.times.ndim, 1)
        self.assertEqual(traj.position.ndim, 3)
        self.assertEqual(traj.position.shape[0], n)
        self.assertEqual(traj.velocity.shape, traj.position.shape)
        self.assertEqual(traj.landing_point.shape, (n, 3))
        self.assertEqual(traj.arrival_time.shape, (n,))
        self.assertTrue(np.all(np.isfinite(traj.landed_within_horizon)))
        # the frozen contract must still reject a mismatched horizon
        with self.assertRaises(BrainBoundaryError):
            PredictedTrajectory(times=traj.times[:3], position=traj.position,
                                velocity=traj.velocity, landing_point=traj.landing_point,
                                arrival_time=traj.arrival_time)

    def test_horizon_is_configurable(self) -> None:
        short = PhysicsTrajectoryPredictor(horizon_s=0.3, dt_s=DT_S).process(make_state(POS, VEL))
        long = PhysicsTrajectoryPredictor(horizon_s=0.6, dt_s=DT_S).process(make_state(POS, VEL))
        self.assertAlmostEqual(float(short.times[-1]), 0.3, places=12)
        self.assertAlmostEqual(float(long.times[-1]), 0.6, places=12)
        self.assertLess(short.times.shape[0], long.times.shape[0])
        self.assertEqual(short.position.shape[0], long.position.shape[0])

    def test_default_horizon_parameter_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsTrajectoryPredictor(horizon_s=0.0)
        with self.assertRaises(ValueError):
            PhysicsTrajectoryPredictor(horizon_s=-1.0)
        with self.assertRaises(ValueError):
            PhysicsTrajectoryPredictor(horizon_s=1.0, dt_s=0.0)
        with self.assertRaises(ValueError):
            PhysicsTrajectoryPredictor(horizon_s=0.01, dt_s=0.5)

    def test_process_rejects_non_unified_state_and_missing_shuttle(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        with self.assertRaises(BrainBoundaryError):
            predictor.process(np.zeros((2, 3)))
        stripped = make_state(POS, VEL)   # a contract-valid state whose shuttle was dropped
        stripped.shuttle_position = None
        with self.assertRaises(BrainBoundaryError):
            predictor.process(stripped)

    def test_module_declares_the_prediction_layer_interface(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=HORIZON_S, dt_s=DT_S)
        self.assertIsInstance(predictor, PredictionModule)
        self.assertEqual(predictor.layer, Layer.PREDICTION)
        self.assertIs(interface_layer_of(predictor), Layer.PREDICTION)
        self.assertTrue(predictor.is_implemented)

    def test_k_is_configurable_and_single_unbatched_input_is_supported(self) -> None:
        predictor = PhysicsTrajectoryPredictor(horizon_s=0.5, dt_s=DT_S, k_per_m=0.05)
        traj = predictor.process(make_state(POS, VEL))
        ref = reference_rollout(POS, VEL, horizon_s=0.5, k_per_m=0.05)
        self.assertEqual(traj.position.shape, (1, ref['position'].shape[0], 3))
        self.assertLessEqual(float(np.max(np.abs(traj.position[0] - ref['position']))), 1e-9)


if __name__ == '__main__':
    unittest.main()
