# -*- coding: utf-8 -*-
"""T10 online adaptation tests (plan docs/superpowers/plans/2026-09-13-brain-modules.md, row T10).

Contract under test (src/badminton_brain/adaptation/online_adaptation.py):
  * consumes (Feedback, UnifiedState), returns a correction dict (AdaptationModule.output_type);
  * Feedback.prediction_error is the PREDICTION residual in metres (DEC-015): predicted shuttle
    position minus measured.  Feedback.tracking_residual is the execution residual and must be
    ignored here;
  * the observation interval is the inter-step interval, state.timestamp - previous state.timestamp,
    because in the pipeline Feedback.timestamp equals the UnifiedState timestamp (same step);
  * prediction error drives a bounded Gauss-Newton / normalized-LMS update of the slow parameters
    drag_scale / wind / delay, with gain semantics theta += -gain * (theta - theta_true);
  * every estimate is clamped into declared bounds, per-step increments are capped;
  * a residual ledger keeps the last K prediction errors per environment;
  * reset(env_ids) clears only the selected environments, including the stored previous state.

The convergence tests never re-use the module's update algebra: the residual fed back comes from
the real quadratic-drag model / RK4 integrator in src/trajectory/shuttle_aerodynamics.py.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'trajectory'))

from shuttle_aerodynamics import acceleration, k_from_aerodynamic_length, rk4_step  # noqa: E402

from badminton_brain.adaptation.online_adaptation import OnlineAdaptation  # noqa: E402
from badminton_brain.interfaces import AdaptationModule, interface_layer_of  # noqa: E402
from badminton_brain.status import AssetStatus  # noqa: E402
from badminton_brain.types import (BrainBoundaryError, Feedback, Layer,  # noqa: E402
                                   PredictedTrajectory, UnifiedState)

K_BASE = k_from_aerodynamic_length(6.5)          # literature feather shuttle L = 6.5 m
GRAVITY = np.array([0.0, 0.0, -9.80665])
DT = 0.2                                          # inter-step interval used by every test


def make_state(position, velocity, timestamp: float) -> UnifiedState:
    n = np.asarray(velocity, dtype=float).shape[0]
    return UnifiedState(
        base_pose=np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)),
        base_twist=np.zeros((n, 6)),
        joint_pos=np.zeros((n, 6)),
        joint_vel=np.zeros((n, 6)),
        racket_contact_pose=np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)),
        racket_contact_twist=np.zeros((n, 6)),
        shuttle_position=np.asarray(position, dtype=float),
        shuttle_velocity=np.asarray(velocity, dtype=float),
        timestamp=timestamp)


def position_residual(position, velocity, *, pred_k, pred_wind, true_k, true_wind) -> np.ndarray:
    """predicted - measured position after DT, both from the real drag model (single env)."""
    p = np.asarray(position, dtype=float)[0]
    v = np.asarray(velocity, dtype=float)[0]
    a_pred = acceleration(p, v, k_per_m=pred_k, gravity=GRAVITY,
                          wind=np.asarray(pred_wind, dtype=float).reshape(-1)[:3])
    a_true = acceleration(p, v, k_per_m=true_k, gravity=GRAVITY,
                          wind=np.asarray(true_wind, dtype=float).reshape(-1)[:3])
    p_pred = p + v * DT + 0.5 * a_pred * DT ** 2
    p_obs = p + v * DT + 0.5 * a_true * DT ** 2
    return (p_pred - p_obs)[None, :]


def rk4_position(position, velocity, duration_s: float) -> np.ndarray:
    """Real RK4 rollout of the shuttle for duration_s (single env)."""
    p, _ = rk4_step(np.asarray(position, dtype=float)[0], np.asarray(velocity, dtype=float)[0],
                    duration_s, k_per_m=K_BASE, gravity=GRAVITY)
    return p[None, :]


def make_prediction(positions, timestamp: float, times) -> PredictedTrajectory:
    """Build a PredictedTrajectory whose grid is ABSOLUTE simulation time (times[0] == timestamp)."""
    positions = np.asarray(positions, dtype=float)
    n, t = positions.shape[0], positions.shape[1]
    times = np.asarray(times, dtype=float)
    return PredictedTrajectory(times=times, position=positions,
                               velocity=np.zeros((n, t, 3)),
                               landing_point=positions[:, -1, :].copy(),
                               arrival_time=np.full(n, float(times[-1])),
                               timestamp=float(timestamp))


def orbit_velocity(step: int, speed: float = 4.0) -> np.ndarray:
    """Deterministic changing velocity direction so the residual is not a single fixed axis."""
    theta = 0.05 * step
    return np.array([[speed * np.cos(theta), speed * np.sin(theta), -0.5]])


def dt_of(out, env: int = 0) -> float:
    return float(np.asarray(out['dt_s'], dtype=float).reshape(-1)[env])


class LayerContractTests(unittest.TestCase):
    def test_module_binds_to_the_adaptation_layer(self) -> None:
        module = OnlineAdaptation(num_envs=2)
        self.assertIsInstance(module, AdaptationModule)
        self.assertIs(interface_layer_of(module), Layer.ADAPTATION)
        self.assertEqual(module.layer, Layer.ADAPTATION)

    def test_process_returns_the_correction_dict(self) -> None:
        module = OnlineAdaptation(num_envs=2)
        state = make_state(np.zeros((2, 3)), orbit_velocity(0).repeat(2, axis=0), 0.0)
        out = module.process(Feedback(prediction_error=np.zeros((2, 3)), timestamp=0.0), state)
        self.assertIsInstance(out, dict)
        for key in ('drag_scale', 'wind', 'delay_s', 'residual', 'updates'):
            self.assertIn(key, out)
        self.assertEqual(out['drag_scale'].shape, (2,))
        self.assertEqual(out['wind'].shape, (2, 3))
        self.assertEqual(out['delay_s'].shape, (2,))


class UpdateLawTests(unittest.TestCase):
    """D1/D2/D3/D4 of the T10 review: gain semantics, inter-step dt, DEC-015 semantics, ledger API."""

    def test_drag_update_gain_matches_the_declared_gain(self) -> None:
        """theta += -gain * (theta - theta_true), for any theta (a k_total factor would break this)."""
        true_scale = 1.0
        gain = 0.4
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        analytic = (-0.5 * DT * DT * K_BASE * float(np.linalg.norm(velocity)) * velocity[0])
        delta = 1e-3
        jac_fd = (position_residual(position, velocity, pred_k=K_BASE * (1.0 + delta),
                                    pred_wind=np.zeros(3), true_k=K_BASE * true_scale,
                                    true_wind=np.zeros(3))
                  - position_residual(position, velocity, pred_k=K_BASE * (1.0 - delta),
                                      pred_wind=np.zeros(3), true_k=K_BASE * true_scale,
                                      true_wind=np.zeros(3))) / (2.0 * delta)
        # the finite difference pins the Jacobian FORM: -0.5*dt^2*k*|r|*r, with no k_total factor
        # (a k_total factor would show up here as a 3x error at drag_scale = 3)
        np.testing.assert_allclose(jac_fd[0], analytic, rtol=1e-5,
                                   err_msg='drag Jacobian must be -0.5*dt^2*k*|r|*r, no k_total')
        for theta in (0.5, 1.0, 3.0):
            # eps is the only thing between the update and the exact gain law here (the default
            # 1e-9 perturbs this denominator by ~4e-7 relative), so shrink it to isolate the gain
            module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',),
                                      gain_drag_scale=gain, max_step_drag_scale=5.0, eps=1e-30)
            module.drag_scale[0] = theta
            error = analytic[None, :] * (theta - true_scale)
            module.process(Feedback(prediction_error=error, timestamp=0.0),
                           make_state(position, velocity, 0.0))
            out = module.process(Feedback(prediction_error=error, timestamp=DT),
                                 make_state(position, velocity, DT))
            self.assertTrue(bool(out['updated'][0]))
            expected = theta - gain * (theta - true_scale)
            self.assertAlmostEqual(float(module.drag_scale[0]), expected, places=9,
                                   msg='gain was not %g at drag_scale=%g' % (gain, theta))

    def test_inter_step_interval_drives_the_update(self) -> None:
        """In the pipeline feedback.timestamp == state.timestamp, so dt must be inter-step."""
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        error = np.array([[0.05, 0.0, 0.0]])
        first = module.process(Feedback(prediction_error=error, timestamp=0.0),
                               make_state(position, velocity, 0.0))
        self.assertFalse(bool(first['updated'][0]), 'the first sample only stores the previous state')
        self.assertEqual(int(module.updates[0]), 0)
        self.assertAlmostEqual(float(module.drag_scale[0]), 1.0, places=15)

        second = module.process(Feedback(prediction_error=error, timestamp=0.05),
                                make_state(position, velocity, 0.05))
        self.assertTrue(bool(second['updated'][0]), 'the second sample must use the inter-step dt')
        self.assertAlmostEqual(dt_of(second), 0.05, places=12)
        self.assertEqual(int(module.updates[0]), 1)
        self.assertNotAlmostEqual(float(module.drag_scale[0]), 1.0, places=6)

        repeat = module.process(Feedback(prediction_error=error, timestamp=0.05),
                                make_state(position, velocity, 0.05))
        self.assertFalse(bool(repeat['updated'][0]), 'replaying the same step must not update twice')
        self.assertAlmostEqual(dt_of(repeat), 0.0, places=12)
        self.assertEqual(int(module.updates[0]), 1, 'one step must produce exactly one update')
        self.assertEqual(int(module.residual_counts()[0]), 3, 'every finite residual is recorded')

    def test_reset_clears_the_previous_state(self) -> None:
        module = OnlineAdaptation(num_envs=2, estimate=('drag_scale',))
        position = np.zeros((2, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = orbit_velocity(0).repeat(2, axis=0)
        error = np.array([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])
        module.process(Feedback(prediction_error=error, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        second = module.process(Feedback(prediction_error=error, timestamp=0.05),
                                make_state(position, velocity, 0.05))
        self.assertTrue(np.all(second['updated']))
        updated_before = float(module.drag_scale[1])

        module.reset([0])

        after = module.process(Feedback(prediction_error=error, timestamp=0.10),
                               make_state(position, velocity, 0.10))
        self.assertFalse(bool(after['updated'][0]), 'reset must clear the stored previous state')
        self.assertTrue(np.isnan(dt_of(after, 0)), 'a cleared environment has no interval yet')
        self.assertAlmostEqual(float(module.drag_scale[0]), 1.0, places=15)
        self.assertTrue(bool(after['updated'][1]), 'the untouched environment keeps updating')
        self.assertAlmostEqual(dt_of(after, 1), 0.05, places=12)
        self.assertNotAlmostEqual(float(module.drag_scale[1]), updated_before, places=9)

    def test_meter_level_prediction_error_does_not_saturate_everything(self) -> None:
        """DEC-015: prediction_error is in metres; one 5 m sample must cap, not saturate."""
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale', 'wind', 'delay'))
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        error = np.array([[5.0, 5.0, 5.0]])
        module.process(Feedback(prediction_error=error, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        out = module.process(Feedback(prediction_error=error, timestamp=0.05),
                             make_state(position, velocity, 0.05))
        self.assertTrue(bool(out['updated'][0]))
        self.assertTrue(bool(out['limited'][0]), 'a 5 m sample must be capped per step')
        self.assertFalse(bool(out['clamped'][0]), 'a single sample must not pin any estimate')
        self.assertLess(module.drag_scale_bounds[0], float(module.drag_scale[0]))
        self.assertLess(float(module.drag_scale[0]), module.drag_scale_bounds[1])
        self.assertLess(np.abs(module.wind[0]).max(), module.wind_bounds_mps[1])
        self.assertLess(abs(float(module.delay_s[0])), module.delay_bounds_s[1])

    def test_tracking_residual_is_not_consumed(self) -> None:
        """DEC-015: the execution residual must never move the shuttle estimates."""
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        quiet = OnlineAdaptation(num_envs=1, estimate=('drag_scale', 'wind', 'delay'))
        noisy = OnlineAdaptation(num_envs=1, estimate=('drag_scale', 'wind', 'delay'))
        for step in range(3):
            t0 = step * DT
            prediction = np.array([[0.01, 0.0, 0.0]])
            quiet.process(Feedback(prediction_error=prediction, timestamp=t0),
                          make_state(position, velocity, t0))
            noisy.process(Feedback(prediction_error=prediction,
                                   tracking_residual=np.array([9.0]), timestamp=t0),
                          make_state(position, velocity, t0))
        np.testing.assert_array_equal(noisy.drag_scale, quiet.drag_scale)
        np.testing.assert_array_equal(noisy.wind, quiet.wind)
        np.testing.assert_array_equal(noisy.delay_s, quiet.delay_s)
        self.assertNotAlmostEqual(float(noisy.drag_scale[0]), 1.0, places=6)

        only_tracking = noisy.process(
            Feedback(prediction_error=None, tracking_residual=np.array([9.0]), timestamp=3 * DT),
            make_state(position, velocity, 3 * DT))
        self.assertFalse(bool(only_tracking['updated'][0]))
        self.assertEqual(int(noisy.residual_counts()[0]), 3, 'no prediction error, no ledger entry')
        np.testing.assert_array_equal(noisy.drag_scale, quiet.drag_scale)

    def test_measurement_requirements_lists_unmeasured_quantities(self) -> None:
        module = OnlineAdaptation(num_envs=2)
        requirements = module.measurement_requirements()
        self.assertIsInstance(requirements, dict)
        self.assertTrue(requirements)
        for name, param in requirements.items():
            self.assertIsNone(param.value, name)
            self.assertIs(param.status, AssetStatus.REQUIRES_MEASUREMENT, name)
            self.assertTrue(str(param.source).strip(), name)
        self.assertTrue(any('gain' in name for name in requirements))
        self.assertIn('wind_bounds_mps', requirements)
        self.assertNotIn('aerodynamic_length_m', requirements,
                         'the literature aerodynamic length is resolved, not a requirement')


class ConvergenceTests(unittest.TestCase):
    """Systematic bias -> slow estimate converges to the true value."""

    def test_drag_scale_converges_to_truth(self) -> None:
        true_scale = 1.25
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        position = np.array([[0.0, 0.0, 3.0]])
        errors = []
        for step in range(60):
            velocity = orbit_velocity(step)
            t0 = step * DT
            residual = position_residual(
                position, velocity,
                pred_k=K_BASE * float(module.drag_scale[0]), pred_wind=np.zeros(3),
                true_k=K_BASE * true_scale, true_wind=np.zeros(3))
            errors.append(abs(float(module.drag_scale[0]) - true_scale))
            module.process(Feedback(prediction_error=residual, timestamp=t0),
                           make_state(position, velocity, t0))
        # the first sample only stores the previous state, so it must not move the estimate
        self.assertAlmostEqual(errors[1], errors[0], places=15)
        # thereafter the error contracts by exactly 1 - gain_drag_scale per sample (0.6 here)
        for before, after in zip(errors[1:20], errors[2:21]):
            self.assertLessEqual(after, 0.61 * before,
                                 'contraction slower than 1 - gain_drag_scale')
        self.assertLess(errors[-1], 1e-9, 'drag scale did not converge: %r' % module.drag_scale[0])

    def test_wind_estimate_converges_to_truth(self) -> None:
        true_wind = np.array([0.6, -0.4, 0.0])
        module = OnlineAdaptation(num_envs=1, estimate=('wind',))
        position = np.array([[0.0, 0.0, 3.0]])
        errors = []
        for step in range(60):
            velocity = orbit_velocity(step)
            t0 = step * DT
            residual = position_residual(
                position, velocity,
                pred_k=K_BASE, pred_wind=module.wind[0],
                true_k=K_BASE, true_wind=true_wind)
            errors.append(float(np.abs(module.wind[0] - true_wind).max()))
            module.process(Feedback(prediction_error=residual, timestamp=t0),
                           make_state(position, velocity, t0))
        np.testing.assert_allclose(module.wind[0], true_wind, atol=1e-4)
        self.assertTrue(all(after < before for before, after in zip(errors[1:30], errors[2:31])),
                        'the wind error must decrease on every sample')
        self.assertLess(errors[19], 2e-2, 'wind did not converge within 20 samples')

    def test_delay_estimate_converges_to_truth(self) -> None:
        """Residual generated by the real RK4 rollout, not by the module's own linear model.

        The true pipeline latency is true_delay: the state the predictor receives is that much
        older than its timestamp.  The predictor compensates the latency it believes in, so
        p_pred = rollout(DT - true_delay + assumed_delay) and the residual vanishes only at
        assumed_delay == true_delay.
        """
        true_delay = 0.02
        module = OnlineAdaptation(num_envs=1, estimate=('delay',))
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        measured = rk4_position(position, velocity, DT)
        # the sign convention itself, checked against RK4 instead of assumed: a longer assumed
        # latency stretches the rollout, so d(residual)/d(delay) is +v (the old -v diverged here)
        probe = 1e-4
        jac_fd = (rk4_position(position, velocity, DT + true_delay + probe)
                  - rk4_position(position, velocity, DT + true_delay - probe)) / (2.0 * probe)
        self.assertGreater(float(jac_fd[0] @ velocity[0]), 0.0,
                           'latency sensitivity must be +v, not -v')
        # d(rollout position)/d(duration) is the velocity AT THE END of the rollout, not the start
        _, end_velocity = rk4_step(position[0], velocity[0], DT + true_delay,
                                   k_per_m=K_BASE, gravity=GRAVITY)
        np.testing.assert_allclose(jac_fd[0], end_velocity, rtol=0.02)
        errors = []
        for step in range(60):
            t0 = step * DT
            assumed = float(module.delay_s[0])
            predicted = rk4_position(position, velocity, DT - true_delay + assumed)
            errors.append(abs(assumed - true_delay))
            module.process(Feedback(prediction_error=predicted - measured, timestamp=t0),
                           make_state(position, velocity, t0))
        self.assertAlmostEqual(errors[1], errors[0], places=15)
        self.assertGreater(float(module.delay_s[0]), module.delay_bounds_s[0],
                           'the estimate must converge, not run into a bound')
        self.assertLess(float(module.delay_s[0]), module.delay_bounds_s[1])
        self.assertLess(errors[-1], 1e-4, 'delay estimate did not converge: %r' % module.delay_s[0])
        self.assertLess(errors[-1], 1e-2 * errors[1])


class BoundedUpdateTests(unittest.TestCase):
    def test_estimates_stay_inside_bounds_and_steps_are_capped(self) -> None:
        module = OnlineAdaptation(
            num_envs=1,
            estimate=('drag_scale', 'wind', 'delay'),
            drag_scale_bounds=(0.5, 2.0),
            wind_bounds_mps=(-1.0, 1.0),
            delay_bounds_s=(-0.02, 0.02),
            max_step_drag_scale=0.05,
            max_step_wind_mps=0.1,
            max_step_delay_s=0.001)
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        previous = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        limited_seen = False
        out = None
        for step in range(400):
            out = module.process(Feedback(prediction_error=np.array([[1.0, 0.5, -0.4]]),
                                          timestamp=step * DT),
                                 make_state(position, velocity, step * DT))
            now = np.array([out['drag_scale'][0], out['wind'][0, 0], out['wind'][0, 1],
                            out['wind'][0, 2], out['delay_s'][0]])
            step_size = np.abs(now - previous)
            self.assertLessEqual(step_size[0], 0.05 + 1e-12)
            self.assertLessEqual(step_size[1:4].max(), 0.1 + 1e-12)
            self.assertLessEqual(step_size[4], 0.001 + 1e-12)
            self.assertLessEqual(now[0], 2.0 + 1e-12)
            self.assertGreaterEqual(now[0], 0.5 - 1e-12)
            self.assertLessEqual(np.abs(now[1:4]).max(), 1.0 + 1e-12)
            self.assertLessEqual(abs(now[4]), 0.02 + 1e-12)
            self.assertTrue(np.all(np.isfinite(now)))
            limited_seen = limited_seen or bool(out['limited'][0])
            previous = now
        self.assertTrue(limited_seen, 'the per-step cap never engaged')
        self.assertTrue(bool(out['clamped'][0]), 'no bound was reported as engaged')
        # the persistent residual keeps pushing every estimate into its bound
        self.assertAlmostEqual(float(module.drag_scale[0]), 2.0, places=9)
        # the residual has a positive component along v, so a predictor that is rolling out too long
        # must reduce the latency it assumes: the delay estimate saturates at its LOWER bound
        self.assertAlmostEqual(float(module.delay_s[0]), -0.02, places=9)
        self.assertAlmostEqual(float(module.wind[0, 0]), -1.0, places=9)
        self.assertAlmostEqual(float(module.wind[0, 2]), 1.0, places=9)

    def test_nonfinite_error_is_ignored_and_keeps_state_finite(self) -> None:
        module = OnlineAdaptation(num_envs=2, estimate=('drag_scale',))
        position = np.zeros((2, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = orbit_velocity(0).repeat(2, axis=0)
        bad = np.array([[np.nan, 0.0, 0.0], [0.2, 0.0, 0.0]])
        module.process(Feedback(prediction_error=bad, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        out = module.process(Feedback(prediction_error=bad, timestamp=DT),
                             make_state(position, velocity, DT))
        self.assertAlmostEqual(float(module.drag_scale[0]), 1.0, places=12)
        self.assertNotAlmostEqual(float(module.drag_scale[1]), 1.0)
        self.assertTrue(np.all(np.isfinite(module.drag_scale)))
        self.assertTrue(np.isnan(out['residual'][0]))
        self.assertTrue(out['updated'][1])
        self.assertFalse(out['updated'][0])
        self.assertEqual(int(module.residual_counts()[0]), 0)
        self.assertEqual(int(module.residual_counts()[1]), 2)

    def test_random_large_errors_never_diverge(self) -> None:
        rng = np.random.default_rng(20260913)
        window = 8
        module = OnlineAdaptation(num_envs=4, estimate=('drag_scale', 'wind', 'delay'),
                                  ledger_size=window)
        position = np.zeros((4, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = np.tile(np.array([[4.0, 1.0, -0.5]]), (4, 1))
        pushed = []
        for step in range(500):
            error = rng.normal(scale=25.0, size=(4, 3))     # metres of nonsense
            pushed.append(error.copy())
            out = module.process(Feedback(prediction_error=error, timestamp=step * DT),
                                 make_state(position, velocity, step * DT))
            self.assertTrue(np.all(np.isfinite(out['drag_scale'])))
            self.assertTrue(np.all(np.isfinite(out['wind'])))
            self.assertTrue(np.all(np.isfinite(out['delay_s'])))
            self.assertLessEqual(np.abs(out['drag_scale']).max(), 4.0 + 1e-12)
            self.assertLessEqual(np.abs(out['wind']).max(), 5.0 + 1e-12)
            self.assertLessEqual(np.abs(out['delay_s']).max(), 0.05 + 1e-12)
        self.assertTrue(np.all(module.drag_scale >= module.drag_scale_bounds[0] - 1e-12))
        self.assertTrue(np.all(module.drag_scale <= module.drag_scale_bounds[1] + 1e-12))
        self.assertTrue(np.all(module.wind >= module.wind_bounds_mps[0] - 1e-12))
        self.assertTrue(np.all(module.wind <= module.wind_bounds_mps[1] + 1e-12))
        self.assertTrue(np.all(np.abs(module.delay_s) <= module.delay_bounds_s[1] + 1e-12))
        np.testing.assert_array_equal(module.residual_counts(), [window] * 4)
        # the ledger must hold the newest window samples, in order, and nothing else
        np.testing.assert_allclose(module.residual_ledger()[:, :, 0],
                                   np.asarray(pushed[-window:])[:, :, 0].T)
        np.testing.assert_allclose(
            module.residual_mean(),
            np.linalg.norm(np.asarray(pushed[-window:]), axis=2).mean(axis=0), atol=1e-12)

    def test_mismatched_batch_and_wrong_types_raise(self) -> None:
        module = OnlineAdaptation(num_envs=2)
        state = make_state(np.zeros((2, 3)), orbit_velocity(0).repeat(2, axis=0), 0.0)
        with self.assertRaises(BrainBoundaryError):
            module.process(Feedback(prediction_error=np.zeros((3, 3)), timestamp=0.0), state)
        with self.assertRaises(BrainBoundaryError):
            module.process(Feedback(prediction_error=np.zeros((2, 3)), timestamp=0.0),
                           make_state(np.zeros((3, 3)), orbit_velocity(0).repeat(3, axis=0), 0.0))
        with self.assertRaises(BrainBoundaryError):
            module.process(object(), state)


class LedgerTests(unittest.TestCase):
    def test_ledger_keeps_the_last_k_errors_per_env(self) -> None:
        window = 4
        module = OnlineAdaptation(num_envs=2, estimate=('drag_scale',), ledger_size=window)
        position = np.zeros((2, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = orbit_velocity(0).repeat(2, axis=0)
        pushed = []
        for step in range(6):
            error = np.zeros((2, 3))
            error[0, 0] = 0.1 * (step + 1)
            error[1, 0] = -0.2 * (step + 1)
            pushed.append(error.copy())
            module.process(Feedback(prediction_error=error, timestamp=step * DT),
                           make_state(position, velocity, step * DT))
        ledger = module.residual_ledger()
        self.assertEqual(ledger.shape, (2, window, 3))
        np.testing.assert_allclose(ledger[0, :, 0], [0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(ledger[1, :, 0], [-0.6, -0.8, -1.0, -1.2])
        np.testing.assert_allclose(ledger[0], np.asarray(pushed[-window:])[:, 0])
        norms = module.residual_norms()
        np.testing.assert_allclose(norms[0], [0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(module.residual_mean(), [0.45, 0.9], atol=1e-12)
        np.testing.assert_array_equal(module.residual_counts(), [window, window])

    def test_ledger_is_nan_padded_before_it_is_full(self) -> None:
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',), ledger_size=3)
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        error = np.array([[0.5, 0.0, 0.0]])
        module.process(Feedback(prediction_error=error, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        ledger = module.residual_ledger()
        np.testing.assert_allclose(ledger[0, -1], error[0])
        self.assertTrue(np.all(np.isnan(ledger[0, :-1])))
        self.assertEqual(int(module.residual_counts()[0]), 1)

    def test_scalar_error_is_interpreted_along_the_shuttle_velocity(self) -> None:
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',), ledger_size=2)
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = np.array([[3.0, 4.0, 0.0]])
        module.process(Feedback(prediction_error=np.array([2.0]), timestamp=0.0),
                       make_state(position, velocity, 0.0))
        np.testing.assert_allclose(module.residual_ledger()[0, -1], [1.2, 1.6, 0.0], atol=1e-12)

    def test_missing_error_does_not_update_but_is_reported(self) -> None:
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        state = make_state(np.array([[0.0, 0.0, 3.0]]), orbit_velocity(0), 0.0)
        out = module.process(Feedback(prediction_error=None, timestamp=0.0), state)
        self.assertFalse(bool(out['updated'][0]))
        self.assertTrue(np.isnan(out['residual'][0]))
        self.assertAlmostEqual(float(module.drag_scale[0]), 1.0, places=12)
        self.assertEqual(int(module.residual_counts()[0]), 0)


class ResetTests(unittest.TestCase):
    def test_reset_clears_only_the_selected_envs(self) -> None:
        module = OnlineAdaptation(num_envs=3, estimate=('drag_scale', 'wind', 'delay'),
                                  ledger_size=4)
        position = np.zeros((3, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = orbit_velocity(0).repeat(3, axis=0)
        for step in range(6):
            error = np.array([[0.3 * (step + 1), 0.0, 0.0],
                              [0.6 * (step + 1), 0.0, 0.0],
                              [-0.2 * (step + 1), 0.0, 0.0]])
            module.process(Feedback(prediction_error=error, timestamp=step * DT),
                           make_state(position, velocity, step * DT))
        self.assertNotAlmostEqual(float(module.drag_scale[0]), 1.0, places=6)
        self.assertNotAlmostEqual(float(module.drag_scale[1]), 1.0, places=6)
        drag_before = module.drag_scale.copy()
        wind_before = module.wind.copy()
        delay_before = module.delay_s.copy()
        counts_before = module.residual_counts().copy()
        updates_before = module.updates.copy()

        module.reset([0])

        np.testing.assert_allclose(module.drag_scale[0], 1.0)
        np.testing.assert_allclose(module.wind[0], np.zeros(3))
        np.testing.assert_allclose(module.delay_s[0], 0.0)
        self.assertEqual(int(module.residual_counts()[0]), 0)
        self.assertEqual(int(module.updates[0]), 0)
        self.assertTrue(np.all(np.isnan(module.residual_ledger()[0])))

        np.testing.assert_array_equal(module.drag_scale[1:], drag_before[1:])
        np.testing.assert_array_equal(module.wind[1:], wind_before[1:])
        np.testing.assert_array_equal(module.delay_s[1:], delay_before[1:])
        np.testing.assert_array_equal(module.residual_counts()[1:], counts_before[1:])
        np.testing.assert_array_equal(module.updates[1:], updates_before[1:])

        module.reset(np.array([2]))
        np.testing.assert_allclose(module.drag_scale[2], 1.0)
        np.testing.assert_array_equal(module.drag_scale[1], drag_before[1])

    def test_reset_rejects_out_of_range_ids(self) -> None:
        module = OnlineAdaptation(num_envs=2)
        with self.assertRaises(BrainBoundaryError):
            module.reset([5])


class PredictionResidualTests(unittest.TestCase):
    """The slow loop computes the residual itself from the prediction pushed by the application.

    Order: the application pushes the trajectory with set_prediction() and the module compares it
    with the state that arrives on the next step, over the inter-step interval (D2).  An explicit
    Feedback.prediction_error still wins when a measurement arrives late.
    """

    def test_residual_is_computed_from_the_pushed_prediction(self) -> None:
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        offset = np.array([[0.06, 0.0, 0.0]])       # the prediction is 6 cm ahead of the truth

        first = module.process(Feedback(prediction_error=None, timestamp=0.0),
                               make_state(position, velocity, 0.0))
        self.assertFalse(bool(first['updated'][0]))
        self.assertEqual(first['residual_source'][0], 'none')

        module.set_prediction(make_prediction(np.stack([position, position + offset], axis=1),
                                              0.0, [0.0, DT]))
        self.assertTrue(bool(module.diagnostics()['prediction_available'][0]))

        second = module.process(Feedback(prediction_error=None, timestamp=DT),
                                make_state(position, velocity, DT))
        self.assertTrue(bool(second['updated'][0]), 'a pushed prediction must drive the update')
        self.assertEqual(second['residual_source'][0], 'prediction')
        np.testing.assert_allclose(module.residual_ledger()[0, -1], offset[0], atol=1e-12)
        self.assertAlmostEqual(float(second['residual'][0]),
                               float(np.linalg.norm(offset)), places=12)
        # predicted ahead of the measurement along +v means the model under-estimates drag
        self.assertGreater(float(module.drag_scale[0]), 1.0)

        inline = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        inline.process(Feedback(prediction_error=None, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        inline_out = inline.process(
            Feedback(prediction_error=None, timestamp=DT), make_state(position, velocity, DT),
            prediction=make_prediction(np.stack([position, position + offset], axis=1),
                                       0.0, [0.0, DT]))
        self.assertTrue(bool(inline_out['updated'][0]), 'the inline prediction= idiom must work too')
        np.testing.assert_allclose(inline.residual_ledger()[0, -1], offset[0], atol=1e-12)

    def test_missing_prediction_does_not_update_and_is_visible_in_diagnostics(self) -> None:
        module = OnlineAdaptation(num_envs=2, estimate=('drag_scale',))
        self.assertFalse(np.any(module.diagnostics()['prediction_available']))
        position = np.zeros((2, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = orbit_velocity(0).repeat(2, axis=0)
        module.process(Feedback(prediction_error=None, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        out = module.process(Feedback(prediction_error=None, timestamp=DT),
                             make_state(position, velocity, DT))
        self.assertFalse(np.any(out['updated']))
        self.assertEqual(list(out['residual_source']), ['none', 'none'])
        self.assertTrue(np.all(np.isnan(out['residual'])))
        np.testing.assert_array_equal(module.residual_counts(), [0, 0])

        offset = np.array([0.06, 0.0, 0.0])
        module.set_prediction(make_prediction(
            np.stack([position, position + offset], axis=1), 0.0, [0.0, DT]))
        self.assertTrue(np.all(module.diagnostics()['prediction_available']))
        module.set_prediction(None)
        self.assertFalse(np.any(module.diagnostics()['prediction_available']),
                         'set_prediction(None) clears the cache')

    def test_explicit_prediction_error_takes_precedence(self) -> None:
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        mirror = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        for other in (module, mirror):
            other.process(Feedback(prediction_error=None, timestamp=0.0),
                          make_state(position, velocity, 0.0))
        big = np.array([[0.5, 0.0, 0.0]])            # what the pushed prediction would imply
        module.set_prediction(make_prediction(np.stack([position, position + big], axis=1),
                                              0.0, [0.0, DT]))
        exact = np.array([[0.01, 0.0, 0.0]])         # the late measurement that really arrived
        out = module.process(Feedback(prediction_error=exact, timestamp=DT),
                             make_state(position, velocity, DT))
        mirror_out = mirror.process(Feedback(prediction_error=exact, timestamp=DT),
                                    make_state(position, velocity, DT))
        self.assertTrue(bool(out['updated'][0]))
        self.assertEqual(out['residual_source'][0], 'feedback')
        np.testing.assert_allclose(module.residual_ledger()[0, -1], exact[0], atol=1e-12)
        self.assertEqual(float(module.drag_scale[0]), float(mirror.drag_scale[0]),
                         'the explicit residual must give exactly the prediction-free result')
        self.assertTrue(bool(mirror_out['updated'][0]))

    def test_same_step_push_falls_back_to_the_previous_prediction(self) -> None:
        module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))
        position = np.array([[0.0, 0.0, 3.0]])
        velocity = orbit_velocity(0)
        older = np.array([[0.06, 0.0, 0.0]])
        module.set_prediction(make_prediction(np.stack([position, position + older], axis=1),
                                              0.0, [0.0, DT]))
        module.process(Feedback(prediction_error=None, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        # the application pushes the prediction of THIS step as well: its grid starts at the current
        # instant, so it cannot be compared with the state of the same instant and must not be used
        newer = np.array([[-9.0, 0.0, 0.0]])
        module.set_prediction(make_prediction(np.stack([position, position + newer], axis=1),
                                              DT, [DT, 2 * DT]))
        out = module.process(Feedback(prediction_error=None, timestamp=DT),
                             make_state(position, velocity, DT))
        self.assertTrue(bool(out['updated'][0]))
        np.testing.assert_allclose(module.residual_ledger()[0, -1], older[0], atol=1e-12)

    def test_reset_clears_the_prediction_cache(self) -> None:
        module = OnlineAdaptation(num_envs=2, estimate=('drag_scale',))
        position = np.zeros((2, 3)) + np.array([0.0, 0.0, 3.0])
        velocity = orbit_velocity(0).repeat(2, axis=0)
        offset = np.array([0.06, 0.0, 0.0])
        module.process(Feedback(prediction_error=None, timestamp=0.0),
                       make_state(position, velocity, 0.0))
        module.set_prediction(make_prediction(np.stack([position, position + offset], axis=1),
                                              0.0, [0.0, DT]))
        self.assertTrue(np.all(module.diagnostics()['prediction_available']))

        module.reset([0])

        available = module.diagnostics()['prediction_available']
        self.assertFalse(bool(available[0]), 'reset must clear the prediction cache of env 0')
        self.assertTrue(bool(available[1]))
        out = module.process(Feedback(prediction_error=None, timestamp=DT),
                             make_state(position, velocity, DT))
        self.assertFalse(bool(out['updated'][0]))
        self.assertEqual(out['residual_source'][0], 'none')
        self.assertTrue(bool(out['updated'][1]))

        # a fresh push for both environments restores the loop for env 0 as well
        module.set_prediction(make_prediction(np.stack([position, position + offset], axis=1),
                                              DT, [DT, 2 * DT]))
        out2 = module.process(Feedback(prediction_error=None, timestamp=2 * DT),
                              make_state(position, velocity, 2 * DT))
        self.assertTrue(np.all(out2['updated']))
        self.assertTrue(np.all(out2['residual_source'] == 'prediction'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
