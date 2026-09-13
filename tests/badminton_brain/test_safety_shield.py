# -*- coding: utf-8 -*-
"""T8 Safety Shield tests: WholeBodyTarget -> SafeCommand (docs/architecture/06_SAFETY.md).

Acceptance (plan 2026-09-13-brain-modules.md, T8 row):
  * every limit has a "construct a violation -> clamped or rejected" test
  * e-stop and communication timeout emit a zero command
  * a scan over random targets never emits a command that violates a limit

All numeric limits used here are TEMP_PARAMETERIZED_PROXY placeholders: no PiPER /
Morph One limit has been measured in this repository, so SafetyLimits() itself
stays REQUIRES_MEASUREMENT with value=None.
"""
from __future__ import annotations
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.safety.safety_shield import (  # noqa: E402
    SafetyContext, SafetyLimits, SafetyShield,
)
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from badminton_brain.registry import ModuleRegistry  # noqa: E402
from badminton_brain.types import (  # noqa: E402
    BrainBoundaryError, Layer, SafeCommand, UnifiedState, WholeBodyTarget,
)

N = 3
DT = 0.02


def make_limits() -> SafetyLimits:
    """TEMP proxies only (see module docstring)."""
    return SafetyLimits.temp_proxy()


def quat_identity(n: int) -> np.ndarray:
    return np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))


def make_state(joint_pos=None, racket_pose=None, timestamp=0.0, n: int = N) -> UnifiedState:
    joint_pos = np.zeros((n, 6)) if joint_pos is None else np.asarray(joint_pos, dtype=float)
    return UnifiedState(base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)),
                        base_twist=np.zeros((n, 6)), joint_pos=joint_pos,
                        joint_vel=np.zeros((n, 6)),
                        racket_contact_pose=(quat_identity(n) if racket_pose is None
                                             else np.asarray(racket_pose, dtype=float)),
                        racket_contact_twist=np.zeros((n, 6)),
                        shuttle_position=np.zeros((n, 3)), shuttle_velocity=np.zeros((n, 3)),
                        timestamp=timestamp)


def make_target(base_twist=None, joints=None, timestamp=0.0, n: int = N) -> WholeBodyTarget:
    base_twist = np.zeros((n, 3)) if base_twist is None else np.asarray(base_twist, dtype=float)
    joints = np.zeros((n, 6)) if joints is None else np.asarray(joints, dtype=float)
    return WholeBodyTarget(base_twist=base_twist, joint_position_target=joints,
                           horizon_s=DT, timestamp=timestamp)


def box_center(limits: SafetyLimits) -> np.ndarray:
    lo = np.asarray(limits.racket_workspace_min.value, dtype=float)
    hi = np.asarray(limits.racket_workspace_max.value, dtype=float)
    return 0.5 * (lo + hi)


def pose_in_box(limits: SafetyLimits, n: int = N) -> np.ndarray:
    pose = np.tile(np.concatenate([box_center(limits), np.array([0.0, 0.0, 0.0, 1.0])]), (n, 1))
    return pose


class LimitHonestyTests(unittest.TestCase):
    """S4/S12: no limit may be invented; unknown limits stay REQUIRES_MEASUREMENT."""

    def test_default_limits_are_unmeasured_placeholders(self) -> None:
        limits = SafetyLimits()
        params = limits.as_dict()
        self.assertGreaterEqual(len(params), 8)
        for name, param in params.items():
            self.assertIsInstance(param, Param, msg=name)
            self.assertIs(param.status, AssetStatus.REQUIRES_MEASUREMENT, msg=name)
            self.assertIsNone(param.value, msg=name)
            self.assertTrue(param.source.strip(), msg=name)
        self.assertIn('joint_position_min', params)
        self.assertIn('joint_velocity_max', params)
        self.assertIn('base_twist_max', params)
        self.assertIn('racket_workspace_min', params)
        self.assertIn('command_timeout_s', params)

    def test_shield_refuses_to_run_with_unmeasured_limits(self) -> None:
        with self.assertRaises(BrainBoundaryError):
            SafetyShield(num_envs=N, limits=SafetyLimits())

    def test_temp_proxy_limits_are_explicitly_marked_and_warn(self) -> None:
        limits = SafetyLimits.temp_proxy()
        for name, param in limits.as_dict().items():
            self.assertIs(param.status, AssetStatus.TEMP_PARAMETERIZED_PROXY, msg=name)
            self.assertIsNotNone(param.value, msg=name)
            self.assertIn('TEMP', str(param.source), msg=name)
        self.assertTrue(limits.unresolved())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SafetyShield(num_envs=N)
        self.assertTrue(any('TEMP' in str(w.message) for w in caught))

    def test_limits_are_configurable_per_instance(self) -> None:
        limits = make_limits()
        limits.joint_velocity_max = Param(np.full(6, 0.05), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                          "TEMP: test-only tightened joint speed limit")
        shield = SafetyShield(num_envs=N, limits=limits)
        out = shield.process(make_target(joints=np.full((N, 6), 0.4)), state=make_state())
        self.assertTrue(out.limited)
        self.assertLessEqual(np.abs(out.joint_position_target).max(), 0.05 * DT + 1e-12)

    def test_shield_registers_as_the_safety_layer(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        registry = ModuleRegistry()
        registry.register(shield)
        self.assertIs(registry.get(Layer.SAFETY), shield)
        self.assertTrue(shield.is_implemented)


class PassThroughTests(unittest.TestCase):
    def test_in_limit_target_is_passed_through_unchanged(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        shield.process(make_target(), state=make_state())          # first step: adopt reference
        twist = np.tile(np.array([0.1, -0.1, 0.2]), (N, 1))
        joints = np.tile(np.array([0.1, 0.2, 0.3, -0.1, 0.05, 0.0]), (N, 1))
        out = shield.process(make_target(base_twist=twist, joints=joints, timestamp=DT),
                             state=make_state(), now=DT)
        self.assertIsInstance(out, SafeCommand)
        self.assertEqual(out.frame, 'court')
        self.assertEqual(out.base_twist.shape, (N, 3))
        self.assertEqual(out.joint_position_target.shape, (N, 6))
        self.assertFalse(out.limited)
        self.assertEqual(out.violations, ())
        np.testing.assert_allclose(out.base_twist, twist)
        np.testing.assert_allclose(out.joint_position_target, joints, atol=1e-12)


class JointLimitTests(unittest.TestCase):
    def test_joint_position_violation_is_clamped_and_reported(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        ref = np.zeros((N, 6))
        shield.process(make_target(), state=make_state(joint_pos=ref))
        lo = np.asarray(limits.joint_position_min.value, dtype=float)
        hi = np.asarray(limits.joint_position_max.value, dtype=float)
        joints = np.tile(hi + 1.0, (N, 1))
        joints[2] = lo - 1.0
        out = shield.process(make_target(joints=joints), state=make_state(joint_pos=ref))
        self.assertTrue(out.limited)
        self.assertTrue(np.all(out.joint_position_target <= hi + 1e-12))
        self.assertTrue(np.all(out.joint_position_target >= lo - 1e-12))
        self.assertTrue(any('joint_position_max' in v for v in out.violations))
        self.assertTrue(any('joint_position_min' in v for v in out.violations))
        self.assertTrue(any(v.startswith('env2:') for v in out.violations))
        self.assertTrue(all(v.startswith('env') for v in out.violations))

    def test_joint_velocity_violation_is_clamped_per_step(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        ref = np.zeros((N, 6))
        shield.process(make_target(), state=make_state(joint_pos=ref))
        out = shield.process(make_target(joints=np.full((N, 6), 5.0)),
                             state=make_state(joint_pos=ref))
        step = np.abs(out.joint_position_target - ref)
        self.assertTrue(out.limited)
        self.assertTrue(np.all(step <= dq * DT + 1e-12), msg=str(step.max()))
        self.assertTrue(any('joint_velocity_max' in v for v in out.violations))

    def test_measured_joint_outside_hard_limit_forces_zero_command(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        hi = np.asarray(limits.joint_position_max.value, dtype=float)
        ref = np.zeros((N, 6))
        ref[1, 4] = hi[4] + 0.3                        # joint feedback beyond the hard limit
        out = shield.process(make_target(joints=np.full((N, 6), 0.5)),
                             state=make_state(joint_pos=ref))
        self.assertTrue(out.limited)
        np.testing.assert_allclose(out.joint_position_target[1], np.zeros(6))
        np.testing.assert_allclose(out.base_twist[1], np.zeros(3))
        self.assertTrue(any('measured_joint_beyond_limit' in v for v in out.violations))
        self.assertFalse(any(v.startswith('env0:measured') for v in out.violations))

    def test_missing_velocity_reference_is_reported_once(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        first = shield.process(make_target(joints=np.full((N, 6), 0.3)))
        self.assertTrue(first.limited)
        self.assertTrue(any('no_velocity_reference' in v for v in first.violations))
        second = shield.process(make_target(joints=np.full((N, 6), 0.3), timestamp=DT), now=DT)
        self.assertFalse(second.limited)


class BaseTwistLimitTests(unittest.TestCase):
    def test_base_twist_violation_is_scaled_without_changing_direction(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        vmax = np.asarray(limits.base_twist_max.value, dtype=float)
        requested = np.tile(np.array([2.0, 1.0, 3.0]), (N, 1))
        out = shield.process(make_target(base_twist=requested), state=make_state())
        self.assertTrue(out.limited)
        self.assertTrue(any('base_twist_max' in v for v in out.violations))
        self.assertTrue(np.all(np.abs(out.base_twist) <= vmax + 1e-12))
        # direction preserving: same ratio between axes as the request
        for env in range(N):
            np.testing.assert_allclose(out.base_twist[env] / out.base_twist[env, 0],
                                       requested[env] / requested[env, 0], atol=1e-12)
        self.assertLessEqual(float(np.hypot(*out.base_twist[0, :2])),
                             float(np.hypot(*vmax[:2])) + 1e-12)


class RacketWorkspaceTests(unittest.TestCase):
    def test_measured_racket_pose_outside_box_is_rejected(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=make_limits())
        shield.process(make_target(), state=make_state())
        pose = pose_in_box(limits)
        pose[1] = box_center(limits) + np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        requested = np.tile(np.array([0.2, 0.0, 0.1]), (N, 1))
        out = shield.process(make_target(base_twist=requested, joints=np.full((N, 6), 0.2)),
                             state=make_state(joint_pos=np.zeros((N, 6)), racket_pose=pose))
        self.assertTrue(out.limited)
        self.assertTrue(any('racket_workspace' in v for v in out.violations))
        np.testing.assert_allclose(out.base_twist[1], np.zeros(3))      # rejected: no motion
        self.assertFalse(any(v.startswith('env0:racket') for v in out.violations))
        self.assertGreater(np.abs(out.joint_position_target[0]).max(), 0.0)

    def test_fk_hook_rejects_a_joint_command_that_leaves_the_box(self) -> None:
        limits = make_limits()
        center = box_center(limits)

        def test_double_fk(q: np.ndarray) -> np.ndarray:
            """TEST DOUBLE (not robot geometry): maps q[0] straight onto racket x."""
            pose = np.tile(np.concatenate([center, np.array([0.0, 0.0, 0.0, 1.0])]), (q.shape[0], 1))
            pose[:, 0] = center[0] + q[:, 0]
            return pose

        shield = SafetyShield(num_envs=N, limits=limits, forward_kinematics=test_double_fk)
        ref = np.zeros((N, 6))
        shield.process(make_target(), state=make_state(joint_pos=ref))
        inside = np.zeros((N, 6))
        inside[:, 0] = 0.05
        ok = shield.process(make_target(joints=inside), state=make_state(joint_pos=ref))
        self.assertFalse(any('racket_workspace' in v for v in ok.violations))
        outside = np.zeros((N, 6))
        outside[:, 0] = 1.0
        out = shield.process(make_target(joints=outside), state=make_state(joint_pos=ref))
        self.assertTrue(out.limited)
        self.assertTrue(any('racket_workspace' in v for v in out.violations))
        np.testing.assert_allclose(out.joint_position_target, ref)      # HOLD, not the request


class EmergencyTests(unittest.TestCase):
    def test_estop_is_per_env_and_zeroes_only_flagged_envs(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        estop = np.array([True, False, False])
        requested = np.tile(np.array([0.3, 0.0, 0.0]), (N, 1))
        joints = np.full((N, 6), 0.2)
        out = shield.process(make_target(base_twist=requested, joints=joints),
                             state=make_state(), context=SafetyContext(estop=estop))
        self.assertTrue(out.limited)
        np.testing.assert_allclose(out.base_twist[0], np.zeros(3))
        np.testing.assert_allclose(out.joint_position_target[0], np.zeros(6))
        self.assertFalse(any(v.startswith('env0:if') for v in out.violations))
        self.assertTrue(any('estop' in v for v in out.violations))
        self.assertFalse(any(v.startswith('env1:estop') for v in out.violations))
        np.testing.assert_allclose(out.base_twist[2], requested[2])

    def test_estop_latches_until_reset_and_reset_is_per_env(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        shield.process(make_target(joints=np.full((N, 6), 0.2)), state=make_state(),
                       context=SafetyContext(estop=np.array([True, False, False])))
        after = shield.process(make_target(joints=np.full((N, 6), 0.2), timestamp=DT),
                               state=make_state(), now=DT,
                               context=SafetyContext(estop=np.array([False, False, False])))
        np.testing.assert_allclose(after.joint_position_target[0], np.zeros(6))
        self.assertTrue(any(v.startswith('env0:estop') for v in after.violations))

        shield.reset([1])                                  # must not clear env 0
        still = shield.process(make_target(joints=np.full((N, 6), 0.2), timestamp=DT),
                               state=make_state(), now=DT)
        np.testing.assert_allclose(still.joint_position_target[0], np.zeros(6))
        self.assertTrue(any(v.startswith('env0:estop') for v in still.violations))

        shield.reset([0])
        cleared = shield.process(make_target(joints=np.full((N, 6), 0.2), timestamp=DT),
                                 state=make_state(), now=DT)
        self.assertFalse(any('estop' in v for v in cleared.violations))
        np.testing.assert_allclose(cleared.joint_position_target[1], np.full(6, 0.2))


class TimeoutTests(unittest.TestCase):
    def test_stale_command_emits_zero_command(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        timeout = float(limits.command_timeout_s.value)
        requested = np.tile(np.array([0.3, 0.0, 0.0]), (N, 1))
        out = shield.process(make_target(base_twist=requested, joints=np.full((N, 6), 0.2)),
                             state=make_state(), now=timeout + 0.5)
        self.assertTrue(out.limited)
        self.assertTrue(any('communication_timeout' in v for v in out.violations))
        np.testing.assert_allclose(out.base_twist, np.zeros((N, 3)))
        np.testing.assert_allclose(out.joint_position_target, np.zeros((N, 6)))

    def test_external_watchdog_timeout_is_per_env(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        out = shield.process(make_target(joints=np.full((N, 6), 0.1)), state=make_state(),
                             context=SafetyContext(timeouts=np.array([False, True, False])))
        np.testing.assert_allclose(out.joint_position_target[1], np.zeros(6))
        self.assertTrue(any(v.startswith('env1:communication_timeout') for v in out.violations))
        self.assertFalse(any(v.startswith('env0:communication_timeout') for v in out.violations))

    def test_future_timestamp_beyond_clock_tolerance_is_rejected(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        tolerance = float(limits.clock_tolerance_s.value)
        out = shield.process(make_target(joints=np.full((N, 6), 0.1), timestamp=tolerance + 1.0),
                             state=make_state(), now=0.0)
        self.assertTrue(out.limited)
        self.assertTrue(any('command_timestamp_future' in v for v in out.violations))
        np.testing.assert_allclose(out.joint_position_target, np.zeros((N, 6)))


class RobustnessTests(unittest.TestCase):
    def test_non_finite_target_is_zeroed_instead_of_raised(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        target = make_target(joints=np.full((N, 6), 0.1))       # bypass __post_init__ on purpose
        target.joint_position_target = np.full((N, 6), 0.1)
        target.joint_position_target[1, 3] = np.nan
        out = shield.process(target, state=make_state())
        self.assertTrue(out.limited)
        self.assertTrue(any('invalid_non_finite' in v for v in out.violations))
        np.testing.assert_allclose(out.joint_position_target[1], np.zeros(6))
        self.assertFalse(any(v.startswith('env0:invalid') for v in out.violations))

    def test_wrong_message_type_and_shape_are_refused(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        with self.assertRaises(BrainBoundaryError):
            shield.process(make_state())
        target = make_target()
        target.base_twist = np.zeros((N, 4))
        with self.assertRaises(BrainBoundaryError):
            shield.process(target)

    def test_env_count_mismatch_is_refused(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        with self.assertRaises(BrainBoundaryError):
            shield.process(make_target(n=N + 1), state=make_state(n=N + 1))


class ScanTests(unittest.TestCase):
    def test_scan_of_random_targets_never_emits_a_limiting_violation(self) -> None:
        limits = make_limits()
        shield = SafetyShield(num_envs=N, limits=limits)
        rng = np.random.default_rng(0)
        lo = np.asarray(limits.joint_position_min.value, dtype=float)
        hi = np.asarray(limits.joint_position_max.value, dtype=float)
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        vmax = np.asarray(limits.base_twist_max.value, dtype=float)
        ref = 0.5 * (lo + hi) * rng.uniform(-0.5, 0.5, size=(N, 6))
        pose = pose_in_box(limits)
        for step in range(200):
            joints = rng.uniform(-20.0, 20.0, size=(N, 6))
            twist = rng.uniform(-5.0, 5.0, size=(N, 3))
            estop = rng.random(N) < 0.1
            target = make_target(base_twist=twist, joints=joints, timestamp=step * DT)
            out = shield.process(target, state=make_state(joint_pos=ref, racket_pose=pose),
                                 now=step * DT, context=SafetyContext(estop=estop))
            q = np.asarray(out.joint_position_target)
            v = np.asarray(out.base_twist)
            self.assertTrue(np.all(q <= hi + 1e-12) and np.all(q >= lo - 1e-12),
                            msg='joint position limit violated at step %d' % step)
            self.assertTrue(np.all(np.abs(q - ref) <= dq * DT + 1e-12),
                            msg='joint velocity limit violated at step %d' % step)
            self.assertTrue(np.all(np.abs(v) <= vmax + 1e-12),
                            msg='base twist limit violated at step %d' % step)
            self.assertTrue(np.all(np.isfinite(q)) and np.all(np.isfinite(v)))
            if estop.any():
                self.assertTrue(out.limited)
                for env in np.flatnonzero(estop):
                    np.testing.assert_allclose(q[env], np.zeros(6))
                    np.testing.assert_allclose(v[env], np.zeros(3))
            applied = np.asarray(out.base_twist)
            if not out.limited:
                np.testing.assert_allclose(applied, twist, atol=1e-12)
                np.testing.assert_allclose(q, joints, atol=1e-12)
            ref = q

    def test_violations_are_deterministic_and_sorted(self) -> None:
        shield = SafetyShield(num_envs=N, limits=make_limits())
        joints = np.tile(np.array([9.0, -9.0, 9.0, -9.0, 9.0, -9.0]), (N, 1))
        out = shield.process(make_target(base_twist=np.tile(np.array([9.0, 9.0, 9.0]), (N, 1)),
                                         joints=joints), state=make_state())
        first = out.violations
        shield.reset(None)
        out2 = shield.process(make_target(base_twist=np.tile(np.array([9.0, 9.0, 9.0]), (N, 1)),
                                          joints=joints), state=make_state())
        self.assertEqual(first, out2.violations)
        self.assertEqual(list(first), sorted(first))
        self.assertEqual(len(set(first)), len(first))


if __name__ == '__main__':
    unittest.main()
