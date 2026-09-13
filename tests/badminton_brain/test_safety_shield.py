# -*- coding: utf-8 -*-
"""T8 Safety Shield tests: WholeBodyTarget -> SafeCommand (docs/architecture/06_SAFETY.md).

Acceptance (plan 2026-09-13-brain-modules.md, T8 row):
  * every limit has a "construct a violation -> clamped or rejected" test
  * e-stop and communication timeout emit a zero command
  * a scan over random targets never emits a command that violates a limit
Plus the coordinator review rulings: set_context() push API (HIGH), feasible zero-command
fallback, FK-verified workspace HOLD, no-jump reference after reset, two named base speed limits.

Every numeric limit used here is a TEMP_PARAMETERIZED_PROXY placeholder: no PiPER /
Morph One / court-workspace limit has been measured in this repository, so
SafetyLimits() itself stays REQUIRES_MEASUREMENT with value=None.
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
TEMP = AssetStatus.TEMP_PARAMETERIZED_PROXY


def make_limits() -> SafetyLimits:
    """TEMP proxies only (see module docstring)."""
    return SafetyLimits.temp_proxy()


def joint_bounds(limits: SafetyLimits):
    return (np.asarray(limits.joint_position_min.value, dtype=float),
            np.asarray(limits.joint_position_max.value, dtype=float))


def box_bounds(limits: SafetyLimits):
    return (np.asarray(limits.racket_workspace_min.value, dtype=float),
            np.asarray(limits.racket_workspace_max.value, dtype=float))


def box_center(limits: SafetyLimits) -> np.ndarray:
    lo, hi = box_bounds(limits)
    return 0.5 * (lo + hi)


def pose_in_box(limits: SafetyLimits, n: int = N) -> np.ndarray:
    """(n, 7) racket pose at the centre of the TEMP racket workspace box."""
    return np.tile(np.concatenate([box_center(limits), np.array([0.0, 0.0, 0.0, 1.0])]), (n, 1))


def limits_with(joint_min: float, joint_max: float) -> SafetyLimits:
    """TEMP limits with an asymmetric joint range (0 must not be inside it)."""
    limits = make_limits()
    limits.joint_position_min = Param(np.full(6, joint_min), TEMP,
                                      "TEMP: test-only asymmetric lower joint limit")
    limits.joint_position_max = Param(np.full(6, joint_max), TEMP,
                                      "TEMP: test-only asymmetric upper joint limit")
    return limits


def make_shield(limits: SafetyLimits = None, n: int = N, **kwargs) -> SafetyShield:
    return SafetyShield(num_envs=n, limits=limits if limits is not None else make_limits(), **kwargs)


def run(shield: SafetyShield, target: WholeBodyTarget, **kwargs):
    """Process with an explicit benign context unless the test supplies its own."""
    kwargs.setdefault('context', SafetyContext())
    return shield.process(target, **kwargs)


def make_state(joint_pos=None, racket_pose=None, timestamp=0.0, n: int = N) -> UnifiedState:
    joint_pos = np.zeros((n, 6)) if joint_pos is None else np.asarray(joint_pos, dtype=float)
    if racket_pose is None:
        racket_pose = pose_in_box(make_limits(), n)          # inside the TEMP workspace box
    return UnifiedState(base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)),
                        base_twist=np.zeros((n, 6)), joint_pos=joint_pos,
                        joint_vel=np.zeros((n, 6)),
                        racket_contact_pose=np.asarray(racket_pose, dtype=float),
                        racket_contact_twist=np.zeros((n, 6)),
                        shuttle_position=np.zeros((n, 3)), shuttle_velocity=np.zeros((n, 3)),
                        timestamp=timestamp)


def make_target(base_twist=None, joints=None, timestamp=0.0, n: int = N) -> WholeBodyTarget:
    base_twist = np.zeros((n, 3)) if base_twist is None else np.asarray(base_twist, dtype=float)
    joints = np.zeros((n, 6)) if joints is None else np.asarray(joints, dtype=float)
    return WholeBodyTarget(base_twist=base_twist, joint_position_target=joints,
                           horizon_s=DT, timestamp=timestamp)


def codes_for(out: SafeCommand, env: int) -> str:
    prefix = 'env%d:' % env
    return ' '.join(v for v in out.violations if v.startswith(prefix))


class LimitHonestyTests(unittest.TestCase):
    """S4/S12: no limit may be invented; unknown limits stay REQUIRES_MEASUREMENT."""

    def test_default_limits_are_unmeasured_placeholders(self) -> None:
        params = SafetyLimits().as_dict()
        self.assertGreaterEqual(len(params), 8)
        for name, param in params.items():
            self.assertIsInstance(param, Param, msg=name)
            self.assertIs(param.status, AssetStatus.REQUIRES_MEASUREMENT, msg=name)
            self.assertIsNone(param.value, msg=name)
            self.assertTrue(param.source.strip(), msg=name)
        for required in ('joint_position_min', 'joint_position_max', 'joint_velocity_max',
                         'base_twist_axis_max', 'base_translation_speed_max',
                         'racket_workspace_min', 'racket_workspace_max',
                         'control_dt', 'command_timeout_s'):
            self.assertIn(required, params)
        self.assertNotIn('base_twist_max', params)          # review finding 4: split by semantics

    def test_shield_refuses_to_run_with_unmeasured_limits(self) -> None:
        with self.assertRaises(BrainBoundaryError):
            SafetyShield(num_envs=N, limits=SafetyLimits())

    def test_temp_proxy_limits_are_explicitly_marked_and_warn(self) -> None:
        limits = SafetyLimits.temp_proxy()
        for name, param in limits.as_dict().items():
            self.assertIs(param.status, TEMP, msg=name)
            self.assertIsNotNone(param.value, msg=name)
            self.assertIn('TEMP', str(param.source), msg=name)
        self.assertTrue(limits.unresolved())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SafetyShield(num_envs=N)
        self.assertTrue(any('TEMP' in str(w.message) for w in caught))

    def test_limits_are_configurable_per_instance(self) -> None:
        limits = make_limits()
        limits.joint_velocity_max = Param(np.full(6, 0.05), TEMP,
                                          "TEMP: test-only tightened joint speed limit")
        shield = make_shield(limits)
        out = run(shield, make_target(joints=np.full((N, 6), 0.4)), state=make_state())
        self.assertTrue(out.limited)
        self.assertLessEqual(np.abs(out.joint_position_target).max(), 0.05 * DT + 1e-12)

    def test_shield_registers_as_the_safety_layer(self) -> None:
        shield = make_shield()
        registry = ModuleRegistry()
        registry.register(shield)
        self.assertIs(registry.get(Layer.SAFETY), shield)
        self.assertTrue(shield.is_implemented)
        self.assertIs(shield.output_type, SafeCommand)


class ContextApiTests(unittest.TestCase):
    """HIGH ruling: the application pushes SafetyContext; nothing may be silently assumed."""

    def test_pushed_estop_really_takes_effect(self) -> None:
        shield = make_shield()
        shield.set_context(SafetyContext(estop=np.array([True, False, False])))
        requested = np.tile(np.array([0.3, 0.0, 0.0]), (N, 1))
        out = shield.process(make_target(base_twist=requested, joints=np.full((N, 6), 0.2)),
                             state=make_state())          # no inline context: push must be used
        self.assertTrue(out.limited)
        np.testing.assert_allclose(out.base_twist[0], np.zeros(3))
        np.testing.assert_allclose(out.joint_position_target[0], np.zeros(6))
        self.assertIn('estop', codes_for(out, 0))
        self.assertNotIn('estop', codes_for(out, 1))
        np.testing.assert_allclose(out.base_twist[2], requested[2])
        snapshot = shield.context_snapshot()
        self.assertEqual(snapshot['source'], 'pushed')
        self.assertEqual(tuple(snapshot['estop_envs']), (0,))
        self.assertEqual(tuple(snapshot['without_context_envs']), ())

    def test_pushed_now_drives_the_watchdog(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        timeout = float(limits.command_timeout_s.value)
        shield.set_context(SafetyContext(now=timeout + 0.5, estop=np.zeros(N)))
        out = shield.process(make_target(joints=np.full((N, 6), 0.2)))   # no inline now
        self.assertTrue(out.limited)
        self.assertTrue(all('communication_timeout' in codes_for(out, e) for e in range(N)))
        np.testing.assert_allclose(out.joint_position_target, np.zeros((N, 6)))
        self.assertEqual(shield.context_snapshot()['now'], timeout + 0.5)

    def test_pushed_context_supplies_the_measured_state(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        ref = np.zeros((N, 6))
        pose = pose_in_box(limits)
        pose[2, 0] = box_center(limits)[0] + 5.0             # env 2 measured outside the box
        shield.set_context(SafetyContext(joint_position=ref, racket_contact_pose=pose))
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        out = shield.process(make_target(joints=np.full((N, 6), 5.0)))
        self.assertTrue(np.all(np.abs(out.joint_position_target - ref) <= dq * DT + 1e-12),
                        msg='pushed joint_position was not used as the velocity reference')
        self.assertIn('racket_workspace', codes_for(out, 2))
        self.assertNotIn('racket_workspace', codes_for(out, 0))
        snapshot = shield.context_snapshot()
        self.assertEqual(tuple(snapshot['measured_joint_envs']), (0, 1, 2))
        self.assertEqual(tuple(snapshot['measured_racket_envs']), (0, 1, 2))

    def test_inline_context_wins_over_the_pushed_one(self) -> None:
        shield = make_shield()
        shield.set_context(SafetyContext(estop=np.array([True, True, True])))
        out = shield.process(make_target(joints=np.full((N, 6), 0.2)), state=make_state(),
                             context=SafetyContext(estop=np.zeros(N)))
        self.assertNotIn('estop', ' '.join(out.violations))
        self.assertGreater(np.abs(out.joint_position_target).max(), 0.0)
        self.assertEqual(shield.context_snapshot()['source'], 'argument')

    def test_missing_context_is_marked_not_silently_normal(self) -> None:
        shield = make_shield()
        out = shield.process(make_target(joints=np.full((N, 6), 0.2)), state=make_state())
        self.assertTrue(out.limited)
        for env in range(N):
            self.assertIn('no_context', codes_for(out, env))
        self.assertGreater(np.abs(out.joint_position_target).max(), 0.0)   # no info != stop
        snapshot = shield.context_snapshot()
        self.assertEqual(snapshot['source'], 'none')
        self.assertEqual(tuple(snapshot['without_context_envs']), (0, 1, 2))

    def test_require_context_false_keeps_the_diagnostic_only(self) -> None:
        shield = make_shield(require_context=False)
        out = shield.process(make_target(joints=np.full((N, 6), 0.2)), state=make_state())
        self.assertNotIn('no_context', ' '.join(out.violations))
        self.assertEqual(tuple(shield.context_snapshot()['without_context_envs']), (0, 1, 2))

    def test_reset_clears_only_the_selected_envs_latch_and_context(self) -> None:
        shield = make_shield()
        shield.set_context(SafetyContext(estop=np.array([True, True, False])))
        first = shield.process(make_target(joints=np.full((N, 6), 0.2)), state=make_state())
        np.testing.assert_allclose(first.joint_position_target[0], np.zeros(6))
        np.testing.assert_allclose(first.joint_position_target[1], np.zeros(6))

        shield.reset([0])                                  # only env 0 latch + context
        after = shield.process(make_target(joints=np.full((N, 6), 0.2)), state=make_state())
        self.assertNotIn('estop', codes_for(after, 0))
        self.assertIn('estop', codes_for(after, 1))         # env 1 still latched
        np.testing.assert_allclose(after.joint_position_target[1], np.zeros(6))
        self.assertIn('no_context', codes_for(after, 0))    # its context was dropped
        self.assertEqual(shield.context_snapshot()['source'], 'pushed')


class PassThroughTests(unittest.TestCase):
    def test_in_limit_target_is_passed_through_unchanged(self) -> None:
        shield = make_shield()
        twist = np.tile(np.array([0.1, -0.1, 0.2]), (N, 1))
        joints = np.tile(np.array([0.1, 0.2, 0.3, -0.1, 0.05, 0.0]), (N, 1))
        out = run(shield, make_target(base_twist=twist, joints=joints, timestamp=DT),
                  state=make_state(joint_pos=joints), now=DT)
        self.assertIsInstance(out, SafeCommand)
        self.assertEqual(out.frame, 'court')
        self.assertEqual(out.base_twist.shape, (N, 3))
        self.assertEqual(out.joint_position_target.shape, (N, 6))
        self.assertFalse(out.limited)
        self.assertEqual(out.violations, ())
        np.testing.assert_allclose(out.base_twist, twist)
        np.testing.assert_allclose(out.joint_position_target, joints, atol=1e-12)
        self.assertEqual(out.timestamp, DT)


class JointLimitTests(unittest.TestCase):
    def test_joint_position_violation_is_clamped_and_reported(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        lo, hi = joint_bounds(limits)
        ref = np.zeros((N, 6))
        ref[0] = hi                       # already at the upper soft limit
        ref[1] = hi
        ref[2] = lo                       # already at the lower soft limit
        joints = np.tile(hi + 1.0, (N, 1))
        joints[2] = lo - 1.0
        out = run(shield, make_target(joints=joints), state=make_state(joint_pos=ref))
        self.assertTrue(out.limited)
        self.assertTrue(np.all(out.joint_position_target <= hi + 1e-12))
        self.assertTrue(np.all(out.joint_position_target >= lo - 1e-12))
        self.assertTrue(any('joint_position_max' in v for v in out.violations))
        self.assertTrue(any('joint_position_min' in v for v in out.violations))
        self.assertTrue(any(v.startswith('env2:') for v in out.violations))
        self.assertTrue(all(v.startswith('env') for v in out.violations))

    def test_joint_velocity_violation_is_clamped_per_step(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        ref = np.zeros((N, 6))
        out = run(shield, make_target(joints=np.full((N, 6), 5.0)), state=make_state(joint_pos=ref))
        step = np.abs(out.joint_position_target - ref)
        self.assertTrue(out.limited)
        self.assertTrue(np.all(step <= dq * DT + 1e-12), msg=str(step.max()))
        self.assertTrue(any('joint_velocity_max' in v for v in out.violations))
        np.testing.assert_allclose(out.joint_position_target, np.tile(dq * DT, (N, 1)), atol=1e-12)

    def test_measured_joint_outside_hard_limit_forces_zero_command(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        hi = np.asarray(limits.joint_position_max.value, dtype=float)
        ref = np.zeros((N, 6))
        ref[1, 4] = hi[4] + 0.3                        # joint feedback beyond the hard limit
        out = run(shield, make_target(joints=np.full((N, 6), 0.5)), state=make_state(joint_pos=ref))
        self.assertTrue(out.limited)
        np.testing.assert_allclose(out.joint_position_target[1], np.zeros(6))
        np.testing.assert_allclose(out.base_twist[1], np.zeros(3))
        self.assertTrue(any('measured_joint_beyond_limit' in v for v in out.violations))
        self.assertFalse(any(v.startswith('env0:measured') for v in out.violations))
        self.assertGreater(np.abs(out.joint_position_target[0]).max(), 0.0)

    def test_zero_command_fallback_respects_asymmetric_joint_limits(self) -> None:
        """Review finding 1: a hardcoded 0.0 fallback can leave [lo, hi]."""
        limits = limits_with(0.5, 1.0)
        shield = make_shield(limits)
        ref = np.full((N, 6), 0.75)
        out = run(shield, make_target(joints=np.full((N, 6), 2.0)), state=make_state(joint_pos=ref),
                  context=SafetyContext(estop=np.array([True, False, False])))
        self.assertTrue(np.all(out.joint_position_target >= 0.5 - 1e-12))
        self.assertTrue(np.all(out.joint_position_target <= 1.0 + 1e-12))
        np.testing.assert_allclose(out.joint_position_target[0], np.full(6, 0.5))
        np.testing.assert_allclose(out.base_twist[0], np.zeros(3))
        self.assertGreater(np.abs(out.joint_position_target[1]).max(), 0.0)

    def test_missing_velocity_reference_uses_the_limit_box_centre(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        lo, hi = joint_bounds(limits)
        centre = 0.5 * (lo + hi)
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        first = run(shield, make_target(joints=centre + 5.0))
        self.assertTrue(first.limited)
        self.assertTrue(any('no_velocity_reference' in v for v in first.violations))
        np.testing.assert_allclose(first.joint_position_target, np.tile(centre + dq * DT, (N, 1)),
                                   atol=1e-12)
        second = run(shield, make_target(joints=centre + dq * DT, timestamp=DT), now=DT)
        self.assertFalse(second.limited)
        self.assertEqual(second.violations, ())

    def test_reference_after_reset_is_the_box_centre_not_a_jump(self) -> None:
        """Review finding 3: after reset() the first command must not jump anywhere."""
        limits = make_limits()
        shield = make_shield(limits)
        lo, hi = joint_bounds(limits)
        centre = 0.5 * (lo + hi)
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        run(shield, make_target(joints=np.full((N, 6), centre + 1.0)))
        before = run(shield, make_target(joints=np.full((N, 6), centre + 1.0), timestamp=DT), now=DT)
        shield.reset([0])
        after = run(shield, make_target(joints=np.full((N, 6), centre + 1.0), timestamp=2 * DT),
                    now=2 * DT)
        # env 0 restarts from the limit box centre, env 1 continues from its last safe command
        np.testing.assert_allclose(after.joint_position_target[0],
                                   np.tile(centre + dq * DT, (N, 1))[0], atol=1e-12)
        np.testing.assert_allclose(after.joint_position_target[1],
                                   before.joint_position_target[1] + dq * DT, atol=1e-12)


class BaseTwistLimitTests(unittest.TestCase):
    def test_axis_and_combined_speed_limits_are_separate_and_enforced(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        vmax = np.asarray(limits.base_twist_axis_max.value, dtype=float)
        v_speed = float(limits.base_translation_speed_max.value)
        requested = np.tile(np.array([2.0, 1.0, 3.0]), (N, 1))
        out = run(shield, make_target(base_twist=requested), state=make_state())
        self.assertTrue(out.limited)
        self.assertTrue(any('base_twist_axis_max' in v for v in out.violations))
        self.assertTrue(np.all(np.abs(out.base_twist) <= vmax + 1e-12))
        for env in range(N):
            np.testing.assert_allclose(out.base_twist[env] / out.base_twist[env, 0],
                                       requested[env] / requested[env, 0], atol=1e-12)
        self.assertTrue(np.all(np.hypot(out.base_twist[:, 0], out.base_twist[:, 1]) <=
                               v_speed + 1e-12))

    def test_axis_limit_binds_alone(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        vmax = np.asarray(limits.base_twist_axis_max.value, dtype=float)
        out = run(shield, make_target(base_twist=np.tile(np.array([0.0, 0.0, 9.0]), (N, 1))),
                  state=make_state())
        self.assertTrue(out.limited)
        self.assertIn('base_twist_axis_max', ' '.join(out.violations))
        self.assertNotIn('base_translation_speed_max', ' '.join(out.violations))
        self.assertAlmostEqual(float(np.abs(out.base_twist[:, 2]).max()), float(vmax[2]), places=12)
        np.testing.assert_allclose(out.base_twist[:, :2], np.zeros((N, 2)))

    def test_combined_translation_limit_binds_alone(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        vmax = np.asarray(limits.base_twist_axis_max.value, dtype=float)
        v_speed = float(limits.base_translation_speed_max.value)
        requested = np.tile(np.array([vmax[0], vmax[1], 0.0]), (N, 1))  # each axis legal, norm is not
        self.assertGreater(float(np.hypot(requested[0, 0], requested[0, 1])), v_speed)
        out = run(shield, make_target(base_twist=requested), state=make_state())
        self.assertTrue(out.limited)
        self.assertIn('base_translation_speed_max', ' '.join(out.violations))
        self.assertNotIn('base_twist_axis_max', ' '.join(out.violations))
        np.testing.assert_allclose(np.hypot(out.base_twist[:, 0], out.base_twist[:, 1]),
                                   np.full(N, v_speed), atol=1e-12)
        np.testing.assert_allclose(out.base_twist[0, 0] / out.base_twist[0, 1],
                                   requested[0, 0] / requested[0, 1], atol=1e-12)


class RacketWorkspaceTests(unittest.TestCase):
    def test_measured_racket_pose_outside_box_is_rejected(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        pose = pose_in_box(limits)                     # every env starts inside the box
        pose[1, 0] = box_center(limits)[0] + 5.0       # env 1 is outside the box
        ref = np.zeros((N, 6))
        requested = np.tile(np.array([0.2, 0.0, 0.1]), (N, 1))
        out = run(shield, make_target(base_twist=requested, joints=np.full((N, 6), 0.2)),
                  state=make_state(joint_pos=ref, racket_pose=pose))
        self.assertTrue(out.limited)
        self.assertIn('racket_workspace', codes_for(out, 1))
        np.testing.assert_allclose(out.base_twist[1], np.zeros(3))      # rejected: no motion
        self.assertNotIn('racket_workspace', codes_for(out, 0))
        self.assertGreater(np.abs(out.joint_position_target[0]).max(), 0.0)

    def test_fk_hook_rejects_a_joint_command_that_leaves_the_box(self) -> None:
        limits = make_limits()
        center = box_center(limits)
        lo, hi = box_bounds(limits)
        gain = 10.0

        def test_double_fk(q: np.ndarray) -> np.ndarray:
            """TEST DOUBLE with an arbitrary gain (NOT robot geometry): racket x = centre_x + gain*q0."""
            pose = pose_in_box(limits, q.shape[0])
            pose[:, 0] = center[0] + gain * q[:, 0]
            return pose

        shield = make_shield(limits, forward_kinematics=test_double_fk)
        inside_ref = np.zeros((N, 6))
        inside = np.zeros((N, 6))
        inside[:, 0] = 0.05
        ok = run(shield, make_target(joints=inside), state=make_state(joint_pos=inside_ref))
        self.assertNotIn('racket_workspace', ' '.join(ok.violations))
        self.assertTrue(ok.limited)                       # joint speed limit still applies

        edge_ref = np.zeros((N, 6))
        edge_ref[:, 0] = (hi[0] - center[0]) / gain      # exactly on the box face
        out = run(shield, make_target(joints=np.full((N, 6), 1.0)),
                  state=make_state(joint_pos=edge_ref))
        self.assertIn('racket_workspace', ' '.join(out.violations))
        np.testing.assert_allclose(out.base_twist, np.zeros((N, 3)))
        emitted = test_double_fk(np.asarray(out.joint_position_target))[:, 0]
        self.assertTrue(np.all(emitted >= lo[0] - 1e-12) and np.all(emitted <= hi[0] + 1e-12),
                        msg='HOLD emitted a racket pose outside the box: %s' % emitted)

    def test_workspace_hold_is_never_out_of_box_in_a_scan(self) -> None:
        """Review finding 2: the HOLD target itself must be FK-verified."""
        limits = make_limits()
        center = box_center(limits)
        lo, hi = box_bounds(limits)
        gain = 10.0

        def test_double_fk(q: np.ndarray) -> np.ndarray:
            pose = pose_in_box(limits, q.shape[0])
            pose[:, 0] = center[0] + gain * q[:, 0]
            return pose

        shield = make_shield(limits, forward_kinematics=test_double_fk)
        rng = np.random.default_rng(1)
        for step in range(60):
            ref = rng.uniform(-1.3, 1.3, size=(N, 6))
            target = make_target(joints=rng.uniform(-1.3, 1.3, size=(N, 6)),
                                 base_twist=rng.uniform(-2.0, 2.0, size=(N, 3)),
                                 timestamp=step * DT)
            out = run(shield, target, state=make_state(joint_pos=ref), now=step * DT)
            poses = test_double_fk(np.asarray(out.joint_position_target))[:, :3]
            inside = ((poses >= lo - 1e-12) & (poses <= hi + 1e-12)).all(axis=1)
            for env in range(N):
                self.assertTrue(inside[env],
                                msg='step %d env %d emitted out-of-box racket pose %s (%s)'
                                    % (step, env, poses[env], codes_for(out, env)))
            self.assertNotIn('racket_workspace_unreachable', ' '.join(out.violations))

    def test_unreachable_workspace_is_reported_loudly(self) -> None:
        limits = make_limits()
        center = box_center(limits)

        def hopeless_fk(q: np.ndarray) -> np.ndarray:
            """TEST DOUBLE: no joint value maps into the box (configuration error scenario)."""
            pose = pose_in_box(limits, q.shape[0])
            pose[:, 0] = center[0] + 50.0 + 10.0 * q[:, 0]
            return pose

        shield = make_shield(limits, forward_kinematics=hopeless_fk)
        out = run(shield, make_target(joints=np.full((N, 6), 0.2)), state=make_state())
        self.assertTrue(out.limited)
        for env in range(N):
            self.assertIn('racket_workspace_unreachable', codes_for(out, env))
        np.testing.assert_allclose(out.base_twist, np.zeros((N, 3)))


class EmergencyTests(unittest.TestCase):
    def test_estop_is_per_env_and_zeroes_only_flagged_envs(self) -> None:
        shield = make_shield()
        estop = np.array([True, False, False])
        requested = np.tile(np.array([0.3, 0.0, 0.0]), (N, 1))
        joints = np.full((N, 6), 0.2)
        out = run(shield, make_target(base_twist=requested, joints=joints),
                  state=make_state(joint_pos=joints), context=SafetyContext(estop=estop))
        self.assertTrue(out.limited)
        np.testing.assert_allclose(out.base_twist[0], np.zeros(3))
        np.testing.assert_allclose(out.joint_position_target[0], np.zeros(6))
        self.assertIn('estop', codes_for(out, 0))
        self.assertNotIn('estop', codes_for(out, 1))
        self.assertNotIn('estop', codes_for(out, 2))
        np.testing.assert_allclose(out.base_twist[2], requested[2])

    def test_estop_latches_until_reset_and_reset_is_per_env(self) -> None:
        shield = make_shield()
        run(shield, make_target(joints=np.full((N, 6), 0.2)), state=make_state(),
            context=SafetyContext(estop=np.array([True, False, False])))
        after = run(shield, make_target(joints=np.full((N, 6), 0.2), timestamp=DT),
                    state=make_state(), now=DT, context=SafetyContext(estop=np.zeros(N)))
        np.testing.assert_allclose(after.joint_position_target[0], np.zeros(6))
        self.assertIn('estop', codes_for(after, 0))

        shield.reset([1])                                  # must not clear env 0
        still = run(shield, make_target(joints=np.full((N, 6), 0.2), timestamp=DT),
                    state=make_state(), now=DT)
        np.testing.assert_allclose(still.joint_position_target[0], np.zeros(6))
        self.assertIn('estop', codes_for(still, 0))

        shield.reset([0])
        cleared = run(shield, make_target(joints=np.full((N, 6), 0.2), timestamp=DT),
                      state=make_state(), now=DT)
        self.assertNotIn('estop', ' '.join(cleared.violations))
        self.assertGreater(np.abs(cleared.joint_position_target[1]).max(), 0.0)


class TimeoutTests(unittest.TestCase):
    def test_stale_command_emits_zero_command(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        timeout = float(limits.command_timeout_s.value)
        requested = np.tile(np.array([0.3, 0.0, 0.0]), (N, 1))
        out = run(shield, make_target(base_twist=requested, joints=np.full((N, 6), 0.2)),
                  state=make_state(), now=timeout + 0.5)
        self.assertTrue(out.limited)
        self.assertTrue(any('communication_timeout' in v for v in out.violations))
        np.testing.assert_allclose(out.base_twist, np.zeros((N, 3)))
        np.testing.assert_allclose(out.joint_position_target, np.zeros((N, 6)))

    def test_fresh_command_is_not_a_timeout(self) -> None:
        shield = make_shield()
        out = run(shield, make_target(timestamp=DT), state=make_state(), now=DT)
        self.assertNotIn('communication_timeout', ' '.join(out.violations))

    def test_external_watchdog_timeout_is_per_env(self) -> None:
        shield = make_shield()
        out = run(shield, make_target(joints=np.full((N, 6), 0.1)), state=make_state(),
                  context=SafetyContext(timeouts=np.array([False, True, False])))
        np.testing.assert_allclose(out.joint_position_target[1], np.zeros(6))
        self.assertIn('communication_timeout', codes_for(out, 1))
        self.assertNotIn('communication_timeout', codes_for(out, 0))

    def test_future_timestamp_beyond_clock_tolerance_is_rejected(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        tolerance = float(limits.clock_tolerance_s.value)
        out = run(shield, make_target(joints=np.full((N, 6), 0.1), timestamp=tolerance + 1.0),
                  state=make_state(), now=0.0)
        self.assertTrue(out.limited)
        self.assertTrue(any('command_timestamp_future' in v for v in out.violations))
        np.testing.assert_allclose(out.joint_position_target, np.zeros((N, 6)))


class RobustnessTests(unittest.TestCase):
    def test_non_finite_target_is_zeroed_instead_of_raised(self) -> None:
        shield = make_shield()
        target = make_target(joints=np.full((N, 6), 0.1))
        target.joint_position_target = np.full((N, 6), 0.1)   # bypass __post_init__ on purpose
        target.joint_position_target[1, 3] = np.nan
        out = run(shield, target, state=make_state())
        self.assertTrue(out.limited)
        self.assertTrue(any('invalid_non_finite' in v for v in out.violations))
        np.testing.assert_allclose(out.joint_position_target[1], np.zeros(6))
        self.assertFalse(any(v.startswith('env0:invalid') for v in out.violations))

    def test_wrong_message_type_and_shape_are_refused(self) -> None:
        shield = make_shield()
        with self.assertRaises(BrainBoundaryError):
            shield.process(make_state())
        target = make_target()
        target.base_twist = np.zeros((N, 4))
        with self.assertRaises(BrainBoundaryError):
            shield.process(target)

    def test_env_count_mismatch_is_refused(self) -> None:
        shield = make_shield()
        with self.assertRaises(BrainBoundaryError):
            shield.process(make_target(n=N + 1), state=make_state(n=N + 1))

    def test_bad_estop_mask_and_bad_context_are_refused(self) -> None:
        shield = make_shield()
        with self.assertRaises(BrainBoundaryError):
            shield.process(make_target(), state=make_state(),
                           context=SafetyContext(estop=np.array([True, False])))
        with self.assertRaises(BrainBoundaryError):
            shield.set_context({'estop': True})


class ScanTests(unittest.TestCase):
    def test_scan_of_random_targets_never_emits_a_limit_violation(self) -> None:
        limits = make_limits()
        shield = make_shield(limits)
        rng = np.random.default_rng(0)
        lo, hi = joint_bounds(limits)
        dq = np.asarray(limits.joint_velocity_max.value, dtype=float)
        vmax = np.asarray(limits.base_twist_axis_max.value, dtype=float)
        v_speed = float(limits.base_translation_speed_max.value)
        ref = 0.5 * (lo + hi) * rng.uniform(-0.5, 0.5, size=(N, 6))
        pose = pose_in_box(limits)
        for step in range(200):
            joints = rng.uniform(-20.0, 20.0, size=(N, 6))
            twist = rng.uniform(-5.0, 5.0, size=(N, 3))
            estop = rng.random(N) < 0.1
            out = run(shield, make_target(base_twist=twist, joints=joints, timestamp=step * DT),
                      state=make_state(joint_pos=ref, racket_pose=pose), now=step * DT,
                      context=SafetyContext(estop=estop))
            q = np.asarray(out.joint_position_target)
            v = np.asarray(out.base_twist)
            self.assertTrue(np.all(q <= hi + 1e-12) and np.all(q >= lo - 1e-12),
                            msg='joint position limit violated at step %d' % step)
            self.assertTrue(np.all(np.abs(v) <= vmax + 1e-12),
                            msg='base twist axis limit violated at step %d' % step)
            self.assertTrue(np.all(np.hypot(v[:, 0], v[:, 1]) <= v_speed + 1e-12),
                            msg='combined translation limit violated at step %d' % step)
            self.assertTrue(np.all(np.isfinite(q)) and np.all(np.isfinite(v)))
            active = ~estop
            if active.any():
                self.assertTrue(np.all(np.abs(q[active] - ref[active]) <= dq * DT + 1e-12),
                                msg='joint velocity limit violated at step %d' % step)
            if estop.any():
                self.assertTrue(out.limited)
                for env in np.flatnonzero(estop):
                    np.testing.assert_allclose(q[env], np.zeros(6))
                    np.testing.assert_allclose(v[env], np.zeros(3))
            if not out.limited:
                np.testing.assert_allclose(v, twist, atol=1e-12)
                np.testing.assert_allclose(q, joints, atol=1e-12)
            ref = q

    def test_violations_are_deterministic_sorted_and_unique(self) -> None:
        shield = make_shield()
        twist = np.tile(np.array([9.0, 9.0, 9.0]), (N, 1))
        joints = np.tile(np.array([9.0, -9.0, 9.0, -9.0, 9.0, -9.0]), (N, 1))
        first = run(shield, make_target(base_twist=twist, joints=joints), state=make_state()).violations
        shield.reset(None)
        second = run(shield, make_target(base_twist=twist, joints=joints), state=make_state()).violations
        self.assertEqual(first, second)
        self.assertEqual(list(first), sorted(first))
        self.assertEqual(len(set(first)), len(first))
        self.assertTrue(all(v.startswith('env') for v in first))


if __name__ == '__main__':
    unittest.main()
