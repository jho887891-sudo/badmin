# PHASE 3 RESULT - Morph One kinematics

**Status: PASS**

## Changed files
- `simulation/robots/badminton_robot/morph_one/kinematics.py`  (WheelTargets, normalise_angle,
  body_twist_to_wheel_targets = S10 IK + S11 optimisation, wheel_targets_to_body_twist least-squares)
- `simulation/robots/badminton_robot/morph_one/__init__.py`
- `tests/simulation/robots/test_morph_one_kinematics.py`

## Tests added
9 tests (S51/S52): pure forward (all wheels aligned, speed = vx/r), pure strafe (|theta| = 90 deg),
pure spin (tangential speeds), mixed motion round-trip (lstsq recovery, atol 1e-9),
zero command keeps steering and zero speed, wheel-radius/missing-wheel validation,
optimisation flips by exactly pi with negated wheel rate and |delta theta| <= 90 deg,
and the equivalence of (theta, w) with (theta+pi, -w).

## Tests executed
```
tests/simulation/robots/test_morph_one_kinematics.py   Ran 9 tests   OK
tests/simulation/robots/test_robot_config.py           Ran 18 tests  OK   (regression)
tests/simulation/robots/test_robot_frames.py           Ran 15 tests  OK   (regression)
```
TDD evidence: RED = `ModuleNotFoundError: robots.badminton_robot.morph_one.kinematics`;
one real bug found by the tests (reconstruction used the unsigned tangential magnitude instead of the
signed wheel rate, so the equivalent steer solution reconstructed a negated twist) -> fixed to the
physical definition v = w * r * (cos theta, sin theta); GREEN afterwards.

## Actual outputs
```
forward [0.5,0,0]     -> theta = 0 for all four wheels, |omega| = 8.333 rad/s (= 0.5/0.06)
strafe  [0,0.4,0]     -> |theta| = pi/2, |omega| = 6.667 rad/s
spin    [0,0,0.8]     -> theta_i = atan2(wz*x_i, -wz*y_i), |omega_i| = |v_i|/r
mixed   [0.35,-0.22,0.45] -> lstsq recovery matches the input within 1e-9
```

## TEMP assumptions
Wheel centres used in tests (0.25, +-0.20) and (-0.25, +-0.20) and wheel radius 0.06 m are passed
explicitly as TEMP inputs because `cfg.morph_one.geometry.wheel_positions_robot` is
`REQUIRES_MEASUREMENT`. The kinematics never invents them itself.

## Unknown real parameters
Wheel radius, wheel width and the four wheel centres (all still REQUIRES_MEASUREMENT in the cfg);
steering rate/wheel-speed limits are engineering baselines.

## Known limitations
No slip, no steering dynamics/lag, no wheel-ground contact model: this is the kinematic layer only
(STEER_DRIVE_WHEEL_MODEL dynamic validation belongs to a later phase).

## Frozen interfaces
Four-steer/four-drive topology; the eight semantic joint names; `[vx, vy, wz]` body-twist input;
`(steer_cmd (N,4), drive_cmd (N,4))` wheel-level output shape; S11 minimum-steering rule.

## Ready for next phase
YES  (Phase 4 = PiPER adapter; needs Isaac, so it is the first phase requiring a Kit run)
