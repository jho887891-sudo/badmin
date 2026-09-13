# PHASE 2 RESULT - Frames

**Status: PASS**

## Changed files
- `simulation/robots/badminton_robot/frames/robot_frames.py`  (Transform, FrameTree, quat helpers, make_robot_frame_tree, compose_racket_contact_court, court_from_world, world_from_court)
- `tests/simulation/robots/test_robot_frames.py`

## Tests added
15 tests (S50/S41/S28): identity + inverse round-trip (max error < 1e-12), composition equals the
manual matrix product, non-unit quaternion rejection, canonical frames present, court->robot_base,
robot_base->piper_base from cfg, link6->tcp->contact chain, alias equivalence, camera baseline and
left/right ordering, unknown frame raises, whole-body composition matches the closed-form product,
explicit MissingTransformError for the unmeasured adapter, env_origin strip/add round-trip.

## Tests executed
```
tests/simulation/robots/test_robot_frames.py   Ran 15 tests   OK
tests/simulation/robots/test_robot_config.py   Ran 18 tests   OK   (Phase 1 regression)
```
TDD evidence: RED = `ModuleNotFoundError: robots.badminton_robot.frames.robot_frames`;
then a signature mismatch was fixed from the failing test; GREEN afterwards.

## Actual outputs
```
T_court_contact composed by make_robot_frame_tree == closed-form product of the five transforms (atol 1e-12)
court_from_world([14.0,-15.5,1.2], env_origin=[15,-15,0]) -> [-1.0,-0.5,1.2]
```

## TEMP assumptions
Camera pitch -4 deg and baseline 0.29 m from cfg; `imu_link` is placed as an identity TEMP child of
`robot_base` because no source specifies its location.

## Unknown real parameters
`racket.t_link6_tcp` and `racket.t_tcp_contact` (both REQUIRES_MEASUREMENT): the frame chain
refuses to build without them and raises `MissingTransformError` naming the missing measurement.

## Known limitations
Runtime FK for `piper_link6` is supplied by the caller (Phase 4 adapter); the tree does not query Isaac yet.
Frame-name aliases are resolved to one truth: `racket_contact_frame -> racket_contact`,
`camera_rig -> camera_center` (see ISSUES).

## Frozen interfaces
`T_A_B` naming (COORDINATE_SYSTEM S2.6); canonical frame names (S2.5); Court Frame (S3.1);
env_origin never leaks above this module (S28); S41 composition order.

## Ready for next phase
YES
