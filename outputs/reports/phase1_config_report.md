# PHASE 1 RESULT - Config schema

**Status: PASS**

## Changed files
- `simulation/robots/__init__.py`, `simulation/robots/badminton_robot/__init__.py`
- `simulation/robots/badminton_robot/badminton_robot_cfg.py`  (AssetStatus, Param, WheelId, DriveMode, MorphOne/Piper/Racket/StereoCamera cfg, BadmintonRobotCfg, make_default_robot_cfg)
- `simulation/robots/badminton_robot/validation/robot_validator.py`  (validate_robot_cfg, ValidationReport, ValidationError)
- `tests/simulation/robots/test_robot_config.py`

## Tests added
18 tests (S49): AssetStatus enum completeness, Param status/source rules, TEMP explicitness,
four-steer/four-drive topology, TEMP vs REQUIRES_MEASUREMENT defaults, PiPER frozen joint order,
quaternion normalization check, stereo baseline ordering, drive modes, development vs final mode,
bad quaternion / wrong joint count / non-positive baseline rejection, engineering-baseline initial pose.

## Tests executed
```
tests/simulation/robots/test_robot_config.py   Ran 18 tests   OK
```
TDD evidence: RED = `ModuleNotFoundError: No module named 'robots'` before implementation;
GREEN after implementation.

## Actual outputs
```
validate_robot_cfg(cfg, mode="development") -> ok=True, warnings=[...TEMP/REQUIRES_MEASUREMENT...]
validate_robot_cfg(cfg, mode="final")       -> ValidationError("final mode requires resolved parameters; still TEMP/UNKNOWN: ...")
```

## TEMP assumptions
Chassis 0.70x0.55 m, wheel radius 0.06 m, wheel width 0.04 m, steer rate 6 rad/s,
wheel speed 40 rad/s, steer range pi, PiPER mount (0,0,0.30)+identity quaternion,
stereo baseline 0.29 m with +-0.145 m y offsets, camera centre (+0.20, 0, +1.20) pitch -4 deg.
All marked `TEMP_PARAMETERIZED_PROXY` with a source string; see `morph_one/TEMP_README.md`.

## Unknown real parameters
morph_one wheel positions, base mass/COM/inertia, all measured racket fields, `T_link6_tcp`,
racket string-bed contact centre, all contact-pair parameters (shuttle-ground/net/racket).

## Known limitations
Prim-path resolution, tensor interfaces and Isaac-side checks are not part of Phase 1
(they arrive with the robot class in later phases).

## Frozen interfaces
AssetStatus vocabulary (S4); TEMP policy (S12); PiPER joint order (S13);
four-steer/four-drive topology and the eight semantic joint names (S8/S63);
Court Frame (S3.1); canonical frame names from COORDINATE_SYSTEM S2.5.

## Ready for next phase
YES
