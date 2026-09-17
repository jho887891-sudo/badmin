# reBot B601-DM badminton capability: delivery package

Assembled by `tools/assemble_b601dm_delivery.py` from `cc1a60aeb0136dd78eecae80f9e016006ffab50b`.

**This folder is generated. Do not edit it in place.** The files above it in the repository are the
single point of truth; a change made here will be overwritten the next time the assembler runs, and
worse, will not be in the repository at all.

Everything below is committed to git. Bulk images and the 14 MB binary geometry payload are
gitignored and are NOT needed to load these assets.

---

## Read this first

| | |
|---|---|
| Gate 1, asset honesty | **PASS** |
| Gate 2, provenance completeness | **PASS** |
| Gate 3, physical plausibility | **UNEVALUATED - no valid measurement exists** |

**The delivery is not accepted, because one gate cannot be evaluated.** The first swing record that was
produced is invalid and is kept only as evidence; see `01_reports/FIRST_SWING_RECORD_INVALID.md`.

## Contents

| Folder | What is in it |
|---|---|
| `01_reports/` | the decision record, the A-M inspection, the gate result, the TEMP register, the plan |
| `02_joint_limits/` | every joint limit with a provenance tag, and the YAML mirror |
| `03_simulation_assets/` | the patched USD trees: real limits, and the official 70 percent variant |
| `04_torque_and_mass/` | the torque convention, and the motor mass error quantified |
| `05_launch_and_racket/` | the launcher for this host, and the racket attachment |
| `06_measurement/` | the speed bound, the swing measurement, the window watcher |
| `07_tests/` | the test suite, 135 passing |
| `08_package/` | the Python package files these modules live in |

## The single most important number in this delivery

The shipped simulation asset permitted joint speeds **9.09x and 9.55x** the real motors, because the
real rpm figures had been used as rad/s. The patched trees here enforce **315 and 1200 deg/s** where the
upstream ones enforced 2864.789 and 11459.156. Everything else in this package exists to make that
correction trustworthy and to have somewhere to measure with it.

## Files

| Folder | File | Size | sha256 (16) | Source in the repository |
|---|---|---|---|---|
| 01_reports | `ARM_IDENTITY_DECISION.md` | 2,530 B | `17a9a502e7c9b8fb` | `management/rebot_b601dm/ARM_IDENTITY_DECISION.md` |
| 01_reports | `REBOT_B601DM_REPORT.md` | 11,619 B | `90ebb75153fc3568` | `management/rebot_b601dm/REBOT_B601DM_REPORT.md` |
| 01_reports | `BADMINTON_CAPABILITY_RESULT.md` | 7,898 B | `bf6b41af3c2cf04d` | `management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md` |
| 01_reports | `DELIVERABLES.md` | 7,792 B | `e36e6a8d55153c35` | `management/rebot_b601dm/DELIVERABLES.md` |
| 01_reports | `TEMP_README.md` | 4,615 B | `3e303e6966459020` | `management/rebot_b601dm/TEMP_README.md` |
| 01_reports | `TASK7_MEASUREMENT_STATUS.md` | 4,125 B | `f7c49e4ed3373e5a` | `management/rebot_b601dm/TASK7_MEASUREMENT_STATUS.md` |
| 01_reports | `FIRST_SWING_RECORD_INVALID.md` | 2,722 B | `f8870e3a84226281` | `management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md` |
| 01_reports | `2026-09-17-rebot-b601dm-badminton-capability-plan.md` | 14,871 B | `745bd5545af3d635` | `docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md` |
| 02_joint_limits | `joint_limits.py` | 7,722 B | `0964366db5985700` | `src/simulation/rebot_b601dm/joint_limits.py` |
| 02_joint_limits | `rebot_b601dm_joint_limits.yaml` | 6,666 B | `fc4498f32f2cb3c4` | `configs/simulation/rebot_b601dm_joint_limits.yaml` |
| 02_joint_limits | `generate_b601dm_limits_yaml.py` | 4,492 B | `7249b6cb15a8ffa9` | `tools/generate_b601dm_limits_yaml.py` |
| 03_simulation_assets | `patch_asset.py` | 5,682 B | `cde5b84c4f80236a` | `src/simulation/rebot_b601dm/patch_asset.py` |
| 03_simulation_assets | `reBot_B601_DM__real_limits.usda` | 9,875 B | `8d21beddc01a8870` | `outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda` |
| 03_simulation_assets | `PATCH_NOTES__real_limits.md` | 1,290 B | `267b1343ec971a2e` | `outputs/simulation/rebot_b601dm/real_limits/PATCH_NOTES.md` |
| 03_simulation_assets | `physx__real_limits.usda` | 4,955 B | `7cab2579fa908bfd` | `outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physx.usda` |
| 03_simulation_assets | `physics__real_limits.usda` | 16,232 B | `734af4dce9466c5f` | `outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physics.usda` |
| 03_simulation_assets | `reBot_B601_DM__recommended_70.usda` | 9,875 B | `8d21beddc01a8870` | `outputs/simulation/rebot_b601dm/recommended_70_limits/reBot_B601_DM.usda` |
| 03_simulation_assets | `PATCH_NOTES__recommended_70.md` | 1,300 B | `23ec323091829928` | `outputs/simulation/rebot_b601dm/recommended_70_limits/PATCH_NOTES.md` |
| 03_simulation_assets | `physx__recommended_70.usda` | 4,952 B | `efc027cf8588435d` | `outputs/simulation/rebot_b601dm/recommended_70_limits/payloads/Physics/physx.usda` |
| 04_torque_and_mass | `torque_convention.py` | 5,059 B | `52fea0a4848c8066` | `src/simulation/rebot_b601dm/torque_convention.py` |
| 04_torque_and_mass | `motor_mass.py` | 7,253 B | `da22887489e30746` | `src/simulation/rebot_b601dm/motor_mass.py` |
| 04_torque_and_mass | `generate_b601dm_temp_readme.py` | 4,497 B | `6aa11921ac4c3a19` | `tools/generate_b601dm_temp_readme.py` |
| 05_launch_and_racket | `launch.py` | 3,737 B | `21c22f4bede13752` | `src/simulation/rebot_b601dm/launch.py` |
| 05_launch_and_racket | `isaacsim_receiver_driver.py` | 3,957 B | `ceccef1171c37dd5` | `scripts/simulation/isaacsim_receiver_driver.py` |
| 05_launch_and_racket | `run_isaacsim_receiver.sh` | 2,612 B | `8711ceb5c9918058` | `scripts/simulation/run_isaacsim_receiver.sh` |
| 05_launch_and_racket | `racket.py` | 6,099 B | `267f095cd2654bbd` | `src/simulation/rebot_b601dm/racket.py` |
| 06_measurement | `speed_bound.py` | 7,350 B | `1fe133bcb373c5a8` | `src/simulation/rebot_b601dm/speed_bound.py` |
| 06_measurement | `measure_racket_speed.py` | 8,596 B | `1696588c29c429d5` | `scripts/simulation/measure_racket_speed.py` |
| 06_measurement | `wait_for_quiet_window.sh` | 2,671 B | `20401430a38ba140` | `scripts/simulation/wait_for_quiet_window.sh` |
| 07_tests | `conftest.py` | 259 B | `1a30dc889d73226a` | `tests\simulation\rebot_b601dm\conftest.py` |
| 07_tests | `test_deliverables_index.py` | 2,093 B | `2693366b69c8a889` | `tests\simulation\rebot_b601dm\test_deliverables_index.py` |
| 07_tests | `test_delivery_package.py` | 3,116 B | `fb1772584977f413` | `tests\simulation\rebot_b601dm\test_delivery_package.py` |
| 07_tests | `test_joint_limits.py` | 8,902 B | `7ba669793fcf2b4d` | `tests\simulation\rebot_b601dm\test_joint_limits.py` |
| 07_tests | `test_launcher_contract.py` | 5,942 B | `a9abf1a45fa51dfd` | `tests\simulation\rebot_b601dm\test_launcher_contract.py` |
| 07_tests | `test_limits_yaml_consistency.py` | 3,168 B | `e3be24ce739027a5` | `tests\simulation\rebot_b601dm\test_limits_yaml_consistency.py` |
| 07_tests | `test_measure_script.py` | 4,087 B | `738998faf12ebc48` | `tests\simulation\rebot_b601dm\test_measure_script.py` |
| 07_tests | `test_motor_mass.py` | 6,634 B | `ccbd433563735cd4` | `tests\simulation\rebot_b601dm\test_motor_mass.py` |
| 07_tests | `test_patch_asset.py` | 5,489 B | `7e92fefaf3566d37` | `tests\simulation\rebot_b601dm\test_patch_asset.py` |
| 07_tests | `test_racket_attachment.py` | 6,273 B | `9fd7b7a5dad71cf9` | `tests\simulation\rebot_b601dm\test_racket_attachment.py` |
| 07_tests | `test_speed_bound.py` | 6,327 B | `f2415eb80ac28541` | `tests\simulation\rebot_b601dm\test_speed_bound.py` |
| 07_tests | `test_torque_convention.py` | 4,396 B | `72cab31ff22565b8` | `tests\simulation\rebot_b601dm\test_torque_convention.py` |
| 08_package | `__init__.py` | 250 B | `c502d63ecf78f2f2` | `src/simulation/__init__.py` |
| 08_package | `__init__.py` | 197 B | `4da0d8103b40643d` | `src/simulation/rebot_b601dm/__init__.py` |

---

## Rebuilding this folder

```bash
python tools/assemble_b601dm_delivery.py
```

## What is deliberately not in here

- a racket-head speed, because none has been validly measured
- the upstream repositories, which are inputs and are unchanged
- the 14 MB binary geometry payload, which is regenerable and gitignored
- V4 motor data, real joint angle limits, racket mass, and a thermal model, none of which were found
