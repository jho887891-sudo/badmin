# reBot B601-DM badminton capability: delivery package

Assembled by `tools/assemble_b601dm_delivery.py` from commit `c798f789c4a9891a9e9704d31502bcb447b576f4`.

**This folder is generated. Do not edit it in place.** The repository above it is the single point of
truth; a change made here is lost the next time the assembler runs, and is not in the repository at all.

**It mirrors the repository layout on purpose.** The tests import `src.simulation.rebot_b601dm.*`
so the paths have to be real for anything here to run. An earlier version filed the files into numbered
folders where no import could reach them and not one of its own tests would execute.

---

## Verify it before trusting it

```bash
cd deliverables/rebot_b601dm
pytest tests/simulation/rebot_b601dm -q
```

That runs the packaged suite against the packaged modules, entirely inside this folder. If it does not
pass, the package is not a package. Tests that skip do so because the upstream repositories are not
in here, which is expected.

**Two test files are deliberately not packaged**: `test_deliverables_index.py` and
`test_delivery_package.py`. They check the repository layout rather than the shipped code - the
first asserts that the repository index lists files that exist, and inside this folder that index points
at repository paths which are not here. Shipping them would make the suite fail on arrival and teach a
reader to ignore failures. They run in the repository.

---

## Gate status

| | |
|---|---|
| Gate 1, asset honesty | **PASS** |
| Gate 2, provenance completeness | **PASS** |
| Gate 3, physical plausibility | **UNEVALUATED - no valid measurement exists** |

**The delivery is not accepted, because one gate cannot be evaluated.** The one swing record that was
produced is invalid; see `management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md`.

## The single most important number here

The shipped simulation asset permitted joint speeds **9.09x and 9.55x** the real motors, because the real
rpm figures had been used as rad/s. The patched trees enforce **315 and 1200 deg/s** where the upstream
ones enforced 2864.789 and 11459.156. Everything else exists to make that correction trustworthy.

## What is in here

| Section | Folder | What it is |
|---|---|---|
| Reports and decision records | `management/rebot_b601dm/` | the decision record, the A-M inspection, the gate result, the TEMP register, and the two records of what went wrong (7 files) |
| The plan | `docs/superpowers/plans/` | the task plan this work follows (1 files) |
| The package | `src/simulation/rebot_b601dm/` | joint limits with provenance, the asset patcher, the torque convention, the mass audit, the launcher, the racket, the speed bound (8 files) |
| Configuration | `configs/simulation/` | the generated YAML mirror of the limit table (1 files) |
| Scripts | `scripts/simulation/` | the launcher, the receiver driver, the swing measurement, the window watcher (4 files) |
| Generators | `tools/` | the two scripts that keep the YAML and the TEMP register from drifting (2 files) |
| Patched assets | `outputs/simulation/rebot_b601dm/` | two USD variants: real limits, and the official 70 percent variant (7 files) |
| Tests | `tests/simulation/rebot_b601dm/` | the suite, runnable from this folder (10 files) |

## Every file, with its hash

| Path in this package | Size | sha256 (16) |
|---|---|---|
| `management/rebot_b601dm/ARM_IDENTITY_DECISION.md` | 2,530 B | `17a9a502e7c9b8fb` |
| `management/rebot_b601dm/REBOT_B601DM_REPORT.md` | 11,619 B | `90ebb75153fc3568` |
| `management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md` | 7,898 B | `bf6b41af3c2cf04d` |
| `management/rebot_b601dm/DELIVERABLES.md` | 7,793 B | `c6480aeb2a0d23b5` |
| `management/rebot_b601dm/TEMP_README.md` | 4,615 B | `3e303e6966459020` |
| `management/rebot_b601dm/TASK7_MEASUREMENT_STATUS.md` | 4,125 B | `f7c49e4ed3373e5a` |
| `management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md` | 2,722 B | `f8870e3a84226281` |
| `docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md` | 14,871 B | `745bd5545af3d635` |
| `src/simulation/__init__.py` | 250 B | `c502d63ecf78f2f2` |
| `src/simulation/rebot_b601dm/__init__.py` | 197 B | `4da0d8103b40643d` |
| `src/simulation/rebot_b601dm/joint_limits.py` | 7,722 B | `0964366db5985700` |
| `src/simulation/rebot_b601dm/patch_asset.py` | 5,682 B | `cde5b84c4f80236a` |
| `src/simulation/rebot_b601dm/torque_convention.py` | 5,059 B | `52fea0a4848c8066` |
| `src/simulation/rebot_b601dm/motor_mass.py` | 7,253 B | `da22887489e30746` |
| `src/simulation/rebot_b601dm/launch.py` | 3,737 B | `21c22f4bede13752` |
| `src/simulation/rebot_b601dm/racket.py` | 6,099 B | `267f095cd2654bbd` |
| `src/simulation/rebot_b601dm/speed_bound.py` | 7,350 B | `1fe133bcb373c5a8` |
| `configs/simulation/rebot_b601dm_joint_limits.yaml` | 6,666 B | `fc4498f32f2cb3c4` |
| `scripts/simulation/isaacsim_receiver_driver.py` | 3,957 B | `ceccef1171c37dd5` |
| `scripts/simulation/run_isaacsim_receiver.sh` | 2,612 B | `8711ceb5c9918058` |
| `scripts/simulation/measure_racket_speed.py` | 12,163 B | `07d2b392dfe6b5a5` |
| `scripts/simulation/wait_for_quiet_window.sh` | 2,671 B | `20401430a38ba140` |
| `tools/generate_b601dm_limits_yaml.py` | 4,492 B | `7249b6cb15a8ffa9` |
| `tools/generate_b601dm_temp_readme.py` | 4,497 B | `6aa11921ac4c3a19` |
| `outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda` | 9,875 B | `8d21beddc01a8870` |
| `outputs/simulation/rebot_b601dm/real_limits/PATCH_NOTES.md` | 1,290 B | `267b1343ec971a2e` |
| `outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physx.usda` | 4,955 B | `7cab2579fa908bfd` |
| `outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physics.usda` | 16,232 B | `734af4dce9466c5f` |
| `outputs/simulation/rebot_b601dm/recommended_70_limits/reBot_B601_DM.usda` | 9,875 B | `8d21beddc01a8870` |
| `outputs/simulation/rebot_b601dm/recommended_70_limits/PATCH_NOTES.md` | 1,300 B | `23ec323091829928` |
| `outputs/simulation/rebot_b601dm/recommended_70_limits/payloads/Physics/physx.usda` | 4,952 B | `efc027cf8588435d` |
| `tests/simulation/rebot_b601dm/conftest.py` | 259 B | `1a30dc889d73226a` |
| `tests/simulation/rebot_b601dm/test_joint_limits.py` | 8,902 B | `7ba669793fcf2b4d` |
| `tests/simulation/rebot_b601dm/test_launcher_contract.py` | 5,942 B | `a9abf1a45fa51dfd` |
| `tests/simulation/rebot_b601dm/test_limits_yaml_consistency.py` | 3,168 B | `e3be24ce739027a5` |
| `tests/simulation/rebot_b601dm/test_measure_script.py` | 8,392 B | `6211830956bf5541` |
| `tests/simulation/rebot_b601dm/test_motor_mass.py` | 6,634 B | `ccbd433563735cd4` |
| `tests/simulation/rebot_b601dm/test_patch_asset.py` | 5,489 B | `7e92fefaf3566d37` |
| `tests/simulation/rebot_b601dm/test_racket_attachment.py` | 6,273 B | `9fd7b7a5dad71cf9` |
| `tests/simulation/rebot_b601dm/test_speed_bound.py` | 6,327 B | `f2415eb80ac28541` |
| `tests/simulation/rebot_b601dm/test_torque_convention.py` | 4,396 B | `72cab31ff22565b8` |

---

## What is deliberately not here

| Missing | Why |
|---|---|
| a valid racket-head speed | none has been measured; the first record measured a joint it had itself told to stop |
| the upstream repositories | they are inputs, unchanged, at 1958511342dab181b94a2b7c068c77e54eb85d3a and f01a1dc189ecb74b5435a336a68b287c146c1b60 |
| the binary geometry payload | 14 MB, regenerable, gitignored; not needed to load these assets |
| V4 motor data | the official table documents -2EC variants while the BOM specifies V4 |
| real joint angle limits | no hardware document in either repository states a range |
| racket mass and inertia | the asset is a mesh with no inertial data |
| a thermal model | the published testing gives a duty-cycle constraint, not resistance or capacity |

## Rebuilding

```bash
python tools/assemble_b601dm_delivery.py
```
