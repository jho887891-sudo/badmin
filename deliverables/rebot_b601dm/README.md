# reBot B601-DM badminton capability: delivery package

Assembled by `tools/assemble_b601dm_delivery.py` from commit `3a679b35fc326ed8eadfdaf299f56ce19d28c321`.

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
| `management/rebot_b601dm/ARM_IDENTITY_DECISION.md` | 2,571 B | `b8fe2226e39a67cc` |
| `management/rebot_b601dm/REBOT_B601DM_REPORT.md` | 11,834 B | `d81d1c612204c915` |
| `management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md` | 8,041 B | `444ad9f02eab7a5a` |
| `management/rebot_b601dm/DELIVERABLES.md` | 7,793 B | `c6480aeb2a0d23b5` |
| `management/rebot_b601dm/TEMP_README.md` | 4,615 B | `3e303e6966459020` |
| `management/rebot_b601dm/TASK7_MEASUREMENT_STATUS.md` | 4,202 B | `725fc9b0e551321a` |
| `management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md` | 2,777 B | `0b65ef2beac385a9` |
| `docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md` | 15,160 B | `575c0d3cf2f70643` |
| `src/simulation/__init__.py` | 259 B | `ea9666627706450b` |
| `src/simulation/rebot_b601dm/__init__.py` | 206 B | `d624155876b65bff` |
| `src/simulation/rebot_b601dm/joint_limits.py` | 7,898 B | `60d106a2560b805c` |
| `src/simulation/rebot_b601dm/patch_asset.py` | 5,825 B | `8023dbf99e06bb49` |
| `src/simulation/rebot_b601dm/torque_convention.py` | 5,180 B | `e38953c7d883eaba` |
| `src/simulation/rebot_b601dm/motor_mass.py` | 7,419 B | `225c5272c524d992` |
| `src/simulation/rebot_b601dm/launch.py` | 3,833 B | `9ee90137dd2d666a` |
| `src/simulation/rebot_b601dm/racket.py` | 6,246 B | `ca2d5f444adc9776` |
| `src/simulation/rebot_b601dm/speed_bound.py` | 7,514 B | `888ba70da03914b1` |
| `configs/simulation/rebot_b601dm_joint_limits.yaml` | 6,666 B | `fc4498f32f2cb3c4` |
| `scripts/simulation/isaacsim_receiver_driver.py` | 4,059 B | `53cc6dcc761f1135` |
| `scripts/simulation/run_isaacsim_receiver.sh` | 2,666 B | `573511192477d1ea` |
| `scripts/simulation/measure_racket_speed.py` | 12,409 B | `a5bf27dfefb195d7` |
| `scripts/simulation/wait_for_quiet_window.sh` | 2,721 B | `dbb2c749807218d1` |
| `tools/generate_b601dm_limits_yaml.py` | 4,575 B | `485523a182b90ada` |
| `tools/generate_b601dm_temp_readme.py` | 4,605 B | `bd9587e7d423bf2f` |
| `outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda` | 9,875 B | `8d21beddc01a8870` |
| `outputs/simulation/rebot_b601dm/real_limits/PATCH_NOTES.md` | 1,290 B | `267b1343ec971a2e` |
| `outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physx.usda` | 4,955 B | `7cab2579fa908bfd` |
| `outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physics.usda` | 16,232 B | `734af4dce9466c5f` |
| `outputs/simulation/rebot_b601dm/recommended_70_limits/reBot_B601_DM.usda` | 9,875 B | `8d21beddc01a8870` |
| `outputs/simulation/rebot_b601dm/recommended_70_limits/PATCH_NOTES.md` | 1,300 B | `23ec323091829928` |
| `outputs/simulation/rebot_b601dm/recommended_70_limits/payloads/Physics/physx.usda` | 4,952 B | `efc027cf8588435d` |
| `tests/simulation/rebot_b601dm/conftest.py` | 267 B | `0789b7c9a5aab484` |
| `tests/simulation/rebot_b601dm/test_joint_limits.py` | 9,084 B | `970fe6e5dd72cabf` |
| `tests/simulation/rebot_b601dm/test_launcher_contract.py` | 6,089 B | `5d34e6d6f46ce77e` |
| `tests/simulation/rebot_b601dm/test_limits_yaml_consistency.py` | 3,247 B | `f2c1e2bb9782f038` |
| `tests/simulation/rebot_b601dm/test_measure_script.py` | 8,565 B | `818197febb5b1a44` |
| `tests/simulation/rebot_b601dm/test_motor_mass.py` | 6,782 B | `86c2c26b2c052a8c` |
| `tests/simulation/rebot_b601dm/test_patch_asset.py` | 5,612 B | `95b8b75afa94541e` |
| `tests/simulation/rebot_b601dm/test_racket_attachment.py` | 6,421 B | `e38f13059e6f9c62` |
| `tests/simulation/rebot_b601dm/test_speed_bound.py` | 6,470 B | `2e1dcce390b59ec0` |
| `tests/simulation/rebot_b601dm/test_torque_convention.py` | 4,490 B | `0db791f8645180d1` |

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
