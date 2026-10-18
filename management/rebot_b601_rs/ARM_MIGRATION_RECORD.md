# reBot B601-RS arm migration record (2026-10)

Status: **asset layer verified, scene layer integrated with one open defect, hardware layer unmeasured.**

The project arm moved from PiPER to the official **reBot B601-RS**. Nothing was renamed: no PiPER file
was turned into a B601-RS file, no B601-DM parameter was reused, and no third-party model was
substituted. Every number below is traceable to a file or a URL in this document.

---

## A. 型号核验 (identity)

| Item | Value | Source |
| --- | --- | --- |
| Model | reBot Arm **B601-RS** | repo README hardware table, B601-RS column |
| Robot name in asset | ReBot_Arm_RS | ReBot_Arm_RS.urdf <robot name> |
| DOF | **6 DOF + 1 gripper** | README table; 6 revolute joints in the URDF |
| Motors | **RobStride RS06 x3 + RS00 x4** | hardware/reBot_B601_RS/README.md BOM line |
| Payload / reach / weight / supply | 2.5 kg / 754 mm / ~6.7 kg / DC 48 V | README table, B601-RS column |
| Repeatability | < 0.2 mm | README table |
| Official asset repo | https://github.com/Seeed-Projects/reBot-DevArm | cloned on the GPU host |
| Commit used | f01a1dc189ecb74b5435a336a68b287c146c1b60 | git rev-parse HEAD |
| License | Seeed reBot-DevArm repository LICENSE (copied alongside the asset) | LICENSE |

**Why the DM column is not used.** The same README carries a second column: B601-DM = 1.5 kg payload,
767 mm reach, ~4.5 kg, DAMIAO motors. management/rebot_b601dm/ARM_IDENTITY_DECISION.md already records
DM as a test bench, not the project arm. Here DM is a *different product*; its reach (767 mm) and
motors (DM4340P/DM4310) appear nowhere in the RS tables.

**Motor cross-check (this is what makes the assignment real, not guessed).** The URDF effort column is
36 for joint1..joint3 and 14 for joint4..joint6. The Seeed product pages state RS06 = 36 N.m peak (9:1)
and RS00 = 14 N.m peak (10:1). The BOM's 3x RS06 + 4x RS00 covers joints 1-3, joints 4-6 and the one
gripper drive. Two independent directions agree.

## B. 资产获取 (assets)

```
/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/    (64 MB staged, source clone untouched)
  urdf/ReBot_Arm_RS.urdf        3838ca8f9d586be22a89b33c7cbdd4abd43d4a92a2de377b294eef4c5bd7654d
  urdf/ReBot_Arm_RS.csv         5b921c678f5c003869a9b410ad866fcc172f67435efc905fd32bd16e9434d959
  meshes/visual (29 MB, 22 STL) / shared (35 MB, 10 STL) / mujoco_collision (188 KB, 10 STL)
  README_RS.md, README_RS_zh.md, LICENSE_reBot-DevArm
  usd/rebot_b601_rs.usd/                                         20 MB conversion output
    config.yaml            importer settings (copied to outputs/simulation/rebot_b601_rs/conversion_config.yaml)
    ReBot_Arm_RS/ReBot_Arm_RS.usda
    ReBot_Arm_RS/payloads/{base,robot,instances,materials}.usd*
    ReBot_Arm_RS/payloads/geometries.usd
    ReBot_Arm_RS/payloads/Physics/{physx,physics,mujoco}.usda
```

Conversion command (reproducible, Isaac Lab convert_urdf.py):

```bash
cd /home/T7/ojh/robot_sim/IsaacLab
PYTHONPATH=$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets \
  /home/T7/ojh/robot_sim/env_isaaclab/bin/python scripts/tools/convert_urdf.py \
  <staged>/urdf/ReBot_Arm_RS.urdf <staged>/usd/rebot_b601_rs.usd \
  --merge-joints --fix-base --joint-stiffness 100 --joint-damping 1.0 \
  --joint-target-type position --headless
```

merge-joints folds the fixed j_gripper_end into link6, which is why link6 carries 0.75 kg (0.1 + 0.65)
and the model still totals 6.0085 kg. The official whole-assembly CAD
(hardware/reBot_B601_RS/reBot_B601_RS_v1.0_20260625.step) was located but **not** ingested: the URDF is
what drives simulation, and a STEP ingest would add a second, unverified geometry source.

## C. 现有项目问题 (what was wrong before this migration)

| Class | Files | Action |
| --- | --- | --- |
| Formal, migrated | simulation/robots/badminton_robot/badminton_robot_cfg.py, frames/robot_frames.py, simulation/badminton_scene/robot_cfg.py, scene_layout.py, src/badminton_brain/safety/safety_shield.py | now selects and describes the B601-RS |
| Historical, kept intact | simulation/piper_stage0/* (5 files), tools/build_piper_with_racket.py, tests/test_piper_with_racket.py, outputs/evidence/screenshots.json, outputs/reports/phase*_report.md | untouched; reachable via BADMINTON_ARM_MODEL=piper |
| Separate candidate line | src/simulation/rebot_b601dm/*, management/rebot_b601dm/*, deliverables/rebot_b601dm/*, configs/simulation/rebot_b601dm_joint_limits.yaml | untouched; DM parameters never imported |
| Still PiPER-worded | docs/simulation/BADMINTON_ROBOT.md (45 hits), docs/RACKET.md, docs/architecture/COORDINATE_SYSTEM.md, PROJECT_STATUS.md, AGENTS.md | not rewritten in this pass: see F5 |

Defects found while migrating (each caught by a test or a real run, not by reading):

1. **Frame-alias inversion.** FrameTree.get_transform inverted the transform when an *alias* was passed
   as the base, flipping every axis sign. Latent because aliases had only ever been used as the source
   frame. Fixed; tests/simulation/rebot_b601_rs/test_frame_aliases.py pins both roles.
2. **fix_root_link=True conflicts with the converted asset.** The conversion already authors a fixed
   root_joint; asking the spawner to fix it again raises NotImplementedError ('/Geometry' does not have
   the RigidBodyAPI). Now False.
3. **Symmetric joint-limit proxy in the safety shield.** SafetyLimits.temp_proxy() used +-2.6 rad on all
   six joints. On this arm joint2/joint3 are [0, 3.14] and joint4/joint5 are +-1.57, so a negative
   joint2 command was permissible and hardware-invalid. Now per-joint official ranges.
4. **A test fixture that assumed symmetry.** test_scan_of_random_targets_never_emits_a_limit_violation
   sampled states as 0.5*(lo+hi)*u(-0.5,0.5), which lands outside the B601-RS ranges; the shield was
   then required to clamp and respect the velocity budget at once. The fixture now samples inside.

Data inconsistencies recorded rather than silently resolved:

* **URDF velocity 50/40 is not a verified joint speed.** Same signature as the B601-DM asset, where rpm
  figures had been written as rad/s (9.09x / 9.55x overshoot). Marked OFFICIAL_URDF and barred from
  capability claims until a RobStride speed figure is obtained.
* **URDF masses are roughly 2x the exporter CSV** (base_link 1.1774 vs 0.5796, link2 1.552 vs 0.7533).
  The difference is consistent with motor mass being added by the exporter, but that is an inference,
  not a document. The URDF total (6.0085 kg) sits 0.69 kg under the 6.7 kg product weight.
* **The gripper has two prismatic finger joints and no mimic tag**, while the BOM has one gripper motor.
  The fingers are hardware-coupled through a rack; the coupling is not modelled.

## D. 实际修改 (changes)

New:

* src/simulation/rebot_b601_rs/{__init__,joint_limits}.py - per-joint limits with Provenance
  (CONFIRMED_REAL / OFFICIAL_URDF / UNKNOWN), gears, link masses, spec sheet, asset pin, and
  angle_range / joint_velocity / joint_torque(..., require_real=True) gates raising LimitNotReal.
* configs/simulation/rebot_b601_rs_joint_limits.yaml + tools/generate_rs_limits_yaml.py (generated).
* tests/simulation/rebot_b601_rs/ - 30 tests (limits, YAML consistency, frame aliases, rest pose,
  safety-shield agreement).
* outputs/simulation/rebot_b601_rs/ - VERIFY_usd_load.json, VERIFY_scene_load.json,
  rest_pose_probe.json, conversion_config.yaml.

Modified:

* simulation/robots/badminton_robot/badminton_robot_cfg.py - new RebotB601RsCfg, arm_model selector
  (default rebot_b601_rs), rebot_* canonical frames; PiperCfg marked HISTORICAL.
* simulation/robots/badminton_robot/frames/robot_frames.py - arm-aware prefix, legacy piper_* names
  aliased to the same frames, alias-inversion fix.
* simulation/badminton_scene/scene_layout.py - REBOT_RS block (mount, geometry-probed rest pose).
* simulation/badminton_scene/robot_cfg.py - spawns the converted B601-RS; PiPER only under
  BADMINTON_ARM_MODEL=piper.
* src/badminton_brain/safety/safety_shield.py - per-joint official ranges.
* tests/badminton_brain/test_safety_shield.py - legal-state fixture.

## E. 测试证据 (evidence)

| Check | Result | Where |
| --- | --- | --- |
| URDF -> USD conversion | exit 0, 20 MB output, importer config recorded | conversion_config.yaml |
| Stage dependencies resolve | missing_asset_dependencies: [] | VERIFY_usd_load.json |
| Mesh geometry present | **35 mesh prims / 730,663 vertices** | VERIFY_usd_load.json |
| Rigid bodies / joints | **9 bodies, 8 joints + fixed root** | VERIFY_usd_load.json |
| Limits as the simulator sees them | +-2.8, [0,3.14], [0,3.14], +-1.57, +-1.57, +-3.14 rad; fingers [0,0.05], [0,0.0715] m | VERIFY_usd_load.json |
| Masses | per-link equal to the URDF; **total 6.008504 kg** (= URDF 6.0085) | VERIFY_usd_load.json |
| Joint motion | joint1 +0.100 rad, joint2 +0.0868 rad over 240 steps, finite, all within limits | VERIFY_usd_load.json |
| Scene assembly | prim {ENV_REGEX_NS}/Robot/RebotB601RS, base at (-1.6, 0, 0.3), 9 bodies / 8 joints, 35 robot meshes, shuttle absent | VERIFY_scene_load.json |
| Rest-pose geometry | 8 candidate poses measured; chosen joint2=0.3, joint3=1.0 keeps every link above the base plane, link6 0.497 m above the mount | rest_pose_probe.json |
| Local test suite | **1,666 passed, 13 skipped**; 4 pre-existing failures in tests/tools/test_usd_glb_common.py (need pxr; files unmodified by this work, git status clean) | pytest tests -q |
| B601-RS package | **30 passed** | pytest tests/simulation/rebot_b601_rs -q |

**Open defect (not hidden).** In the assembled scene the rest pose does not hold: over 60 steps joint3
drifts from +1.0 rad to -0.067 rad, i.e. **below its own lower limit of 0**, and link6 ends at
z = 0.108 m instead of about 0.80 m. Raising the implicit drive stiffness from 100 to 5000 changed the
result by less than 3 mrad, which points at the drives/limits not being applied rather than at a gain
choice. Recorded in VERIFY_scene_load.json (rest_error_rad, all_within_limits: false).

## F. 剩余工作 (open items)

1. **Scene rest-pose hold (blocks any scene/HUD evidence).** Isolate why the command does not track:
   compare robot.data.default_joint_pos against init_state.joint_pos, inspect the USD drive attributes
   on link2/link3, and check whether the PhysX joint limits are hard. Until fixed, no screenshot or
   capability number may come from the B601-RS scene.
2. **Joint speed limit.** Obtain the official RobStride RS00/RS06 output speed. Until then
   joint_velocity(..., require_real=True) raises and the 50/40 values stay OFFICIAL_URDF.
3. **Racket adapter.** configs/racket.yaml keeps mount_transform.status = REQUIRES_MEASUREMENT with null
   translation/quaternion, so no B601-RS racket USD can be authored. The racket is not mounted on this
   arm yet; gripper_end / link6 is the flange to measure from.
4. **Gripper coupling direction** (rack/pinion, two URDF fingers, one motor) is not modelled.
5. **Docs still say PiPER**: docs/simulation/BADMINTON_ROBOT.md (45 references), docs/RACKET.md,
   docs/architecture/COORDINATE_SYSTEM.md, PROJECT_STATUS.md, AGENTS.md. Rewriting the spec is a
   separate reviewable change; doing it here would have buried the migration diff.
6. **Hardware confirmation** of robot_model: rs + camera_mount: chassis_fixed is still the user's call.

## G. Git 状态

Worked on feature/shuttle-detection on top of 98487bf. Only this migration's files are committed;
another session's in-flight files (deliverables/rebot_b601dm/*, management/rebot_b601dm/DELIVERABLES.md,
scripts/simulation/render_shuttle_3d_probe.py, outputs/three_d_render_probe/,
outputs/yolo26s_speed_opt/render_manifest.json) are left untouched. Nothing is pushed.
