# Phases B / C / D - what is blocked, what is delivered, and the design that must be verified

Binding rule from the task: 阶段 A 的 CAD 分类如遇不可证明的关节结构，停止进一步的物理装配宣称。
Phase A classified **no** geometry (282 bodies all UNKNOWN, see PHASE_A_CAD_ASSETIZATION.md), so this
document does not claim any physical assembly. It delivers the interface layer, the mounting design, and
an explicit list of what would be needed to verify each acceptance item.

## Phase B - four-steer / four-drive kinematic assembly: interface delivered, geometry blocked

Delivered (code, tested, no Isaac Sim needed): `simulation/badminton_scene/morph_one_chassis.py`

| Content | What it guarantees |
| --- | --- |
| SEMANTIC_JOINTS, exactly 8, canonical order | fl_steer, fl_drive, fr_steer, fr_drive, rl_steer, rl_drive, rr_steer, rr_drive - identical to the project's existing STEER_DRIVE_JOINT_NAMES builder |
| validate_topology() | rejects duplicate modules, duplicate joint names, malformed names, wrong count |
| joint_index_map() | one canonical index per semantic joint, so a test can address joints by meaning |
| CHASSIS_PARAMETERS | every geometric quantity carries an explicit status |
| REQUIRED_FRAMES | the frames that must exist once geometry is real (chassis_base, four steer axes, four wheel centres, arm_mount) |
| ChassisParameter validation | refuses an unknown status; refuses a value under REQUIRES_MEASUREMENT; refuses VERIFIED_MEASURED without a source |

Parameter status table (nothing is measured):

| Parameter | Status | Source |
| --- | --- | --- |
| wheel_radius_m | REQUIRES_MEASUREMENT | no wheel geometry identified from the official CAD |
| wheel_width_m | REQUIRES_MEASUREMENT | ditto |
| wheel_base_m | REQUIRES_MEASUREMENT | front/rear axle spacing not derivable |
| track_width_m | REQUIRES_MEASUREMENT | left/right spacing not derivable |
| steer_axis_height_m | REQUIRES_MEASUREMENT | kingpin geometry not identified |
| total_mass_kg | REQUIRES_MEASUREMENT | no mass in the CAD text |
| base_length_m | TEMP_PARAMETERIZED_PROXY (0.70 m) | project engineering baseline in docs/simulation/BADMINTON_ROBOT.md, NOT a Morph One value |
| base_width_m | TEMP_PARAMETERIZED_PROXY (0.55 m) | same |

Tests: `tests/simulation/rebot_b601_rs/test_morph_one_topology.py`, 8 tests, all passing locally
(run log in outputs/simulation/rebot_b601_rs/morph_one/logs/). They assert the interface, not the
geometry. **No joint is created, no proxy chassis is fabricated, and no wheel motion is claimed.**

Why no proxy articulation: the task forbids claiming a movable chassis without the real geometry,
and a proxy with invented wheel radii/positions would make every downstream motion number meaningless.

## Phase C - mounting B601-RS on Morph One: the real obstacle, measured

The arm asset's base is world-fixed by construction. From the converted USD
(`payloads/Physics/physics.usda`, line 308):

    def PhysicsFixedJoint "root_joint"
        physics:body0 = </ReBot_Arm_RS>                    # the arm root prim
        physics:body1 = </ReBot_Arm_RS/Geometry/base_link> # the arm base body
        localPos/Rot 0, identity

So `base_link` is pinned to the arm's own root prim (the standard `--fix-base` output), which is why
the earlier F1 work recorded that writing the root pose after the fact detaches the fixed base and the
robot falls through the floor. Consequences, stated plainly:

* Placing the arm on the chassis purely visually, or re-writing its root pose at runtime, **is not a
  physical mount**. Both are excluded.
* With the current asset there is no correct way to make the arm ride a moving chassis, because a
  PhysX articulation cannot be jointed to a body of another articulation.

Required design (not yet built, needs a re-converted asset):

1. Re-convert `ReBot_Arm_RS.urdf` **without** `--fix-base`, keeping `base_link` free.
2. Build one combined articulation whose root is the chassis base link, referencing the free-base arm,
   and author a `PhysicsFixedJoint` from `chassis_base_link` to the arm `base_link` at the measured
   `arm_mount` transform.
3. Only then can a test prove: arm base transform relative to the chassis base is constant while the
   chassis moves, the five rest-pose joints hold their targets, and no body separates.

The mount transform itself is REQUIRES_MEASUREMENT: there is no bracket geometry, no mounting plate in
the classified set, and no measurement record. Inventing one would violate the same rule as inventing
wheel radii.

## Phase D - minimal whole-machine verification: status of each acceptance item

| # | Item | Status | Evidence / reason |
| --- | --- | --- | --- |
| 1 | Real CAD loads, units correct | **partial** | units proven at text level (mm, SI_UNIT); no USD exists, so loading is not demonstrated |
| 2 | 8 semantic joints map correctly | **interface verified** | test_morph_one_topology.py (8 tests); no joints exist in simulation yet |
| 3 | Arm holds rest pose 5 s on a static base | **already verified elsewhere** | evidence/b601-rs-acceptance-173966b: worst error 0.013724 rad, standalone and scene identical |
| 4 | Individual steer / individual wheel / low-speed straight / in-place turn | **blocked** | requires the chassis articulation, which requires classified geometry |
| 5 | Mount transform preserved while the base moves | **blocked** | requires the combined articulation described above |
| 6 | No limit violation, no NaN/Inf, no through-floor, no rigid-body drift | **blocked for the chassis** | the arm-only checks already pass; chassis checks need the chassis |

Nothing in this round was accepted on the basis of a rendered screenshot; there are no chassis runs to
render because no chassis assembly exists yet.

## What was deliberately not done

* No fake wheels, no invented joint axes, no proxy chassis presented as Morph One.
* No chassis motion claim, no arm-on-moving-base claim, no PPO, no real-robot command.
* No modification of any acceptance rule or tolerance.
* The 115 MB official CAD file and any derived binary stay on the asset host; only the 5 KB text
  inventory, the interface module and the tests are committed.

## Deliverables in this branch

| Path | Content |
| --- | --- |
| management/morph_one_rs/PHASE_A_CAD_ASSETIZATION.md | CAD facts, entity census, extents, outlier rule, classification policy, conversion blocker |
| management/morph_one_rs/PHASE_BCD_BLOCKERS_AND_DESIGN.md | this document |
| simulation/badminton_scene/morph_one_chassis.py | 8-joint interface + parameter discipline (no geometry) |
| tests/simulation/rebot_b601_rs/test_morph_one_topology.py | 8 CPU tests, passing |
| outputs/simulation/rebot_b601_rs/morph_one/step_inventory.json | raw text-level inventory (5,064 B) |
| outputs/simulation/rebot_b601_rs/morph_one/logs/ | inventory run log and pytest log |
