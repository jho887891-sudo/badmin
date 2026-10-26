# P0: B601-RS fixed-base scene and racket contact-frame adapter

Status: **implementation scaffold; physical mount and Morph One chassis integration remain BLOCKED**.

## Scope and provenance

This branch is based on p6/dynamic-fk at a0a0fb3b18928f4284dab7980111c21b67ee1f6e. It imports the existing single-environment B601-RS scene route from feature/morph-one-rs-integrated-assembly, plus a fail-closed visual-racket adapter and FK contact-pose composition. No files on main are changed; no robot deployment or GPU run is performed by this change.

The existing scene's B601-RS direct-articulation path is single-environment only. The racket visual is opt-in: include_racket=False by default. Collision geometry is separately opt-in with include_racket_collision=True (which requires include_racket=True). If enabled, callers must pass an explicit mount record with status MEASURED or SIMULATION_ASSUMPTION, translation_m, quaternion_wxyz, and source. The adapter refuses UNKNOWN, missing values, non-unit quaternions, missing source provenance, duplicate mount prims, invalid collision-proxy geometry, and absent assets before simulation initialization where possible.

## What is computed

The contact-pose module composes

    T_world_contact = T_world_arm_root
                      * T_arm_root_gripper_end(q)
                      * T_gripper_end_racket_root
                      * T_racket_root_contact

The B601-RS FK supplies only T_arm_root_gripper_end(q). The remaining transforms are inputs with explicit evidence labels. Unknown transforms are never replaced by identity.

The helper racket_stringbed_center_proxy derives the midpoint of the committed visual-only string-bed bounding box. For the currently committed proxy, that is approximately (0, 0, 0.559585) metres in racket_root. It is a geometry-derived centre proxy, NOT a measured sweet spot. Using it requires allow_proxy_geometry=True and the result remains PROXY_DERIVED.

## Known blockers

1. world_from_arm_root: physical world/chassis/arm-root relation is not calibrated. The scene-layout mount is a temporary parameterized proxy.
2. gripper_end_from_racket_root: the actual adapter/handle clamp offset and orientation are UNKNOWN.
3. racket_root_from_contact: the code can derive a visual string-bed centre proxy, but no physical contact/sweet-spot calibration exists.
4. The visual attachment is render-only by default. With include_racket_collision=True, the 43-shape engineering proxy is authored under the link6 rigid-body hierarchy, so contact geometry can be exercised without inventing a separate fixed-joint anchor. The proxy does not add racket mass/inertia and is not validated physical contact dynamics.
5. The official Morph One CAD has not been classified into a verified chassis geometry. This branch does not fabricate or claim a physical Morph One chassis model.
6. The B601-RS scene's InteractiveScene drive-binding issue is worked around by direct articulation; formal create_scene remains num_envs=1.

## Automated CI evidence (2026-10-26)

- Source commit: `2ed5db8175a55171f3a8170e945509538e1cd8b2`
- GitHub Actions: [run 40285332369](https://github.com/jho887891-sudo/badmin/actions/runs/40285332369)
- Environment: Ubuntu runner, Python 3.10.22, NumPy 2.2.6, pytest 8.4.2, OpenUSD `usd-core==25.11`
- Result: **128 passed, 1 skipped, 0 failed**; the Isaac Lab smoke script also passed `py_compile`.
- The one skip is `tests/simulation/p6_joint_anchors_test.py:507`: the runner does not contain the read-only `physics.usda` layer copy, and the test uses the committed SHA-256-pinned anchors instead.
- This validates Python kinematics, P6 frame/joint/proxy regression checks and OpenUSD in-memory authoring. It does **not** validate Isaac Sim / PhysX runtime behavior, a GPU simulation, the real racket mount, or shuttle-racket contact dynamics.

## Reproducible verification

Install the CI-only dependencies and run the same regression suite:

    python -m pip install "numpy>=1.24,<3" "pytest>=7,<9" "usd-core==25.11"
    python -m pytest -q -rs \
      tests/simulation/rebot_b601_rs/test_scene_arm_route.py \
      tests/simulation/rebot_b601_rs/test_racket_mount_adapter.py \
      tests/simulation/p6_racket_contact_kinematics_test.py \
      tests/simulation/p6_dynamic_fk_test.py \
      tests/simulation/p6_frame_tree_test.py \
      tests/simulation/p6_joint_anchors_test.py \
      tests/simulation/p6_racket_proxy_test.py \
      tests/simulation/rebot_b601_rs/test_racket_usd_authoring.py

For the next gate, run the dedicated Isaac Lab smoke script on the target host after verifying `ISAACLAB_ROOT`, `BADMINTON_ROOT`, and the actual B601-RS USD path. The test-only identity mount is deliberately explicit and is **not** a physical calibration:

    "$ISAACLAB_ROOT/isaaclab.sh" -p "$BADMINTON_ROOT/scripts/simulation/p0_b601_racket_scene_smoke.py" \
      --headless --robot-usd "$REBOT_RS_USD" --test-identity-mount --steps 120 \
      --output "$BADMINTON_ROOT/outputs/simulation/rebot_b601_rs/morph_one/p6_frames/evidence/p0_racket_scene_smoke_20261026.json"

The smoke script runs only a fixed-base simulation of the B601-RS and checks that the racket visual and 43 collision proxies survive reset and stepping. It does not create or deploy a real robot, does not include the Morph One chassis or a shuttle, and does not claim physical contact dynamics. No target-host Isaac Sim run is claimed yet.
