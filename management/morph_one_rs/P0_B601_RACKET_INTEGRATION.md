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

- Source commit: `b22c1e753174ed827cff311d39277aff9aa57420`
- GitHub Actions: [run 40286480569](https://github.com/jho887891-sudo/badmin/actions/runs/40286480569)
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

The smoke script runs only a fixed-base simulation of the B601-RS. In addition to checking that the racket visual and all 43 collision proxies survive reset and stepping, it checks that link6 has a rigid-body API, every collider resolves to link6 as its nearest rigid-body ancestor, the referenced racket root matches the mount transform within 1e-5 m, and the racket mount does not create a separate rigid body. These are runtime assertions in the script, not yet target-host results. It does not create or deploy a real robot, does not include the Morph One chassis or a shuttle, and does not claim physical contact dynamics. No target-host Isaac Sim run is claimed yet.


## PR #63 GitHub-direct remediation and automated acceptance gate (2026-10-27)

The target host reported a successful 120-step simulation on *a different local commit*:
`9ff4ab8adaeb24c95327df6ee07d55d1ade61ec9`, which never reached GitHub
because the target host's HTTPS Git traffic was directed at an unavailable
`127.0.0.1:10080` proxy. Its JSON evidence is only on the target host
(`p0_racket_scene_smoke_formal_20261027T074316Z.json`, reported SHA256
`70b2080d91709fcf40a4cb69762cae3321c816aa88e8a53089a6896eeec2e088`).
Do **not** relabel that run as having exercised this PR's new code commit.

The GitHub-side equivalent fixes were separately implemented and verified by CI:

1. `scripts/simulation/p0_b601_racket_scene_smoke.py` authors the racket only
   after `sim_utils.make_uninstanceable(REBOT_RS_PRIM_PATH, stage)`, without
   modifying the source USD asset.
2. A native Isaac Sim `app.close()` may terminate with `exit_group(0)`;
   the original code lost its report. The PR now uses
   `try: write_report_exclusive(output, report) finally: app.close()` so
   the structured PASS/FAIL evidence is persisted before native shutdown.
3. `tests/simulation/rebot_b601_rs/test_p0_smoke_regressions.py` executes the
   actual Python finalizer body with a simulated non-returning close and guards
   mount authoring order. RED: [CI #40472083100](https://github.com/jho887891-sudo/badmin/actions/runs/40472083100)
   **5 failed**, GREEN: [CI #40472215330](https://github.com/jho887891-sudo/badmin/actions/runs/40472215330)
   **133 passed, 1 skipped**.
4. `tools/p0/verify_racket_smoke.py` independently verifies the final
   smoke JSON from *outside* Isaac Sim. It refuses missing reports, incomplete
   or nonfinite data, status!=PASS, wrong step/count/ancestry checks, or any
   unvalidated physical mount claim. RED: [CI #40472606837](https://github.com/jho887891-sudo/badmin/actions/runs/40472606837)
   **19 failed** before the validator existed; GREEN:
   [CI #40472770599](https://github.com/jho887891-sudo/badmin/actions/runs/40472770599)
   **152 passed, 1 skipped**.

### Direct external report check (no Isaac dependency)

```bash
python tools/p0/verify_racket_smoke.py --report /absolute/path/to/new_120step_report.json
```

The checker prints `PASS` and returns 0 only if every required field in the
JSON confirms the fixed-base B601-RS + **proxy**-racket smoke contract.
A failure/missing report returns 2. The checker is not a substitute for running
the actual Isaac Lab smoke with the **current GitHub PR SHA**.

### Final outstanding gate

Perform a new 120-step Isaac Lab run from a **fresh clean checkout of this PR's
actual head** on `jxxy`, keeping `/home/dgut/badmin_project_pr63_integration`
(the host's original local `9ff4ab8` worktree) untouched. Pin the newly checked
out commit, preserve the new report/log hashes, then run the verifier above.
There is no GitHub-side capability here to execute commands on the target host.
The real racket mount and shuttle-racket impact response remain UNKNOWN.
