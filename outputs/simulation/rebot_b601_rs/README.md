# B601-RS verification evidence

Every file here was produced by a real run on the GPU host; none of it is transcribed by hand.

| File | Produced by | What it proves |
| --- | --- | --- |
| VERIFY_usd_load.json | remote verify_rs_usd.py (Isaac Sim 6.0.1 + Isaac Lab) | the converted asset resolves, carries 35 meshes / 730,663 vertices, 9 rigid bodies, 8 joints, the official limits, per-link masses totalling 6.008504 kg, and moves under a position command within its limits |
| VERIFY_scene_load.json | remote scene_rs_check.py (spawns simulation/badminton_scene) | the migrated scene assembles the B601-RS at the configured mount, and records the OPEN rest-pose defect (all_within_limits false) |
| rest_pose_probe.json | remote rest_pose_probe.py | 8 candidate rest poses measured from link geometry; the chosen pose keeps every link above the base plane |
| conversion_config.yaml | Isaac Lab convert_urdf.py | the exact importer settings behind the USD |

| STATE_SYNC_dt_scaling.json | tools/merge_state_sync_evidence.py from 4 remote runs | KNOWN ISSUE: the reported joint velocity is not dq/dt - it equals the PhysX DOF velocity but the position does not change, and the value scales with the physics step (0.0367 / 0.0676 / 0.115 / 0.167 rad/s at 480 / 240 / 120 / 60 Hz) |
| STATE_SYNC_dt_1_<N>.json | remote state_sync.py | raw per-step-size probe behind the table above |
| MULTIENV_curve_<N>_probe_240hz.csv | check_multienv_hold.py --curve | 480 rows of raw 240 Hz joint POSITION and VELOCITY for the probe env |
| MULTIENV_hold.json | remote check_multienv_hold.py via run_multienv.sh | merged climb 2/4/16/32/64/128 envs: every env holds its own target (worst 0.0188 rad), limits respected, independent reset verified, throughput per count |
| MULTIENV_run_<N>.json | same harness, per count | per-env target, final joint positions/velocities, hold error, limit status, wrist height, independent-reset drift, performance |
| MULTIENV_curve_<N>.csv | same harness --curve (2, 16, 128) | joint angles over time for every env |
| MULTIENV_log_<N>.txt | same runs, full stdout | complete run logs for each env count |
| HOLD_rest_pose.json | remote check_arm_rest_pose_hold.py (bare arm + full scene) | the B601-RS holds its rest pose for 5 s in both assemblies: worst error 0.0423 rad, joint2/joint3 inside their official ranges, wrist 0.487 m above the mount, no body below the mount plane |
| HOLD_run_standalone.json / HOLD_run_scene.json | same harness, per-mode output | the raw per-mode runs that HOLD_rest_pose.json merges (joint names/indices, default_joint_pos, targets, velocities, limits, drive gains, checkpoints) |
| HOLD_curve_standalone.csv / HOLD_curve_scene.csv | same harness --curve | 60 Hz joint-angle curve over the full 5 s (301 rows) |

Regenerate the hold evidence with:

```bash
ssh dgut@172.31.68.251 bash /home/T7/dgut/robot_sim/run_hold_f1.sh   # writes migration_check/hold_f1/
python tools/merge_hold_evidence.py                                # after copying the two run_*.json here
```

VERIFY_scene_load.json is the **pre-fix** record of the F1 defect (rest_error_rad, all_within_limits:
false) and is kept on purpose; do not read it as a current result.

Regenerate the conversion/USD evidence (host jxxy = dgut@172.31.68.251):

```bash
ssh dgut@172.31.68.251 bash /home/T7/dgut/robot_sim/run_verify_rs.sh
```

The scripts live at /home/T7/dgut/robot_sim/{verify_rs_usd,scene_rs_check,rest_pose_probe}.py.
Local copies of the two verification scripts are kept in this repository as _scratch_* files only
while the migration is in flight; they are not deliverables.

Assets themselves stay on the host under
/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/ (64 MB staged source + 20 MB USD).
