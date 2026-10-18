# B601-RS verification evidence

Every file here was produced by a real run on the GPU host; none of it is transcribed by hand.

| File | Produced by | What it proves |
| --- | --- | --- |
| VERIFY_usd_load.json | remote verify_rs_usd.py (Isaac Sim 6.0.1 + Isaac Lab) | the converted asset resolves, carries 35 meshes / 730,663 vertices, 9 rigid bodies, 8 joints, the official limits, per-link masses totalling 6.008504 kg, and moves under a position command within its limits |
| VERIFY_scene_load.json | remote scene_rs_check.py (spawns simulation/badminton_scene) | the migrated scene assembles the B601-RS at the configured mount, and records the OPEN rest-pose defect (all_within_limits false) |
| rest_pose_probe.json | remote rest_pose_probe.py | 8 candidate rest poses measured from link geometry; the chosen pose keeps every link above the base plane |
| conversion_config.yaml | Isaac Lab convert_urdf.py | the exact importer settings behind the USD |

Regenerate (asset paths are on the GPU host, host jxxy = dgut@172.31.68.251):

```bash
ssh dgut@172.31.68.251 bash /home/T7/dgut/robot_sim/run_verify_rs.sh
```

The scripts live at /home/T7/dgut/robot_sim/{verify_rs_usd,scene_rs_check,rest_pose_probe}.py.
Local copies of the two verification scripts are kept in this repository as _scratch_* files only
while the migration is in flight; they are not deliverables.

Assets themselves stay on the host under
/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/ (64 MB staged source + 20 MB USD).
