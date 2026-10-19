# -*- coding: utf-8 -*-
"""Merge the step-size probe into the recorded known issue (velocity channel vs position)."""
import json
from pathlib import Path

root = Path("outputs/simulation/rebot_b601_rs")
rows = []
for d in (480, 240, 120, 60):
    p = root / ("STATE_SYNC_dt_1_%d.json" % d)
    if not p.is_file():
        continue
    s = json.loads(p.read_text(encoding="utf-8"))["summary"]
    i2 = s["joint_names"].index("joint2")
    i3 = s["joint_names"].index("joint3")
    rows.append({
        "physics_hz": d,
        "physics_dt": s["physics_dt"],
        "reported_vel_joint2_rad_s": s["data_vel_joint2_last"],
        "reported_vel_joint3_rad_s": s["data_vel_joint3_last"],
        "numeric_dqdt_joint2_rad_s": s["numeric_vel_joint2_last"],
        "numeric_dqdt_joint3_rad_s": s["numeric_vel_joint3_last"],
        "position_delta_joint2_rad": s["pos_delta_joint2"],
        "position_delta_joint3_rad": s["pos_delta_joint3"],
        "data_equals_physx": abs(s["data_vel_joint2_last"] - (s.get("physx_vel_joint2_last") or 0.0)) < 1e-9,
    })
doc = {
    "question": "is the joint velocity channel consistent with the joint position channel?",
    "answer": "no",
    "method": "MultiEnvArm, 2 envs, gains 300/20, rest target, 240 settle steps then 40 measured steps",
    "finding": [
        "IsaacLab data.joint_vel equals the PhysX DOF velocity, so the buffer is not stale.",
        "The numeric derivative of the same DOF position is exactly 0 while the reported velocity is not.",
        "The reported velocity grows with the physics step (see rows), i.e. it behaves like a per-step"
        " residual rather than a rate; it is therefore not dq/dt.",
    ],
    "rows": rows,
    "rows_sorted_by_hz": sorted(rows, key=lambda r: r["physics_hz"]),
}
out = root / "STATE_SYNC_dt_scaling.json"
out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
for r in sorted(rows, key=lambda r: r["physics_hz"]):
    print(" %4d Hz  vel2=%8.5f  numeric2=%9.6f  pos_delta2=%9.6f" % (
        r["physics_hz"], r["reported_vel_joint2_rad_s"], r["numeric_dqdt_joint2_rad_s"],
        r["position_delta_joint2_rad"]))
