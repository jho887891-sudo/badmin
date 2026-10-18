# -*- coding: utf-8 -*-
"""Merge the per-mode F1 runs into the single evidence file the regression test reads."""
import json
from pathlib import Path

ROOT = Path("outputs/simulation/rebot_b601_rs")
runs = []
for mode in ("standalone", "scene"):
    p = ROOT / ("HOLD_run_" + mode + ".json")
    runs.append(json.loads(p.read_text(encoding="utf-8")))
doc = {
    "generated_by": "scripts/simulation/check_arm_rest_pose_hold.py",
    "host": "dgut@172.31.68.251 (Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2)",
    "acceptance": {
        "hold_seconds": 5.0,
        "physics_dt": runs[0]["physics_dt"],
        "steps": runs[0]["steps_total"],
        "max_hold_error_rad": 0.05,
    },
    "runs": runs,
}
(ROOT / "HOLD_rest_pose.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
print("merged", len(runs), "runs ->", ROOT / "HOLD_rest_pose.json")
for r in runs:
    print(r["name"], r["verdict"])
