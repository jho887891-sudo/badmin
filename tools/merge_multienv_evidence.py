# -*- coding: utf-8 -*-
"""Merge the per-count multi-env runs into the single evidence file the gate test reads."""
import json
from pathlib import Path

ROOT = Path("outputs/simulation/rebot_b601_rs")
runs = []
for n in (2, 4, 16, 32, 64, 128):
    p = ROOT / ("MULTIENV_run_%d.json" % n)
    if p.is_file():
        runs.append(json.loads(p.read_text(encoding="utf-8")))
doc = {
    "generated_by": "scripts/simulation/check_multienv_hold.py",
    "host": "dgut@172.31.68.251 (Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2)",
    "assembly": "direct Articulation per env (simulation/badminton_scene/multienv_arm.py)",
    "acceptance": {"hold_seconds": 2.0, "max_hold_error_rad": 0.05,
                   "climb_order": [2, 4, 16, 32, 64, 128]},
    "runs": runs,
}
(ROOT / "MULTIENV_hold.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print("merged", len(runs), "runs")
for r in runs:
    v = r["verdict"]
    print(r["num_envs"], "hold", v["all_envs_hold"], "worst", v["worst_hold_error_rad"],
          "limits", v["all_within_limits"], "reset", v["independent_reset_ok"],
          "throughput", r["performance"]["steps_per_second"], "steps/s")
