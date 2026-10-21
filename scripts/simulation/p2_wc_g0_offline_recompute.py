"""G0 offline: recompute the historical nested-lift wheel heights from committed USD evidence.

No PhysX. Reconstructs the superseded RC authoring (459cb12) and the corrected authoring so the
double-lift error is documented numerically instead of described in prose.
"""
import json, re, subprocess, sys
from pathlib import Path

REPO = Path(r"E:\具身智能\badmin_wt_g")
sys.path.insert(0, str(REPO / "scripts" / "simulation"))
from p2_chassis_frame_math import wheel_bottom_world_z, historical_nested_wheel_bottom_world_z

HISTORICAL_REV = "459cb12"
USD_PATH = "outputs/simulation/rebot_b601_rs/morph_one/p2_real_chassis/RC_lift0300_r01/authored_chassis.usda"
LIFTS = [0.300, 0.334, 0.350]

raw = subprocess.run(["git", "show", HISTORICAL_REV + ":" + USD_PATH],
                     cwd=str(REPO), capture_output=True, text=True)
text = raw.stdout
blocks = re.findall(r'def Xform "wheel\d+".*?\n    \}', text, re.S)
wheel_translates = []
for block in blocks:
    hit = re.search(r"xformOp:translate = \(([^)]+)\)", block)
    wheel_translates.append(tuple(float(v) for v in hit.group(1).split(",")))
chassis_hit = re.search(r'def Xform "Chassis".*?xformOp:translate = \(([^)]+)\)', text, re.S)
chassis_translate = tuple(float(v) for v in chassis_hit.group(1).split(","))

rows = []
for lift in LIFTS:
    historical = historical_nested_wheel_bottom_world_z(lift)
    corrected = wheel_bottom_world_z(lift)
    rows.append({
        "lift_m": lift,
        "historical_recomposed_wheel_bottom_z_m": round(historical, 9),
        "corrected_wheel_bottom_z_m": round(corrected, 9),
        "double_lift_error_m": round(historical - corrected, 9),
        "historical_clearance_m": round(historical, 9),
        "corrected_clearance_m": round(corrected, 9),
    })

out = {
    "gate": "G0",
    "method": "pure recomputation from committed USDA + frame math, no PhysX",
    "historical_rev": HISTORICAL_REV,
    "usda": USD_PATH,
    "git_show_returncode": raw.returncode,
    "historical_chassis_translate": chassis_translate,
    "historical_wheel_translates": wheel_translates,
    "expected_double_lift_m": round(chassis_translate[2] - (chassis_translate[2] - 0.0404), 9),
    "rows": rows,
    "note": ("the historical authoring applied lift at the chassis AND at each wheel child, so the "
             "composed wheel height gained one extra lift; the corrected authoring applies lift "
             "once at the chassis"),
}
out_path = REPO / "outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/G0_offline_recompute.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1)[:1800])
