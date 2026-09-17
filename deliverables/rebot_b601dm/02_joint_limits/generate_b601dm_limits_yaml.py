"""Generate the YAML mirror of the joint-limit table, so the two cannot drift apart."""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(".").resolve()))
from src.simulation.rebot_b601dm.joint_limits import JOINT_LIMITS, MOTOR_ASSIGNMENT, Provenance
from src.simulation.rebot_b601dm.torque_convention import DEFAULT_TORQUE_CONVENTION

def render(value):
    return "null" if value is None else repr(round(value, 7))

lines = [
    "# reBot B601-DM joint limits, mirroring src/simulation/rebot_b601dm/joint_limits.py.",
    "#",
    "# GENERATED - edit the Python table and regenerate; a test asserts the two agree.",
    "#",
    "# Every value carries `provenance` and, unless UNKNOWN, a `source`. A value tagged SIM_ONLY may not be",
    "# used to support a capability claim: the shipped asset permits 9.09x and 9.55x the real motor speeds",
    "# because the real rpm figures were used as rad/s.",
    "#",
    "# Candidate arm only: see management/rebot_b601dm/ARM_IDENTITY_DECISION.md. PiPER keeps the project",
    "# identity until this arm passes the Task 8 gates.",
    "",
    "robot: reBot_B601_DM",
    "identity: test_bench_candidate_arm",
    "reach_m: 0.767          # README_zh.md:176, B601-DM maximum",
    "payload_kg: 1.5         # README_zh.md:174",
    "recommended_radius_fraction: 0.70    # Performance_Testing_zh.md:86",
    "recommended_speed_fraction: 0.70     # Performance_Testing_zh.md:87",
    "mass_kg: 4.5            # README_zh.md:177, approximate",
    "",
    "# An experiment must record which torque convention it ran under. The shipped asset and the URDF carry",
    "# PEAK torques (27 and 7 N m) while the rated figures are 9 and 3 N m - a factor of three. The default",
    "# here is the conservative one, and resolve_torque_limit refuses to guess when asked without one.",
    "torque_convention:",
    "  default: {}".format(DEFAULT_TORQUE_CONVENTION.value),
    "  allowed: [peak, rated]",
    "  peak_torque_nm: {DM4340P: 27.0, DM4310: 7.0}    # what the shipped asset carries",
    "  rated_torque_nm: {DM4340P: 9.0, DM4310: 3.0}    # what the motor can sustain",
    "",
    "motors:",
]
for model, spec in (("DM4340P", "DM4340"), ("DM4310", "DM4310")):
    from src.simulation.rebot_b601dm import joint_limits as jl
    d = jl.DM4340 if spec == "DM4340" else jl.DM4310
    lines.append("  {}:".format(model))
    lines.append("    rated_torque_nm: {}".format(render(d["rated_nm"])))
    lines.append("    peak_torque_nm: {}".format(render(d["peak_nm"])))
    lines.append("    no_load_rpm: {}".format(render(d["rpm_no_load"])))
    lines.append("    rated_rpm: {}".format(render(d["rpm_rated"])))
    lines.append("    reduction_ratio: {}".format(d["ratio"]))
    lines.append("    provenance: CONFIRMED_REAL")
    lines.append("    source: {}".format(jl.MOTOR_SRC))
    lines.append("    caveat: official table documents -2EC; the BOM specifies V4")
lines.append("")
lines.append("joints:")
for name, s in JOINT_LIMITS.items():
    lines.append("  {}:".format(name))
    lines.append("    kind: {}".format(s.kind))
    lines.append("    motor: {}".format(MOTOR_ASSIGNMENT[name]))
    lines.append("    reduction_ratio: {}".format(s.reduction_ratio))
    for kind in ("angle", "velocity", "torque"):
        lim = getattr(s, kind)
        lines.append("    {}:".format(kind))
        lines.append("      value: {}".format(render(lim.value)))
        lines.append("      provenance: {}".format(lim.provenance.value))
        if lim.source:
            lines.append("      source: {}".format(lim.source.replace('"', "")))
        if lim.sim_value is not None:
            lines.append("      sim_value: {}".format(render(lim.sim_value)))
        if lim.sim_overshoot is not None:
            lines.append("      sim_overshoot: {}".format(round(lim.sim_overshoot, 4)))
        if lim.lower is not None:
            lines.append("      lower: {}".format(render(lim.lower)))
            lines.append("      upper: {}".format(render(lim.upper)))
        if lim.rated is not None:
            lines.append("      rated: {}".format(render(lim.rated)))
        if lim.is_peak:
            lines.append("      is_peak: true")

out = Path("configs/simulation/rebot_b601dm_joint_limits.yaml")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", out, "({} lines)".format(len(lines)))