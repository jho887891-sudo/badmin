"""Landing analysis for a gravity run: per-step z / v_z / contact / tilt / penetration.

Turns a single peak-contact number into a support story.  Works on both the G5 chassis
telemetry (raw_poses.jsonl with max_contact_force_n only) and the G6 variant telemetry
(adds tilt_rad, bodies, contact_by_sensor_n).  Pure numpy, no Isaac, no GPU.  Read-only.

Usage:
    python p2_wc_landing_analysis.py --run-dir <dir> [--wheel-bottom-offset -0.334]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

TAIL_STEPS = 30


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="landing/support analysis")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wheel-bottom-offset", type=float, default=-0.334,
                        help="wheel bottom = chassis z + offset (identity rotation)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    frames = load_jsonl(os.path.join(run_dir, "raw_poses.jsonl"))
    result_path = os.path.join(run_dir, "result.json")
    result = {}
    if os.path.isfile(result_path):
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)

    z = np.array([frame["pos"][2] for frame in frames])
    vz = np.array([frame["lin_vel"][2] for frame in frames])
    speed = np.array([float(np.linalg.norm(frame["lin_vel"])) for frame in frames])
    forces = [frame.get("max_contact_force_n") for frame in frames]
    known = [(frame["step"], value) for frame, value in zip(frames, forces)
             if value is not None]
    peak = max((value for _, value in known), default=None)
    peak_steps = [step for step, value in known if peak is not None and value >= peak - 1e-9]
    contact_steps = [step for step, value in known if value is not None and value > 1.0]
    first_contact = contact_steps[0] if contact_steps else None
    tilt = [frame.get("tilt_rad") for frame in frames]
    tilt_known = [value for value in tilt if value is not None]

    bottom = z + args.wheel_bottom_offset
    tail = slice(max(0, len(frames) - TAIL_STEPS), len(frames))
    tail_speed = float(np.max(speed[tail]))
    tail_z_span = float(np.max(z[tail]) - np.min(z[tail]))
    tail_penetration = float(max(0.0, -float(np.min(bottom[tail]))))
    bounce_count = int(np.sum(np.abs(np.diff(np.sign(vz[tail]))) > 0))
    nonzero_vz_after_contact = (
        int(np.sum(np.abs(vz[first_contact:]) > 0.02)) if first_contact is not None else None)

    analysis = {
        "run_id": result.get("run_id", os.path.basename(run_dir)),
        "scene_variant": result.get("scene_variant"),
        "gate_result": result.get("gate_result"),
        "steps": len(frames) - 1,
        "contact": {
            "peak_contact_force_n": peak,
            "peak_at_steps": peak_steps,
            "first_contact_step": first_contact,
            "contact_steps_gt_1n": len(contact_steps),
            "contact_fraction": round(len(contact_steps) / max(1, len(frames) - 1), 6),
            "contact_sensor_available": bool(known),
        },
        "support": {
            "final_chassis_z_m": round(float(z[-1]), 9),
            "final_speed_m_s": round(float(speed[-1]), 9),
            "final_vz_m_s": round(float(vz[-1]), 9),
            "tail_window_steps": TAIL_STEPS,
            "tail_speed_max_m_s": round(tail_speed, 9),
            "tail_z_span_m": round(tail_z_span, 9),
            "tail_bounce_sign_changes": bounce_count,
            "steps_with_|vz|>0.02_after_first_contact": nonzero_vz_after_contact,
            "min_chassis_z_m": round(float(np.min(z)), 9),
            "max_chassis_z_m": round(float(np.max(z)), 9),
            "initial_chassis_z_m": round(float(z[0]), 9),
        },
        "penetration": {
            "wheel_bottom_offset_m": args.wheel_bottom_offset,
            "initial_wheel_bottom_m": round(float(bottom[0]), 9),
            "min_wheel_bottom_m": round(float(np.min(bottom)), 9),
            "max_penetration_any_step_m": round(float(max(0.0, -float(np.min(bottom)))), 9),
            "tail_max_penetration_m": round(tail_penetration, 9),
        },
        "tilt": {
            "measured": bool(tilt_known),
            "max_tilt_rad": (round(float(max(tilt_known)), 9) if tilt_known else None),
        },
        "per_step": [
            {"step": frame["step"],
             "z": round(float(z[index]), 6),
             "vz": round(float(vz[index]), 6),
             "force_n": forces[index],
             "tilt_rad": (round(frame["tilt_rad"], 6)
                          if frame.get("tilt_rad") is not None else None)}
            for index, frame in enumerate(frames)
        ],
    }
    out_path = args.out or os.path.join(run_dir, "landing_analysis.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=1)
    compact = {key: analysis[key] for key in
               ("run_id", "gate_result", "contact", "support", "penetration", "tilt")}
    print(json.dumps(compact, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
