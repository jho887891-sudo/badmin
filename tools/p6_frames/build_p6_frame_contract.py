#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6-02 / P6-04: emit the machine-readable frame tree and camera metadata contract.

Rule this file exists to enforce: an edge either carries a transform WITH the file and
sha256 it came from, or it is UNKNOWN and names the missing datum.  There is no third
state, and no field is ever filled with a plausible number.  Anything a placeholder
config says is recorded as a REJECTED placeholder so a later reader cannot mistake it
for calibration.

    python tools/p6_frames/build_p6_frame_contract.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
EVIDENCE = OUT / "evidence"

RACKET_USD_REL = "assets/third_party/racket_visual.usd"
ROBOT_JOINTS_REL = "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/evidence/robot_usd_joint_chain.json"
RACKET_AUDIT_REL = "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/evidence/racket_usd_audit.json"

CAMERA_SOURCES = {
    "scene_layout": ("simulation/badminton_scene/scene_layout.py", 87),
    "camera_rig_cfg": ("simulation/badminton_scene/camera_rig_cfg.py", 1),
    "no_calibrated_fx": ("management/shuttle_detection/plan5_input_requirements.md", 57),
    "proxy_fx_700": ("docs/ALGORITHM_AUDIT.md", 56),
}


def sha256_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def quat_wxyz_to_matrix(q) -> list:
    """USD / Isaac-usd convention: (w, x, y, z). Normalised here, not assumed normal."""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_xyzw_to_matrix(q) -> list:
    """The OTHER convention, kept as a separate function on purpose: xyzw never silently
    flows into the wxyz decoder above."""
    x, y, z, w = [float(v) for v in q]
    return quat_wxyz_to_matrix((w, x, y, z))


def det3(m) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def unknown(missing: str, would_come_from: str, note: str = "") -> dict:
    edge = {
        "status": "UNKNOWN",
        "translation_m": None,
        "rotation_quat": None,
        "rotation_matrix": None,
        "missing_datum": missing,
    }
    if would_come_from:
        edge["would_come_from"] = would_come_from
    if note:
        edge["note"] = note
    return edge


def asserted(translation, quat_wxyz, src_file, src_sha, locator, note="") -> dict:
    return {
        "status": "ASSERTED_BY_ASSET",
        "translation_m": [float(v) for v in translation],
        "rotation_quat": {"order": "wxyz", "values": [float(v) for v in quat_wxyz]},
        "rotation_matrix": quat_wxyz_to_matrix(quat_wxyz),
        "source": {"file": src_file, "sha256": src_sha, "locator": locator},
        "note": note,
    }


def build_frame_tree() -> dict:
    joints = load(ROBOT_JOINTS_REL)
    audit = load(RACKET_AUDIT_REL)
    phys_rel = "payloads/Physics/physics.usda"
    base_rel = "payloads/base.usda"
    phys_sha = joints["asset"]["layers"][phys_rel]["sha256"]
    base_sha = joints["asset"]["layers"][base_rel]["sha256"]
    arm_sha = joints["asset"]["layers"]["ReBot_Arm_RS.usda"]["sha256"]
    racket_sha = sha256_of(RACKET_USD_REL)

    # The racket asset must still be the bytes the audit described.
    if racket_sha != audit["usd"]["sha256"]:
        raise SystemExit("racket usd sha256 changed since the audit; rerun audit_racket_usd.py")

    by_name = {j["prim"]: j for j in joints["joints"]}
    chain = []
    for name, link_child in (
        ("joint1", "link1"), ("joint2", "link2"), ("joint3", "link3"),
        ("joint4", "link4"), ("joint5", "link5"), ("joint6", "link6"),
    ):
        j = by_name[name]
        chain.append({
            "from": j["body0"].split("/")[-1],
            "to": link_child,
            "edge_kind": "revolute_joint",
            "joint_name": name,
            "joint_axis_in_child": j["axis"],
            "joint_limits_rad": [j["limit_lower_rad"], j["limit_upper_rad"]],
            "joint_limits_as_authored_deg": [j["limit_lower_as_authored"], j["limit_upper_as_authored"]],
            "link_prim_path": j["body1"],
            "transform_link0_from_link1": asserted(
                j["local_pos0"], j["local_rot0_wxyz"], phys_rel, phys_sha,
                "%s / physics:localPos0 + physics:localRot0" % name,
                "authored joint frame: child pose with joint coordinate = 0",
            ),
        })

    gripper_end = joints["named_frames"][0]

    edges = [
        {
            "edge": 1,
            "from": "world",
            "to": "chassis_morph_one",
            "edge_kind": "pose",
            **unknown(
                "the Morph One chassis world pose (x, y, yaw) for the actual deployment",
                "a metrology record of the robot's placement on the court, or the runtime odometry "
                "plus the court registration procedure; neither is committed",
                "simulation/badminton_scene/scene_layout.py ROBOT world_pos=(-1.60,0.00,0.00) is an "
                "ENGINEERING_V0_1 scene placeholder, not a measurement of the real rig.",
            ),
        },
        {
            "edge": 2,
            "from": "chassis_morph_one",
            "to": "arm_base_link",
            "edge_kind": "mount",
            **unknown(
                "T_chassis_armbase: the bolt-pattern offset and yaw of the B601-RS base on the Morph One top plate",
                "a CAD or hand-measured mount record for THIS chassis + THIS arm; "
                "management/morph_one_rs/ holds the chassis CAD provenance but no arm mount transform",
                "scene_layout.py PIPER mount_rel=(0,0,0.30) belongs to the PiPER arm and must not be reused here.",
            ),
        },
        {
            "edge": 3,
            "from": "arm_root_prim",
            "to": "arm_base_link",
            "edge_kind": "root_joint_identity",
            "joint_name": "root_joint",
            **asserted(
                by_name["root_joint"]["local_pos0"], by_name["root_joint"]["local_rot0_wxyz"],
                phys_rel, phys_sha, "root_joint / physics:localPos0 + physics:localRot0",
                "fixed joint /ReBot_Arm_RS -> base_link, identity; establishes that the arm root prim "
                "IS the base_link frame, so no naming gap exists at the arm boundary.",
            ),
        },
    ]

    idx = 4
    for seg in chain:
        edges.append({
            "edge": idx,
            "from": seg["from"],
            "to": seg["to"],
            "edge_kind": seg["edge_kind"],
            "joint_name": seg["joint_name"],
            "joint_axis_in_child": seg["joint_axis_in_child"],
            "joint_limits_rad": seg["joint_limits_rad"],
            "joint_limits_as_authored_deg": seg["joint_limits_as_authored_deg"],
            "link_prim_path": seg["link_prim_path"],
            **seg["transform_link0_from_link1"],
        })
        idx += 1

    for jaw, joint in (("gripper_left", "gripper_joint1"), ("gripper_right", "gripper_joint2")):
        j = by_name[joint]
        edges.append({
            "edge": idx,
            "from": "link6",
            "to": jaw,
            "edge_kind": "prismatic_jaw",
            "joint_name": joint,
            "joint_limits_as_authored_m": [j["limit_lower_as_authored"], j["limit_upper_as_authored"]],
            "mount_allowed": False,
            "mount_rejection": (
                "the jaws slide (0..%s m on %s), so a racket clamped by them is not a fixed tool; "
                "src/simulation/rebot_b601dm/racket.py:LOCKED_JOINTS already refuses them"
                % (j["limit_upper_as_authored"], joint)
            ),
            **asserted(
                j["local_pos0"], j["local_rot0_wxyz"], phys_rel, phys_sha,
                "%s / physics:localPos0 + physics:localRot0" % joint,
            ),
        })
        idx += 1

    edges.append({
        "edge": idx,
        "from": "link6",
        "to": "gripper_end",
        "edge_kind": "authored_frame",
        "rigid_body": False,
        "physical_meaning": (
            "an instanceable Xform child of link6 with no rigid body and no joint: it is a GEOMETRIC "
            "frame (0.16621 m down link6 +Z), not a physics link. Using it as a body in a dynamics "
            "graph would silently create a static child; the dynamics body stays link6."
        ),
        **asserted(
            gripper_end["xformOp:translate"], gripper_end["xformOp:orient_wxyz"], base_rel, base_sha,
            'def Xform "gripper_end" / xformOp:translate + xformOp:orient',
        ),
    })
    idx += 1

    edges.append({
        "edge": idx,
        "from": "gripper_end",
        "to": "racket_mount",
        "edge_kind": "tool_mount",
        **unknown(
            "T_flange_tcp (a.k.a. T_link6_tcp): where the racket handle axis leaves the flange and how it is clocked about that axis",
            "configs/racket.yaml piper_attachment.mount_transform.translation_m / quaternion_xyzw, which are null and "
            "carry status REQUIRES_MEASUREMENT; filling them needs a physical measurement of the printed adapter on the "
            "installed tool, plus docs/RACKET.md:117 T_link6_tcp procedure",
            "the arm side of this edge is asserted (link6/gripper_end); the tool side is the gap.",
        ),
    })
    idx += 1

    edges.append({
        "edge": idx,
        "from": "racket_mount",
        "to": "racket_root",
        "edge_kind": "asset_placement",
        **unknown(
            "how deep the handle butt sits inside the mount, and the clocking of the racket's +X face normal about the handle axis",
            "a measurement of the physical assembly (handle insertion depth + face normal direction). The asset cannot "
            "answer it: /RacketVisual origin is the handle butt at z=0, which is a point on the tool, not the mating face",
            "assets/third_party/racket_visual.usd has no mount prim, no joint and no flange geometry "
            "(evidence/racket_usd_audit.json flange_or_end_link_in_asset=false).",
        ),
    })
    idx += 1

    edges.append({
        "edge": idx,
        "from": "racket_root",
        "to": "racket_contact_frame",
        "edge_kind": "derived_frame",
        **unknown(
            "the measured head centre / sweet-spot centre in racket-root coordinates",
            "geometry measured on the real strung racket, or a fitted ellipse centre on the collision proxy; "
            "simulation/badminton_scene/racket_cfg.py head_center_from_tcp=0.50 is explicitly labelled approx "
            "and configs/racket.yaml measured_unit is REQUIRES_MEASUREMENT",
            "a value CAN be read off the visual mesh (string-bed bbox z 0.44391..0.67526 m), but the visual is 682.7 mm "
            "long versus 675 mm for the real racket, so that centre is not the real racket's centre and is not asserted here.",
        ),
    })

    nodes = {}
    for e in edges:
        for key in ("from", "to"):
            nodes.setdefault(e[key], []).append(e["edge"])
    node_records = [
        {"node": "world", "definition": "right-handed, Z up, origin at the court net centre on the floor plane (simulation/badminton_scene/scene_layout.py:16-20)"},
        {"node": "chassis_morph_one", "definition": "Morph One chassis body frame, Z up, +X forward"},
        {"node": "arm_root_prim", "definition": "the defaultPrim /ReBot_Arm_RS of the shipped RS USD (the stage root the arm is referenced at)"},
        {"node": "arm_base_link", "definition": "/ReBot_Arm_RS/Geometry/base_link in the shipped RS USD"},
    ]
    for name in ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_left", "gripper_right"):
        node_records.append({"node": name, "definition": "/ReBot_Arm_RS/Geometry/base_link/" + "/".join(
            ["link%d" % i for i in range(1, int(name[-1]) + 1)] if name.startswith("link") else
            ["link1", "link2", "link3", "link4", "link5", "link6", name])})
    node_records.append({"node": "gripper_end", "definition": "geometric frame under link6, translate (0,0,0.16621) m"})
    node_records.append({"node": "racket_mount", "definition": "flange mating face / handle clamp axis - position UNKNOWN"})
    node_records.append({"node": "racket_root", "definition": "/RacketVisual in assets/third_party/racket_visual.usd: +Z handle->head, +X face normal toward the opponent, +Y = Z x X, origin at the handle butt"})
    node_records.append({"node": "racket_contact_frame", "definition": "centre of the racket face - position UNKNOWN"})

    assumed = [e for e in edges if e["status"] == "UNKNOWN"]
    return {
        "schema": "p6_frame_tree/1",
        "generated_by": "tools/p6_frames/build_p6_frame_contract.py",
        "scope": "world -> chassis -> arm base -> arm links (by name, from the shipped USD) -> end link -> racket mount -> racket",
        "conventions": {
            "length_unit": "metre",
            "handedness": "right-handed",
            "up_axis_of_source_assets": "Z (both the RS USD and the racket USD) - no axis swap is applied anywhere in this contract",
            "quaternion_order_by_source": {
                "usd_physics_and_usdgeom": "wxyz",
                "gltf_glb_json": "xyzw",
                "isaac_body_quat_w": "xyzw",
                "configs/racket.yaml:mount_transform.quaternion_xyzw": "xyzw",
            },
            "quaternion_rule": (
                "SOURCE-TAGGED. Every rotation in this file is written as {order, values} and the order is decoded by a "
                "dedicated function per source. USD wxyz and Isaac xyzw must never share one decoder or one helper."
            ),
            "degree_radian_rule": "USD Physics revolute limits are authored in DEGREES; both the authored value and the radian value are recorded per joint.",
        },
        "nodes": node_records,
        "edges": edges,
        "summary": {
            "edges_total": len(edges),
            "asserted_by_asset": sum(1 for e in edges if e["status"] == "ASSERTED_BY_ASSET"),
            "assumptions_or_unknown": len(assumed),
            "unknown_edge_ids": [e["edge"] for e in assumed],
            "unknown_edge_list": [
                {"edge": e["edge"], "from": e["from"], "to": e["to"], "missing_datum": e["missing_datum"]}
                for e in assumed
            ],
            "verdict": (
                "PASS for the arm chain (base_link..link6 + gripper_end, by name, from sha256-pinned USD); "
                "UNKNOWN for every transform that crosses a mount or a tool boundary"
            ),
        },
        "evidence": {
            "robot_usd": {
                "file": ROBOT_JOINTS_REL,
                "layers": joints["asset"]["layers"],
                "metersPerUnit": joints["asset"]["metersPerUnit"],
                "kilogramsPerUnit": joints["asset"]["kilogramsPerUnit"],
                "upAxis": joints["asset"]["upAxis"],
                "remote_root": joints["asset"]["remote_root"],
            },
            "racket_usd": {
                "file": RACKET_USD_REL,
                "sha256": racket_sha,
                "metersPerUnit": audit["usd"].get("metersPerUnit"),
                "upAxis": audit["usd"].get("upAxis"),
                "collision_count": audit["usd"].get("collision_count"),
                "audit": RACKET_AUDIT_REL,
            },
        },
    }


# Mount(FLU) -> optical(RDF): the columns are the mount axes expressed in optical
# coordinates. Verified by hand: X_mount->(0,0,1), Y_mount->(-1,0,0), Z_mount->(0,-1,0),
# det = +1.  One constant, decoded once, so the quaternion and the matrix cannot drift apart.
MOUNT_TO_OPTICAL_WXYZ = (0.5, 0.5, -0.5, 0.5)


def build_camera() -> dict:
    layout_sha = sha256_of(CAMERA_SOURCES["scene_layout"][0])
    rig_sha = sha256_of(CAMERA_SOURCES["camera_rig_cfg"][0])
    return {
        "schema": "p6_camera_metadata/1",
        "generated_by": "tools/p6_frames/build_p6_frame_contract.py",
        "device": {
            "model": "Intel RealSense D455",
            "model_provenance": {
                "status": "UNKNOWN",
                "why": "the model name appears only in task text and in dataset-scope lists such as "
                       "configs/tiny_recovery_yolo26s_v1.yaml:40 ('D455 hard negatives'); no committed device record "
                       "(serial number, firmware, rs-enumerate-devices output) exists in this repository",
                "missing_datum": "the serial number and firmware version of the unit actually mounted",
            },
        },
        "fields": {
            "intrinsics": {
                "status": "UNKNOWN",
                "fx_px": None, "fy_px": None, "cx_px": None, "cy_px": None,
                "missing_datum": "one committed intrinsics record for the exact depth/colour streams in the exact "
                                 "resolution profile that will be used",
                "would_come_from": "on the device: rs-enumerate-devices -c (or pyrealsense2 "
                                   "pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()) "
                                   "saved as configs/camera/d455_<serial>_<mode>_intrinsics.json and committed",
                "rejected_substitutes": [
                    {"file": CAMERA_SOURCES["scene_layout"][0], "line": CAMERA_SOURCES["scene_layout"][1],
                     "value": "rig_world=(-1.40,0,1.20), pitch_deg=-4.0, baseline=0.29, left_y/right_y=+-0.145",
                     "why": "the file header itself says 'ENGINEERING_V0_1 camera rig placeholder (no sensor/render)'. "
                            "It has no intrinsics and no D455 identity, so it cannot stand in for a calibration; the "
                            "0.29 m baseline is rig geometry, not a measured stereo baseline."},
                    {"file": CAMERA_SOURCES["proxy_fx_700"][0], "line": CAMERA_SOURCES["proxy_fx_700"][1],
                     "value": "fx=700 px",
                     "why": "an engineered proxy calibration for rendering/synthetic data, explicitly labelled "
                            "'代理标定'; it is not a measurement of any D455"},
                    {"file": "outputs/shuttle_detection/capability/CONTROLLED_MATRIX_RUN1.md", "line": 20,
                     "value": "fx=11200 px",
                     "why": "a long-lens render camera used to draw the size ladder; physically unrelated to a D455"},
                    {"file": CAMERA_SOURCES["no_calibrated_fx"][0], "line": CAMERA_SOURCES["no_calibrated_fx"][1],
                     "value": "'The repository holds no calibrated fx'", "why": "the repository states the gap itself"},
                ],
            },
            "resolution": {
                "status": "UNKNOWN", "width_px": None, "height_px": None,
                "missing_datum": "the actual stream profile the perception stack runs (D455 supports several; the "
                                 "choice sets the intrinsics, so it cannot be assumed independently of them)",
                "would_come_from": "the same intrinsics record; profile name e.g. 848x480x30 or 1280x720x30",
            },
            "rgb_depth_alignment": {
                "status": "UNKNOWN", "aligned": None,
                "missing_datum": "whether the pipeline runs align(rs.stream.color) (depth expressed in the colour "
                                 "optical frame) or publishes the raw depth frame, and T_depth_color in the chosen mode",
                "would_come_from": "the pipeline configuration in code plus "
                                   "depth_profile.get_extrinsics_to(color_profile) on the device",
                "why_it_matters": "with alignment off, a pixel (u,v) in RGB and the same (u,v) in depth are DIFFERENT "
                                  "rays; every 3D projection in P7 depends on which one is true",
            },
            "frame_rate": {
                "status": "UNKNOWN", "fps": None,
                "missing_datum": "the configured sensor frame rate and whether it is the same for depth and colour",
                "would_come_from": "the pipeline config plus a timestamp log (P6-06 owns the timing protocol)",
            },
            "depth_units": {
                "status": "UNKNOWN", "depth_scale_m_per_unit": None, "depth_encoding": None,
                "missing_datum": "the depth scale factor of the unit (a per-device value, NOT a constant to hardcode)",
                "would_come_from": "depth_sensor.get_depth_scale() at connect time, stored alongside the serial number. "
                                   "Assuming 0.001 m would silently put every ball 10x off if the device reports 0.0001.",
            },
            "distortion": {
                "status": "UNKNOWN", "model": None, "coefficients": None,
                "missing_datum": "the distortion model and coefficients for the colour stream (the D455 RGB stream is "
                                 "typically Brown-Conrady; the VALUES are still per-device)",
                "would_come_from": "the intrinsics record; a factory calibration read from the device is the minimum, "
                                   "a chessboard/charuco re-calibration is the honest maximum",
            },
            "extrinsics_depth_to_color": {
                "status": "UNKNOWN", "translation_m": None, "rotation_quaternion": None,
                "missing_datum": "T_depth_color for this unit",
                "would_come_from": "depth_profile.get_extrinsics_to(color_profile) (factory, per-unit) committed with the serial",
            },
            "extrinsics_mount_to_camera": {
                "status": "UNKNOWN", "translation_m": None, "rotation_quaternion": None,
                "missing_datum": "T_mount_camera: where the camera sits on the robot and how it is aimed",
                "would_come_from": "hand-eye calibration (or a measured CAD mount) against the chassis/arm base frame; "
                                   "the FOV of a camera is not a substitute for its pose",
            },
        },
        "frames": {
            "robot_mount_frame": {
                "convention": "REP-103 body frame: +X forward, +Y left, +Z up, right-handed",
                "note": "this is the frame the arm and chassis use; the frame tree in frame_tree.json is expressed in it",
            },
            "camera_optical_frame": {
                "convention": "REP-103 optical frame: +Z forward out of the lens, +X right in the image, +Y down, right-handed",
                "rotation_from_mount_to_optical": {
                    "quat_order": "wxyz",
                    "values": list(MOUNT_TO_OPTICAL_WXYZ),
                    "matrix": quat_wxyz_to_matrix(MOUNT_TO_OPTICAL_WXYZ),
                    "derivation": "the matrix columns are the mount axes expressed in optical coordinates: "
                                  "X_mount->(0,0,1)=optical +Z, Y_mount->(-1,0,0)=optical -X, Z_mount->(0,-1,0)=optical -Y. "
                                  "The matrix itself is the record; the quaternion is its wxyz encoding.",
                    "status": "CONVENTION_NOT_CALIBRATION",
                    "note": "the AXES are fixed by the two conventions; the camera's POSE on the robot is a different "
                            "quantity and stays UNKNOWN above",
                },
                "handedness_check": "determinant of the matrix above is +1 (both frames are right-handed)",
            },
            "distinction_rule": (
                "camera_optical_frame != robot_mount_frame: they differ by a fixed 90-degree permutation "
                "(R_x(-90) then R_z(-90)). A depth point (X,Y,Z)_optical is (Z, -X, -Y) in the mount frame. "
                "Mixing them rotates every ball position by 90 degrees and is invisible in a 2D overlay."
            ),
        },
        "repository_sources_checked": [
            {"file": CAMERA_SOURCES["scene_layout"][0], "sha256": layout_sha, "line": CAMERA_SOURCES["scene_layout"][1],
             "content": "CAMERA = dict(rig_height=1.20, rig_rel_x=0.20, rig_world=(-1.40,0,1.20), pitch_deg=-4.0, "
                        "baseline=0.29, left_y=0.145, right_y=-0.145)",
             "verdict": "PLACEHOLDER_NOT_CALIBRATION",
             "why": "the file header calls it 'ENGINEERING_V0_1 camera rig placeholder (no sensor/render)'. It carries "
                    "NO intrinsics, NO resolution, NO distortion and NO D455 identity; baseline 0.29 m is a rig geometry "
                    "placeholder, not a measured stereo baseline."},
            {"file": CAMERA_SOURCES["camera_rig_cfg"][0], "sha256": rig_sha, "line": 1,
             "content": "RIG = dict(mast_height=..., mast_radius=0.02, arm_y=CAMERA['baseline']/2+0.03, cam_box=(0.05,0.05,0.05))",
             "verdict": "PLACEHOLDER_NOT_CALIBRATION",
             "why": "derives a 50 mm cube and a mast from the placeholder above; it contains no camera model at all."},
            {"file": CAMERA_SOURCES["no_calibrated_fx"][0], "sha256": sha256_of(CAMERA_SOURCES["no_calibrated_fx"][0]),
             "line": CAMERA_SOURCES["no_calibrated_fx"][1],
             "content": "The repository holds no calibrated fx",
             "verdict": "CONFIRMS_UNKNOWN",
             "why": "the project already recorded that any fx in this repo is absent, so filling these fields here "
                    "would be invention."},
        ],
        "verdict": {
            "intrinsics": "UNKNOWN", "resolution": "UNKNOWN", "rgb_depth_alignment": "UNKNOWN",
            "frame_rate": "UNKNOWN", "depth_units": "UNKNOWN", "distortion": "UNKNOWN",
            "extrinsics": "UNKNOWN",
            "summary": "NO D455 field is known from a committed source. What IS proven: the repository's only camera "
                       "numbers are labelled placeholders, and the optical-vs-mount axis convention is stated exactly.",
        },
        "fill_procedure": [
            "1. On a host with the device attached: rs-enumerate-devices -c > d455_<serial>_<fw>_enumerate.txt",
            "2. pyrealsense2: pick the exact stream profile the perception stack will run; read intrinsics + distortion "
            "for colour, intrinsics for depth, depth_sensor.get_depth_scale(), and "
            "depth_profile.get_extrinsics_to(color_profile)",
            "3. Write configs/camera/d455_<serial>_<mode>.json with those numbers plus the profile name and firmware",
            "4. Commit it, hash it, and point build_p6_frame_contract.py at it so these fields stop being UNKNOWN",
            "5. Hand-eye calibrate T_mount_camera separately and commit that as its own record - it is a robot "
            "measurement, not a camera fact",
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tree = build_frame_tree()
    cam = build_camera()

    # Every asserted edge must be right-handed (det = +1) and every unknown must be null.
    for e in tree["edges"]:
        if e["status"] == "ASSERTED_BY_ASSET":
            d = det3(e["rotation_matrix"])
            if abs(d - 1.0) > 1e-6:
                raise SystemExit("edge %s has det=%r; not a proper rotation" % (e["edge"], d))
        else:
            if e["translation_m"] is not None or e["rotation_matrix"] is not None:
                raise SystemExit("edge %s is UNKNOWN but carries numbers" % e["edge"])
            if not e.get("missing_datum"):
                raise SystemExit("edge %s is UNKNOWN without naming the missing datum" % e["edge"])

    (OUT / "frame_tree.json").write_text(json.dumps(tree, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "camera_metadata.json").write_text(json.dumps(cam, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT / "frame_tree.json")
    print("  edges", tree["summary"]["edges_total"], "asserted", tree["summary"]["asserted_by_asset"],
          "unknown", tree["summary"]["assumptions_or_unknown"])
    print("wrote", OUT / "camera_metadata.json")
    print("  camera verdict:", cam["verdict"]["summary"][:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
