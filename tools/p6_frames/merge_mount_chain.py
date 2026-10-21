# -*- coding: utf-8 -*-
"""P6-02: merge the P3-01 mount anchor into the P6 frame tree as ONE chain of record.

Two delivered slices describe the same robot from opposite ends:

  * P6 (p6/racket-camera-frames) : world -> chassis-> arm -> racket -> camera, with every
    boundary that the shipped assets cannot answer marked UNKNOWN (5 of 15 edges).
  * P3-01 (p3/mount-frame-tree, 0bbed19) : the REAL root_joint anchor, whose physics:localPos0/1
    and localRot0/1 are all identity - i.e. the arm is authored AT THE CHASSIS ORIGIN.

This tool emits a single file that carries the whole chain world -> chassis -> arm_base ->
base_link -> link1..link6 -> gripper_end -> racket_mount -> racket_root -> racket_contact_frame
without copying a single measured number: every numeric edge is a POINTER (file + sha256 +
JSON pointer) into the already-committed evidence, so the two slices cannot silently disagree
and no number exists twice.  The 5 UNKNOWN edges stay UNKNOWN, and the P3 identity is recorded
as the SIM-AUTHORED anchor for edge 2 - never as a measurement of the physical mount.

Pure python.  No Isaac, no GPU, no asset writes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P6_DIR = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
FRAME_TREE = P6_DIR / "frame_tree.json"
P3_TREE = P6_DIR / "evidence/p3_mount_frame_tree_at_0bbed19.json"
OUT = P6_DIR / "mounted_chain.json"

P3_COMMIT = "0bbed19"
P3_BRANCH = "p3/mount-frame-tree"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_norm(path: Path) -> str:
    """sha256 over LF-normalised bytes.

    Text evidence is hashed modulo the checkout line ending on purpose: the same blob is CRLF
    in this Windows working tree and LF in the object store, and a hash that only matches in one
    of them is not a hash of the evidence, it is a hash of a lucky checkout.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ref(path: Path, pointer: str) -> dict:
    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_norm(path), "sha256_basis": "LF-normalised bytes",
            "json_pointer": pointer}


def resolve_pointer(doc, pointer: str):
    cur = doc
    for tok in pointer.strip("/").split("/"):
        if tok == "":
            continue
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def mat4(R, t):
    return [[R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0]]


def mat4_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def det3(R):
    return (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
            - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
            + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))


def compose(edges, tree):
    """Compose the numeric edges (resolved through their pointers) into one 4x4."""
    T = mat4([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [0.0, 0.0, 0.0])
    used = []
    for e in edges:
        tref = e.get("value_ref")
        if tref is None:
            continue
        src = tree if tref["file"].endswith("frame_tree.json") else None
        if src is None:
            raise SystemExit(f"unexpected value_ref file {tref['file']}")
        node = resolve_pointer(src, tref["json_pointer"])
        T = mat4_mul(T, mat4(quat_wxyz_to_mat(node["rotation_quat"]["values"]),
                             [float(v) for v in node["translation_m"]]))
        used.append(e["edge"])
    return T, used


def main():
    tree = json.loads(FRAME_TREE.read_text(encoding="utf-8"))
    p3 = json.loads(P3_TREE.read_text(encoding="utf-8"))

    by_id = {e["edge"]: e for e in tree["edges"]}
    p3_anchor_node = resolve_pointer(p3, "/frame_tree/edges/1/anchor")
    p3_mount_tf = resolve_pointer(p3, "/chassis_to_arm_base_transform")

    # P3 is the authority for the anchor; prove the two P3 blocks agree instead of trusting it
    lp0 = p3_anchor_node["local_pos0"]["value"]
    lr0 = p3_anchor_node["local_rot0"]["value"]
    lp1 = p3_anchor_node["local_pos1"]["value"]
    lr1 = p3_anchor_node["local_rot1"]["value"]
    t_from_anchor = [lp0[i] - lp1[i] for i in range(3)]  # identical rotations => pure difference
    rot_equal = all(abs(float(a) - float(b)) < 1e-12 for a, b in zip(lr0, lr1))

    edges_out = []
    for e in tree["edges"]:
        out = {"edge": e["edge"], "from": e["from"], "to": e["to"],
               "edge_kind": e["edge_kind"], "status": e["status"]}
        if e["status"] != "UNKNOWN":
            # pointer only: no number is copied into this file
            out["value_ref"] = ref(FRAME_TREE, f"/edges/{e['edge'] - 1}")
            out["asserted_by"] = "P6 asset evidence (sha256-pinned USD layer), unchanged by this merge"
            if e["edge"] in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
                out["arm_boundary_note"] = (
                    "arm-internal: identical to P6 frame_tree.json, not restated here")
        else:
            out["value_ref"] = None
            out["missing_datum"] = e["missing_datum"]
            out["would_come_from"] = e["would_come_from"]
            out["unknown_guard"] = "UNKNOWN: no numeric datum may be added without a new measurement"
        if e["edge"] == 2:
            out["p3_01_anchor"] = {
                "status": "SIM_AUTHORED_IDENTITY (not a measurement of the physical mount)",
                "value_ref": ref(P3_TREE, "/chassis_to_arm_base_transform"),
                "anchor_ref": ref(P3_TREE, "/frame_tree/edges/1/anchor"),
                "p3_commit": P3_COMMIT,
                "p3_branch": P3_BRANCH,
                "depth_of_evidence": (
                    "payloads/Physics/physics.usda root_joint: physics:localPos0=localPos1=0 and "
                    "physics:localRot0=localRot1=identity (sha256 8d18a8c9...), so the authored "
                    "chassis frame and the arm base frame COINCIDE"),
                "authoring_consequence": (
                    "the sim composes chassis==arm_base; the PHYSICAL bolt-pattern offset on the "
                    "Morph One top plate is still unmeasured, so this edge's status stays UNKNOWN "
                    "and no mount offset is invented"),
                "verified_in_this_merge": {
                    "local_rot0_equals_local_rot1": rot_equal,
                    "translation_from_anchor_m": [float(f"{v:.12g}") for v in t_from_anchor],
                    "matches_p3_derived_transform": (
                        all(abs(a - b) < 1e-12 for a, b in
                            zip(t_from_anchor, p3_mount_tf["position_m"]["value"]))
                        and all(abs(float(a) - float(b)) < 1e-12 for a, b in
                                zip(lr0, p3_mount_tf["rotation_quat_wxyz"]["value"]))),
                },
            }
        edges_out.append(out)

    # --- composition A: the P6 physics-joint anchors, edges 3..12 ----------------------- #
    arm_edges = [o for o in edges_out if 3 <= o["edge"] <= 12
                 and o["edge_kind"] in ("root_joint_identity", "revolute_joint", "authored_frame")]
    T_arm, used = compose(arm_edges, tree)
    R = [row[:3] for row in T_arm[:3]]
    rt_r = [[R[j][i] for j in range(3)] for i in range(3)]
    ortho = [[sum(rt_r[i][k] * R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    # --- composition B: the P3-01 UsdGeom xformOp hops (pxr-verified end pose) ---------- #
    hops = p3["arm_chain_rest_pose"]["hops"]
    T_vis = mat4([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [0.0, 0.0, 0.0])
    for h in hops:
        lt = h["local_transform"]
        T_vis = mat4_mul(T_vis, mat4(quat_wxyz_to_mat(lt["quat_wxyz"]),
                                     [float(v) for v in lt["position"]]))
    Rv = [row[:3] for row in T_vis[:3]]
    rtv = [[Rv[j][i] for j in range(3)] for i in range(3)]
    orthov = [[sum(rtv[i][k] * Rv[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    p3_gripper = p3["arm_chain_rest_pose"]["gripper_end_world_rest_pose"]
    delta = [float(T_vis[i][3] - p3_gripper["world_position"][i]) for i in range(3)]
    delta_joint_anchor_vs_vis = [float(T_arm[i][3] - T_vis[i][3]) for i in range(3)]

    doc = {
        "schema": "p6_mounted_chain/1",
        "task": "P6-02",
        "title": "ONE chain of record: world -> chassis -> arm_base -> base_link -> link1..link6 "
                 "-> gripper_end -> racket_mount -> racket_root -> racket_contact_frame",
        "rule": "no measured number is stored twice: every numeric edge is a pointer "
                "(file + sha256 + JSON pointer) into the committed evidence, and every edge "
                "that the assets cannot answer is UNKNOWN with no numeric payload",
        "generated_by": "tools/p6_frames/merge_mount_chain.py",
        "units": {"length": "metre", "angle": "radian", "quaternion": "wxyz (USD convention)",
                  "source_note": "the source USD declares metersPerUnit=1.0 (both assets); the "
                                 "merge applies no scale factor"},
        "sources": [
            {"role": "P6 frame tree (arm edges, 5 UNKNOWN boundaries)",
             "ref": ref(FRAME_TREE, ""), "sha256_file": sha256_norm(FRAME_TREE),
             "generated_by": tree.get("generated_by")},
            {"role": "P3-01 mount anchor, pinned copy of the delivered branch",
             "ref": ref(P3_TREE, ""), "sha256_file": sha256_norm(P3_TREE),
             "p3_commit": P3_COMMIT, "p3_branch": P3_BRANCH,
             "how_pinned": "git show origin/p3/mount-frame-tree:"
                           "outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_frame_tree.json "
                           "> evidence/p3_mount_frame_tree_at_0bbed19.json (bytes unmodified)"},
        ],
        "chain_order": [
            "world", "chassis", "arm_base", "base_link", "link1", "link2", "link3", "link4",
            "link5", "link6", "gripper_end", "racket_mount", "racket_root", "racket_contact_frame",
        ],
        "node_names": {
            "world": "Isaac stage /world (right-handed, Z up)",
            "chassis": "Morph One chassis body frame (P3: /RW/Chassis proxy body)",
            "arm_base": "P3: /ReBot_Arm_RS defaultPrim == body0 of root_joint; P6 calls the same "
                        "prim arm_root_prim",
            "base_link": "/ReBot_Arm_RS/Geometry/base_link",
            "racket_root": "/RacketVisual in assets/third_party/racket_visual.usd; +Z handle "
                           "butt -> head, +X face normal; origin at the handle BUTT",
            "racket_contact_frame": "centre of the racket face - position UNKNOWN",
        },
        "naming_reconciliation": {
            "p6_arm_root_prim == p3_arm_base": (
                "P6 edge 3 asserts arm_root_prim -> arm_base_link with the identity root_joint, "
                "and P3 names the same prim 'arm_base'; this merge keeps ONE node name "
                "(arm_base) for that frame and records the alias instead of forking the chain"),
            "p6_chassis_morph_one == p3_chassis": "same body frame, same name kept: chassis",
        },
        "edges": edges_out,
        "summary": {
            "edges_total": len(edges_out),
            "asserted": sum(1 for e in edges_out if e["status"] == "ASSERTED_BY_ASSET"),
            "unknown": sum(1 for e in edges_out if e["status"] == "UNKNOWN"),
            "unknown_edge_ids": [e["edge"] for e in edges_out if e["status"] == "UNKNOWN"],
            "numeric_edges_are_pointers_only": True,
        },
        "composition": {
            "A_usdgeom_xformops_authoritative": {
                "what": "chassis -> arm_base -> base_link -> link1..link6 -> gripper_end composed "
                        "from the P3-01 UsdGeom xformOp hops (pxr-verified slice)",
                "hops_used": len(hops),
                "hops_ref": ref(P3_TREE, "/arm_chain_rest_pose/hops"),
                "T": [[round(v, 12) for v in row] for row in T_vis],
                "det_R": round(det3(Rv), 12),
                "R_orthonormal_max_error": round(max(abs(orthov[i][j] - (1.0 if i == j else 0.0))
                                                      for i in range(3) for j in range(3)), 12),
                "vs_pxr_end_pose": {
                    "p3_gripper_end_world_rest_pose_ref": ref(
                        P3_TREE, "/arm_chain_rest_pose/gripper_end_world_rest_pose"),
                    "position_delta_m": [round(v, 9) for v in delta],
                    "max_abs_delta_m": round(max(abs(v) for v in delta), 9),
                },
            },
            "B_p6_physics_joint_anchors": {
                "what": "chassis -> arm_base -> base_link -> link1..link6 -> gripper_end composed "
                        "from the P6 joint anchors (physics:localPos0 + physics:localRot0)",
                "edges_used": used,
                "T": [[round(v, 12) for v in row] for row in T_arm],
                "det_R": round(det3(R), 12),
                "R_orthonormal_max_error": round(max(abs(ortho[i][j] - (1.0 if i == j else 0.0))
                                                     for i in range(3) for j in range(3)), 12),
                "vs_A": {
                    "position_delta_m": [round(v, 9) for v in delta_joint_anchor_vs_vis],
                    "max_abs_delta_m": round(max(abs(v) for v in delta_joint_anchor_vs_vis), 9),
                },
            },
            "open_issue": {
                "id": "P6-02-OI-1",
                "status": "OPEN",
                "finding": (
                    "composing the P6 revolute edges as child-link poses does NOT reproduce the "
                    "pxr-composed gripper_end pose: the USD joint frame of a body carries TWO "
                    "local poses (physics:localPos0/localRot0 on body0 and "
                    "physics:localPos1/localRot1 on body1), and the P6 frame tree records only "
                    "the body0 side. Both compositions are proper rotations (det R = +1), so "
                    "neither is a handedness bug - the gap is a missing localPos1/localRot1 per "
                    "joint, not a wrong number."),
                "max_abs_delta_m": round(max(abs(v) for v in delta_joint_anchor_vs_vis), 9),
                "consequence_for_this_merge": (
                    "the merged chain uses composition A (UsdGeom xformOps) as the chain of "
                    "record; the P6 layer is kept for its per-edge provenance and joint limits, "
                    "and its revolute edge numbers are NOT reinterpreted as child-link poses"),
                "next_action": (
                    "extract physics:localPos1/localRot1 for joints 1..6 from the read-only arm "
                    "USD and re-check A vs B; until then B stays a diagnostic, not a datum"),
            },
        },
        "guards": [
            {"guard": "no_duplicate_numbers",
             "rule": "a numeric edge carries value_ref only; tests fail if a translation_m / "
                     "rotation_quat payload is added next to the pointer"},
            {"guard": "unknown_stays_unknown",
             "rule": "edges 1, 2, 13, 14, 15 must keep translation_m/rotation_quat absent; a test "
                     "fails if anyone silently fills the mount with a guess"},
            {"guard": "edge2_is_sim_authored",
             "rule": "the P3 identity is attached to edge 2 as p3_01_anchor and never changes the "
                     "physical status: the mount offset stays unmeasured"},
        ],
        "unknown": [{"edge": e["edge"], "from": e["from"], "to": e["to"],
                     "missing_datum": e["missing_datum"]}
                    for e in edges_out if e["status"] == "UNKNOWN"],
        "provenance": {
            "pxr_used": False,
            "isaac_sim_used": False,
            "gpu_used": False,
            "assets_modified": False,
            "deterministic": True,
            "command": "python tools/p6_frames/merge_mount_chain.py",
        },
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"edges: {doc['summary']['edges_total']} asserted={doc['summary']['asserted']} "
          f"unknown={doc['summary']['unknown']} ids={doc['summary']['unknown_edge_ids']}")
    A = doc["composition"]["A_usdgeom_xformops_authoritative"]
    B = doc["composition"]["B_p6_physics_joint_anchors"]
    print(f"A (P3 xformOps): det(R)={A['det_R']} ortho_err={A['R_orthonormal_max_error']} "
          f"vs pxr end pose max delta={A['vs_pxr_end_pose']['max_abs_delta_m']} m")
    print(f"B (P6 joint anchors, edges {used}): det(R)={B['det_R']} "
          f"ortho_err={B['R_orthonormal_max_error']} vs A max delta={B['vs_A']['max_abs_delta_m']} m")
    print(f"open issue {doc['composition']['open_issue']['id']}: {doc['composition']['open_issue']['status']}")
    print(f"wrote {OUT}")
    ok = (doc["composition"]["A_usdgeom_xformops_authoritative"]["det_R"] > 0
          and doc["composition"]["B_p6_physics_joint_anchors"]["det_R"] > 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
