#!/usr/bin/env python3
"""P6 frame-transform consistency auditor (P6-05).

Pure Python + numpy.  It imports nothing from other P6 slices and reads the
committed JSON only by path: every numeric datum is re-resolved from the file it
claims to come from.

Audited artifacts (all under --frames-dir):

  frame_tree.json        required  (P6-01 / PR #26, this branch)
  camera_metadata.json   required  (P6-01 / PR #26, this branch)
  mounted_chain.json     optional  (P6-02 / PR #36)
  mounted_chain_v2.json  optional  (P6-02-OI-1 / PR #37)
  joint_anchors.json     optional  (P6-02-OI-1 / PR #37)

An optional artifact missing from the frames dir is looked up in
<frames-dir>/evidence/upstream/<name> (the pinned-copy convention this repo
already uses, cf. evidence/p3_mount_frame_tree_at_0bbed19.json).  Nothing is
ever written to an audited file.

Exit codes
  0  clean: no inconsistency
  2  at least one inconsistency (every finding carries a typed code)
  1  usage error / a required artifact is missing or unparsable

Typed codes
  E_ARTIFACT_MISSING             required artifact absent (exit 1)
  E_JSON_INVALID                 artifact is not valid JSON (exit 2)
  E_SCHEMA_FIELD_MISSING         a field the audit needs is absent
  E_POINTER_UNRESOLVED           value_ref.json_pointer does not resolve
  E_SHA256_MISMATCH              recorded sha256 != sha256 of the referenced file
  E_REF_EDGE_MISMATCH            pointer resolves to a different edge than claimed
  E_POINTER_ONLY_VIOLATED        a pointer-only edge also carries numbers
  E_UNKNOWN_HAS_NUMBER           an UNKNOWN edge carries a numeric payload
  E_UNKNOWN_HAS_POINTER          an UNKNOWN edge carries a value_ref
  E_UNKNOWN_SET_CHANGED          the set of UNKNOWN edges is no longer {1,2,13,14,15}
  E_QUANTITY_DUPLICATE_MISMATCH  the same physical quantity recorded twice with
                                 different values (across files)
  E_ROTATION_NOT_ORTHONORMAL     max|R^T R - I| > tol
  E_DETERMINANT_NOT_PLUS_ONE     det(R) != +1
  E_QUAT_MATRIX_MISMATCH         {order,values} does not reproduce the matrix
  E_UNIT_NOT_METRE               a length/angle declaration is not metre/radian
  E_COMPOSITION_MISMATCH         recomposing the asserted sub-chain does not
                                 reproduce the recorded end pose
  E_JOINT_ANCHOR_MISMATCH        X0 * inv(X1) != the recorded T_body0_body1
  E_HISTORICAL_ERROR_CHANGED     mounted_chain.json no longer records 0.33242 m
  E_V2_ERROR_TOO_LARGE           mounted_chain_v2.json > 1e-6 m from the pxr pose
  E_OPTICAL_PERMUTATION_MISMATCH camera_metadata optical<->mount block is not
                                 the documented 90-degree permutation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

FRAMES_DIR_DEFAULT = "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
PINNED_SUBDIR = "evidence/upstream"

REQUIRED = ("frame_tree.json", "camera_metadata.json")
OPTIONAL = ("mounted_chain.json", "mounted_chain_v2.json", "joint_anchors.json")
UNKNOWN_EDGE_IDS = (1, 2, 13, 14, 15)

TOL_ORTHO = 1e-9
TOL_DET = 1e-9
TOL_QUAT_MATRIX = 1e-6
TOL_COMPOSITION_M = 1e-9
TOL_V2_M = 1e-6
HISTORICAL_DELTA_M = 0.33242
TOL_HISTORICAL_M = 1e-12

# A subtree that is an explicit anchor AROUND an UNKNOWN edge, whose own status
# declares itself a non-measurement, may legitimately carry numbers.  Anything
# else next to an UNKNOWN is an error.
ANCHOR_EXEMPT_KEYS = ("p3_01_anchor",)


class Report:
    def __init__(self) -> None:
        self.errors = []
        self.checks = []
        self.notes = []
        self.out_of_scope = []

    def ok(self, check, detail):
        self.checks.append({"check": check, "status": "PASS", "detail": detail})

    def fail(self, code, check, detail, where=None):
        self.errors.append({"code": code, "check": check, "detail": detail, "where": where})
        self.checks.append({"check": check, "status": "FAIL", "code": code, "detail": detail})

    def note(self, text):
        self.notes.append(text)

    @property
    def clean(self):
        return not self.errors


def sha256_lf(path):
    """sha256 over the LF-normalised bytes (the basis every P6 file declares)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    return hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_pointer(doc, pointer):
    """RFC 6901 JSON pointer; raises KeyError/IndexError/TypeError/ValueError."""
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(token)]
        elif isinstance(cur, dict):
            cur = cur[token]
        else:
            raise TypeError("pointer step " + repr(token) + " into " + type(cur).__name__)
    return cur


def edge_by_id(doc, edge_id):
    for edge in doc.get("edges", []):
        if edge.get("edge") == edge_id:
            return edge
    return None


def quat_to_matrix(values, order):
    v = [float(x) for x in values]
    if len(v) != 4:
        raise ValueError("quaternion needs 4 values")
    if order == "wxyz":
        w, x, y, z = v
    elif order == "xyzw":
        x, y, z, w = v
    else:
        raise ValueError("unknown quaternion order " + repr(order))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def as_matrix(rot):
    if rot is None:
        return None
    return np.asarray(rot, dtype=float)


def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def position_delta(Ta, Tb):
    """max component-wise position delta of two 4x4 poses (metres)."""
    return float(np.max(np.abs(np.asarray(Ta, float)[:3, 3] - np.asarray(Tb, float)[:3, 3])))


def rotation_delta(Ta, Tb):
    """max element-wise rotation delta of two 4x4 poses (radians, small-angle)."""
    return float(np.max(np.abs(np.asarray(Ta, float)[:3, :3] - np.asarray(Tb, float)[:3, :3])))


def numbers_in(obj, skip_keys=("edge",), skip_subtrees=()):
    """Every int/float leaf under obj, honouring the anchor exemptions."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if k in skip_keys or k in skip_subtrees:
                continue
            out.extend(numbers_in(v, skip_keys, skip_subtrees))
        return out
    if isinstance(obj, list):
        out = []
        for v in obj:
            out.extend(numbers_in(v, skip_keys, skip_subtrees))
        return out
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    return []


def discover(frames_dir):
    found = {}
    for name in REQUIRED + OPTIONAL:
        plain = os.path.join(frames_dir, name)
        pinned = os.path.join(frames_dir, PINNED_SUBDIR, name)
        if os.path.exists(plain):
            found[name] = (plain, "frames-dir")
        elif os.path.exists(pinned):
            found[name] = (pinned, "pinned-upstream-copy")
        else:
            found[name] = (None, None)
    return found


def check_rotation(edge, edge_id, report, source):
    rot = edge.get("rotation_matrix")
    quat = edge.get("rotation_quat")
    if rot is None and quat is None:
        return
    if rot is None:
        report.fail("E_SCHEMA_FIELD_MISSING", "rotation.present",
                    "edge " + str(edge_id) + " has a quaternion but no rotation_matrix", source)
        return
    R = as_matrix(rot)
    if R.shape != (3, 3):
        report.fail("E_SCHEMA_FIELD_MISSING", "rotation.shape",
                    "edge " + str(edge_id) + " rotation_matrix is " + str(R.shape), source)
        return
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if ortho_err > TOL_ORTHO:
        report.fail("E_ROTATION_NOT_ORTHONORMAL", "rotation.orthonormal",
                    "edge " + str(edge_id) + ": max|R^T R - I| = %.3e" % ortho_err, source)
    else:
        report.ok("rotation.orthonormal", "edge " + str(edge_id) + " max|R^T R - I| = %.3e" % ortho_err)
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > TOL_DET:
        report.fail("E_DETERMINANT_NOT_PLUS_ONE", "rotation.det",
                    "edge " + str(edge_id) + ": det(R) = " + repr(det), source)
    else:
        report.ok("rotation.det", "edge " + str(edge_id) + " det(R) = %.15g" % det)
    if quat is not None:
        order = quat.get("order")
        try:
            Rq = quat_to_matrix(quat.get("values"), order)
        except (ValueError, TypeError) as exc:
            report.fail("E_QUAT_MATRIX_MISMATCH", "rotation.quat_matrix",
                        "edge " + str(edge_id) + ": quaternion not decodable (" + str(exc) + ")", source)
            return
        err = max_abs(R, Rq)
        if err > TOL_QUAT_MATRIX:
            report.fail("E_QUAT_MATRIX_MISMATCH", "rotation.quat_matrix",
                        "edge " + str(edge_id) + ": max|R - R(q)| = %.3e" % err, source)
        else:
            report.ok("rotation.quat_matrix",
                      "edge " + str(edge_id) + " max|R - R(q[%s])| = %.3e" % (order, err))


def check_units(report, frame_tree, mounted_chain, joint_anchors):
    decl = (frame_tree.get("conventions") or {}).get("length_unit")
    if decl != "metre":
        report.fail("E_UNIT_NOT_METRE", "units.frame_tree",
                    "frame_tree conventions.length_unit = " + repr(decl))
    else:
        report.ok("units.frame_tree", "frame_tree conventions.length_unit = 'metre'")
    if mounted_chain is not None:
        u = mounted_chain.get("units") or {}
        if u.get("length") != "metre" or u.get("angle") != "radian":
            report.fail("E_UNIT_NOT_METRE", "units.mounted_chain", "mounted_chain units = " + repr(u))
        else:
            report.ok("units.mounted_chain", "mounted_chain units length=metre angle=radian")
    if joint_anchors is not None:
        u = joint_anchors.get("units") or {}
        if u.get("length") != "metre" or u.get("angle") != "radian":
            report.fail("E_UNIT_NOT_METRE", "units.joint_anchors", "joint_anchors units = " + repr(u))
        else:
            report.ok("units.joint_anchors", "joint_anchors units length=metre angle=radian")


def check_unknowns(report, frame_tree, mounted_chain, camera_metadata):
    ids = [e.get("edge") for e in frame_tree.get("edges", []) if e.get("status") == "UNKNOWN"]
    if tuple(ids) != UNKNOWN_EDGE_IDS:
        report.fail("E_UNKNOWN_SET_CHANGED", "unknown.set",
                    "frame_tree UNKNOWN edges are " + repr(ids))
    else:
        report.ok("unknown.set", "frame_tree UNKNOWN edges are 1,2,13,14,15")

    for edge in frame_tree.get("edges", []):
        if edge.get("status") != "UNKNOWN":
            continue
        eid = edge.get("edge")
        nums = numbers_in(edge)
        if nums:
            report.fail("E_UNKNOWN_HAS_NUMBER", "unknown.no_number",
                        "frame_tree edge " + str(eid) + " is UNKNOWN but carries " + repr(nums[:6]),
                        "frame_tree.json")
        for key in ("translation_m", "rotation_quat", "rotation_matrix"):
            if edge.get(key) is not None:
                report.fail("E_UNKNOWN_HAS_NUMBER", "unknown.no_payload",
                            "frame_tree edge " + str(eid) + " is UNKNOWN but " + key + " = "
                            + repr(edge.get(key)), "frame_tree.json")
        if not nums:
            report.ok("unknown.no_number", "frame_tree edge " + str(eid) + " UNKNOWN carries no number")

    if mounted_chain is not None:
        for edge in mounted_chain.get("edges", []):
            if edge.get("status") != "UNKNOWN":
                continue
            eid = edge.get("edge")
            if edge.get("value_ref") is not None:
                report.fail("E_UNKNOWN_HAS_POINTER", "unknown.no_pointer",
                            "mounted_chain edge " + str(eid) + " is UNKNOWN but has a value_ref",
                            "mounted_chain.json")
            nums = numbers_in(edge, skip_keys=("edge",), skip_subtrees=ANCHOR_EXEMPT_KEYS)
            if nums:
                report.fail("E_UNKNOWN_HAS_NUMBER", "unknown.no_number",
                            "mounted_chain edge " + str(eid) + " is UNKNOWN but carries "
                            + repr(nums[:6]), "mounted_chain.json")
            anchor = edge.get("p3_01_anchor")
            if anchor is not None:
                status = str(anchor.get("status", "")).upper()
                if "NOT" not in status and "SIM_AUTHORED" not in status:
                    report.fail("E_UNKNOWN_HAS_NUMBER", "unknown.anchor_status",
                                "mounted_chain edge " + str(eid) + " carries a numeric anchor whose "
                                "status does not declare itself a non-measurement", "mounted_chain.json")
            if not nums:
                report.ok("unknown.no_number", "mounted_chain edge " + str(eid) + " UNKNOWN carries no number")

    for fname, field in (camera_metadata.get("fields") or {}).items():
        if field.get("status") != "UNKNOWN":
            continue
        for key, val in field.items():
            if key == "status":
                continue
            bad = None
            if isinstance(val, bool):
                bad = None
            elif isinstance(val, (int, float)):
                bad = val
            elif (isinstance(val, list) and val
                  and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val)):
                bad = val
            if bad is not None:
                report.fail("E_UNKNOWN_HAS_NUMBER", "unknown.camera_field",
                            "camera_metadata field " + fname + "." + key + " is UNKNOWN but = "
                            + repr(bad), "camera_metadata.json")
    report.ok("unknown.camera_fields", "no numeric payload on an UNKNOWN camera field")


def check_refs(report, frames_dir, found, docs):
    by_basename = {os.path.basename(p): p for p, _ in found.values() if p}

    def audited_ref(ref, owner, label):
        target = ref.get("file")
        pointer = ref.get("json_pointer", "")
        if not target:
            report.fail("E_SCHEMA_FIELD_MISSING", "ref.file", owner + ": " + label + " has no file")
            return None
        base = os.path.basename(target)
        local = None
        if base in dict((n, True) for n in REQUIRED + OPTIONAL):
            cand = os.path.join(frames_dir, base)
            if os.path.exists(cand):
                local = cand
            else:
                cand = os.path.join(frames_dir, PINNED_SUBDIR, base)
                if os.path.exists(cand):
                    local = cand
            if local is None:
                local = by_basename.get(base)
        if local is None:
            report.out_of_scope.append({"owner": owner, "label": label, "file": target,
                                        "reason": "referenced file is not one of the audited artifacts"})
            return None
        want = ref.get("sha256")
        if want:
            got = sha256_lf(local)
            if got != want:
                report.fail("E_SHA256_MISMATCH", "ref.sha256",
                            owner + ": " + label + " records sha256 " + want + " but " + base
                            + " hashes to " + got, base)
            else:
                report.ok("ref.sha256", owner + ": " + label + " sha256 == " + base)
        doc = docs.get(base)
        if doc is None:
            try:
                doc = load_json(local)
            except Exception as exc:
                report.fail("E_JSON_INVALID", "ref.parse",
                            owner + ": " + label + " target " + base + " is not valid JSON (" + str(exc) + ")", base)
                return None
        try:
            return resolve_pointer(doc, pointer)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            report.fail("E_POINTER_UNRESOLVED", "ref.pointer",
                        owner + ": " + label + " pointer " + repr(pointer) + " does not resolve in "
                        + base + " (" + type(exc).__name__ + ": " + str(exc) + ")", base)
            return None

    mc = docs.get("mounted_chain.json")
    if mc is not None:
        for i, src in enumerate(mc.get("sources", [])):
            audited_ref(src.get("ref") or {}, "mounted_chain.sources[" + str(i) + "]", "source")
        for edge in mc.get("edges", []):
            ref = edge.get("value_ref")
            if not ref:
                continue
            eid = edge.get("edge")
            tgt = audited_ref(ref, "mounted_chain.edge " + str(eid), "value_ref")
            if isinstance(tgt, dict):
                for key in ("edge", "from", "to", "edge_kind", "status"):
                    if key in tgt and tgt.get(key) != edge.get(key):
                        report.fail("E_REF_EDGE_MISMATCH", "ref.edge_identity",
                                    "mounted_chain edge " + str(eid) + " claims " + key + "="
                                    + repr(edge.get(key)) + " but the pointer resolves to "
                                    + repr(tgt.get(key)), "mounted_chain.json")
            payload = [k for k in ("translation_m", "rotation_quat", "rotation_matrix")
                       if edge.get(k) is not None]
            if payload:
                report.fail("E_POINTER_ONLY_VIOLATED", "pointer_only",
                            "mounted_chain edge " + str(eid) + " is pointer-only but also carries "
                            + repr(payload), "mounted_chain.json")
            if isinstance(tgt, dict) and not payload:
                report.ok("ref.edge_identity",
                          "mounted_chain edge " + str(eid) + " value_ref resolves to the same edge")
        for block_name, block in (mc.get("composition") or {}).items():
            if not isinstance(block, dict):
                continue
            ref = block.get("hops_ref")
            if ref:
                audited_ref(ref, "mounted_chain.composition." + block_name, "hops_ref")
            ref = (block.get("vs_pxr_end_pose") or {}).get("p3_gripper_end_world_rest_pose_ref")
            if ref:
                audited_ref(ref, "mounted_chain.composition." + block_name, "vs_pxr_end_pose")

    v2 = docs.get("mounted_chain_v2.json")
    if v2 is not None:
        for label in ("base_chain", "joint_anchors"):
            ref = v2.get(label)
            if ref:
                audited_ref(ref, "mounted_chain_v2." + label, "ref")
        for block_name, block in (v2.get("composition") or {}).items():
            if not isinstance(block, dict):
                continue
            ref = block.get("hops_ref")
            if ref:
                audited_ref(ref, "mounted_chain_v2.composition." + block_name, "hops_ref")
        hops = ((v2.get("composition_v2") or {}).get("B_v2_body0_and_body1") or {}).get("hops", [])
        for i, hop in enumerate(hops):
            for label in ("value_ref", "body1_anchor_ref"):
                ref = hop.get(label)
                if ref:
                    audited_ref(ref, "mounted_chain_v2.composition_v2.hops[" + str(i) + "]", label)
        unk = v2.get("unknown_edges_unchanged") or {}
        if unk.get("ref"):
            resolved = audited_ref(unk["ref"], "mounted_chain_v2.unknown_edges_unchanged", "ref")
            if resolved is not None and not isinstance(resolved, list):
                report.fail("E_UNKNOWN_HAS_POINTER", "unknown.no_pointer",
                            "mounted_chain_v2.unknown_edges_unchanged.ref does not point at the "
                            "unknown list", "mounted_chain_v2.json")


def check_duplicates(report, docs):
    frame_tree = docs.get("frame_tree.json")
    mounted_chain = docs.get("mounted_chain.json")
    mounted_chain_v2 = docs.get("mounted_chain_v2.json")
    joint_anchors = docs.get("joint_anchors.json")

    def compare(label, a, b, where_a, where_b):
        if a is None or b is None:
            return
        try:
            aa = np.asarray(a, dtype=float).ravel()
            bb = np.asarray(b, dtype=float).ravel()
        except (TypeError, ValueError):
            return
        if aa.shape != bb.shape:
            report.fail("E_QUANTITY_DUPLICATE_MISMATCH", "duplicate.value",
                        label + ": shapes differ " + str(aa.shape) + " vs " + str(bb.shape),
                        where_a + " vs " + where_b)
            return
        if not np.array_equal(aa, bb):
            delta = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
            report.fail("E_QUANTITY_DUPLICATE_MISMATCH", "duplicate.value",
                        label + ": " + where_a + " = " + repr(aa.tolist()) + " but " + where_b
                        + " = " + repr(bb.tolist()) + " (max|delta| = %.3e)" % delta
                        + "; the same datum must be byte-equal in both files",
                        where_a + " vs " + where_b)
        else:
            report.ok("duplicate.value",
                      label + ": byte-equal across " + where_a + " / " + where_b)

    if frame_tree is not None and joint_anchors is not None:
        for joint in joint_anchors.get("joints", []):
            eid = joint.get("merged_chain_edge")
            edge = edge_by_id(frame_tree, eid) if eid is not None else None
            if edge is None:
                continue
            compare("edge " + str(eid) + " translation",
                    edge.get("translation_m"), joint.get("local_pos0_m"),
                    "frame_tree edge " + str(eid) + " translation_m",
                    "joint_anchors " + str(joint.get("prim")) + " local_pos0_m")
            q = edge.get("rotation_quat") or {}
            compare("edge " + str(eid) + " rotation",
                    q.get("values"), joint.get("local_rot0_wxyz"),
                    "frame_tree edge " + str(eid) + " rotation_quat.values",
                    "joint_anchors " + str(joint.get("prim")) + " local_rot0_wxyz")

    if frame_tree is not None and mounted_chain is not None:
        for m in mounted_chain.get("edges", []):
            eid = m.get("edge")
            ft_edge = edge_by_id(frame_tree, eid) if eid is not None else None
            if ft_edge is None:
                continue
            compare("mounted_chain edge " + str(eid) + " translation",
                    m.get("translation_m"), ft_edge.get("translation_m"),
                    "mounted_chain edge " + str(eid) + " translation_m",
                    "frame_tree edge " + str(eid) + " translation_m")
            compare("mounted_chain edge " + str(eid) + " rotation",
                    (m.get("rotation_quat") or {}).get("values"),
                    (ft_edge.get("rotation_quat") or {}).get("values"),
                    "mounted_chain edge " + str(eid) + " rotation_quat.values",
                    "frame_tree edge " + str(eid) + " rotation_quat.values")

    if mounted_chain is not None and mounted_chain_v2 is not None:
        c1 = mounted_chain.get("composition") or {}
        c2 = mounted_chain_v2.get("composition") or {}
        a1 = (c1.get("A_usdgeom_xformops_authoritative") or {}).get("T")
        a2 = (c2.get("A_usdgeom_xformops_authoritative") or {}).get("T")
        compare("composition A end pose", a1, a2,
                "mounted_chain composition.A.T", "mounted_chain_v2 composition.A.T")
        b1 = (c1.get("B_p6_physics_joint_anchors") or {}).get("vs_A", {}).get("max_abs_delta_m")
        b2 = (c2.get("B_p6_physics_joint_anchors") or {}).get("vs_A", {}).get("max_abs_delta_m")
        oi = (c1.get("open_issue") or {}).get("max_abs_delta_m")
        compare("composition B v1 error", b1, b2,
                "mounted_chain.composition.B.vs_A.max_abs_delta_m",
                "mounted_chain_v2.composition.B.vs_A.max_abs_delta_m")
        compare("composition B v1 error vs open_issue", b1, oi,
                "mounted_chain.composition.B.vs_A.max_abs_delta_m",
                "mounted_chain.composition.open_issue.max_abs_delta_m")
        before = (((mounted_chain_v2.get("composition_v2") or {}).get("error_progression") or {})
                  .get("before") or {}).get("max_abs_delta_m")
        compare("composition B v1 error (error_progression.before)", before, oi,
                "mounted_chain_v2.composition_v2.error_progression.before.max_abs_delta_m",
                "mounted_chain.composition.open_issue.max_abs_delta_m")


def check_camera(report, camera_metadata):
    opt = ((camera_metadata.get("frames") or {}).get("camera_optical_frame") or {})
    block = opt.get("rotation_from_mount_to_optical")
    if not block:
        report.fail("E_SCHEMA_FIELD_MISSING", "camera.optical_block",
                    "camera_metadata lacks frames.camera_optical_frame.rotation_from_mount_to_optical")
        return
    R = as_matrix(block.get("matrix"))
    if R.shape != (3, 3):
        report.fail("E_SCHEMA_FIELD_MISSING", "camera.optical_shape",
                    "optical matrix is " + str(R.shape))
        return
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if ortho_err > TOL_ORTHO:
        report.fail("E_ROTATION_NOT_ORTHONORMAL", "camera.optical_orthonormal",
                    "optical matrix max|R^T R - I| = %.3e" % ortho_err)
    else:
        report.ok("camera.optical_orthonormal", "optical max|R^T R - I| = %.3e" % ortho_err)
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > TOL_DET:
        report.fail("E_DETERMINANT_NOT_PLUS_ONE", "camera.optical_det",
                    "optical matrix det = " + repr(det))
    else:
        report.ok("camera.optical_det", "optical det = %.15g" % det)
    order = block.get("quat_order")
    try:
        Rq = quat_to_matrix(block.get("values"), order)
    except (ValueError, TypeError) as exc:
        report.fail("E_QUAT_MATRIX_MISMATCH", "camera.quat_matrix",
                    "optical quaternion not decodable (" + str(exc) + ")")
        return
    err = max_abs(R, Rq)
    if err > TOL_QUAT_MATRIX:
        report.fail("E_QUAT_MATRIX_MISMATCH", "camera.quat_matrix",
                    "optical max|R - R(q[%s])| = %.3e" % (order, err))
    else:
        report.ok("camera.quat_matrix", "optical max|R - R(q[%s])| = %.3e" % (order, err))
    want = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    perr = max_abs(R, want)
    if perr > TOL_ORTHO:
        report.fail("E_OPTICAL_PERMUTATION_MISMATCH", "camera.optical_permutation",
                    "optical matrix is not the documented permutation "
                    "X_mount->+Z_opt / Y_mount->-X_opt / Z_mount->-Y_opt: max|R - P| = %.3e" % perr)
    else:
        report.ok("camera.optical_permutation",
                  "optical matrix == documented permutation (max|R - P| = %.3e)" % perr)
    ext = ((camera_metadata.get("fields") or {}).get("extrinsics_mount_to_camera") or {})
    if ext.get("status") == "UNKNOWN":
        if ext.get("translation_m") is not None or ext.get("rotation_quaternion") is not None:
            report.fail("E_UNKNOWN_HAS_NUMBER", "camera.mount_extrinsics",
                        "extrinsics_mount_to_camera is UNKNOWN but carries a value")
        else:
            report.ok("camera.mount_extrinsics", "T_mount_camera stays UNKNOWN with no number")


def compose_edges(frame_tree, ids):
    T = np.eye(4)
    for eid in ids:
        edge = edge_by_id(frame_tree, eid)
        if edge is None:
            return None, "edge " + str(eid) + " is absent from frame_tree"
        q = edge.get("rotation_quat") or {}
        if edge.get("translation_m") is None or not q.get("values"):
            return None, "edge " + str(eid) + " has no numeric datum"
        T = T @ homogeneous(quat_to_matrix(q["values"], q.get("order")), edge["translation_m"])
    return T, None


def compose_joint_anchors(joint_anchors, frame_tree, ids):
    T = np.eye(4)
    by_edge = {j.get("merged_chain_edge"): j for j in joint_anchors.get("joints", [])}
    for eid in ids:
        joint = by_edge.get(eid)
        if joint is None:
            edge = edge_by_id(frame_tree, eid)
            if edge is None or edge.get("translation_m") is None:
                return None, "edge " + str(eid) + " has neither an anchor nor a frame_tree datum"
            q = edge.get("rotation_quat") or {}
            T = T @ homogeneous(quat_to_matrix(q["values"], q.get("order")), edge["translation_m"])
            continue
        X0 = as_matrix(joint.get("X0_joint_in_body0"))
        X1 = as_matrix(joint.get("X1_joint_in_body1"))
        if X0 is None or X1 is None:
            return None, str(joint.get("prim")) + " lacks X0/X1"
        T = T @ (X0 @ np.linalg.inv(X1))
    return T, None


def check_composition(report, docs):
    frame_tree = docs.get("frame_tree.json")
    mounted_chain = docs.get("mounted_chain.json")
    mounted_chain_v2 = docs.get("mounted_chain_v2.json")
    joint_anchors = docs.get("joint_anchors.json")

    if joint_anchors is not None:
        n = 0
        for joint in joint_anchors.get("joints", []):
            X0 = as_matrix(joint.get("X0_joint_in_body0"))
            X1 = as_matrix(joint.get("X1_joint_in_body1"))
            T = as_matrix(joint.get("T_body0_body1"))
            if X0 is None or X1 is None or T is None:
                report.fail("E_SCHEMA_FIELD_MISSING", "joint_anchors.compose",
                            str(joint.get("prim")) + " lacks X0/X1/T_body0_body1",
                            "joint_anchors.json")
                continue
            err = max_abs(X0 @ np.linalg.inv(X1), T)
            if err > TOL_COMPOSITION_M:
                report.fail("E_JOINT_ANCHOR_MISMATCH", "joint_anchors.compose",
                            str(joint.get("prim")) + ": X0*inv(X1) differs from T_body0_body1 by "
                            "%.3e" % err, "joint_anchors.json")
            det = float(np.linalg.det(T[:3, :3]))
            if abs(det - 1.0) > TOL_DET:
                report.fail("E_DETERMINANT_NOT_PLUS_ONE", "joint_anchors.det",
                            str(joint.get("prim")) + ": det(R) = " + repr(det), "joint_anchors.json")
            n += 1
        report.ok("joint_anchors.compose", str(n) + " joints: X0*inv(X1) == T_body0_body1")

    if mounted_chain is not None and frame_tree is not None:
        comp = mounted_chain.get("composition") or {}
        b = comp.get("B_p6_physics_joint_anchors") or {}
        ids = b.get("edges_used") or []
        recorded = as_matrix(b.get("T"))
        if recorded is not None and ids:
            recomposed, why = compose_edges(frame_tree, ids)
            if recomposed is None:
                report.fail("E_COMPOSITION_MISMATCH", "composition.B_recompose", why,
                            "mounted_chain.json")
            else:
                err = max_abs(recomposed, recorded)
                if err > TOL_COMPOSITION_M:
                    report.fail("E_COMPOSITION_MISMATCH", "composition.B_recompose",
                                "recomposing frame_tree edges " + repr(ids) + " differs from the "
                                "recorded composition B end pose by %.3e m (tol %.0e)"
                                % (err, TOL_COMPOSITION_M), "mounted_chain.json")
                else:
                    report.ok("composition.B_recompose",
                              "frame_tree edges " + repr(ids) + " reproduce composition B end pose "
                              "(max|delta| = %.3e m)" % err)
            a = as_matrix((comp.get("A_usdgeom_xformops_authoritative") or {}).get("T"))
            hist = float((comp.get("open_issue") or {}).get("max_abs_delta_m", float("nan")))
            if abs(hist - HISTORICAL_DELTA_M) > TOL_HISTORICAL_M:
                report.fail("E_HISTORICAL_ERROR_CHANGED", "historical.0.33242",
                            "mounted_chain.json open_issue.max_abs_delta_m = " + repr(hist)
                            + "; the historical %r m must never be overwritten" % HISTORICAL_DELTA_M,
                            "mounted_chain.json")
            else:
                report.ok("historical.0.33242",
                          "mounted_chain.json keeps its historical %r m composition error"
                          % HISTORICAL_DELTA_M)
            if a is not None:
                b_vs_a = position_delta(recorded, a)
                if abs(b_vs_a - HISTORICAL_DELTA_M) > 1e-5:
                    report.fail("E_COMPOSITION_MISMATCH", "composition.B_vs_A",
                                "max position delta |B - A| = %.6f m but the recorded history "
                                "says %r m" % (b_vs_a, HISTORICAL_DELTA_M), "mounted_chain.json")
                else:
                    report.ok("composition.B_vs_A", "max position delta |B - A| = %.8f m" % b_vs_a)

    if mounted_chain_v2 is not None:
        a = as_matrix(((mounted_chain_v2.get("composition") or {})
                       .get("A_usdgeom_xformops_authoritative") or {}).get("T"))
        b2 = ((mounted_chain_v2.get("composition_v2") or {}).get("B_v2_body0_and_body1") or {})
        recorded = as_matrix(b2.get("T"))
        ids = b2.get("edges_used") or []
        if a is not None and recorded is not None:
            delta = position_delta(recorded, a)
            rec = float((b2.get("vs_A") or {}).get("max_abs_delta_m", float("nan")))
            if delta > TOL_V2_M:
                report.fail("E_V2_ERROR_TOO_LARGE", "v2.within_tolerance",
                            "composition B v2 is %.3e m from the pxr pose, tolerance %.0e m"
                            % (delta, TOL_V2_M), "mounted_chain_v2.json")
            else:
                report.ok("v2.within_tolerance",
                          "composition B v2 is %.3e m from the pxr pose (< %.0e m)"
                          % (delta, TOL_V2_M))
            if not math.isnan(rec) and abs(rec - delta) > 1e-12:
                report.fail("E_COMPOSITION_MISMATCH", "v2.recorded_vs_actual",
                            "mounted_chain_v2 records vs_A.max_abs_delta_m = %.3e but the recorded "
                            "poses differ by %.3e m in position" % (rec, delta),
                            "mounted_chain_v2.json")
            rot = rotation_delta(recorded, a)
            rot_rec = float((b2.get("vs_A") or {}).get("max_abs_rotation_element_rad", float("nan")))
            if not math.isnan(rot_rec) and abs(rot_rec - rot) > 1e-12:
                report.fail("E_COMPOSITION_MISMATCH", "v2.recorded_rotation",
                            "mounted_chain_v2 records max_abs_rotation_element_rad = %.3e but the "
                            "recorded poses differ by %.3e" % (rot_rec, rot), "mounted_chain_v2.json")
            else:
                report.ok("v2.recorded_rotation",
                          "recorded vs_A rotation delta %.3e matches the matrices" % rot)
        if joint_anchors is not None and frame_tree is not None and recorded is not None and ids:
            recomposed, why = compose_joint_anchors(joint_anchors, frame_tree, ids)
            if recomposed is None:
                report.fail("E_COMPOSITION_MISMATCH", "v2.recompose", why, "mounted_chain_v2.json")
            else:
                err = max_abs(recomposed, recorded)
                if err > TOL_COMPOSITION_M:
                    report.fail("E_COMPOSITION_MISMATCH", "v2.recompose",
                                "recomposing joint_anchors X0*inv(X1) over edges " + repr(ids)
                                + " differs from the recorded v2 end pose by %.3e m" % err,
                                "mounted_chain_v2.json")
                else:
                    report.ok("v2.recompose",
                              "joint_anchors X0*inv(X1) over edges " + repr(ids)
                              + " reproduce the v2 end pose (max|delta| = %.3e m)" % err)


def run(frames_dir):
    report = Report()
    found = discover(frames_dir)
    for name in REQUIRED:
        if found[name][0] is None:
            sys.stderr.write("[E_ARTIFACT_MISSING] required artifact " + name + " is not in "
                             + frames_dir + "\n")
            return 1, {"verdict": "USAGE", "reason": "missing required artifact " + name,
                       "errors": [], "checks": [], "artifacts": {}}
    docs = {}
    for name, (path, origin) in found.items():
        if path is None:
            continue
        try:
            docs[name] = load_json(path)
        except Exception as exc:
            if name in REQUIRED:
                sys.stderr.write("[E_JSON_INVALID] " + name + ": " + str(exc) + "\n")
                return 1, {"verdict": "USAGE", "reason": name + " is not valid JSON: " + str(exc),
                           "errors": [], "checks": [], "artifacts": {}}
            report.fail("E_JSON_INVALID", "artifact.parse",
                        name + " is not valid JSON (" + str(exc) + ")", name)
            docs.pop(name, None)

    frame_tree = docs["frame_tree.json"]
    camera_metadata = docs["camera_metadata.json"]
    mounted_chain = docs.get("mounted_chain.json")
    joint_anchors = docs.get("joint_anchors.json")

    for edge in frame_tree.get("edges", []):
        if edge.get("status") == "UNKNOWN":
            continue
        check_rotation(edge, edge.get("edge"), report, "frame_tree.json")
    check_units(report, frame_tree, mounted_chain, joint_anchors)
    check_unknowns(report, frame_tree, mounted_chain, camera_metadata)
    check_refs(report, frames_dir, found, docs)
    check_duplicates(report, docs)
    check_camera(report, camera_metadata)
    check_composition(report, docs)

    artifacts = {}
    for name in REQUIRED + OPTIONAL:
        path, origin = found[name]
        artifacts[name] = {
            "present": path is not None,
            "origin": origin,
            "sha256_lf": sha256_lf(path) if path else None,
        }
    payload = {
        "schema": "p6_frame_consistency_audit/1",
        "tool": "tools/p6_frames/audit_frame_consistency.py",
        "frames_dir": frames_dir,
        "artifacts": artifacts,
        "checks_total": len(report.checks),
        "checks_passed": sum(1 for c in report.checks if c["status"] == "PASS"),
        "errors_total": len(report.errors),
        "errors": report.errors,
        "checks": report.checks,
        "notes": report.notes,
        "out_of_scope_refs": report.out_of_scope,
        "tolerances": {
            "rotation_orthonormal": TOL_ORTHO,
            "determinant": TOL_DET,
            "quat_vs_matrix": TOL_QUAT_MATRIX,
            "composition_m": TOL_COMPOSITION_M,
            "v2_acceptance_m": TOL_V2_M,
            "historical_delta_m": HISTORICAL_DELTA_M,
        },
        "unknown_edges": list(UNKNOWN_EDGE_IDS),
        "verdict": "PASS" if report.clean else "FAIL",
    }
    if report.clean:
        return 0, payload
    payload["error_codes"] = sorted({e["code"] for e in report.errors})
    return 2, payload


def main(argv=None):
    ap = argparse.ArgumentParser(description="P6 frame-transform consistency auditor")
    ap.add_argument("--frames-dir", default=FRAMES_DIR_DEFAULT,
                    help="directory holding the P6 artifacts (default: " + FRAMES_DIR_DEFAULT + ")")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON on stdout")
    ap.add_argument("--report", default=None, help="also write the JSON report to this path")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.frames_dir):
        sys.stderr.write("[E_ARTIFACT_MISSING] frames dir " + repr(args.frames_dir)
                         + " does not exist\n")
        return 1

    code, payload = run(args.frames_dir)
    if args.report:
        parent = os.path.dirname(args.report)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print("P6 frame consistency audit  frames_dir=" + args.frames_dir)
        for name, info in payload.get("artifacts", {}).items():
            print("  artifact %-24s %s" % (name, info["origin"] if info["present"] else "ABSENT"))
        print("  checks: %s/%s passed" % (payload.get("checks_passed"), payload.get("checks_total")))
        for err in payload.get("errors", []):
            print("  [FAIL] " + err["code"] + ": " + err["check"] + ": " + err["detail"])
        for ref in payload.get("out_of_scope_refs", []):
            print("  [scope] " + ref["owner"] + " -> " + ref["file"] + " (not an audited artifact)")
        print("  verdict: " + str(payload.get("verdict")) + "  (exit " + str(code) + ")")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
