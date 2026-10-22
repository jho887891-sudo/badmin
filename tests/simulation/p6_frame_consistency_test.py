"""P6-05 frame-transform consistency: audit + regression suite.

Two layers:
  * real assertions over the COMMITTED P6 artifacts (frame_tree.json,
    camera_metadata.json on this branch; mounted_chain.json,
    mounted_chain_v2.json, joint_anchors.json from the pinned upstream copies in
    evidence/upstream/), re-derived here with numpy instead of trusting
    tools/p6_frames/audit_frame_consistency.py;
  * synthetic mutations of a COPY in tmp_path, each of which must make the
    auditor exit 2 with the right typed code.

The committed JSON is never modified: every mutation happens on a copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES_DIR = os.path.join(REPO_ROOT, "outputs", "simulation", "rebot_b601_rs",
                          "morph_one", "p6_frames")
PINNED_DIR = os.path.join(FRAMES_DIR, "evidence", "upstream")
AUDITOR = os.path.join(REPO_ROOT, "tools", "p6_frames", "audit_frame_consistency.py")

ARTIFACTS = ("frame_tree.json", "camera_metadata.json",
             "mounted_chain.json", "mounted_chain_v2.json", "joint_anchors.json")
UNKNOWN_EDGE_IDS = [1, 2, 13, 14, 15]
TOL_ORTHO = 1e-9
TOL_QUAT_MATRIX = 1e-6
V2_TOL_M = 1e-6
HISTORICAL_DELTA_M = 0.33242


# --------------------------------------------------------------------------- #
# sources and plumbing
# --------------------------------------------------------------------------- #
def source_path(name):
    plain = os.path.join(FRAMES_DIR, name)
    if os.path.exists(plain):
        return plain
    pinned = os.path.join(PINNED_DIR, name)
    if os.path.exists(pinned):
        return pinned
    return None


@pytest.fixture(scope="module")
def frames_dir():
    missing = [n for n in ARTIFACTS if source_path(n) is None]
    assert not missing, "committed P6 artifacts are missing: " + repr(missing)
    return FRAMES_DIR


def load(name):
    return json.load(open(source_path(name), "r", encoding="utf-8"))


def sha256_lf(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    return hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def run_audit(frames):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run([sys.executable, AUDITOR, "--frames-dir", frames, "--json"],
                          cwd=REPO_ROOT, capture_output=True, env=env)
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    payload = None
    if out.strip():
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload, out + err


def codes_of(payload):
    return {e["code"] for e in (payload or {}).get("errors", [])}


@pytest.fixture()
def workspace(tmp_path):
    """A flat copy of all five artifacts, so the auditor reads only this copy."""
    dst = tmp_path / "p6_frames"
    dst.mkdir()
    for name in ARTIFACTS:
        shutil.copyfile(source_path(name), str(dst / name))
    return str(dst)


def read_json(path):
    return json.load(open(path, "r", encoding="utf-8"))


def write_json(path, doc):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def refresh_hashes(frames):
    """Re-pin the recorded sha256 chain after a deliberate value mutation.

    frame_tree -> mounted_chain.sources[0] -> mounted_chain_v2.base_chain
    joint_anchors -> mounted_chain_v2.joint_anchors
    Keeps a mutation test pointed at ONE semantic typed code instead of tripping
    the (also correct) E_SHA256_MISMATCH guard.
    """
    mc_path = os.path.join(frames, "mounted_chain.json")
    v2_path = os.path.join(frames, "mounted_chain_v2.json")
    ja_path = os.path.join(frames, "joint_anchors.json")

    mc = read_json(mc_path)
    old_ft = mc["sources"][0]["ref"]["sha256"]
    new_ft = sha256_lf(os.path.join(frames, "frame_tree.json"))
    with open(mc_path, "r", encoding="utf-8") as fh:
        mc_text = fh.read()
    with open(mc_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(mc_text.replace(old_ft, new_ft))

    v2 = read_json(v2_path)
    with open(v2_path, "r", encoding="utf-8") as fh:
        v2_text = fh.read()
    v2_text = v2_text.replace(v2["joint_anchors"]["sha256"], sha256_lf(ja_path))
    v2_text = v2_text.replace(v2["base_chain"]["sha256"], sha256_lf(mc_path))
    with open(v2_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(v2_text)


# --------------------------------------------------------------------------- #
# independent maths (deliberately duplicated: the test must not import the tool)
# --------------------------------------------------------------------------- #
def quat_matrix(values, order="wxyz"):
    v = [float(x) for x in values]
    if order == "wxyz":
        w, x, y, z = v
    else:
        x, y, z, w = v
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def edge(doc, eid):
    return next(e for e in doc["edges"] if e["edge"] == eid)


def numbers(obj):
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if k != "edge":
                out.extend(numbers(v))
        return out
    if isinstance(obj, list):
        out = []
        for v in obj:
            out.extend(numbers(v))
        return out
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    return []


# --------------------------------------------------------------------------- #
# 1. real assertions over the committed artifacts
# --------------------------------------------------------------------------- #
def test_p6_audit_clean_on_committed_artifacts(frames_dir):
    code, payload, log = run_audit(frames_dir)
    assert code == 0, log
    assert payload is not None, log
    assert payload["verdict"] == "PASS"
    assert payload["errors_total"] == 0
    assert payload["checks_total"] > 50
    assert payload["checks_passed"] == payload["checks_total"]
    for name, info in payload["artifacts"].items():
        assert info["present"], name + " was not audited at all"
    assert payload["unknown_edges"] == UNKNOWN_EDGE_IDS


def test_p6_frame_tree_rotations_are_rigid(frames_dir):
    doc = load("frame_tree.json")
    seen = 0
    for e in doc["edges"]:
        if e["status"] == "UNKNOWN":
            continue
        R = np.asarray(e["rotation_matrix"], dtype=float)
        assert R.shape == (3, 3), e["edge"]
        assert float(np.max(np.abs(R.T @ R - np.eye(3)))) <= TOL_ORTHO, e["edge"]
        assert abs(float(np.linalg.det(R)) - 1.0) <= TOL_ORTHO, e["edge"]
        order = e["rotation_quat"]["order"]
        Rq = quat_matrix(e["rotation_quat"]["values"], order)
        assert float(np.max(np.abs(R - Rq))) <= TOL_QUAT_MATRIX, e["edge"]
        seen += 1
    assert seen == 10, "expected 10 asserted edges, saw " + str(seen)


def test_p6_unknown_edges_carry_no_number(frames_dir):
    ft = load("frame_tree.json")
    mc = load("mounted_chain.json")
    unknown = [e["edge"] for e in ft["edges"] if e["status"] == "UNKNOWN"]
    assert unknown == UNKNOWN_EDGE_IDS
    for e in ft["edges"]:
        if e["status"] != "UNKNOWN":
            continue
        assert numbers(e) == [], "frame_tree edge " + str(e["edge"]) + " carries a number"
        for key in ("translation_m", "rotation_quat", "rotation_matrix"):
            assert e.get(key) is None
    for e in mc["edges"]:
        if e["status"] != "UNKNOWN":
            continue
        assert e.get("value_ref") is None
        nums = numbers({k: v for k, v in e.items() if k != "p3_01_anchor"})
        assert nums == [], "mounted_chain edge " + str(e["edge"]) + " carries a number"
    # the one numeric anchor that sits next to edge 2 must say it is not a measurement
    anchor = edge(mc, 2)["p3_01_anchor"]
    assert "NOT" in anchor["status"].upper() or "SIM_AUTHORED" in anchor["status"].upper()
    # camera side: every UNKNOWN field is null
    cam = load("camera_metadata.json")
    for fname, field in cam["fields"].items():
        if field.get("status") != "UNKNOWN":
            continue
        for key, val in field.items():
            if key != "status" and isinstance(val, (int, float)) and not isinstance(val, bool):
                pytest.fail("camera field " + fname + "." + key + " is UNKNOWN but numeric")


def test_p6_asserted_subchain_reproduces_recorded_end_pose(frames_dir):
    ft = load("frame_tree.json")
    mc = load("mounted_chain.json")
    b = mc["composition"]["B_p6_physics_joint_anchors"]
    ids = b["edges_used"]
    T = np.eye(4)
    for eid in ids:
        e = edge(ft, eid)
        T = T @ pose(quat_matrix(e["rotation_quat"]["values"], e["rotation_quat"]["order"]),
                     e["translation_m"])
    recorded = np.asarray(b["T"], dtype=float)
    assert float(np.max(np.abs(T - recorded))) <= 1e-9
    # B (body0 only) stays 0.33242 m off the pxr-composed pose: history is immutable
    A = np.asarray(mc["composition"]["A_usdgeom_xformops_authoritative"]["T"], dtype=float)
    hist = float(mc["composition"]["open_issue"]["max_abs_delta_m"])
    assert abs(hist - HISTORICAL_DELTA_M) <= 1e-12
    assert abs(float(np.max(np.abs(recorded[:3, 3] - A[:3, 3]))) - HISTORICAL_DELTA_M) <= 1e-5
    assert mc["composition"]["open_issue"]["status"] == "OPEN"


def test_p6_mounted_chain_v2_reduces_error_below_1e6_and_keeps_v1(frames_dir):
    mc = load("mounted_chain.json")
    v2 = load("mounted_chain_v2.json")
    ft = load("frame_tree.json")
    ja = load("joint_anchors.json")
    assert v2["base_chain"]["sha256"] == sha256_lf(source_path("mounted_chain.json"))
    assert v2["joint_anchors"]["sha256"] == sha256_lf(source_path("joint_anchors.json"))
    # v1 is untouched inside the historical file
    assert mc["schema"] == "p6_mounted_chain/1"
    assert abs(float(mc["composition"]["open_issue"]["max_abs_delta_m"]) - HISTORICAL_DELTA_M) <= 1e-12
    # v2 recomposes from BOTH joint body anchors and lands within 1e-6 m
    by_edge = {j["merged_chain_edge"]: j for j in ja["joints"]}
    b2 = v2["composition_v2"]["B_v2_body0_and_body1"]
    T = np.eye(4)
    for eid in b2["edges_used"]:
        joint = by_edge.get(eid)
        if joint is not None:
            T = T @ (np.asarray(joint["X0_joint_in_body0"]) @
                     np.linalg.inv(np.asarray(joint["X1_joint_in_body1"])))
        else:
            e = edge(ft, eid)
            T = T @ pose(quat_matrix(e["rotation_quat"]["values"], e["rotation_quat"]["order"]),
                         e["translation_m"])
    assert float(np.max(np.abs(T - np.asarray(b2["T"], dtype=float)))) <= 1e-9
    A = np.asarray(v2["composition"]["A_usdgeom_xformops_authoritative"]["T"], dtype=float)
    delta = float(np.max(np.abs(np.asarray(b2["T"], dtype=float)[:3, 3] - A[:3, 3])))
    assert delta < V2_TOL_M, "v2 residual " + repr(delta)
    assert abs(delta - float(b2["vs_A"]["max_abs_delta_m"])) <= 1e-12
    assert v2["issue_resolution"]["status"] == "RESOLVED"


def test_p6_joint_anchors_are_self_consistent(frames_dir):
    ja = load("joint_anchors.json")
    assert ja["counts"]["joints_extracted"] == 7
    for j in ja["joints"]:
        X0 = np.asarray(j["X0_joint_in_body0"], dtype=float)
        X1 = np.asarray(j["X1_joint_in_body1"], dtype=float)
        assert float(np.max(np.abs(X0 @ np.linalg.inv(X1) - np.asarray(j["T_body0_body1"], dtype=float)))) <= 1e-9
        assert abs(float(np.linalg.det(X0[:3, :3])) - 1.0) <= TOL_ORTHO
        assert abs(float(np.linalg.det(X1[:3, :3])) - 1.0) <= TOL_ORTHO


def test_p6_camera_optical_mount_permutation_is_self_consistent(frames_dir):
    cam = load("camera_metadata.json")
    block = cam["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]
    R = np.asarray(block["matrix"], dtype=float)
    assert float(np.max(np.abs(R.T @ R - np.eye(3)))) <= TOL_ORTHO
    assert abs(float(np.linalg.det(R)) - 1.0) <= TOL_ORTHO
    Rq = quat_matrix(block["values"], block["quat_order"])
    assert float(np.max(np.abs(R - Rq))) <= TOL_QUAT_MATRIX
    expected = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    assert float(np.max(np.abs(R - expected))) <= TOL_ORTHO
    # the mount POSE is a different quantity and must stay UNKNOWN
    ext = cam["fields"]["extrinsics_mount_to_camera"]
    assert ext["status"] == "UNKNOWN"
    assert ext["translation_m"] is None and ext["rotation_quaternion"] is None
    for key in ("intrinsics", "resolution", "depth_units", "distortion", "extrinsics_depth_to_color"):
        assert cam["fields"][key]["status"] == "UNKNOWN"


# --------------------------------------------------------------------------- #
# 2. synthetic mutations of a COPY (each caught with its typed code, exit 2)
# --------------------------------------------------------------------------- #
def test_p6_mutation_flipped_quaternion_sign_is_caught(workspace):
    path = os.path.join(workspace, "frame_tree.json")
    doc = read_json(path)
    q = edge(doc, 5)["rotation_quat"]["values"]
    edge(doc, 5)["rotation_quat"]["values"] = [-v for v in q]  # q and -q: same rotation
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    assert "E_QUANTITY_DUPLICATE_MISMATCH" in codes_of(payload), log


def test_p6_mutation_one_millimetre_translation_is_caught(workspace):
    path = os.path.join(workspace, "frame_tree.json")
    doc = read_json(path)
    e = edge(doc, 6)
    e["translation_m"] = [e["translation_m"][0] + 0.001, e["translation_m"][1], e["translation_m"][2]]
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    codes = codes_of(payload)
    assert "E_QUANTITY_DUPLICATE_MISMATCH" in codes, log
    assert "E_COMPOSITION_MISMATCH" in codes, log


def test_p6_mutation_unknown_edge_filled_with_a_number_is_caught(workspace):
    path = os.path.join(workspace, "frame_tree.json")
    doc = read_json(path)
    e = edge(doc, 13)
    e["translation_m"] = [0.0, 0.0, 0.05]
    e["rotation_quat"] = {"order": "wxyz", "values": [1.0, 0.0, 0.0, 0.0]}
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    assert "E_UNKNOWN_HAS_NUMBER" in codes_of(payload), log


def test_p6_mutation_duplicate_quantity_with_a_different_value_is_caught(workspace):
    path = os.path.join(workspace, "mounted_chain.json")
    doc = read_json(path)
    edge(doc, 4)["translation_m"] = [-0.0003, -0.0009, 0.0745]  # restates the pointer datum
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    codes = codes_of(payload)
    assert "E_POINTER_ONLY_VIOLATED" in codes, log
    assert "E_QUANTITY_DUPLICATE_MISMATCH" in codes, log


def test_p6_mutation_broken_json_pointer_is_caught(workspace):
    path = os.path.join(workspace, "mounted_chain.json")
    doc = read_json(path)
    edge(doc, 5)["value_ref"]["json_pointer"] = "/edges/99"
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    assert "E_POINTER_UNRESOLVED" in codes_of(payload), log


def test_p6_mutation_tampered_sha256_is_caught(workspace):
    path = os.path.join(workspace, "mounted_chain.json")
    doc = read_json(path)
    doc["sources"][0]["ref"]["sha256"] = "0" * 64  # deliberately NOT re-pinned
    write_json(path, doc)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    assert "E_SHA256_MISMATCH" in codes_of(payload), log


def test_p6_mutation_historical_0_33242_overwrite_is_caught(workspace):
    path = os.path.join(workspace, "mounted_chain.json")
    doc = read_json(path)
    doc["composition"]["open_issue"]["max_abs_delta_m"] = 1.85e-09
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    assert "E_HISTORICAL_ERROR_CHANGED" in codes_of(payload), log


def test_p6_mutation_v2_error_blown_up_is_caught(workspace):
    path = os.path.join(workspace, "mounted_chain_v2.json")
    doc = read_json(path)
    T = doc["composition_v2"]["B_v2_body0_and_body1"]["T"]
    T[0][3] += 0.01
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    codes = codes_of(payload)
    assert "E_V2_ERROR_TOO_LARGE" in codes, log
    assert "E_COMPOSITION_MISMATCH" in codes, log


def test_p6_mutation_non_orthonormal_rotation_is_caught(workspace):
    path = os.path.join(workspace, "frame_tree.json")
    doc = read_json(path)
    edge(doc, 7)["rotation_matrix"][0][0] = 1.01
    write_json(path, doc)
    refresh_hashes(workspace)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    codes = codes_of(payload)
    assert "E_ROTATION_NOT_ORTHONORMAL" in codes, log
    assert "E_QUAT_MATRIX_MISMATCH" in codes, log


def test_p6_mutation_optical_permutation_break_is_caught(workspace):
    path = os.path.join(workspace, "camera_metadata.json")
    doc = read_json(path)
    block = doc["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]
    block["matrix"] = [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]  # det still +1
    write_json(path, doc)
    code, payload, log = run_audit(workspace)
    assert code == 2, log
    codes = codes_of(payload)
    assert "E_OPTICAL_PERMUTATION_MISMATCH" in codes, log
    assert "E_QUAT_MATRIX_MISMATCH" in codes, log


def test_p6_missing_required_artifact_is_a_usage_error(workspace):
    os.remove(os.path.join(workspace, "camera_metadata.json"))
    code, payload, log = run_audit(workspace)
    assert code == 1, log
    assert "[E_ARTIFACT_MISSING]" in log


def test_p6_committed_artifacts_are_not_touched_by_the_suite(frames_dir):
    before = {n: sha256_lf(source_path(n)) for n in ARTIFACTS}
    code, payload, log = run_audit(frames_dir)
    assert code == 0, log
    after = {n: sha256_lf(source_path(n)) for n in ARTIFACTS}
    assert before == after
