"""P3: mounting stability of the REAL B601-RS arm while the chassis actually moves.

P3 asks a different question from P2: with the chassis commanded through a NON-ZERO
translation and/or a NON-ZERO yaw, does the measured world motion really happen, and does
the arm stay rigidly attached (relative mount pose constant, joints holding their target)?

The motion here is a DIAGNOSTIC root-pose drive (write_root_pose_to_sim).  It is explicitly
NOT wheel drive and must never be reported as P4 evidence.

Thresholds are pre-registered in management/morph_one_rs/PHASE_P3_DYNAMIC_MOUNT_GATE.md.

Usage:
    python p3_dynamic_mount.py --run-id P3_xy_r01 --out-dir <dir> --profile translate_xy --steps 360
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

REAL_ARM = ("/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
            "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda")
WHEEL_CENTRES = [(0.250002, -0.297, -0.244), (0.250002, 0.297, -0.244),
                 (-0.250002, -0.297, -0.244), (-0.250002, 0.297, -0.244)]
WHEEL_HALF = (0.0243, 0.0789, 0.0900)
BODY_HALF = (0.198, 0.400, 0.066)
BODY_CENTRE = (0.0, -0.0024, -0.0404)
CHASSIS_MASS_KG = 20.0
DT = 1.0 / 120.0
LIFT = 0.40

# profile -> dict(ramp_s, x, y, z, yaw_rad)
PROFILES = {
    "static": {"ramp_s": 1.0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
    "translate_xy": {"ramp_s": 1.5, "x": 0.30, "y": 0.0, "z": 0.0, "yaw": 0.0},
    "translate_z": {"ramp_s": 1.0, "x": 0.0, "y": 0.0, "z": 0.10, "yaw": 0.0},
    "yaw": {"ramp_s": 1.5, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.35},
    "compound": {"ramp_s": 2.0, "x": 0.20, "y": -0.15, "z": 0.05, "yaw": 0.30},
}

GATE_VERSION = "v1.1"

PARSER = argparse.ArgumentParser(description="P3 dynamic mount stability (diagnostic root drive)")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--profile", required=True, choices=sorted(PROFILES))
PARSER.add_argument("--steps", type=int, default=360)
PARSER.add_argument("--gravity-z", type=float, default=0.0)
PARSER.add_argument("--read-contacts", action="store_true")
PARSER.add_argument("--freeze-axis", choices=("none", "x", "y", "z", "yaw"), default="none",
                    help="NEGATIVE CONTROL: deliberately drop this command component so the "
                         "directed/yaw gate must FAIL. Recorded in manifest as negative_control.")
PARSER.add_argument("--inject-fault", choices=("none", "joint_limit", "python_exception"),
                    default="none",
                    help="P0-07/P3-05 fault injection. 'joint_limit' submits a joint target "
                         "outside the USD limit and must be REFUSED before any write; "
                         "'python_exception' raises inside the stepping loop so the gate runner "
                         "must report a non-zero child code instead of a false PASS.")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from p3_motion_gate import directed_displacement_track  # noqa: E402


def _np(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(values):
    arr = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion")
    return arr / norm


def raw_to_wxyz(raw):
    """Isaac Lab body_quat_w in THIS build is raw (x, y, z, w)."""
    arr = np.asarray(raw, dtype=float)
    return unit(np.array([arr[3], arr[0], arr[1], arr[2]]))


def rotation_matrix(quat_wxyz):
    w, x, y, z = unit(quat_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(quat_wxyz):
    rot = rotation_matrix(quat_wxyz)
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def angle_between(a, b):
    delta = rotation_matrix(a).T @ rotation_matrix(b)
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return float(math.acos(cosine))


def author(path):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/PM")
    stage.SetDefaultPrim(root.GetPrim())

    chassis = UsdGeom.Xform.Define(stage, "/PM/Chassis")
    UsdGeom.Xformable(chassis).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, LIFT))
    body = UsdGeom.Xform.Define(stage, "/PM/Chassis/body_geometry")
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3f(*BODY_CENTRE))
    geom = UsdGeom.Cube.Define(stage, "/PM/Chassis/body_geometry/geometry")
    geom.CreateSizeAttr(1.0)
    UsdGeom.Xformable(geom.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2 * BODY_HALF[0], 2 * BODY_HALF[1], 2 * BODY_HALF[2]))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
    mass_api.CreateMassAttr(CHASSIS_MASS_KG)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    # No ArticulationRootAPI here: the real arm asset already carries exactly one, and a
    # second root under the same articulation is rejected by PhysX (seen in G7B_static_r03).

    for index, centre in enumerate(WHEEL_CENTRES, start=1):
        wheel = UsdGeom.Xform.Define(stage, Sdf.Path("/PM/Chassis/wheel%d" % index))
        UsdGeom.Xformable(wheel).AddTranslateOp().Set(Gf.Vec3f(*centre))
        wheel_geom = UsdGeom.Cube.Define(stage, Sdf.Path("/PM/Chassis/wheel%d/geometry" % index))
        wheel_geom.CreateSizeAttr(1.0)
        UsdGeom.Xformable(wheel_geom.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(2 * WHEEL_HALF[0], 2 * WHEEL_HALF[1], 2 * WHEEL_HALF[2]))
        UsdPhysics.CollisionAPI.Apply(wheel_geom.GetPrim())

    arm = UsdGeom.Xform.Define(stage, "/PM/Arm")
    UsdGeom.Xformable(arm).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, LIFT))
    arm.GetPrim().GetReferences().AddReference(REAL_ARM)
    joint_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/PM/Arm")):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == "root_joint":
            joint_path = prim.GetPath()
            break
    if joint_path is None:
        raise SystemExit("real arm has no root_joint to retarget")
    over = stage.OverridePrim(joint_path)
    UsdPhysics.FixedJoint(over).CreateBody0Rel().SetTargets([chassis.GetPath()])
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path)}


def command_at(profile, step):
    """Cosine ramp 0->1 over ramp_s, then hold. Returns (pos_xyz, yaw_rad, smooth)."""
    spec = PROFILES[profile]
    ramp_steps = max(1, int(spec["ramp_s"] / DT))
    t = min(1.0, step / ramp_steps)
    smooth = 0.5 - 0.5 * math.cos(math.pi * t)
    pos = np.array([spec["x"] * smooth, spec["y"] * smooth, LIFT + spec["z"] * smooth])
    return pos, spec["yaw"] * smooth, smooth


def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    arm_hash_before = sha256(REAL_ARM)
    spec = PROFILES[ARGS.profile]
    authored = author(os.path.join(out_dir, "authored_p3_scene.usda"))

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0", gravity=(0.0, 0.0, float(ARGS.gravity_z)), physics=physics))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"], activate_contact_sensors=True)
    spawn.func("/World/PM", spawn)

    sensors = {}
    contact_api = "UNKNOWN"
    if ARGS.read_contacts:
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            for name, prim_path in (("chassis", "/World/PM/Chassis"),
                                    ("arm", "/World/PM/Arm/.*")):
                try:
                    sensors[name] = ContactSensor(ContactSensorCfg(
                        prim_path=prim_path, update_period=0.0, history_length=0,
                        debug_vis=False))
                except Exception as exc:
                    contact_api = "partial: %s failed (%s)" % (name, type(exc).__name__)
            if sensors:
                contact_api = "isaaclab.sensors.ContactSensor(%s)" % ",".join(sensors)
        except Exception as exc:  # pragma: no cover
            contact_api = "UNKNOWN: %s" % type(exc).__name__

    actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0,
                                            damping=20.0)}
    robot = Articulation(ArticulationCfg(prim_path="/World/PM", spawn=None, actuators=actuators))
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    index_of = {name: position for position, name in enumerate(body_names)}
    chassis_index = index_of.get("Chassis")
    base_index = index_of.get("base_link")
    if chassis_index is None:
        raise SystemExit("chassis body not found in the articulation")

    quat_order = {"value": "wxyz", "reason": "probe matched the first convention tried"}

    # The articulation ROOT is the arm's own root prim, whose authored world pose is NOT
    # (0,0,LIFT) with identity orientation: the referenced asset carries its own base
    # transform.  Writing an absolute "identity at (0,0,LIFT)" therefore teleports the root
    # somewhere the chassis does not follow exactly, which is why P3_static_r05/r07 reported
    # t0_pose_write_error_m = 0.2615 m while mount_stable stayed at 3e-8.  The command is
    # therefore applied as a DELTA on top of the root pose measured right after reset.
    base_root_pos = [float(v) for v in _np(robot.data.root_pos_w[0]).reshape(-1)[:3]]
    base_root_quat = raw_to_wxyz(_np(robot.data.root_quat_w[0]).reshape(-1)[:4])
    base_root_yaw = yaw_of(base_root_quat)
    base_chassis_pos = [float(v) for v in
                        _np(robot.data.body_pos_w[0, chassis_index]).reshape(-1)[:3]]
    # The chassis is a CHILD body of the articulation root, so the two frames differ by a
    # constant offset (~0.208 m here).  A root-pose command is therefore NOT a chassis
    # command: P3_compound_r09 realized only 0.1812 m of a commanded 0.25 m because the yaw
    # arc partly cancelled the translation.  The offset is expressed in the ROOT frame and
    # subtracted when commanding, so the CHASSIS lands where the gate expects it.
    offset_world = np.array(base_chassis_pos) - np.array(base_root_pos)
    base_rotation = rotation_matrix(base_root_quat)
    offset_root_frame = base_rotation.T @ offset_world

    def negative_control_record(spec):
        """A negative control is only meaningful if the frozen component was non-zero.

        P3_neg_frozen_y_r01 froze Y on the translate_xy profile, whose command is
        x=0.3, y=0.0: the freeze was a no-op, the run was bit-identical to P3_xy_r14 and
        the gate PASSED. That is a VACUOUS negative control, not a gate hole, so it is
        recorded as such and forced to FAIL (a vacuous control may never count as PASS).
        """
        component = {"x": float(spec["x"]), "y": float(spec["y"]), "z": float(spec["z"]),
                     "yaw": float(spec["yaw"])}.get(ARGS.freeze_axis, 0.0)
        vacuous = abs(component) < 1e-9
        return {"frozen_axis": ARGS.freeze_axis,
                "frozen_command_component": component,
                "vacuous": vacuous,
                "note": ("the frozen command component was ZERO, so dropping it changes "
                         "nothing: this run cannot serve as a negative control"
                         if vacuous else
                         "command component dropped on purpose; the corresponding motion "
                         "gate MUST fail")}

    def write_pose(pos_cmd, yaw_cmd):
        """pos_cmd/yaw_cmd are absolute CHASSIS commands; converted to a root pose."""
        pos_cmd = [float(pos_cmd[0]), float(pos_cmd[1]), float(pos_cmd[2])]
        yaw_cmd = float(yaw_cmd)
        if ARGS.freeze_axis == "x":
            pos_cmd[0] = 0.0
        elif ARGS.freeze_axis == "y":
            pos_cmd[1] = 0.0
        elif ARGS.freeze_axis == "z":
            pos_cmd[2] = LIFT
        elif ARGS.freeze_axis == "yaw":
            yaw_cmd = 0.0
        yaw = base_root_yaw + float(yaw_cmd)
        half = 0.5 * yaw
        if quat_order["value"] == "wxyz":
            quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
        else:
            quat = [0.0, 0.0, math.sin(half), math.cos(half)]
        rotation = rotation_matrix([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
        chassis_cmd = np.array([float(pos_cmd[0]), float(pos_cmd[1]), float(pos_cmd[2])])
        root_pos = chassis_cmd - rotation @ offset_root_frame
        pos = [float(root_pos[0]), float(root_pos[1]), float(root_pos[2])]
        pose = [pos[0], pos[1], pos[2]] + [float(v) for v in quat]
        robot.write_root_pose_to_sim(
            torch.tensor([pose], dtype=torch.float32, device=robot.device))
        # A teleport without clearing the velocity leaves residual linear/angular velocity in
        # PhysX, which then yanks the arm joints (seen in P3_static_r01: joint deviation
        # 0.205 rad and 1.49 m/s peak from a "static" profile).  Diagnostics drive = kinematic.
        robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=torch.float32, device=robot.device))

    def read_chassis():
        raw = [float(v) for v in
               _np(robot.data.body_quat_w[0, chassis_index]).reshape(-1)[:4]]
        pos = [float(v) for v in
               _np(robot.data.body_pos_w[0, chassis_index]).reshape(-1)[:3]]
        return pos, raw_to_wxyz(raw)

    # Convention probe: a unit quaternion cannot distinguish wxyz from xyzw, so write a known
    # 0.2 rad yaw in BOTH candidate orders and keep the one whose MEASURED yaw matches.
    #
    # Two defects made the first two attempts useless and are fixed here:
    #   * P3_static_r01/P3_static_r03 decided the order from a single candidate whose wrong-order
    #     quaternion is a ~180 deg flip, and robot.update() without a physics advance reports a
    #     STALE pose, so the probe "measured" yaw 0 even for the correct order and picked the
    #     wrong one.  The probe now advances the physics (sim.forward()) before reading, and
    #     tests both orders instead of trusting the first.
    def probe_order(candidate, target_yaw=0.2):
        saved = quat_order["value"]
        quat_order["value"] = candidate
        write_pose(np.array([0.0, 0.0, LIFT]), target_yaw)
        sim.forward()
        robot.update(DT)
        _, probe_quat = read_chassis()
        measured = yaw_of(probe_quat) - base_chassis_yaw   # delta, not absolute
        quat_order["value"] = saved
        return float(measured)

    sim.reset()
    base_chassis_yaw = yaw_of(read_chassis()[1])
    probe_results = {"wxyz": probe_order("wxyz"), "xyzw": probe_order("xyzw")}
    sim.reset()
    best = min(probe_results, key=lambda name: abs(probe_results[name] - 0.2))
    quat_order["value"] = best
    quat_order["reason"] = ("both candidate orders were written and measured; %s reproduced "
                            "the commanded 0.2 rad yaw" % best)
    quat_order["probe_measured_yaw_rad"] = {name: round(value, 9)
                                            for name, value in probe_results.items()}
    quat_order["probe_commanded_yaw_rad"] = 0.2
    quat_order["probe_absolute_error_rad"] = {
        name: round(abs(value - 0.2), 9) for name, value in probe_results.items()}
    sim.reset()
    # Establish step 0 explicitly with the confirmed order, verify it took effect.
    write_pose(np.array([0.0, 0.0, LIFT]), 0.0)
    sim.forward()
    robot.update(DT)
    start_pos, start_quat = read_chassis()
    start_error = float(np.linalg.norm(np.asarray(start_pos) - np.array([0.0, 0.0, LIFT])))

    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []
    joint_target = [float(v) for v in _np(robot.data.joint_pos[0]).reshape(-1)]

    joint_target_tensor = torch.tensor([joint_target], dtype=torch.float32,
                                       device=robot.device)

    def set_joint_targets():
        # Fail closed. Previously exceptions here were silently swallowed,
        # allowing "joint_hold PASS" when no joint target had been submitted.
        robot.set_joint_position_target(joint_target_tensor)

    def capture(tag, step, commanded_pos, commanded_yaw, smooth):
        row = {"run_id": ARGS.run_id, "profile": ARGS.profile, "tag": tag, "step": step,
               "t_s": round(step * DT, 9), "smooth": round(smooth, 9),
               "commanded": {"pos": [round(float(v), 9) for v in commanded_pos],
                             "yaw_rad": round(float(commanded_yaw), 9)},
               "bodies": {}}
        for name in body_names:
            index = index_of[name]
            raw = [float(v) for v in _np(robot.data.body_quat_w[0, index]).reshape(-1)[:4]]
            row["bodies"][name] = {
                "pos": [round(float(v), 12) for v in
                        _np(robot.data.body_pos_w[0, index]).reshape(-1)[:3]],
                "raw_quat": [round(v, 12) for v in raw],
                "normalized_quat_wxyz": [round(v, 12) for v in raw_to_wxyz(raw)],
                "lin_vel": [round(float(v), 9) for v in
                            _np(robot.data.body_lin_vel_w[0, index]).reshape(-1)[:3]],
                "ang_vel": [round(float(v), 9) for v in
                            _np(robot.data.body_ang_vel_w[0, index]).reshape(-1)[:3]],
            }
        chassis = row["bodies"]["Chassis"]
        row["measured_chassis_yaw_rad"] = round(yaw_of(chassis["normalized_quat_wxyz"]), 9)
        row["joint_pos"] = [round(float(v), 9) for v in _np(robot.data.joint_pos[0]).reshape(-1)]
        row["joint_vel"] = [round(float(v), 9) for v in _np(robot.data.joint_vel[0]).reshape(-1)]
        if base_index is not None:
            rotation = rotation_matrix(chassis["normalized_quat_wxyz"])
            relative = rotation.T @ (np.asarray(row["bodies"]["base_link"]["pos"])
                                    - np.asarray(chassis["pos"]))
            row["mount_relative_pos"] = [round(float(v), 12) for v in relative]
            row["mount_relative_rot_rad"] = round(angle_between(
                chassis["normalized_quat_wxyz"],
                row["bodies"]["base_link"]["normalized_quat_wxyz"]), 12)
        contacts = {}
        for name, sensor in sensors.items():
            try:
                data = _np(sensor.data.net_forces_w).reshape(-1, 3)
                contacts[name] = round(float(np.max(np.linalg.norm(data, axis=1))), 9)
            except Exception:
                contacts[name] = None
        row["contact_by_sensor_n"] = contacts
        row["finite"] = all(
            math.isfinite(v) for entry in row["bodies"].values()
            for key in ("pos", "raw_quat", "lin_vel", "ang_vel") for v in entry[key])
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)

    # EVERY frame, including step 0, is produced by "write the commanded root pose, then step
    # once".  The chassis is a CHILD rigid body linked to the arm root by a FixedJoint, so a
    # teleport of the root does not relocate it -- only the solver does.  Reading it before a
    # solver step gave the flipped leftover pose: P3_static_r05 reported
    # t0_pose_write_error_m = 0.2615 m even though the write itself used the now-confirmed
    # xyzw order (probe: xyzw 0.199999853 rad vs commanded 0.2; wxyz 9.8e-08).
    def write_fault_record(code, detail):
        try:
            with open(os.path.join(ARGS.out_dir, "fault_record.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"run_id": ARGS.run_id, "fault": ARGS.inject_fault,
                           "code": code, "detail": detail,
                           "sim_steps_taken": len(frames),
                           "note": ("typed refusal: the gate runner must surface this as a "
                                    "non-zero child code, never as a PASS")},
                          handle, ensure_ascii=False, indent=1)
        except Exception:
            pass

    if ARGS.inject_fault == "joint_limit":
        # Refuse BEFORE the loop.  Raising SystemExit from inside the stepping callback made
        # Kit die with SIGSEGV (child rc 139, P0F_jointlimit_r01) - fail-closed but ugly and
        # not a typed refusal, so the check is done once, up front, with a clean exit code.
        index = 2 if len(joint_names) > 2 else 0
        upper = float(_np(robot.data.joint_pos_limits[0, index, 1]))
        bad_target = upper + 0.5
        if bad_target > upper:
            write_fault_record("JOINT_LIMIT_REFUSED",
                               {"joint": joint_names[index], "target_rad": bad_target,
                                "upper_limit_rad": upper, "targets_written": 0})
            print("JOINT_LIMIT_REFUSED: joint %r target %.6f rad exceeds the upper limit "
                  "%.6f rad; refusing before any write" % (joint_names[index], bad_target,
                                                           upper), flush=True)
            jsonl.close()
            # P0-07 finding 2: returning from main() WITHOUT closing the Kit app left the
            # process exiting 0 (child_returncode 0 in P0F_jointlimit_r02), so any tool keyed
            # on the child code would read a false PASS.  Close, then raise.
            try:
                SIMULATION_APP.close()
            except Exception:
                pass
            raise SystemExit(3)

    for step in range(ARGS.steps + 1):
        commanded_pos, commanded_yaw, smooth = command_at(ARGS.profile, step)
        write_pose(commanded_pos, commanded_yaw)
        set_joint_targets()
        if ARGS.inject_fault == "python_exception" and step == 60:
            # P0-07 finding: an exception raised inside the loop was caught by Kit, the
            # traceback printed, and the process still exited 0 (P0F_exception_r01 child rc 0).
            # The script therefore records a typed fault and exits non-zero itself.
            write_fault_record("INJECTED_PYTHON_EXCEPTION", {"step": step})
            print("INJECTED_PYTHON_EXCEPTION at step %d" % step, flush=True)
            jsonl.close()
            try:
                SIMULATION_APP.close()
            except Exception:
                pass
            raise SystemExit(4)
        sim.step()
        robot.update(DT)
        for sensor in sensors.values():
            try:
                sensor.update(DT)
            except Exception:
                pass
        capture("drive", step, commanded_pos, commanded_yaw, smooth)
    jsonl.close()

    chassis_pos = np.array([frame["bodies"]["Chassis"]["pos"] for frame in frames])
    yaws = np.array([frame["measured_chassis_yaw_rad"] for frame in frames])
    commanded_pos_all = np.array([frame["commanded"]["pos"] for frame in frames])
    commanded_yaw_all = np.array([frame["commanded"]["yaw_rad"] for frame in frames])
    tracking_pos_max = float(np.max(np.linalg.norm(chassis_pos - commanded_pos_all, axis=1)))
    tracking_yaw_max = float(np.max(np.abs(yaws - commanded_yaw_all)))
    realized_translation = float(np.max(np.linalg.norm(chassis_pos - chassis_pos[0], axis=1)))
    realized_yaw = float(np.max(np.abs(yaws - yaws[0])))

    mount_pos = mount_rot = mount_pos_max = mount_rot_max = None
    if base_index is not None:
        relative0 = np.array(frames[0]["mount_relative_pos"])
        mount_pos = float(np.linalg.norm(relative0))
        mount_rot = float(frames[0]["mount_relative_rot_rad"])
        mount_pos_max = max(float(np.linalg.norm(np.array(frame["mount_relative_pos"]) - relative0))
                            for frame in frames)
        mount_rot_max = max(abs(frame["mount_relative_rot_rad"] - mount_rot) for frame in frames)
    joint_dev = max(float(np.max(np.abs(np.array(frame["joint_pos"]) - np.array(joint_target))))
                    for frame in frames)
    speeds = [float(np.linalg.norm(frame["bodies"]["Chassis"]["lin_vel"])) for frame in frames]
    omegas = [float(np.linalg.norm(frame["bodies"]["Chassis"]["ang_vel"])) for frame in frames]
    accelerations = np.diff(np.array(speeds)) / DT
    jerks = np.diff(accelerations) / DT
    arm_forces = [frame["contact_by_sensor_n"].get("arm") for frame in frames]
    arm_peak = max([v for v in arm_forces if v is not None], default=None)

    thresholds = {"mount_pos_m": 0.001, "mount_rot_rad": 0.001, "joint_hold_rad": 0.05,
                  "motion_ratio": 0.9, "arm_ground_contact_n": 1.0}
    gates = {
        "all_finite": all(frame["finite"] for frame in frames),
        "mount_stable": bool(mount_pos_max is not None
                             and mount_pos_max <= thresholds["mount_pos_m"]
                             and mount_rot_max <= thresholds["mount_rot_rad"]),
        "joint_hold": bool(joint_dev <= thresholds["joint_hold_rad"]),
    }
    # P3 v1.1: include Z and require directed displacement, not arbitrary
    # same-magnitude motion. The old hypot(x,y) skipped translate_z entirely.
    directed = directed_displacement_track(
        [spec["x"], spec["y"], spec["z"]], chassis_pos, thresholds["motion_ratio"])
    if directed is not None:
        gates["translation_actually_measured"] = directed["pass"]
    if abs(spec["yaw"]) > 1e-9:
        gates["yaw_actually_measured"] = bool(
            realized_yaw >= thresholds["motion_ratio"] * abs(spec["yaw"]))
    if arm_peak is not None:
        gates["arm_does_not_touch_ground"] = bool(arm_peak <= thresholds["arm_ground_contact_n"])
    if ARGS.freeze_axis != "none" and negative_control_record(spec)["vacuous"]:
        # a negative control whose frozen component was already zero proves nothing, so it
        # may never be counted as a PASS (P3_neg_frozen_y_r01 was exactly this case)
        gates["negative_control_is_meaningful"] = False
    gate = "PASS" if all(gates.values()) else "FAIL"

    arm_hash_after = sha256(REAL_ARM)
    result = {
        "run_id": ARGS.run_id, "scene_variant": "P3_DYNAMIC_MOUNT",
        "profile": ARGS.profile, "drive_mode": "DIAGNOSTIC_ROOT_POSE (NOT wheel drive)",
        "gate_result": gate,
        "metrics": {
            "command_amplitude": {"x_m": spec["x"], "y_m": spec["y"], "z_m": spec["z"],
                                  "yaw_rad": spec["yaw"], "ramp_s": spec["ramp_s"]},
            "realized_translation_max_m": round(realized_translation, 9),
            "directed_translation": directed,
            "realized_yaw_max_rad": round(realized_yaw, 9),
            "tracking_position_error_max_m": round(tracking_pos_max, 9),
            "tracking_yaw_error_max_rad": round(tracking_yaw_max, 9),
            "t0_mount_position_error_m": None if mount_pos is None else round(mount_pos, 12),
            "mount_position_deviation_max_m": (None if mount_pos_max is None
                                               else round(mount_pos_max, 12)),
            "mount_rotation_deviation_max_rad": (None if mount_rot_max is None
                                                 else round(mount_rot_max, 12)),
            "joint_deviation_max_rad": round(joint_dev, 9),
            "peak_lin_speed_m_s": round(max(speeds), 9),
            "peak_ang_speed_rad_s": round(max(omegas), 9),
            "peak_jerk_m_s3": round(float(np.max(np.abs(jerks))), 6) if jerks.size else None,
            "arm_peak_contact_n": arm_peak,
            "steps": ARGS.steps,
        },
        "thresholds": thresholds, "gates": gates,
        # explicit generation tag: P3_EVIDENCE_SUMMARY had to guess v1.0 vs v1.1 from the
        # metrics shape, and mis-classified the yaw profile (no directed_translation block).
        "gate_version": GATE_VERSION,
        "gate_preregistration": "management/morph_one_rs/PHASE_P3_GATE_V1_1_AMENDMENT.md",
        "diagnostics": {
            "negative_control": (negative_control_record(spec) if ARGS.freeze_axis != "none"
                                 else None),
            "quat_order_for_write_root_pose": quat_order,
            "t0_pose_write_error_m": round(start_error, 12),
            "root_to_chassis_offset_m": round(float(np.linalg.norm(offset_world)), 9),
            "root_to_chassis_offset_root_frame_m": [round(float(v), 9)
                                                    for v in offset_root_frame],
            "root_pose_is_offset_compensated": True,
            "joint_targets_set_each_step": True,
            "root_velocity_zeroed_with_each_teleport": True,
            "contact_api": contact_api,
            "body_names": body_names, "joint_names": joint_names,
            "num_bodies": len(body_names), "dofs": int(robot.data.joint_pos.shape[1]),
            "arm_sha256_before": arm_hash_before, "arm_sha256_after": arm_hash_after,
            "arm_asset_unchanged": arm_hash_before == arm_hash_after,
            "note": ("root-pose drive is diagnostic only; it must never be reported as "
                     "P4 wheel-drive evidence"),
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage": "P3", "run_id": ARGS.run_id, "scene_variant": "P3_DYNAMIC_MOUNT",
                   "profile": ARGS.profile, "physics_dt_s": DT, "steps": ARGS.steps,
                   "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
                   "lift_m": LIFT, "arm_asset": REAL_ARM, "arm_sha256": arm_hash_after,
                   "arm_asset_unchanged": arm_hash_before == arm_hash_after,
                   "drive_mode": "DIAGNOSTIC_ROOT_POSE (NOT wheel drive)",
                   "freeze_axis": ARGS.freeze_axis,
                   "preregistration": "management/morph_one_rs/PHASE_P3_DYNAMIC_MOUNT_GATE.md",
                   "command": " ".join(os.sys.argv)}, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"run_id": ARGS.run_id, "profile": ARGS.profile, "gate_result": gate,
                      "metrics": result["metrics"], "gates": gates},
                     ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate, flush=True)
    SIMULATION_APP.close()
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
