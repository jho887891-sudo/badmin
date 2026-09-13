# -*- coding: utf-8 -*-
"""
t01_convert_urdf.py - Convert AgileX PiPER (6-DOF, no gripper) URDF to USD.

Uses the official Isaac Sim URDF importer extension (isaacsim.asset.importer.urdf).
Outputs: assets/piper_no_gripper.usd (canonical binary USD)
Run (after sourcing robot_sim env.sh):
    ./isaaclab.sh -p <this_file>
"""
from pathlib import Path

from isaacsim import SimulationApp

SIM_APP = SimulationApp({"headless": True})
print("[t01] SimulationApp launched", flush=True)

try:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
except Exception as exc:  # noqa: BLE001
    print("[t01] enable_extension skipped:", exc, flush=True)

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

STAGE = Path("/home/T7/ojh/robot_sim/assets/piper_stage0")
SRC_URDF = STAGE / "assets/piper_isaac_sim/piper_description/urdf/piper_no_gripper_description.urdf"
MESHES_DIR = STAGE / "assets/piper_isaac_sim/piper_description/meshes"
WORK_DIR = STAGE / "work"
USD_DIR = STAGE / "usd_piper_no_gripper"
CANON_USD = STAGE / "assets" / "piper_no_gripper.usd"


def main() -> None:
    assert SRC_URDF.exists(), f"source URDF missing: {SRC_URDF}"
    for d in (WORK_DIR, USD_DIR, CANON_USD.parent):
        d.mkdir(parents=True, exist_ok=True)

    abs_urdf = WORK_DIR / "piper_no_gripper_abs.urdf"
    text = SRC_URDF.read_text(encoding="utf-8")
    text = text.replace("package://piper_description/meshes/", str(MESHES_DIR) + "/")
    abs_urdf.write_text(text, encoding="utf-8")
    print(f"[t01] rewrote URDF mesh paths -> {abs_urdf}", flush=True)

    cfg = URDFImporterConfig(
        urdf_path=str(abs_urdf),
        usd_path=str(USD_DIR),
        merge_fixed_joints=False,
        merge_mesh=False,
        fix_base=True,
        collision_from_visuals=False,
        allow_self_collision=False,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    importer = URDFImporter(config=cfg)
    out = Path(importer.import_urdf(cfg))
    print(f"[t01] import_urdf returned: {out}", flush=True)
    assert out.exists(), f"returned USD missing: {out}"

    from pxr import Usd, UsdPhysics  # noqa: E402

    stage = Usd.Stage.Open(str(out))
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    print(f"[t01] articulation_roots={roots}", flush=True)
    print(f"[t01] joint_prims_count={len(joints)}", flush=True)
    print("[t01] joint_prims=" + ",".join(joints), flush=True)

    stage.Export(str(CANON_USD))
    print(f"[t01] canonical USD -> {CANON_USD}", flush=True)
    print(f"[t01] canonical size bytes={CANON_USD.stat().st_size}", flush=True)
    SIM_APP.close()
    print("[t01] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        try:
            SIM_APP.close()
        finally:
            pass