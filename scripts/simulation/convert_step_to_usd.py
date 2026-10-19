"""Headless STEP/IGES -> USD conversion through the Isaac Sim CAD converter.

The CAD converter in Isaac Sim is shipped as Kit extensions
(omni.kit.converter.cad + omni.kit.converter.hoops_core).  Those extensions
cannot be imported from a plain interpreter: they only exist inside a running
Kit application.  This script therefore starts a headless Kit app, enables the
converter, and drives the documented HoopsCoreConverter API
(create_converter_task) from inside that app - no GUI, no HTTP service, no
clicking in the UI.

The script never reports success just because a process exited cleanly.  After
conversion it re-opens the produced USD and records prim types, mesh and point
counts, stage units, up-axis and the world bounding box, so a reviewer can check
the numbers instead of trusting a log line.

Exit codes:
  0 - conversion returned success AND the output verifies
  2 - converter returned a non-zero status / did not complete
  3 - converter claimed success but the output is missing, unreadable or empty
  4 - the converter extensions could not be enabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_format_args(options: dict) -> dict:
    """Translate converter options into the FileFormatArguments the API expects.

    HoopsCoreConverter.create_converter_task takes Dict[str, str]: option keys
    stay in their camelCase spelling and every value must be a string, with
    booleans spelled in lower case.  Passing native Python types raises
    "parseArgs(): incompatible function arguments".
    """
    args: dict = {}
    for key, value in options.items():
        if isinstance(value, bool):
            args[key] = "true" if value else "false"
        else:
            args[key] = str(value)
    return args


def _ensure_converter_enabled() -> dict:
    """Enable the CAD converter extensions inside the running Kit app."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    wanted = [
        "omni.kit.converter.cad",
        "omni.kit.converter.hoops",
        "omni.kit.converter.hoops_core",
    ]
    state = {}
    for name in wanted:
        enabled = manager.is_extension_enabled(name)
        if not enabled:
            enabled = bool(manager.set_extension_enabled_immediate(name, True))
        state[name] = bool(enabled)
    return state


def _run_conversion(input_path: Path, output_path: Path, options: dict, max_seconds: float) -> dict:
    """Drive HoopsCoreConverter.create_converter_task inside the Kit app."""
    import omni.kit.app

    from omni.kit.converter.hoops_core import get_instance

    app = omni.kit.app.get_app()
    converter = get_instance()
    coroutine = converter.create_converter_task(str(input_path), str(output_path), options)

    started = time.time()
    deadline = started + max_seconds
    result: dict = {
        "completed": False,
        "error": None,
        "elapsed_seconds": None,
        "output_url": "",
        "status_error_code": None,
        "status_error_msg": None,
    }

    def _record(value) -> None:
        output_url, status = value
        result["output_url"] = str(output_url)
        result["status_error_code"] = getattr(status, "error_code", None)
        result["status_error_msg"] = getattr(status, "error_msg", None)
        result["completed"] = True

    try:
        import omni.kit.async_engine

        future = omni.kit.async_engine.run_coroutine(coroutine)
        while not future.done():
            app.update()
            if time.time() > deadline:
                future.cancel()
                result["error"] = "TimeoutError: conversion exceeded %.0f s" % max_seconds
                break
        else:
            _record(future.result())
    except ImportError:
        import asyncio

        try:
            _record(asyncio.run(coroutine))
        except Exception as exc:  # noqa: BLE001 - status must be reported, not hidden
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001
        result["error"] = "%s: %s" % (type(exc).__name__, exc)

    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def _verify_usd(path: Path) -> dict:
    """Re-open the produced USD and describe what is actually inside it."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return {"opened": False}

    # The CAD converter stores geometry under an abstract "Prototypes" scope
    # (Usd class prims) and references it from instanceable prims.  The default
    # Stage.Traverse() skips abstract prims, so a naive traversal reports zero
    # meshes for a perfectly good conversion.  Count prototypes with
    # TraverseAll() and rendered mesh instances with TraverseInstanceProxies().
    all_prims = list(stage.TraverseAll())
    type_counts: dict = {}
    prim_count = 0
    mesh_count = 0
    total_points = 0
    total_faces = 0
    abstract_prims = 0
    instanceable_prims = 0
    for prim in all_prims:
        prim_count += 1
        type_name = prim.GetTypeName()
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        if prim.IsAbstract():
            abstract_prims += 1
        if prim.IsInstanceable():
            instanceable_prims += 1
        if type_name == "Mesh":
            mesh_count += 1
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                total_points += len(points)
            counts = mesh.GetFaceVertexCountsAttr().Get()
            if counts:
                total_faces += len(counts)

    rendered_meshes = 0
    rendered_points = 0
    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        if prim.GetTypeName() != "Mesh":
            continue
        rendered_meshes += 1
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points:
            rendered_points += len(points)

    bbox = None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    root_bound = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    if not root_bound.IsEmpty():
        minimum = root_bound.GetMin()
        maximum = root_bound.GetMax()
        bbox = {
            "min": [minimum[0], minimum[1], minimum[2]],
            "max": [maximum[0], maximum[1], maximum[2]],
            "size": [
                maximum[0] - minimum[0],
                maximum[1] - minimum[1],
                maximum[2] - minimum[2],
            ],
        }

    default_prim = stage.GetDefaultPrim()
    return {
        "opened": True,
        "default_prim": str(default_prim.GetPath()) if default_prim else "",
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "prim_count": prim_count,
        "abstract_prim_count": abstract_prims,
        "instanceable_prim_count": instanceable_prims,
        "mesh_count": mesh_count,
        "total_mesh_points": total_points,
        "total_mesh_faces": total_faces,
        "rendered_mesh_instances": rendered_meshes,
        "rendered_mesh_points": rendered_points,
        "prim_type_counts": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])[:15]),
        "world_bbox": bbox,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert a CAD file to USD headlessly.")
    parser.add_argument("--input", required=True, help="STEP/IGES file to convert")
    parser.add_argument("--output", required=True, help="destination USD path")
    parser.add_argument("--report", default="", help="JSON report path")
    parser.add_argument(
        "--options-json",
        default="",
        help="JSON dict of converter options (default: tessLOD=2, useMaterials/useNormals on)",
    )
    parser.add_argument("--max-seconds", type=float, default=7200.0)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)
    args_cli.headless = True
    simulation_app = AppLauncher(args_cli).app

    exit_code = 0
    report: dict = {
        "input": str(Path(args_cli.input).resolve()),
        "output": str(Path(args_cli.output).resolve()),
        "isaac_sim_version": _safe_version(),
    }
    try:
        input_path = Path(args_cli.input)
        output_path = Path(args_cli.output)
        if not input_path.is_file():
            report["fatal"] = "input file not found"
            report["verdict"] = "input file not found"
            exit_code = 4
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            options = {
                "tessLOD": 2,
                "useMaterials": True,
                "useNormals": True,
                "dedup": True,
                "instancing": True,
            }
            if args_cli.options_json:
                options = json.loads(Path(args_cli.options_json).read_text(encoding="utf-8"))
            format_args = _to_format_args(options)

            report["input_sha256"] = _sha256(input_path)
            report["input_bytes"] = input_path.stat().st_size
            report["extensions"] = _ensure_converter_enabled()
            report["options"] = options
            report["file_format_args"] = format_args

            if not all(report["extensions"].values()):
                report["verdict"] = "converter extensions could not be enabled"
                exit_code = 4
            else:
                result = _run_conversion(
                    input_path, output_path, format_args, args_cli.max_seconds
                )
                report["conversion"] = result

                if not result["completed"]:
                    report["verdict"] = "conversion did not complete - see conversion.error"
                    exit_code = 2
                elif result["status_error_code"] not in (0, None):
                    report["verdict"] = "converter returned error code %s" % result[
                        "status_error_code"
                    ]
                    exit_code = 2
                elif not output_path.is_file() or output_path.stat().st_size == 0:
                    report["verdict"] = "converter reported success but no output file exists"
                    exit_code = 3
                else:
                    report["output_bytes"] = output_path.stat().st_size
                    report["output_sha256"] = _sha256(output_path)
                    usd = _verify_usd(output_path)
                    report["usd"] = usd
                    if (
                        usd.get("opened")
                        and usd.get("mesh_count", 0) > 0
                        and usd.get("rendered_mesh_instances", 0) > 0
                        and usd.get("world_bbox")
                    ):
                        report["verdict"] = (
                            "converted and verified: %d prototype meshes / %d points, "
                            "%d rendered mesh instances, bbox size %s"
                            % (
                                usd["mesh_count"],
                                usd["total_mesh_points"],
                                usd["rendered_mesh_instances"],
                                [round(v, 2) for v in usd["world_bbox"]["size"]],
                            )
                        )
                        exit_code = 0
                    else:
                        report["verdict"] = "output exists but contains no meshes"
                        exit_code = 3
    finally:
        report["exit_code"] = exit_code
        text = json.dumps(report, indent=2)
        if args_cli.report:
            report_path = Path(args_cli.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        simulation_app.close()

    return exit_code


def _safe_version() -> str:
    try:
        from isaacsim import __version__ as isaacsim_version

        return str(isaacsim_version)
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
