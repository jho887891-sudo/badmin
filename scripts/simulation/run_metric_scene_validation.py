"""Run the metres-stage assembly validation inside Isaac Sim (headless).

It builds two scenes that reference the converted Morph One visual USD:

  A  metres stage, plain reference, no scale
  B  metres stage, reference with xformOp:scale = 0.001

and then reads the composed stage: prim census, the visual node's composed
local-to-world transform, the world bbox obtained by transforming every mesh
point of every instance proxy, and the bbox from UsdGeom.BBoxCache as a second
opinion.  No collider, mass, joint or arm attachment is created.

A screenshot is attempted and reported as available/unavailable together with the
reason; the run never pretends to have a picture it does not have.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

# Allow running this file directly (sys.path[0] would then be the package
# directory and "scripts.simulation" would be unimportable).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = ["main"]

SCALE = (0.001, 0.001, 0.001)


def _attempt_screenshot(scene_path: Path, png_path: Path, budget_seconds: float = 45.0):
    started = time.time()
    try:
        import omni.kit.viewport.utility as viewport_utility

        viewport = viewport_utility.get_active_viewport()
        if viewport is None:
            return {"available": False, "reason": "no active viewport in headless app"}
        import omni.usd

        context = omni.usd.get_context()
        context.open_stage(str(scene_path))
        while context.get_stage_loading_status()[2] > 0 and time.time() - started < budget_seconds:
            time.sleep(0.1)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        viewport_utility.capture_viewport_to_file(viewport, str(png_path))
        return {
            "available": png_path.is_file(),
            "path": str(png_path),
            "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001 - availability is reported, not raised
        return {
            "available": False,
            "reason": "%s: %s" % (type(exc).__name__, exc),
            "seconds": round(time.time() - started, 2),
        }


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the metres-stage assembly in Isaac Sim")
    parser.add_argument("--visual-usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-m", type=float, nargs=3, default=[0.580, 0.373, 0.803])
    parser.add_argument("--screenshot", action="store_true", help="best-effort viewport capture")
    parser.add_argument("--prim-sample", type=int, default=5)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)
    args_cli.headless = True
    simulation_app = AppLauncher(args_cli).app

    from scripts.simulation.morph_one_metric_scene import (
        build_metric_scene,
        envelope_within,
        evaluate,
        measure_scene,
        ratio_between,
        write_json,
    )

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_usd = Path(args_cli.visual_usd)

    scene_a = build_metric_scene(out_dir / "A_no_scale_metre_stage.usda", visual_usd, scale=None)
    scene_b = build_metric_scene(
        out_dir / "B_scale_0.001_metre_stage.usda", visual_usd, scale=SCALE
    )

    report_a = evaluate(measure_scene(scene_a, limit_prims=args_cli.prim_sample), args_cli.expected_m)
    report_b = evaluate(measure_scene(scene_b, limit_prims=args_cli.prim_sample), args_cli.expected_m)

    ratio = ratio_between(
        report_b.get("world_bbox_from_points_meters"), report_a.get("world_bbox_from_points_meters")
    )

    screenshots = {}
    if args_cli.screenshot:
        screenshots["B_scale_0.001_metre_stage"] = _attempt_screenshot(
            Path(scene_b), out_dir / "B_scale_0.001_metre_stage.png"
        )
        screenshots["A_no_scale_metre_stage"] = _attempt_screenshot(
            Path(scene_a), out_dir / "A_no_scale_metre_stage.png"
        )

    payload = {
        "runner": "isaac sim app (headless)",
        "visual_usd": str(visual_usd),
        "expected_envelope_m": list(args_cli.expected_m),
        "cases": {
            "A_no_scale": report_a,
            "B_scale_0.001": report_b,
        },
        "ratio_B_over_A": ratio,
        "verdict": {
            "A_matches_expected": report_a.get("matches_expected_envelope"),
            "B_matches_expected": report_b.get("matches_expected_envelope"),
            "B_is_thousandth_of_A": ratio == [0.001, 0.001, 0.001],
            "both_have_meshes": report_a.get("mesh_instances", 0) > 0
            and report_b.get("mesh_instances", 0) > 0,
            "both_free_of_physics_apis": all(
                value == 0
                for case in (report_a, report_b)
                for value in (case.get("physx_apis") or {}).values()
            ),
        },
        "screenshots": screenshots,
        "notes": [
            "unit and composition test only: no collider, mass, joint or arm attachment",
            "the measured envelope is conversion geometry, not a verified chassis dimension",
        ],
    }
    payload["verdict"]["ok"] = (
        bool(payload["verdict"]["B_matches_expected"])
        and not bool(payload["verdict"]["A_matches_expected"])
        and bool(payload["verdict"]["B_is_thousandth_of_A"])
        and bool(payload["verdict"]["both_have_meshes"])
        and bool(payload["verdict"]["both_free_of_physics_apis"])
    )

    write_json(args_cli.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1), flush=True)
    simulation_app.close()
    return 0 if payload["verdict"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
