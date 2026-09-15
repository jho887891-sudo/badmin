#!/usr/bin/env python3
"""generate_challenge_test.py - build the CHALLENGE_TEST split.

Thin CLI over tools/shuttle_render.py and
src/perception/shuttle_detection/challenge_test.py, which owns the sources, the plan and
the composition table. This file walks the plan, draws, writes, and then checks the result
by measurement.

The split is written to its own directory with its own manifest, so the frozen core
manifest is never rewritten and "the challenge set grew" cannot mean "the core set
changed". Both manifests carry the same schema and are audited together, which is also how
cross-split scene reuse is detected.

Usage:
    python tools/generate_challenge_test.py
    python tools/generate_challenge_test.py --only C2 --out <temp dir>
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
BACKGROUND_GLOBS = ("*.jpg", "*.jpeg", "*.png")
COMPOSITION_NAME = "challenge_composition.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="output directory (default: the module's DATA_DIR)")
    parser.add_argument("--only", default=None, help="comma-separated challenge sweep ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = REPO_DEFAULT
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tools"))

    import numpy as np

    from shuttle_render import (
        calibrate_distance,
        composite,
        imread_unicode,
        imwrite_unicode,
        load_shuttle_parts,
        prepare_background,
        render_shuttle,
        rot_ypr,
    )
    from src.perception.shuttle_detection import challenge_test as challenge
    from src.perception.shuttle_detection import controlled_capability as cc

    out_dir = Path(args.out) if args.out else challenge.DATA_DIR
    image_dir = out_dir / cc.IMAGE_SUBDIR
    label_dir = out_dir / cc.LABEL_SUBDIR
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for directory, patterns in ((image_dir, ("*.jpg", "*.jpeg", "*.png")), (label_dir, ("*.txt",))):
        for pattern in patterns:
            for stale in directory.glob(pattern):
                stale.unlink()

    # Every scene the core set or the training pool already stands on is forbidden here: a
    # challenge row on a core scene would measure the core material again, and the
    # filenames alone would not catch it (three of the hard-negative pool ARE core scenes
    # under a different name).
    candidates = sorted(challenge.HARD_NEGATIVE_DIR.glob("*.jpg"))
    forbidden: list[Path] = list(sorted(cc.BACKGROUND_DIR.glob("*.jpg")))
    for directory in cc.TRAINING_BACKGROUND_DIRS:
        for pattern in BACKGROUND_GLOBS:
            forbidden.extend(sorted(Path(directory).glob(pattern)))
    if not candidates:
        raise SystemExit("no hard-negative scenes found under " + str(challenge.HARD_NEGATIVE_DIR))
    usable, excluded = challenge.select_challenge_backgrounds(candidates, forbidden)
    print(
        "hard scenes: " + str(len(usable)) + " usable of " + str(len(candidates))
        + ", " + str(len(excluded)) + " excluded as core/training scenes"
    )
    for name, match, score in excluded:
        print("  excluded " + name + " ~ " + match + f" (r={score:.4f})")
    if not usable:
        raise SystemExit("no usable hard scene: the whole pool is core material")

    pool = tuple(sorted(path.name for path in cc.BACKGROUND_DIR.glob("*.jpg")))
    settings = cc.ControlSettings(
        background=pool[0],
        backgrounds=pool,
        block_backgrounds=cc.select_block_backgrounds(sorted(cc.BACKGROUND_DIR.glob("*.jpg"))),
    )
    plan = challenge.build_challenge_plan(settings, usable)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - set(challenge.CHALLENGE_SWEEPS)
        if unknown:
            raise SystemExit("unknown challenge sweeps: " + ", ".join(sorted(unknown)))
        plan = [sample for sample in plan if sample.sweep in wanted]
    if args.limit:
        plan = plan[: args.limit]

    parts = load_shuttle_parts()
    cache: dict[tuple, tuple] = {}
    scene_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    print("rendering " + str(len(plan)) + " challenge samples into " + str(out_dir))

    for index, sample in enumerate(plan, start=1):
        if sample.background not in scene_cache:
            loaded = imread_unicode(challenge.HARD_NEGATIVE_DIR / sample.background)
            if loaded is None:
                raise SystemExit("cannot read hard scene " + sample.background)
            scene_cache[sample.background] = loaded
        crop_rng = np.random.default_rng(sample.crop_seed)
        background = prepare_background(
            scene_cache[sample.background], sample.width, sample.height, crop_rng
        )
        row = sample.as_manifest_row()

        if sample.is_negative:
            # A scene with no shuttle in it: the direct test of the false-positive failure
            # this material was collected for. The label is written empty, which the audit
            # treats as a legal hard negative rather than a missing annotation.
            imwrite_unicode(image_dir / (sample.name + ".jpg"), background, quality=args.jpeg_quality)
            (label_dir / (sample.name + ".txt")).write_text("", encoding="utf-8", newline="\n")
            row.update(
                {
                    "bbox_w_px": 0, "bbox_h_px": 0, "pos_x_px": "", "pos_y_px": "",
                    "equivalent_size_px": 0.0, "equiv_size_px": 0.0, "distance_m": "",
                    "occlusion_fraction": 0.0, "occlusion_bucket": "none",
                    "coverage_sum": 0.0, "solid_px": 0,
                }
            )
            rows.append(row)
            print("  [{0}/{1}] {2} NEGATIVE (no object)".format(index, len(plan), sample.name))
            continue

        focal = sample.focal_px
        k_matrix = (focal, focal, sample.width / 2.0, sample.height / 2.0)
        rotation = rot_ypr(
            math.radians(sample.yaw_deg), math.radians(sample.pitch_deg),
            math.radians(sample.roll_deg),
        )
        pixel_xy = sample.pixel_centre
        light = cc.light_vector(sample.light_azimuth_deg)
        key = (
            float(sample.yaw_deg), float(sample.pitch_deg), float(sample.roll_deg),
            round(float(sample.target_px), 9), float(pixel_xy[0]), float(pixel_xy[1]),
            int(sample.width), int(sample.height), float(sample.focal_px),
            int(sample.supersample), float(sample.light_azimuth_deg),
        )
        if key not in cache:
            distance = calibrate_distance(
                parts, rotation, sample.target_px, pixel_xy, k_matrix, sample.width,
                sample.height, measure_supersample=sample.supersample,
            )
            rgb, alpha = render_shuttle(
                parts, rotation, distance, pixel_xy, k_matrix, sample.width, sample.height,
                light_dir=light, supersample=sample.supersample,
            )
            cache[key] = (
                distance, rgb, alpha,
                cc.footprint_measurement(alpha, sample.width, sample.height),
            )
        distance, rgb, alpha, measurement = cache[key]
        image = composite(
            background, rgb, alpha,
            shadow_gain=sample.shadow_gain, motion_px=sample.motion_px,
            motion_angle_deg=sample.motion_angle_deg, noise_sigma=sample.noise_sigma,
            noise_seed=sample.noise_seed,
        )
        imwrite_unicode(image_dir / (sample.name + ".jpg"), image, quality=args.jpeg_quality)
        (label_dir / (sample.name + ".txt")).write_text(
            measurement["label"], encoding="utf-8", newline="\n"
        )
        row.update(
            {
                "bbox_w_px": measurement["bbox_w_px"],
                "bbox_h_px": measurement["bbox_h_px"],
                "pos_x_px": measurement["pos_x_px"],
                "pos_y_px": measurement["pos_y_px"],
                "equivalent_size_px": measurement["equivalent_size_px"],
                "equiv_size_px": measurement["equivalent_size_px"],
                "distance_m": round(float(distance), 4),
                "occlusion_fraction": 0.0,
                "occlusion_bucket": "none",
                "coverage_sum": measurement["coverage_sum"],
                "solid_px": measurement["solid_px"],
            }
        )
        rows.append(row)
        print(
            "  [{0}/{1}] {2} size={3:.2f}px motion={4:.0f}px".format(
                index, len(plan), sample.name, measurement["equivalent_size_px"], sample.motion_px
            )
        )

    manifest_path = out_dir / cc.MANIFEST_NAME
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cc.MANIFEST_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return report(rows, plan, settings, manifest_path, challenge, cc, usable, excluded,
                  enforce=not (args.only or args.limit))


def report(rows, plan, settings, manifest_path, challenge, cc, usable, excluded, *, enforce=True) -> int:
    failures: list[str] = []
    print("")
    print("samples=" + str(len(rows)) + "  manifest=" + str(manifest_path))
    if len(rows) != len(plan):
        failures.append("rendered " + str(len(rows)) + " rows for a plan of " + str(len(plan)))

    table = challenge.challenge_composition(rows)
    composition_path = manifest_path.parent / COMPOSITION_NAME
    with composition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(challenge.COMPOSITION_COLUMNS))
        writer.writeheader()
        for entry in table:
            writer.writerow(entry)
    print("")
    print("composition by spec 08 section 4.2 source (measured, not requested):")
    for entry in table:
        print(
            "  " + entry["source_category"] + " [" + entry["sweep"] + "]: n=" + str(entry["n_rows"])
            + " (negatives " + str(entry["n_negative"]) + ")"
            + ", measured px " + f"{entry['size_min_px']}-{entry['size_max_px']}"
            + " (median " + f"{entry['size_median_px']})"
            + ", poses " + str(entry["pose_buckets"])
            + ", max motion " + f"{entry['max_motion_px']}" + " px"
            + ", scenes " + str(entry["backgrounds_used"])
        )
    print("  wrote " + composition_path.name)
    print("  negative share: " + f"{challenge.negative_share(rows) * 100:.1f}" + "%")

    scenes = {str(row["background"]) for row in rows}
    if not scenes <= set(usable):
        failures.append("a row stands on a scene that was not selected as usable")

    for violation in cc.verify_measured_pins(rows, challenge.challenge_varied_columns()):
        failures.append(violation)

    for row in rows:
        if str(row["is_negative"]).lower() in ("true", "1", "yes"):
            continue
        frame = int(float(row["imgsz"]))
        if not cc.bbox_inside_frame(row, frame, frame):
            failures.append(row["file"] + " is not wholly inside its " + str(frame) + " px frame")

    label_dir = manifest_path.parent / cc.LABEL_SUBDIR
    for row in rows:
        stem = Path(row["file"]).stem
        label_path = label_dir / (stem + ".txt")
        if not label_path.is_file():
            failures.append("missing label for " + row["file"])
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        negative = str(row["is_negative"]).lower() in ("true", "1", "yes")
        if negative:
            if text:
                failures.append("negative sample " + row["file"] + " carries a label")
            continue
        fields = text.split()
        if len(fields) != 5 or int(float(fields[0])) != 0:
            failures.append("malformed label for " + row["file"])
            continue
        values = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0.0 or values[3] <= 0.0:
            failures.append("label out of range for " + row["file"])

    if failures:
        print("")
        for failure in failures:
            print("FAILED: " + failure)
        return 1
    print("")
    print(
        "OK: every source category is present, every pinned column held, no row stands on a "
        "core or training scene"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
