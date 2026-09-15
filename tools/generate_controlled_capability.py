#!/usr/bin/env python3
"""generate_controlled_capability.py - render the controlled capability test set.

Thin CLI over two things that are already tested: tools/shuttle_render.py (the
calibrated renderer, its ground truth and its compositing) and
src/perception/shuttle_detection/controlled_capability.py (the sweep plan -- which
variable each row moves and every value that is held -- plus the bucket vocabulary
the evaluator reads and the occluder). This file walks the plan, draws, writes, and
then checks the result by measurement instead of by intent.

What it does NOT do: render_sample(). That helper randomises pose, position,
background crop, blur and noise internally, which is exactly what a controlled sweep
must not do, so every draw here goes through render_shuttle() + composite() with
each non-varied value pinned.

Usage:
    python tools/generate_controlled_capability.py
    python tools/generate_controlled_capability.py --only S4 --out <temp dir>
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]

BACKGROUND_GLOBS = ("*.jpg", "*.jpeg", "*.png")

# calibrate_distance() is left at its own 5% tolerance. Tightening it to 2% was tried
# and measured: it did NOT tighten the pinned size in the pose sweep (0.57 px of
# spread at the default versus 1.06 px at 2%), because the measured footprint is a
# whole-pixel quantity and a longer search only wanders further along its staircase.
# The residual spread is therefore reported in the manifest's equivalent_size_px
# column and checked after the run, rather than chased with a longer loop.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="output directory (default: the module's DATA_DIR)")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--only", default=None, help="comma-separated sweep ids, for smoke runs")
    parser.add_argument("--limit", type=int, default=0, help="render at most this many rows")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def collect_backgrounds(background_dir: Path, training_dirs, find_blank_backgrounds,
                        find_leaked_backgrounds) -> list[Path]:
    """The real-image pool, checked against the training pools before anything renders.

    A test image sharing a training background measures the detector's memory of the
    scene, not its capability on the target, and the filename check alone would miss a
    re-encoded copy -- hence the content correlation, which is the same check the
    frozen P3 set uses.
    """
    paths: list[Path] = []
    for pattern in BACKGROUND_GLOBS:
        paths.extend(sorted(background_dir.glob(pattern)))
    if not paths:
        raise SystemExit("no backgrounds found under " + str(background_dir))
    blank = find_blank_backgrounds(paths)
    if blank:
        raise SystemExit("blank backgrounds in the pool: " + ", ".join(p.name for p in blank))
    training: list[Path] = []
    for directory in training_dirs:
        for pattern in BACKGROUND_GLOBS:
            training.extend(sorted(Path(directory).glob(pattern)))
    if not training:
        raise SystemExit("no training backgrounds found to check the test pool against")
    leaked = find_leaked_backgrounds(paths, training)
    if leaked:
        raise SystemExit(
            "test backgrounds overlap the training pools: "
            + ", ".join(candidate.name + " ~ " + match.name + f" (r={score:.4f})"
                        for candidate, match, score in leaked)
        )
    return paths


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
        find_blank_backgrounds,
        find_leaked_backgrounds,
        load_shuttle_parts,
        prepare_background,
        render_shuttle,
        rot_ypr,
    )
    from src.perception.shuttle_detection import controlled_capability as cc

    out_dir = Path(args.out) if args.out else cc.DATA_DIR
    image_dir = out_dir / cc.IMAGE_SUBDIR
    label_dir = out_dir / cc.LABEL_SUBDIR
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    background_paths = collect_backgrounds(
        cc.BACKGROUND_DIR, cc.TRAINING_BACKGROUND_DIRS, find_blank_backgrounds,
        find_leaked_backgrounds,
    )
    settings = cc.ControlSettings(
        width=args.imgsz,
        height=args.imgsz,
        supersample=args.supersample,
        background=background_paths[0].name,
        backgrounds=tuple(path.name for path in background_paths),
    )
    plan = cc.build_plan(settings)
    violations = cc.verify_single_variable(plan)
    if violations:
        for violation in violations:
            print("CONTROL VIOLATION: " + violation, file=sys.stderr)
        return 2
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - set(cc.SWEEP_IDS)
        if unknown:
            raise SystemExit("unknown sweep ids: " + ", ".join(sorted(unknown)))
        plan = [sample for sample in plan if sample.sweep in wanted]
    if args.limit:
        plan = plan[: args.limit]

    parts = load_shuttle_parts()
    cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    print("rendering " + str(len(plan)) + " controlled samples into " + str(out_dir))

    for index, sample in enumerate(plan, start=1):
        if sample.background not in cache:
            loaded = imread_unicode(cc.BACKGROUND_DIR / sample.background)
            if loaded is None:
                raise SystemExit("cannot read background " + sample.background)
            cache[sample.background] = loaded
        # The crop is pinned per background, not per row and not per sweep: an unpinned
        # crop would move the scene behind the target and turn a size sweep into a
        # background sweep, and a per-sweep crop would leave the size, pose, position,
        # blur and occlusion curves standing on five different patches of the same
        # photograph, which is not comparable across curves.
        crop_rng = np.random.default_rng(cc.stable_seed("background-crop", sample.background))
        background = prepare_background(
            cache[sample.background], sample.width, sample.height, crop_rng
        )

        focal = sample.focal_px
        k_matrix = (focal, focal, sample.width / 2.0, sample.height / 2.0)
        rotation = rot_ypr(
            math.radians(sample.yaw_deg),
            math.radians(sample.pitch_deg),
            math.radians(sample.roll_deg),
        )
        pixel_xy = sample.pixel_centre
        light = cc.light_vector(sample.light_azimuth_deg)

        # The rendered footprint is quantised to whole pixels, so a size request near a
        # bucket edge can land in the neighbouring bucket. The sweep walks the other
        # candidates of the SAME bucket and keeps the first that lands inside it; the
        # bucket edges themselves are never moved.
        requested_bucket = cc.size_bucket(sample.target_px)
        candidates = (sample.target_px,)
        if sample.sweep == "S1":
            candidates = candidates + tuple(
                value
                for value in cc.SIZE_BUCKET_TARGETS_PX[requested_bucket]
                if abs(value - sample.target_px) > 1e-9
            )
        measurement = None
        for attempt, target_px in enumerate(candidates):
            distance = calibrate_distance(
                parts, rotation, target_px, pixel_xy, k_matrix, sample.width,
                sample.height, measure_supersample=sample.supersample,
            )
            rgb, alpha = render_shuttle(
                parts, rotation, distance, pixel_xy, k_matrix, sample.width,
                sample.height, light_dir=light, supersample=sample.supersample,
            )
            measurement = cc.footprint_measurement(alpha, sample.width, sample.height)
            inside = cc.size_bucket(measurement["equivalent_size_px"]) == requested_bucket
            if inside or attempt == len(candidates) - 1:
                break

        image = composite(
            background, rgb, alpha,
            shadow_gain=sample.shadow_gain,
            motion_px=sample.motion_px,
            motion_angle_deg=sample.motion_angle_deg,
            noise_sigma=sample.noise_sigma,
            noise_seed=sample.noise_seed,
        )
        # Spec 03 section 4.5: the occluder goes over the composited image, and the
        # fraction of the object footprint it hides is measured from what was drawn.
        occlusion = cc.apply_occlusion(
            image, measurement["mask"],
            target_fraction=sample.occlusion_target_fraction,
            side=cc.OCCLUDER_SIDE,
            seed=sample.seed,
        )
        imwrite_unicode(
            image_dir / (sample.name + ".jpg"), occlusion.image, quality=args.jpeg_quality
        )
        (label_dir / (sample.name + ".txt")).write_text(
            measurement["label"], encoding="utf-8", newline="\n"
        )

        row = sample.as_manifest_row()
        row.update(
            {
                "target_px": float(target_px),
                "bbox_w_px": measurement["bbox_w_px"],
                "bbox_h_px": measurement["bbox_h_px"],
                "pos_x_px": measurement["pos_x_px"],
                "pos_y_px": measurement["pos_y_px"],
                "equivalent_size_px": measurement["equivalent_size_px"],
                "equiv_size_px": measurement["equivalent_size_px"],
                "distance_m": round(float(distance), 4),
                "occlusion_fraction": round(occlusion.measured_fraction, 6),
                "occlusion_bucket": occlusion.bucket,
                "coverage_sum": measurement["coverage_sum"],
                "solid_px": measurement["solid_px"],
            }
        )
        rows.append(row)
        print(
            "  [{0}/{1}] {2} size={3:.2f}px bucket={4} occl={5:.3f}".format(
                index, len(plan), sample.name, measurement["equivalent_size_px"],
                cc.size_bucket(measurement["equivalent_size_px"]),
                occlusion.measured_fraction,
            )
        )

    manifest_path = out_dir / cc.MANIFEST_NAME
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cc.MANIFEST_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return report(rows, settings, manifest_path, plan, cc)


def report(rows, settings, manifest_path: Path, plan, cc) -> int:
    """Check the finished manifest by measurement and print what was produced."""
    failures: list[str] = []
    print("")
    print("samples=" + str(len(rows)) + "  manifest=" + str(manifest_path))
    if len(rows) != len(plan):
        failures.append("rendered " + str(len(rows)) + " rows for a plan of " + str(len(plan)))

    measured_sizes = [cc.size_bucket(float(row["equivalent_size_px"])) for row in rows]
    dimensions = {
        "size_bucket (measured)": measured_sizes,
        "pose_bucket": [row["pose_bucket"] for row in rows],
        "position_bucket": [row["position_bucket"] for row in rows],
        "blur_bucket": [row["blur_bucket"] for row in rows],
        "occlusion_bucket (measured)": [row["occlusion_bucket"] for row in rows],
        "sweep": [row["sweep"] for row in rows],
    }
    for name, values in dimensions.items():
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        print("  " + name + ": " + ", ".join(
            key + "=" + str(counts[key]) for key in sorted(counts)
        ))

    # A dimension is only required when the sweep that fills it was rendered: a smoke
    # run of one sweep must not be reported as a set with seven empty buckets.
    expected = {
        "S1": ("size_bucket (measured)", cc.SIZE_BUCKETS),
        "S2": ("pose_bucket", cc.POSE_BUCKETS),
        "S3": ("position_bucket", tuple(cc.POSITION_TARGETS)),
        "S4": ("blur_bucket", tuple(name for name, _ in cc.BLUR_LEVELS)),
        "S5": ("occlusion_bucket (measured)", tuple(name for name, _ in cc.OCCLUSION_LEVELS)),
    }
    rendered_sweeps = {row["sweep"] for row in rows}
    for sweep, (name, vocabulary) in expected.items():
        if sweep not in rendered_sweeps:
            continue
        counts = {}
        for value in dimensions[name]:
            counts[value] = counts.get(value, 0) + 1
        empty = [value for value in vocabulary if counts.get(value, 0) == 0]
        if empty:
            failures.append(name + " has no sample for: " + ", ".join(empty))

    for violation in cc.verify_measured_pins(rows):
        failures.append(violation)

    for row in rows:
        if not cc.bbox_inside_frame(row, settings.width, settings.height):
            failures.append(row["file"] + " is not wholly inside the frame")

    backgrounds = {row["background"] for row in rows}
    training_names = set()
    for directory in cc.TRAINING_BACKGROUND_DIRS:
        for pattern in BACKGROUND_GLOBS:
            training_names.update(path.name for path in Path(directory).glob(pattern))
    shared = sorted(backgrounds & training_names)
    if shared:
        failures.append("training backgrounds used in the test set: " + ", ".join(shared))
    print("  backgrounds used: " + str(len(backgrounds)) + " (training pool shares "
          + str(len(shared)) + ")")

    label_dir = manifest_path.parent / cc.LABEL_SUBDIR
    for row in rows:
        stem = Path(row["file"]).stem
        label_path = label_dir / (stem + ".txt")
        if not label_path.is_file():
            failures.append("missing label for " + row["file"])
            continue
        fields = label_path.read_text(encoding="utf-8").split()
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
    print("OK: every bucket has samples, every sweep held its pinned columns, "
          "no training background was used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
