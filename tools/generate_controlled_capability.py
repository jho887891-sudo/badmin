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
CONDITION_COUNTS_NAME = "condition_counts.csv"
BLOCK_CELLS_NAME = "block_cells.csv"

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
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="keep the rows already in the output manifest and append this run's rows "
             "to them, so a sweep can be added to a shipped set without re-rendering it",
    )
    return parser.parse_args()


def read_manifest_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Read an existing manifest, or nothing when there is none yet."""
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


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
    # The blocked size sweep spans scenes chosen by measurement, not by file order:
    # bg_001 is kept because S1 used it, and the other four are spread over the pool's
    # high-frequency energy so the block covers different kinds of scene.
    block_backgrounds = cc.select_block_backgrounds(background_paths)
    print(
        "blocked size sweep scenes: "
        + ", ".join(block_backgrounds)
        + " (evenly spaced ranks of measured scene busyness)"
    )
    settings = cc.ControlSettings(
        width=args.imgsz,
        height=args.imgsz,
        supersample=args.supersample,
        background=background_paths[0].name,
        backgrounds=tuple(path.name for path in background_paths),
        block_backgrounds=block_backgrounds,
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

    if not args.only and not args.merge_existing:
        # A rebuild replaces the set, so stale images and labels from the previous
        # revision (whose names no longer appear in the new manifest) are removed
        # rather than left to look like part of it.
        for directory, patterns in ((image_dir, ("*.jpg", "*.jpeg", "*.png")),
                                    (label_dir, ("*.txt",))):
            for pattern in patterns:
                for stale in directory.glob(pattern):
                    stale.unlink()

    parts = load_shuttle_parts()
    # Rendering is the expensive half of this generator and it is a pure function of
    # its arguments, so the repeats of one condition reuse one render: they differ in
    # the background crop and the noise draw, both of which are applied after the
    # render. The cache key is every argument that reaches render_shuttle() and
    # calibrate_distance(), so a hit is the same image, not a similar one.
    cache: dict[tuple, tuple] = {}
    background_cache: dict[str, np.ndarray] = {}
    block_contrast: dict[str, float] = {}
    rows: list[dict] = []
    print("rendering " + str(len(plan)) + " controlled samples into " + str(out_dir))

    def stimulus(sample, target_px: float, rotation, pixel_xy, k_matrix, light):
        """The calibrated render for one (pose, size, position, camera, light) key."""
        key = (
            float(sample.yaw_deg), float(sample.pitch_deg), float(sample.roll_deg),
            round(float(target_px), 9), float(pixel_xy[0]), float(pixel_xy[1]),
            int(sample.width), int(sample.height), float(sample.focal_px),
            int(sample.supersample), float(sample.light_azimuth_deg),
        )
        if key not in cache:
            distance = calibrate_distance(
                parts, rotation, target_px, pixel_xy, k_matrix, sample.width,
                sample.height, measure_supersample=sample.supersample,
            )
            rgb, alpha = render_shuttle(
                parts, rotation, distance, pixel_xy, k_matrix, sample.width,
                sample.height, light_dir=light, supersample=sample.supersample,
            )
            cache[key] = (distance, rgb, alpha, cc.footprint_measurement(alpha, sample.width, sample.height))
        return cache[key]

    for index, sample in enumerate(plan, start=1):
        if sample.background not in background_cache:
            loaded = imread_unicode(cc.BACKGROUND_DIR / sample.background)
            if loaded is None:
                raise SystemExit("cannot read background " + sample.background)
            background_cache[sample.background] = loaded
        # The crop is NOT fixed any more: repeats of a condition have to differ in
        # something other than the conditioned variable, and a fresh crop of the same
        # background is exactly that. What IS fixed is the crop SEQUENCE - crop_seed
        # depends on the sweep and the repeat index only - so no condition in a sweep is
        # measured on a luckier scene than another, and the background column itself
        # stays pinned to one image.
        crop_rng = np.random.default_rng(sample.crop_seed)
        background = prepare_background(
            background_cache[sample.background], sample.width, sample.height, crop_rng
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
        if sample.sweep in cc.SIZE_DRIVEN_SWEEPS:
            candidates = candidates + tuple(
                value
                for value in cc.SIZE_BUCKET_TARGETS_PX[requested_bucket]
                if abs(value - sample.target_px) > 1e-9
            )
        target_px = candidates[0]
        distance, rgb, alpha, measurement = stimulus(
            sample, target_px, rotation, pixel_xy, k_matrix, light
        )
        for attempt in range(1, len(candidates)):
            if cc.size_bucket(measurement["equivalent_size_px"]) == requested_bucket:
                break
            target_px = candidates[attempt]
            distance, rgb, alpha, measurement = stimulus(
                sample, target_px, rotation, pixel_xy, k_matrix, light
            )

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
        if sample.sweep == cc.BLOCK_SWEEP:
            # A stimulus-level number for the per-scene reading: if one scene's 6-12 px
            # cells were simply less visible than another's, that is the first thing a
            # reader of the per-scene curves needs to know.
            block_contrast[sample.name] = round(
                cc.target_contrast(occlusion.image, measurement["mask"]), 3
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

    if len(rows) != len(plan):
        print(
            "FAILED: rendered " + str(len(rows)) + " rows for a plan of " + str(len(plan)),
            file=sys.stderr,
        )
        return 1

    manifest_path = out_dir / cc.MANIFEST_NAME
    merged_rows = rows
    if args.merge_existing:
        existing_rows, existing_header = read_manifest_rows(manifest_path)
        if existing_rows:
            # A header that no longer matches the schema would silently drop columns for
            # every row this run appends, so it is a refusal rather than a warning.
            if list(existing_header) != list(cc.MANIFEST_COLUMNS):
                raise SystemExit(
                    "existing manifest header does not match the schema: "
                    + str(existing_header)
                )
            merged_rows = cc.merge_manifest_rows(existing_rows, rows)
            print(
                "keeping " + str(len(existing_rows)) + " existing rows and appending "
                + str(len(rows)) + " new ones (" + str(len(merged_rows)) + " total)"
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cc.MANIFEST_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    return report(
        merged_rows, settings, manifest_path, cc,
        enforce_counts=not (args.only or args.limit),
        contrasts=block_contrast,
    )


def write_condition_counts(rows, out_dir: Path, cc, settings, *, enforce: bool):
    """Write the sample count and error bar of every conditioned group next to the manifest.

    A curve without its sample size is a claim rather than a measurement: at n=3 a
    binomial proportion carries a 95% half-width of 57%, and at n=1 it carries none at
    all. Every group is written with its n and that half-width so a reader never has to
    go looking for it, and with the count the sweep requires.
    """
    counts = cc.measured_condition_counts(rows)
    expected = cc.expected_conditions(settings)
    path = out_dir / CONDITION_COUNTS_NAME
    block_cells = cc.block_cell_counts(rows, cc.BLOCK_SWEEP, measured=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sweep", "dimension", "condition", "n_samples", "half_width_95",
            "required", "meets_required",
        ])
        for sweep in cc.SWEEP_IDS:
            if sweep not in counts and sweep not in expected:
                continue
            required = cc.REPLICATES_PER_GROUP[sweep]
            groups = counts.get(sweep, {})
            labels = list(groups)
            for label in expected.get(sweep, ()):
                if label not in groups:
                    labels.append(label)
            for label in sorted(labels):
                n = int(groups.get(label, 0))
                writer.writerow([
                    sweep, "condition", label, n, f"{cc.proportion_half_width(n):.4f}",
                    required, "yes" if n >= required else "no",
                ])
        # The block's cells are listed separately and at their own required count: the
        # per-scene curves are read from these rows, and a cell at n=8 carries 35%, not
        # the 15% the marginal size bucket carries.
        if block_cells or cc.BLOCK_SWEEP in expected:
            for bucket in cc.SIZE_BUCKETS:
                for background in settings.block_backgrounds:
                    n = int(block_cells.get((bucket, background), 0))
                    writer.writerow([
                        cc.BLOCK_SWEEP, "size_bucket x background",
                        bucket + " @ " + background, n,
                        f"{cc.proportion_half_width(n):.4f}", cc.BLOCK_CELL_REQUIRED,
                        "yes" if n >= cc.BLOCK_CELL_REQUIRED else "no",
                    ])
    violations = cc.verify_group_counts(counts, expected) if enforce else []
    if enforce:
        violations.extend(
            cc.verify_block_cells(
                block_cells, cc.SIZE_BUCKETS, settings.block_backgrounds
            )
        )
    return path, counts, violations


def write_block_cells(rows, contrasts, out_dir: Path, cc, settings) -> Path:
    """Write the per (size, scene) cell diagnostic of the blocked sweep.

    Two numbers a reader needs before comparing per-scene curves: how many rows the cell
    has, and how visible the target was in it. A per-scene difference in detection is a
    detector effect only if the target was equally visible in every scene; this is the
    stimulus-side check of that precondition, and it is deliberately a separate file so
    it is not mistaken for a detection result.
    """
    path = out_dir / BLOCK_CELLS_NAME
    size_sums: dict[tuple[str, str], float] = {}
    contrast_sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if str(row.get("sweep", "")) != cc.BLOCK_SWEEP:
            continue
        key = (
            cc.size_bucket(float(row["equivalent_size_px"])),
            str(row["background"]),
        )
        counts[key] = counts.get(key, 0) + 1
        size_sums[key] = size_sums.get(key, 0.0) + float(row["equivalent_size_px"])
        name = Path(row["file"]).stem
        if name in contrasts:
            contrast_sums[key] = contrast_sums.get(key, 0.0) + contrasts[name]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sweep", "size_bucket", "background", "n_samples", "half_width_95",
            "required", "meets_required", "mean_equivalent_size_px", "mean_target_contrast",
        ])
        for bucket in cc.SIZE_BUCKETS:
            for background in settings.block_backgrounds:
                key = (bucket, background)
                count = int(counts.get(key, 0))
                writer.writerow([
                    cc.BLOCK_SWEEP, bucket, background, count,
                    f"{cc.proportion_half_width(count):.4f}", cc.BLOCK_CELL_REQUIRED,
                    "yes" if count >= cc.BLOCK_CELL_REQUIRED else "no",
                    f"{size_sums.get(key, 0.0) / count:.3f}" if count else "",
                    f"{contrast_sums.get(key, 0.0) / count:.3f}" if count and key in contrast_sums else "",
                ])
    return path


def report(rows, settings, manifest_path: Path, cc, *, enforce_counts: bool = True,
           contrasts=None) -> int:
    """Check the finished manifest by measurement and print what was produced."""
    failures: list[str] = []
    print("")
    print("samples=" + str(len(rows)) + "  manifest=" + str(manifest_path))

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
    # run of one sweep must not be reported as a set with seven empty buckets. The size
    # dimension is filled by both size sweeps, so S1 or S7 alone still has to cover it.
    expected = {
        "size_bucket (measured)": (("S1", "S7"), cc.SIZE_BUCKETS),
        "pose_bucket": (("S2",), cc.POSE_BUCKETS),
        "position_bucket": (("S3",), tuple(cc.POSITION_TARGETS)),
        "blur_bucket": (("S4",), tuple(name for name, _ in cc.BLUR_LEVELS)),
        "occlusion_bucket (measured)": (("S5",), tuple(name for name, _ in cc.OCCLUSION_LEVELS)),
    }
    rendered_sweeps = {row["sweep"] for row in rows}
    for name, (sweeps, vocabulary) in expected.items():
        if not (set(sweeps) & rendered_sweeps):
            continue
        counts = {}
        for value in dimensions[name]:
            counts[value] = counts.get(value, 0) + 1
        empty = [value for value in vocabulary if counts.get(value, 0) == 0]
        if empty:
            failures.append(name + " has no sample for: " + ", ".join(empty))

    counts_path, counts, count_violations = write_condition_counts(
        rows, manifest_path.parent, cc, settings, enforce=enforce_counts
    )
    print("")
    print("conditioned groups (n, and the 95% half-width of a proportion at that n):")
    for sweep in cc.SWEEP_IDS:
        groups = counts.get(sweep, {})
        if not groups:
            continue
        required = cc.REPLICATES_PER_GROUP[sweep]
        print(
            "  " + sweep + " (need " + str(required) + " each): "
            + "; ".join(
                label + " n=" + str(groups[label])
                + " +/-" + f"{cc.proportion_half_width(groups[label]) * 100:.1f}%"
                for label in sorted(groups)
            )
        )
    print("  wrote " + counts_path.name)
    block_path = write_block_cells(rows, contrasts or {}, manifest_path.parent, cc, settings)
    block_rows = [row for row in rows if str(row.get("sweep", "")) == cc.BLOCK_SWEEP]
    if block_rows:
        print("blocked size x scene cells (n, and the 35% a cell at n=8 carries):")
        for bucket in cc.SIZE_BUCKETS:
            cells = [
                row for row in block_rows
                if cc.size_bucket(float(row["equivalent_size_px"])) == bucket
            ]
            contrasts_for_bucket = [
                (contrasts or {}).get(Path(row["file"]).stem) for row in cells
            ]
            known = [value for value in contrasts_for_bucket if value is not None]
            print(
                "  " + bucket + ": n=" + str(len(cells)) + " over "
                + str(len({row["background"] for row in cells})) + " scenes"
                + (
                    ", contrast "
                    + f"{min(known):.1f}-{max(known):.1f}"
                    if known
                    else ""
                )
            )
        print("  wrote " + block_path.name)
    if enforce_counts:
        failures.extend(count_violations)
    else:
        print("  NOTE: partial run (--only/--limit), group counts not enforced")

    for violation in cc.verify_measured_pins(rows):
        failures.append(violation)

    # Each row is checked against its OWN frame: the near-field ladder is rendered in a
    # larger frame than the rest of the set, and a check against the wrong one would
    # either pass a clipped box or fail a good one.
    for row in rows:
        frame = int(float(row["imgsz"]))
        if not cc.bbox_inside_frame(row, frame, frame):
            failures.append(row["file"] + " is not wholly inside its " + str(frame) + " px frame")

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
