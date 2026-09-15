#!/usr/bin/env python3
"""The CHALLENGE_TEST split: samples whose job is to expose NEW capability gaps.

Why this module exists: spec 08 section 4.2 requires a challenge pool that is separate
from the frozen core, that grows over time, and that never enters training. Spec 01
section 67 lists both splits and the P0 asset-readiness report recorded that neither
existed; Plan 3 built and froze the core set and the challenge split was never added, so
spec 08 section 12 ("challenge test = NO UNEXPLAINED SYSTEMIC HARD FAILURE") had nothing
to run against.

The material comes from the sources spec 08 section 4.2 names, as far as this repository
can supply them:

  extreme_background   the 36 images acquired in experiments/shuttle_detection/
                       03_acquire_hard_negatives.py, which are the scene character that was
                       MEASURED to cause false positives (31 of 36 fire the baseline). They
                       are used as the scenes for every row in this split.
  very_small_target    sizes at and below the measured floor, where the baseline scores 0.
  fast_blur            motion blur beyond the core sweep's heaviest level.
  extreme_pose         the orientation the measurements found hardest (side view, 0.150).
  historical_failure   the real-photograph size range 838-1578 px, above the core sweep's
                       ceiling, which is the regime the near-field data return was for.
  real_machine_footage NOT COVERED: no real-machine footage exists in this repository.

The split is built with the SAME row machinery as the core set, so it carries one schema
and one definition of a ground-truth box and the existing evaluator runs over it unchanged.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_challenge_test.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .controlled_capability import (  # noqa: E402
    BLUR_LEVELS,
    LIGHT_AZIMUTHS_DEG,
    POSE_FAMILIES,
    SIZE_BUCKET_TARGETS_PX,
    ControlSettings,
    PlannedSample,
    light_vector,
    near_field_camera,
    sample_row,
    stable_seed,
)

# The pool this split is built on, and the split value every row carries.
HARD_NEGATIVE_DIR = ROOT / "outputs/shuttle_capability/hard_negatives/raw"
CHALLENGE_SPLIT = "challenge_test"
DATA_DIR = ROOT / "outputs/shuttle_capability/challenge_test"

# Spec 08 section 4.2 sources, in the order the spec gives them. The last one is listed
# with its own note rather than omitted, so a reader sees the gap instead of a silently
# shorter list.
CHALLENGE_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "extreme_background",
        "scenes measured to cause false positives: halls with distant players in light kit, "
        "group photographs in light clothing, bright specks on floors",
    ),
    ("very_small_targets", "sizes at and below the measured floor, where the baseline scores 0.000"),
    ("fast_blur", "motion blur beyond the core blur sweep's heaviest level"),
    ("extreme_pose", "the orientation measured hardest: side view, 0.150"),
    ("historical_failure", "the measured real-photograph range 838-1578 px, above the core sweep"),
    ("real_machine_footage", "NOT COVERED - no real-machine footage exists in this repository"),
)

# Which spec source each challenge sweep belongs to.
SWEEP_SOURCE: dict[str, str] = {
    "C1": "extreme_background",
    "C1n": "extreme_background",
    "C2": "very_small_targets",
    "C3": "fast_blur",
    "C4": "extreme_pose",
    "C5": "historical_failure",
}
CHALLENGE_SWEEPS: tuple[str, ...] = tuple(SWEEP_SOURCE)

# The scenes used in the extreme-background sweep are excluded from the challenge split if
# they are the same pictures the frozen core already renders on: a challenge row standing
# on a core scene would measure the core set's material again and make the challenge signal
# fake. Measured, not assumed - see select_challenge_backgrounds.
CORE_LEAK_THRESHOLD = 0.95

# Rows per challenge material. Sized by what the source needs, not by a target count:
# every usable hard scene gets two rows with a shuttle and one with none, so the false
# positive rate on a scene can be compared with the true positive rate on the same scene.
HARD_SCENE_SHUTTLE_ROWS = 2
HARD_SCENE_NEGATIVE_ROWS = 1
HARD_SCENE_TARGET_PX = 10.0

# At and below the core set's smallest measured footprint (2 px), which is where the
# baseline scores 0.000. 1.2 px is below anything the renderer has produced so far.
TINY_TARGETS_PX: tuple[float, ...] = (1.2, 1.5, 2.0, 3.0)
TINY_REPEATS = 10

# Beyond the core blur sweep, whose heaviest level is 7 px.
FAST_BLUR_PX: tuple[float, ...] = (10.0, 14.0, 20.0)
FAST_BLUR_REPEATS = 8

# The side family is the orientation the measured curves found hardest, so the challenge
# pose sweep is that family rather than a new spread of orientations.
CHALLENGE_POSE_FAMILY = "side"
CHALLENGE_POSE_REPEATS = 5

# The measured real-photograph range. The core near-field sweep tops out at 1024 px, so
# these start above it: this is the part of 838-1578 px the core set cannot reach.
HISTORICAL_SIZES_PX: tuple[float, ...] = (1100.0, 1300.0, 1578.0)
HISTORICAL_REPEATS = 8
HISTORICAL_FRAME_PX = 1792
HISTORICAL_FOCAL_PX = 17300.0
HISTORICAL_SUPERSAMPLE = 1


def _renderer():
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import shuttle_render

    return shuttle_render


def select_challenge_backgrounds(
    candidates: Sequence[Any],
    forbidden: Sequence[Any],
    *,
    threshold: float = CORE_LEAK_THRESHOLD,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, float], ...]]:
    """The scenes this split may use, and the ones it must not, both by measurement.

    Returns (usable names, excluded (name, matched name, correlation)). Filenames prove
    nothing here: the hard-negative pool and the frozen pool are different files, and three
    of them are still the same pictures. Only a content comparison catches that, which is
    the same lesson find_leaked_backgrounds was written for.
    """
    find_leaked_backgrounds = _renderer().find_leaked_backgrounds
    leaks = find_leaked_backgrounds(candidates, forbidden, threshold=threshold)
    excluded = {
        Path(candidate).name: (Path(match).name, float(score))
        for candidate, match, score in leaks
    }
    usable = tuple(
        sorted(Path(path).name for path in candidates if Path(path).name not in excluded)
    )
    ordered = tuple((name, info[0], info[1]) for name, info in sorted(excluded.items()))
    return usable, ordered


def build_challenge_plan(
    settings: ControlSettings, backgrounds: Sequence[str]
) -> list[PlannedSample]:
    """The challenge sample list, one entry per image.

    Every row stands on one of the hard scenes, so the split shares no scene with the
    frozen core. Which scene a row gets is deterministic (row index modulo the pool), and
    the background is declared as a varied column for every challenge sweep: this split is
    deliberately heterogeneous, and pretending otherwise would let the pinned-column check
    pass on a claim that is not true.
    """
    if not backgrounds:
        raise ValueError("the challenge split needs at least one usable hard scene")
    scenes = tuple(backgrounds)
    varied_background = ("background",)
    samples: list[PlannedSample] = []
    counter = {"value": 0}

    def add(name: str, sweep: str, varied: tuple[str, ...], **kwargs: Any) -> None:
        index = counter["value"]
        counter["value"] = index + 1
        kwargs.setdefault("background", scenes[index % len(scenes)])
        kwargs.setdefault("pose", {
            "yaw_deg": settings.pose_deg[0],
            "pitch_deg": settings.pose_deg[1],
            "roll_deg": settings.pose_deg[2],
        })
        kwargs.setdefault("pose_bucket", "pinned_oblique")
        kwargs.setdefault("target_px", settings.size_px)
        kwargs.setdefault("position_bucket", "center")
        kwargs.setdefault("light_azimuth_deg", settings.light_azimuth_deg)
        kwargs.setdefault("motion_px", 0.0)
        kwargs.setdefault("occlusion_name", "none")
        kwargs.setdefault("occlusion_target_fraction", 0.0)
        kwargs.setdefault("repeat", index)
        kwargs.setdefault("split", CHALLENGE_SPLIT)
        kwargs["varied"] = tuple(dict.fromkeys(varied + varied_background))
        samples.append(sample_row(settings, name=name, sweep=sweep, **kwargs))

    # C1: a small shuttle on a scene that already produces false positives, and C1n: the
    # same scene with nothing in it. The pair is what separates "it fires on the clutter"
    # from "it found the shuttle".
    for scene_index, scene in enumerate(scenes):
        for repeat in range(HARD_SCENE_SHUTTLE_ROWS):
            add(
                "c1_hard_scene_" + str(scene_index).zfill(2) + "_" + str(repeat),
                "C1",
                ("background",),
                target_px=HARD_SCENE_TARGET_PX,
                background=scene,
                repeat=repeat,
            )
        for repeat in range(HARD_SCENE_NEGATIVE_ROWS):
            add(
                "c1n_hard_negative_" + str(scene_index).zfill(2) + "_" + str(repeat),
                "C1n",
                ("background",),
                target_px=0.0,
                background=scene,
                repeat=repeat,
                is_negative=True,
            )

    # C2: at and below the measured floor.
    for target in TINY_TARGETS_PX:
        for repeat in range(TINY_REPEATS):
            add(
                "c2_tiny_" + f"{target:.1f}".replace(".", "p") + "_" + str(repeat),
                "C2",
                ("target_px",),
                target_px=target,
                repeat=repeat,
            )

    # C3: blur past the core sweep's heaviest level.
    for motion in FAST_BLUR_PX:
        for repeat in range(FAST_BLUR_REPEATS):
            add(
                "c3_fast_blur_" + str(int(motion)).zfill(2) + "_" + str(repeat),
                "C3",
                ("motion_px",),
                motion_px=motion,
                repeat=repeat,
            )

    # C4: the orientation the measured curves found hardest.
    orientations = POSE_FAMILIES[CHALLENGE_POSE_FAMILY]
    for repeat in range(CHALLENGE_POSE_REPEATS):
        for orientation_index, pose in enumerate(orientations):
            add(
                "c4_side_pose_" + str(orientation_index).zfill(2) + "_" + str(repeat),
                "C4",
                ("yaw_deg", "pitch_deg", "roll_deg", "pose_bucket"),
                pose=pose,
                pose_bucket=CHALLENGE_POSE_FAMILY,
                repeat=repeat,
            )

    # C5: the measured real-photograph size range, above the core sweep's ceiling.
    camera = near_field_camera(
        HISTORICAL_SIZES_PX,
        frame_px=HISTORICAL_FRAME_PX,
        focal_px=HISTORICAL_FOCAL_PX,
        supersample=HISTORICAL_SUPERSAMPLE,
    )
    for target in HISTORICAL_SIZES_PX:
        for repeat in range(HISTORICAL_REPEATS):
            add(
                "c5_historical_" + str(int(target)).zfill(4) + "_" + str(repeat),
                "C5",
                ("target_px",),
                target_px=target,
                frame_px=camera.frame_px,
                focal_px=camera.focal_px,
                supersample=camera.supersample,
                repeat=repeat,
            )
    return samples


def challenge_varied_columns() -> dict[str, tuple[str, ...]]:
    """The columns each challenge sweep is allowed to move, for verify_measured_pins.

    Every challenge sweep may move the scene, because this split is deliberately
    heterogeneous; the rest of each sweep's controls stay pinned and are checked.
    """
    mapping: dict[str, tuple[str, ...]] = {}
    for sweep in CHALLENGE_SWEEPS:
        mapping[sweep] = ("background",)
    mapping["C2"] = ("background", "target_px", "equivalent_size_px")
    mapping["C3"] = ("background", "motion_px", "blur_bucket")
    mapping["C4"] = ("background", "yaw_deg", "pitch_deg", "roll_deg", "pose_bucket")
    mapping["C5"] = ("background", "target_px", "equivalent_size_px")
    return mapping


COMPOSITION_COLUMNS: tuple[str, ...] = (
    "sweep",
    "source_category",
    "n_rows",
    "n_negative",
    "size_min_px",
    "size_median_px",
    "size_max_px",
    "pose_buckets",
    "max_motion_px",
    "backgrounds_used",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def challenge_composition(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per source category: how many rows, and the spread actually achieved.

    Measured values only. A composition table built from what was requested would report
    a 1.2 px target that rendered as 1.41 px and a blur level that did not reach the image.
    """
    table: list[dict[str, Any]] = []
    for sweep in CHALLENGE_SWEEPS:
        subset = [row for row in rows if str(row.get("sweep", "")) == sweep]
        if not subset:
            continue
        sizes = [float(row["equivalent_size_px"]) for row in subset]
        motions = [float(row.get("motion_px") or 0.0) for row in subset]
        table.append(
            {
                "sweep": sweep,
                "source_category": SWEEP_SOURCE[sweep],
                "n_rows": len(subset),
                "n_negative": sum(1 for row in subset if _is_negative(row)),
                "size_min_px": round(min(sizes), 3),
                "size_median_px": round(_percentile(sizes, 0.5), 3),
                "size_max_px": round(max(sizes), 3),
                "pose_buckets": len({str(row.get("pose_bucket", "")) for row in subset}),
                "max_motion_px": round(max(motions), 1),
                "backgrounds_used": len({str(row.get("background", "")) for row in subset}),
            }
        )
    return table


def _is_negative(record: Any) -> bool:
    """Whether a planned sample or a manifest row carries no object.

    Accepts either shape so the same helper works on the plan and on the shipped manifest,
    where the value has been through a CSV as the text "True".
    """
    value = record.get("is_negative", "") if isinstance(record, Mapping) else getattr(
        record, "is_negative", False
    )
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def negative_share(rows: Sequence[Any]) -> float:
    """The share of the split that contains no shuttle at all."""
    if not rows:
        return 0.0
    return sum(1 for row in rows if _is_negative(row)) / float(len(rows))
