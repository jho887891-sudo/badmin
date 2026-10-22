# -*- coding: utf-8 -*-
"""P6-03 (R+1 item C3): D455 replay validator.

Given a directory of (rgb, depth, timestamp) triplets this checks the things a
replay can actually be wrong about without anybody noticing:

  * timestamps that are not strictly increasing (a reordered or duplicated frame)
  * gaps wider than a documented threshold (dropped frames that the file count
    does not reveal, e.g. a writer that skipped 40 frames)
  * rgb/depth frame-count disagreement or a missing file
  * depth units that are millimetres per unit while the pipeline assumes metres
  * an UNDECLARED rgb/depth alignment flag (the flag is a decision, not a default)
  * intrinsics that are UNKNOWN, or a stream resolution that disagrees with the
    PNG headers on disk

It emits a machine-readable report and exits non-zero on ANY violation, so it
can gate CI. Exit codes: 0 = PASS, 1 = violations, 2 = usage/IO error.

Data class: this repository contains NO real D455 capture. Every fixture here is
synthetic and the report says so ("OFFLINE-SYNTHETIC"). The validator is written
so a real capture can be dropped in unchanged; it never claims to have measured
anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import d455_metadata as md

REPORT_SCHEMA = "p6_d455_replay_report/1"
INDEX_SCHEMA = "p6_d455_replay_index/1"

DATA_CLASS_SYNTHETIC = "OFFLINE-SYNTHETIC"
DATA_CLASS_REAL = "REAL-CAPTURE"
DATA_CLASSES = (DATA_CLASS_SYNTHETIC, DATA_CLASS_REAL)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

DEFAULT_MAX_GAP_FACTOR = 1.5  # a gap is dt > factor / fps
MAX_GAP_FACTOR_LIMIT = 100.0  # refuse an absurdly permissive threshold


class Violation:
    """One concrete, named reason a replay cannot be trusted."""

    __slots__ = ("code", "detail")

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}

    def __repr__(self) -> str:
        return f"Violation({self.code}: {self.detail})"


@dataclass
class Check:
    id: str
    status: str  # PASS | FAIL | SKIP
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "detail": self.detail,
        }
        if self.metrics:
            out["metrics"] = self.metrics
        return out


@dataclass
class ReplayReport:
    root: Path
    index_path: Path | None = None
    metadata_path: Path | None = None
    data_class: str = "UNDECLARED"
    thresholds: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    tool: str = "src/simulation/sensors/d455_replay.py"

    @property
    def verdict(self) -> str:
        return "FAIL" if self.violations else "PASS"

    @property
    def exit_code(self) -> int:
        return 1 if self.violations else 0

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def fail(self, code: str, detail: str) -> None:
        self.violations.append(Violation(code, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "tool": self.tool,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "data_class": self.data_class,
            "calibration_claimed": False,
            "input": {
                "root": str(self.root),
                "index": str(self.index_path) if self.index_path else None,
                "metadata": str(self.metadata_path) if self.metadata_path else None,
            },
            "thresholds": self.thresholds,
            "counts": self.counts,
            "metrics": self.metrics,
            "checks": [c.to_dict() for c in self.checks],
            "violations": [v.to_dict() for v in self.violations],
        }

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False)
            + "\n",
            encoding="utf-8",
        )
        return target


def png_size(path: Path | str) -> tuple[int, int]:
    """Width/height from a PNG IHDR header. Raises ValueError on a non-PNG."""
    raw = Path(path).read_bytes()
    if not raw.startswith(PNG_MAGIC):
        raise ValueError(f"{path} is not a PNG (bad signature)")
    if len(raw) < 24 or raw[12:16] != b"IHDR":
        raise ValueError(f"{path} has no IHDR chunk")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} declares a {width}x{height} image")
    return width, height


def _declared_depth_scale(index: dict[str, Any]) -> tuple[float, str, str | None] | None:
    """Pull (value, unit, encoding) out of the index's depth block, if present."""
    block = index.get("depth")
    if not isinstance(block, dict):
        return None
    value = block.get("scale_m_per_unit", block.get("depth_scale"))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    unit = block.get("unit", block.get("scale_unit"))
    encoding = block.get("encoding")
    return float(value), str(unit) if unit is not None else "", (
        str(encoding) if encoding is not None else None
    )


def validate_replay(
    root: Path | str,
    *,
    metadata_path: Path | str | None = None,
    max_gap_factor: float = DEFAULT_MAX_GAP_FACTOR,
) -> ReplayReport:
    """Validate one replay directory. Never raises for a data problem."""
    root = Path(root)
    report = ReplayReport(root=root)
    if metadata_path is not None:
        report.metadata_path = Path(metadata_path)

    if max_gap_factor <= 0 or max_gap_factor > MAX_GAP_FACTOR_LIMIT:
        raise ValueError(
            f"max_gap_factor={max_gap_factor} outside (0, {MAX_GAP_FACTOR_LIMIT}]"
        )
    report.thresholds = {
        "max_gap_factor": max_gap_factor,
        "gap_rule": "any dt > max_gap_factor / fps is a gap",
        "monotonic_rule": "timestamps must be strictly increasing (dt > 0)",
        "depth_scale_canonical_unit": md.M_PER_UNIT,
        "depth_scale_upper_bound": md.DEPTH_SCALE_M_PER_UNIT_MAX,
        "data_classes": list(DATA_CLASSES),
    }

    index_path = root if root.is_file() else root / "index.json"
    if not index_path.is_file():
        report.fail("INDEX_MISSING", f"no replay index at {index_path}")
        report.add(Check("index_present", "FAIL", f"missing {index_path.name}"))
        return report
    report.index_path = index_path
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.fail("INDEX_UNPARSABLE", f"{index_path.name}: {exc}")
        report.add(Check("index_parsable", "FAIL", str(exc)))
        return report
    report.add(Check("index_parsable", "PASS", f"parsed {index_path.name}"))

    if index.get("schema") != INDEX_SCHEMA:
        report.fail(
            "INDEX_SCHEMA",
            f"index schema {index.get('schema')!r} != {INDEX_SCHEMA!r}",
        )
        report.add(Check("index_schema", "FAIL", str(index.get("schema"))))
    else:
        report.add(Check("index_schema", "PASS", INDEX_SCHEMA))

    declared_class = index.get("data_class")
    if declared_class in DATA_CLASSES:
        report.data_class = str(declared_class)
        report.add(
            Check("data_class", "PASS", f"declared {report.data_class}")
        )
    else:
        report.fail(
            "DATA_CLASS_UNDECLARED",
            f"data_class {declared_class!r} is not one of {list(DATA_CLASSES)}; "
            "synthetic fixtures and real captures must not be confused",
        )
        report.add(Check("data_class", "FAIL", repr(declared_class)))

    frames = index.get("frames")
    if not isinstance(frames, list) or not frames:
        report.fail("EMPTY_INDEX", "index has no frames")
        report.add(Check("frames_present", "FAIL", "no frames"))
        return report
    report.counts["index_frames"] = len(frames)
    report.add(Check("frames_present", "PASS", f"{len(frames)} frames in index"))

    # --- per-frame file existence and image headers ----------------------
    rgb_ok: list[Path] = []
    depth_ok: list[Path] = []
    missing: list[str] = []
    unreadable: list[str] = []
    mismatched: list[str] = []
    resolution = index.get("resolution")
    declared_res: tuple[int, int] | None = None
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        declared_res = (int(resolution[0]), int(resolution[1]))

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            report.fail("FRAME_MALFORMED", f"frame[{i}] is not an object")
            continue
        for key, bucket in (("rgb", rgb_ok), ("depth", depth_ok)):
            rel = frame.get(key)
            if not isinstance(rel, str):
                missing.append(f"frame[{i}].{key} absent")
                continue
            candidate = index_path.parent / rel
            if not candidate.is_file():
                missing.append(f"frame[{i}].{key} -> {rel} not found")
                continue
            try:
                size = png_size(candidate)
            except ValueError as exc:
                unreadable.append(str(exc))
                bucket.append(candidate)
                continue
            if declared_res is not None and size != declared_res:
                mismatched.append(
                    f"{rel} is {size[0]}x{size[1]}, index declares "
                    f"{declared_res[0]}x{declared_res[1]}"
                )
            bucket.append(candidate)

    report.counts["rgb_files"] = len(rgb_ok)
    report.counts["depth_files"] = len(depth_ok)

    if missing:
        report.fail("MISSING_FRAME_FILE", "; ".join(missing[:6])
                    + ("" if len(missing) <= 6 else f" (+{len(missing) - 6} more)"))
        report.add(Check("frame_files", "FAIL", f"{len(missing)} missing"))
    else:
        report.add(Check("frame_files", "PASS", f"{len(frames)} triplets resolve"))

    if unreadable:
        report.fail("IMAGE_UNREADABLE", "; ".join(unreadable[:4]))
        report.add(Check("image_headers", "FAIL", f"{len(unreadable)} unreadable"))
    else:
        report.add(Check("image_headers", "PASS", "PNG headers readable"))

    if mismatched:
        report.fail("IMAGE_SIZE_MISMATCH", "; ".join(mismatched[:4]))
        report.add(Check("image_size", "FAIL", f"{len(mismatched)} mismatched"))
    else:
        report.add(
            Check(
                "image_size",
                "PASS",
                f"all images {declared_res[0]}x{declared_res[1]}" if declared_res
                else "no index resolution declared",
            )
        )

    if len(rgb_ok) != len(depth_ok):
        report.fail(
            "FRAME_COUNT_MISMATCH",
            f"{len(rgb_ok)} rgb files vs {len(depth_ok)} depth files",
        )
        report.add(
            Check("frame_counts", "FAIL", f"rgb={len(rgb_ok)} depth={len(depth_ok)}")
        )
    else:
        report.add(
            Check("frame_counts", "PASS", f"rgb={len(rgb_ok)} depth={len(depth_ok)}")
        )

    # directory-level count: catches files on disk that the index forgot
    for sub in ("rgb", "depth"):
        subdir = index_path.parent / sub
        if subdir.is_dir():
            on_disk = len([p for p in subdir.iterdir() if p.is_file()])
            report.counts[f"{sub}_files_on_disk"] = on_disk
            if on_disk != len(frames):
                report.fail(
                    "DIRECTORY_COUNT_MISMATCH",
                    f"{sub}/ holds {on_disk} files but the index lists {len(frames)}",
                )

    # --- timestamps ------------------------------------------------------
    stamps: list[float] = []
    for i, frame in enumerate(frames):
        value = frame.get("timestamp_s") if isinstance(frame, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            report.fail("TIMESTAMP_MISSING", f"frame[{i}] has no numeric timestamp_s")
            continue
        stamps.append(float(value))
    report.metrics["frame_count_with_timestamp"] = len(stamps)

    if len(stamps) >= 2:
        deltas = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
        report.metrics["min_dt_s"] = min(deltas)
        report.metrics["max_dt_s"] = max(deltas)
        report.metrics["mean_dt_s"] = sum(deltas) / len(deltas)
        bad = [
            f"frame[{i}]->[{i + 1}] dt={deltas[i]:.9f}s"
            for i in range(len(deltas))
            if not deltas[i] > 0.0
        ]
        if bad:
            report.fail(
                "NON_MONOTONIC_TIMESTAMP",
                "timestamps are not strictly increasing: " + "; ".join(bad[:5]),
            )
            report.add(Check("timestamps_monotonic", "FAIL", f"{len(bad)} bad steps"))
        else:
            report.add(
                Check(
                    "timestamps_monotonic",
                    "PASS",
                    f"{len(stamps)} strictly increasing stamps",
                )
            )
    else:
        report.add(Check("timestamps_monotonic", "SKIP", "fewer than 2 timestamps"))

    # --- fps and gaps ----------------------------------------------------
    fps: float | None = None
    for holder, label in ((index, "index"),):
        value = holder.get("fps")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fps = float(value)
            report.metrics["fps_source"] = label
    if fps is None and report.metadata_path is not None:
        try:
            meta_for_fps = md.load_metadata(
                report.metadata_path, require_consistent_values=False
            )
            if md.is_known(meta_for_fps.fps):
                fps = float(meta_for_fps.fps.value)
                report.metrics["fps_source"] = "metadata"
        except md.D455MetadataError:
            pass

    if fps is None or fps <= 0:
        report.fail(
            "FPS_UNKNOWN",
            "no positive fps in the index or the metadata record: a gap "
            "threshold cannot be derived, so timing is unverified",
        )
        report.add(Check("frame_rate", "FAIL", f"fps={fps!r}"))
        report.add(Check("timestamp_gaps", "SKIP", "no fps"))
    else:
        report.add(Check("frame_rate", "PASS", f"{fps} fps"))
        gap_threshold = max_gap_factor / fps
        report.metrics["fps"] = fps
        report.metrics["gap_threshold_s"] = gap_threshold
        gaps = [
            f"frame[{i}]->[{i + 1}] dt={stamps[i + 1] - stamps[i]:.6f}s"
            for i in range(len(stamps) - 1)
            if stamps[i + 1] - stamps[i] > gap_threshold
        ]
        if gaps:
            report.fail(
                "TIMESTAMP_GAP_EXCEEDED",
                f"gap > {gap_threshold:.6f}s (={max_gap_factor}/fps): "
                + "; ".join(gaps[:5]),
            )
            report.add(Check("timestamp_gaps", "FAIL", f"{len(gaps)} gaps"))
        else:
            report.add(
                Check(
                    "timestamp_gaps",
                    "PASS",
                    f"no dt > {gap_threshold:.6f}s",
                )
            )

    # --- depth units -----------------------------------------------------
    depth_block = _declared_depth_scale(index)
    if depth_block is None:
        report.fail(
            "DEPTH_SCALE_ABSENT",
            "index declares no depth scale; a raw depth image has no meaning "
            "without metres-per-unit",
        )
        report.add(Check("depth_units", "FAIL", "no depth scale declared"))
    else:
        value, unit, encoding = depth_block
        if unit != md.M_PER_UNIT:
            report.fail(
                "DEPTH_UNIT_NOT_METRES",
                f"depth scale unit {unit!r} is not {md.M_PER_UNIT!r} "
                f"(value {value!r}); millimetres-per-unit read as metres is a "
                "1000x error",
            )
            report.add(Check("depth_units", "FAIL", f"unit={unit!r}"))
        elif not (0.0 < value < md.DEPTH_SCALE_M_PER_UNIT_MAX):
            report.fail(
                "DEPTH_SCALE_IMPLAUSIBLE",
                f"depth scale {value!r} m/unit outside (0, "
                f"{md.DEPTH_SCALE_M_PER_UNIT_MAX})",
            )
            report.add(Check("depth_units", "FAIL", f"value={value!r}"))
        else:
            report.add(
                Check(
                    "depth_units",
                    "PASS",
                    f"{value} {md.M_PER_UNIT}",
                    {"depth_scale_m_per_unit": value, "encoding": encoding},
                )
            )
        if encoding is None:
            report.add(Check("depth_encoding", "SKIP", "encoding not declared"))

    # --- alignment flag --------------------------------------------------
    aligned = index.get("rgb_depth_aligned")
    if isinstance(aligned, bool):
        report.add(Check("rgb_depth_alignment", "PASS", f"aligned={aligned}"))
        report.metrics["rgb_depth_aligned"] = aligned
    else:
        report.fail(
            "ALIGNMENT_UNKNOWN",
            f"rgb_depth_aligned={aligned!r}: the alignment flag is a pipeline "
            "decision, not a default; with it UNKNOWN the same (u,v) in RGB and "
            "depth may be different rays",
        )
        report.add(Check("rgb_depth_alignment", "FAIL", repr(aligned)))

    # --- intrinsics from a metadata record -------------------------------
    if report.metadata_path is None:
        report.fail(
            "INTRINSICS_UNKNOWN",
            "no metadata record supplied: a replay alone cannot prove fx/fy/cx/cy",
        )
        report.add(Check("intrinsics", "FAIL", "no metadata record"))
    else:
        try:
            meta = md.load_metadata(report.metadata_path)
        except md.D455MetadataError as exc:
            report.fail(
                "METADATA_INVALID",
                f"{type(exc).__name__}: {exc}",
            )
            report.add(Check("intrinsics", "FAIL", "metadata invalid"))
            meta = None
        if meta is not None:
            missing_intr = meta.missing_required("intrinsics")
            if missing_intr:
                report.fail(
                    "INTRINSICS_UNKNOWN",
                    "metadata record is missing required intrinsics fields: "
                    + ", ".join(missing_intr),
                )
                report.add(
                    Check("intrinsics", "FAIL", f"{len(missing_intr)} fields UNKNOWN")
                )
            else:
                report.add(
                    Check(
                        "intrinsics",
                        "PASS",
                        f"fx={meta.fx_px.value} fy={meta.fy_px.value} "
                        f"cx={meta.cx_px.value} cy={meta.cy_px.value}",
                    )
                )
                report.metrics["intrinsics"] = {
                    "fx_px": meta.fx_px.value,
                    "fy_px": meta.fy_px.value,
                    "cx_px": meta.cx_px.value,
                    "cy_px": meta.cy_px.value,
                    "record_sha256": meta.record_sha256,
                }
                meta_res = (int(meta.width_px.value), int(meta.height_px.value))
                if declared_res is not None and meta_res != declared_res:
                    report.fail(
                        "RESOLUTION_MISMATCH",
                        f"metadata resolution {meta_res} != index resolution "
                        f"{declared_res}",
                    )
                    report.add(Check("resolution", "FAIL", f"{meta_res} vs {declared_res}"))
                elif declared_res is None:
                    report.fail(
                        "RESOLUTION_UNDECLARED",
                        f"metadata says {meta_res} but the index declares no "
                        "resolution to check it against",
                    )
                    report.add(Check("resolution", "FAIL", "index silent"))
                else:
                    report.add(
                        Check("resolution", "PASS", f"{meta_res[0]}x{meta_res[1]}")
                    )
                if md.is_known(meta.rgb_depth_aligned):
                    meta_aligned = bool(meta.rgb_depth_aligned.value)
                    if isinstance(aligned, bool) and meta_aligned != aligned:
                        report.fail(
                            "ALIGNMENT_CONFLICT",
                            f"index says aligned={aligned}, metadata says "
                            f"{meta_aligned}",
                        )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="d455_replay",
        description=(
            "Validate a D455 replay directory (rgb/depth/timestamp triplets). "
            "Exit 0 = PASS, 1 = violations, 2 = usage/IO error."
        ),
    )
    parser.add_argument("replay", help="replay directory (or its index.json)")
    parser.add_argument(
        "--metadata",
        default=None,
        help="D455 metadata record (JSON) to check intrinsics against",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write the machine-readable report to this path",
    )
    parser.add_argument(
        "--max-gap-factor",
        type=float,
        default=DEFAULT_MAX_GAP_FACTOR,
        help=f"a gap is dt > factor/fps (default {DEFAULT_MAX_GAP_FACTOR})",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not print the report to stdout"
    )
    args = parser.parse_args(argv)

    try:
        report = validate_replay(
            args.replay,
            metadata_path=args.metadata,
            max_gap_factor=args.max_gap_factor,
        )
    except (ValueError, OSError) as exc:
        print(f"usage/IO error: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    if args.report:
        try:
            written = report.write(args.report)
        except OSError as exc:
            print(f"cannot write report: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {written}")
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"verdict={report.verdict} data_class={report.data_class} "
        f"violations={len(report.violations)} exit={report.exit_code}"
    )
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
