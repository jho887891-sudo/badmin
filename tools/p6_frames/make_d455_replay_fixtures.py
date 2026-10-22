# -*- coding: utf-8 -*-
"""Generate the P6-03 D455 replay fixtures (deterministic, no device needed).

The repository contains NO real D455 capture, so the replay validator is
exercised on synthetic fixtures that are labelled OFFLINE-SYNTHETIC everywhere.
The numbers in these fixtures are NOT D455 measurements: they describe an 8x6
toy pinhole so that nobody can mistake them for a calibration (a real D455
colour stream is not 8x6 and its fx is not 100 px).

Every generated file is hashed into manifest.json, and the test suite verifies
that the files on disk still match, so a hand-edited fixture cannot pass.

Usage:
    python tools/p6_frames/make_d455_replay_fixtures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "tests/simulation/fixtures/p6_d455_replay"

FIXTURE_WIDTH = 8
FIXTURE_HEIGHT = 6
FIXTURE_FPS = 30.0
FIXTURE_FRAMES = 6
# Synthetic pinhole: a 90-degree FOV on an 8x6 image. Deliberately NOT a D455.
FIXTURE_FX = 4.0
FIXTURE_FY = 4.0
FIXTURE_CX = 3.5
FIXTURE_CY = 2.5

SYNTHETIC_NOTE = (
    "SYNTHETIC FIXTURE - NOT A D455 MEASUREMENT. Values describe an 8x6 toy "
    "pinhole used only to exercise the P6-03 contract and replay validator."
)


def write_png(path: Path, width: int, height: int, gray: int) -> None:
    """Minimal deterministic grayscale PNG (no PIL, no external tooling)."""
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # filter type 0
        raw.extend(bytes([gray]) * width)
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _datum(value, locator: str, unit=None) -> dict:
    node = {
        "value": value,
        "source": {"file": "@self", "locator": locator},
    }
    if unit is not None:
        node["unit"] = unit
    return node


def _unknown(missing: str) -> dict:
    return {"status": "UNKNOWN", "missing_datum": missing}


def full_metadata(prefix: str = "fields") -> dict:
    """A complete, self-consistent SYNTHETIC record (8x6 toy pinhole)."""
    def loc(name: str) -> str:
        return f"{prefix}.{name}.value"

    fields = {
        "fx_px": _datum(FIXTURE_FX, loc("fx_px"), "px"),
        "fy_px": _datum(FIXTURE_FY, loc("fy_px"), "px"),
        "cx_px": _datum(FIXTURE_CX, loc("cx_px"), "px"),
        "cy_px": _datum(FIXTURE_CY, loc("cy_px"), "px"),
        "width_px": _datum(FIXTURE_WIDTH, loc("width_px"), "px"),
        "height_px": _datum(FIXTURE_HEIGHT, loc("height_px"), "px"),
        "depth_scale_m_per_unit": _datum(0.001, loc("depth_scale_m_per_unit"),
                                         "m_per_unit"),
        "depth_encoding": _datum("uint16", loc("depth_encoding")),
        "fps": _datum(FIXTURE_FPS, loc("fps"), "Hz"),
        "rgb_depth_aligned": _datum(True, loc("rgb_depth_aligned")),
        "distortion_model": _datum("brown_conrady", loc("distortion_model")),
        "distortion_coefficients": _datum([0.0, 0.0, 0.0, 0.0, 0.0],
                                          loc("distortion_coefficients"),
                                          "dimensionless"),
        "hfov_rad": _datum(1.5707963267948966, loc("hfov_rad"), "rad"),
        "T_mount_camera": _datum(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            loc("T_mount_camera"),
            "m",
        ),
        "device_model": _datum("SYNTHETIC-8x6-PINHOLE", loc("device_model")),
    }
    return {
        "schema": "p6_d455_metadata/1",
        "synthetic_fixture": {
            "is_real_d455_data": False,
            "note": SYNTHETIC_NOTE,
            "T_mount_camera_note": (
                "rotation = the proven optical->mount convention; translation "
                "fixed at zero because no mount measurement exists. NOT a "
                "hand-eye calibration."
            ),
        },
        "units": {
            "intrinsics": "px",
            "depth_scale": "m_per_unit",
            "length": "m",
            "angle": "rad",
            "rate": "Hz",
            "distortion": "dimensionless",
            "time": "s",
        },
        "fields": fields,
    }


def unknown_metadata() -> dict:
    """A record that honestly declares every field UNKNOWN (the real repo state)."""
    reasons = {
        "fx_px": "no committed intrinsics record for the exact stream profile",
        "fy_px": "no committed intrinsics record for the exact stream profile",
        "cx_px": "no committed intrinsics record for the exact stream profile",
        "cy_px": "no committed intrinsics record for the exact stream profile",
        "width_px": "the stream profile actually used is not recorded",
        "height_px": "the stream profile actually used is not recorded",
        "depth_scale_m_per_unit": "per-device depth scale never read from a unit",
        "depth_encoding": "depth frame format never recorded",
        "fps": "configured frame rate never recorded",
        "rgb_depth_aligned": "whether align(rs.stream.color) runs is not recorded",
        "distortion_model": "no colour-stream calibration record",
        "distortion_coefficients": "no colour-stream calibration record",
        "T_mount_camera": "no hand-eye calibration and no measured CAD mount",
        "device_serial": "no rs-enumerate-devices output in the repository",
    }
    return {
        "schema": "p6_d455_metadata/1",
        "synthetic_fixture": {
            "is_real_d455_data": False,
            "note": SYNTHETIC_NOTE,
        },
        "units": {
            "intrinsics": "px",
            "depth_scale": "m_per_unit",
            "length": "m",
            "angle": "rad",
            "rate": "Hz",
            "distortion": "dimensionless",
            "time": "s",
        },
        "fields": {name: _unknown(reason) for name, reason in reasons.items()},
    }


def _index(frames: int, *, timestamps=None, depth_unit="m_per_unit",
           depth_scale=0.001, aligned=True, data_class="OFFLINE-SYNTHETIC",
           resolution=(FIXTURE_WIDTH, FIXTURE_HEIGHT), fps=FIXTURE_FPS) -> dict:
    if timestamps is None:
        timestamps = [round(i / FIXTURE_FPS, 9) for i in range(frames)]
    entries = []
    for i in range(frames):
        entries.append(
            {
                "rgb": f"rgb/{i:06d}.png",
                "depth": f"depth/{i:06d}.png",
                "timestamp_s": timestamps[i] if i < len(timestamps) else 0.0,
            }
        )
    return {
        "schema": "p6_d455_replay_index/1",
        "data_class": data_class,
        "resolution": list(resolution),
        "fps": fps,
        "rgb_depth_aligned": aligned,
        "depth": {
            "scale_m_per_unit": depth_scale,
            "unit": depth_unit,
            "encoding": "uint16",
        },
        "frames": entries,
    }


def _write_frames(root: Path, count: int, *, skip_depth: tuple[int, ...] = ()) -> None:
    for i in range(count):
        write_png(root / "rgb" / f"{i:06d}.png", FIXTURE_WIDTH, FIXTURE_HEIGHT,
                  40 + i * 7)
        if i not in skip_depth:
            write_png(root / "depth" / f"{i:06d}.png", FIXTURE_WIDTH,
                      FIXTURE_HEIGHT, 10 + i * 5)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build(out_root: Path = DEFAULT_OUT) -> Path:
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # 1. good: everything consistent
    good = out_root / "good"
    _write_frames(good, FIXTURE_FRAMES)
    _write_json(good / "index.json", _index(FIXTURE_FRAMES))
    _write_json(good / "metadata.json", full_metadata())

    # also a YAML twin of the good record, for the YAML code path
    (good / "metadata.yaml").write_text(
        "# SYNTHETIC fixture, not a D455 measurement. JSON twin: metadata.json\n"
        + json.dumps(full_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )

    # 2. missing frames: the index lists 6, depth frame 3 never landed
    missing = out_root / "missing_frames"
    _write_frames(missing, FIXTURE_FRAMES, skip_depth=(3,))
    _write_json(missing / "index.json", _index(FIXTURE_FRAMES))
    _write_json(missing / "metadata.json", full_metadata())

    # 3. non-monotonic timestamps: frame 3 repeats frame 2, frame 4 goes back
    stamps = [0.0, 1 / 30, 2 / 30, 2 / 30, 1 / 30, 5 / 30]
    non_mono = out_root / "non_monotonic"
    _write_frames(non_mono, FIXTURE_FRAMES)
    _write_json(non_mono / "index.json", _index(FIXTURE_FRAMES, timestamps=stamps))
    _write_json(non_mono / "metadata.json", full_metadata())

    # 4. gap: a 0.5 s hole at frame 4
    gap_stamps = [0.0, 1 / 30, 2 / 30, 3 / 30, 3 / 30 + 0.5, 3 / 30 + 0.5 + 1 / 30]
    gap = out_root / "gap"
    _write_frames(gap, FIXTURE_FRAMES)
    _write_json(gap / "index.json", _index(FIXTURE_FRAMES, timestamps=gap_stamps))
    _write_json(gap / "metadata.json", full_metadata())

    # 5. millimetre depth scale: value 1.0 mm/unit declared as such
    mm = out_root / "mm_depth"
    _write_frames(mm, FIXTURE_FRAMES)
    _write_json(
        mm / "index.json",
        _index(FIXTURE_FRAMES, depth_unit="mm_per_unit", depth_scale=1.0),
    )
    _write_json(mm / "metadata.json", full_metadata())

    # 5b. millimetre value smuggled into a metres field (the 1000x error)
    mm_val = out_root / "mm_value_in_metres_field"
    _write_frames(mm_val, FIXTURE_FRAMES)
    _write_json(mm_val / "index.json", _index(FIXTURE_FRAMES))
    record = full_metadata()
    record["fields"]["depth_scale_m_per_unit"] = _datum(
        1.0, "fields.depth_scale_m_per_unit.value", "m_per_unit"
    )
    _write_json(mm_val / "metadata.json", record)

    # 6. unknown intrinsics: the truthful repo state
    unknown = out_root / "unknown_intrinsics"
    _write_frames(unknown, FIXTURE_FRAMES)
    _write_json(unknown / "index.json", _index(FIXTURE_FRAMES))
    _write_json(unknown / "metadata.json", unknown_metadata())

    # 7. bad provenance: a witness file whose sha256 does not match
    bad_sha = out_root / "bad_sha"
    _write_frames(bad_sha, FIXTURE_FRAMES)
    _write_json(bad_sha / "index.json", _index(FIXTURE_FRAMES))
    witness = out_root / "witness_lab_note.json"
    _write_json(
        witness,
        {
            "note": "SYNTHETIC witness file: stands in for a committed lab record",
            "fx_px": FIXTURE_FX,
        },
    )
    record = full_metadata()
    record["fields"]["fx_px"] = {
        "value": FIXTURE_FX,
        "unit": "px",
        "source": {
            "file": "../witness_lab_note.json",
            "sha256": "0" * 64,
            "locator": "fx_px",
        },
    }
    _write_json(bad_sha / "metadata.json", record)

    # 7b. good provenance against an external witness (the real-data code path)
    ext = out_root / "external_witness"
    _write_frames(ext, FIXTURE_FRAMES)
    _write_json(ext / "index.json", _index(FIXTURE_FRAMES))
    record = full_metadata()
    digest = hashlib.sha256(witness.read_bytes()).hexdigest()
    record["fields"]["fx_px"] = {
        "value": FIXTURE_FX,
        "unit": "px",
        "source": {"file": "../witness_lab_note.json", "sha256": digest,
                   "locator": "fx_px"},
    }
    _write_json(ext / "metadata.json", record)

    # 8. degree-valued FOV smuggled into a radian field
    deg = out_root / "deg_fov"
    _write_frames(deg, FIXTURE_FRAMES)
    _write_json(deg / "index.json", _index(FIXTURE_FRAMES))
    record = full_metadata()
    record["fields"]["hfov_rad"] = _datum(87.0, "fields.hfov_rad.value", "rad")
    _write_json(deg / "metadata.json", record)

    # 9. no fps anywhere: gap threshold cannot be derived
    no_fps = out_root / "no_fps"
    _write_frames(no_fps, FIXTURE_FRAMES)
    index = _index(FIXTURE_FRAMES)
    del index["fps"]
    _write_json(no_fps / "index.json", index)
    record = full_metadata()
    del record["fields"]["fps"]
    _write_json(no_fps / "metadata.json", record)

    # 10. undeclared alignment flag
    no_align = out_root / "alignment_undeclared"
    _write_frames(no_align, FIXTURE_FRAMES)
    index = _index(FIXTURE_FRAMES)
    del index["rgb_depth_aligned"]
    _write_json(no_align / "index.json", index)
    record = full_metadata()
    del record["fields"]["rgb_depth_aligned"]
    _write_json(no_align / "metadata.json", record)

    # manifest last: sha256 of every produced file
    manifest = {}
    for path in sorted(out_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest[str(path.relative_to(out_root)).replace("\\", "/")] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
    _write_json(
        out_root / "manifest.json",
        {
            "schema": "p6_d455_fixture_manifest/1",
            "data_class": "OFFLINE-SYNTHETIC",
            "note": SYNTHETIC_NOTE,
            "generated_by": "tools/p6_frames/make_d455_replay_fixtures.py",
            "files": manifest,
        },
    )
    return out_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    root = build(Path(args.out))
    count = len(list(root.rglob("*")))
    print(f"wrote {root} ({count} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
