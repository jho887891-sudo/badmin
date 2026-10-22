# -*- coding: utf-8 -*-
"""P6-03 (R+1 item C3): Intel RealSense D455 metadata contract.

Why this module exists
----------------------
The repository holds NO calibrated D455 quantity (see
outputs/simulation/rebot_b601_rs/morph_one/p6_frames/camera_metadata.json: every
number is UNKNOWN with the exact missing datum named). The failure mode this
module exists to prevent is that someone later writes

    fx = 391.0

into a config file and everybody downstream believes it. So here a quantity is
not a float, it is a *datum*: a value bound to the bytes that carry it
{file, sha256, locator}. If the binding cannot be verified the load fails.
Absent data is the explicit UNKNOWN sentinel, never a plausible default.

The one thing that IS proven in this repository, and is therefore implemented
here as a convention (not as a calibration):
the camera OPTICAL frame and the robot MOUNT frame differ by a fixed cyclic
permutation with det = +1.  Repo statement (P6-02 record, authoritative):

    (X, Y, Z)_mount   = (Z, -X, -Y)_optical
    (X, Y, Z)_optical = (-Y, -Z, X)_mount          <- inverse of the line above
    R(optical <- mount) = [[0,-1,0],[0,0,-1],[1,0,0]], det = +1

Note on the task text: "P6-03 ... (X,Y,Z)_optical = (Z,-X,-Y)_mount" labels a
direction that contradicts the committed matrix. The committed matrix is the
record (this module is checked against it in the test suite), so both directions
are implemented explicitly and never conflated. The permutation is the same
proven fact; only the arrow differs.

No calibration is claimed by this module. It can CONSUME a real record; it can
never produce one.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_ID = "p6_d455_metadata/1"


class _Unknown:
    """Singleton sentinel for "not known". Deliberately falsy, not a number."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash("<p6-d455-unknown>")


UNKNOWN = _Unknown()


def is_known(value: Any) -> bool:
    """True only for a real Datum carrying a verified provenance."""
    return isinstance(value, Datum)


@dataclass(frozen=True)
class Provenance:
    """Where a value comes from. Every verified datum carries one.

    file    : repo-relative path of the bytes that state the value, or "@self"
              meaning "the metadata record that is being loaded".
    sha256  : lowercase hex digest of those bytes. For "@self" the loader
              substitutes the digest it computes from the record itself and
              does not accept a caller-supplied digest.
    locator : a dotted path that must resolve inside that file.
    """

    file: str
    sha256: str
    locator: str

    @property
    def is_self(self) -> bool:
        return self.file == "@self"


@dataclass(frozen=True)
class Datum:
    """A verified value plus its provenance and declared unit."""

    value: Any
    source: Provenance
    unit: str | None = None

    def as_float(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"Datum({self.value!r}, unit={self.unit!r}, "
            f"from={self.source.file}#{self.source.locator})"
        )


PX = "px"
M_PER_UNIT = "m_per_unit"
M = "m"
RAD = "rad"
HZ = "Hz"
DIMENSIONLESS = "dimensionless"
S = "s"

CANONICAL_UNITS: dict[str, str] = {
    "intrinsics": PX,
    "depth_scale": M_PER_UNIT,
    "length": M,
    "angle": RAD,
    "rate": HZ,
    "distortion": DIMENSIONLESS,
    "time": S,
}

# Accepted spellings that mean exactly the canonical unit. Everything else is
# rejected outright instead of being silently converted: a conversion is a
# decision, and it belongs to whoever owns the record, not to a loader.
UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    PX: ("px", "pixel", "pixels"),
    M_PER_UNIT: ("m_per_unit", "meter_per_unit", "metre_per_unit"),
    M: ("m", "meter", "metre", "meters", "metres"),
    RAD: ("rad", "radian", "radians"),
    HZ: ("Hz", "hz", "fps", "1/s"),
    DIMENSIONLESS: ("dimensionless", "1", "-", "none"),
    S: ("s", "sec", "second", "seconds"),
}


class D455MetadataError(Exception):
    """Base class for every refusal raised by this module."""


class UnitViolationError(D455MetadataError):
    """A declared unit is not the canonical unit for that field."""


class ProvenanceViolationError(D455MetadataError):
    """A value is not bound to verifiable bytes."""


class ConsistencyViolationError(D455MetadataError):
    """Values are individually present but mutually impossible."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("metadata is inconsistent: " + "; ".join(self.problems))


class MetadataIncompleteError(D455MetadataError):
    """A calibrated object was requested but required fields are UNKNOWN."""

    def __init__(self, missing: Sequence[str], level: str):
        self.missing = list(missing)
        self.level = level
        super().__init__(
            f"cannot produce {level}: {len(self.missing)} required field(s) UNKNOWN: "
            + ", ".join(self.missing)
        )


class MountFrameUnknownError(D455MetadataError):
    """A frame conversion was requested while T_mount_camera is UNKNOWN."""


class FrameConventionError(D455MetadataError):
    """A rotation matrix that is not a right-handed rotation (det != +1)."""


DISTORTION_MODELS: dict[str, int] = {
    "none": 0,
    "brown_conrady": 5,
    "modified_brown_conrady": 5,
    "inverse_brown_conrady": 5,
    "brown_conrady_8": 8,
    "rational_polynomial": 8,
    "fisheye": 4,
    "kannala_brandt4": 4,
}

# p_mount = OPTICAL_TO_MOUNT @ p_optical  ->  (Z, -X, -Y)
OPTICAL_TO_MOUNT: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)

# p_optical = MOUNT_TO_OPTICAL @ p_mount  ->  (-Y, -Z, X)
MOUNT_TO_OPTICAL: tuple[tuple[float, ...], ...] = (
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
)

OPTICAL_TO_MOUNT_QWXYZ: tuple[float, ...] = (0.5, 0.5, -0.5, 0.5)


def matmul3(a, b) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def transpose3(a) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def det3(m) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def apply3(m, v) -> tuple[float, float, float]:
    if len(v) != 3:
        raise ValueError(f"expected a 3-vector, got {len(v)} components")
    return tuple(
        sum(m[i][k] * float(v[k]) for k in range(3)) for i in range(3)
    )


def mat_from_quat_wxyz(q) -> list[list[float]]:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise FrameConventionError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def require_rotation(matrix, *, name: str, atol: float = 1e-9) -> list[list[float]]:
    """Right-handed orthonormal 3x3 or refuse. det must be +1, not -1."""
    m = [[float(v) for v in row] for row in matrix]
    if len(m) != 3 or any(len(row) != 3 for row in m):
        raise FrameConventionError(f"{name}: expected a 3x3 matrix")
    for i in range(3):
        for j in range(3):
            dot = sum(m[k][i] * m[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > atol:
                raise FrameConventionError(
                    f"{name}: columns {i},{j} are not orthonormal (dot={dot!r})"
                )
    d = det3(m)
    if abs(d - 1.0) > 1e-6:
        raise FrameConventionError(
            f"{name}: det={d!r} != +1 (left-handed / mirrored basis)"
        )
    return m


def optical_to_mount(p) -> tuple[float, float, float]:
    """(X,Y,Z)_optical -> (Z,-X,-Y)_mount. Pure convention, no metadata needed."""
    return apply3(OPTICAL_TO_MOUNT, p)


def mount_to_optical(p) -> tuple[float, float, float]:
    """(X,Y,Z)_mount -> (-Y,-Z,X)_optical. Exact inverse of optical_to_mount."""
    return apply3(MOUNT_TO_OPTICAL, p)


def check_optical_mount_contract(camera_metadata_path) -> dict[str, Any]:
    """Verify this module's permutation against the committed P6-02 record.

    Reads (never writes) outputs/.../p6_frames/camera_metadata.json and checks
    that its rotation_from_mount_to_optical matrix equals MOUNT_TO_OPTICAL and
    that both directions are det = +1. Raises FrameConventionError on drift.
    """
    path = Path(camera_metadata_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    node = record["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]
    committed = [[float(v) for v in row] for row in node["matrix"]]
    require_rotation(committed, name="committed R(optical<-mount)")
    for i in range(3):
        for j in range(3):
            if abs(committed[i][j] - MOUNT_TO_OPTICAL[i][j]) > 1e-12:
                raise FrameConventionError(
                    "committed camera_metadata.json rotation does not match "
                    f"MOUNT_TO_OPTICAL at ({i},{j})"
                )
    quat = tuple(float(v) for v in node["values"])
    require_rotation(mat_from_quat_wxyz(quat), name="committed wxyz quaternion")
    require_rotation(OPTICAL_TO_MOUNT, name="OPTICAL_TO_MOUNT")
    require_rotation(MOUNT_TO_OPTICAL, name="MOUNT_TO_OPTICAL")
    identity = matmul3(OPTICAL_TO_MOUNT, MOUNT_TO_OPTICAL)
    for i in range(3):
        for j in range(3):
            if abs(identity[i][j] - (1.0 if i == j else 0.0)) > 1e-12:
                raise FrameConventionError("optical->mount->optical is not identity")
    return {
        "committed_matrix_equals_MOUNT_TO_OPTICAL": True,
        "quaternion_matches_matrix": True,
        "det_optical_to_mount": det3(OPTICAL_TO_MOUNT),
        "det_mount_to_optical": det3(MOUNT_TO_OPTICAL),
        "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "CONVENTION_NOT_CALIBRATION",
    }


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    unit_key: str | None
    required_for: tuple[str, ...]
    doc: str


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("fx_px", "float", "intrinsics", ("intrinsics", "full"),
              "focal length x, pixels"),
    FieldSpec("fy_px", "float", "intrinsics", ("intrinsics", "full"),
              "focal length y, pixels"),
    FieldSpec("cx_px", "float", "intrinsics", ("intrinsics", "full"),
              "principal point u, pixels"),
    FieldSpec("cy_px", "float", "intrinsics", ("intrinsics", "full"),
              "principal point v, pixels"),
    FieldSpec("width_px", "int", "intrinsics", ("intrinsics", "full"),
              "stream width, pixels"),
    FieldSpec("height_px", "int", "intrinsics", ("intrinsics", "full"),
              "stream height, pixels"),
    FieldSpec("depth_scale_m_per_unit", "float", "depth_scale",
              ("intrinsics", "full"),
              "metres per raw depth unit (per device; never a constant)"),
    FieldSpec("depth_encoding", "str", None, ("intrinsics", "full"),
              "e.g. uint16"),
    FieldSpec("fps", "float", "rate", ("intrinsics", "full"),
              "configured frame rate"),
    FieldSpec("rgb_depth_aligned", "bool", None, ("intrinsics", "full"),
              "True iff depth is expressed in the colour optical frame"),
    FieldSpec("distortion_model", "str", None, ("intrinsics", "full"),
              "model name, see DISTORTION_MODELS"),
    FieldSpec("distortion_coefficients", "floats", "distortion",
              ("intrinsics", "full"), "coefficients for that model"),
    FieldSpec("hfov_rad", "float", "angle", (), "horizontal FOV, radians"),
    FieldSpec("vfov_rad", "float", "angle", (), "vertical FOV, radians"),
    FieldSpec("T_mount_camera", "matrix4", "length", ("full",),
              "4x4 SE(3) mapping camera-optical coordinates into the mount frame"),
    FieldSpec("device_model", "str", None, (), "device model recorded by the unit"),
    FieldSpec("device_serial", "str", None, (), "serial number of the mounted unit"),
    FieldSpec("device_firmware", "str", None, (), "firmware version"),
)

FIELD_BY_NAME = {spec.name: spec for spec in FIELD_SPECS}

# Rejection envelope for depth scale. NOT a claimed device value: it only exists
# so a millimetre-per-unit number copied into a metre-per-unit field
# (0.001 -> 1.0, the classic 1000x error) cannot pass. A 16-bit stream at
# 0.1 m/unit would already span 6.5 km, so anything >= 0.1 is not metres/unit.
DEPTH_SCALE_M_PER_UNIT_MAX = 0.1


@dataclass
class Units:
    """Declared units of the record. Guarded, never guessed."""

    intrinsics: str = PX
    depth_scale: str = M_PER_UNIT
    length: str = M
    angle: str = RAD
    rate: str = HZ
    distortion: str = DIMENSIONLESS
    time: str = S
    raw: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Units":
        raw = {str(k): str(v) for k, v in mapping.items()}
        for key, canonical in CANONICAL_UNITS.items():
            if key not in raw:
                continue
            declared = raw[key].strip()
            accepted = UNIT_ALIASES[canonical]
            if declared not in accepted:
                raise UnitViolationError(
                    f"units.{key} = {declared!r} is not the canonical unit "
                    f"{canonical!r} (accepted spellings: {', '.join(accepted)}). "
                    "Convert and re-declare in the owning record; this loader "
                    "never converts silently."
                )
        kwargs = {key: raw.get(key, CANONICAL_UNITS[key]) for key in CANONICAL_UNITS}
        for key, canonical in CANONICAL_UNITS.items():
            if raw.get(key) in UNIT_ALIASES[canonical]:
                kwargs[key] = canonical
        return cls(**kwargs, raw=raw)

    def canonical(self, key: str) -> str:
        return str(getattr(self, key))


@dataclass
class D455Metadata:
    """Every field is either a verified Datum or the UNKNOWN sentinel.

    There are no numeric defaults anywhere in this class: constructing it
    without evidence yields an all-UNKNOWN object, which is the truthful state
    of this repository.
    """

    fx_px: Any = UNKNOWN
    fy_px: Any = UNKNOWN
    cx_px: Any = UNKNOWN
    cy_px: Any = UNKNOWN
    width_px: Any = UNKNOWN
    height_px: Any = UNKNOWN
    depth_scale_m_per_unit: Any = UNKNOWN
    depth_encoding: Any = UNKNOWN
    fps: Any = UNKNOWN
    rgb_depth_aligned: Any = UNKNOWN
    distortion_model: Any = UNKNOWN
    distortion_coefficients: Any = UNKNOWN
    hfov_rad: Any = UNKNOWN
    vfov_rad: Any = UNKNOWN
    T_mount_camera: Any = UNKNOWN
    device_model: Any = UNKNOWN
    device_serial: Any = UNKNOWN
    device_firmware: Any = UNKNOWN

    units: Units = field(default_factory=Units)
    record_path: Path | None = None
    record_sha256: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        if name not in FIELD_BY_NAME:
            raise KeyError(f"unknown field {name!r}")
        return getattr(self, name)

    def known_fields(self) -> list[str]:
        return [spec.name for spec in FIELD_SPECS if is_known(self.get(spec.name))]

    def unknown_fields(self) -> list[str]:
        return [spec.name for spec in FIELD_SPECS if not is_known(self.get(spec.name))]

    def missing_required(self, level: str) -> list[str]:
        return [
            spec.name
            for spec in FIELD_SPECS
            if level in spec.required_for and not is_known(self.get(spec.name))
        ]

    @property
    def status(self) -> str:
        if not self.missing_required("full"):
            return "CALIBRATED_FULL"
        if not self.missing_required("intrinsics"):
            return "CALIBRATED_INTRINSICS"
        return "INCOMPLETE"

    @property
    def is_calibrated(self) -> bool:
        return self.status in ("CALIBRATED_INTRINSICS", "CALIBRATED_FULL")

    def require_calibrated_intrinsics(self) -> "D455Metadata":
        missing = self.missing_required("intrinsics")
        if missing:
            raise MetadataIncompleteError(missing, "CALIBRATED_INTRINSICS")
        return self

    def require_calibrated_full(self) -> "D455Metadata":
        missing = self.missing_required("full")
        if missing:
            raise MetadataIncompleteError(missing, "CALIBRATED_FULL")
        return self

    def intrinsics_matrix(self) -> list[list[float]]:
        self.require_calibrated_intrinsics()
        return [
            [self.fx_px.as_float(), 0.0, self.cx_px.as_float()],
            [0.0, self.fy_px.as_float(), self.cy_px.as_float()],
            [0.0, 0.0, 1.0],
        ]

    def T_mount_camera_matrix(self) -> list[list[float]]:
        if not is_known(self.T_mount_camera):
            raise MountFrameUnknownError(
                "T_mount_camera is UNKNOWN: the camera pose on the robot is a "
                "hand-eye measurement (or a measured CAD mount), not a camera "
                "fact. The axis convention alone cannot place a point on the robot."
            )
        return [[float(v) for v in row] for row in self.T_mount_camera.value]

    def optical_point_to_mount(self, p_optical) -> tuple[float, float, float]:
        """Apply the SE(3): p_mount = R @ p_optical + t. Refuses when R/t UNKNOWN."""
        T = self.T_mount_camera_matrix()
        R = [row[:3] for row in T[:3]]
        t = [T[0][3], T[1][3], T[2][3]]
        rotated = apply3(R, p_optical)
        return (rotated[0] + t[0], rotated[1] + t[1], rotated[2] + t[2])

    def axis_convention_only_point_to_mount(self, p_optical):
        """The convention half of the transform, with NO translation.

        Only meaningful when the camera and mount origins coincide; it exists so
        callers can see the 90-degree failure on its own. It does not place a
        point on the robot and must not be used for geometry.
        """
        return optical_to_mount(p_optical)


def validate_consistency(meta: D455Metadata) -> list[str]:
    """Return the list of mutually-impossible combinations. Empty == consistent."""
    problems: list[str] = []

    if meta.units.depth_scale != M_PER_UNIT:
        problems.append(
            f"units.depth_scale={meta.units.depth_scale!r}, expected {M_PER_UNIT!r}"
        )

    fx = meta.fx_px if is_known(meta.fx_px) else None
    fy = meta.fy_px if is_known(meta.fy_px) else None
    cx = meta.cx_px if is_known(meta.cx_px) else None
    cy = meta.cy_px if is_known(meta.cy_px) else None
    w = meta.width_px if is_known(meta.width_px) else None
    h = meta.height_px if is_known(meta.height_px) else None

    if fx is not None and not (fx.as_float() > 0.0):
        problems.append(f"fx_px={fx.value!r} must be > 0")
    if fy is not None and not (fy.as_float() > 0.0):
        problems.append(f"fy_px={fy.value!r} must be > 0")
    if w is not None and int(w.value) <= 0:
        problems.append(f"width_px={w.value!r} must be > 0")
    if h is not None and int(h.value) <= 0:
        problems.append(f"height_px={h.value!r} must be > 0")
    if cx is not None and w is not None:
        if not (0.0 <= cx.as_float() < float(w.value)):
            problems.append(
                f"cx_px={cx.value!r} outside the image [0, {int(w.value)}) "
                f"for width {int(w.value)}"
            )
    if cy is not None and h is not None:
        if not (0.0 <= cy.as_float() < float(h.value)):
            problems.append(
                f"cy_px={cy.value!r} outside the image [0, {int(h.value)}) "
                f"for height {int(h.value)}"
            )

    scale = (
        meta.depth_scale_m_per_unit
        if is_known(meta.depth_scale_m_per_unit)
        else None
    )
    if scale is not None:
        v = scale.as_float()
        if not (v > 0.0):
            problems.append(f"depth_scale_m_per_unit={scale.value!r} must be > 0")
        elif v >= DEPTH_SCALE_M_PER_UNIT_MAX:
            problems.append(
                f"depth_scale_m_per_unit={scale.value!r} >= "
                f"{DEPTH_SCALE_M_PER_UNIT_MAX} (looks like millimetres-per-unit "
                "written into a metres-per-unit field)"
            )

    fps = meta.fps if is_known(meta.fps) else None
    if fps is not None and not (fps.as_float() > 0.0):
        problems.append(f"fps={fps.value!r} must be > 0")

    aligned = meta.rgb_depth_aligned if is_known(meta.rgb_depth_aligned) else None
    if aligned is not None and not isinstance(aligned.value, bool):
        problems.append(
            f"rgb_depth_aligned={aligned.value!r} must be a bool, not a truthy non-bool"
        )

    model = meta.distortion_model if is_known(meta.distortion_model) else None
    coeffs = (
        meta.distortion_coefficients
        if is_known(meta.distortion_coefficients)
        else None
    )
    if model is not None and str(model.value) not in DISTORTION_MODELS:
        problems.append(
            f"distortion_model={model.value!r} is not one of {sorted(DISTORTION_MODELS)}"
        )
    if model is not None and str(model.value) in DISTORTION_MODELS:
        expected = DISTORTION_MODELS[str(model.value)]
        if coeffs is not None and len(coeffs.value) != expected:
            problems.append(
                f"distortion_model={model.value!r} needs {expected} coefficients, "
                f"got {len(coeffs.value)}"
            )
        if coeffs is None and expected > 0:
            problems.append(
                f"distortion_model={model.value!r} needs {expected} coefficients, "
                "but the coefficient vector is UNKNOWN"
            )
    if coeffs is not None and model is None:
        problems.append("distortion_coefficients present but distortion_model UNKNOWN")

    for name in ("hfov_rad", "vfov_rad"):
        datum = meta.get(name)
        if is_known(datum):
            if meta.units.angle != RAD:
                problems.append(f"units.angle={meta.units.angle!r}, expected {RAD!r}")
                break
            v = datum.as_float()
            if not (0.0 < v < math.pi + 1e-9):
                problems.append(
                    f"{name}={datum.value!r} rad is outside (0, pi); "
                    "a value near 87 is degrees"
                )

    T = meta.T_mount_camera if is_known(meta.T_mount_camera) else None
    if T is not None:
        matrix = [[float(v) for v in row] for row in T.value]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            problems.append("T_mount_camera must be 4x4")
        else:
            last = matrix[3]
            if any(abs(last[i] - (1.0 if i == 3 else 0.0)) > 1e-12 for i in range(4)):
                problems.append(
                    f"T_mount_camera bottom row must be [0,0,0,1], got {last!r}"
                )
            try:
                require_rotation([row[:3] for row in matrix[:3]],
                                 name="T_mount_camera")
            except FrameConventionError as exc:
                problems.append(str(exc))

    return problems


def require_consistent(meta: D455Metadata) -> D455Metadata:
    problems = validate_consistency(meta)
    if problems:
        raise ConsistencyViolationError(problems)
    return meta


_MISSING_MARKERS = {"UNKNOWN", "unknown", "MISSING", "missing", "TBD", "null"}


def _resolve_locator(document: Any, locator: str) -> Any:
    node = document
    for part in locator.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                raise ProvenanceViolationError(
                    f"locator {locator!r} does not resolve (missing key {part!r})"
                )
            node = node[part]
        elif isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                raise ProvenanceViolationError(
                    f"locator {locator!r} does not resolve in a list"
                ) from None
        else:
            raise ProvenanceViolationError(
                f"locator {locator!r} descends into a {type(node).__name__}"
            )
    return node


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_provenance(
    source_node,
    *,
    record_path: Path,
    record_document,
    repo_root: Path | None,
    field_name: str,
    claimed_value: Any,
) -> Provenance:
    """Verify bytes, locator AND witnessed value; a real hash alone is not enough."""
    if not isinstance(source_node, Mapping):
        raise ProvenanceViolationError(
            "every value needs a 'source' object with {file, sha256, locator}; got "
            f"{type(source_node).__name__}"
        )
    file_field = source_node.get("file")
    locator = source_node.get("locator")
    declared_sha = source_node.get("sha256")
    if not isinstance(file_field, str) or not file_field:
        raise ProvenanceViolationError("source.file must be a non-empty string")
    if not isinstance(locator, str) or not locator:
        raise ProvenanceViolationError("source.locator must be a non-empty string")

    if file_field == "@self":
        actual = hashlib.sha256(record_path.read_bytes()).hexdigest()
        if declared_sha is not None and str(declared_sha) != actual:
            raise ProvenanceViolationError(
                f"source.sha256 for @self is {declared_sha!r} but the record hashes "
                f"to {actual!r}"
            )
        # Self-references provide only self-consistency, not an independent measurement.
        # Do not let fx silently cite fy (which may happen to have the same value).
        expected = f"fields.{field_name}.value"
        witnessed_value = _resolve_locator(record_document, locator)
        if locator != expected:
            raise ProvenanceViolationError(
                f"{field_name}: @self locator {locator!r} must equal {expected!r}"
            )
        if witnessed_value != claimed_value:
            raise ProvenanceViolationError(
                f"{field_name}: @self locator value does not match the field value"
            )
        return Provenance(file="@self", sha256=actual, locator=locator)

    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        raise ProvenanceViolationError(
            f"source.sha256 for {file_field!r} must be a 64-char hex digest"
        )
    root = repo_root if repo_root is not None else record_path.parent
    target = Path(file_field)
    if not target.is_absolute():
        target = Path(root) / target
    if not target.is_file():
        raise ProvenanceViolationError(
            f"source.file {file_field!r} does not exist (resolved to {target})"
        )
    actual = sha256_file(target)
    if actual != declared_sha:
        raise ProvenanceViolationError(
            f"source.file {file_field!r} hashes to {actual}, record declares "
            f"{declared_sha}"
        )
    # SHA proves which witness file is referenced, not that the claimed field value
    # agrees with the witness. Without this check fx=600 could cite a valid fx=300.
    try:
        witness = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceViolationError(
            f"{field_name}: source {file_field!r} is not structured JSON; "
            "no verified locator reader is registered for this file type"
        ) from exc
    witnessed_value = _resolve_locator(witness, locator)
    if witnessed_value != claimed_value:
        raise ProvenanceViolationError(
            f"{field_name}: value {claimed_value!r} disagrees with witness "
            f"{file_field!r} locator {locator!r} = {witnessed_value!r}"
        )
    return Provenance(file=file_field, sha256=actual, locator=locator)


def _coerce(spec: FieldSpec, value: Any, unit):
    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise D455MetadataError(f"{spec.name}: expected a number, got {value!r}")
        return float(value)
    if spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise D455MetadataError(f"{spec.name}: expected an integer, got {value!r}")
        as_float = float(value)
        if as_float != int(as_float):
            raise D455MetadataError(f"{spec.name}: expected an integer, got {value!r}")
        return int(as_float)
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise D455MetadataError(
                f"{spec.name}: expected a JSON bool, got {value!r}"
            )
        return value
    if spec.kind == "str":
        if not isinstance(value, str) or not value:
            raise D455MetadataError(
                f"{spec.name}: expected a non-empty string, got {value!r}"
            )
        return value
    if spec.kind == "floats":
        if not isinstance(value, (list, tuple)) or not value:
            raise D455MetadataError(
                f"{spec.name}: expected a non-empty list of numbers"
            )
        out = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise D455MetadataError(
                    f"{spec.name}: non-numeric coefficient {item!r}"
                )
            out.append(float(item))
        return out
    if spec.kind == "matrix4":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise D455MetadataError(f"{spec.name}: expected a 4x4 nested list")
        matrix = []
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise D455MetadataError(f"{spec.name}: expected a 4x4 nested list")
            matrix.append([float(v) for v in row])
        return matrix
    raise D455MetadataError(f"{spec.name}: unsupported kind {spec.kind}")


def parse_metadata_document(
    document: Mapping[str, Any],
    *,
    record_path: Path,
    repo_root: Path | None = None,
    require_complete: bool = False,
    require_consistent_values: bool = True,
) -> D455Metadata:
    """Parse an already-loaded record. Never invents a value."""
    record_path = Path(record_path)
    schema = document.get("schema")
    if schema != SCHEMA_ID:
        raise D455MetadataError(
            f"schema mismatch: expected {SCHEMA_ID!r}, got {schema!r}"
        )
    units_node = document.get("units")
    if not isinstance(units_node, Mapping):
        raise UnitViolationError(
            "record has no 'units' block; units must be explicit for every quantity"
        )
    units = Units.from_mapping(units_node)

    meta = D455Metadata(units=units, record_path=record_path)
    meta.record_sha256 = hashlib.sha256(record_path.read_bytes()).hexdigest()

    fields_node = document.get("fields")
    if not isinstance(fields_node, Mapping):
        raise D455MetadataError("record has no 'fields' object")

    missing_reasons: dict[str, str] = {}
    for name, raw in fields_node.items():
        if name not in FIELD_BY_NAME:
            raise D455MetadataError(f"unknown field {name!r} in record")
        spec = FIELD_BY_NAME[name]
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise D455MetadataError(
                f"field {name!r} must be an object, got {type(raw).__name__}"
            )
        status = raw.get("status")
        has_value = "value" in raw
        if status in _MISSING_MARKERS or (not has_value and status is not None):
            if has_value:
                raise ProvenanceViolationError(
                    f"field {name!r} is marked {status!r} but also carries a value"
                )
            reason = raw.get("missing_datum")
            missing_reasons[name] = (
                str(reason) if reason else "no missing_datum recorded"
            )
            continue
        if not has_value:
            missing_reasons[name] = "record omits the value"
            continue
        source = resolve_provenance(
            raw.get("source"),
            record_path=record_path,
            record_document=document,
            repo_root=repo_root,
            field_name=name,
            claimed_value=raw["value"],
        )
        declared_unit = raw.get("unit")
        if spec.unit_key is not None:
            canonical = units.canonical(spec.unit_key)
            if declared_unit is None:
                raise UnitViolationError(
                    f"field {name!r} needs an explicit 'unit' "
                    f"(canonical {canonical!r})"
                )
            declared = str(declared_unit).strip()
            accepted = UNIT_ALIASES[canonical]
            if declared not in accepted:
                raise UnitViolationError(
                    f"field {name!r} unit {declared!r} is not the canonical "
                    f"{canonical!r} (accepted: {', '.join(accepted)})"
                )
        value = _coerce(spec, raw["value"], declared_unit)
        setattr(meta, name, Datum(value=value, source=source, unit=declared_unit))

    if missing_reasons:
        meta.notes["unknown_reasons"] = missing_reasons

    if require_consistent_values:
        require_consistent(meta)
    meta.notes["state"] = meta.status
    if require_complete and meta.status != "CALIBRATED_FULL":
        raise MetadataIncompleteError(meta.missing_required("full"),
                                      "CALIBRATED_FULL")
    return meta


def load_metadata(
    path,
    *,
    repo_root=None,
    require_complete: bool = False,
    require_consistent_values: bool = True,
) -> D455Metadata:
    """Load a D455 metadata record (JSON, or YAML when PyYAML is installed).

    There is no fallback parser that could quietly misinterpret a number: if
    YAML is requested and PyYAML is absent, the load fails.
    """
    record_path = Path(path)
    text = record_path.read_text(encoding="utf-8")
    if record_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise D455MetadataError(
                "PyYAML is not installed; convert the record to JSON or install "
                "PyYAML"
            ) from exc
        document = yaml.safe_load(text)
    else:
        document = json.loads(text)
    if not isinstance(document, Mapping):
        raise D455MetadataError("record root must be a mapping")
    return parse_metadata_document(
        document,
        record_path=record_path,
        repo_root=Path(repo_root) if repo_root is not None else None,
        require_complete=require_complete,
        require_consistent_values=require_consistent_values,
    )


@dataclass
class FrameIndexProbe:
    """What a folder of (rgb, depth, timestamp) triplets can prove on its own.

    It cannot prove intrinsics, depth scale or alignment; those are record
    facts. It is here so a real capture can be consumed the moment one exists.
    """

    root: Path
    rgb_count: int
    depth_count: int
    timestamps_s: list[float]
    declared_resolution: tuple[int, int] | None = None
    index_path: Path | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return not self.problems


def probe_frame_index(index_path) -> FrameIndexProbe:
    """Read a timestamp index (JSON) describing an rgb/depth capture.

    Expected shape::

        {"schema": "p6_d455_replay_index/1",
         "frames": [{"rgb": "rgb/000000.png", "depth": "depth/000000.png",
                     "timestamp_s": 0.0}, ...]}

    Paths are resolved relative to the index's own directory.
    """
    index_path = Path(index_path)
    document = json.loads(index_path.read_text(encoding="utf-8"))
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise D455MetadataError("frame index has no 'frames' list")
    problems: list[str] = []
    rgb_paths: list[Path] = []
    depth_paths: list[Path] = []
    ts: list[float] = []
    for i, frame in enumerate(frames):
        for key, bucket in (("rgb", rgb_paths), ("depth", depth_paths)):
            rel = frame.get(key)
            if not isinstance(rel, str):
                problems.append(f"frame[{i}] has no {key} path")
                continue
            candidate = index_path.parent / rel
            if candidate.is_file():
                bucket.append(candidate)
            else:
                problems.append(f"frame[{i}] {key} file missing: {rel}")
        stamp = frame.get("timestamp_s")
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
            ts.append(float(stamp))
        else:
            problems.append(f"frame[{i}] has no numeric timestamp_s")
    resolution = document.get("resolution")
    declared = None
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        declared = (int(resolution[0]), int(resolution[1]))
    return FrameIndexProbe(
        root=index_path.parent,
        rgb_count=len(rgb_paths),
        depth_count=len(depth_paths),
        timestamps_s=ts,
        declared_resolution=declared,
        index_path=index_path,
        problems=problems,
    )


@dataclass
class RosbagInfo:
    """Subset of the ros2 bag info output that matters to this contract."""

    path: str | None
    duration_s: float | None
    message_count: int | None
    topics: dict[str, dict[str, Any]]
    raw: str


def parse_rosbag_info(text: str, *, path: str | None = None) -> RosbagInfo:
    """Parse the ros2 bag info text. No ROS import, no bag reading.

    It yields topic/message-count facts only; it can never yield intrinsics.
    """
    duration: float | None = None
    count: int | None = None
    topics: dict[str, dict[str, Any]] = {}

    # ros2 prints "Duration: 2.033s" / "Messages: 147"; rosbag1 prints
    # "duration:" / "message_count:". Accept both spellings, never guess.
    match = re.search(r"^\s*duration:\s*([0-9]+(?:\.[0-9]+)?)\s*s?", text,
                      re.IGNORECASE | re.MULTILINE)
    if match:
        duration = float(match.group(1))
    match = re.search(r"^\s*(?:messages|message count):\s*([0-9]+)", text,
                      re.IGNORECASE | re.MULTILINE)
    if match:
        count = int(match.group(1))
    for match in re.finditer(
        r"Topic:\s*(\S+)\s*\|\s*Type:\s*(\S+)\s*\|\s*Count:\s*([0-9]+)",
        text,
        re.IGNORECASE,
    ):
        topics[match.group(1)] = {
            "type": match.group(2),
            "messages": int(match.group(3)),
        }
    return RosbagInfo(
        path=path, duration_s=duration, message_count=count, topics=topics, raw=text
    )
