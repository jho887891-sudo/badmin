"""Time- and calibration-tagged D455 observations, separate from simulation court frames."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class CaptureStamp:
    frame_seq: int
    capture_time_ns: int
    capture_clock_domain: str
    sync_uncertainty_ns: int

    def __post_init__(self):
        if type(self.frame_seq) is not int or self.frame_seq < 0:
            raise ValueError('frame_seq must be a nonnegative integer')
        for field in ('capture_time_ns', 'sync_uncertainty_ns'):
            x = getattr(self, field)
            if type(x) is not int or x < 0:
                raise ValueError(f'{field} must be a nonnegative integer')
        if not isinstance(self.capture_clock_domain, str) or not self.capture_clock_domain.strip():
            raise ValueError('capture_clock_domain must identify a clock')

@dataclass(frozen=True)
class CalibrationRef:
    calibration_id: str
    d455_serial: str
    color_stream_profile: str
    depth_stream_profile: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (
                self.calibration_id, self.d455_serial, self.color_stream_profile,
                self.depth_stream_profile)):
            raise ValueError('calibration metadata must be nonempty')

@dataclass(frozen=True)
class BallObservation3D:
    stamp: CaptureStamp
    coordinate_frame: str
    xyz_m: tuple[float, float, float] | None
    covariance_3x3_m2: tuple[float, ...] | None
    depth_status: str
    source: str
    calibration: CalibrationRef

    def __post_init__(self):
        if not isinstance(self.stamp, CaptureStamp):
            raise TypeError('stamp must be CaptureStamp')
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            raise ValueError('coordinate_frame must be a name')
        if not isinstance(self.calibration, CalibrationRef):
            raise TypeError('calibration must be CalibrationRef')
        if not self.depth_status or not self.source:
            raise ValueError('depth status and source are required')
        if self.xyz_m is not None and (len(self.xyz_m) != 3 or not all(isfinite(x) for x in self.xyz_m)):
            raise ValueError('xyz_m must be finite xyz')
        if self.xyz_m is None and self.depth_status == 'VALID':
            raise ValueError('valid depth requires an actual observation')
        if self.xyz_m is not None and self.depth_status != 'VALID':
            raise ValueError('invalid depth cannot produce xyz')
        if self.covariance_3x3_m2 is not None and (len(self.covariance_3x3_m2) != 9 or not all(isfinite(x) for x in self.covariance_3x3_m2)):
            raise ValueError('covariance must be finite 3x3')

@dataclass(frozen=True)
class TransformedBall3D:
    stamp: CaptureStamp
    coordinate_frame: str
    xyz_m: tuple[float, float, float] | None
    covariance_3x3_m2: tuple[float, ...] | None
    transform_calibration_id: str
    transform_status: str
