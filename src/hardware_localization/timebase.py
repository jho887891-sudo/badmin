"""Explicit clock-domain bridges; no implicit hardware/ROS epoch mixing."""
from dataclasses import dataclass
from .types import CaptureStamp

@dataclass(frozen=True)
class ClockBridge:
    source_domain: str
    target_domain: str
    offset_ns: int
    uncertainty_ns: int

    def __post_init__(self):
        if not self.source_domain or not self.target_domain:
            raise ValueError('clock domains required')
        if type(self.offset_ns) is not int:
            raise ValueError('offset_ns must be an integer')
        if type(self.uncertainty_ns) is not int or self.uncertainty_ns < 0:
            raise ValueError('uncertainty_ns must be nonnegative')

def convert_capture_time(stamp: CaptureStamp, bridge: ClockBridge | None, expected_domain: str) -> int:
    if stamp.capture_clock_domain == expected_domain:
        return stamp.capture_time_ns
    if bridge is None or bridge.source_domain != stamp.capture_clock_domain or bridge.target_domain != expected_domain:
        raise ValueError('missing or incompatible clock bridge')
    result = stamp.capture_time_ns + bridge.offset_ns
    if result < 0:
        raise ValueError('converted timestamp cannot be negative')
    return result
