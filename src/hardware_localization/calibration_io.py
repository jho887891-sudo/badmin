"""Strict versioned calibration loader. Never synthesizes real extrinsics."""
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
import yaml
from .types import CalibrationRef

@dataclass(frozen=True)
class RigidTransform:
    parent_frame: str
    child_frame: str
    translation_m: tuple[float,float,float]
    quaternion_xyzw: tuple[float,float,float,float]

    def __post_init__(self):
        if not self.parent_frame or not self.child_frame or self.parent_frame == self.child_frame:
            raise ValueError('invalid frame relation')
        if len(self.translation_m)!=3 or not all(isfinite(v) for v in self.translation_m):
            raise ValueError('invalid translation')
        q=self.quaternion_xyzw
        if len(q)!=4 or not all(isfinite(v) for v in q) or abs(sqrt(sum(v*v for v in q))-1)>1e-6:
            raise ValueError('quaternion must be normalized')

def convert_translation_mm_to_m(xyz_mm):
    if len(xyz_mm)!=3 or not all(isinstance(v,(float,int)) and isfinite(v) for v in xyz_mm):
        raise ValueError('finite xyz millimeters required')
    return tuple(float(v)/1000 for v in xyz_mm)

def load_verified_transform(path: str | Path, expected_calibration: CalibrationRef) -> RigidTransform:
    d=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(d,dict): raise ValueError('calibration must be a mapping')
    fields=('parent_frame','child_frame','translation_m','quaternion_xyzw','source','calibration_id',
            'timestamp','verification_status','d455_serial','color_stream_profile','depth_stream_profile')
    if any(k not in d for k in fields): raise ValueError('calibration fields missing')
    if d['verification_status']!='VERIFIED' or not d['source'] or not d['timestamp']:
        raise ValueError('not a verified measurement')
    for k,wanted in [('calibration_id',expected_calibration.calibration_id),('d455_serial',expected_calibration.d455_serial),
                     ('color_stream_profile',expected_calibration.color_stream_profile),
                     ('depth_stream_profile',expected_calibration.depth_stream_profile)]:
        if d[k]!=wanted: raise ValueError(f'calibration mismatch: {k}')
    if not all(isinstance(x,(int,float)) for x in d['translation_m']): raise ValueError('translation must be meters')
    return RigidTransform(d['parent_frame'],d['child_frame'],tuple(d['translation_m']),tuple(d['quaternion_xyzw']))
