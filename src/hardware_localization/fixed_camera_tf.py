"""Strict SE(3) transforms for a rigid D455 on the stationary chassis."""
import numpy as np
from .calibration_io import RigidTransform

def _matrix(t: RigidTransform) -> np.ndarray:
    x,y,z,w=t.quaternion_xyzw
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=float)
    out=np.eye(4);out[:3,:3]=R;out[:3,3]=t.translation_m
    return out

def _from_matrix(parent: str,child: str,m: np.ndarray) -> RigidTransform:
    from scipy.spatial.transform import Rotation
    return RigidTransform(parent,child,tuple(float(x) for x in m[:3,3]),
                          tuple(float(x) for x in Rotation.from_matrix(m[:3,:3]).as_quat()))

def invert_transform(T: RigidTransform) -> RigidTransform:
    m=_matrix(T);inv=np.eye(4);inv[:3,:3]=m[:3,:3].T;inv[:3,3]=-m[:3,:3].T @ m[:3,3]
    return _from_matrix(T.child_frame,T.parent_frame,inv)

def compose_transform(T_X_Y: RigidTransform,T_Y_Z: RigidTransform) -> RigidTransform:
    if T_X_Y.child_frame != T_Y_Z.parent_frame:
        raise ValueError('inconsistent transform frame chain')
    if T_X_Y.parent_frame==T_Y_Z.child_frame:
        raise ValueError('identity result should not be stored as a named rigid transform')
    return _from_matrix(T_X_Y.parent_frame,T_Y_Z.child_frame,_matrix(T_X_Y)@_matrix(T_Y_Z))

def camera_to_arm(point_c_xyz_m,T_B_D: RigidTransform,T_D_C: RigidTransform,T_B_A: RigidTransform)->tuple[float,float,float]:
    if T_B_D.child_frame != T_D_C.parent_frame or T_B_D.parent_frame != T_B_A.parent_frame:
        raise ValueError('camera/arm are not rooted in same chassis frame')
    p=np.asarray(point_c_xyz_m,dtype=float)
    if p.shape!=(3,) or not np.all(np.isfinite(p)):
        raise ValueError('point must be finite 3D vector')
    T=_matrix(invert_transform(T_B_A))@_matrix(T_B_D)@_matrix(T_D_C)
    return tuple(float(v) for v in (T[:3,:3]@p+T[:3,3]))
