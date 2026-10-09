import numpy as np
import pytest
from hardware_localization.calibration_io import RigidTransform
from hardware_localization.fixed_camera_tf import invert_transform, camera_to_arm

def test_transform_chain_inverse():
    rot=RigidTransform('chassis','d455_link',(1,2,0),(0,0,2**-.5,2**-.5))
    eye=RigidTransform('d455_link','color_optical',(0,0,0),(0,0,0,1))
    arm=RigidTransform('chassis','arm_base',(0,0,0),(0,0,0,1))
    assert np.allclose(camera_to_arm((1,0,0),rot,eye,arm),(1,3,0),atol=1e-9)
    inv=invert_transform(rot)
    assert np.allclose(np.asarray(inv.translation_m),(-2,1,0),atol=1e-9)
    assert abs(abs(inv.quaternion_xyzw[2])-2**-.5)<1e-9

def test_optical_frame_convention():
    from scipy.spatial.transform import Rotation
    R=np.array([[0,0,1],[-1,0,0],[0,-1,0]],dtype=float)
    q=tuple(Rotation.from_matrix(R).as_quat())
    t=RigidTransform('chassis','d455_link',(0,0,0),(0,0,0,1))
    o=RigidTransform('d455_link','optical',(0,0,0),q)
    a=RigidTransform('chassis','arm',(0,0,0),(0,0,0,1))
    assert np.allclose(camera_to_arm((0,0,1),t,o,a),(1,0,0))
    assert np.allclose(camera_to_arm((1,0,0),t,o,a),(0,-1,0))

def test_wrong_frame_chain_rejected():
    t=RigidTransform('base','d455',(0,0,0),(0,0,0,1))
    wrong=RigidTransform('wrong','optical',(0,0,0),(0,0,0,1))
    arm=RigidTransform('base','arm',(0,0,0),(0,0,0,1))
    with pytest.raises(ValueError): camera_to_arm((0,0,1),t,wrong,arm)
