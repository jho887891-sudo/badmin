import pytest
from hardware_localization.types import CaptureStamp, CalibrationRef, BallObservation3D
from hardware_localization.timebase import ClockBridge, convert_capture_time

def test_frame_seq_and_coordinate_frame_are_distinct():
    stamp = CaptureStamp(42, 2000, 'REALSENSE_HW', 5)
    obs = BallObservation3D(stamp, 'd455_color_optical_frame', None, None, 'MISSING', 'D455', CalibrationRef('cal1','123','color1280x720','depth848x480'))
    assert stamp.frame_seq == 42
    assert obs.coordinate_frame == 'd455_color_optical_frame'
    with pytest.raises((TypeError, ValueError)):
        CaptureStamp('d455_color_optical_frame', 2000, 'REALSENSE_HW', 5)
    with pytest.raises((TypeError, ValueError)):
        BallObservation3D(stamp, 42, None, None, 'MISSING', 'D455', obs.calibration)

def test_timestamp_clock_domain_conversion():
    stamp=CaptureStamp(42,2000,'REALSENSE_HW',5)
    with pytest.raises(ValueError):
        convert_capture_time(stamp,None,'ROS_TIME')
    bridge=ClockBridge('REALSENSE_HW','ROS_TIME',1000,10)
    assert convert_capture_time(stamp,bridge,'ROS_TIME') == 3000
    assert stamp.sync_uncertainty_ns + bridge.uncertainty_ns == 15
    with pytest.raises(ValueError):
        convert_capture_time(stamp,ClockBridge('HOST_MONOTONIC','ROS_TIME',1000,10),'ROS_TIME')
    with pytest.raises(ValueError):
        CaptureStamp(0,-1,'REALSENSE_HW',0)
    with pytest.raises(ValueError):
        CaptureStamp(0,1,'REALSENSE_HW',-1)
