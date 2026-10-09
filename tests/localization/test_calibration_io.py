import pytest
from hardware_localization.calibration_io import RigidTransform, convert_translation_mm_to_m, load_verified_transform
from hardware_localization.types import CalibrationRef

def test_units_mm_to_m():
    assert convert_translation_mm_to_m((1000,-200,50)) == (1,-.2,.05)
    with pytest.raises(ValueError):
        convert_translation_mm_to_m((1000,float('nan'),50))

def test_calibration_version_mismatch(tmp_path):
    import yaml
    cfg={'parent_frame':'chassis_base','child_frame':'d455_link','translation_m':[0,0,1],
         'quaternion_xyzw':[0,0,0,1],'source':'survey','calibration_id':'C1',
         'timestamp':'2026-10-09','verification_status':'VERIFIED','d455_serial':'123',
         'color_stream_profile':'1280x720','depth_stream_profile':'848x480'}
    p=tmp_path/'cal.yaml';p.write_text(yaml.safe_dump(cfg))
    expected=CalibrationRef('C1','123','1280x720','848x480')
    assert load_verified_transform(p,expected).parent_frame == 'chassis_base'
    for field,value in [('verification_status','UNCALIBRATED'),('d455_serial','wrong'),('color_stream_profile','wrong'),('calibration_id','old')]:
        changed=dict(cfg,**{field:value});p.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError): load_verified_transform(p,expected)
    with pytest.raises(ValueError): RigidTransform('a','b',(0,0,0),(0,0,0,2))
