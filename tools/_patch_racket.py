from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/tools/build_racket.py')
s = p.read_text()
# ---- A) orient 精度兼容 ----
helper = (
    'def _set_orient(op, quat) -> None:\n'
    '    """Set an orient xformOp with the precision the stage actually created.\n'
    '\n'
    '    USD 25.11 creates freshly added xformOp:orient attributes as Quatf while\n'
    '    Gf.Quatd is the double-precision variant; writing the wrong type raises\n'
    '    Tf.ErrorException. Match the attribute type instead of guessing.\n'
    '    """\n'
    '    from pxr import Gf, Sdf\n'
    '\n'
    '    attr = op.GetAttr()\n'
    '    if attr.GetTypeName() == Sdf.ValueTypeNames.Quatf:\n'
    '        op.Set(Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(*[float(v) for v in quat.GetImaginary()])))\n'
    '    else:\n'
    '        op.Set(quat)\n'
    '\n'
    '\n'
)
if '_set_orient' not in s:
    s = s.replace('def _quat_from_z_to_direction(direction: Vec3):', helper + 'def _quat_from_z_to_direction(direction: Vec3):', 1)
    s = s.replace('        cyl.AddOrientOp().Set(_quat_from_z_to_direction(direction))',
                  '        _set_orient(cyl.AddOrientOp(), _quat_from_z_to_direction(direction))')
# ---- B) customData 类型兼容 ----
s = s.replace('contact.GetPrim().SetCustomDataByKey("face_normal_local", [1.0, 0.0, 0.0])',
              'contact.GetPrim().SetCustomDataByKey("face_normal_local", Vt.FloatArray([1.0, 0.0, 0.0]))')
import re
m = re.search(r'^from pxr import ([^\n]+)$', s, re.M)
if m and 'Vt' not in m.group(1):
    s = s[:m.start()] + 'from pxr import ' + m.group(1).strip() + ', Vt' + s[m.end():]
p.write_text(s)
import ast; ast.parse(s)
print('OK: syntax valid')
print('helper present :', '_set_orient' in s)
print('orient call    :', '_set_orient(cyl.AddOrientOp()' in s)
print('customData Vt  :', 'Vt.FloatArray([1.0, 0.0, 0.0])' in s)