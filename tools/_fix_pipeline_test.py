from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/tests/architecture/test_brain_pipeline.py')
s = p.read_text()
old = '''        return Feedback(timestamp=command.timestamp, prediction_error=np.zeros((N,)),
                        contact_detected=np.zeros((N,), dtype=bool))'''
new = '''        # DEC-015: execution reports its tracking residual; the prediction residual belongs to
        # the estimation/prediction side and must not be fabricated here.
        return Feedback(timestamp=command.timestamp, prediction_error=None,
                        tracking_residual=np.zeros((N,)),
                        contact_detected=np.zeros((N,), dtype=bool))'''
if old not in s:
    raise SystemExit('fake execution Feedback block not found')
p.write_text(s.replace(old, new))
import ast; ast.parse(p.read_text())
print('contract test fake updated to DEC-015 semantics')