from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/src/badminton_brain/types.py')
s = p.read_text()
if 'tracking_residual' in s:
    print('already patched'); raise SystemExit
old = '''    prediction_error: Any = None
    contact_detected: Any = None

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'Feedback')'''
new = '''    # Prediction residual: predicted shuttle position/velocity minus the measured one
    # (metres / metres per second).  Consumed by the adaptation layer.
    prediction_error: Any = None
    contact_detected: Any = None
    # Execution tracking residual (DEC-015): how far the executed command stayed from the
    # requested one.  Kept separate so the two meanings cannot be confused again.
    tracking_residual: Any = None

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'Feedback')
        if self.tracking_residual is not None:
            self.tracking_residual = check_per_env_scalar(
                'Feedback.tracking_residual', self.tracking_residual)'''
if old not in s:
    raise SystemExit('Feedback block not found verbatim')
p.write_text(s.replace(old, new))
import ast; ast.parse(p.read_text())
print('Feedback.tracking_residual added (DEC-015)')