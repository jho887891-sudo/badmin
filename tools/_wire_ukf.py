from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/src/badminton_brain/estimation/estimator.py')
s = p.read_text()
if 'ShuttleEstimatorBridge' in s:
    print('already wired'); raise SystemExit
old = '''    def __init__(self, num_envs: int = 1, *, localization: Optional[RobotLocalization] = None,
                 shuttle_filter: Any = None) -> None:
        self.num_envs = int(num_envs)
        self.localization = localization or RobotLocalization(num_envs=self.num_envs)
        self.shuttle_filter = shuttle_filter
        self._last_timestamp: Optional[float] = None
        if self.shuttle_filter is None:
            self.shuttle_source = "TEMP: measurement passthrough until the T3 UKF is injected"
        else:
            self.shuttle_source = type(self.shuttle_filter).__name__'''
new = '''    def __init__(self, num_envs: int = 1, *, localization: Optional[RobotLocalization] = None,
                 shuttle_filter: Any = None, use_ukf: bool = True) -> None:
        self.num_envs = int(num_envs)
        self.localization = localization or RobotLocalization(num_envs=self.num_envs)
        if shuttle_filter is None and use_ukf:
            try:
                from .shuttle_ukf import ShuttleEstimatorBridge
                shuttle_filter = ShuttleEstimatorBridge(num_envs=self.num_envs)
            except Exception:  # the UKF is optional; passthrough is labelled TEMP below
                shuttle_filter = None
        self.shuttle_filter = shuttle_filter
        self._last_timestamp: Optional[float] = None
        if self.shuttle_filter is None:
            self.shuttle_source = 'TEMP: measurement passthrough (T3 UKF not available)'
        else:
            self.shuttle_source = type(self.shuttle_filter).__name__'''
if old not in s:
    raise SystemExit('constructor block not found verbatim')
p.write_text(s.replace(old, new))
print('estimator now auto-injects the T3 shuttle bridge')