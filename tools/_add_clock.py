from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/src/badminton_brain/decision/decision_module.py')
s = p.read_text()
if 'now_provider' in s:
    print('already patched'); raise SystemExit
s = s.replace('''    def __init__(self, *, limits: Optional[FeasibilityLimits] = None,
                 search_config: Optional[InterceptSearchConfig] = None, num_envs: int = 1) -> None:
        self.num_envs = int(num_envs)''',
'''    def __init__(self, *, limits: Optional[FeasibilityLimits] = None,
                 search_config: Optional[InterceptSearchConfig] = None, num_envs: int = 1,
                 now_provider: Optional[Callable[[], float]] = None) -> None:
        self.num_envs = int(num_envs)
        self.now_provider = now_provider
        self._pushed_now: Optional[float] = None
        self.last_now: Optional[float] = None''')
s = s.replace('''from typing import Any, Optional, Sequence, Tuple''',
              '''from typing import Any, Callable, Optional, Sequence, Tuple''')
s = s.replace('''        decisions = tuple(self.gate.evaluate_batch(state, trajectory))''',
'''        now = self._resolve_now(state, trajectory)
        self.last_now = now
        decisions = tuple(self.gate.evaluate_batch(state, trajectory, now=now))''')
s = s.replace('''    @staticmethod
    def _reason(''',
'''    def set_now(self, now: float) -> None:
        """Push the runtime clock (T5 review C2): without it the gate cannot see a stale state."""
        self._pushed_now = float(now)

    def _resolve_now(self, state: UnifiedState, trajectory: PredictedTrajectory) -> float:
        """Runtime clock if available, else the newest message timestamp (documented fallback)."""
        if self.now_provider is not None:
            return float(self.now_provider())
        if self._pushed_now is not None:
            return float(self._pushed_now)
        return max(float(state.timestamp), float(trajectory.timestamp))

    @staticmethod
    def _reason(''')
p.write_text(s)
import ast; ast.parse(s)
print('decision module now accepts a runtime clock')