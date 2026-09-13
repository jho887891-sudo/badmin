from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/src/badminton_brain/apps/full_brain.py')
s = p.read_text()
if 'truth_provider' in s:
    print('already patched'); raise SystemExit
s = s.replace(
    "def _instantiate(cls, num_envs: int):\n    last_error: Optional[Exception] = None\n    for kwargs in ({'num_envs': num_envs}, {}, {'n': num_envs}):",
    "def _instantiate(cls, num_envs: int, truth_provider=None):\n    last_error: Optional[Exception] = None\n    attempts = [{'num_envs': num_envs, 'truth_provider': truth_provider},\n                {'num_envs': num_envs}, {}, {'n': num_envs}] if truth_provider is not None else [\n                {'num_envs': num_envs}, {}, {'n': num_envs}]\n    for kwargs in attempts:")
s = s.replace(
    "def build_full_brain(*, num_envs: int = 1, config: Optional[PipelineConfig] = None):",
    "def build_full_brain(*, num_envs: int = 1, config: Optional[PipelineConfig] = None,\n                     truth_provider=None):")
s = s.replace("                chosen = _instantiate(candidates[0], num_envs)",
              "                chosen = _instantiate(candidates[0], num_envs, truth_provider)")
p.write_text(s)
import ast; ast.parse(s)
print('full_brain accepts a scenario truth provider')
