from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/tests/badminton_brain/test_full_brain.py')
s = p.read_text()

# 1) add an always-moving stub planner so the e-stop test has a non-zero baseline
if 'AlwaysMovePlanner' not in s:
    helper = '''

class AlwaysMovePlanner(PlanningModule):
    """Stub planner that always commands motion (the canonical scenario may command none),
    so an e-stop comparison has a non-zero baseline to differ from."""

    name = 'always_move'
    is_implemented = True

    def process(self, state, decision, intercept, trajectory):
        return WholeBodyTarget(base_twist=np.full((N, 3), 0.4),
                               joint_position_target=np.zeros((N, 6)),
                               horizon_s=0.2, timestamp=state.timestamp)

'''
    anchor = 'class FullBrainIntegrationTests(unittest.TestCase):'
    assert anchor in s
    s = s.replace(anchor, helper + anchor)
    s = s.replace('from badminton_brain.apps.full_brain import build_full_brain  # noqa: E402',
                  'from badminton_brain.apps.full_brain import build_full_brain  # noqa: E402' + chr(10) +
                  'from badminton_brain.interfaces import PlanningModule  # noqa: E402' + chr(10) +
                  'from badminton_brain.types import WholeBodyTarget  # noqa: E402')

# 2) strengthen the e-stop test with the stub planner
start = s.index('    def test_estop_context_reaches_the_command_end_to_end(self) -> None:')
end = s.index('    def test_context_free_run_is_flagged_and_not_silently_normal(self) -> None:')
new_test = '''    def test_estop_context_reaches_the_command_end_to_end(self) -> None:
        from badminton_brain.safety.safety_shield import SafetyContext
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        registry.replace(AlwaysMovePlanner())   # guarantee a non-zero baseline command
        reference = pipeline.step(sensors(0.0))
        reference_twist = np.asarray(reference.safe_command.base_twist)
        assert np.max(np.abs(reference_twist)) > 0.0, 'the baseline must actually command motion'

        registry_b, pipeline_b = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        registry_b.replace(AlwaysMovePlanner())
        safety = registry_b.get(Layer.SAFETY)
        safety.set_context(SafetyContext(now=0.0, estop=np.array([True, False])))
        result = pipeline_b.step(sensors(0.0))
        twist = np.asarray(result.safe_command.base_twist)
        np.testing.assert_allclose(twist[0], np.zeros(3), atol=1e-12, err_msg='e-stopped env must not move')
        np.testing.assert_allclose(twist[1], reference_twist[1], atol=1e-12,
                                   err_msg='the non-e-stopped env must keep its command')
        self.assertTrue(result.safe_command.limited, 'an e-stop must be reported in the command')

'''
s = s[:start] + new_test + s[end:]

# 3) give the reset test a real assertion (broad review D8)
start_r = s.index('    def test_reset_is_propagated_to_every_layer(self) -> None:')
end_r = s.index("\n\nif __name__ == '__main__':")
new_reset = '''    def test_reset_is_propagated_to_every_layer(self) -> None:
        registry, pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        seen = []
        for module in registry.modules():
            original = module.reset

            def spy(env_ids, _module=module, _original=original):
                seen.append((_module.layer.value, list(env_ids)))
                return _original(env_ids)

            module.reset = spy
        pipeline.reset([1])
        self.assertEqual(len(seen), len(registry.modules()),
                         'reset must reach every registered layer')
        self.assertEqual(sorted(layer for layer, _ in seen),
                         sorted(m.layer.value for m in registry.modules()))
        self.assertTrue(all(ids == [1] for _, ids in seen),
                        'every layer must receive exactly the requested env ids')
'''
s = s[:start_r] + new_reset + s[end_r:]
p.write_text(s)
import ast; ast.parse(s)
print('T11 tests strengthened (D3 differential e-stop, D8 assertions)')
