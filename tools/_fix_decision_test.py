from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/tests/badminton_brain/test_decision_module.py')
s = p.read_text()
old = """    def test_aggregate_is_conservative_when_any_env_is_rejected(self) -> None:
        mod = FeasibilityDecisionModule(num_envs=N)
        state = make_state()
        traj = make_trajectory()
        state.shuttle_velocity[1] = np.array([120.0, 0.0, 0.0])  # absurd speed for env 1
        decision, intercept = mod.process(state, traj)
        self.assertFalse(decision.feasible)
        self.assertIsNone(intercept)"""
new = """    def test_aggregate_is_conservative_when_any_env_is_rejected(self) -> None:
        # The gate judges the PREDICTED trajectory (its first sample speed / landing), not the
        # instantaneous shuttle_velocity, so env 1 is made infeasible by pushing its predicted
        # path far outside the playable volume.
        mod = FeasibilityDecisionModule(num_envs=N)
        state = make_state()
        traj = make_trajectory()
        traj.position[1, :, 0] += 60.0  # env 1 lands far outside the court
        traj.landing_point[1] = np.array([60.0, 0.0, 0.0])
        decision, intercept = mod.process(state, traj)
        self.assertFalse(decision.feasible)
        self.assertIsNone(intercept)
        self.assertFalse(bool(mod.last_decisions[1].feasible))
        self.assertTrue(bool(mod.last_decisions[0].feasible))"""
if old not in s:
    raise SystemExit('decision test block not found verbatim')
p.write_text(s.replace(old, new))
print('decision test premise fixed')
