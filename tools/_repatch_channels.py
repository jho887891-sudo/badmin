from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/tests/architecture/test_sensor_channels.py')
s = p.read_text()
old = """    def test_adapter_reset_only_touches_selected_envs(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        before = mod.state_snapshot()
        mod.reset([1])
        after = mod.state_snapshot()
        self.assertFalse(np.array_equal(before[1], after[1]))
        self.assertTrue(np.array_equal(before[0], after[0]))
        self.assertTrue(np.array_equal(before[2], after[2]))"""
new = """    def test_adapter_reset_only_touches_selected_envs(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        # Drive the estimator away from the home pose first: "reset to home" is otherwise
        # indistinguishable from "not touched" (the first version of this test asserted a
        # change a correct implementation does not have to make).
        moved = np.tile(np.array([-1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 1.0]), (N, 1))
        for step in range(3):
            t = 0.01 * (step + 1)
            sensors = RobotSensorState(base_pose=moved, joint_pos=np.zeros((N, 6)),
                                       joint_vel=np.zeros((N, 6)), timestamp=t,
                                       odom_twist=np.tile(np.array([0.1, 0.0, 0.0]), (N, 1)),
                                       imu_yaw_rate=np.zeros((N,)))
            perception = ShuttleMeasurement(position=np.zeros((N, 3)), velocity=np.zeros((N, 3)),
                                            covariance=np.zeros((N, 3, 3)), timestamp=t)
            mod.process(perception, sensors)
        perturbed = mod.state_snapshot()
        self.assertFalse(np.array_equal(perturbed[1], np.zeros_like(perturbed[1])),
                         'env 1 must have moved away from its initial state')
        mod.reset([1])
        after = mod.state_snapshot()
        self.assertFalse(np.array_equal(perturbed[1], after[1]), 'selected env must be reset')
        self.assertTrue(np.array_equal(perturbed[0], after[0]), 'env 0 must be untouched')
        self.assertTrue(np.array_equal(perturbed[2], after[2]), 'env 2 must be untouched')"""
if old not in s:
    raise SystemExit('old test body not found')
p.write_text(s.replace(old, new))
print('remote test re-patched')
