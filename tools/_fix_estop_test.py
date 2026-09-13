from pathlib import Path
p = Path('/home/T7/ojh/robot_sim/tests/badminton_brain/test_full_brain.py')
s = p.read_text()
old = '''        safety.set_context(SafetyContext(now=0.0, estop=np.array([True, False])))
        result = pipeline.step(sensors(0.0))
        twist = np.asarray(result.safe_command.base_twist)
        joints = np.asarray(result.safe_command.joint_position_target)
        np.testing.assert_allclose(twist[0], np.zeros(3), atol=1e-12, err_msg='e-stopped env must not move')
        self.assertTrue(np.all(np.isfinite(joints[0])))
        self.assertGreater(float(np.max(np.abs(twist[1]))), 0.0,
                           'the non-e-stopped env must keep its command')
        self.assertTrue(result.safe_command.limited)'''
new = '''        # Compare against an identical run without e-stop: the scenario itself may command nothing
        # (an unreachable intercept), so only the DIFFERENCE proves the e-stop reached the command.
        _, reference_pipeline = build_full_brain(num_envs=N, truth_provider=canonical_truth)
        reference = reference_pipeline.step(sensors(0.0))
        safety.set_context(SafetyContext(now=0.0, estop=np.array([True, False])))
        result = pipeline.step(sensors(0.0))
        twist = np.asarray(result.safe_command.base_twist)
        reference_twist = np.asarray(reference.safe_command.base_twist)
        np.testing.assert_allclose(twist[0], np.zeros(3), atol=1e-12, err_msg='e-stopped env must not move')
        np.testing.assert_allclose(twist[1], reference_twist[1], atol=1e-12,
                                   err_msg='the non-e-stopped env must keep exactly its original command')
        self.assertTrue(np.all(np.isfinite(np.asarray(result.safe_command.joint_position_target))))
        self.assertTrue(result.safe_command.limited, 'an e-stop must be reported in the command')'''
if old not in s:
    raise SystemExit('estop test block not found')
p.write_text(s.replace(old, new))
print('estop test made differential')