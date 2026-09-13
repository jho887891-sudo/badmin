import sys, numpy as np
sys.path.insert(0, '/home/T7/ojh/robot_sim/src')
from badminton_brain.apps.full_brain import build_full_runtime
from badminton_brain.types import Layer, RobotSensorState

N = 2
def canonical_truth(num_envs, timestamp):
    start = np.array([5.20, 0.30, 2.10]); velocity = np.array([-8.0, -0.10, 1.20])
    return np.stack([start + velocity * timestamp for _ in range(num_envs)])
def sensors(t):
    return RobotSensorState(base_pose=np.tile(np.array([-1.6,0.0,0.0,0.0,0.0,0.0,1.0]),(N,1)),
                            joint_pos=np.zeros((N,6)), joint_vel=np.zeros((N,6)), timestamp=t,
                            odom_twist=np.zeros((N,3)), imu_yaw_rate=np.zeros((N,)))

runtime = build_full_runtime(num_envs=N, truth_provider=canonical_truth)
adaptation = runtime.registry.get(Layer.ADAPTATION)
print('adaptation module:', type(adaptation).__name__)
print('%4s %8s %10s %10s %12s %12s %12s %10s %10s %18s' % (
      'step','sensor_t','state_t','traj_t','times[0]','times[-1]','feedback_pe','dt','source','residual'))
for step in range(6):
    t = 0.05*step
    result = runtime.step(sensors(t))
    st = result.unified_state; traj = result.trajectory; fb = result.feedback
    d = adaptation.diagnostics()
    last = adaptation._last_source if adaptation._last_source else ['-']
    pe = None if fb is None else fb.prediction_error
    print('%4d %8.3f %10.3f %10.3f %12.3f %12.3f %12s %10s %10s %18s' % (
        step, t, float(st.timestamp), float(traj.timestamp), float(np.asarray(traj.times)[0]),
        float(np.asarray(traj.times)[-1]),
        'None' if pe is None else str(np.asarray(pe).ravel()[:2]),
        np.round(d['last_dt_s'],4), last[:1], np.round(adaptation.correction()['residual'],5)))
print()
print('prediction_available:', d['prediction_available'], 'slots:', d['prediction_slots'])
print('updates:', adaptation.updates, 'residual_counts:', adaptation.residual_counts())
print('shuttle state position[0]:', np.round(np.asarray(st.shuttle_position)[0],4))
print('shuttle state velocity[0]:', np.round(np.asarray(st.shuttle_velocity)[0],4))
