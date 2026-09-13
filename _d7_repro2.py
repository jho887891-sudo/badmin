import sys, numpy as np
sys.path.insert(0, '/home/T7/ojh/robot_sim/src')
from badminton_brain.apps.full_brain import build_full_runtime
from badminton_brain.types import Layer, RobotSensorState

N = 2
def canonical_truth(n, t):
    return np.stack([np.array([5.20,0.30,2.10]) + np.array([-8.0,-0.10,1.20])*t for _ in range(n)])
def sensors(t):
    return RobotSensorState(base_pose=np.tile(np.array([-1.6,0.0,0.0,0.0,0.0,0.0,1.0]),(N,1)),
                            joint_pos=np.zeros((N,6)), joint_vel=np.zeros((N,6)), timestamp=t,
                            odom_twist=np.zeros((N,3)), imu_yaw_rate=np.zeros((N,)))
runtime = build_full_runtime(num_envs=N, truth_provider=canonical_truth)
a = runtime.registry.get(Layer.ADAPTATION)
norms = []
for step in range(40):
    runtime.step(sensors(0.05*step))
print('updates           :', a.updates)
print('residual_counts   :', a.residual_counts())
print('residual per step :', np.round(a.residual_norms()[0], 5))
print('residual_mean     :', np.round(a.residual_mean(), 6))
print('estimates drag    :', np.round(a.drag_scale, 6))
print('estimates wind    :', np.round(a.wind[0], 5))
print('estimates delay   :', np.round(a.delay_s, 6))
print('clamped/limited   :', a.clamped, a.limited)
print('correction() residual (read-only query):', a.correction()['residual'])
print('correction() keys :', sorted(a.correction()))
