import sys, numpy as np
sys.path.insert(0, '/home/T7/ojh/robot_sim/src'); sys.path.insert(0, '/home/T7/ojh/robot_sim/src/trajectory')
from badminton_brain.apps.full_brain import build_full_runtime
from badminton_brain.types import Layer, RobotSensorState
from shuttle_aerodynamics import k_from_aerodynamic_length

N = 2; DT = 0.05
START = np.array([5.20, 0.30, 2.10]); VEL = np.array([-8.0, -0.10, 1.20])
def canonical_truth(n, t): return np.stack([START + VEL*t for _ in range(n)])
def sensors(t):
    return RobotSensorState(base_pose=np.tile(np.array([-1.6,0.0,0.0,0.0,0.0,0.0,1.0]),(N,1)),
                            joint_pos=np.zeros((N,6)), joint_vel=np.zeros((N,6)), timestamp=t,
                            odom_twist=np.zeros((N,3)), imu_yaw_rate=np.zeros((N,)))
runtime = build_full_runtime(num_envs=N, truth_provider=canonical_truth)
a = runtime.registry.get(Layer.ADAPTATION)
for step in range(8):
    runtime.step(sensors(DT*step))
led = a.residual_ledger()[0]
print('residual vectors (last 5):'); print(np.round(led[-5:], 6))
print('|residual| =', round(float(np.linalg.norm(led[-1])), 6))
print('residual vs straight-line truth: expected gravity term over dt =',
      round(0.5*9.80665*DT*DT, 6), 'm in -z')
print('truth at t=0.35 :', np.round(START + VEL*0.35, 4))
print('=> the canonical truth_provider is a STRAIGHT LINE, the predictor integrates gravity+drag')
print('estimates after 8 steps: drag', np.round(a.drag_scale,5), 'wind', np.round(a.wind[0],4),
      'delay', np.round(a.delay_s,6))
# same runtime, but a PHYSICS-CONSISTENT truth (real RK4 flight instead of a straight line)
sys.path.insert(0, '/home/T7/ojh/robot_sim/src/trajectory')
from shuttle_aerodynamics import rk4_step
K = k_from_aerodynamic_length(6.5)
grid = np.arange(0.0, 5.0, 0.001)
p = START.copy(); v = VEL*0.35
traj = [p.copy()]
for _ in range(len(grid)-1):
    p, v = rk4_step(p, v, 0.001, k_per_m=K)
    traj.append(p.copy())
traj = np.asarray(traj)
def rk4_truth(n, t):
    pos = np.array([np.interp(t, grid, traj[:, i]) for i in range(3)])
    return np.stack([pos for _ in range(n)])
runtime2 = build_full_runtime(num_envs=N, truth_provider=rk4_truth)
a2 = runtime2.registry.get(Layer.ADAPTATION)
for step in range(40):
    runtime2.step(sensors(DT*step))
print()
print('PHYSICS-CONSISTENT truth: updates', a2.updates, 'residual_counts', a2.residual_counts(),
      'residual_mean', np.round(a2.residual_mean(), 6))
print('  residual vectors (last 3):', np.round(a2.residual_ledger()[0][-3:], 6))
print('  estimates: drag', np.round(a2.drag_scale,5), 'wind', np.round(a2.wind[0],4),
      'delay', np.round(a2.delay_s,6), 'clamped', a2.clamped)
