import sys, numpy as np
sys.path.insert(0, '/home/T7/ojh/robot_sim/src'); sys.path.insert(0, '/home/T7/ojh/robot_sim/src/trajectory')
sys.path.insert(0, '/home/T7/ojh/robot_sim/tests/badminton_brain')
from badminton_brain.adaptation.online_adaptation import OnlineAdaptation
from badminton_brain.types import Feedback
import test_online_adaptation as T

POS = np.array([[0.0,0.0,3.0]]); VEL = lambda s: T.orbit_velocity(s)

for theta in (0.5, 1.0, 3.0):
    m = OnlineAdaptation(num_envs=1, estimate=('drag_scale',), gain_drag_scale=0.4,
                         max_step_drag_scale=5.0, eps=1e-30)
    m.drag_scale[0] = theta
    v = VEL(0); jac = -0.5*T.DT*T.DT*T.K_BASE*float(np.linalg.norm(v))*v[0]
    e = (jac*(theta-1.0))[None,:]
    m.process(Feedback(prediction_error=e, timestamp=0.0), T.make_state(POS, v, 0.0))
    m.process(Feedback(prediction_error=e, timestamp=T.DT), T.make_state(POS, v, T.DT))
    print('gain theta=%.1f -> %.12f | exact %.12f | gain/theta-bug would give %.12f'
          % (theta, m.drag_scale[0], theta-0.4*(theta-1.0), theta-0.4*(theta-1.0)/theta))

m = OnlineAdaptation(num_envs=1, estimate=('drag_scale',)); errs = []
for s in range(60):
    v = VEL(s); t0 = s*T.DT
    e = T.position_residual(POS, v, pred_k=T.K_BASE*float(m.drag_scale[0]), pred_wind=np.zeros(3),
                            true_k=T.K_BASE*1.25, true_wind=np.zeros(3))
    errs.append(abs(float(m.drag_scale[0])-1.25))
    m.process(Feedback(prediction_error=e, timestamp=t0), T.make_state(POS, v, t0))
print('drag: err[1]=%.3e err[20]=%.3e err[59]=%.3e rates=%s (expect 0.6)'
      % (errs[1], errs[20], errs[59], np.round([errs[i+1]/errs[i] for i in range(1,8)], 4)))

m = OnlineAdaptation(num_envs=1, estimate=('delay',))
meas = T.rk4_position(POS, VEL(0), T.DT); derr = []
for s in range(60):
    t0 = s*T.DT; a = float(m.delay_s[0])
    pred = T.rk4_position(POS, VEL(0), T.DT - 0.02 + a)
    derr.append(abs(a-0.02))
    m.process(Feedback(prediction_error=pred-meas, timestamp=t0), T.make_state(POS, VEL(0), t0))
print('delay(RK4): err[1]=%.3e err[20]=%.3e err[59]=%.3e final=%.12f'
      % (derr[1], derr[20], derr[59], m.delay_s[0]))

m = OnlineAdaptation(num_envs=1, estimate=('wind',)); tw = np.array([0.6,-0.4,0.0]); werr = []
for s in range(60):
    v = VEL(s); t0 = s*T.DT
    e = T.position_residual(POS, v, pred_k=T.K_BASE, pred_wind=m.wind[0], true_k=T.K_BASE, true_wind=tw)
    werr.append(float(np.abs(m.wind[0]-tw).max()))
    m.process(Feedback(prediction_error=e, timestamp=t0), T.make_state(POS, v, t0))
print('wind: err[1]=%.3e err[19]=%.3e err[59]=%.3e final=%s'
      % (werr[1], werr[19], werr[59], np.round(m.wind[0], 9)))

m = OnlineAdaptation(num_envs=1, estimate=('drag_scale','wind','delay')); e = np.array([[5.0,5.0,5.0]])
m.process(Feedback(prediction_error=e, timestamp=0.0), T.make_state(POS, VEL(0), 0.0))
out = m.process(Feedback(prediction_error=e, timestamp=0.05), T.make_state(POS, VEL(0), 0.05))
print('5 m sample: drag=%.6f wind=%s delay=%.6f limited=%s clamped=%s'
      % (m.drag_scale[0], np.round(m.wind[0],6), m.delay_s[0], out['limited'], out['clamped']))
print('measurement_requirements: %d entries, sample=%s'
      % (len(m.measurement_requirements()), sorted(m.measurement_requirements())[:3]))
