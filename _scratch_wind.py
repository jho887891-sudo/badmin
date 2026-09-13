import sys
sys.path.insert(0, '/home/T7/ojh/robot_sim/src')
import numpy as np
from badminton_brain.estimation.shuttle_ukf import ShuttleUKF
from badminton_brain.types import ShuttleMeasurement
from trajectory.shuttle_aerodynamics import rollout

K_TRUE = 1/6.5; DT = 0.02; SIG = 0.01
W = np.array([0.0, 1.5, 0.0])
def make(p0, v0, wind, seed, duration, dt_meas=DT, sigma=SIG):
    n = int(round(duration/dt_meas))
    tr = rollout(p0, v0, duration_s=duration, dt_s=5e-4, k_per_m=K_TRUE, gravity=(0,0,-9.80665), wind=wind)
    t = dt_meas*np.arange(n+1)
    p = np.stack([np.interp(t, tr['time'], tr['position'][:, i]) for i in range(3)], axis=1)
    v = np.stack([np.interp(t, tr['time'], tr['velocity'][:, i]) for i in range(3)], axis=1)
    rng = np.random.default_rng(seed)
    return t, p, v, p + sigma*rng.standard_normal(p.shape), v + sigma*rng.standard_normal(v.shape)

for label, p0, v0, duration in (('canonical', np.array([1.0,0.5,1.5]), np.array([-8.0,-1.0,4.0]), 1.0),
                                ('high clear', np.array([2.0,0.5,2.0]), np.array([-9.0,-1.0,6.5]), 1.6)):
    t, p, v, mp, mv = make(p0, v0, W, 5, duration)
    print('--- %s  z-range %.2f..%.2f, |v| %.1f..%.1f' % (label, p[:,2].min(), p[:,2].max(),
          np.linalg.norm(v[0]), np.linalg.norm(v[-1])))
    for wp in (0.3, 1.0):
        ukf = ShuttleUKF(num_envs=1, enable_wind=True, wind_process_std=wp, initial_wind_std=2.0)
        ukf.initialize(position=mp[0][None,:], velocity=(mp[1]-mp[0])[None,:]/DT)
        marks = []
        for k in range(1, len(t)):
            ukf.predict(DT)
            ukf.update(ShuttleMeasurement(position=mp[k][None,:], velocity=mv[k][None,:],
                                          covariance=np.eye(3)[None,:,:]*SIG**2, timestamp=float(t[k])))
            if k % max(1, len(t)//6) == 0:
                marks.append('t=%.2f w=%s k=%.3f' % (t[k], np.array2string(ukf.wind[0], precision=2), ukf.drag_k[0]))
        print('  wp=%.1f  %s' % (wp, ' | '.join(marks)))
        print('        final |werr|=%.3f  |verr|=%.3f' % (np.linalg.norm(ukf.wind[0]-W),
              np.linalg.norm(ukf.velocity[0]-v[-1])))
