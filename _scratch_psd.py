import sys
sys.path.insert(0, '/home/T7/ojh/robot_sim/src')
import numpy as np
from badminton_brain.estimation.shuttle_ukf import ShuttleUKF
from badminton_brain.types import ShuttleMeasurement
from trajectory.shuttle_aerodynamics import rollout

K_TRUE = 1/6.5; DT = 0.02; SIG = 0.01
P0 = np.array([1.0, 0.5, 1.5]); V0 = np.array([-8.0, -1.0, 4.0])
def synthetic_flight(sigma=SIG, seed=7, duration=1.0, dt_meas=DT):
    n_steps = int(round(duration/dt_meas))
    truth = rollout(P0, V0, duration_s=duration, dt_s=5e-4, k_per_m=K_TRUE,
                    gravity=(0.0,0.0,-9.80665), wind=np.zeros(3))
    times = dt_meas*np.arange(n_steps+1)
    pos_true = np.stack([np.interp(times, truth['time'], truth['position'][:, i]) for i in range(3)], axis=1)
    vel_true = np.stack([np.interp(times, truth['time'], truth['velocity'][:, i]) for i in range(3)], axis=1)
    rng = np.random.default_rng(seed)
    noise = sigma*rng.standard_normal(pos_true.shape)
    vel_noise = sigma*rng.standard_normal(vel_true.shape)
    return times, pos_true, vel_true, pos_true+noise, vel_true+vel_noise

times, pos_true, vel_true, mp, mv = synthetic_flight()
ukf = ShuttleUKF(num_envs=1, initial_drag_std=0.5)
ukf.initialize(position=mp[0][None,:], velocity=np.zeros((1,3)))
worst = 1e9
for k in range(1, len(times)):
    ukf.predict(5*DT if k % 5 == 0 else DT)
    if k % 5 == 0:
        ukf.update(ShuttleMeasurement(position=mp[k][None,:], velocity=mv[k][None,:],
                                      covariance=np.eye(3)[None,:,:]*SIG**2, timestamp=float(times[k])))
    cov = ukf.covariance[0]; eig = np.linalg.eigvalsh(0.5*(cov+cov.T))
    if eig.min() < worst:
        worst = eig.min()
        print('worst so far k=%2d eigmin=%+.6e eigmax=%.3e k_hat=%.4f' % (k, eig.min(), eig.max(), ukf.drag_k[0]))
    if eig.min() < -1e-9:
        print('NEGATIVE at k=%d' % k); print(np.array2string(cov, precision=3, suppress_small=True)); break
print('final k_hat', ukf.drag_k[0])
