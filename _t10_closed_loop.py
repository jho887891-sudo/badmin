import sys, numpy as np
sys.path.insert(0, '/home/T7/ojh/robot_sim/src'); sys.path.insert(0, '/home/T7/ojh/robot_sim/src/trajectory')
from shuttle_aerodynamics import rk4_step, k_from_aerodynamic_length
from badminton_brain.adaptation.online_adaptation import OnlineAdaptation
from badminton_brain.prediction.physics_predictor import PhysicsTrajectoryPredictor
from badminton_brain.types import Feedback, UnifiedState

K_BASE = k_from_aerodynamic_length(6.5); TRUE_SCALE = 1.25; DT = 0.05
predictor = PhysicsTrajectoryPredictor()
module = OnlineAdaptation(num_envs=1, estimate=('drag_scale',))

p = np.array([0.0, 0.0, 3.0]); v = np.array([4.0, 0.5, -0.5]); t = 0.0
def state(t, p, v):
    return UnifiedState(base_pose=np.tile([0,0,0,0,0,0,1.0],(1,1)), base_twist=np.zeros((1,6)),
                        joint_pos=np.zeros((1,6)), joint_vel=np.zeros((1,6)),
                        racket_contact_pose=np.tile([0,0,0,0,0,0,1.0],(1,1)),
                        racket_contact_twist=np.zeros((1,6)),
                        shuttle_position=p[None,:], shuttle_velocity=v[None,:], timestamp=t)
for step in range(80):
    out = module.process(Feedback(prediction_error=None, timestamp=t), state(t, p, v))
    if step in (1, 5, 20, 40, 79):
        print('step %2d  source=%-10s residual=%.6f m  drag_scale=%.9f  (truth %.2f)'
              % (step, out['residual_source'][0], out['residual'][0], module.drag_scale[0], TRUE_SCALE))
    traj = predictor.predict(p, v, k_per_m=K_BASE*float(module.drag_scale[0]), timestamp=t)
    module.set_prediction(traj)                       # application pushes the current prediction
    p, v = rk4_step(p, v, DT, k_per_m=K_BASE*TRUE_SCALE)   # reality flies with the TRUE drag
    t += DT
d = module.diagnostics()
print('final drag_scale=%.9f  err=%.3e  updates=%d  prediction_available=%s age=%.4f s'
      % (module.drag_scale[0], abs(module.drag_scale[0]-TRUE_SCALE), module.updates[0],
         d['prediction_available'], d['prediction_age_s'][0]))
print('residual_mean(last window)=%.6f m' % d['residual_mean'][0])
