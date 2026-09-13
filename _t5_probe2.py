
import sys, math
sys.path.insert(0,'/home/T7/ojh/robot_sim/src')
import numpy as np
from badminton_brain.decision.feasibility import HitFeasibilityGate
from badminton_brain.types import PredictedTrajectory, UnifiedState

G=9.81
def traj():
    p0=np.array([1.2,0.0,1.8]); v0=np.array([-3.0,0.0,1.5])
    tl=(v0[2]+math.sqrt(v0[2]**2+2*G*p0[2]))/G
    ts=np.arange(int(tl/0.004)+1)*0.004
    ts=np.append(ts,tl)
    acc=np.array([0.,0.,-G])
    pos=p0+v0*ts[:,None]+0.5*acc*ts[:,None]**2
    vel=v0+acc*ts[:,None]
    land=(p0+v0*tl+0.5*acc*tl*tl)
    return PredictedTrajectory(times=ts, position=pos[None], velocity=vel[None],
                               landing_point=land[None], arrival_time=np.array([tl]), timestamp=0.0)

def state():
    return UnifiedState(base_pose=np.array([[-1.6,0,0,0,0,0,1.0]]), base_twist=np.zeros((1,6)),
        joint_pos=np.zeros((1,6)), joint_vel=np.zeros((1,6)), racket_contact_pose=np.zeros((1,7)),
        racket_contact_twist=np.zeros((1,6)), shuttle_position=np.array([[1.2,0,1.8]]),
        shuttle_velocity=np.array([[-3.0,0,1.5]]), timestamp=0.0)

g=HitFeasibilityGate()
print("baseline            :", g.evaluate(state(), traj()).reason)

s=state(); s.shuttle_position[0,0]=np.nan
print("state pos NaN       :", g.evaluate(s, traj()).reason, "(expect INVALID_STATE)")

s=state(); s.base_pose[0,0]=np.nan
print("state base NaN      :", g.evaluate(s, traj()).reason, "(expect INVALID_STATE)")

t=traj(); t.velocity[0,0,0]=np.nan      # in-place mutation after construction
print("traj velocity NaN   :", g.evaluate(state(), t).reason, "(expect INVALID_PREDICTION)")

t=traj(); t.landing_point[0,0]=np.nan
print("traj landing NaN    :", g.evaluate(state(), t).reason, "(expect INVALID_PREDICTION)")

t=traj(); t.position[0,:,2]=np.nan
print("traj position NaN   :", g.evaluate(state(), t).reason, "(expect INVALID_PREDICTION)")

t=traj(); t.times[:]=np.nan
print("traj times NaN      :", g.evaluate(state(), t).reason, "(expect INVALID_PREDICTION)")

t=traj(); t.arrival_time[0]=np.nan
print("traj arrival NaN    :", g.evaluate(state(), t).reason, "(expect INVALID_PREDICTION)")

t=traj(); t.velocity[0,0,:]=np.array([np.inf,0,0])
print("traj velocity Inf   :", g.evaluate(state(), t).reason, "(expect INVALID_PREDICTION / TOO_FAST)")

# state and prediction disagree about where the shuttle is
s=state(); s.shuttle_position[0]=np.array([5.0,0.0,1.8]); s.shuttle_velocity[0]=np.array([9.0,0.0,0.0])
print("state/pred mismatch :", g.evaluate(s, traj()).reason, "(gate uses trajectory only)")

# boundary + degenerate limits
import dataclasses
from badminton_brain.status import AssetStatus, Param
from badminton_brain.decision.feasibility import FeasibilityLimits
lim = dataclasses.replace(FeasibilityLimits(),
        station_x_range_m=Param((0.0,0.0), AssetStatus.TEMP_PARAMETERIZED_PROXY,'degenerate station'),
        arm_reach_x_m=Param(0.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,'zero reach'))
try:
    print("degenerate limits   :", g.__class__(lim).evaluate(state(), traj()).reason)
except Exception as e:
    print("degenerate limits   : EXC", type(e).__name__, e)
