
import math, sys
sys.path.insert(0,'/home/T7/ojh/robot_sim/src')
import numpy as np
from badminton_brain.decision.feasibility import (FeasibilityLimits, HitFeasibilityGate,
                                                  HitReason, base_min_travel_time_s)
from badminton_brain.status import AssetStatus, Param, UNRESOLVED_STATUSES
from badminton_brain.types import PredictedTrajectory, UnifiedState

G = 9.81
def t_land(p0, v0):
    return (v0[2] + math.sqrt(v0[2]**2 + 2.0*G*p0[2])) / G

def make_traj(p0=(1.2,0.0,1.8), v0=(-3.0,0.0,1.5), horizon=None, landing=None,
              dt=0.004, shift=0.0, stamp=0.0, arrival=None):
    p0 = np.array(p0, float); v0 = np.array(v0, float)
    tl = t_land(p0, v0)
    tmax = tl if horizon is None else min(horizon, tl)
    n = int(math.floor(tmax/dt)) + 1
    ts = np.arange(n)*dt
    if ts[-1] < tmax - 1e-12:
        ts = np.append(ts, tmax)
    acc = np.array([0.0, 0.0, -G])
    pos = p0 + v0*ts[:,None] + 0.5*acc*ts[:,None]**2
    vel = v0 + acc*ts[:,None]
    land = p0 + v0*tl + 0.5*acc*tl*tl if landing is None else np.array(landing, float)
    arr = (tl if arrival is None else float(arrival)) + shift
    return PredictedTrajectory(times=ts+shift, position=pos[None], velocity=vel[None],
                               landing_point=land[None], arrival_time=np.array([arr]),
                               timestamp=stamp)

def make_state(base=(-1.60,0.0,0.0), p=(1.2,0.0,1.8), v=(-3.0,0.0,1.5), ts=0.0):
    bp = np.array([[base[0],base[1],base[2],0.0,0.0,0.0,1.0]])
    return UnifiedState(base_pose=bp, base_twist=np.zeros((1,6)), joint_pos=np.zeros((1,6)),
                        joint_vel=np.zeros((1,6)), racket_contact_pose=np.zeros((1,7)),
                        racket_contact_twist=np.zeros((1,6)),
                        shuttle_position=np.array([p],float), shuttle_velocity=np.array([v],float),
                        timestamp=ts)

# ---- independent numeric integration of the canonical flight (rk4, drag-free) ----
def rk4(p0, v0, dt=1e-5):
    p = np.array(p0,float); v = np.array(v0,float); t = 0.0
    while p[2] > 0.0:
        def f(pp, vv): return vv, np.array([0.0,0.0,-G])
        k1p,k1v = f(p,v)
        k2p,k2v = f(p+0.5*dt*k1p, v+0.5*dt*k1v)
        k3p,k3v = f(p+0.5*dt*k2p, v+0.5*dt*k2v)
        k4p,k4v = f(p+dt*k3p, v+dt*k3v)
        p = p + dt/6.0*(k1p+2*k2p+2*k3p+k4p)
        v = v + dt/6.0*(k1v+2*k2v+2*k3v+k4v)
        t += dt
    return t, p

tl_an = t_land(np.array([1.2,0.0,1.8]), np.array([-3.0,0.0,1.5]))
x_an = 1.2 - 3.0*tl_an
t_rk, p_rk = rk4((1.2,0.0,1.8), (-3.0,0.0,1.5))
print("ANALYTIC  t_land=%.6f  landing_x=%.6f" % (tl_an, x_an))
print("RK4       t_land=%.6f  landing=(%.6f, %.6f, %.6f)" % (t_rk, p_rk[0], p_rk[1], p_rk[2]))

gate = HitFeasibilityGate()
canon_traj = make_traj()
print("canonical traj landing_point[x]=%.6f arrival=%.6f  (test asserts -1.133 / 0.778)"
      % (float(canon_traj.landing_point[0,0]), float(canon_traj.arrival_time[0])))

cases = [
  ("invalid_state(None)",              None,                              canon_traj, None, "INVALID_STATE"),
  ("invalid_state('str')",             "not-a-state",                     canon_traj, None, "INVALID_STATE"),
  ("invalid_prediction(None)",         make_state(),                      None,       None, "INVALID_PREDICTION"),
  ("stale_state(now=0.06)",            make_state(ts=0.0),                make_traj(stamp=0.0), 0.06, "STALE_STATE"),
  ("stale_prediction(state ts=0.06)",  make_state(ts=0.06),               make_traj(stamp=0.0), 0.06, "STALE_PREDICTION"),
  ("too_fast(v=40m/s)",                make_state(v=(-40,0,1.0)),         make_traj(v0=(-40.0,0.0,1.0)), None, "SHUTTLE_TOO_FAST"),
  ("too_slow(v=0.10m/s)",              make_state(p=(0.5,0,0.5), v=(-0.1,0,0.001)),
                                                                          make_traj(p0=(0.5,0.0,0.5), v0=(-0.1,0.0,0.001)), None, "SHUTTLE_TOO_SLOW"),
  ("wrong_direction(vx=+1)",           make_state(p=(0.5,0,1.5), v=(1.0,0,1.0)),
                                                                          make_traj(p0=(0.5,0.0,1.5), v0=(1.0,0.0,1.0)), None, "WRONG_DIRECTION"),
  ("oob_x(landing -7.2)",              make_state(),                      make_traj(landing=(-7.2,0.0,0.0)), None, "OUT_OF_BOUNDS"),
  ("oob_y(landing +3.2)",              make_state(),                      make_traj(landing=(-1.0,3.2,0.0)), None, "OUT_OF_BOUNDS"),
  ("opponent_side(landing +1.5)",      make_state(),                      make_traj(landing=(1.5,0.0,0.0)), None, "OUTSIDE_RESPONSIBILITY"),
  ("flight_over(shift=-1.0)",          make_state(),                      make_traj(shift=-1.0), None, "NO_TIME_MARGIN"),
  ("base_far(-6.0)",                   make_state(base=(-6.0,0,0)),       canon_traj, None, "NO_TIME_MARGIN"),
  ("above_workspace(z~3.5)",           make_state(p=(-1.0,0,3.5), v=(-0.2,0,1.0)),
                                                                          make_traj(p0=(-1.0,0.0,3.5), v0=(-0.2,0.0,1.0), horizon=0.5,
                                                                                    landing=(-1.1,0.0,0.0)), None, "UNREACHABLE"),
  ("ACCEPT canonical(home)",           make_state(),                      canon_traj, None, "FEASIBLE"),
  ("ACCEPT canonical(base -1.30)",     make_state(base=(-1.30,0,0)),       canon_traj, None, "FEASIBLE"),
  ("boundary x=-6.70 (inside)",        make_state(),                      make_traj(landing=(-6.70,0.0,0.0)), None, "not OUT_OF_BOUNDS"),
  ("boundary x=-6.71 (outside)",       make_state(),                      make_traj(landing=(-6.71,0.0,0.0)), None, "OUT_OF_BOUNDS"),
]
bad = 0
for name, st, tr, now, exp in cases:
    try:
        d = gate.evaluate(st, tr) if now is None else gate.evaluate(st, tr, now=now)
        got = d.reason
    except Exception as e:
        got = "EXC:%s: %s" % (type(e).__name__, e)
    ok = (got == exp) or (exp == "not OUT_OF_BOUNDS" and got != "OUT_OF_BOUNDS")
    if not ok: bad += 1
    print("%-34s expect=%-22s got=%-22s %s" % (name, exp, got, "OK" if ok else "*** MISMATCH ***"))

print("--- determinism (3 repeats) ---")
for i in range(3):
    r1 = gate.evaluate(make_state(), make_traj(landing=(-7.2,0.0,0.0))).reason
    r2 = gate.evaluate(make_state(p=(-1.0,0,3.5), v=(-0.2,0,1.0)),
                       make_traj(p0=(-1.0,0.0,3.5), v0=(-0.2,0.0,1.0), horizon=0.5, landing=(-1.1,0.0,0.0))).reason
    print("   repeat%d: %s / %s" % (i, r1, r2))

print("--- batch (mixed near/far base) ---")
near, far = make_state(), make_state(base=(-6.0,0,0))
st2 = UnifiedState(base_pose=np.vstack([near.base_pose, far.base_pose]),
                   base_twist=np.zeros((2,6)), joint_pos=np.zeros((2,6)), joint_vel=np.zeros((2,6)),
                   racket_contact_pose=np.zeros((2,7)), racket_contact_twist=np.zeros((2,6)),
                   shuttle_position=np.tile(near.shuttle_position,(2,1)),
                   shuttle_velocity=np.tile(near.shuttle_velocity,(2,1)), timestamp=0.0)
tr2 = PredictedTrajectory(times=canon_traj.times, position=np.tile(canon_traj.position,(2,1,1)),
                          velocity=np.tile(canon_traj.velocity,(2,1,1)),
                          landing_point=np.tile(canon_traj.landing_point,(2,1)),
                          arrival_time=np.tile(canon_traj.arrival_time,2), timestamp=0.0)
print("   ", [d.reason for d in gate.evaluate_batch(st2, tr2)])

print("--- base travel model cross-check (my own closed form) ---")
def myT(d, v, a):
    if d <= 0: return 0.0
    if d <= v*v/a: return 2.0*math.sqrt(d/a)
    return d/v + v/a
for (d, v, a) in [(0.0,1.0,1.5),(0.375,1.0,1.5),(0.6666666667,1.0,1.5),(3.0,1.0,1.5),(1.0,2.0,0.5)]:
    print("   d=%.4f v=%.1f a=%.1f  impl=%.9f  mine=%.9f" %
          (d, v, a, base_min_travel_time_s(d,v,a), myT(d,v,a)))

print("--- TEMP / authenticity discipline ---")
lim = FeasibilityLimits()
params = lim.param_limits()
print("   total Param limits:", len(params))
for n, p in sorted(params.items()):
    print("     %-28s %-26s value=%s" % (n, p.status.value, p.value))
print("   unresolved_limits:", lim.unresolved_limits())
mr = lim.measurement_requirements()
print("   measurement_requirements: n=%d all_value_None=%s statuses=%s" %
      (len(mr), all(p.value is None for p in mr.values()), sorted({p.status.value for p in mr.values()})))
print("   non-TEMP limits:", [n for n,p in params.items() if p.status != AssetStatus.TEMP_PARAMETERIZED_PROXY])
print("   VERIFIED_* anywhere:", [n for n,p in params.items() if 'VERIFIED' in p.status.value])
print("   empty sources:", [n for n,p in params.items() if not str(p.source).strip()])

print("--- untyped/degenerate inputs ---")
try:
    print("   zero v_max:", base_min_travel_time_s(1.0, 0.0, 1.5))
except Exception as e:
    print("   zero v_max ->", type(e).__name__, e)
try:
    print("   nan d:", base_min_travel_time_s(float('nan'), 1.0, 1.5))
except Exception as e:
    print("   nan d ->", type(e).__name__, e)
print("   nan state:", gate.evaluate(make_state(p=(float('nan'),0,1.5)), canon_traj).reason)
print("   inf state:", gate.evaluate(make_state(p=(float('inf'),0,1.5)), canon_traj).reason)
print("   env_id out of range:", gate.evaluate(make_state(), canon_traj, env_id=3).reason)
