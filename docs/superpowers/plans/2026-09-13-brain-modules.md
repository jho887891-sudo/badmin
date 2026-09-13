# Brain Modules Implementation Plan (2026-09-13)

Announcement: created with the `writing-plans` skill; executed with `subagent-driven-development`
(fresh implementer per task + per-task review + broad final review) and `test-driven-development`.

## Goal
Turn the 8 frozen layer contracts in `src/badminton_brain/` into working modules with tests,
without changing the contracts (`types.py`, `interfaces.py`, `registry.py`, `pipeline.py`, `validation.py`).
Spec: `docs/architecture/ROBOT_BRAIN.md` (S11/S12/S13/S15), `docs/architecture/COORDINATE_SYSTEM.md`,
`docs/simulation/BADMINTON_ROBOT.md` (S4 authenticity levels, S12 TEMP policy, S34 final mode).

## Environment facts every task must use
- Remote repo: `/home/T7/ojh/robot_sim` on `dgut@172.31.68.251` (key-based ssh, never any credential).
- Python: `/home/T7/ojh/robot_sim/env_isaaclab/bin/python` (numpy available). **No Isaac/Kit** in these tasks
  (two Kit instances block each other on this box).
- Run a test: `cd /home/T7/ojh/robot_sim && ./env_isaaclab/bin/python tests/<file>.py`
- Full regression command list is in `outputs/reports/architecture_implementation.md`.

## File structure (locked; one responsibility per file)
```
src/badminton_brain/perception/stereo_geometry.py   ROI gating, subpixel centroid, triangulation (pure numpy)
src/badminton_brain/perception/synthetic_detector.py deterministic synthetic detector (test double, no weights)
src/badminton_brain/estimation/robot_localization.py EKF over odom+IMU(+visual) -> base pose/twist (court)
src/badminton_brain/estimation/shuttle_ukf.py        augmented-state UKF (p, v, drag) -> position/velocity
src/badminton_brain/prediction/physics_predictor.py  RK4 physics rollout -> PredictedTrajectory
src/badminton_brain/decision/feasibility.py          hit feasibility gate -> HitDecision
src/badminton_brain/decision/intercept_search.py     candidate intercepts -> BestIntercept
src/badminton_brain/planning/expert_planner.py       station + racket timing -> WholeBodyTarget
src/badminton_brain/planning/ppo_policy_stub.py      explicit NOT_IMPLEMENTED policy slot
src/badminton_brain/safety/safety_shield.py          limits -> SafeCommand (only executable source)
src/badminton_brain/execution/sim_adapter.py         SafeCommand -> wheel/arm targets (no Isaac calls)
src/badminton_brain/adaptation/online_adaptation.py  slow-loop drag/wind/delay correction + error ledger
src/badminton_brain/apps/full_brain.py               wires all layers into BrainPipeline (T10)
tests/badminton_brain/test_<module>.py               one test file per module
```

## Tasks (each task = one fresh implementer subagent, TDD: RED -> GREEN)
| # | task | deliverable | key acceptance |
|---|---|---|---|
| T1 | perception geometry | `stereo_geometry.py` + `synthetic_detector.py` | synthetic stereo pair → 3D within 1e-6 m; ROI gating rejects outside; no calibration constant invented (TEMP with status) |
| T2 | robot localization | `robot_localization.py` (EKF) | straight-line + yaw-rate synthetic scenario: pose error bounded, covariance PSD, reset(env_ids) isolates envs |
| T3 | shuttle UKF | `shuttle_ukf.py` | synthetic quadratic-drag flight: position RMSE smaller than raw measurement; drag estimate converges; no NaN |
| T4 | physics prediction | `physics_predictor.py` | rollout matches `src/trajectory/shuttle_aerodynamics.rollout` within 1e-9; returns PredictedTrajectory with landing point + arrival time |
| T5 | hit feasibility | `feasibility.py` | rejects out-of-workspace / too-late / overspeed cases; accepts the canonical incoming inside limits; all limits are Param with status |
| T6 | intercept search | `intercept_search.py` | finds the earliest feasible intercept on the predicted trajectory; deterministic ordering; no magic offsets |
| T7 | expert planner | `expert_planner.py` + `ppo_policy_stub.py` | WholeBodyTarget respects Morph One steer limits (uses `morph_one/kinematics.py`); ppo stub is NOT_IMPLEMENTED and final mode refuses it |
| T8 | safety shield | `safety_shield.py` | clamps joint/velocity/workspace, honours e-stop + timeout, marks `limited`/`violations`; never emits a command that violates a limit |
| T9 | execution adapter | `sim_adapter.py` | SafeCommand → 4×(steer, drive) via kinematics + 6 joint targets; returns Feedback; pure (no Isaac) |
| T10 | adaptation | `online_adaptation.py` | drag/wind estimate updated from prediction error; bounded update; ledger of residuals |
| T11 | integration | `apps/full_brain.py` + `tests/badminton_brain/test_full_brain.py` | canonical scenario end-to-end through BrainPipeline; dev mode ok; final mode fails only for the declared NOT_IMPLEMENTED items |
| T12 | broad review + records | review report + management records | independent review of all modules; regressions green; report in `outputs/reports/` |

## Rules every task must obey
1. Contracts in `src/badminton_brain/*.py` (types/interfaces/registry/pipeline) are **frozen**: read them, do not edit them.
2. TDD: failing test first (show the RED output), then minimal implementation, then GREEN. Report both.
3. Unknown real values → `Param(value=None, status=REQUIRES_MEASUREMENT, source=...)`; TEMP values must be marked
   `TEMP_PARAMETERIZED_PROXY` with a source. Never present a guess as measured.
4. No PPO/RL training, no Isaac/Kit, no new third-party dependency (numpy only).
5. Only touch the files the task owns; other modules belong to other agents working in parallel.
6. Commit is done by the coordinator after review (agents report, they do not commit).

## Ledger
Decisions and deviations are recorded here as `Ruling: ... — why — cost if wrong`.
- Ruling: contracts frozen, implementations parallel — why: independent files, no shared state — cost: interface friction must be fixed by the coordinator in integration.
- Ruling: T11 wiring uses **interface discovery** (each layer module must expose exactly one
  implementer of its layer interface) instead of hard-coded class names — why: ten parallel
  implementers choose their own class names and the coordinator must not guess — cost if wrong:
  a module exposing two implementers is rejected loudly (safe failure, one-line fix).
- Ruling: a project-wide regression runner (`tools/run_all_tests.py`) is the single gate for
  "全量回归" — why: SDD requires verifying after every task, and ad-hoc command lists drift —
  cost if wrong: the gate could miss a suite that does not follow the `tests/**/test_*.py` pattern.
- Ruling: `RobotSensorState` gains **optional** `odom_twist (N,3)` and `imu_yaw_rate (N,)` channels
  (defaults None, validated when present) so the estimation layer can be plugged in; the frozen
  architecture tests still pass unchanged — why: T2's EKF needs proprioception the contract lacked —
  cost if wrong: two extra optional fields on one message (no caller breaks).
- Ruling: the estimation adapter `estimation/estimator.py` is the ONLY bridge between T2's batched EKF
  API and `EstimationModule`; shuttle estimation is measurement passthrough until the T3 UKF is
  injected, and that shortcut is labelled TEMP in the module (never presented as an estimate).
- Ruling: the decision layer reports **batch level**: aggregate `HitDecision` (feasible only if every
  environment is feasible) plus the batched `BestIntercept` that T6 already returns; per-environment
  decisions stay available as `last_decisions` — why: the frozen interface returns one 2-tuple per step
  and T7 already reads `decision.feasible` + (N,.) arrays — cost if wrong: partial batches are handled
  conservatively (nothing is planned) instead of per-env.
- Ruling: the Morph One kinematics stays a **single source** at
  `simulation/robots/badminton_robot/morph_one/kinematics.py`; `src/` imports it through a documented
  lazy loader (T7/T9 both did this) — why: duplicating it would create two truths — cost if wrong:
  `src/` depends on a simulation path (documented, no functional impact).
