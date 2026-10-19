# Velocity-channel experiment on commit 55112d2 (execution record)

Role: ChatGPT authored the diagnostics; this is the execution record. The probe script was run
unchanged - no source, tolerance, gain, joint limit or asset was touched, and no earlier evidence file
was overwritten (all artefacts below are new files in a new directory).

## 1. Provenance

| Item | Value |
| --- | --- |
| Target commit | 55112d26e1e3e830254702a77b5972eae8a920b1 |
| Commit tree | ea9a25547284c313f3ed91ac467e973dd3a9ab1d |
| Executed tree (git write-tree on the host) | ea9a2554...ab1d - MATCH |
| Per-file blob check | 2238 / 2238 files match (0 blobdiff, 0 missing) |
| Transport | git archive, sha256 56cb5ec9...93de identical on both sides |
| Only file changed vs the previous audit target 454197e | scripts/simulation/probe_rs_velocity_channels.py (md5 56934296894e096a7d03467beb995e27) |
| Host checkout | /home/T7/dgut/runs/review-55112d2 |

Module provenance printed before any simulation pointed at that checkout (badminton_scene,
multienv_arm, scene_layout), gains 300 / 20 unchanged. GPU was shared again: 24.8 GiB already in use
at start.

## 2. Commands and exit codes (11 runs, all exit 0)

Common: probe_rs_velocity_channels.py; module import path = the review checkout;
`--headless`; group 1 settle 240 / sample 480 at dt 1/240 (= 1 s settle + 2 s measure);
group 2 settle 480 / sample 960 at dt 1/480 (same 1 s + 2 s); group 3 same as group 1/2 with --mode step.

| run | mode | solver flags | exit | s | log |
| --- | --- | --- | --- | --- | --- |
| A | hold | tgs, external-forces off | 0 | 181 | logs/A_tgs_extoff_240.log |
| B | hold | tgs, external-forces on | 0 | 51 | logs/B_tgs_exton_240.log |
| C | hold | tgs, on, min-velocity-iterations 2 | 0 | 37 | logs/C_tgs_exton_mv2_240.log |
| D | hold | pgs | 0 | 48 | logs/D_pgs_240.log |
| E | hold | tgs, off, gravity off | 0 | 35 | logs/E_tgs_extoff_zerog_240.log |
| A@480 | hold | tgs, off | 0 | 45 | logs/A2_tgs_extoff_480.log |
| B@480 | hold | tgs, on | 0 | 44 | logs/B2_tgs_exton_480.log |
| G3 A | step | tgs, off | 0 | 43 | logs/G3_step_A_tgs_extoff_240.log |
| G3 B | step | tgs, on | 0 | 41 | logs/G3_step_B_tgs_exton_240.log |
| G3 C | step | tgs, on, min-vel 2 | 0 | 43 | logs/G3_step_C_tgs_exton_mv2_240.log |
| G3 B@480 | step | tgs, on | 0 | 53 | logs/G3_step_B_tgs_exton_480.log |

**Effective solver configuration** was recorded inside every JSON (solver_config.physics_kwargs) and is
corroborated by IsaacLab's own warning, which is printed exactly when TGS runs with external forces
per-iteration disabled:

| configuration | warning 'TGS solver with enable_external_forces_every_iteration=False may cause noisy velocities' |
| --- | --- |
| A @240, A @480, E @240, G3-A @240 (external forces OFF) | present in all four logs |
| B, B@480, C, G3-B, G3-B@480, G3-C (external forces ON) | absent |
| D (PGS) | absent |

## 3. Results - the three metrics requested

joint2, unrounded. measured_delta is the actual position change over the 2 s window;
integrated_physx_velocity is the sum of the reported velocities x dt; integral_minus_position is the
discrepancy that should be ~0.

| run | mode | dt | external forces | mean v | integrated | measured dq | integral - dq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | hold | 1/240 | **off** | 0.067574 | 0.134867 | 0.000000 | **0.134867** |
| B | hold | 1/240 | **on** | -0.000072 | -0.000143 | 0.000000 | **-0.000143** |
| C | hold | 1/240 | on, min-vel 2 | -0.000072 | -0.000143 | 0.000000 | -0.000143 |
| D | hold | 1/240 | PGS | -0.000002 | -0.000004 | 0.000000 | **-0.000004** |
| E | hold | 1/240 | off, zero g | 0.000000 | 0.000000 | 0.000000 | **0.000000** |
| A@480 | hold | 1/480 | off | 0.036709 | 0.073341 | 0.000000 | 0.073341 |
| B@480 | hold | 1/480 | on | -0.000156 | -0.000311 | 0.000000 | -0.000311 |
| G3 A | step | 1/240 | off | 0.065353 | 0.128416 | -0.001151 | 0.129567 |
| G3 B | step | 1/240 | on | -0.000098 | -0.002203 | -0.001151 | -0.001052 |
| G3 C | step | 1/240 | on, min-vel 2 | -0.000098 | -0.002203 | -0.001151 | -0.001052 |
| G3 B@480 | step | 1/480 | on | -0.000052 | -0.000694 | -0.000323 | -0.000371 |

Supporting columns in the JSONs, all consistent with the above: max|dq per step| is 0.0 in every hold
run (the arm genuinely does not move) and 0.0042 / 0.0021 rad in the step runs (the arm genuinely
does move), while max_abs_lab_physx_q_difference and max_abs_lab_physx_qd_difference are 0.0 in all
eleven runs - Isaac Lab and PhysX still agree exactly, so this is physics, not wrapper desync.

## 4. What this shows (facts only; the interpretation is ChatGPT's call)

1. **External-forces-per-iteration is the switch.** Turning it ON reduces the hold-window discrepancy
   from 0.134867 rad to 0.000143 rad (about 940x, i.e. from 7.7 deg of phantom motion to 0.008 deg)
   and the same switch reduces the step-mode discrepancy from 0.129567 to 0.001052 rad.
2. **It is not a dt artefact.** At 1/480 the OFF configuration still shows 0.073341 rad while the ON
   configuration shows 0.000311 rad; the ordering is unchanged at both rates.
3. **PGS is also clean** (-0.000004 rad), so this is not unique to TGS; the defect is specific to TGS
   with external forces applied only in the first iteration.
4. **min-velocity-iterations = 2 changes nothing** - C is bit-identical to B in both hold and step
   (same mean, same integral, same measured delta). It was applied (physics_kwargs records
   min 2 / max 4), it just has no effect on this residual.
5. **Zero gravity reproduces the clean case even with the bad solver setting** (0.000000), which pins
   the residual on the gravity load interacting with that solver configuration, matching the earlier
   finding from the 454197e round.

Note on the step-mode residual: with external forces ON, integral - dq is -0.001052 rad while the
commanded motion was -0.001151 rad, i.e. the same order of magnitude as the motion itself, so step-mode
position/velocity agreement is much better than before but not exact. At 1/480 it is -0.000371 against a
-0.000323 rad motion. Whoever decides the acceptance threshold should look at those numbers rather than
at the hold number alone.

## 5. Problems encountered

None blocked the run. Every command exited 0 and no traceback occurred. Two operational notes:

* the GPU was shared with another Isaac Sim session (24.8 GiB already used), so wall-clock times vary
  (A took 181 s, the rest 35-53 s) and are not comparable as performance numbers;
* log filenames keep the runner's internal names for the 480 Hz runs (A2_/B2_), while the JSONs use
  the requested A_/B_ names - the mapping is in the table above.

## 6. Files

| Path | Contents |
| --- | --- |
| A_*.json, B_*.json, C_*.json, D_*.json, E_*.json, G3_*.json | one per configuration: full 480/960 unrounded samples (lab q/qd, PhysX q/qd, command, target), solver_config, summary |
| logs/ | stdout/stderr per run, the runner summary 00_summary.txt, and both runner console captures |
| MANIFEST.sha256 | sha256 and size of every file here |
