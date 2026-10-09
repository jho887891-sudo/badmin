# Morph One / M1-4WD4S chassis CAD - asset record

**Received:** 2026-10-09 (user-supplied attachment). **Status: stored on the resource host; NOT yet a simulator asset.**

## What it is

The real CAD of the Morph One mobile base - a **4WD4S (four-wheel-drive, four-wheel-steer)** chassis. The
project has been running on a TEMP placeholder until now: `docs/simulation/BADMINTON_ROBOT.md` section 5 lists
`assets/robots/badminton_robot/morph_one/morph_one_temp.usd`, and section 6 fixes the required prim hierarchy as
`/MorphOne/Chassis` plus `FL/FR/RL/RR_Steer` -> `FL/FR/RL/RR_Wheel`, i.e. exactly the 4-steer/4-drive topology this
CAD describes (`docs/architecture/COORDINATE_SYSTEM.md` marks the current geometry `SOURCE=TEMP_PLACEHOLDER`).

## Identity (verified)

| item | value |
|---|---|
| original filename | `M1-4WD4S 模型2601.STEP` |
| stored filename | `M1-4WD4S_model2601.step` (ASCII only: the resource host is a fuseblk/NTFS mount where non-ASCII and spaced names are a known hazard) |
| bytes | 115,496,178 |
| sha256 | `388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5` |
| format | STEP AP203 (`CONFIG_CONTROL_DESIGN`), written by `SwSTEP 2.0` / SolidWorks 2022 |
| internal FILE_NAME | `M1-1.00V3.STEP`, dated `2026-03-06T09:12:09` |
| remote path | `/home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step` |
| local copy | attachment only (see below) - the 115 MB binary is deliberately **not** in this repository |

Verification: the remote copy was byte-verified against the supplied file (same byte count and the same sha256
recorded above) at 2026-10-09 08:57 UTC. Refetch with:

```text
scp dgut@jxxy.taildd42cc.ts.net:/home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step .
```

## What this record does NOT claim

- no geometry inspection was performed: units/scale, part naming, wheel radius, steer axis directions and the
  number of solid bodies are all **unverified**;
- no mass, inertia, friction or joint-limit data has been extracted, so `ALGORITHM_AUDIT.md` P0 items about
  "Morph One geometry/mass" remain open;
- no USD conversion, no articulation, no Isaac wiring: `morph_one_temp.usd` stays in force until a converted asset
  is produced and the topology is validated against section 6 of the robot design;
- licence / redistribution terms were not stated with the file. Treat as internal project material; do not
  redistribute or publish the binary until the terms are confirmed.

## Next steps this asset unblocks

1. convert STEP -> USD (or URDF) and check the 4-steer/4-drive prim hierarchy;
2. measure the real chassis footprint/height and replace the placeholder box in
   `docs/architecture/COORDINATE_SYSTEM.md` (currently L0.70 x W0.55 x H0.25 m @ z0.125, TEMP_PLACEHOLDER);
3. extract mass/inertia or record explicitly that they are still assumed, then rerun the affected audits;
4. only then revisit the whole-body controller / PPO work that depends on real base dynamics.
