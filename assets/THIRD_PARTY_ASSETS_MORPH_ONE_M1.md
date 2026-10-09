# Morph One / M1-4WD4S chassis CAD - asset record

**Received:** 2026-10-09 (user-supplied attachment). **Status: stored on the resource host and fingerprinted; NOT yet a simulator asset.**

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
| internal PRODUCT name | `M1-1.00V3` |
| PRODUCT_DEFINITION name | `未知` (decodes only as GB18030) - the file carries non-ASCII bytes, i.e. a GBK-era SolidWorks export |
| line endings / terminator | CRLF; ends correctly with `END-ISO-10303-21;` |
| units | **millimetres** - entity `#120813 = ( LENGTH_UNIT ( ) NAMED_UNIT ( * ) SI_UNIT ( .MILLI., .METRE. ) )`; length uncertainty `1.0E-05` |
| structure | **multi-body part, not an assembly**: 281 `MANIFOLD_SOLID_BREP`, 398 `CLOSED_SHELL`, 14,714 `ADVANCED_FACE`, 5,692 `CYLINDRICAL_SURFACE`, 720,342 `CARTESIAN_POINT`, 1 `SHAPE_REPRESENTATION`, 0 `NEXT_ASSEMBLY_USAGE_OCCURRENCE` |
| remote path | `/home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step` |
| local copy | attachment only (see below) - the 115 MB binary is deliberately **not** in this repository |

Verification: the remote copy was byte-verified against the supplied file (same byte count and the same sha256
recorded above) at 2026-10-09 08:57 UTC, and **independently re-checked in the follow-up session** -
`sha256sum` on the resource host and `Get-FileHash -Algorithm SHA256` over the supplied attachment both return
the value above. Refetch with:

```text
scp dgut@jxxy.taildd42cc.ts.net:/home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step .
```

## Measured geometry (added in the follow-up session, 2026-10-09)

Envelope over **all** `CARTESIAN_POINT` entities (720,342 points, mm):

| axis | all points | after dropping the 373 outliers |
|---|---|---|
| x | [-500149.495, 500149.495] | [-372.07, 582.10] - span 954.17 |
| y | [-499027.900, 498995.000] | [-1032.43, 715.02] - span 1747.45 |
| z | [-447521.095, 447521.095] | [-609.21, 562.28] - span 1171.50 |

- 373 points (0.052 % of all points) lie beyond ±2 m and **363 of them lie beyond ±100 km**. The raw
  all-point envelope (~1 km) is therefore meaningless as a size check and **must not** be quoted as the model size.
- The right-hand column drops exactly those 373 points. It is still only a *vertex envelope of all 281 solids* -
  an upper bound, **not** a verified chassis outline; construction geometry may still be included.
- **Conversion hazard:** a naive STEP -> USD import would drag the km-scale outlier geometry into the stage,
  which wrecks viewpoint framing, camera clipping and physics bounds. The converter step must drop/clip those
  curves or bodies, and the converted result must be re-measured against
  `docs/architecture/COORDINATE_SYSTEM.md` (placeholder L0.70 x W0.55 x H0.25 m @ z0.125, TEMP_PLACEHOLDER).

## What this record does NOT claim

- no part-level identification: which of the 281 solid bodies are chassis / steering columns / wheels / unrelated
  hardware is **unknown**, so the 4-steer/4-drive topology is still unconfirmed *in the file*;
- the file header fixes the unit as mm, but the model origin, up-axis and forward direction are **not** yet
  aligned or even measured against the project frame;
- no mass, inertia, friction or joint-limit data has been extracted, so `ALGORITHM_AUDIT.md` P0 items about
  "Morph One geometry/mass" remain open;
- no USD conversion, no articulation, no Isaac wiring: `morph_one_temp.usd` stays in force until a converted asset
  is produced and the topology is validated against section 6 of the robot design;
- licence / redistribution terms were not stated with the file. Treat as internal project material; do not
  redistribute or publish the binary until the terms are confirmed.

## Next steps this asset unblocks

1. convert STEP -> USD (or URDF), **dropping the outlier geometry first**, and check the 4-steer/4-drive prim hierarchy;
2. measure the real chassis footprint/height and replace the placeholder box in
   `docs/architecture/COORDINATE_SYSTEM.md` (currently L0.70 x W0.55 x H0.25 m @ z0.125, TEMP_PLACEHOLDER);
3. extract mass/inertia or record explicitly that they are still assumed, then rerun the affected audits;
4. only then revisit the whole-body controller / PPO work that depends on real base dynamics.
