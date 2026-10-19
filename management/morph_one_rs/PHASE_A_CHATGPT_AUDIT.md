# ChatGPT independent audit of Morph One phase A

Source implementation: `feature/morph-one-rs-integrated-assembly` @
`59f89488b92608552c5f0c58b6cfd70554419bf6`.

## Verdict

- The STEP header, original binary SHA-256, flattened AP203 structure, units,
  and count of 281 MANIFOLD_SOLID_BREP plus one BREP_WITH_VOIDS are
  well-supported by the submitted log and inventory.
- The 8 semantic joints are **API declarations only**. Nothing here proves
  there are eight physical joints in the STEP or a working chassis articulation.
- Actual part classification, CAD tessellation/USD export, a moving chassis,
  and a rigid moving mount for B601-RS remain unimplemented.
- This is an acceptable **blocker report**, not completion of CAD assetization.

## Independent audit corrections

1. `outlier_count=469` identifies percentile-band **candidates**, not 469
   invalid mechanical solids or 469 points at hundreds of meters.
   Submitted `outlier_sample[0]` is only y=243.76 mm. Other samples do show
   genuine extreme coordinates of hundreds of thousands of mm.
   Do not delete points or entire BREP solids without tracing dependency
   chains, classifying shape function and verifying the converted result.
2. The absence of `.py` files under `omni.kit.converter.cad` does NOT prove
   that headless conversion is impossible. NVIDIA documents a CAD Converter
   Service/Kit CLI batch path:
   https://docs.omniverse.nvidia.com/kit/docs/omni.services.convert.cad/latest/Usage.html
   Whether this host has `omni.services.convert.cad` and appropriate core
   converter binaries installed, and whether conversion succeeds for this
   particular STEP, is **not yet established**.
3. The STEP inventory JSON + console log are committed, but the generator
   that calculated percentile statistics and mapped point samples to indices
   is not committed. The next phase must include a deterministic source
   script/command plus explicit rules for point/entity tracing.
4. The fixed-root joint in the B601-RS USD is a strong reason not to assume
   static arm-on-chassis visual placement is a correct physical mount.
   The combined-articulation design is a recommendation pending GPU testing,
   not a proven prohibition of all other PhysX constraint arrangements.
5. `ChassisParameter` previously permitted `VERIFIED_MEASURED` with
   `value=None` or nonfinite numbers. The review patch now rejects those
   unsupported states and adds pure-CPU tests.

## Canonical B601-RS evidence restored WITHOUT fabricating a new run

Earlier default `outputs/simulation/rebot_b601_rs/MULTIENV_hold.json`
was a pre-TGS snapshot, while current test suite requires post-TGS
`velocity_position_consistent=True` and a `physics_configuration` object.

The audit patch performs a Git-object exact replacement only:

- Original canonical blob `7ba89bb59f5601e1dd5290ad60d2320679456fb4`
  retained at `outputs/simulation/rebot_b601_rs/archive/MULTIENV_hold_pre_tgs.json`.
- Replacement canonical blob `2340a73c06c0d15c23a6b0eeed0ea92257f665a8`,
  copied unchanged from
  `evidence/b601-rs-acceptance-173966b` @
  `972819366fba3dcb841c3dcb1909455d1e40ed72`,
  path `outputs/simulation/rebot_b601_rs/production_review_173966b/MULTIENV_hold.json`.

No tolerances were weakened. The repaired canonical JSON includes all
six counts (2,4,16,32,64,128), the TGS force-per-iteration setting, and
`velocity_position_consistent=True`. The historical result remains
available for regression analyses.

**The snapshot itself does not constitute a fresh GPU rerun on the Morph One
branch.** Re-run the complete CPU test suite on this review SHA (no GPU
required to read committed evidence) and report actual output before merging.

## Next builder task (DeepSeek, new feature author)

Priority 1: investigate installed NVIDIA CAD Converter *service* / Kit CLI
and capture an actual conversion smoke test with stdout/stderr and an
out-of-repository USD for visual inspection. Inspect core converter format
support for STEP specifically; do not assume all converter cores read all
formats. If absent, document and pursue a licensed FreeCAD/OCP fallback.

Priority 2: implement a committed, reproducible STEP geometry inventory
script with per-BREP/face/edge dependency tracing. Only then perform
candidate solid classification, validation with available engineering
drawings, and import staging. Keep all unknown/temporary values labelled.

Priority 3 (separate future change): design/fabricate an articulation
only after CAD evidence resolves wheel geometry, axes and mount geometry.

**Do not claim full chassis assembly until geometry, contact, joint
actuation and chassis-arm transform have been validated in Isaac Sim.**
