# Independent review: Morph One CAD headless pipeline (2026-10-20)

Reviewed source: `dev/morph-one-cad-pipeline` at
`a042f51f6ddf65c957f5d39feb44c73c4b496ef7`.

**Verdict: STEP->visual USD conversion is supported by code, metadata and
real execution records. This is a valid CAD assetization milestone but NOT
physically usable 4WD4S chassis assembly.**

## Evidence verified from GitHub

- STEP SHA-256 in the saved execution record:
  `388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5`.
- The `full_step.json` reports a successful native converter status (0),
  output SHA-256 `1573488daff493579722f8c6426f81bfa77e186d37f9b917217fa3887a4c6716`,
  58,658,300-byte USDA, 282 prototype meshes, 282 instance proxies,
  455,255 mesh points and stage bbox [580, 372.996, 802.852] *source units*.
- `usd_audit.json` records zero Collision/RigidBody/Mass/ArticulationRoot
  schema applications. The USD is **visual only** at this point.
- `step_trace.json` reports 469 candidate outlier points classified as
  auxiliary geometry, with direct usage 419 LINE and 50 AXIS2_PLACEMENT_3D,
  not BREP vertex or B-spline control point according to the implemented
  ancestry classifier. There remains a risk of incompletely explored
  ancestor/reference paths, and these labels must not be upgraded into
  exact geometric clearance/physical validity certificates.
- A matching total count (282 STEP solid entities, 282 USD prototype meshes)
  is strong supporting evidence, but **does not independently establish**
  a bijective STEP-entity-to-USD-prim mapping or lossless face-level topology.
  That requires an exported source-ID mapping and/or kernel-based volume/face
  comparisons on representative solids.
- 247 names recovered by GB18030 decoding are useful designer intent and
  grouping cues, but *not* measured wheel axes or verified kinematic joints.

**Independent-tool limitation:** ChatGPT inspected GitHub source and text
evidence. No remote USD binary was downloaded or re-imported into Isaac Sim,
and no GPU execution happened from this reviewer environment.

## Code defects fixed by ChatGPT

1. **Native converter result reuse:** old script accepted a `None` error
   code and could accept a pre-existing USD file if the converter did not
   replace it, allowing false-positive "new conversion" verification. The
   converter now requires `status_error_code == 0`, refuses an existing
   output destination, and offers optional explicit mesh-count,
   rendered-instance-count, metersPerUnit and upAxis gates for reproducible
   Morph One conversion. Use a NEW output path on the GPU host.
2. **Incomplete solid attribution:** `attribute_points_to_solids` previously
   stopped *all* reverse-path traversal after discovering the first solid.
   That can silently miss another owner along a longer chain. The traversal
   now continues each uncompleted branch, with a new multi-owner regression.
   **The 469-point ownership histogram is stale until regenerated** from
   this corrected code; don't silently carry its counts forward.
3. **Working-tree side effects during pytest:** the historical B601-DM
   index test executed a generator that rewrote a tracked
   `management/rebot_b601dm/DELIVERABLES.md` file. Generator now supports
   `--output`, and tests write only inside pytest's `tmp_path`.
4. **Unit mismatch when composing USD:** stage `metersPerUnit=0.001`
   means existing positions are in millimetres. OpenUSD references do NOT
   automatically rescale differing stage metrics. Referencing into a metre
   stage requires a 0.001 corrective scaling transform (or an equivalent,
   verified geometry-unit conversion). The USD audit now explicitly reports
   `required_xform_scale_for_metre_stage` and
   `raw_bbox_size_in_source_meters`, expected near
   [0.580, 0.373, 0.803]. Confirm in a **new composed metre-scale scene**;
   do not trust stage metadata alone.
   Reference: https://docs.nvidia.com/learn-openusd/latest/beyond-basics/units.html
5. **Shape fingerprint semantics:** the quantized, translation-invariant
   vertex/circle hash groups duplicate *candidates*. It is not a geometric
   equality proof; it is not rotation invariant, and equal digests do not
   guarantee matching topology. Code comments/output now explicitly say so.

## GPU/CAD runner acceptance before any physical assembly

- Run CPU suites: `pytest tests/simulation/cad
  tests/simulation/rebot_b601_rs tests/simulation/rebot_b601dm/test_deliverables_index.py -q`.
  Show `git status --porcelain` before/after, which should be identical.
- Rerun attribute/trace/census and geometry fingerprint reports against
  the same 115 MB STEP, using *new paths* for output reports. Compare
  469 candidate usage labels and owners against the original run. Any
  changed counts must be explained rather than discarded.
- Rerun converter at an unused destination with
  `--expect-meshes 282 --expect-rendered-instances 282
  --expect-meters-per-unit 0.001 --expect-up-axis Z` and verify the
  new file SHA-256 and reported PASS gates. Do not overwrite prior USD.
- Build a small *separate* Isaac Lab stage with metersPerUnit=1.0,
  reference the visual USD under an Xform scaled 0.001, read back the
  composed world bounds in metres and confirm ~0.580 x 0.373 x 0.803 m.
  Run a negative control with no correction and record the result.
  No physics, joints, collisions or mass on this validation stage.
- Preserve original and generated assets on the GPU host; push only code,
  manifests, JSON logs, trace statistics and the execution SHA to GitHub.
- Do not edit this audit code or assertions during the run. DeepSeek
  develops new integration code on its own branch after these gates pass.

## Remaining research/development unknowns

- Actual wheel, steer axis, chassis root/base frame and mounting geometry,
  wheel centers and rolling radius, whole-body mass/inertia and friction.
- Whether unnamed solids #435700/#759491/#773650/#871375 are
  particular wheel parts; matching extrema/radius are insufficient.
- Whether BREP_WITH_VOIDS #894929 should be retained, hidden, or split.
- Physical articulation: the B601-RS fixed-base asset requires a
  separately validated moving-mount design. Not achieved here.

Do not start PPO or a real-robot motion task based on these CAD-only facts.
