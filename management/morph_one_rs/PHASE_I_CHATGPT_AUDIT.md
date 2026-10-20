# ChatGPT review of P0–P4 metric scene and STEP–USD matching (2026-10-20)

Reviewed developer commit: `dev/morph-one-metric-scene-and-mapping` @
`3ce42c09cc0c86133a673c3756fd1fbf79d02a81`.

## Scope and evidence

- **P0**: source log reports 1 failed / 142 passed / 3 skipped; the
  failing assertion `len(report["caveats"]) == 3` was stale after the
  reviewer added a fourth essential caveat about fingerprints.
- **P1**: the submitted regenerated evidence reports 469 candidate
  Cartesian points (not 470); all 469 classified as auxiliary, 419
  directly referenced by LINE and 50 by AXIS2_PLACEMENT_3D. The
  reported unchanged Git blobs support consistency of this re-run, but
  do not independently establish CAD solid-level semantic identities.
- **P2**: the reported converted USD has 282 prototype meshes, 282
  rendered mesh instances, correct millimetre stage metric
  (`metersPerUnit=0.001`), Z-up and a measured source-unit world bbox
  near [580, 372.996, 802.852]. Hashes differ across conversions:
  accept structured mesh/geometry verification, not a fixed USDA hash.
- **P3: ACCEPTED FOR VISUAL GEOMETRY UNIT COMPOSITION**: The submitted
  `metric_scene_report.json` and source code show mesh instance
  vertices actually transformed in composed stages: A (unscaled) world
  bbox [580,372.996,802.852] in a metre stage; B (0.001 scale) world
  bbox [0.580000028,0.372995715,0.802851662] metres. BBoxCache
  agrees. Both stages have 282 mesh instances and no authored physics
  APIs. This is NOT validation of chassis physics, joint motion,
  mounting, mass/inertia or wheel geometry.
- **P4**: the original 199 VERIFIED / 25 CANDIDATE / 58 unmatched
  output is a *pre-audit heuristic report*, not independently
  established source-entity-to-USD identity. Do not reuse those counts
  after algorithm correction.

## Repaired defects

1. `sandwich()` mistakenly implemented the min-side vertex coverage
   inequality backwards. Correct condition is
   `mesh_min <= step_vertex_min` and
   `mesh_max >= step_vertex_max` (within disclosed tolerance).
   Tests now cover min-side legitimate surface bulge and missed
   STEP-vertex coverage.
2. `match_parts()` previously greedily selected the lowest-cost
   name-group pairs, potentially leaving matchable solids stranded.
   It now uses deterministic augmenting paths for a
   **maximum-cardinality** feasible matching within each name group.
   This maximizes matched *count*, not total assignment likelihood
   or correctness; ambiguous assignments remain candidates.
   New regression uses a synthetic A-with-two-partners/B-with-one
   graph that global greedy cannot fully match.
3. The former VERIFIED definition needed only a unique normalized
   name plus approximate bounding-box containment. Neither proves
   entity identity. This is now `STRONG_CANDIDATE`;
   duplicate/unnamed-name matches are `CANDIDATE`. `VERIFIED`
   is reserved for explicit per-solid provenance or a stronger
   independently validated identity link. `bijection_proven=False`
   until such evidence is provided.
4. `test_solid_geometry.py` previously assumed exactly three
   caveats; it now asserts the meaningful fingerprint caveat exists.
5. Visual composition verdict previously did not enforce
   BBoxCache agreement, stage metric and Z-up, and could silently
   accept missing PhysX audit data as zero APIs. The composed scene
   runner now gates on all recorded scene checks and CPU tests
   cover the fail-closed interpretation.

## Limitations and acceptance gates

- The numeric containment tolerance `2.5 mm` is heuristic;
  a claimed tessellation-chord-error origin has not been verified
  independently. Upper bounds can be loose because auxiliary
  LINE/placement origins are included in closure. Passing
  a containment test is evidence of plausibility, not proof of
  source-to-mesh identity.
- Maximum-cardinality matching may find a different solution than
  the prior cost-greedy algorithm. It does not minimize total cost.
  The new grade and pairing figures **must be regenerated from the
  original STEP and USD with this exact review commit**.
- ChatGPT audited code and the committed JSON/USDA text evidence
  over GitHub. The 58 MB USD binary remains on the GPU host and was
  not independently executed/read back by the reviewer. This review
  branch's CPU tests and Isaac Sim runs have **not yet been executed**.

## Runner task (DeepSeek executes, does not change this review code)

1. Verify exact review SHA and a clean worktree.
2. Run `pytest tests/simulation/cad tests/simulation/rebot_b601_rs
   tests/simulation/rebot_b601dm/test_deliverables_index.py -q`;
   run the pxr-dependent tests in Isaac venv; show exit codes and
   before/after `git status --porcelain`.
3. Rerun `map_step_solids_to_usd_meshes.py` on the original STEP
   and verified current USD. Compare new counts, grades,
   rejected pairs and unmatched IDs vs original 224/199/25/58.
   Document specifically which pairs changed due to the corrected
   min-side inequality or augmenting-path assignment.
4. Rerun `run_metric_scene_validation.py` on the existing visual USD
   under a distinct output directory; confirm both new
   `both_have_valid_scene_checks=True` and `verdict.ok=True`,
   along with the raw two-geometry-method measurements.
5. Push all new raw logs and JSON to an independent evidence
   branch, not source. Do not start collision/dynamics, PPO or
   whole-body mounting under this task.

Do not merge the audit until execution is reviewed.
