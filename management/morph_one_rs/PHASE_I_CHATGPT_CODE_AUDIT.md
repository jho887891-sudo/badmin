# Morph One CAD developer branch audit

**Audited source:** `jho887891-sudo/badmin@3ce42c09cc0c86133a673c3756fd1fbf79d02a81`,
branch `dev/morph-one-metric-scene-and-mapping`.
**Corrections are exclusively on** `audit/morph-one-sandwich-review`.
The developer branch and main were not modified.

## Verified evidence (bounded claims)

- The committed P4 JSON reports 282 STEP solids, 282 USD mesh instances,
  224 recorded pairs, 199 `VERIFIED`, 25 `CANDIDATE`, 58 unmatched
  on each side, and `bijection_proven=false`.
- Recomputing **all 224 archived matched-pair bboxes**, using the corrected
  containment inequalities and existing 2.5 mm / 1% tolerance, gives
  **224 pass, 0 fail**. This does *not* recompute the 58 unmatched elements;
  the complete STEP/USD source datasets are not available in this GitHub
  audit runtime. Run `python scripts/simulation/audit_archived_step_usd_mapping.py`
  to reproduce the 224/224 result with no CAD/pxr dependencies.
- Committed P3 scene report shows 282 mesh instances and 455,255 points
  in both cases; stage `metersPerUnit=1`, Z up, scale 0.001 produced
  measured 0.580000028 x 0.372995715 x 0.802851662 metres.
  Transformed-point and BBoxCache sizes agree. This verifies reference
  scaling/composition, **not** physics, chassis dimensions, collision,
  mass, drive constraints or articulated success.
- P2 report records converter exit 0, 282 meshes, output
  58,659,744 bytes, SHA-256 `3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7`.
  The large CAD/USD files are remote-only and cannot be independently
  hashed from this GitHub branch.
- P1 attributes and traces are committed; no source STEP file is included
  here for independently rerunning a 1,070,664-entity census.

## Code defects found and corrected in the isolated audit branch

### 1. Min-side inequality in the "sandwich" geometry gate was reversed

Intended lower containment:
`mesh_min <= vertex_min && mesh_max >= vertex_max` per axis.
Old source incorrectly penalized `vertex_min - mesh_min` instead of
`mesh_min - vertex_min`.

Consequences: a mesh correctly extending below a curved solid's vertex
extremum could be **falsely rejected**; a mesh missing the minimum
vertex extremum could be **falsely accepted** if its max extremum
contained the vertex max. Both are covered by separate CPU regression
tests, plus a 2.5 mm threshold boundary test.

All 224 previously accepted pairs still satisfy the corrected test, based
on the **archived rounded bboxes**. A complete CAD re-extraction is needed
to discover which, if any, of the original 58 unmatched become admissible.

### 2. Greedy matching can undercount matchable ambiguous parts

Old matching sorted candidate edges by heuristic `pair_cost` and
immediately consumed a solid/mesh. A flexible solid could consume the
only admissible mesh of a constrained solid; that leaves a feasible
perfect matching undiscovered. The audit branch uses a deterministic
maximum-cardinality bipartite augmenting-path matching within each
name key, preferring low-cost candidate edges.

This maximizes **the number of admissible pairs**. It does **not**
establish a globally minimum-cost assignment or recover semantic identity
for identical-name parts. `NONE` and duplicate names remain
`CANDIDATE`, never automatically `VERIFIED`.
A two-solid/two-mesh regression reproduces the greedy failure.

### 3. P4 rejections were not exhaustively explainable

Archived P4 has `rejected_pair_count=172` but retains only the
first 50 records in `rejected_pairs`. The 58 `unmatched_solids`
entries record the closest key ratio, not each explicit geometric
rejection reason. The correction adds per-unmatched solid
`unmatched_reason`, same-key candidate/admissible counts, and
`rejected_pairs_truncated` metadata. The full 58-item outcomes
must be regenerated remotely; old evidence remains immutable.

### 4. Original metre-scene acceptance was too permissive

Original scene verdict checked that at least one mesh existed and that
the global bounding envelope matched, but did not require exactly
282 mesh instances, both BBox methods to agree, or valid metre/Z-stage
metadata. A scene could lose parts yet still pass the global envelope.
The revised CPU-testable acceptance gate requires both scenes:
exactly the expected mesh count (default 282), BBox agreement, opened,
`metersPerUnit=1`, `upAxis=Z`, all four inspected UsdPhysics APIs
present with count zero, correct 0.001 ratio, and expected B envelope.

**Important:** four zero UsdPhysics API counts are *not* a complete
inventory of PhysX-specific APIs or a simulation-ready certification.

## Remaining limitations and high-priority review areas

1. **2.5 mm tolerance is a heuristic, not independently calibrated.**
   The source says tessellation `tessLOD=2` justifies 2.5 mm, but no
   per-face chord tolerance from the actual converter was inspected.
   The combined tolerance is `max(2.5 mm, 1% of extent)`, so a large
   part may tolerate more than 2.5 mm. Run a sensitivity sweep and a
   STEP-vs-mesh physical point-distance test before claiming strict
   CAD identity.
2. Upper `closure` point+circle-radius bbox is a conservative geometry
   heuristic. A rigorous guarantee for all STEP analytic/NURBS/placement
   constructions has not been established from the implementation.
   If a corrected-sandwich failure remains, do not immediately label
   the converter bad without checking coordinate frames and surface bounds.
3. Equal 282 counts and unique normalized name keys do not prove true
   one-to-one semantic identity. `VERIFIED` is a **local algorithm grade**
   indicating two checks passed, not an independently certified CAD
   identity. In particular ASCII-only name normalization loses Chinese
   and lowercase distinctions.
4. A maximally admissible matching of same-key AABBs still needs
   visual/CAD topology evidence for duplicate parts. Keep those as
   `CANDIDATE` even if they are newly matched.
5. GitHub evidence completeness: `MANIFEST.sha256` lists **17** relative
   evidence paths, but only **11** are present in the committed Git tree.
   Missing: `p0/pytest_full.log`, `p1/attribute_run.log`,
   `p1/trace_run.log`, `p2/convert.log`,
   `p3/metric_scene.log`, `p4/p4_mapping.log`.
   Hashes in the manifest are not independently verifiable without
   their contents. The reported `1 failed, 142 passed, 3 skipped`
   belongs to the remote execution summary; its full pytest log is
   not currently verifiable from this GitHub archive.
6. The reported failing audit test
   `test_build_report_aggregates_groups_and_caveats` asserts exactly
   3 caveats despite production now emitting 4. It remains separately
   unresolved: confirm the 4th caveat is intended before changing any
   production logic or test assertion. This audit branch deliberately
   does not alter that unrelated test.

## Validation and rerun requirements

Audit-branch GitHub CI (CPU) compiles source and runs the amended
STEP→USD matching and metre-scene tests, where `pxr`-dependent tests
are skipped outside Isaac Sim. It also rechecks the 224 archived pairs.
Read the actual run conclusion before claiming success.

**Remote operator only**: rerun full
`scripts/simulation/map_step_solids_to_usd_meshes.py` against the same
STEP and the exact USD identified in P2, into a fresh output directory.
Preserve the prior 224/58 mapping and produce a per-EID diff of
`ADDED / REMOVED / CHANGED`, new grades and geometric rejections.
Then rerun the stricter metric scene runner, recording actual mesh count,
bbox cross-checks and stage metadata. Do not call any new mapping 282/282
without inspecting all 58 originally unmatched solids.

No CAD conversion or new Isaac GPU job has been started by this audit,
and neither historical evidence nor the developer branch was overwritten.
