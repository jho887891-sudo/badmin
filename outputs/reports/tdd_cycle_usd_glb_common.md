# TDD CYCLE REPORT - shared GLB->USD helper (tools/usd_glb_common.py)

**Process: full TDD loop (RED -> verify RED -> GREEN -> verify GREEN -> REFACTOR -> verify again)**
Mandated by the user on 2026-09-13: "以后都用 tdd 的完整路程".

## RED - write the failing test first
`tests/tools/test_usd_glb_common.py` (8 tests) written against a stub module whose functions
return deliberately empty/wrong values, so the failures are **assertion failures** and not typos.

Observed first run:
```
Ran 8 tests ... FAILED (failures=2, errors=6)
AssertionError: Lists differ: [] != ['Obj_Feather_0', 'Obj_Cork_0']
```
Every red came from "feature missing", which is the expected reason.

## GREEN - minimal implementation, slice by slice (watching green grow)
| slice | implemented | result after the slice |
|---|---|---|
| 1 | `parse_glb`, `read_accessor`, `iter_mesh_nodes` | parsing tests green, 4 errors left (remaining stubs) |
| 2 | `set_orient_compat` (Quatf/Quatd type matching -> ISSUE-005) | 2 errors left |
| 3 | `author_mesh_from_glb`, `make_material_from_glb` | **Ran 8 tests ... OK** |

## Verify GREEN - full regression, nothing else broke
```
test_badminton_court        16 OK      test_robot_config            18 OK
test_shuttlecock            14 OK      test_robot_frames            15 OK
test_shuttle_aerodynamics    7 OK      test_morph_one_kinematics     9 OK
test_racket                 16 OK      test_usd_glb_common           8 OK
test_piper_with_racket       6 OK
                                                          total 109 OK
```

## REFACTOR - both extractors now share one implementation (tests stayed green)
- `tools/glb_shuttle_extract.py` 168 -> 78 lines; `tools/glb_racket_extract.py` 144 -> 88 lines;
  both delegate to `usd_glb_common` (no duplicated GLB parsing, material building or USD authoring).
- Removes the duplicated `_set_orient`-style code that ISSUE-005 was about.

## Verify GREEN again - real extraction reproduced
```
shuttle : Obj_Feather_0 1920 tri  z[0.0006,0.0780]  |  Obj_Cork_0 1502 tri  z[0.0002,0.0411]   total 3422
racket  : Obj_Racket_0   828 tri  x[-0.0158,0.0158] (face-normal axis) z[0.0000,0.6827]
          Obj_Strings_0   40 tri  materials White + Strings(texture)                            total  868
assets/shuttle/shuttlecock.usd rebuilt successfully; test_usd_glb_common / test_shuttlecock / test_racket green
```
Byte-identical output was not required (authoring order changed); the invariant properties
(triangle counts, bounding boxes, frame mapping, material assignment) are identical.

## Process honesty notes (deviations still present)
1. The 8 tests were authored in one file at once rather than one-test-at-a-time; the strict
   one-behaviour cycle was applied to the *implementation* slices instead (documented above).
2. RED contained errors (6) as well as failures (2); they were all "feature missing" errors.
   Later cycles will create a runnable stub first so RED is failure-only.

## Frozen interfaces
`parse_glb / read_accessor / iter_mesh_nodes / set_orient_compat / author_mesh_from_glb / make_material_from_glb`
and the frame conventions of both visual assets (shuttle +Z toward skirt; racket +X face normal, +Z handle->head).
