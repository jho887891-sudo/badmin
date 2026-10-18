# B601-RS simulation asset package v1

Location on the GPU host: /home/T7/dgut/robot_sim/packages/badminton_b601_rs_asset_v1 (83 MB du, 59 files,
86,510,524 bytes of payload). The assets stay on the host; this directory holds the portable manifest and
the verifier, exactly as the repository rule requires (results local, resources remote).

## What the package is

| Item | Value |
| --- | --- |
| Upstream | https://github.com/Seeed-Projects/reBot-DevArm @ f01a1dc189ecb74b5435a336a68b287c146c1b60 |
| License | Seeed reBot-DevArm LICENSE, shipped as LICENSE_reBot-DevArm inside the package |
| Robot | ReBot_Arm_RS (reBot Arm B601-RS): 6 revolute + 2 prismatic fingers + 1 fixed root |
| Entry (simulation) | usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda |
| Entry (source) | urdf/ReBot_Arm_RS.urdf (mesh references are relative: ../meshes/...) |
| Conversion | Isaac Lab convert_urdf.py, --merge-joints --fix-base, gains 100 / 1.0 |

## Files in this directory

| File | Purpose |
| --- | --- |
| asset_package_manifest.json | machine-readable manifest: every file with sha256 + size, structural facts, source pin |
| asset_package_MANIFEST.sha256 | the same list in sha256sum format (59 lines), so plain sha256sum -c works |
| ../tools/verify_asset_package.py | verifier: manifest-only mode anywhere, full mode against the package directory |

## Checklist (all of it was actually run, not asserted)

| Check | Result |
| --- | --- |
| Every reference inside the layers is relative and resolves | 0 broken references |
| Asset references outside the package | none |
| Absolute host paths inside the layers | 70 matches, all of them in converter doc/comment strings (/tmp build paths); no functional dependency |
| URDF mesh closure | 42 mesh references, 0 missing, all under meshes/ |
| Joint structure | 6 revolute, 2 prismatic, 1 fixed, 6 angular drive-stiffness attributes |
| sha256sum -c on the package | MANIFEST_OK, all 59 files match |
| Independent re-hash through the verifier (--root) | see outputs/simulation/rebot_b601_rs/asset_package_verify.json |

## How to verify on another machine

    python tools/verify_asset_package.py --manifest management/rebot_b601_rs/asset_package_manifest.json
    python tools/verify_asset_package.py --manifest <package>/manifest.json --root <package>

The first form needs no assets at all and fails if the manifest drifts (schema, relative paths, counts,
structure, license presence). The second form re-hashes every file and reports any missing, resized,
rehashed or unlisted file. Exit code 0 means the handover is clean.

## Deliberately not in the package

usd_regen/ (re-conversion with gains multiplied by 57.2958, used to prove the importer's unit behaviour)
and usd_flat/ (a 103 MB flattened copy) stay in the working asset directory for the record. Both produce
identical simulation behaviour, so neither is needed to use or verify the package.

## Caveat that belongs with the package

The asset's authored drive gains are 1.745 / 0.0175 because the URDF importer divides the requested
100 / 1.0 by 57.2958. The project's control gains come from simulation/badminton_scene/scene_layout.py at
runtime; do not read the authored number as the controller gain.
