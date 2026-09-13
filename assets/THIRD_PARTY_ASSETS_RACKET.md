# Third-Party Asset Record — Racket

## Final visual candidate

- Title: Badminton Racket And Shuttlecock (Low Poly)
- Author: game_travel
- Platform: Sketchfab
- Source UID: 795e4cca2e544d7ab86a8e5c7ded2362
- URL: https://sketchfab.com/3d-models/badminton-racket-and-shuttlecock-low-poly-795e4cca2e544d7ab86a8e5c7ded2362
- License shown on source page: CC Attribution (CC-BY-4.0)
- Intended use: extract only the racket visual mesh, align/scale it to the measured physical racket, convert to USD.
- Physics use: none; dedicated measured colliders are authored separately.
- Local target: `assets/third_party/racket_visual.usd`
- Retrieval date: 2026-09-13
- Original-file SHA256: 8f6c2baac0bbd2f2f1aceaa295b06fd74a10aed59de8686d2d76cdcc2e22bfa5
  (badminton_racket_and_shuttlecock_low_poly.glb, 308444 bytes, glTF binary from Sketchfab)
- Extracted-racket SHA256: 643fb67003c6429f98ec132d2db056c83ed02811b594ad27904b618bf7c1ec29

## Extraction record (2026-09-13)

Tool: `tools/glb_racket_extract.py` (pxr only, no Blender).

- GLB audit: two rackets are present (`Obj_Racket`, `Obj_Racket.001`). **Only the first racket**
  (`Obj_Racket` 828 tri + its `Obj_Strings` 40 tri = **868 tri**) is extracted; the second racket
  and the shuttlecock are dropped.
- Materials kept from the GLB: `White` (frame, no texture) and `Strings` (string-bed, textured).
  Texture extracted to `assets/third_party/textures/racket_strings_basecolor.png`
  (sha256 42c87519eb08d52ad1fb6a5ecba1e168f32167bc3a68cebcfc8728b28242e324).
- Frame mapping: the GLB racket lies with the string-bed plane in X-Z (face normal +Y). Points and
  normals were rotated -90° about Z, `(x, y, z) -> (y, -x, z)`, so the asset now matches
  `configs/racket.yaml`: **+Z handle→head, +X face normal toward the opponent, origin at the handle end**.
- Measured world bbox after mapping:
  - frame: x ±0.0158 m, y ±0.1080 m, z 0.0000–0.6827 m
  - strings: x ±0.0010 m, y ±0.1007 m, z 0.4439–0.6753 m
- Dimension check vs config:
  - overall length **682.7 mm** vs BWF max 680 mm and cited product length 675 mm → **visual must be
    rescaled to the measured racket** before final authoring (not done: measurement pending).
  - head width 216 mm ≤ BWF 230 mm; string-bed 231.4 × 201.4 mm ≤ BWF 280 × 220 mm.
- No rescale applied at this stage by design: `racket.yaml:measured_unit` is still
  `REQUIRES_MEASUREMENT`, and the builder refuses final authoring until it is filled.

## Attribution requirement

Preserve author/title/source/license information in the project's consolidated `assets/THIRD_PARTY_ASSETS.md` and any redistribution package that includes the visual asset.
## Visual verification (2026-09-13)

Rendered standalone previews with the project camera pipeline (Isaac RTX, headless):

| image | content |
|---|---|
| `outputs/evidence/racket_preview_threequarter.png` | 整体三视角 1400×1000：白色拍框 + 拍杆 + 握柄，拍面（线床）呈深色 |
| `outputs/evidence/racket_preview_face.png` | 沿 +X（面法向）正视 1400×1000 |

Renderer: `tools/render_racket_preview.py`；日志 `outputs/runs/racket_preview4.log`。

Checked visually:
- frame / shaft / handle: **white, correct shape and proportions** ✅
- orientation: +Z handle→head, +X face normal ✅ (camera along +X sees the string-bed)
- string-bed plane renders **dark**; diagnosis: the source string-bed mesh is a thin double-sided plane whose
  vertex normals cancel out (average normal ≈ 0), so the lit shading comes out dark.
  The USD itself is correct — `st` primvar present (44 values) and the material is bound to the extracted
  `badminton_strings` texture (512×512, white ground with a black string grid).
  This is a **cosmetic** issue only (physics uses the procedural colliders, not this mesh).
  Options if a lighter string-bed look is wanted: (a) make the string-bed shading emissive from the texture,
  (b) replace it with a translucent light-grey plane, (c) keep as-is — the reference model also shows dark strings.
- **Still blocked for final authoring**: `racket.yaml:measured_unit` is `REQUIRES_MEASUREMENT`, so the visual
  has not been rescaled to the measured racket and the final racket/wrapper USD cannot be produced yet.
