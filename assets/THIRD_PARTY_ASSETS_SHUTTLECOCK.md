# Third-Party Asset Entry — Shuttlecock Visual

- Asset: Badminton Racket And Shuttlecock (Low Poly)
- Intended extracted component: shuttlecock only
- Author: game_travel
- Platform: Sketchfab
- Source UID: 795e4cca2e544d7ab86a8e5c7ded2362
- URL: https://sketchfab.com/3d-models/badminton-racket-and-shuttlecock-low-poly-795e4cca2e544d7ab86a8e5c7ded2362
- License: CC Attribution (CC-BY-4.0)
- Public listing: downloadable, approximately 5.2k triangles / 2.7k vertices for the combined model
- Local target: assets/third_party/shuttlecock_visual.usd
- SHA256: see conversion record below
- Conversion requirements: shuttlecock only; meters; preserve visual proportions; align local frame to project convention; do not use visual mesh as the sole high-cost physics collision mesh.

## Conversion record (2026-09-13)

**Downloaded artefact (original, unmodified)**
```
assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb
size   : 308444 bytes
sha256 : 8f6c2baac0bbd2f2f1aceaa295b06fd74a10aed59de8686d2d76cdcc2e22bfa5
format : glTF binary (Sketchfab autoconverted, texture 512 px embedded)
```

**Extraction** (tool: `tools/glb_shuttle_extract.py`, pxr only — no Blender)
- GLB node audit: `Gp_Shuttle` = `Obj_Feather` + `Obj_Cork` (kept) ；`Obj_Racket`/`Obj_Racket.001`
  (+ `.Strings`) 为球拍（丢弃）。
- Kept triangles: feather 1920 + cork 1502 = **3422 tri**（丢弃球拍 2×868 tri）。
- No rescale applied (source already in metres and in project frame convention).

**Produced visual**
```
assets/third_party/shuttlecock_visual.usd      sha256 e47066655b7f09a013b27bcfdd869ec86f2c845b4a62a268721d2fbb473c4227
```
- defaultPrim `/ShuttlecockVisual`；metersPerUnit 1.0；upAxis Z
- frame 约定：原点在球托平面中心，+Z 由球托指向裙边（与 `configs/shuttlecock.yaml` 一致）
- geometry check (world bbox):
  - feather: x/y ±0.0309 m (裙尖 Ø **61.8 mm**，与配置 61.804 mm 一致) ，z 0.0006–0.0780 m
  - cork: x/y ±0.0178 m (Ø **35.6 mm**)，z 0.0002–0.0411 m
  - 总高 78.0 mm（配置参考总长 ≈ 79 mm）
- material: 按 GLB 材质定义生成 —— 材质 0 `White` = **baseColorFactor 白色 (1,1,1)**、roughness 0.6、metallic 0、doubleSided=True
  - 修正记录：首版导出误把 GLB 中属于**球拍拍线**的贴图（材质 1 `Strings`）绑到了羽毛球上，渲染成黑色；已改为按材质索引分别生成（羽毛球无贴图）。渲染复核为白色。

**Final shuttlecock asset**
```
assets/shuttle/shuttlecock.usd   sha256 108ceb1aa0ca3084673b71338bfb2b598b237ef6e74b5fbbe4259ae59830f7fb
```
- `/Shuttlecock` = Visual (External reference -> shuttlecock_visual.usd) + Physics (CorkCollider + 16 SkirtColliders) + Frames
- mass 0.005000 kg，COM [0, 0, 0.011978] m，inertia [3.543e-06, 3.543e-06, 1.341e-06] kg·m²
- `project:contactCalibrationStatus = REQUIRES_PAIR_CALIBRATION`
- **Note**: the low-poly cork is slightly wider than the BWF reference value used by the physics proxy
  (visual Ø35.6 mm vs config cork Ø26.5 mm). Physics still uses the configured proxy; visual/collision
  are intentionally decoupled. Recorded here for traceability.
- `shuttlecock_physics_proxy.obj` mentioned in `docs/SHUTTLECOCK.md` was **not** delivered in the download
  set; the collision proxy is generated procedurally from `shuttlecock.yaml` instead.
## Visual verification (2026-09-13)

Rendered standalone previews with the project camera pipeline (Isaac RTX, headless):

| image | content |
|---|---|
| `outputs/evidence/shuttle_preview_side.png` | 侧视 1400×1000：球托在下、羽毛裙在上，16 片羽毛 + 羽毛杆笼，白色材质 |
| `outputs/evidence/shuttle_preview_top.png` | 俯视 1400×1000 |

Checked visually: **scale / orientation / material / mesh OK**（纹理不适用于羽毛球：GLB 唯一贴图属于拍线）。
Renderer: `tools/render_shuttle_preview.py`；日志 `outputs/runs/shuttle_preview2.log`。
