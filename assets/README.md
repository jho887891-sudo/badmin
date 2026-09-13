# Third-Party Visual Assets — Review (Stage 1)

**Reviewer: visual inspection performed 2026-09-10** (1920x1080 Sketchfab thumbnails, actually viewed).
No asset downloaded. No change to the training scene / physics / PiPER Stage0.

## Visual findings (from actual image inspection)

### A — Badminton Field (23,820 tri, CC-BY, author timmy)
- Shows a **real badminton net** (black frame, mesh netting, two posts) on a **flat plain green floor**;
  clean, no gymnasium/buildings.
- **Court lines NOT visible** in the preview -> lines must still come from our own geometry.
- Materials: flat untextured colours; net has a grid texture. Very basic overall.
- Verdict: **CANDIDATE_A_PARTIAL** — good net, incomplete court (no lines, flat material);
  not clearly better than our current primitive scene overall.

### B — test badminton court (532 tri, CC-BY, author VLQ)
- Shows a **white untextured court with a full line set** (doubles/singles sidelines, short & long
  service lines, centre line) plus a **net panel with posts**.
- Extremely simple; all white, no materials; net is a translucent panel, not netting.
- Verdict: useful as a **line-layout reference / conversion smoke test**; 128-env friendly (tiny).

### C — BADMINTON COURT (1,097,620 tri, CC-BY, author proleed)
- Preview shows an **architectural steel-structure wireframe of an indoor hall**, not a court surface.
- Verdict: **not usable as a court visual** (training or presentation). Dropped.

### D — Badminton Racket And Shuttlecock (Low Poly) (5,158 tri, CC-BY, author game_travel)
- Shows **two rackets** (oval head, string grid, shaft, tapered handle) **+ one shuttlecock**
  (cork base + feather skirt); proportions plausible (shuttle ~1/6 of racket length).
- Plain white, untextured; strings/feathers are geometry (no alpha cards).
- Verdict: **best training visual candidate**; needs sub-object extraction (1 racket + 1 shuttle).

### E — Badminton Racket (13,488 tri, CC-BY, author farooq.smurf)
- Shows **two rackets with yellow/blue rims and black grips**, visible string grids, dark backdrop.
- Verdict: viable **presentation RacketVisual**; D is cheaper (5.2k) for training.

### FAB-1 (JG3D Badminton Court, paid + NoAI) / FAB-2 (Shuttlecock, paid, offers USD)
- **Not reviewable programmatically**: fab.com returns HTTP 403 to non-browser fetch (JS/login wall).
- No preview retrieved, no download, no purchase. **User decision only.** Per the NoAI notice,
  FAB-1 must not be fed to AI tooling even if purchased.

## Recommendation

**TRAINING_SCENE**: CourtVisual = our own geometry (lines from scene_layout); optional B for a
debug "clean white" look. RacketVisual = **D** (extracted single racket). ShuttleVisual = **D** (shuttle).
Physics unchanged (racket thin paddle proxy, shuttle sphere r=0.014, net thin box, 240 Hz).

**PRESENTATION_SCENE**: CourtVisual = A's net + our lines, or user-purchased FAB-1 (PBR) if licensed.
RacketVisual = **E**. ShuttleVisual = FAB-2 (has USD) or D's shuttle.

**Never**: C.

## Blocking issue found by visual check (retraction)
The current `evidence/*.png` renders are **invalid**: `cam.set_world_poses()` did not take effect in
the render, so images show only floor/ROI slabs. The projection-based `visible_objects` claims in
`screenshots.json` are **false positives** and are **retracted**. Fix: author the camera transform
directly in USD (or verify pose readback) and re-verify every shot visually before any PASS claim.

## Verified metadata (Sketchfab API)
| ID | Faces | Verts | Downloadable | License |
|---|---|---|---|---|
| A | 23,820 | 12,296 | yes (login required) | CC-BY 4.0 |
| B | 532 | 292 | yes (login) | CC-BY 4.0 |
| C | 1,097,620 | 952,209 | yes (login) | CC-BY 4.0 |
| D | 5,158 | 2,697 | yes (login) | CC-BY 4.0 |
| E | 13,488 | 6,184 | yes (login) | CC-BY 4.0 |
