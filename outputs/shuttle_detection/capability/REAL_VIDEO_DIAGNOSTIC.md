# REAL_VIDEO — diagnostic continuity result (NO recall is claimed)

Date: 2026-09-15  |  Model: `baseline_synthetic_only`  |  Control: untrained COCO yolo26s

> **This is a DIAGNOSTIC, not an acceptance measurement.** The 150 frames have no per-frame
> ground truth, so recall, miss-streak and reacquisition CANNOT be computed. Spec 04 permits a
> diagnostic continuity test in exactly this situation and forbids presenting it as recall.

## 1. Why there is no ground truth, re-verified rather than inherited

An earlier phase recorded that only one of four clips is a real match and that the shuttle "cannot
be localised at broadcast scale". I re-checked that claim on the actual frames instead of trusting
the note:

- The only match-like clip is named `ball_badminton_ccby3_1280`. Ball badminton is played with a
  BALL, not a shuttlecock.
- Zooming the net region at 2x across four frames of that clip shows players holding badminton
  rackets, but **no projectile visible in any of the four sampled frames**.
- The frames are 1280x720 wide shots; at that scale a shuttlecock is a few pixels at best.

So the blocker is not merely "no labels" - it is that the available footage does not reliably
contain an identifiable shuttlecock at all. Both halves are recorded.

## 2. The diagnostic result

| Clip | Frames | Frames with any detection | Max confidence | Mean candidates/frame |
|---|---|---|---|---|
| `ball_badminton_ccby3_1280` | 48 | **15 (31%)** | 0.457 | 0.40 |
| `bim_cc0_1440` | 6 | **1 (17%)** | 0.109 | 0.33 |
| `eurogames_ccbysa4_619` | 48 | **32 (67%)** | 0.478 | 1.04 |
| `school_cc0_320` | 48 | **17 (35%)** | **0.745** | 0.50 |
| control: COCO yolo26s, any class | 150 | 150 (100%) | 0.948 | 17.0 - 27.0 |

The COCO control row is there to show what "a model that finds everything" looks like on the same
frames - 17 to 27 boxes per frame. Our model emits well under one candidate per frame on average,
so it is not simply firing indiscriminately.

## 3. What can and cannot be concluded

**Cannot be concluded:** whether any of those detections is a shuttlecock. Without ground truth,
"67% of frames have a detection" is not recall and must never be reported as such.

**Can be noted as a suspicion only:** the model was measured to respond to small white blobs, and
these frames are outdoor courts full of players in white clothing, which spec 06 lists as a known
false-positive source (衣物白点). The prudent reading is that the observed detections are MORE
likely false positives than shuttlecocks until labels exist - but that is a hypothesis, not a
measurement, and it is labelled as one.

**What would unblock this:** per-frame ground truth on footage that actually contains a shuttlecock,
which is the same camera/private-footage dependency already recorded for the REAL_VIDEO domain.

## 4. Reproduction

```bash
# 150 frames from 4 free-licensed clips; conf 0.05; imgsz 640; device 0
#   outputs/shuttle_capability/real_video/frames/*.png
#   model: baseline_synthetic_only/weights/best.pt            -> diagnostic_baseline.json
#   control: yolo26s.pt with any class                        -> diagnostic_coco_any.json
```
Per-frame counts are in `/tmp/video_baseline.json` and `/tmp/video_coco_any.json` on the host; the
contact sheets used for the visual check are `video_frames_sheet.jpg` and `video_net_zoom.jpg`.
