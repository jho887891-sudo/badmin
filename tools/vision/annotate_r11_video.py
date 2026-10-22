"""Manual, model-blind R11 shuttlecock video annotation.

Only a human keypress can label a frame positive/negative/unscorable.
Skipped frames remain absent from GT JSONL, never automatic negatives.

Run on a computer with a real display and OpenCV:
  python tools/vision/annotate_r11_video.py --video doubles0.mp4 \\
      --output doubles0_human_gt.jsonl --annotator operator1

Mouse drag: bbox; x: undo; p: positive; n: negative; u: unscorable;
s: skip without label; q/ESC: save and quit. Existing labels are immutable.
This tool does not import/load YOLO or show any model suggestion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


class AnnotationError(ValueError):
    """Unsafe or inconsistent annotation or source-video state."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_timestamp(frame_id: int, pts_ms: float, fps: float) -> tuple[float, str]:
    """Mirror the existing R11 VideoFrameSource timestamp convention."""
    if (type(frame_id) is not int or frame_id < 0 or
            not math.isfinite(pts_ms) or pts_ms < 0 or
            not math.isfinite(fps) or fps <= 0):
        raise AnnotationError("invalid frame index, PTS or video fps")
    if pts_ms > 0:
        return pts_ms / 1000.0, "PTS_VIDEO"
    return frame_id / fps, "ASSUMED_FPS"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         delete=False, newline="\n") as stream:
            temp_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _validate_sha(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value):
        raise AnnotationError("video_sha256 must be lowercase 64-char hex")
    return value


class ReviewStore:
    """Immutable reviewed frames, source SHA-locked and atomically checkpointed.

    The sidecar is an operator *claim* about video identity and participation;
    it is not proof of independent second-person verification or held-out data.
    """

    def __init__(self, output: str | Path, *, video_sha256: str, width: int,
                 height: int, frame_count: int, fps: float, annotator: str) -> None:
        self.output = Path(output)
        self.meta_path = self.output.with_suffix(self.output.suffix + ".meta.json")
        _validate_sha(video_sha256)
        if (any(type(x) is not int or x < 1 for x in (width, height, frame_count)) or
                width < 2 or height < 2 or not math.isfinite(fps) or fps <= 0):
            raise AnnotationError("invalid source video metadata")
        if not isinstance(annotator, str) or not annotator.strip():
            raise AnnotationError("nonempty manual annotator identifier required")
        self.metadata = {
            "schema": "r11_manual_video_annotation_meta/v1",
            "video_sha256": video_sha256,
            "width_px": width, "height_px": height,
            "frame_count": frame_count, "fps": float(fps),
            "annotation_method": "MANUAL_BLIND_NO_MODEL_BOXES",
            "annotator_self_declared": annotator.strip(),
            "held_out_split_verified": False,
            "independent_annotation_verified": False,
        }
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.rows: dict[int, dict] = {}
        if self.output.exists() and not self.meta_path.exists():
            raise AnnotationError("metadata missing for existing GT; refusing to mix videos")
        if self.meta_path.exists():
            if not self.output.exists():
                raise AnnotationError("saved GT missing while metadata exists; refusing silent reset")
            try:
                stored = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise AnnotationError("corrupt annotation metadata") from exc
            # Annotation may resume only if the same exact source AND operator.
            if stored != self.metadata:
                raise AnnotationError("source mismatch with existing annotation metadata")
            if self.output.exists():
                try:
                    for line in self.output.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        fid = item["frame_id"]
                        if fid in self.rows:
                            raise AnnotationError("corrupt duplicate frame in saved GT")
                        self._check_saved(item)
                        self.rows[fid] = item
                    self._check_time_order(self.rows)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise AnnotationError("corrupt saved annotation GT") from exc
        else:
            if self.output.exists():
                raise AnnotationError("metadata missing for existing GT")
            _atomic_write(self.meta_path, json.dumps(self.metadata, sort_keys=True, indent=2) + "\n")
            # An empty GT file is intentional: missing GT on resume then signals data loss.
            _atomic_write(self.output, "")

    def _bbox(self, box: tuple | list) -> list[float]:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise AnnotationError("manual bbox must be xyxy[4]")
        try:
            a = [float(v) for v in box]
        except (ValueError, TypeError) as exc:
            raise AnnotationError("nonnumeric manual bbox") from exc
        if not all(math.isfinite(v) for v in a):
            raise AnnotationError("nonfinite manual bbox")
        x1, y1, x2, y2 = a
        if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
            raise AnnotationError("manual bbox out of image or degenerate")
        return a

    def _check_saved(self, row: dict) -> None:
        if not isinstance(row, dict):
            raise AnnotationError("saved annotation must be mapping")
        fid = row["frame_id"]
        if type(fid) is not int or not 0 <= fid < self.frame_count:
            raise AnnotationError("invalid saved frame id")
        t = row["timestamp_s"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise AnnotationError("invalid saved timestamp")
        if row.get("image_size") != {"width_px": self.width, "height_px": self.height}:
            raise AnnotationError("saved GT image size mismatch")
        objects = row.get("objects")
        if not isinstance(objects, list):
            raise AnnotationError("invalid saved GT objects")
        boxes = []
        for obj in objects:
            if not isinstance(obj, dict):
                raise AnnotationError("invalid saved GT object")
            boxes.append(self._bbox(obj.get("bbox_xyxy_px")))
        if (row.get("status") not in ("labeled", "unscorable") or
                (row["status"] == "unscorable" and boxes)):
            raise AnnotationError("invalid saved GT status")

    @staticmethod
    def _check_time_order(rows: dict[int, dict]) -> None:
        prev_t = -1.0
        for fid in sorted(rows):
            t = rows[fid]["timestamp_s"]
            if t <= prev_t:
                raise AnnotationError("time order mismatch between reviewed frames")
            prev_t = t

    @property
    def reviewed_frames(self) -> set[int]:
        return set(self.rows)

    def record(self, frame_id: int, timestamp_s: float, kind: str,
               boxes: list | tuple, *, timestamp_source: str = "ASSUMED_FPS") -> bool:
        """A single explicit manual decision; skip means no saved GT row."""
        if kind not in ("positive", "negative", "unscorable", "skip"):
            raise AnnotationError("unknown manual annotation decision")
        if type(frame_id) is not int or not 0 <= frame_id < self.frame_count:
            raise AnnotationError("frame_id outside video range")
        if (isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or
                not math.isfinite(timestamp_s) or timestamp_s < 0):
            raise AnnotationError("invalid capture timestamp")
        if timestamp_source not in ("PTS_VIDEO", "ASSUMED_FPS"):
            raise AnnotationError("invalid timestamp provenance")
        if frame_id in self.rows:
            if kind == "skip":
                return False
            raise AnnotationError("already reviewed: saved decisions are immutable")
        if kind == "skip":
            return False
        if not isinstance(boxes, (tuple, list)):
            raise AnnotationError("bbox list required")
        valid = [self._bbox(box) for box in boxes]
        if kind == "positive" and not valid:
            raise AnnotationError("positive review requires manually drawn bbox")
        if kind != "positive" and valid:
            raise AnnotationError("negative/unscorable review cannot contain boxes")
        row = {
            "frame_id": frame_id, "timestamp_s": float(timestamp_s),
            "timestamp_provenance": timestamp_source,
            "status": "unscorable" if kind == "unscorable" else "labeled",
            "image_size": {"width_px": self.width, "height_px": self.height},
            "objects": [{"bbox_xyxy_px": coords} for coords in valid],
        }
        proposal = {**self.rows, frame_id: row}
        self._check_time_order(proposal)
        text = "".join(json.dumps(proposal[fid], sort_keys=True, ensure_ascii=False) + "\n"
                       for fid in sorted(proposal))
        _atomic_write(self.output, text)
        self.rows = proposal
        return True

    def coverage(self) -> dict:
        positive = sum(bool(r["objects"]) for r in self.rows.values())
        negative = sum(r["status"] == "labeled" and not r["objects"] for r in self.rows.values())
        unscorable = sum(r["status"] == "unscorable" for r in self.rows.values())
        return {
            "expected_frames": self.frame_count,
            "reviewed_frames": len(self.rows),
            "positive_frames": positive,
            "negative_frames": negative,
            "unscorable_frames": unscorable,
            "unreviewed_frames": self.frame_count - len(self.rows),
        }





def display_to_source(x: int, y: int, *, scale: float,
                      width: int, height: int) -> tuple[int, int]:
    """Convert fixed-size OpenCV autosize display coords to source pixel coords."""
    if not math.isfinite(scale) or scale <= 0:
        raise AnnotationError("invalid display-to-source scale")
    return (min(width - 1, max(0, int(round(x / scale)))),
            min(height - 1, max(0, int(round(y / scale)))))


def make_pixel_magnifier(image, u: int, v: int, *, radius: int = 24,
                         zoom: int = 6):
    """Model-blind nearest-neighbour enlarged crop for reviewing 4-6px objects.

    It uses the raw source pixels only, so the preview contains no detector
    suggestion or learned super-resolution that could create false structure.
    """
    import numpy as np
    if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise AnnotationError("magnifier requires HxWxC source image")
    if (type(radius) is not int or radius < 1 or
            type(zoom) is not int or zoom < 1):
        raise AnnotationError("magnifier radius/zoom must be positive integers")
    height, width = image.shape[:2]
    u = min(width - 1, max(0, int(u)))
    v = min(height - 1, max(0, int(v)))
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)),
                    mode="edge")
    patch = padded[v:v + 2 * radius, u:u + 2 * radius]
    return np.repeat(np.repeat(patch, zoom, axis=0), zoom, axis=1)


def iter_decoded_frames(cap, *, start: int, end: int, stride: int):
    """Decode sequentially from 0: never trust approximate codec seek identity.

    Yield (absolute_frame_id, decoded_image) for selected positions only.
    Keeping the full decode sequence also means timestamps from CAP_PROP_POS_MSEC
    correspond to the frame that was actually read rather than an assumed seek.
    """
    if (type(start) is not int or start < 0 or
            type(end) is not int or end <= start or
            type(stride) is not int or stride < 1):
        raise AnnotationError("invalid sequential review frame range")
    for frame_id in range(end):
        ok, image = cap.read()
        if not ok:
            raise AnnotationError(f"video stopped decoding at frame {frame_id}")
        if frame_id >= start and (frame_id - start) % stride == 0:
            yield frame_id, image


def review_video(video_path: Path, store: ReviewStore, *, start: int = 0,
                 end: int | None = None, stride: int = 1, display_scale: float = 1.0) -> dict:
    """OpenCV interactive review, requiring actual user keypress for each GT row."""
    import cv2
    if type(start) is not int or start < 0 or type(stride) is not int or stride <= 0:
        raise AnnotationError("start/stride invalid")
    if end is None:
        end = store.frame_count
    if type(end) is not int or not start < end <= store.frame_count:
        raise AnnotationError("end must be exclusive and within video")
    if not math.isfinite(display_scale) or not 0.1 <= display_scale <= 2:
        raise AnnotationError("invalid display scale")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnnotationError("cannot open video for manual review")
    window = "R11 HUMAN BLIND GT (p/n/u/s/q, mouse drag, x undo)"
    ui = {"boxes": [], "down": None, "cursor": None,
          "hover": (store.width // 2, store.height // 2), "magnify": True}
    view_width = max(2, int(round(store.width * display_scale)))
    view_height = max(2, int(round(store.height * display_scale)))

    def to_native(x, y):
        return display_to_source(x, y, scale=display_scale,
                                 width=store.width, height=store.height)

    def mouse(event, x, y, flags, userdata):
        pt = to_native(x, y)
        ui["hover"] = pt
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["down"] = pt
            ui["cursor"] = pt
        elif event == cv2.EVENT_MOUSEMOVE and ui["down"] is not None:
            ui["cursor"] = pt
        elif event == cv2.EVENT_LBUTTONUP and ui["down"] is not None:
            px, py = ui["down"]
            x1, x2 = sorted((px, pt[0]))
            y1, y2 = sorted((py, pt[1]))
            if x2 > x1 and y2 > y1:
                ui["boxes"].append((x1, y1, min(store.width, x2 + 1), min(store.height, y2 + 1)))
            ui["down"] = None
            ui["cursor"] = None

    try:
        # WINDOW_AUTOSIZE preserves 1:1 mouse coords with the resized image.
        # WINDOW_NORMAL could allow arbitrary user resizing and corrupt GT xyxy.
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse)
        for frame_idx, image in iter_decoded_frames(cap, start=start, end=end, stride=stride):
            if frame_idx in store.reviewed_frames:
                continue
            if image.shape[1] != store.width or image.shape[0] != store.height:
                raise AnnotationError("video decoded geometry differs from source manifest")
            pts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            ts, clock_src = frame_timestamp(frame_idx, pts_ms, store.metadata["fps"])
            ui["boxes"].clear()
            while True:
                display = image.copy()
                for coords in ui["boxes"]:
                    x1, y1, x2, y2 = coords
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 80), 2)
                if ui["down"] is not None and ui["cursor"] is not None:
                    cv2.rectangle(display, ui["down"], ui["cursor"], (0, 255, 255), 1)
                if ui["magnify"]:
                    inset = make_pixel_magnifier(image, *ui["hover"])
                    ih = min(display.shape[0], inset.shape[0])
                    iw = min(display.shape[1], inset.shape[1])
                    display[0:ih, display.shape[1] - iw:] = inset[:ih, :iw]
                    cv2.rectangle(display, (display.shape[1] - iw, 0),
                                  (display.shape[1] - 1, ih - 1), (255, 255, 0), 2)
                msg = f"frame {frame_idx}/{store.frame_count}  t={ts:.3f}({clock_src})  boxes={len(ui['boxes'])}"
                cv2.rectangle(display, (0, 0), (min(store.width, 950), 58), (0, 0, 0), -1)
                cv2.putText(display, msg, (7, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display, "p=positive n=no-ball u=unscorable s=skip x=undo m=zoom q=quit",
                            (7, 44), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1,
                            cv2.LINE_AA)
                if display_scale != 1:
                    display = cv2.resize(display, (view_width, view_height))
                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("m"):
                    ui["magnify"] = not ui["magnify"]
                    continue
                if key == ord("x"):
                    if ui["boxes"]:
                        ui["boxes"].pop()
                    continue
                if key in (27, ord("q")):
                    return store.coverage()
                decision = {ord("p"): "positive", ord("n"): "negative",
                            ord("u"): "unscorable", ord("s"): "skip"}.get(key)
                if decision is None:
                    continue
                try:
                    store.record(frame_idx, ts, decision,
                                 list(ui["boxes"]), timestamp_source=clock_src)
                except AnnotationError as exc:
                    print(f"REJECTED frame {frame_idx}: {exc}")
                    continue
                print(f"frame {frame_idx}: {decision}, saved={decision != 'skip'}")
                break
        return store.coverage()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manual blind GT; never loads YOLO")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator", type=str, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--display-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise AnnotationError("video not found or unsupported codec")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            store = ReviewStore(args.output, video_sha256=sha256_file(args.video),
                                width=width, height=height, frame_count=frame_count,
                                fps=fps, annotator=args.annotator)
        finally:
            cap.release()
        outcome = review_video(args.video, store, start=args.start_frame,
                               end=args.end_frame, stride=args.stride,
                               display_scale=args.display_scale)
        print(json.dumps({"schema": "r11_manual_review_summary/v1", "output": str(store.output),
                          "source_video_sha256": store.metadata["video_sha256"],
                          "coverage": outcome, "human_verified_independently": False}, indent=2))
        return 0
    except (AnnotationError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
