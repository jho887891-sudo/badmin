"""RED/GREEN tests for manual, model-blind R11 video ground truth."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.vision.annotate_r11_video import (  # noqa: E402
    AnnotationError, ReviewStore, frame_timestamp, sha256_file, iter_decoded_frames,
)


def store(tmp_path, video_sha="a" * 64, fps=30.):
    return ReviewStore(
        tmp_path / "doubles0_gt.jsonl", video_sha256=video_sha,
        width=1280, height=720, frame_count=336, fps=fps,
        annotator="manual_review_operator",
    )


def test_positive_negative_and_unscorable_have_different_states(tmp_path):
    s = store(tmp_path)
    s.record(0, 0.0, "positive", [(100, 80, 106, 86)])
    s.record(1, 1 / 30, "negative", [])
    s.record(2, 2 / 30, "unscorable", [])
    rows = [json.loads(line) for line in s.output.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["labeled", "labeled", "unscorable"]
    assert [len(row["objects"]) for row in rows] == [1, 0, 0]
    assert [row["frame_id"] for row in rows] == [0, 1, 2]
    assert s.coverage() == {
        "expected_frames": 336, "reviewed_frames": 3,
        "positive_frames": 1, "negative_frames": 1,
        "unscorable_frames": 1, "unreviewed_frames": 333,
    }


def test_skip_is_not_labeled_negative(tmp_path):
    s = store(tmp_path)
    assert not s.record(0, 0, "skip", [])
    assert s.coverage()["reviewed_frames"] == 0
    assert s.output.exists() and s.output.read_text() == ""


def test_resume_only_same_video_and_geometry(tmp_path):
    s = store(tmp_path)
    s.record(0, 0, "positive", [(2, 3, 7, 9)])
    again = store(tmp_path)
    assert again.reviewed_frames == {0}
    with pytest.raises(AnnotationError, match="source mismatch"):
        store(tmp_path, video_sha="b" * 64)
    with pytest.raises(AnnotationError, match="source mismatch"):
        store(tmp_path, fps=25.0)
    assert json.loads(s.output.read_text().splitlines()[0])["objects"][0]["bbox_xyxy_px"] == [2.0, 3.0, 7.0, 9.0]


def test_duplicate_review_cannot_be_overwritten(tmp_path):
    s = store(tmp_path)
    s.record(0, 0.0, "negative", [])
    with pytest.raises(AnnotationError, match="already reviewed"):
        s.record(0, 0.0, "positive", [(1, 1, 3, 3)])
    assert len(s.output.read_text().splitlines()) == 1


@pytest.mark.parametrize("kind,boxes", [
    ("positive", []), ("negative", [(1, 1, 4, 4)]),
    ("unscorable", [(1, 1, 4, 4)]), ("positive", [(-1, 2, 3, 5)]),
    ("positive", [(100, 100, 100, 120)]),
    ("positive", [(1275, 1, 1290, 10)]),
    ("positive", [(1, 2, float("nan"), 4)]),
    ("undetermined", []),
])
def test_invalid_manual_status_and_geometry_fails_closed(tmp_path, kind, boxes):
    s = store(tmp_path)
    with pytest.raises(AnnotationError):
        s.record(0, 0., kind, boxes)
    assert s.output.exists() and s.output.read_text() == ""


@pytest.mark.parametrize("frame,clock", [(-1, 0.), (336, 12.), (0, -1.), (0, math.nan)])
def test_invalid_frame_time_not_written(tmp_path, frame, clock):
    s = store(tmp_path)
    with pytest.raises(AnnotationError):
        s.record(frame, clock, "negative", [])
    assert s.output.exists() and s.output.read_text() == ""


def test_out_of_order_review_sorts_and_preserves_real_time_order(tmp_path):
    s = store(tmp_path)
    s.record(12, .4, "negative", [])
    s.record(3, .1, "negative", [])
    rows = [json.loads(line) for line in s.output.read_text().splitlines()]
    assert [r["frame_id"] for r in rows] == [3, 12]
    with pytest.raises(AnnotationError, match="time order"):
        s.record(4, .8, "negative", [])


def test_rejects_corrupted_existing_gt_and_missing_manifest(tmp_path):
    s = store(tmp_path)
    s.record(0, 0, "negative", [])
    s.output.write_text("not JSON\n")
    with pytest.raises(AnnotationError, match="corrupt"):
        store(tmp_path)
    s.output.unlink()
    s.meta_path.unlink()
    s.output.write_text('{"frame_id":0}\n')
    with pytest.raises(AnnotationError, match="metadata missing"):
        store(tmp_path)


def test_pts_and_fps_source_labels_are_explicit():
    assert frame_timestamp(0, 0.0, 30.) == (0., "ASSUMED_FPS")
    assert frame_timestamp(5, 0.0, 25.) == (.2, "ASSUMED_FPS")
    assert frame_timestamp(5, 200.0, 25.) == (.2, "PTS_VIDEO")
    with pytest.raises(AnnotationError):
        frame_timestamp(0, -100., 30.)


def test_source_file_sha_depends_on_actual_video_bytes(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"a video")
    old = sha256_file(video)
    video.write_bytes(b"a video altered")
    assert sha256_file(video) != old


def test_sequential_decoding_never_assumes_keyframe_seek_is_exact():
    class FakeCapture:
        def __init__(self):
            self.index = 0
            self.seeks = 0
        def set(self, *args):
            self.seeks += 1
            raise AssertionError("seeking risks mismatched frame identity")
        def read(self):
            pos = self.index
            self.index += 1
            return True, {"decoded_origin_frame": pos}
    cap = FakeCapture()
    out = list(iter_decoded_frames(cap, start=3, end=7, stride=2))
    assert [(fid, img["decoded_origin_frame"]) for fid, img in out] == [(3, 3), (5, 5)]
    assert cap.seeks == 0 and cap.index == 7


def test_missing_saved_gt_with_existing_manifest_fails_closed(tmp_path):
    s = store(tmp_path)
    assert s.output.exists()
    s.record(0, 0, "negative", [])
    s.output.unlink()
    with pytest.raises(AnnotationError, match="saved GT missing"):
        store(tmp_path)


def test_magnifier_for_4px_objects_keeps_original_pixels_and_image_edges():
    import numpy as np
    from tools.vision.annotate_r11_video import make_pixel_magnifier
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[3:7, 4:8] = [11, 99, 201]
    zoom = make_pixel_magnifier(image, 6, 5, radius=4, zoom=5)
    assert zoom.shape == (40, 40, 3)
    assert (zoom[20, 20] == [11, 99, 201]).all()
    edge = make_pixel_magnifier(image, 0, 0, radius=4, zoom=5)
    assert edge.shape == (40, 40, 3)
    assert (edge[0, 0] == image[0, 0]).all()


def test_display_coordinate_conversion_never_distorts_native_box():
    from tools.vision.annotate_r11_video import display_to_source
    assert display_to_source(400, 200, scale=2.0, width=1280, height=720) == (200, 100)
    assert display_to_source(200, 100, scale=.5, width=1280, height=720) == (400, 200)
    assert display_to_source(-8, 1000, scale=1., width=1280, height=720) == (0, 719)
