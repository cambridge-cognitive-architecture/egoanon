import cv2
import numpy as np
import pytest

from egoanon import Region, apply_level, load_plan, render_video, save_plan, scan_video
from egoanon.demo import N, colour_detector, write_clip
from egoanon.detect import ModelDetector, person_regions
from egoanon.render import LEVELS
from egoanon.tracking import IouTracker, interpolate_dropouts
from egoanon.video import is_anonymised


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("size", [4, 8, 12, 20, 64, 150])
@pytest.mark.parametrize("source", ["face", "screen"])
def test_centre_is_covered_at_every_size(level, size, source):
    checker = ((np.indices((400, 400)).sum(0) % 2) * 255).astype(np.uint8)
    image = np.dstack([checker] * 3)
    out = apply_level(image, [Region(200 - size / 2, 200 - size / 2, size, size, source)],
                      level).astype(int)
    assert abs(out[200, 200, 0] - out[200, 201, 0]) <= 8


@pytest.mark.parametrize("level", ["silhouette", "boxes"])
def test_abstract_levels_keep_no_pixels(level):
    regions = [Region(40, 40, 50, 60)]
    a = np.zeros((200, 200, 3), np.uint8)
    b = np.full((200, 200, 3), 99, np.uint8)
    assert np.array_equal(apply_level(a, regions, level), apply_level(b, regions, level))


def test_track_survives_a_dropout():
    tracker = IouTracker()
    first = tracker.step([Region(10, 10, 50, 50)])
    tracker.step([])
    again = tracker.step([Region(16, 10, 50, 50)])
    assert first[0].track_id == again[0].track_id
    assert interpolate_dropouts({0: first, 2: again})[1][0].x == 13


def test_wide_person_covered_but_not_own_arm():
    assert person_regions((100, 100, 400, 250), 640, 480)[0].w == 300
    assert person_regions((100, 350, 500, 480), 640, 480) == []


def test_missing_weights_stop_the_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelDetector(tmp_path / "none.pt")


def test_detector_error_stops_the_scan(tmp_path):
    clip = tmp_path / "a.mp4"
    write_clip(clip)

    def broken(_):
        raise RuntimeError("detector failed")
    with pytest.raises(RuntimeError):
        scan_video(clip, broken)


def test_scan_render_and_guard(tmp_path):
    clip, out = tmp_path / "a.mp4", tmp_path / "b.mp4"
    write_clip(clip)
    plan = scan_video(clip, colour_detector, stride=4)
    save_plan(plan, tmp_path / "plan.json")
    plan = load_plan(tmp_path / "plan.json")
    assert all(plan.regions_at(i) for i in range(N))
    assert render_video(clip, plan, out)["n_frames"] == N
    assert is_anonymised(out) and not is_anonymised(clip)
    plan.frame_to -= 1
    with pytest.raises(ValueError):
        render_video(clip, plan, tmp_path / "c.mp4")
    with pytest.raises(ValueError):
        plan.regions_at(N - 1)
