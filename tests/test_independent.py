"""Tests for egoanon.tracking and egoanon.yolo."""

import copy
import importlib
import itertools
import os
import sys
import types

import numpy as np
import pytest

from egoanon import yolo
from egoanon.plan import Region
from egoanon.tracking import IouTracker, _iou_matrix, interpolate_dropouts


def box(x=0, y=0, w=100, h=100, **kwargs):
    return Region(x, y, w, h, **kwargs)


def ids(regions):
    return [r.track_id for r in regions]


# IoU


def test_iou_values():
    got = _iou_matrix([box()], [box(), box(x=50), box(x=100), box(x=25, y=25, w=50, h=50)])
    np.testing.assert_allclose(got, [[1, 1 / 3, 0, 0.25]])


@pytest.mark.parametrize("n, m", [(0, 0), (0, 3), (3, 0)])
def test_iou_empty(n, m):
    assert _iou_matrix([box()] * n, [box()] * m).shape == (n, m)


def test_iou_degenerate_boxes():
    flat = [box(w=0), box(h=0), box(w=0, h=0), box(w=-5)]
    with np.errstate(all="raise"):
        assert not _iou_matrix(flat, flat + [box()]).any()


# Tracker


def test_step_returns_new_regions_in_input_order():
    tracker = IouTracker()
    first = tracker.step([box(), box(x=500)])
    inputs = [box(x=505, score=0.7), box(x=900), box(x=5)]
    before = copy.deepcopy(inputs)
    out = tracker.step(inputs)
    assert [r.x for r in out] == [505, 900, 5]
    assert ids(out) == [first[1].track_id, 2, first[0].track_id]
    assert out[0].score == 0.7
    assert inputs == before
    assert all(a is not b for a, b in zip(inputs, out))


def test_callers_cannot_alter_track_state():
    tracker = IouTracker()
    tracker.step([box()])[0].x = 999
    assert ids(tracker.step([box()])) == [0]


def test_sources_never_match():
    tracker = IouTracker(min_iou=0)
    tracker.step([box(source="face")])
    assert ids(tracker.step([box(source="screen"), box(source="face")])) == [1, 0]


@pytest.mark.parametrize("width, expected", [(50, 0), (49, 1)])
def test_threshold_is_inclusive(width, expected):
    tracker = IouTracker(min_iou=0.5)
    tracker.step([box()])
    assert ids(tracker.step([box(w=width)])) == [expected]


def test_assignment_is_optimal_rather_than_greedy():
    tracker = IouTracker()
    tracker.step([box(x=0), box(x=30)])
    # Greedy would pair x=20 with the second track and lose the other match.
    assert ids(tracker.step([box(x=20), box(x=55)])) == [0, 1]


def test_rejected_pairs_do_not_displace_a_valid_match():
    tracker = IouTracker()
    tracker.step([box(x=0), box(x=103)])
    assert ids(tracker.step([box(x=48), box(x=-55)])) == [0, 2]


def test_assignment_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        old = [box(x=x) for x in rng.uniform(0, 150, 4)]
        new = [box(x=x) for x in rng.uniform(0, 150, 4)]
        iou = _iou_matrix(old, new)
        valid = np.where(iou >= 0.3, iou, 0)
        best = max(valid[range(4), p].sum() for p in itertools.permutations(range(4)))
        tracker = IouTracker()
        tracker.step(old)
        out = tracker.step(new)
        got = sum(valid[r.track_id, j] for j, r in enumerate(out) if r.track_id < 4)
        assert got == pytest.approx(best)


@pytest.mark.parametrize("max_missed", [0, 1, 20])
@pytest.mark.parametrize("extra, expected", [(0, 0), (1, 1)])
def test_track_lifetime(max_missed, extra, expected):
    tracker = IouTracker(max_missed=max_missed)
    tracker.step([box()])
    for _ in range(max_missed + extra):
        assert tracker.step([]) == []
    assert ids(tracker.step([box()])) == [expected]


def test_a_match_resets_the_miss_count():
    tracker = IouTracker(max_missed=2)
    for _ in range(4):
        assert ids(tracker.step([box()])) == [0]
        tracker.step([])
        tracker.step([])


def test_ids_are_never_reused():
    tracker = IouTracker(max_missed=0)
    seen = []
    for _ in range(5):
        seen += ids(tracker.step([box(), box(x=300)]))
        tracker.step([])
    assert seen == list(range(10))


def test_nothing_is_dropped():
    out = IouTracker().step([box(), box(), box(w=0), box(w=1, h=1, score=0.001)])
    assert ids(out) == [0, 1, 2, 3]


# Interpolation


def test_interpolation_is_linear_and_takes_lower_score():
    a = box(0, 0, 10, 20, score=0.9, track_id=4, source="screen")
    b = box(40, 80, 50, 100, score=0.6, track_id=4, source="screen")
    out = interpolate_dropouts({10: [a], 14: [b]})
    assert list(out) == [10, 11, 12, 13, 14]
    assert out[11] == [box(10, 20, 20, 40, score=0.6, track_id=4, source="screen")]
    assert out[12] == [box(20, 40, 30, 60, score=0.6, track_id=4, source="screen")]
    assert out[10] == [a] and out[14] == [b]


@pytest.mark.parametrize("gap, added", [(1, 0), (2, 1), (15, 14), (16, 0)])
def test_interpolation_gap_limits(gap, added):
    data = {0: [box(track_id=1)], gap: [box(track_id=1)]}
    assert len(interpolate_dropouts(data)) == 2 + added


def test_interpolation_custom_gap():
    data = {0: [box(track_id=0)], 4: [box(track_id=0)]}
    assert interpolate_dropouts(data, max_gap=3) == data


def test_interpolation_only_fills_between_consecutive_sightings():
    data = {f: [box(x=f, track_id=0)] for f in (6, 0, 3)}
    out = interpolate_dropouts(data)
    assert {f: [r.x for r in regs] for f, regs in out.items()} == {f: [f] for f in range(7)}


@pytest.mark.parametrize(
    "a, b",
    [
        (box(track_id=-1), box(track_id=-1)),
        (box(track_id=1), box(track_id=2)),
        (box(track_id=1, source="face"), box(track_id=1, source="screen")),
    ],
)
def test_interpolation_does_not_join(a, b):
    assert interpolate_dropouts({0: [a], 3: [b]}) == {0: [a], 3: [b]}


def test_interpolation_keeps_existing_and_does_not_mutate():
    data = {0: [box(track_id=1)], 1: [box(x=700)], 2: [box(track_id=1)], 9: []}
    before = copy.deepcopy(data)
    out = interpolate_dropouts(data)
    assert out[1] == [box(x=700), box(track_id=1)]
    assert out[9] == []
    out[0][0].x = 123
    assert data == before


def test_interpolation_empty():
    assert interpolate_dropouts({}) == {}


def test_tracker_and_interpolation_together():
    tracker = IouTracker()
    frames = [[box(x=0)], [], [], [box(x=30)]]
    out = interpolate_dropouts({f: tracker.step(r) for f, r in enumerate(frames)})
    assert [[r.x for r in out[f]] for f in range(4)] == [[0], [10], [20], [30]]


# YOLO

NAMES = {0: "person", 2: "car", 62: "tv", 63: "laptop", 67: "cell phone"}


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def fake(monkeypatch):
    state = types.SimpleNamespace(names=NAMES, rows=[], loaded=[], calls=[], error=None)

    class FakeYOLO:
        def __init__(self, weights):
            state.loaded.append(weights)
            self.names = state.names

        def predict(self, *args, **kwargs):
            state.calls.append((args, kwargs))
            if state.error:
                raise state.error
            data = np.array(state.rows, dtype=np.float32).reshape(-1, 6)
            boxes = types.SimpleNamespace(
                xyxy=FakeTensor(data[:, :4]),
                conf=FakeTensor(data[:, 4]),
                cls=FakeTensor(data[:, 5]),
            )
            return [types.SimpleNamespace(boxes=boxes)]

    class FakeModule(types.ModuleType):
        def __getattr__(self, name):
            if name != "YOLO":
                raise AttributeError(name)
            state.offline_at_import = os.environ.get("YOLO_OFFLINE")
            return FakeYOLO

    monkeypatch.setenv("YOLO_OFFLINE", "0")
    monkeypatch.setitem(sys.modules, "ultralytics", FakeModule("ultralytics"))
    return state


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake")
    return path


def test_module_imports_without_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # makes importing it fail
    importlib.reload(yolo)


@pytest.mark.parametrize("name", ["missing.pt", "", "folder"])
def test_weights_must_be_an_existing_file(fake, tmp_path, name):
    (tmp_path / "folder").mkdir()
    with pytest.raises(FileNotFoundError):
        yolo.YoloObjects(tmp_path / name)
    assert not fake.loaded


def test_loads_local_path_offline(fake, weights, monkeypatch):
    monkeypatch.chdir(weights.parent)
    yolo.YoloObjects(weights.name)
    assert fake.offline_at_import == "1"
    assert fake.loaded == [str(weights)]


@pytest.mark.parametrize("names", [{0: "car", 1: "tv"}, {0: "person", 1: "car"}, {}])
def test_required_classes(fake, weights, names):
    fake.names = names
    with pytest.raises(ValueError):
        yolo.YoloObjects(weights)


@pytest.mark.parametrize("screen", ["tv", "laptop", "cell phone", "monitor", "screen"])
def test_one_screen_class_is_enough(fake, weights, screen):
    fake.names = {0: screen, 1: "person"}
    yolo.YoloObjects(weights)


@pytest.mark.parametrize("person, screen", [(0.40, 0.25), (0.2, 0.6)])
def test_predict_arguments(fake, weights, person, screen):
    image = np.zeros((8, 8, 3), np.uint8)
    yolo.YoloObjects(weights, person, screen)(image)
    (args, kwargs), = fake.calls
    assert len(args) == 1 and args[0] is image
    assert set(kwargs) == {"classes", "conf", "verbose"}
    assert sorted(kwargs["classes"]) == [0, 62, 63, 67]
    assert kwargs["conf"] == min(person, screen)
    assert kwargs["verbose"] is False


def test_output_split_and_filtered(fake, weights):
    fake.rows = [
        [1, 2, 30, 40, 0.40, 0],
        [1, 2, 30, 40, 0.39, 0],
        [5, 6, 70, 80, 0.25, 62],
        [5, 6, 70, 80, 0.24, 63],
        [7, 7, 8, 8, 0.90, 67],  # a single pixel is still kept
        [0, 0, 50, 50, 0.99, 2],
    ]
    persons, screens = yolo.YoloObjects(weights)(None)
    np.testing.assert_allclose(persons, [[1, 2, 30, 40, 0.40]])
    np.testing.assert_allclose(screens, [[5, 6, 70, 80, 0.25], [7, 7, 8, 8, 0.90]])
    assert persons.dtype.kind == screens.dtype.kind == "f"


def test_output_shape_without_detections(fake, weights):
    persons, screens = yolo.YoloObjects(weights)(None)
    assert persons.shape == screens.shape == (0, 5)
    assert persons.dtype.kind == screens.dtype.kind == "f"


def test_errors_propagate(fake, weights):
    detector = yolo.YoloObjects(weights)
    fake.error = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        detector(None)
