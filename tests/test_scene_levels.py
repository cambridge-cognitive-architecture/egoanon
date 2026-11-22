import json
import os
import socket
import subprocess
import sys
import types

import cv2
import numpy as np
import pytest

from egoanon import MODEL_LEVELS, Pose, Region, SceneModels, SceneObject, apply_level
from egoanon import load_plan, render_video, save_plan, scan_video
from egoanon import demo_scene as demo
from egoanon.models import (ClipSeg, DepthAnything, LamaInpainter, YoloPose, fingerprint,
                            load_models)
from egoanon.offline import NetworkBlocked, offline
from egoanon.record import (TEMPERATURES, WITHHELD, MlxVlm, describe, render_record,
                            window_frames)
from egoanon.render import LEVELS
from egoanon.scene import colour_depth, draw_labels, remove, removal_mask, select_poses
from egoanon.screen import NUMBER_WORDS, OBJECT_WORDS, screen
from egoanon.video import is_anonymised, manifest_path

W, H = 320, 240


def noise(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), np.uint8)


def flat(image, mask):
    return np.full_like(image, 77)


def models(**kw):
    defaults = dict(inpaint=flat, depth=lambda im: im[..., 0].astype(float),
                    signs=lambda im: np.zeros(im.shape[:2], bool))
    return SceneModels(**{**defaults, **kw})


def points(x, y):
    out = np.zeros((17, 2))
    out[5], out[6] = (x, y), (x + 20, y)
    return out


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "a.mp4"
    demo.write_clip(path)
    return path, scan_video(path, demo.ColourDetector())


# inpaint

@pytest.mark.parametrize("level", MODEL_LEVELS)
@pytest.mark.parametrize("source", ["body", "face", "person", "tag", "screen", "manual"])
def test_nothing_under_a_region_reaches_the_output(level, source):
    regions = [Region(100, 80, 60, 70, source)]
    mask = removal_mask(regions, W, H)
    a, b = noise(), noise()
    b[mask] = 255 - b[mask]
    echo = models(inpaint=lambda im, m: im.copy())  # the worst inpainter: returns its input
    assert np.array_equal(apply_level(a, regions, level, echo), apply_level(b, regions, level, echo))


def test_whole_body_box_is_removed_to_its_corners():
    out = apply_level(noise(), [Region(100, 50, 60, 120, "body")], "inpaint", models())
    assert (out[50:170, 100:160] == 77).all()


@pytest.mark.parametrize("size", [1, 2, 4, 8, 20, 64])
def test_small_regions_are_fully_replaced(size):
    out = apply_level(noise(), [Region(150, 120, size, size, "body")], "inpaint", models())
    assert (out[120:120 + size, 150:150 + size] == 77).all()


def test_only_masked_pixels_change():
    image, mask = noise(), removal_mask([Region(100, 80, 60, 70, "body")], W, H)
    assert np.array_equal(remove(image, mask, flat)[~mask], image[~mask])


def test_padded_model_output_is_cropped_not_resized():
    image, mask = noise(), np.ones((H, W), bool)
    padded = np.zeros((H + 4, W + 4, 3), np.uint8)
    padded[:H, :W] = noise(1)
    assert np.array_equal(remove(image, mask, lambda i, m: padded), noise(1))


def test_short_model_output_stops():
    with pytest.raises(RuntimeError):
        remove(noise(), np.ones((H, W), bool), lambda i, m: np.zeros((8, 8, 3), np.uint8))


def test_failed_inpainter_has_no_fallback():
    def broken(image, mask):
        raise RuntimeError("no model")
    with pytest.raises(RuntimeError):
        apply_level(noise(), [Region(10, 10, 40, 40, "body")], "inpaint", models(inpaint=broken))


def test_person_seen_only_by_the_pose_model_is_removed():
    m = models(pose=lambda im: [Pose((100, 50, 160, 170), np.zeros((17, 2)))])
    out = apply_level(noise(), [], "inpaint", m)
    assert (out[50:170, 100:160] == 77).all()


def test_skeletons_are_drawn():
    regions = [Region(100, 50, 60, 120, "body")]
    m = models(pose=lambda im: [Pose((100, 50, 160, 170), points(110, 90))])
    assert (apply_level(noise(), regions, "inpaint", m)[90, 120] != 77).any()


def test_duplicate_and_manual_poses_are_not_drawn():
    best = Pose((10, 10, 70, 100), points(20, 30), 0.9)
    double = Pose((12, 10, 72, 100), points(22, 30), 0.5)
    manual = Pose((200, 100, 260, 200), points(210, 120), 0.8)
    kept = select_poses([double, manual, best], [Region(200, 100, 60, 100, "manual")], W, H)
    assert len(kept) == 1 and kept[0] is best


# depth

def test_depth_and_signs_see_only_the_people_free_frame_without_skeletons():
    seen = []

    def depth(im):
        seen.append(im.copy())
        return im[..., 0].astype(float)

    def signs(im):
        seen.append(im.copy())
        return np.zeros((H, W), bool)

    regions, pose = [Region(100, 50, 60, 120, "body")], Pose((100, 50, 160, 170), points(110, 90))
    apply_level(noise(), regions, "depth", models(depth=depth, signs=signs, pose=lambda im: [pose]))
    assert len(seen) == 2
    assert all((im[50:170, 100:160] == 77).all() for im in seen)


def test_near_is_dark_and_far_is_light():
    depth = np.tile(np.linspace(0, 1, W), (H, 1))  # right side nearer
    out = colour_depth(depth, np.zeros((H, W), bool)).astype(int)
    assert out[100, 5].sum() > out[100, W - 5].sum() + 200


def test_flat_depth_is_accepted():
    assert colour_depth(np.ones((H, W)), np.zeros((H, W), bool)).shape == (H, W, 3)


def test_bad_depth_stops():
    for bad in (np.full((H, W), np.nan), np.ones((4, 4))):
        with pytest.raises(RuntimeError):
            colour_depth(bad, np.zeros((H, W), bool))


@pytest.mark.parametrize("via", ["sign", "screen", "manual"])
def test_detail_is_blurred_in_screens_signs_and_manual_regions(via):
    stripes = np.tile((np.arange(W) // 3 % 2).astype(float), (H, 1))
    area = np.zeros((H, W), bool)
    area[60:180, 100:220] = True
    regions = [] if via == "sign" else [Region(100, 60, 120, 120, via)]
    m = models(depth=lambda im: stripes,
               signs=lambda im: area if via == "sign" else np.zeros((H, W), bool))
    out = apply_level(noise(), regions, "depth", m).astype(float)
    assert out[110:130, 140:180].std(axis=1).mean() < 0.1 * out[10:30, 10:50].std(axis=1).mean()


def test_wrong_sign_mask_stops():
    with pytest.raises(RuntimeError):
        apply_level(noise(), [], "depth", models(signs=lambda im: np.zeros((2, 2), bool)))


@pytest.mark.parametrize("kind", ["pin", "box"])
def test_object_labels_are_drawn(kind):
    canvas, area = np.zeros((H, W, 3), np.uint8), np.zeros((H, W), bool)
    area[90:150, 100:180] = True
    assert draw_labels(canvas, [SceneObject("desk", area, kind)], np.zeros((H, W), bool)).any()


def test_label_without_room_is_left_out():
    canvas, area = np.zeros((20, 20, 3), np.uint8), np.ones((20, 20), bool)
    assert draw_labels(canvas, [SceneObject("desk", area)], np.zeros((20, 20), bool)).shape == canvas.shape


# both model levels

@pytest.mark.parametrize("level", LEVELS)
def test_bodies_do_not_change_the_existing_levels(level):
    face = Region(30, 30, 40, 40, "face")
    body = Region(100, 50, 60, 120, "body")
    assert np.array_equal(apply_level(noise(), [face], level), apply_level(noise(), [face, body], level))


@pytest.mark.parametrize("level,missing", [("inpaint", "inpaint"), ("depth", "inpaint"),
                                           ("depth", "depth"), ("depth", "signs")])
def test_missing_model_stops(level, missing):
    with pytest.raises(ValueError, match=missing):
        apply_level(noise(), [], level, models(**{missing: None}))


@pytest.mark.parametrize("level", MODEL_LEVELS)
def test_no_models_stops(level):
    with pytest.raises(ValueError):
        apply_level(noise(), [], level)


@pytest.mark.parametrize("level", MODEL_LEVELS)
def test_plan_without_bodies_stops(level, clip, tmp_path):
    path, plan = clip
    plan.params.pop("bodies")
    with pytest.raises(ValueError, match="--bodies"):
        render_video(path, plan, tmp_path / "out.mp4", level, demo.standin_models(0))
    assert not list(tmp_path.glob("out*"))


@pytest.mark.parametrize("level", MODEL_LEVELS)
def test_end_to_end(level, clip, tmp_path):
    path, plan = clip
    save_plan(plan, tmp_path / "plan.json")
    out = tmp_path / "out.mp4"
    m = render_video(path, load_plan(tmp_path / "plan.json"), out, level, demo.standin_models(0))
    assert m["n_frames"] == m["n_frames_covered"] == demo.N and is_anonymised(out)
    cap = cv2.VideoCapture(str(out))
    try:
        for _ in range(demo.N):
            ok, frame = cap.read()
            assert ok
            if level == "inpaint":
                assert not demo.near(frame, demo.FIGURE).any()
    finally:
        cap.release()


@pytest.mark.parametrize("level", MODEL_LEVELS)
def test_changed_source_stops(level, clip, tmp_path):
    path, plan = clip
    plan.source_sha256 = "0" * 64
    with pytest.raises(ValueError):
        render_video(path, plan, tmp_path / "out.mp4", level, demo.standin_models(0))


# record

HOLES = {
    "name as the first word": "Morgan, desk",
    "name in lower case": "morgan, desk",
    "name later in the line": "desk, Morgan",
    "lower-case name later in the line": "desk, morgan",
    "non-Latin text": "desk, door, \u75c5\u4eba",
    "joined by a comma": "desk,man",
    "spelled-out number": "desk, seven",
    "paraphrase for a person": "desk, robot",
    "the real model's line": "Door, robot, wall, pipe, tray, computer",
}


@pytest.mark.parametrize("line", HOLES.values(), ids=HOLES.keys())
def test_each_known_hole_is_closed(line):
    assert screen(line)


@pytest.mark.parametrize("line", HOLES.values(), ids=HOLES.keys())
def test_no_hole_reaches_a_record(line, clip, tmp_path):
    path, plan = clip
    out = tmp_path / "out.json"
    render_record(path, plan, out, lambda *a: line, window_s=2)
    written = out.read_text() + manifest_path(out).read_text()
    assert all(r["scene"] == WITHHELD for r in json.loads(out.read_text())["windows"])
    assert line not in written and "morgan" not in written.lower()


@pytest.mark.parametrize("word", [
    "morgan", "Morgan", "MORGAN", "smith", "someone", "man", "woman", "person", "people",
    "robot", "figure", "mannequin", "child", "hand", "face", "wound", "oxygen", "mask",
    "uniform", "room", "scene", "area", "layout", "thing", "deskman", "doorMorgan",
])
@pytest.mark.parametrize("line", ["{}, desk", "desk, {}", "desk, {}, door", "desk {}"])
def test_unlisted_word_rejects_in_any_position(word, line):
    assert "word not on the object list" in screen(line.format(word))


@pytest.mark.parametrize("line", [
    "desk, \u533b\u751f", "desk, \u0432\u0440\u0430\u0447", "desk, \u0637\u0628\u064a\u0628",
    "desk, \u03b3\u03b9\u03b1\u03c4\u03c1\u03cc\u03c2", "desk, \u00c9mile", "desk, d\u043eor", "desk, \uff44oor",
    "desk\uff0c door", "desk\u00a0door", "desk, door\u200b", "\u202edesk, door",
    "desk, door \U0001f600", "des\u212a, door", "desk, \u017fink",
])
def test_non_latin_and_lookalike_text_rejects(line):
    assert "non-Latin or non-ASCII text" in screen(line)


def test_no_non_ascii_character_gets_through():
    for code in (*range(0x80, 0x3400), 0xFEFF, 0xFF0C, 0xFF44, 0x1F600, 0xE0041):
        char = chr(code)
        for line in (f"desk, door{char}", f"desk{char}door", f"{char}desk, door"):
            assert screen(line)


@pytest.mark.parametrize("word", sorted(NUMBER_WORDS))
@pytest.mark.parametrize("line", ["desk, {}", "{} desk, door", "desk, {}-door"])
def test_spelled_out_numbers_reject(word, line):
    assert "spelled-out number" in screen(line.format(word))


@pytest.mark.parametrize("line", ["desk, twenty-one", "desk, Seven", "desk, door 4", "desk, door4",
                                  "desk, \u0663", "desk, door \u00b2"])
def test_other_numbers_reject(line):
    assert screen(line)


@pytest.mark.parametrize("line", [
    "desk,man", "desk,door", "desk ,door", "desk,  door", "desk  door", "desk-door", "desk/door",
    "desk_door", "desk.door", "desk;door", "desk|door", "desk\tdoor", "desk\ndoor", "desk\rdoor",
    "desk, door\n", " desk, door", "desk, door,", "desk: door", "desk & door", "desk's door",
    "desk,,door", "desk\x00door",
])
def test_joined_tokens_reject(line):
    assert "joined tokens or other characters" in screen(line)


@pytest.mark.parametrize("line", ["deskdoor, sink", "desk, Door", "DESK, door", "dEsk, door", "deskDoor, sink"])
def test_run_together_words_and_capitals_reject(line):
    assert screen(line)


@pytest.mark.parametrize("line", ["", "desk", "desk, door, sink, bin, shelf, chair, lamp"])
def test_length_rejects(line):
    assert screen(line)


def test_list_is_plain_lower_case_words():
    assert all(w.isascii() and w.isalpha() and w.islower() for w in OBJECT_WORDS)
    assert not OBJECT_WORDS & NUMBER_WORDS


@pytest.mark.parametrize("word", sorted(OBJECT_WORDS))
def test_every_listed_word_passes(word):
    assert screen(f"{word}, desk") == []


@pytest.mark.parametrize("line", ["door, desk", "Desk, door, computer", "office chair, waste bin",
                                  "desk, door, sink, bin, shelf, chair"])
def test_screen_passes_objects(line):
    assert screen(line) == []


def test_screen_reasons_never_quote_the_line():
    assert "morgan" not in " ".join(screen("desk, Morgan")).lower()


def test_hallucination_warning_is_in_the_record(clip, tmp_path):
    path, plan = clip
    out = tmp_path / "out.json"
    # Listed words pass whether or not they are in the frame, hence the warning.
    render_record(path, plan, out, lambda *a: "sofa, printer", window_s=2)
    for text in (out.read_text(), manifest_path(out).read_text()):
        assert "hallucination" in text and "not present" in text


def test_rejected_text_is_never_returned():
    row = describe(noise(), lambda *a: "someone named Morgan")
    assert row["scene"] == WITHHELD and row["attempts"] == len(TEMPERATURES)
    assert "morgan" not in json.dumps(row).lower()


def test_only_the_first_line_is_kept():
    assert describe(noise(), lambda *a: '"door, desk.\nsomeone"')["scene"] == "door, desk"


def test_retry_stops_at_the_first_pass():
    replies = iter(["someone", "door, desk"])
    row = describe(noise(), lambda *a: next(replies))
    assert row["passed"] and row["attempts"] == 2
    assert "word not on the object list" in row["reject_reasons"]


def test_windows():
    assert window_frames(60, 10, 2) == [(0, 19, 9), (20, 39, 29), (40, 59, 49)]
    assert window_frames(45, 10, 2)[-1] == (40, 44, 42)
    with pytest.raises(ValueError):
        window_frames(60, 10, 0.01)


def test_record_end_to_end(clip, tmp_path, capsys):
    path, plan = clip
    out = tmp_path / "out.json"
    m = render_record(path, plan, out, demo.StandInVlm(), window_s=2)
    rows = json.loads(out.read_text())["windows"]
    assert [r["scene"] for r in rows] == ["door, desk, screen", WITHHELD, "desk, door"]
    assert m["n_withheld"] == 1 and is_anonymised(out)
    assert "someone" not in out.read_text() + capsys.readouterr().out
    assert {p.name for p in tmp_path.iterdir()} == {"a.mp4", "out.json", "out_record_manifest.json"}


def test_record_and_video_manifests_do_not_collide(tmp_path):
    assert manifest_path(tmp_path / "x.mp4") != manifest_path(tmp_path / "x.json")


def test_model_error_writes_no_record(clip, tmp_path):
    def broken(*args):
        raise RuntimeError("no model")
    path, plan = clip
    with pytest.raises(RuntimeError):
        render_record(path, plan, tmp_path / "out.json", broken)
    assert not list(tmp_path.glob("out*"))


def test_record_refuses_changed_source_existing_output_and_wrong_suffix(clip, tmp_path):
    path, plan = clip
    existing = tmp_path / "there.json"
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        render_record(path, plan, existing, demo.StandInVlm())
    with pytest.raises(ValueError):
        render_record(path, plan, tmp_path / "out.txt", demo.StandInVlm())
    plan.source_sha256 = "0" * 64
    with pytest.raises(ValueError):
        render_record(path, plan, tmp_path / "out.json", demo.StandInVlm())


# offline and local models only

@pytest.mark.parametrize("adapter", [LamaInpainter, YoloPose, DepthAnything, ClipSeg, MlxVlm])
def test_adapters_take_only_local_paths(adapter, tmp_path):
    for path in (tmp_path / "none", "some-org/some-model"):
        with pytest.raises(FileNotFoundError):
            adapter(path)


def test_lama_is_given_the_local_file(tmp_path, monkeypatch):
    weights, seen = tmp_path / "big-lama.pt", []
    weights.write_bytes(b"x")
    fake = types.SimpleNamespace(SimpleLama=lambda: seen.append(os.environ.get("LAMA_MODEL")))
    monkeypatch.setitem(sys.modules, "simple_lama_inpainting", fake)
    monkeypatch.delenv("LAMA_MODEL", raising=False)
    LamaInpainter(weights)
    assert seen == [str(weights)] and "LAMA_MODEL" not in os.environ


def test_loader_names_what_is_missing():
    with pytest.raises(ValueError, match="lama"):
        load_models("inpaint")
    with pytest.raises(ValueError, match="clipseg"):
        load_models("depth", lama="x", depth="y")


def test_fingerprint_follows_content(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    before = fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert fingerprint(tmp_path) != before


def test_offline_blocks_the_network_and_restores_it():
    connect = socket.socket.connect
    with offline():
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("127.0.0.1", 9))
        with socket.socket() as s, pytest.raises(NetworkBlocked):
            s.connect(("127.0.0.1", 9))
    assert socket.socket.connect is connect


@pytest.mark.parametrize("level", ["inpaint", "depth", "record"])
def test_demo(level, tmp_path):
    out = tmp_path / "demo.png"
    subprocess.run([sys.executable, "-m", "egoanon.demo_scene", level, "--out", str(out)], check=True)
    assert cv2.imread(str(out)) is not None
