"""CPU tests for evaluation helpers and real-photo preprocessing (numpy/OpenCV only):  pytest -q"""

import cv2
import numpy as np

from garmentcanon.data.crumple_procedural import CrumpleParams, crumple, procedural_state
from garmentcanon.data.real_photos import frame_garment, frame_prediction, segment_flat_photo


def _shirt(h=1024, w=768, color=(40, 70, 160)):
    img = np.full((h, w, 3), 255, np.uint8)
    pts = (np.array([[230, 120], [538, 120], [700, 260], [620, 360], [560, 320], [560, 900],
                     [208, 900], [208, 320], [148, 360], [68, 260]]) * [w / 768, h / 1024]).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    img[m > 0] = color
    stripes = ((np.arange(h) + 30) % 60 < 15)[:, None] & (m > 0)   # white stripes, none on the outline
    img[stripes] = 255
    return img, m.astype(np.float32)


def _phone_photo(garment, mask, bg_color, scale=0.6, offset=(150, 120), gradient=(0.8, 1.15)):
    h, w = 1024, 768
    g = cv2.resize(garment, None, fx=scale, fy=scale)
    m = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    photo = np.ones((h, w, 3), np.float32) * bg_color
    oy, ox = offset
    region = photo[oy:oy + g.shape[0], ox:ox + g.shape[1]]
    region[:] = g * m[..., None] + region * (1 - m[..., None])
    photo *= np.linspace(*gradient, h)[:, None, None]       # uneven room lighting
    true = np.zeros((h, w), np.float32)
    true[oy:oy + g.shape[0], ox:ox + g.shape[1]] = m
    return photo.clip(0, 255).astype(np.uint8), true


def test_segment_flat_photo_with_uneven_lighting():
    garment, mask = _shirt()
    for bg in [(200, 190, 175), (120, 90, 60), (225, 225, 230)]:
        photo, true = _phone_photo(garment, mask, np.array(bg, np.float32))
        pred = segment_flat_photo(photo) > 0.5
        iou = (pred & (true > 0.5)).sum() / (pred | (true > 0.5)).sum()
        assert iou > 0.96, (bg, iou)


def test_frame_garment_is_position_and_scale_invariant():
    """The same garment photographed at different places/sizes frames to the same image."""
    garment, mask = _shirt()
    a, ma = _phone_photo(garment, mask, np.array([200, 190, 175], np.float32), 0.5, (100, 80), (1, 1))
    b, mb = _phone_photo(garment, mask, np.array([200, 190, 175], np.float32), 0.7, (250, 150), (1, 1))
    fa, fb = frame_garment(a, ma), frame_garment(b, mb)
    assert fa.shape == (1024, 768, 3)
    assert np.abs(fa.astype(float) - fb).mean() < 6


def test_frame_prediction_on_white_background():
    garment, mask = _shirt()
    small = cv2.resize(garment, (384, 512))
    canvas = np.full((1024, 768, 3), 255, np.uint8)
    canvas[300:812, 50:434] = small                          # off-center, smaller
    ref = frame_garment(garment, mask)
    assert np.abs(frame_prediction(canvas).astype(float) - ref).mean() < 6


def test_procedural_state_label_matches_fold():
    img, _ = _shirt(512, 384)
    _, p = crumple(img, params=CrumpleParams(seed=0, fold_prob=1.0, out_size=128))
    assert procedural_state(p.record) == "folded"
    _, p = crumple(img, params=CrumpleParams(seed=0, fold_prob=0.0, out_size=128))
    assert procedural_state(p.record) == "crumpled"


def test_gap_closed():
    import pytest
    pytest.importorskip("torch")
    from garmentcanon.eval.tryon_eval import gap_closed
    assert gap_closed(raw=0.5, ours=0.3, clean=0.1) == pytest.approx(0.5)
    assert gap_closed(raw=0.6, ours=0.8, clean=0.9, lower_is_better=False) == pytest.approx(2 / 3)


def test_viton_pairs_file_one_or_two_columns(tmp_path):
    from garmentcanon.data.viton_hd import VitonHD
    (tmp_path / "two.txt").write_text("00001_00.jpg 00002_00.jpg\n")
    (tmp_path / "one.txt").write_text("00003_00.jpg\n\n")
    assert VitonHD(tmp_path, "test", pairs_file=tmp_path / "two.txt").items[0].key == "00001_00"   # paired
    assert VitonHD(tmp_path, "test", pairs_file=tmp_path / "two.txt", paired=False).items[0].key == "00002_00"
    assert VitonHD(tmp_path, "test", pairs_file=tmp_path / "one.txt").items[0].key == "00003_00"
