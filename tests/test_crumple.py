"""Fast CPU tests for the data pipeline:  pytest -q"""
import numpy as np
import cv2

from garmentcanon.data.crumple_procedural import CrumpleParams, crumple, mask_from_white_background


def _shirt(h=512, w=384):
    img = np.full((h, w, 3), 255, np.uint8)
    pts = np.array([[115, 60], [269, 60], [350, 130], [310, 180], [280, 160], [280, 450],
                    [104, 450], [104, 160], [74, 180], [34, 130]], np.int32)
    cv2.fillPoly(img, [pts], (40, 70, 160))
    return img


def test_output_shape_and_dtype():
    out, p = crumple(_shirt(), params=CrumpleParams(seed=1, out_size=256))
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8
    assert "background" in p.record


def test_deterministic_with_seed():
    a, _ = crumple(_shirt(), params=CrumpleParams(seed=7, out_size=128))
    b, _ = crumple(_shirt(), params=CrumpleParams(seed=7, out_size=128))
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a, _ = crumple(_shirt(), params=CrumpleParams(seed=1, out_size=128))
    b, _ = crumple(_shirt(), params=CrumpleParams(seed=2, out_size=128))
    assert not np.array_equal(a, b)


def test_garment_colors_survive():
    """Pixels are resampled, not invented: the garment's blue should still dominate somewhere."""
    out, _ = crumple(_shirt(), params=CrumpleParams(seed=3, out_size=256, background="solid"))
    blueish = (out[..., 2] > out[..., 0] + 40).mean()
    assert blueish > 0.05


def test_mask_from_white_background():
    m = mask_from_white_background(_shirt())
    assert 0.2 < (m > 0.5).mean() < 0.6
