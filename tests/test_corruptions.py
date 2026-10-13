import io

import numpy as np
import pytest
from PIL import Image

from src.corruptions import (CODC_CORRUPTIONS, apply_corruption, corrupt_for_benchmark, image_seed,
                             load_protocol)

P = load_protocol()


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(1)
    base = rng.integers(40, 216, (96, 128, 3), dtype=np.uint8)   # keep away from 0/255 so clipping is rare
    base[:, 60:] = 200                                           # a flat region and an edge
    return base


# ---- protocol --------------------------------------------------------------------------
def test_protocol_matches_cod10k_c():
    """Values transcribed from COD10K-C Sec. 3 -- guard against accidental edits."""
    pr = P["params"]
    assert set(P["corruptions"]) == set(CODC_CORRUPTIONS) and len(P["corruptions"]) == 8
    assert pr["gaussian_noise"] == [8, 16, 24, 32, 40]
    assert pr["motion_blur"] == [3, 5, 7, 9, 11]
    assert pr["gaussian_blur"] == [0.8, 1.2, 1.6, 2.0, 2.5]
    assert pr["brightness"] == pr["contrast"] == [0.85, 0.70, 0.55, 0.40, 0.30]
    assert pr["fog"] == [0.12, 0.20, 0.28, 0.36, 0.44]
    assert pr["jpeg_compression"] == [70, 55, 40, 30, 20]
    assert set(P["seen"]).isdisjoint(P["unseen"]) and len(P["seen"]) == len(P["unseen"]) == 4


def test_seed_is_stable_and_distinct():
    a = image_seed(1, "COD10K", "x.jpg", "fog", 3)
    assert a == image_seed(1, "COD10K", "x.png", "fog", 3)   # extension-independent
    assert a != image_seed(1, "COD10K", "x.jpg", "fog", 4)
    assert a != image_seed(1, "CAMO", "x.jpg", "fog", 3)


# ---- generic properties ----------------------------------------------------------------
@pytest.mark.parametrize("c", P["corruptions"] + P["extended_unseen"])
def test_runs_deterministic_and_changes_image(img, c):
    a = corrupt_for_benchmark(img, "COD10K", "x", c, 3, base_seed=7)
    np.random.seed(999)  # global RNG state in between must not matter
    b = corrupt_for_benchmark(img, "COD10K", "x", c, 3, base_seed=7)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)


@pytest.mark.parametrize("c", P["corruptions"])
def test_severity_is_monotonic(img, c):
    errs = [np.abs(corrupt_for_benchmark(img, "D", "x", c, s, 0).astype(int) - img).mean() for s in range(1, 6)]
    assert all(e1 < e2 for e1, e2 in zip(errs, errs[1:])), (c, errs)


def test_severity_zero_is_identity(img):
    assert apply_corruption(img, "fog", 0) is img
    assert corrupt_for_benchmark(img, "D", "x", "clean", 0, 0) is img


def test_bad_args(img):
    with pytest.raises(ValueError):
        apply_corruption(img, "not_a_corruption", 1)
    with pytest.raises(ValueError):
        apply_corruption(img, "fog", 6)
    with pytest.raises(ValueError):
        apply_corruption(img.astype(np.float32), "fog", 1)


# ---- parameter-level checks against the COD10K-C definitions ---------------------------
def test_gaussian_noise_sigma(img):
    flat = np.full((256, 256, 3), 128, np.uint8)
    for s, sigma in zip(range(1, 6), P["params"]["gaussian_noise"]):
        diff = apply_corruption(flat, "gaussian_noise", s, seed=0).astype(float) - 128
        assert diff.std() == pytest.approx(sigma, rel=0.03)
        assert abs(diff.mean()) < 0.5


def test_motion_blur_is_horizontal_box(img):
    out = apply_corruption(img, "motion_blur", 1, seed=0)          # k = 3
    x = img.astype(float)
    expected = (x[5, 9] + x[5, 10] + x[5, 11]) / 3
    assert np.allclose(out[5, 10], expected, atol=1)
    vert = np.zeros((20, 20, 3), np.uint8)
    vert[10] = 255                                                  # horizontal line
    assert np.array_equal(apply_corruption(vert, "motion_blur", 5, seed=0), vert)  # unchanged by horizontal blur


def test_brightness_is_multiplicative(img):
    out = apply_corruption(img, "brightness", 3, seed=0)            # factor 0.55
    assert np.allclose(out, img * 0.55, atol=1)


def test_contrast_shrinks_spread(img):
    out = apply_corruption(img, "contrast", 5, seed=0)              # factor 0.30
    assert out.astype(float).std() == pytest.approx(img.astype(float).std() * 0.30, rel=0.05)


def test_fog_formula(img):
    out = apply_corruption(img, "fog", 2, seed=0)                   # alpha 0.20
    assert np.allclose(out, 0.8 * img + 0.2 * 255, atol=0.51)


def test_jpeg_quality(img):
    out = apply_corruption(img, "jpeg_compression", 5, seed=0)      # quality 20
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=20)
    assert np.array_equal(out, np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))


def test_rain_only_brightens(img):
    out = apply_corruption(img, "rain", 3, seed=0)
    assert (out.astype(int) >= img.astype(int) - 0).all()
    assert (out != img).any(axis=2).mean() > 0.02
