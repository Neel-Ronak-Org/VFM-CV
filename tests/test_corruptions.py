import numpy as np
import pytest

from src.corruptions import apply_corruption, corrupt_for_benchmark, image_seed, load_protocol


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(1).integers(0, 256, (96, 128, 3), dtype=np.uint8)


def test_protocol_partition():
    p = load_protocol()
    assert len(p["corruptions"]) == 8
    assert set(p["seen"]).isdisjoint(p["unseen"])
    assert p["severities"] == [1, 2, 3, 4, 5]


def test_seed_is_stable_and_distinct():
    a = image_seed(1, "COD10K", "x.jpg", "fog", 3)
    assert a == image_seed(1, "COD10K", "x.png", "fog", 3)   # extension-independent
    assert a != image_seed(1, "COD10K", "x.jpg", "fog", 4)
    assert a != image_seed(1, "CAMO", "x.jpg", "fog", 3)


@pytest.mark.parametrize("c", load_protocol()["corruptions"] + ["gaussian_blur"])
def test_all_corruptions_run_and_are_deterministic(img, c):
    a = corrupt_for_benchmark(img, "COD10K", "x", c, 3, base_seed=7)
    np.random.seed(999)  # global state in between must not matter
    b = corrupt_for_benchmark(img, "COD10K", "x", c, 3, base_seed=7)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)


def test_severity_increases_damage(img):
    errs = [np.abs(apply_corruption(img, "gaussian_noise", s, seed=0).astype(int) - img).mean() for s in range(1, 6)]
    assert all(e1 < e2 for e1, e2 in zip(errs, errs[1:]))


def test_severity_zero_is_identity(img):
    assert apply_corruption(img, "fog", 0) is img
    assert apply_corruption(img, "clean", 3) is img


def test_bad_args(img):
    with pytest.raises(ValueError):
        apply_corruption(img, "not_a_corruption", 1)
    with pytest.raises(ValueError):
        apply_corruption(img, "fog", 6)
