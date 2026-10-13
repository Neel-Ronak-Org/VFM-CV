import os

import numpy as np

from src.datasets import CODDataset, verify_dataset


def test_loads_pairs(tiny_dataset):
    ds = CODDataset(tiny_dataset)
    assert len(ds) == 4
    name, img, mask = ds[0]
    assert name == "img_0"
    assert img.dtype == np.uint8 and img.shape[2] == 3
    assert mask.dtype == bool and mask.shape == img.shape[:2]
    assert mask.sum() == 30 * 30


def test_verify_ok(tiny_dataset):
    r = verify_dataset(CODDataset(tiny_dataset), expected=4)
    assert r["ok"] and r["nonbinary_masks"] == 0 and r["empty_masks"] == 0


def test_verify_detects_problems(tiny_dataset):
    os.remove(tiny_dataset / "GT" / "img_1.png")
    r = verify_dataset(CODDataset(tiny_dataset), expected=4)
    assert not r["ok"] and r["missing_masks"] == 1 and not r["count_ok"]
