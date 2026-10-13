import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tiny_dataset(tmp_path):
    """A fake COD split: 4 random images + square masks in SINet-V2 layout."""
    rng = np.random.default_rng(0)
    root = tmp_path / "TestDataset" / "COD10K"
    (root / "Imgs").mkdir(parents=True)
    (root / "GT").mkdir()
    for i in range(4):
        img = rng.integers(0, 256, (60 + 4 * i, 80, 3), dtype=np.uint8)
        m = np.zeros(img.shape[:2], np.uint8)
        m[10:40, 20:50] = 255
        cv2.imwrite(str(root / "Imgs" / f"img_{i}.jpg"), img)
        cv2.imwrite(str(root / "GT" / f"img_{i}.png"), m)
    return root
