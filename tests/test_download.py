import importlib.util
import zipfile
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("download_data", Path(__file__).parents[1] / "scripts" / "download_data.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def _make_zip(path: Path, layout: dict[str, int]):
    """layout: folder -> number of (image, mask) pairs, e.g. {'TestDataset/CAMO': 2}."""
    img = np.zeros((8, 8, 3), np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    ok, png = cv2.imencode(".png", np.zeros((8, 8), np.uint8))
    with zipfile.ZipFile(path, "w") as zf:
        for folder, n in layout.items():
            mask_dir = "GT_Object" if folder.endswith("NC4K") else "GT"
            for i in range(n):
                zf.writestr(f"{folder}/Imgs/{i}.jpg", jpg.tobytes())
                zf.writestr(f"{folder}/{mask_dir}/{i}.png", png.tobytes())


def test_extract_and_organise(tmp_path):
    test_zip, train_zip = tmp_path / "test.zip", tmp_path / "train.zip"
    _make_zip(test_zip, {f"TestDataset/{d}": 2 for d in ("CAMO", "CHAMELEON", "COD10K", "NC4K")})
    _make_zip(train_zip, {"TrainValDataset": 3})
    raw = tmp_path / "raw"
    for z in (test_zip, train_zip):
        dd.safe_extract(z, tmp_path / "x" / z.stem)
        dd.organise(tmp_path / "x" / z.stem, raw)
    for d in ("CAMO", "CHAMELEON", "COD10K", "NC4K"):
        ds = dd.CODDataset(raw / "TestDataset" / d)
        assert len(ds) == 2
    assert len(dd.CODDataset(raw / "TrainDataset")) == 3


def test_rejects_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.txt", "x")
    try:
        dd.safe_extract(z, tmp_path / "out")
        raise AssertionError("should have refused")
    except RuntimeError:
        pass
    assert not (tmp_path / "escape.txt").exists()
