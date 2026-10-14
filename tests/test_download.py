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


def test_fetch_is_crash_safe_and_folder_needs_marker(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(dd, "REPO", tmp_path)

    def boom(url, path):  # simulate a crash halfway through writing
        Path(path).write_bytes(b"half")
        raise ConnectionError("cut")
    monkeypatch.setattr(urllib.request, "urlretrieve", boom)
    assert dd.fetch({"url": "http://x", "dest": "w/model.pth"}) is None
    assert not (tmp_path / "w" / "model.pth").exists()          # no truncated file left behind
    assert not (tmp_path / "w" / "model.pth.part").exists()

    folder = tmp_path / "maps"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"x")                       # partial folder from an earlier crash
    import gdown

    def fail_folder(**kw):
        raise RuntimeError("rate limit")
    monkeypatch.setattr(gdown, "download_folder", fail_folder)
    assert dd.fetch({"gdrive_folder": "id", "dest": "maps/"}) is None  # NOT treated as complete
    assert (folder / "a.png").exists()                          # partial files kept for resume
    monkeypatch.setattr(gdown, "download_folder", lambda **kw: [str(folder / "a.png")])
    assert dd.fetch({"gdrive_folder": "id", "dest": "maps/"}) == folder
    assert (folder / dd.COMPLETE_MARKER).exists()
