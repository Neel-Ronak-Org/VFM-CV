"""Demo logic without a browser: corruption + side-by-side predictions, results tables, UI construction."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
import app as demo  # noqa: E402

from src.models import build_model  # noqa: E402


@pytest.fixture
def models():
    return {"debug-centre": build_model("debug-centre", device="cpu"), "Missing": "unavailable: no weights"}


def test_detect_side_by_side(models):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    gt = np.zeros((90, 120), bool)
    gt[30:60, 40:80] = True
    corrupted, res, gt_img = demo.detect(img, "fog", 3, gt, models, show_gt=True)
    assert corrupted.shape == img.shape and not np.array_equal(corrupted, img)
    (n1, ov, text), (n2, none, msg) = res
    assert ov.shape == img.shape and "ms" in text and "S-measure" in text
    assert none is None and "unavailable" in msg
    assert gt_img.shape == img.shape
    clean, res, gt_img = demo.detect(img, "clean", 0, None, models)
    assert np.array_equal(clean, img) and gt_img is None and "S-measure" not in res[0][2]


def test_large_upload_is_downscaled(models):
    img = np.zeros((2000, 3000, 3), np.uint8)
    corrupted, res, _ = demo.detect(img, "clean", 0, None, models)
    assert max(corrupted.shape[:2]) == demo.MAX_SIDE and res[0][1].shape == corrupted.shape


def test_corruption_labels_mark_unseen():
    ch = demo.corruption_choices(demo.load_protocol())
    assert ch["none (clean)"] == "clean"
    assert any("UNSEEN" in k and v == "motion_blur" for k, v in ch.items())
    assert len(ch) == 9


def test_results_tables_and_app_build(models):
    tables = demo.results_tables()  # reads the committed results/*.csv
    assert all(len(t.columns) for t in tables.values())
    app = demo.build_app(models, [], "CPU")
    assert app is not None


def test_demo_prefers_finished_ours(monkeypatch):
    class Fake:
        def __init__(self, name, finished):
            self.name, self.finished = name, finished

    available = {"Ours-Full": Fake("Ours-Full@ep3.0-unfinished", False), "Ours-Full-15ep": Fake("Ours-Full-15ep", True)}

    def fake_build(name, device=None):
        if name not in available:
            raise FileNotFoundError(name)
        return available[name]

    monkeypatch.setattr(demo, "build_model", fake_build)
    assert demo.pick_ours("cpu").name == "Ours-Full-15ep"
    available["Ours-Full-15ep"] = Fake("Ours-Full-15ep@ep9.0-unfinished", False)
    assert demo.pick_ours("cpu").name == "Ours-Full@ep3.0-unfinished"
    available.clear()
    assert demo.pick_ours("cpu") is None
