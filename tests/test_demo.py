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
    pool = demo.ModelPool("cpu")
    pool.models.update(models)
    app = demo.build_app(pool, [], "CPU", list(models))
    assert app is not None


def test_demo_prefers_finished_ours(monkeypatch):
    status = {"Ours-Full": "unfinished", "Ours-Full-15ep": "finished", "Ours-Aug": "missing", "Ours-Clean": "missing"}
    monkeypatch.setattr(demo, "ours_status", lambda n: status.get(n, "missing"))
    assert demo.pick_ours() == "Ours-Full-15ep"
    assert demo.default_selection(None) == ["SINet-V2", "Ours-Full-15ep"]
    assert demo.default_selection(["PFNet"]) == ["PFNet"]
    status["Ours-Full-15ep"] = "unfinished"
    assert demo.pick_ours() == "Ours-Full"
    assert "Ours-Aug" not in demo.pickable() and "SINet-V2" in demo.pickable()
    status.update({k: "missing" for k in status})
    assert demo.pick_ours() is None and demo.default_selection(None) == ["SINet-V2"]


def test_model_pool_loads_lazily_and_reports_errors(monkeypatch):
    calls = []

    def fake_build(name, device=None):
        calls.append(name)
        if name == "Broken":
            raise FileNotFoundError("no weights")
        return build_model("debug-centre", device="cpu")

    monkeypatch.setattr(demo, "build_model", fake_build)
    pool = demo.ModelPool("cpu")
    assert calls == []
    got = pool.subset(["A", "Broken"])
    assert not isinstance(got["A"], str) and got["Broken"].startswith("unavailable")
    pool.subset(["A"])
    assert calls == ["A", "Broken"]  # cached, not rebuilt
    pool.release(keep=["A"])
    assert list(pool.models) == ["A"]
