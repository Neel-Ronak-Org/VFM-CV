import numpy as np
import py_sod_metrics as psm
import pytest

from src.metrics import CODMetrics


def _square(h=64, w=64, box=(16, 48, 16, 48), val=1.0):
    m = np.zeros((h, w), np.float32)
    m[box[0]:box[1], box[2]:box[3]] = val
    return m


def test_perfect_prediction():
    gt = _square() > 0
    m = CODMetrics()
    m.step(gt.astype(np.float32), gt)
    r = m.results()
    assert r["Sm"] == pytest.approx(1.0)
    assert r["wFm"] == pytest.approx(1.0)
    assert r["Em"] > 0.99  # mean over 256 thresholds; threshold 0 marks every pixel foreground
    assert r["MAE"] == pytest.approx(0.0)
    assert r["Dice"] == pytest.approx(1.0) and r["IoU"] == pytest.approx(1.0)


def test_matches_pysodmetrics_directly():
    """Our wrapper must give exactly what PySODMetrics gives on uint8 inputs."""
    gt = _square() > 0
    pred = _square(box=(20, 50, 18, 46), val=200 / 255)
    m = CODMetrics()
    m.step(pred, gt)
    r = m.results()
    p8, g8 = np.round(pred * 255).astype(np.uint8), gt.astype(np.uint8) * 255
    S, E, W, M = psm.Smeasure(), psm.Emeasure(), psm.WeightedFmeasure(), psm.MAE()
    for x in (S, E, W, M):
        x.step(pred=p8, gt=g8)
    assert r["Sm"] == pytest.approx(S.get_results()["sm"])
    assert r["Em"] == pytest.approx(E.get_results()["em"]["curve"].mean())
    assert r["wFm"] == pytest.approx(W.get_results()["wfm"])
    assert r["MAE"] == pytest.approx(M.get_results()["mae"])
    # Known values for this example with pysodmetrics==1.6.2 (regression guard).
    assert r["Sm"] == pytest.approx(0.83228, abs=1e-4)
    assert r["MAE"] == pytest.approx(0.072266, abs=1e-5)


def test_dice_iou_by_hand():
    gt = _square(box=(0, 32, 0, 32)) > 0          # 1024 px
    pred = _square(box=(0, 32, 16, 48))           # 1024 px, overlap 512
    m = CODMetrics()
    m.step(pred, gt)
    r = m.results()
    assert r["Dice"] == pytest.approx(2 * 512 / 2048)
    assert r["IoU"] == pytest.approx(512 / 1536)


def test_averaging_and_gt_formats():
    gt255 = (_square() * 255).astype(np.uint8)
    m = CODMetrics()
    m.step(np.zeros_like(gt255, np.float32), gt255)   # all wrong
    m.step(gt255 / 255.0, gt255)                      # perfect
    assert len(m) == 2
    assert m.results()["Dice"] == pytest.approx(0.5)


def test_rejects_bad_input():
    m = CODMetrics()
    with pytest.raises(ValueError):
        m.step(np.full((8, 8), 2.0), np.zeros((8, 8), bool))
    with pytest.raises(ValueError):
        m.step(np.zeros((8, 9)), np.zeros((8, 8), bool))
