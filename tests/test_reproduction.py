import csv
import importlib.util
from pathlib import Path

import cv2
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_released_maps_perfect_prediction(tiny_dataset, tmp_path):
    epm = _load("eval_prediction_maps")
    maps = tmp_path / "maps" / "results" / "COD10K"
    maps.mkdir(parents=True)
    for p in (tiny_dataset / "GT").glob("*.png"):  # perfect maps = the masks themselves
        cv2.imwrite(str(maps / p.name), cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    out = tmp_path / "maps.csv"
    epm.main(["--model", "X", "--pred-root", str(tmp_path / "maps"), "--datasets", "COD10K",
              "--data-root", str(tiny_dataset.parents[1]), "--out", str(out)])
    row = next(csv.DictReader(open(out)))
    assert float(row["Sm"]) == pytest.approx(1.0) and float(row["MAE"]) == 0.0
    assert row["n_images"] == "4" and row["n_missing"] == "0"


def test_reproduction_table(tmp_path):
    mr = _load("make_reproduction")
    pd.DataFrame([
        {"model": "SINet-V2", "dataset": "COD10K", "corruption": "clean", "severity": 0, "Sm": 0.820, "Em": 0.9,
         "wFm": 0.7, "MAE": 0.04, "n_images": 2026, "precision": "fp32"},
        {"model": "SINet-V2", "dataset": "CAMO", "corruption": "clean", "severity": 0, "Sm": 0.790, "Em": 0.9,
         "wFm": 0.7, "MAE": 0.07, "n_images": 250, "precision": "fp32"},
        {"model": "SINet-V2", "dataset": "COD10K", "corruption": "fog", "severity": 3, "Sm": 0.7, "Em": 0.8,
         "wFm": 0.6, "MAE": 0.05, "n_images": 2026, "precision": "fp32"},
    ]).to_csv(tmp_path / "bench.csv", index=False)
    pd.DataFrame([{"model": "SINet-V2", "dataset": "COD10K", "Sm": 0.818}]).to_csv(tmp_path / "maps.csv", index=False)
    df = mr.build(tmp_path / "bench.csv", tmp_path / "maps.csv", Path(__file__).parents[1] / "configs" / "reported.yaml")
    df = df.set_index("dataset")
    assert len(df) == 2                                   # corrupted rows ignored
    assert df.loc["COD10K", "within_tolerance"]           # 0.820 vs 0.815
    assert not df.loc["CAMO", "within_tolerance"]         # 0.790 vs 0.820
    assert df.loc["COD10K", "delta_Sm_vs_released_maps"] == pytest.approx(0.002)
