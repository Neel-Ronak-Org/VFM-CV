import csv

import yaml

from src import evaluate


def test_smoke_and_resume(tiny_dataset, tmp_path, monkeypatch):
    data_root = tiny_dataset.parents[1]
    out = tmp_path / "bench.csv"
    args = ["--models", "debug-centre", "--datasets", "COD10K", "--corruptions", "clean", "fog",
            "--severities", "1", "--out", str(out), "--data-root", str(data_root), "--device", "cpu"]
    evaluate.main(args)
    rows = list(csv.DictReader(open(out)))
    assert {(r["corruption"], r["severity"]) for r in rows} == {("clean", "0"), ("fog", "1")}
    assert all(int(r["n_images"]) == 4 for r in rows)
    evaluate.main(args)  # second run must skip everything
    assert len(list(csv.DictReader(open(out)))) == 2


def test_prediction_resized_to_mask_size(tiny_dataset, tmp_path):
    import cv2
    import numpy as np
    cv2.imwrite(str(tiny_dataset / "GT" / "img_0.png"), np.zeros((30, 40), np.uint8))  # mask smaller than image
    out = tmp_path / "bench.csv"
    evaluate.main(["--models", "debug-centre", "--datasets", "COD10K", "--corruptions", "clean",
                   "--out", str(out), "--data-root", str(tiny_dataset.parents[1]), "--device", "cpu"])
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["n_images"] == "4" and rows[0]["precision"] == "fp32"


def test_interrupted_condition_resumes_to_identical_result(tiny_dataset, tmp_path, monkeypatch):
    import gzip
    from src.models.base import _Constant
    data_root = str(tiny_dataset.parents[1])
    common = ["--models", "debug-centre", "--datasets", "COD10K", "--corruptions", "fog", "--severities", "2",
              "--data-root", data_root, "--device", "cpu"]
    # reference: uninterrupted
    evaluate.main(common + ["--out", str(tmp_path / "ref.csv")])
    ref = next(csv.DictReader(open(tmp_path / "ref.csv")))
    # crash after 2 of 4 images
    calls = {"n": 0}
    orig = _Constant._forward

    def flaky(self, img):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("laptop crashed")
        return orig(self, img)
    monkeypatch.setattr(_Constant, "_forward", flaky)
    out = tmp_path / "bench.csv"
    try:
        evaluate.main(common + ["--out", str(out)])
    except KeyboardInterrupt:
        pass
    part = tmp_path / "per_image" / "bench" / "debug-centre" / "COD10K" / "fog-2.csv"
    assert part.exists() and not out.exists()
    assert len(list(csv.DictReader(open(part)))) == 2          # 2 images saved before the crash
    monkeypatch.setattr(_Constant, "_forward", orig)
    calls["n"] = 0
    evaluate.main(common + ["--out", str(out)])                 # resume
    row = next(csv.DictReader(open(out)))
    for k in ("Sm", "Em", "wFm", "MAE", "Dice", "IoU", "n_images"):
        assert row[k] == ref[k], k
    assert not part.exists() and part.with_suffix(".csv.gz").exists()
    with gzip.open(part.with_suffix(".csv.gz"), "rt") as f:
        names = [r["name"] for r in csv.DictReader(f)]
    assert sorted(names) == ["img_0", "img_1", "img_2", "img_3"]  # no duplicates


def test_parallel_pipeline_matches_sequential(tiny_dataset, tmp_path):
    common = ["--models", "debug-centre", "--datasets", "COD10K", "--corruptions", "clean", "gaussian_noise",
              "--severities", "3", "--data-root", str(tiny_dataset.parents[1]), "--device", "cpu"]
    evaluate.main(common + ["--out", str(tmp_path / "seq.csv"), "--workers", "0"])
    evaluate.main(common + ["--out", str(tmp_path / "par.csv"), "--workers", "3"])
    a, b = list(csv.DictReader(open(tmp_path / "seq.csv"))), list(csv.DictReader(open(tmp_path / "par.csv")))
    assert len(a) == len(b) == 2
    for ra, rb in zip(a, b):
        for k in ("corruption", "Sm", "Em", "wFm", "MAE", "Dice", "IoU", "n_images"):
            assert ra[k] == rb[k], k
