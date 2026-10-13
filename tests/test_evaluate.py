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
