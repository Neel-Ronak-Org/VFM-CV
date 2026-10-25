"""Tables/figures/significance helpers (src/reporting.py, scripts/make_tables.py, scripts/significance.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.corruptions import load_protocol  # noqa: E402
from src import reporting as R  # noqa: E402


def fake_bench(models: dict) -> pd.DataFrame:
    """models: name -> (clean Sm, drop per severity). Every condition, every metric."""
    p = load_protocol()
    rows = []
    for m, (clean, drop) in models.items():
        for ds in ("COD10K",):
            rows.append(dict(model=m, dataset=ds, corruption="clean", severity=0, Sm=clean, Em=clean, wFm=clean - 0.1,
                             MAE=0.04, Dice=clean - 0.05, IoU=clean - 0.1))
            for c in p["corruptions"]:
                for s in p["severities"]:
                    v = clean - drop * s * (2 if c in R.LOCAL else 1)
                    rows.append(dict(model=m, dataset=ds, corruption=c, severity=s, Sm=v, Em=v, wFm=v - 0.1,
                                     MAE=0.04 + drop * s, Dice=v - 0.05, IoU=v - 0.1))
    return pd.DataFrame(rows)


@pytest.fixture
def bench():
    return fake_bench({"SINet-V2": (0.80, 0.004), "Ours-Full": (0.84, 0.006),
                       "Ours-Full-512": (0.82, 0.003), "Ours-Full-512-s1": (0.83, 0.003)})


def test_robustness_numbers(bench):
    r = R.robustness(bench, "SINet-V2", "COD10K")
    sub = bench[(bench.model == "SINet-V2") & (bench.corruption != "clean")]
    assert r["mcorr_Sm"] == pytest.approx(sub.Sm.mean())
    assert r["ret_Sm"] == pytest.approx(sub.Sm.mean() / 0.80)
    assert r["ret_local"] < r["ret_global"]
    assert r["mCE"] == pytest.approx(1.0)  # the reference model
    assert R.robustness(bench, "SINet-V2", "NC4K") is None  # incomplete -> None


def test_seed_aggregation(bench):
    mean, std, n = R.aggregate_seeds(lambda m, d: R.robustness(bench, m, d), "Ours-Full-512", "COD10K")
    assert n == 2 and mean["clean_Sm"] == pytest.approx(0.825) and std["clean_Sm"] == pytest.approx(np.std([0.82, 0.83], ddof=1))
    mean, std, n = R.aggregate_seeds(lambda m, d: R.robustness(bench, m, d), "Ours-Full", "COD10K")
    assert n == 1 and std is None


def test_paired_bootstrap_ci():
    rng = np.random.default_rng(1)
    d = rng.normal(0.02, 0.05, 2000)
    lo, hi = R.paired_bootstrap(d, n_boot=2000)
    assert lo < d.mean() < hi and lo > 0
    lo, hi = R.paired_bootstrap(rng.normal(0, 0.05, 2000), n_boot=2000)
    assert lo < 0 < hi


def test_significance_on_per_image_files(tmp_path):
    import significance
    p = load_protocol()
    rng = np.random.default_rng(0)
    names = [f"img{i}" for i in range(50)]
    for m, shift in (("A", 0.03), ("B", 0.0)):
        d = tmp_path / m / "COD10K"
        d.mkdir(parents=True)
        for c in p["corruptions"]:
            for s in p["severities"]:
                pd.DataFrame({"name": names, "Sm": 0.7 + shift + rng.normal(0, 0.01, 50)}).to_csv(d / f"{c}-{s}.csv.gz", index=False)
    df = significance.run([("A", "B", "all"), ("A", "B", "unseen"), ("A", "C", "all")], datasets=["COD10K", "CAMO"],
                          root=tmp_path, n_boot=500)
    assert len(df) == 2  # missing model C and dataset CAMO are skipped
    row = df[df.group == "all"].iloc[0]
    assert row.significant and row.ci_lo > 0 and row.n_images == 50 and row.mean_diff == pytest.approx(0.03, abs=0.005)


def test_table_formatting_and_bold(bench):
    import make_tables as T
    assert T.fmt(None) == T.DASH
    assert T.fmt(0.81234, 0.0012) == "0.812{\\tiny$\\pm$.001}"
    assert T.best_flags({"a": 0.5, "b": 0.7, "c": None}, lower=False) == {"b"}
    assert T.best_flags({"a": 0.5, "b": 0.7}, lower=True) == {"a"}
    t2 = T.table2(bench, "COD10K", load_protocol())
    assert "\\begin{table*}" in t2 and "Ours (512)" in t2 and "\\pm" in t2  # seeds -> mean +- std
    assert "\\textbf{0.840}" in t2  # best clean score bold
    assert "mean $\\pm$ std over 2 seeds" in t2
    t3 = T.table3(bench, load_protocol())
    assert "Training components" in t3 and T.DASH in t3  # missing variants shown as --
