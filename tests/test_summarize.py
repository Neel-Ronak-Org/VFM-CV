import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.corruptions import load_protocol

spec = importlib.util.spec_from_file_location("sb", Path(__file__).parents[1] / "scripts" / "summarize_benchmark.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)
P = load_protocol()


def _bench(model, clean_sm, corr_sm):
    rows = [{"model": model, "dataset": "COD10K", "corruption": "clean", "severity": 0,
             "Sm": clean_sm, "Em": clean_sm, "wFm": clean_sm, "Dice": clean_sm, "IoU": clean_sm, "MAE": 0.05}]
    for c in P["corruptions"]:
        for s in P["severities"]:
            v = corr_sm(c, s)
            rows.append({"model": model, "dataset": "COD10K", "corruption": c, "severity": s,
                         "Sm": v, "Em": v, "wFm": v, "Dice": v, "IoU": v, "MAE": 0.10})
    return rows


def test_retention_seen_unseen_and_mce():
    ref = _bench("SINet-V2", 0.8, lambda c, s: 0.6)                                   # error 0.4 everywhere
    ours = _bench("Ours", 0.8, lambda c, s: 0.7 if c in P["seen"] else 0.6)          # better on seen only
    df = sb.summarize(pd.DataFrame(ref + ours), P).set_index(["model", "group"])
    assert df.loc[("SINet-V2", "all"), "Sm_retention"] == pytest.approx(0.75)
    assert df.loc[("Ours", "seen"), "Sm_retention"] == pytest.approx(0.875)
    assert df.loc[("Ours", "unseen"), "Sm_retention"] == pytest.approx(0.75)
    assert df.loc[("SINet-V2", "all"), "mCE_Sm"] == pytest.approx(1.0)
    assert df.loc[("Ours", "all"), "mCE_Sm"] == pytest.approx((0.75 * 4 + 1.0 * 4) / 8)   # 0.3/0.4 on seen
    assert df.loc[("Ours", "all"), "MAE_ratio"] == pytest.approx(2.0)
    assert bool(df.loc[("Ours", "all"), "complete"])


def test_incomplete_groups_are_flagged():
    rows = _bench("SINet-V2", 0.8, lambda c, s: 0.6)
    rows = [r for r in rows if not (r["corruption"] == "fog" and r["severity"] == 5)]
    df = sb.summarize(pd.DataFrame(rows), P).set_index(["model", "group"])
    assert not df.loc[("SINet-V2", "all"), "complete"] and df.loc[("SINet-V2", "all"), "n_conditions"] == 39
    assert pd.isna(df.loc[("SINet-V2", "all"), "mCE_Sm"])
