import importlib.util
from pathlib import Path

import pandas as pd
import yaml

spec = importlib.util.spec_from_file_location("check_phase2", Path(__file__).parents[1] / "scripts" / "check_phase2.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def _write_cfg(tmp_path, **enc_overrides):
    enc = dict(name="sam", img_size=1024, mlp_ratio=4, patch_size=16, qkv_bias=True, use_rel_pos=True,
               window_size=14, out_chans=256, input_type="fft", freq_nums=0.25, prompt_type="highpass",
               prompt_embed_dim=256, tuning_stage=1234, handcrafted_tune=True, embedding_tune=True,
               adaptor="adaptor", embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31])
    enc.update(enc_overrides)
    d = tmp_path / "checkpoints" / "sam_adapter"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"model": {"name": "sam", "args": {"inp_size": 1024, "encoder_mode": enc}}}))


def test_sam_adapter_config_matches_released_demo_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "REPO", tmp_path)
    _write_cfg(tmp_path)                                    # = the repo's demo.yaml / cod-sam-vit-h.yaml
    r = cp.Report()
    cp.check_sam_adapter_config(r)
    assert not r.failed and any("[OK]" in line for line in r.lines)


def test_sam_adapter_config_mismatch_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "REPO", tmp_path)
    _write_cfg(tmp_path, global_attn_indexes=[5, 11, 17, 23], freq_nums=0.5)
    r = cp.Report()
    cp.check_sam_adapter_config(r)
    assert r.failed and "global_attn_indexes" in r.lines[-1] and "freq_nums" in r.lines[-1]


def test_feder_clean_only_counts_as_done(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    rows = [{"model": m, "dataset": d, "corruption": "clean", "severity": 0, "Sm": 0.8, "precision": "fp32"}
            for m in cp.EVALUATED for d in cp.DATASETS]
    pd.DataFrame(rows).to_csv(tmp_path / "results" / "benchmark.csv", index=False)
    maps = [{"model": m, "dataset": d, "Sm": 0.8, "n_missing": 0} for m in ("FEDER", "SINet-V2", "ZoomNet") for d in cp.DATASETS]
    pd.DataFrame(maps).to_csv(tmp_path / "results" / "released_maps.csv", index=False)
    r = cp.Report()
    cp.check_results(r, have={m: True for m in cp.EVALUATED} | {"FEDER": False})
    assert not r.incomplete and not r.failed
    assert any("clean-only" in line for line in r.lines)


def test_missing_results_produce_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    r = cp.Report()
    cp.check_results(r, have={"PFNet": True, "FEDER": False})
    assert r.incomplete
    assert "python scripts/run_clean_baselines.py --models PFNet" in r.todo
