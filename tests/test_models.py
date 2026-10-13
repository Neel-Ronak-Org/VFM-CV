import numpy as np
import pytest
import torch

from src.models import build_model
from src.models.base import load_state_dict_file
from src.models.sinet_v2 import SINetV2


@pytest.fixture(scope="module")
def sinet():
    torch.manual_seed(0)
    return SINetV2(weights=None, device="cpu")


def test_sinet_v2_predict_contract(sinet):
    img = np.random.default_rng(0).integers(0, 256, (123, 171, 3), dtype=np.uint8)
    pred = sinet.predict(img)
    assert pred.shape == (123, 171) and pred.dtype == np.float32
    assert pred.min() == pytest.approx(0.0, abs=1e-6) and pred.max() == pytest.approx(1.0, abs=1e-4)  # min-max
    assert sinet.last_latency_ms > 0 and not sinet.fp16


def test_sinet_v2_architecture_size(sinet):
    n = sum(p.numel() for p in sinet.net.parameters())
    assert 26.5e6 < n < 27.5e6, n   # SINet-V2 (Res2Net-50) is ~27M parameters


def test_checkpoint_roundtrip_with_dataparallel_prefix(sinet, tmp_path):
    sd = {"module." + k: v for k, v in sinet.net.state_dict().items()}
    torch.save({"state_dict": sd}, tmp_path / "ckpt.pth")
    fresh = SINetV2(weights=None, device="cpu")
    load_state_dict_file(fresh.net, tmp_path / "ckpt.pth")
    k = "rfb2_1.conv_cat.conv.weight"
    assert torch.equal(fresh.net.state_dict()[k], sinet.net.state_dict()[k])


def test_checkpoint_mismatch_fails_loudly(sinet, tmp_path):
    torch.save({"foo.weight": torch.zeros(1)}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="does not match"):
        load_state_dict_file(sinet.net, tmp_path / "bad.pth")
    with pytest.raises(FileNotFoundError):
        load_state_dict_file(sinet.net, tmp_path / "missing.pth")


def test_registry():
    m = build_model("debug-centre", device="cpu")
    assert m.predict(np.zeros((40, 60, 3), np.uint8)).shape == (40, 60)
    with pytest.raises(KeyError):
        build_model("nope")
