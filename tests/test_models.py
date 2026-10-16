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


# ---- every baseline wrapper: random init on CPU, predict contract + architecture size ----------
BASELINES = [  # (class path, kwargs for random init, expected parameter range in millions)
    ("src.models.sam2_unet:SAM2UNet", {"weights": None}, (216, 217)),         # Hiera-L + adapters + decoder
    ("src.models.zoomnet:ZoomNet", {"weights": None}, (32.3, 32.5)),          # paper: 32.38M
    ("src.models.pfnet:PFNet", {"weights": None}, (46.4, 46.6)),              # paper: 46.50M
    ("src.models.feder:FEDER", {"weights": None}, (44.0, 44.3)),
    ("src.models.sam_adapter:SAMAdapter", {"weights": None, "embed_dim": 768}, (93, 95)),  # ViT-B variant
]


def _cls(path):
    import importlib
    mod, name = path.split(":")
    return getattr(importlib.import_module(mod), name)


@pytest.mark.parametrize("path,kw,prange", BASELINES, ids=[b[0].split(":")[1] for b in BASELINES])
def test_baseline_wrapper_contract(path, kw, prange):
    torch.manual_seed(0)
    m = _cls(path)(device="cpu", **kw)
    net = m.net
    n = sum(p.numel() for p in net.parameters()) / 1e6
    assert prange[0] <= n <= prange[1], n
    img = np.random.default_rng(0).integers(0, 256, (90, 130, 3), dtype=np.uint8)
    pred = m.predict(img)
    assert pred.shape == (90, 130) and pred.dtype == np.float32
    assert 0.0 <= pred.min() and pred.max() <= 1.0 and np.isfinite(pred).all()


def test_oracle_box_sam_uses_gt_and_refuses_without_it():
    from src.models.oracle_box_sam import OracleBoxMobileSAM, gt_box
    m = OracleBoxMobileSAM(weights=None, device="cpu")
    assert m.needs_gt and m.name.startswith("Oracle")
    img = np.random.default_rng(0).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        m.predict(img)
    gt = np.zeros((32, 48), bool)
    gt[8:16, 12:24] = True
    assert gt_box(gt, (64, 96)).tolist() == [24.0, 16.0, 48.0, 32.0]   # scaled to image size
    m.set_gt(gt)
    assert m.predict(img).shape == (64, 96)
    m.set_gt(np.zeros((64, 96), bool))
    assert m.predict(img).max() == 0.0                                   # empty mask -> empty map


def test_registry_lists_all_baselines():
    from src.models import REGISTRY
    for name in ("SINet-V2", "SAM2-UNet", "ZoomNet", "PFNet", "FEDER", "SAM-Adapter", "Oracle-Box-MobileSAM"):
        assert name in REGISTRY


def test_sam_adapter_detects_backbone_from_checkpoint(tmp_path):
    from src.models.sam_adapter import SAMAdapter
    src = SAMAdapter(weights=None, embed_dim=768, device="cpu")
    torch.save({"module." + k: v for k, v in src.net.state_dict().items()}, tmp_path / "sa.pth")
    m = SAMAdapter(weights=tmp_path / "sa.pth", device="cpu")
    assert m.variant == "ViT-B"


def test_sam_adapter_chunked_attention_is_exact():
    from src.models.sam_adapter import chunked_attention_forward
    from src.models.third_party.sam_adapter.image_encoder import Attention
    torch.manual_seed(0)
    att = Attention(dim=64, num_heads=8, use_rel_pos=True, input_size=(12, 12)).eval()
    torch.nn.init.normal_(att.rel_pos_h, std=0.5)
    torch.nn.init.normal_(att.rel_pos_w, std=0.5)
    x = torch.randn(2, 12, 12, 64)
    with torch.no_grad():
        ref = att(x)
        for chunk in (1, 3, 8):
            assert torch.allclose(chunked_attention_forward(att, x, chunk), ref, atol=1e-5)


def test_sam_adapter_lowmem_matches_original():
    from src.models.sam_adapter import SAMAdapter
    torch.manual_seed(0)
    plain = SAMAdapter(weights=None, embed_dim=768, device="cpu", lowmem=False)
    low = SAMAdapter(weights=None, embed_dim=768, device="cpu", lowmem=True)
    low.net.load_state_dict(plain.net.state_dict())
    img = np.random.default_rng(1).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    assert np.allclose(plain.predict(img), low.predict(img), atol=1e-4)
