"""Our adapter: architecture invariants, losses, training data, and the training loop (CPU, random weights)."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.datasets import CODDataset, CODTrainSet
from src.losses import box_filter, boundary_window, consistency_loss, structure_loss
from src.models.robust_adapter import RobustAdapterCOD, RobustSAMAdapter, load_trained
from src.models.third_party.mobile_sam.modeling import TinyViT


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return RobustSAMAdapter(img_size=256, sam_weights=None).eval()


@pytest.fixture
def train_root(tmp_path):
    root = tmp_path / "TrainDataset"
    (root / "Imgs").mkdir(parents=True)
    (root / "GT").mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        img = rng.integers(0, 256, (90 + 2 * i, 120, 3), dtype=np.uint8)
        m = np.zeros(img.shape[:2], np.uint8)
        m[30:60, 40:90] = 255
        cv2.imwrite(str(root / "Imgs" / f"t{i}.jpg"), img)
        cv2.imwrite(str(root / "GT" / f"t{i}.png"), m)
    return tmp_path


def test_encoder_with_zero_adapters_equals_tinyvit_1024():
    """At init the adapted encoder must compute exactly MobileSAM's TinyViT.forward_features."""
    torch.manual_seed(0)
    net = RobustSAMAdapter(img_size=1024, sam_weights=None).eval()
    plain = TinyViT(img_size=1024, num_classes=1000, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
                    num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], drop_path_rate=0.0, layer_lr_decay=0.8)
    sd = {k.replace("mlp.mlp.", "mlp."): v for k, v in net.encoder.state_dict().items() if ".adapter." not in k}
    plain.load_state_dict(sd)
    plain.eval()
    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        assert torch.allclose(net.encode(x), plain.forward_features(x), atol=1e-5)


def test_batched_decode_matches_sam_decoder_per_image(net):
    emb = torch.randn(2, 256, net.grid, net.grid)
    with torch.no_grad():
        ours = net.decode(emb)
        pe = net.pe_layer((net.grid, net.grid)).unsqueeze(0)
        dense = net.dense_embed.view(1, 256, 1, 1).expand(1, 256, net.grid, net.grid)
        for i in range(2):
            ref, _ = net.decoder(emb[i:i + 1], pe, net.prompt_tokens, dense, multimask_output=False)
            assert torch.allclose(ours[i:i + 1], ref, atol=1e-4)


def test_output_shape_and_frozen_parts(net):
    with torch.no_grad():
        out = net(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1, 256, 256)
    names = net.trainable_names()
    assert names and all(not n.startswith("encoder.") or ".adapter." in n for n in names)
    assert not any("iou_prediction_head" in n for n in names)
    assert "prompt_tokens" in names and "dense_embed" in names
    adapter = sum(p.numel() for n, p in net.named_parameters() if ".adapter." in n)
    assert 100_000 < adapter < 400_000  # ~0.2M bottleneck parameters


@pytest.mark.parametrize("adapter", ["bottleneck", "lora", "none"])
def test_one_step_updates_only_trainable(adapter):
    torch.manual_seed(0)
    net = RobustSAMAdapter(img_size=128, adapter=adapter, sam_weights=None).train()
    assert not net.encoder.training  # BatchNorm stays frozen
    frozen = {k: v.clone() for k, v in net.encoder.state_dict().items() if "adapter" not in k and "lora" not in k}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-2)
    loss = structure_loss(net(torch.randn(2, 3, 128, 128)), (torch.rand(2, 1, 128, 128) > 0.5).float())
    loss.backward()
    opt.step()
    for k, v in net.encoder.state_dict().items():
        if k in frozen:
            assert torch.equal(v, frozen[k]), k
    if adapter != "none":
        ad = [p for n, p in net.named_parameters() if "adapter" in n or "lora" in n]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ad)


def test_gradient_checkpointing_same_gradients():
    torch.manual_seed(0)
    a = RobustSAMAdapter(img_size=128, sam_weights=None).train()
    b = RobustSAMAdapter(img_size=128, sam_weights=None, grad_checkpointing=True).train()
    b.load_state_dict(a.state_dict())
    for n, p in a.named_parameters():  # make adapter outputs non-zero so their grads are meaningful
        if n.endswith("adapter.up.weight"):
            torch.nn.init.normal_(p, std=0.01)
    b.load_state_dict(a.state_dict())
    x, m = torch.randn(1, 3, 128, 128), (torch.rand(1, 1, 128, 128) > 0.5).float()
    for net in (a, b):
        structure_loss(net(x), m).backward()
    ga = dict((n, p.grad) for n, p in a.named_parameters() if p.requires_grad)
    for n, p in b.named_parameters():
        if p.requires_grad and ga[n] is not None:
            assert torch.allclose(p.grad, ga[n], atol=1e-5), n


def test_checkpoint_roundtrip_through_wrapper(net, tmp_path):
    # A fake "mobile_sam.pt" with the frozen weights, plus our trainable checkpoint.
    sam_sd = {f"image_encoder.{k.replace('mlp.mlp.', 'mlp.')}": v for k, v in net.encoder.state_dict().items()
              if ".adapter." not in k}
    from src.models.robust_adapter import build_mobile_sam_parts
    _, pe, _ = build_mobile_sam_parts(256)
    sam_sd.update({f"prompt_encoder.{k}": v for k, v in pe.state_dict().items()})
    sam_sd.update({f"prompt_encoder.pe_layer.{k}": v for k, v in net.pe_layer.state_dict().items()})
    sam_sd.update({f"mask_decoder.{k}": v for k, v in net.decoder.state_dict().items()})
    torch.save(sam_sd, tmp_path / "sam.pt")
    torch.save({"trainable": net.trainable_state_dict(), "model_config": net.config, "epoch": 3, "finished": True},
               tmp_path / "final.pt")
    loaded, _ = load_trained(tmp_path / "final.pt", sam_weights=tmp_path / "sam.pt")
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        assert torch.allclose(loaded(x), net(x), atol=1e-5)
    model = RobustAdapterCOD(checkpoint=tmp_path / "final.pt", sam_weights=tmp_path / "sam.pt", device="cpu")
    pred = model.predict(np.random.default_rng(0).integers(0, 256, (70, 90, 3), dtype=np.uint8))
    assert pred.shape == (70, 90) and pred.dtype == np.float32 and 0 <= pred.min() and pred.max() <= 1
    assert model.epoch == 3 and model.name == "Ours"
    torch.save({"trainable": net.trainable_state_dict(), "model_config": net.config, "epoch": 1.5, "finished": False},
               tmp_path / "last.pt")
    partial = RobustAdapterCOD(checkpoint=tmp_path / "last.pt", sam_weights=tmp_path / "sam.pt", device="cpu")
    assert partial.name == "Ours@ep1.5-unfinished"


def test_load_rejects_mismatched_checkpoint(net):
    sd = net.trainable_state_dict()
    sd.pop("prompt_tokens")
    with pytest.raises(RuntimeError, match="mismatch"):
        net.load_trainable_state_dict(sd)


def test_structure_loss_and_box_filter():
    x = torch.rand(2, 1, 33, 41)
    assert torch.allclose(box_filter(x, 7), F.avg_pool2d(x, 7, 1, 3), atol=1e-6)
    assert boundary_window(352) == 31 and boundary_window(1024) % 2 == 1
    m = torch.zeros(1, 1, 64, 64)
    m[..., 16:48, 16:48] = 1
    good, bad = structure_loss((m * 2 - 1) * 10, m), structure_loss((1 - 2 * m) * 10, m)
    assert good < 0.05 < bad


@pytest.mark.parametrize("kind", ["mse", "sym_kl"])
def test_consistency_loss(kind):
    a = torch.randn(2, 1, 8, 8, requires_grad=True)
    b = torch.randn(2, 1, 8, 8, requires_grad=True)
    assert consistency_loss(a, a.detach().clone(), kind).item() == pytest.approx(0, abs=1e-7)
    consistency_loss(a, b, kind, stopgrad_clean=True).backward()
    assert a.grad is None and b.grad is not None
    a2 = a.detach().clone().requires_grad_()
    consistency_loss(a2, b, kind, stopgrad_clean=False).backward()
    assert a2.grad is not None


def test_trainset_deterministic_and_paired(train_root):
    base = CODDataset(train_root / "TrainDataset")
    ds = CODTrainSet(base, 64, corruption_prob=0.5, corruptions=["gaussian_noise", "brightness"])
    a, b = ds[1], ds[1]
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])
    assert a["image"].shape == (3, 64, 64) and a["image"].dtype == np.uint8
    assert a["mask"].shape == (1, 64, 64) and 0 <= a["mask"].min() and a["mask"].max() <= 1
    ds.set_epoch(1)
    assert not np.array_equal(ds[1]["image"], a["image"]) or not np.array_equal(ds[1]["mask"], a["mask"])
    flags = [ds[i]["corrupted"] for i in range(len(ds))]
    assert any(flags) or not all(flags)
    clean = CODTrainSet(base, 64)
    assert not any(clean[i]["corrupted"] for i in range(len(clean)))
    paired = CODTrainSet(base, 64, corruptions=["gaussian_noise"], paired=True)
    p = paired[0]
    assert p["image"].shape == p["image_corr"].shape and not np.array_equal(p["image"], p["image_corr"])


def _cfg(name, *extra):
    from src.train import load_config
    return load_config(Path("configs/train") / f"{name}.yaml",
                       ["data.img_size=128", "data.num_workers=0", "train.batch_size=2", "train.grad_accum=2",
                        "train.epochs=2", "train.log_every=1", "train.warmup_steps=1", "model.sam_weights=null", *extra])


def test_configs_load_and_respect_held_out_protocol():
    from src.corruptions import load_protocol
    from src.train import load_config, resolve_corruptions
    proto = load_protocol()
    for p in Path("configs/train").glob("*.yaml"):
        cfg = load_config(p)
        assert cfg["run_name"] == p.stem
        assert set(resolve_corruptions(cfg["augment"], proto)) <= set(proto["seen"])
    assert load_config("configs/train/ours_full.yaml")["consistency"]["weight"] > 0
    assert load_config("configs/train/ours_aug.yaml")["augment"]["corruption_prob"] == 0.5
    with pytest.raises(ValueError, match="UNSEEN"):
        resolve_corruptions({"corruptions": ["motion_blur"]}, proto)


def test_train_end_to_end_and_resume(train_root, tmp_path):
    from src.train import train
    kw = dict(device="cpu", data_root=train_root, ckpt_root=tmp_path / "ck", log_root=tmp_path / "logs")
    s = train(_cfg("ours_full"), max_hours=1e-9, **kw)  # stops after the first optimizer step
    assert s["status"] == "stopped" and (tmp_path / "ck/ours_full/last.pt").exists()
    assert not (tmp_path / "ck/ours_full/final.pt").exists()
    with pytest.raises(RuntimeError, match="config differs"):
        train(_cfg("ours_full", "consistency.weight=0.5"), resume=True, **kw)
    s = train(_cfg("ours_full"), resume=True, **kw)
    assert s["status"] == "finished" and s["epochs_done"] == 2 and s["optimizer_steps"] == 4
    ck = torch.load(tmp_path / "ck/ours_full/final.pt", weights_only=False)
    assert ck["finished"] and "optimizer" not in ck
    log = (tmp_path / "logs/ours_full/log.csv").read_text().splitlines()
    assert log[0].startswith("epoch,step") and len(log) == 5  # header + 4 optimizer steps
    summary = (tmp_path / "logs/ours_full/summary.json").read_text()
    # resuming a finished run changes nothing (it used to overwrite summary.json with empty timings)
    assert train(_cfg("ours_full"), resume=True, **kw)["status"] == "already_finished"
    assert (tmp_path / "logs/ours_full/summary.json").read_text() == summary


@pytest.mark.parametrize("budget,epochs", [(1e-9, 1), (100, 2)])
def test_time_budget_fits_epochs_after_first_epoch(train_root, tmp_path, budget, epochs):
    from src.train import train
    s = train(_cfg("ours_clean", f"train.time_budget_hours={budget}"), device="cpu", data_root=train_root,
              ckpt_root=tmp_path / "ck", log_root=tmp_path / "logs")
    assert s["status"] == "finished" and s["epochs_planned"] == epochs == s["epochs_done"]


def test_profile_writes_row_and_no_checkpoint(train_root, tmp_path):
    from src.train import train
    row = train(_cfg("ours_clean"), profile=2, device="cpu", data_root=train_root,
                ckpt_root=tmp_path / "ck", log_root=tmp_path / "logs")
    assert "epoch_minutes" in row and (tmp_path / "logs/profile.csv").exists()
    assert not (tmp_path / "ck").exists()
