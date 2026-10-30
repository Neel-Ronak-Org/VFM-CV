# Our model: a prompt-free MobileSAM adapter

Implementation: `src/models/robust_adapter.py` (model), `src/losses.py` (losses), `src/train.py` (training), configs in `configs/train/`.

## Architecture

| Part | Role | Trained | Parameters |
|---|---|---|---|
| TinyViT image encoder (MobileSAM) | image → grid of 256-d features (64×64 at 1024 px input) | frozen | 6.07 M |
| Bottleneck adapters, one per transformer block (10) | `x + MLP(x) + W_up GELU(W_down LN(x))`, hidden width = block width / 4 | yes | 0.20 M |
| Learned prompts: 4 tokens + a dense prompt | replace the clicks or boxes SAM needs | yes | 1.3 k |
| SAM mask decoder | two-way transformer, 4× upsampling | yes (from MobileSAM) | 3.51 M |

3.71 M of the 10.33 M parameters are trained (`RobustSAMAdapter.count_params()`).

- **Zero-initialised adapters.** `W_up` starts at zero, so training starts from exactly MobileSAM's features (checked by `tests/test_robust_adapter.py`).
- **Prompt-free decoding.** Two prompt tokens start as SAM's embedding of a box around the whole image and two start small and random. SAM's decoder repeats each image embedding once per prompt, which would mix images inside a batch; `decode()` computes the same result once per image (a test checks it matches SAM's decoder).
- **Input size.** 1024 px (Ours (1024)) or 512 px with interpolated position grid (Ours (512)). The output is upsampled to the input image's size.

## Training

- **Data:** COD10K-train + CAMO-train (4,040 images). Random crop (75–100 % of each side) and horizontal flip only; no colour jitter or noise, since these overlap with the test corruptions.
- **Loss:** weighted BCE + weighted IoU (the structure loss of PraNet and SINet-V2); the boundary window is scaled from 31 px at 352 px input to the training resolution.
- **Corruption augmentation (Ours-Aug):** 50 % of samples get one of the four seen corruptions (Gaussian noise, Gaussian blur, brightness, JPEG) at a random severity 1–5.
- **Consistency loss (Ours-Full):** every image gives a clean and a corrupted view with the same crop and flip; the supervised loss is averaged over both views, plus λ · MSE between the two predicted maps, with stop-gradient on the clean view (λ = 1). Ablations: λ ∈ {0.1, 0.5}, symmetric KL, no stop-gradient, LoRA instead of adapters, no adapters.
- **Optimiser:** AdamW, learning rate 2e-4, 100 warm-up steps then cosine decay, effective batch 8 (2 × 4 accumulation), fp16 mixed precision, gradient clipping at 1.0, 30 epochs.
- **Memory:** the frozen convolutional stage runs without gradients; gradient checkpointing inside the transformer blocks. If the GPU is full or slow (`--profile`), step down: (1) 1024 px, batch 2; (2) `--set train.batch_size=1 train.grad_accum=8`; (3) `--set data.img_size=512`.
- **Reproducibility:** image order and every crop, flip and corruption are seeded by (seed, epoch, image index); `--resume` continues on exactly the same batch sequence. Each run writes its config, git hash, loss curve and timings to `results/train/<run>/`.
- **Model selection:** the last epoch is used; checkpoints are never selected by test score. A checkpoint from an unfinished run is labelled `…@ep<x>-unfinished` in every output.

## Variants

| Name | Config | Difference |
|---|---|---|
| Ours-Clean | `ours_clean.yaml` | adapters + decoder, clean images only |
| Ours-Aug | `ours_aug.yaml` | + corruption augmentation |
| Ours-Full = Ours (1024) | `ours_full.yaml` | + consistency loss |
| Ours-Full-512 = Ours (512) | `ours_full_512.yaml` (+ `_s1`, `_s2` seeds) | trained and tested at 512 px |
