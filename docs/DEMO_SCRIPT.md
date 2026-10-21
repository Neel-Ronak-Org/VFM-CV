# Demo script (about 60 seconds)

**Before the presentation**
1. Stop any benchmark or training run, because the demo needs the GPU.
2. `.venv\Scripts\activate`, then `python demo/app.py`. Wait for "ready" and for the browser tab to open. Models warm up at start, so the first latency shown is already representative.
3. Pick your images in advance:
   - Choose 2–3 gallery images where the object is clearly camouflaged, and list them in `demo/examples.txt` as `COD10K/<name>` (one per line) to put them first.
   - Add 1–2 of your own photos to `demo/my_photos/` (for example, a plant-covered wall or an insect on bark).
4. Open the "Benchmark results" tab once and click Refresh, so the latest committed numbers are loaded.

**Shot list**

| Time | Show | Say |
|---|---|---|
| 0–8 s | Title + gallery | "Camouflaged objects blend into the background, like animals for wildlife cameras or people in search and rescue. Here are test images from COD10K, CAMO and CHAMELEON." |
| 8–20 s | Click a gallery image; tick *Show ground truth* | "Left: SINet-V2, a well-known specialist model. Right: our adapter on a frozen MobileSAM. White is the true object. Each model's latency on this laptop's 4 GB GPU is shown under it." |
| 20–35 s | Corruption = *gaussian_blur (seen)*, severity 1 → 5 | "Real cameras blur, compress and add noise. As blur gets stronger, watch how each model's mask changes. Our model was trained with this kind of corruption." |
| 35–48 s | Corruption = *motion_blur (UNSEEN)* or *fog (UNSEEN)*, severity 3–4 | "This corruption was **never** used in training. Testing on held-out corruptions shows whether robustness transfers, which is a key part of our protocol." |
| 48–55 s | Drag your own photo onto the input box; tick one or two more models (e.g. Ours-Clean, SAM2-UNet) | "It works on new photos with no clicks or prompts, and I can compare any of the models side by side." |
| 55–60 s | *Benchmark results* tab | "Every number here is read from the benchmark's results files: clean accuracy, robustness retention, and speed on this GPU." |

**Be honest if asked:**
- Results on single images vary. The benchmark tables, not one example, are the evidence.
- If a checkpoint is labelled `@ep…-unfinished`, say training wasn't complete at demo time.
- The robustness benchmark may still be running. The `complete` column shows which groups are finished.
