# ECE 285 Final Project — Industrial Anomaly Detection
## By Tien-Hao Chen, Yi-Chi Wang, Jasmine Lou
This project contains two main notebook pipelines for image anomaly detection:
1. **Baseline VAE** (`vae.ipynb`)
2. **VQ-VAE + Structural Likelihood** (`vqvae.ipynb`)

The notebooks are the **primary, up-to-date versions**. The `.py` scripts are older/less featured variants and are kept for reference.

---

## Project Structure
- `vae.ipynb` — Baseline VAE anomaly detection (primary).
- `vqvae.ipynb` — VQ-VAE + structural stats pipeline (primary).
- `vae.py` — Legacy VAE script.
- `vqvae.py` — Legacy VQ-VAE + structural stats script.

---

## Dataset
We use the **Real-Life Industrial Dataset of Casting Product** (Kaggle).  
The dataset is not included in this repo; please download it yourself from:
```
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
```

---

## Data Layout
Both notebooks auto-detect **either** of the following formats:

Option A — **Flat folder** (all images in one folder):
```
train2/
  img001.png
  img002.png
  ...

test4/
  img101.png
  img102.png
  ...
```

Option B — **ImageFolder** style (with class subfolders):
```
train2/
  good/
    ...
  defect/
    ...

test4/
  good/
    ...
  defect/
    ...
```

Notes:
- The loader **forces `good=0`, `defect=1`** if folder names contain `good` / `defect` / `bad` / `ng`.
- If no subfolders are found, it falls back to flat-folder mode.

---

## Baseline VAE — `vae.ipynb`
### What it does
- Trains a CNN VAE with **MSE reconstruction loss + KLD**.
- Scores anomalies by **top-k pixel reconstruction error** (configurable).
- Produces AUROC and reconstruction quality metrics (MSE/PSNR/SSIM).

### How to run
1. Open `vae.ipynb`.
2. Update `CFG["train_dir"]` and `CFG["test_dir"]`.
3. Run all cells from top to bottom.

### Key config (Config cell)
- `img_size`, `channels`, `batch_size`, `epochs`
- `latent_dim`, `kld_weight`
- `aggregate`, `topk`

### Outputs (default `./runs/vae_marble`)
- `hparams.json`
- `loss_history.json`
- `last.pt` (checkpoint)
- `infer_scores.json`
- `recon_metrics.json` (MSE/PSNR/SSIM + AUROC)
- Optional visualization images (reconstructions, plots)

---

## VQ-VAE + Structural Likelihood — `vqvae.ipynb`
### What it does
- **Stage I**: Train VQ-VAE (EMA codebook + dead-code restart).
- **Stage II**: Fit structural stats on discrete latent maps (unary + pairwise).
- **Stage III**: Infer anomaly scores using structural likelihood.

### How to run
1. Open `vqvae.ipynb`.
2. Update `CFG["train_dir"]` and `CFG["test_dir"]`.
3. Run all cells from top to bottom.

### Key config (Config cell)
- `img_size`, `channels`, `batch_size`, `epochs`
- `num_codes`, `code_dim`, `vq_beta`
- `vq_ema`, `vq_ema_decay`, `vq_restart_threshold`
- `alpha`, `use_h`, `use_v` (structural stats)
- `lambda1`, `lambda2`, `aggregate`, `topk`

### Outputs (default `./runs/vqvae_marble`)
- `hparams.json`
- `loss_history.json`
- `last.pt` (VQ-VAE checkpoint)
- `struct_stats.pt` (unary/pairwise statistics)
- `infer_scores.json`
- Optional visualization images (reconstructions, plots)

---

## Notes on `.py` Scripts
The `.py` files are **older script versions** and differ in model details:
- `vqvae.py` uses a standard VQ-VAE (no EMA, different decoder).
- `vae.py` is a plain training/inference script with fewer visuals.

If you want the most complete workflow, **use the notebooks**.
