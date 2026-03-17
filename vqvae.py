"""
vqvae_structural.py

PyTorch VQ-VAE + Structural Likelihood Model (Unary + Pairwise) for anomaly detection.

Stages
------
Stage I  : Train VQ-VAE on normal samples.
           Objective: L = ||x - x_hat||^2 + ||sg[z_e] - z_q||^2 + beta*||z_e - sg[z_q]||^2

Stage II : Estimate structural normality stats on discrete latent maps Z:
           * unary   p_{i,j}(k): positional frequency with Laplace smoothing
           * pairwise p(b|a)   : conditional transition probability for neighbors
             pair_prob[a,b] = p(b|a), normalized row-wise (sum over b = 1)

Stage III: Inference on test images:
           * Z = Z(x)                                   
           * S_unary(i,j) = -log p_{i,j}(Z_{i,j})     
           * S_pair(i,j)  = -log p(Z_neighbor | Z_{i,j})
           * S(i,j) = lambda1*S_unary + lambda2*S_pair  
           * s(x) = Aggregate({S(i,j)})              

Boundary note: pixels on the rightmost column / bottom row have no right/bottom neighbor,
so their S_pair contribution from that direction is 0 (correct behaviour).

Example (Marble / single-class anomaly detection):
  python vqvae_structural.py --mode train_and_fit \
      --data_dir ./data --img_size 128 --channels 3 --epochs 30 \
      --out_dir ./runs/marble_vqvae_struct
  python vqvae_structural.py --mode infer \
      --data_dir ./test --out_dir ./runs/marble_vqvae_struct \
      --ckpt ./runs/marble_vqvae_struct/last.pt \
      --stats ./runs/marble_vqvae_struct/struct_stats.pt

Example (local ImageFolder, train + fit_stats):
  python vqvae_structural.py --mode train_and_fit \
      --data_dir ./data/medical/train_normal \
      --img_size 128 --channels 1 --epochs 30 --out_dir ./runs/vqvae_struct

Example (Kaggle chest xray):
  python vqvae_structural.py --mode train_and_fit \\
      --kaggle_dataset paultimothymooney/chest-xray-pneumonia --kaggle_split train \\
      --img_size 128 --channels 1 --epochs 30 --out_dir ./runs/chestxray_vqvae_struct \\
      --class_order NORMAL,PNEUMONIA

Inference (use trained ckpt + stats):
  python vqvae_structural.py --mode infer \\
      --data_dir ./data/medical/test \\
      --out_dir ./runs/vqvae_struct \\
      --ckpt ./runs/vqvae_struct/last.pt \\
      --stats ./runs/vqvae_struct/struct_stats.pt \\
      --aggregate topk --topk 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image


# =========================================================
# Kaggle download helpers
# =========================================================

def try_kagglehub_download(dataset: str) -> Path:
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "kagglehub is not installed. Install it first:\n"
            "  pip install --user kagglehub\n"
        ) from e

    try:
        p = kagglehub.dataset_download(dataset)
        return Path(p)
    except Exception as e:
        raise RuntimeError(
            "Failed to download Kaggle dataset via kagglehub.\n"
            "Try:\n"
            "  python -c \"import kagglehub; kagglehub.login()\"\n"
            f"Original error: {e}"
        ) from e


def infer_chest_xray_split_root(download_root: Path, split: str) -> Path:
    split = split.lower()
    if split not in {"train", "test", "val", "valid", "validation"}:
        raise ValueError("--kaggle_split must be one of: train, test, val")

    if split in {"valid", "validation"}:
        split = "val"

    candidates: List[Path] = [
        download_root / "chest_xray" / split,
        download_root / "chest_xray" / "chest_xray" / split,
    ]
    for p in [download_root, download_root / "chest_xray", download_root / "chest_xray" / "chest_xray"]:
        if p.exists() and p.is_dir():
            for sub in ["chest_xray", "ChestXRay", "Chest_XRay", "data", "dataset"]:
                candidates.append(p / sub / split)

    for c in candidates:
        if c.exists() and c.is_dir():
            if any(x.is_dir() for x in c.iterdir()):
                return c

    raise FileNotFoundError(
        f"Could not locate split folder chest_xray/{split} under: {download_root}\n"
        "Please inspect the downloaded directory and pass --data_dir to the correct ImageFolder root."
    )


# =========================================================
# Dataset / Dataloader
# =========================================================

def remap_imagefolder_labels(
    dataset: torchvision.datasets.ImageFolder,
    class_order: List[str],
) -> Dict[str, int]:
    wanted = [c for c in class_order if c in dataset.class_to_idx]
    if len(wanted) != len(class_order):
        missing = [c for c in class_order if c not in dataset.class_to_idx]
        raise ValueError(f"Requested classes not found: {missing}. Found: {list(dataset.class_to_idx.keys())}")

    new_map = {cls: i for i, cls in enumerate(class_order)}
    new_samples: List[Tuple[str, int]] = []
    for fp, old_t in dataset.samples:
        cls_name = dataset.classes[old_t]
        if cls_name in new_map:
            new_samples.append((fp, new_map[cls_name]))
    dataset.samples = new_samples
    dataset.targets = [t for _, t in new_samples]
    dataset.class_to_idx = new_map
    dataset.classes = list(class_order)
    return new_map


def build_dataloader(
    data_dir: str,
    img_size: int,
    channels: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    class_order: Optional[List[str]] = None,
) -> Tuple[DataLoader, Dict[str, int]]:
    tfms = []
    if channels == 1:
        tfms.append(T.Grayscale(num_output_channels=1))
    tfms += [
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0, 1]
    ]
    transform = T.Compose(tfms)

    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    label_map = dataset.class_to_idx
    if class_order:
        label_map = remap_imagefolder_labels(dataset, class_order)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, label_map


# =========================================================
# Model: VQ-VAE
# =========================================================

class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),      # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, 4, 2, 1),            # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1),        # /8
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * c, 4 * c, 3, 1, 1),        # keep spatial size
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 4 * c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1),   # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1),        # x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, out_channels, 4, 2, 1), # x8
            nn.Sigmoid(),
        )

    def forward(self, zq: torch.Tensor) -> torch.Tensor:
        return self.net(zq)


class VectorQuantizer(nn.Module):
    """
    Standard VQ layer:
      codebook_loss = ||sg[z_e] - z_q||^2
      commit_loss   = beta * ||z_e - sg[z_q]||^2
      vq_loss       = codebook_loss + commit_loss
    Straight-through estimator passes gradients back to encoder.
    """

    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e: [B, D, H', W']
        returns:
          z_q_st  : [B, D, H', W']  (straight-through)
          indices : [B, H', W']     (long) — this is Z(x)
          vq_loss : scalar
        """
        B, D, H, W = z_e.shape
        assert D == self.code_dim, f"code_dim mismatch: got {D}, expected {self.code_dim}"

        # [B*H*W, D]
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # ||z - e||^2 = z^2 + e^2 - 2*z*e
        e = self.codebook.weight  # [K, D]
        z2 = (z_flat ** 2).sum(dim=1, keepdim=True)   # [B*H*W, 1]
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)          # [1, K]
        ze = 2 * (z_flat @ e.t())                      # [B*H*W, K]
        dist = z2 + e2 - ze                            # [B*H*W, K]

        indices = torch.argmin(dist, dim=1)            # [B*H*W]
        z_q = self.codebook(indices).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # VQ losses 
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss   = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        indices_map = indices.view(B, H, W)
        return z_q_st, indices_map, vq_loss


class VQVAE(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 1,
        base_channels: int = 64,
        num_codes: int = 512,
        code_dim: int = 64,
        vq_beta: float = 0.25,
    ):
        super().__init__()
        assert img_size % 8 == 0, f"img_size must be divisible by 8, got {img_size}"

        self.img_size = img_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_codes = num_codes
        self.code_dim = code_dim

        self.enc     = Encoder(in_channels, base_channels)
        self.pre_vq  = nn.Conv2d(4 * base_channels, code_dim, kernel_size=1)
        self.vq      = VectorQuantizer(num_codes=num_codes, code_dim=code_dim, beta=vq_beta)
        self.post_vq = nn.Conv2d(code_dim, 4 * base_channels, kernel_size=1)
        self.dec     = Decoder(out_channels=in_channels, base_channels=base_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns z_e: [B, code_dim, H', W']"""
        return self.pre_vq(self.enc(x))

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_q_st, indices [B,H',W'], vq_loss)"""
        return self.vq(z_e)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.dec(self.post_vq(z_q))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x)
        z_q, indices, vq_loss = self.quantize(z_e)
        x_hat = self.decode(z_q)
        return x_hat, indices, vq_loss


# =========================================================
# Stage I loss: recon + vq_loss  
# =========================================================

def vqvae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    vq_loss: torch.Tensor,
    recon_loss: str = "bce",
    recon_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = recon_weight * recon(x, x_hat) + codebook_loss + beta * commit_loss
    recon_loss='bce' uses binary cross-entropy (appropriate for [0,1] images).
    """
    if recon_loss == "bce":
        recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
    elif recon_loss == "mse":
        recon = F.mse_loss(x_hat, x, reduction="mean")
    else:
        raise ValueError("recon_loss must be 'bce' or 'mse'")
    total = recon_weight * recon + vq_loss
    return total, recon, vq_loss


# =========================================================
# Stage II: structural stats (unary + pairwise) 
# =========================================================

@torch.no_grad()
def fit_structural_stats(
    model: VQVAE,
    loader: DataLoader,
    device: torch.device,
    num_codes: int,
    alpha: float = 1.0,
    use_h: bool = True,
    use_v: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Stage II: estimate structural normality statistics from normal training images.

    Unary spatial distribution with Laplace smoothing α:
        p_{i,j}(k) = (#{n | Z^(n)_{i,j}=k} + α) / (Σ_{k'} #{...=k'} + K*α)
        → unary_prob[i, j, k]

    Pairwise spatial co-occurrence with Laplace smoothing α:
        p(b|a) = (#{(n,i,j) | Z=a, Z_neighbor=b} + α) / (Σ_{b'} #{...=b'} + K*α)
        → pair_prob[a, b]  where pair_prob[a, b] = p(b | a)
        Normalization: row-wise (sum over b = 1), i.e. pair_prob.sum(dim=1) ≈ ones(K).

    Returns dict saved as struct_stats.pt:
        unary_counts : [H', W', K]  — raw counts
        unary_prob   : [H', W', K]  — smoothed probabilities
        pair_counts  : [K, K]       — raw co-occurrence counts
        pair_prob    : [K, K]       — p(b|a), row-wise normalized
        meta         : [H', W', K]
        alpha        : [1]
        use_hv       : [use_h, use_v]
    """
    model.eval()

    unary_counts: Optional[torch.Tensor] = None
    pair_counts = torch.zeros(num_codes, num_codes, device="cpu", dtype=torch.float64)
    Hq = Wq = None

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z_e = model.encode(x)
        _, Z, _ = model.quantize(z_e)  # Z: [B, H', W']，Z^(n) = Z(x^(n))
        Z = Z.detach().to("cpu").long()

        B, H, W = Z.shape
        if unary_counts is None:
            Hq, Wq = H, W
            unary_counts = torch.zeros(H, W, num_codes, dtype=torch.float64)

        # ---- Unary counts ----
        for i in range(H):
            for j in range(W):
                idx = Z[:, i, j]  # [B]
                unary_counts[i, j].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))

        # ---- Pairwise counts ----
        # Horizontal: Z_{i,j} -> Z_{i,j+1}
        if use_h and W > 1:
            a = Z[:, :, :-1].reshape(-1)   # source (left)
            b = Z[:, :, 1:].reshape(-1)    # right neighbor
            pair_counts.index_put_((a, b), torch.ones_like(a, dtype=torch.float64), accumulate=True)

        # Vertical: Z_{i,j} -> Z_{i+1,j}
        if use_v and H > 1:
            a = Z[:, :-1, :].reshape(-1)   # source (top)
            b = Z[:, 1:, :].reshape(-1)    # bottom neighbor
            pair_counts.index_put_((a, b), torch.ones_like(a, dtype=torch.float64), accumulate=True)

    assert unary_counts is not None and Hq is not None and Wq is not None

    # ---- Laplace smoothing + normalization ----
    # normalize over k (dim=2)
    unary_prob = (unary_counts + alpha) / (unary_counts.sum(dim=2, keepdim=True) + num_codes * alpha)

    # normalize over b (dim=1) — row-wise, so pair_prob[a, b] = p(b|a)
    pair_prob = (pair_counts + alpha) / (pair_counts.sum(dim=1, keepdim=True) + num_codes * alpha)

    stats = {
        "unary_counts": unary_counts.float(),
        "unary_prob":   unary_prob.float(),    # [H', W', K]
        "pair_counts":  pair_counts.float(),
        "pair_prob":    pair_prob.float(),     # [K, K], pair_prob[a,b] = p(b|a)
        "meta":  torch.tensor([Hq, Wq, num_codes], dtype=torch.int32),
        "alpha": torch.tensor([alpha], dtype=torch.float32),
        "use_hv": torch.tensor([1 if use_h else 0, 1 if use_v else 0], dtype=torch.int32),
    }
    return stats


# =========================================================
# Stage III: inference 
# =========================================================

@torch.no_grad()
def compute_scores_for_batch(
    Z: torch.Tensor,                 # [B, H', W'] on device
    unary_prob: torch.Tensor,        # [H', W', K]
    pair_prob: torch.Tensor,         # [K, K], pair_prob[a,b] = p(b|a)
    lambda1: float,
    lambda2: float,
    use_h: bool = True,
    use_v: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute spatial anomaly score map S: [B, H', W'] (float32, CPU).

    S_unary(i,j) = -log p_{i,j}(Z_{i,j})
    S_pair(i,j)  = -log p(Z_{i,j+1} | Z_{i,j})  [horizontal]
                           + -log p(Z_{i+1,j} | Z_{i,j})  [vertical]
        Scores are accumulated at the source pixel (i,j).
        Boundary pixels (rightmost col / bottom row) have no corresponding neighbor,
        so their S_pair contribution from that direction is 0 — this is correct.
    S(i,j) = lambda1 * S_unary(i,j) + lambda2 * S_pair(i,j)
    """
    Zc = Z.detach().to("cpu").long()
    B, H, W = Zc.shape

    up = unary_prob.to("cpu")   # [H, W, K]
    pp = pair_prob.to("cpu")    # [K, K]

    # ---- S_unary ----
    S_unary = torch.zeros(B, H, W, dtype=torch.float32)
    for i in range(H):
        for j in range(W):
            k = Zc[:, i, j]                              # [B]
            p = up[i, j].gather(0, k).clamp_min(eps)    # [B]
            S_unary[:, i, j] = -torch.log(p)

    # ---- S_pair ----
    S_pair = torch.zeros(B, H, W, dtype=torch.float32)

    # Horizontal: S_pair(i,j) += -log p(Z_{i,j+1} | Z_{i,j})
    if use_h and W > 1:
        a = Zc[:, :, :-1]   # [B, H, W-1]  source
        b = Zc[:, :, 1:]    # [B, H, W-1]  right neighbor
        p = pp[a.reshape(-1), b.reshape(-1)].clamp_min(eps).reshape(B, H, W - 1)
        S_pair[:, :, :-1] += -torch.log(p)

    # Vertical: S_pair(i,j) += -log p(Z_{i+1,j} | Z_{i,j})
    if use_v and H > 1:
        a = Zc[:, :-1, :]   # [B, H-1, W]  source
        b = Zc[:, 1:, :]    # [B, H-1, W]  bottom neighbor
        p = pp[a.reshape(-1), b.reshape(-1)].clamp_min(eps).reshape(B, H - 1, W)
        S_pair[:, :-1, :] += -torch.log(p)

    # ---- Combined spatial score ----
    S = lambda1 * S_unary + lambda2 * S_pair
    return S  # [B, H, W]


def aggregate_score(S: torch.Tensor, mode: str = "max", topk: float = 0.02) -> torch.Tensor:
    """
    s(x) = Aggregate({S(i,j)})
    S   : [B, H', W'] (CPU)
    mode: max | mean | topk (top-k fraction averaging)
    topk: fraction in (0, 1], e.g. 0.02 = top 2% pixels
    Returns: [B]
    """
    B = S.size(0)
    flat = S.view(B, -1)

    if mode == "max":
        return flat.max(dim=1).values
    if mode == "mean":
        return flat.mean(dim=1)
    if mode == "topk":
        frac = max(min(topk, 1.0), 1e-6)
        k = max(int(frac * flat.size(1)), 1)
        vals, _ = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
        return vals.mean(dim=1)
    raise ValueError("aggregate must be one of: max, mean, topk")


# =========================================================
# Visualization helpers
# =========================================================

@torch.no_grad()
def save_reconstructions_vqvae(model: VQVAE, x: torch.Tensor, out_path: Path, n: int = 16):
    model.eval()
    x = x[:n]
    x_hat, _, _ = model(x)
    grid = make_grid(torch.cat([x.cpu(), x_hat.cpu()], dim=0), nrow=n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))


@torch.no_grad()
def save_samples_vqvae(
    model: VQVAE,
    out_path: Path,
    n: int = 16,
    device: torch.device | str = "cpu",
):
    """
    Naive debug sampler: random codes -> decode.
    Note: proper VQ-VAE generation requires a learned code prior (PixelCNN / Transformer).
    """
    model.eval()
    dummy = torch.zeros(1, model.in_channels, model.img_size, model.img_size, device=device)
    z_e = model.encode(dummy)
    _, _, Hq, Wq = z_e.shape
    idx = torch.randint(0, model.num_codes, (n, Hq, Wq), device=device)
    z_q = (
        model.vq.codebook(idx.reshape(-1))
        .view(n, Hq, Wq, model.code_dim)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    x_hat = model.decode(z_q).cpu()
    grid = make_grid(x_hat, nrow=int(n ** 0.5))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))


# =========================================================
# Checkpoint helpers
# =========================================================

def load_ckpt(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt


# =========================================================
# Main
# =========================================================

def main():
    p = argparse.ArgumentParser()

    # mode
    p.add_argument(
        "--mode", type=str, default="train_and_fit",
        choices=["train", "fit_stats", "train_and_fit", "infer"],
        help=(
            "train: Stage I only; "
            "fit_stats: Stage II only; "
            "train_and_fit: Stage I + II; "
            "infer: Stage III"
        ),
    )

    # data source (choose ONE)
    p.add_argument("--data_dir", type=str, default="./data", help="Path to ImageFolder dataset root (e.g. ./data, needs ./data/good/ subfolder).")
    p.add_argument("--kaggle_dataset", type=str, default="", help="Kaggle dataset slug.")
    p.add_argument("--kaggle_split", type=str, default="train", help="train/test/val for chest xray.")

    # output
    p.add_argument("--out_dir", type=str, default="./runs/vqvae_struct", help="Output directory.")

    # image & model
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_codes", type=int, default=512, help="K — codebook size.")
    p.add_argument("--code_dim", type=int, default=64, help="Embedding dimension of each code.")
    p.add_argument("--vq_beta", type=float, default=0.25, help="Commitment weight in VQ loss.")
    p.add_argument("--recon_loss", type=str, default="bce", choices=["bce", "mse"])
    p.add_argument("--recon_weight", type=float, default=1.0)

    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=1)

    # label mapping (optional)
    p.add_argument(
        "--class_order", type=str, default="",
        help="Comma-separated class names, e.g. NORMAL,PNEUMONIA",
    )

    # structural stats (Stage II)
    p.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha")
    p.add_argument("--use_h", action="store_true", help="Use horizontal pairwise stats (i,j)->(i,j+1)")
    p.add_argument("--use_v", action="store_true", help="Use vertical pairwise stats (i,j)->(i+1,j)")
    p.add_argument("--stats_out", type=str, default="", help="Where to save stats (default: out_dir/struct_stats.pt)")

    # inference weights & aggregation (Stage III)
    p.add_argument("--lambda1", type=float, default=1.0, help="Weight for S_unary")
    p.add_argument("--lambda2", type=float, default=1.0, help="Weight for S_pair ")
    p.add_argument("--aggregate", type=str, default="topk", choices=["max", "mean", "topk"],
                   help="Image-level aggregation mode")
    p.add_argument("--topk", type=float, default=0.02, help="Top-k fraction for aggregate=topk")
    p.add_argument("--save_heatmap", action="store_true", help="Save per-image score map as .pt")
    p.add_argument("--save_recon", action="store_true", help="Save reconstructions during inference")

    # load paths for infer / fit_stats
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path (default: out_dir/last.pt)")
    p.add_argument("--stats", type=str, default="", help="Stats path for infer (default: out_dir/struct_stats.pt)")

    args = p.parse_args()

    # default: use both directions if user does not specify
    if not args.use_h and not args.use_v:
        args.use_h = True
        args.use_v = True

    # resolve data_dir
    if args.kaggle_dataset:
        download_root = try_kagglehub_download(args.kaggle_dataset)
        data_root = infer_chest_xray_split_root(download_root, args.kaggle_split)
        args.data_dir = str(data_root)
    if not args.data_dir and args.mode in {"train", "fit_stats", "train_and_fit", "infer"}:
        raise SystemExit("You must provide --data_dir, or --kaggle_dataset to auto-download.")

    # parse class order
    class_order = None
    if args.class_order.strip():
        class_order = [c.strip() for c in args.class_order.split(",") if c.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save hparams
    (out_dir / "hparams.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    shuffle = False if args.mode == "infer" else True
    loader, label_map = build_dataloader(
        data_dir=args.data_dir,
        img_size=args.img_size,
        channels=args.channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        class_order=class_order,
    )

    with open(out_dir / "label_map.txt", "w", encoding="utf-8") as f:
        for k, v in label_map.items():
            f.write(f"{k}\t{v}\n")

    model = VQVAE(
        img_size=args.img_size,
        in_channels=args.channels,
        base_channels=args.base_channels,
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        vq_beta=args.vq_beta,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] VQVAE params: {total_params:,}")
    print(f"[Info] Latent map size: {args.img_size}x{args.img_size} -> "
          f"{args.img_size//8}x{args.img_size//8}, K={args.num_codes}")

    ckpt_path  = args.ckpt.strip()  or str(out_dir / "last.pt")
    stats_out  = args.stats_out.strip() or str(out_dir / "struct_stats.pt")
    stats_path = args.stats.strip() or str(out_dir / "struct_stats.pt")

    # -------------------------
    # Stage I: Train VQ-VAE
    # -------------------------
    if args.mode in {"train", "train_and_fit"}:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_hist: Dict[str, List[float]] = {"total": [], "recon": [], "vq": []}

        for epoch in range(1, args.epochs + 1):
            model.train()
            tot = rec = vqv = 0.0
            n_batches = 0

            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                x_hat, _, vq_loss_val = model(x)
                loss, recon, vq_loss_scalar = vqvae_loss(
                    x_hat, x, vq_loss_val,
                    recon_loss=args.recon_loss,
                    recon_weight=args.recon_weight,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                tot += float(loss.item())
                rec += float(recon.item())
                vqv += float(vq_loss_scalar.item())
                n_batches += 1

            scale = max(n_batches, 1)
            tot_m, rec_m, vqv_m = tot / scale, rec / scale, vqv / scale
            print(f"Epoch {epoch:03d} | loss={tot_m:.6f} | recon={rec_m:.6f} | vq={vqv_m:.6f}")

            loss_hist["total"].append(tot_m)
            loss_hist["recon"].append(rec_m)
            loss_hist["vq"].append(vqv_m)
            (out_dir / "loss_history.json").write_text(json.dumps(loss_hist, indent=2), encoding="utf-8")

            if epoch % args.save_every == 0:
                x_vis, _ = next(iter(loader))
                x_vis = x_vis.to(device, non_blocking=True)
                save_reconstructions_vqvae(model, x_vis, out_dir / f"recon_epoch{epoch:03d}.png", n=16)
                save_samples_vqvae(model, out_dir / f"samples_epoch{epoch:03d}.png", n=16, device=device)

                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "args": vars(args),
                    "label_map": label_map,
                }
                torch.save(ckpt, out_dir / "last.pt")

        print(f"[Info] Checkpoint saved to: {out_dir / 'last.pt'}")

    # -------------------------
    # Stage II: Fit Stats
    # -------------------------
    if args.mode in {"fit_stats", "train_and_fit"}:
        if args.mode == "fit_stats":
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            load_ckpt(model, ckpt_path, device)

        stats = fit_structural_stats(
            model=model,
            loader=loader,
            device=device,
            num_codes=args.num_codes,
            alpha=args.alpha,
            use_h=args.use_h,
            use_v=args.use_v,
        )

        # Sanity checks
        unary_sum    = stats["unary_prob"].sum(dim=2)
        pair_row_sum = stats["pair_prob"].sum(dim=1)
        print(f"[Info] unary_prob  sum check (should be ~1.0): "
              f"min={unary_sum.min():.6f}, max={unary_sum.max():.6f}")
        print(f"[Info] pair_prob row-sum check (should be ~1.0): "
              f"min={pair_row_sum.min():.6f}, max={pair_row_sum.max():.6f}")

        torch.save(stats, stats_out)
        print(f"[Info] Structural stats saved to: {stats_out}")

    # -------------------------
    # Stage III: Inference
    # -------------------------
    if args.mode == "infer":
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"Stats not found: {stats_path}")

        load_ckpt(model, ckpt_path, device)
        stats = torch.load(stats_path, map_location="cpu")

        unary_prob = stats["unary_prob"]   # [H', W', K]
        pair_prob  = stats["pair_prob"]    # [K, K], pair_prob[a,b] = p(b|a)
        use_hv = stats.get("use_hv", torch.tensor([1, 1], dtype=torch.int32))
        use_h  = bool(int(use_hv[0].item()))
        use_v  = bool(int(use_hv[1].item()))

        model.eval()
        results: List[Dict] = []
        heatmap_dir = out_dir / "heatmaps"
        recon_dir   = out_dir / "infer_recon"

        for idx_batch, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            z_e = model.encode(x)
            _, Z, _ = model.quantize(z_e)  # Z: [B, H', W']

            # Spatial score map 
            S = compute_scores_for_batch(
                Z=Z,
                unary_prob=unary_prob,
                pair_prob=pair_prob,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                use_h=use_h,
                use_v=use_v,
            )  # [B, H', W'] CPU

            # Image-level score 
            img_scores = aggregate_score(S, mode=args.aggregate, topk=args.topk).cpu()

            B = x.size(0)
            for b in range(B):
                results.append({
                    "index": idx_batch * loader.batch_size + b,
                    "label": int(y[b].item()) if torch.is_tensor(y) else int(y[b]),
                    "score": float(img_scores[b].item()),
                })

            if args.save_heatmap:
                heatmap_dir.mkdir(parents=True, exist_ok=True)
                torch.save(S, heatmap_dir / f"S_batch{idx_batch:05d}.pt")

            if args.save_recon:
                recon_dir.mkdir(parents=True, exist_ok=True)
                x_hat, _, _ = model(x)
                grid = make_grid(torch.cat([x.cpu(), x_hat.cpu()], dim=0), nrow=min(16, B))
                save_image(grid, str(recon_dir / f"recon_batch{idx_batch:05d}.png"))

        (out_dir / "infer_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[Info] Inference scores saved to: {out_dir / 'infer_scores.json'}")
        print(f"[Info] Total samples scored: {len(results)}")


if __name__ == "__main__":
    main()
