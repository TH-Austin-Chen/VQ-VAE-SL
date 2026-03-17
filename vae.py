"""
vae.py — VAE Anomaly Detection 

"""
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ======================================================================
# Config
# ======================================================================
CFG = {
    "train_dir":     "./data",
    "test_dir":      "./test",
    "img_size":      256,
    "channels":      1,
    "normalize":     True,
    "batch_size":    32,
    "num_workers":   2,
    "out_dir":       "./runs/vae_marble",
    "save_every":    5,
    "base_channels": 64,
    "latent_dim":    128,
    "dropout":       0.1,
    "epochs":        80,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "kld_weight":    0.001,
    "aggregate":     "topk",
    "topk":          0.05,
}

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path(CFG["out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Dataset / Dataloader
# ======================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

class FlatFolderDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.filepaths = sorted([
            p for p in self.data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])
        if len(self.filepaths) == 0:
            raise FileNotFoundError(f"No images found in {data_dir}")
        class_name        = self.data_dir.name
        self.classes      = [class_name]
        self.class_to_idx = {class_name: 0}
        self.targets      = [0] * len(self.filepaths)

    def __len__(self): return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, 0


def build_transform(img_size, channels, augment=False):
    tfms = []
    if channels == 1:
        tfms.append(T.Grayscale(num_output_channels=1))
    if augment:
        crop_size = int(img_size * 1.15)
        tfms += [
            T.Resize((crop_size, crop_size)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=(90,90))], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
        ]
    else:
        tfms.append(T.Resize((img_size, img_size)))
    tfms.append(T.ToTensor())
    if CFG.get("normalize", False):
        mean = [0.5] * channels
        tfms.append(T.Normalize(mean=mean, std=mean))
    return T.Compose(tfms)


def make_loader(data_dir, shuffle):
    transform = build_transform(CFG["img_size"], CFG["channels"], augment=shuffle)
    root = Path(data_dir)
    valid_subdirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')]

    if valid_subdirs:
        import torchvision.datasets as tvd
        class FilteredImageFolder(tvd.ImageFolder):
            def find_classes(self, directory):
                classes = [d.name for d in Path(directory).iterdir()
                           if d.is_dir() and not d.name.startswith('.')]
                classes.sort()
                priority = {}
                for c in classes:
                    if 'good' in c.lower(): priority[c] = 0
                    elif any(k in c.lower() for k in ['defect','bad','ng']): priority[c] = 1
                    else: priority[c] = 2 + classes.index(c)
                classes = sorted(classes, key=lambda c: priority[c])
                return classes, {c: i for i, c in enumerate(classes)}
        ds = FilteredImageFolder(root=data_dir, transform=transform)
    else:
        ds = FlatFolderDataset(data_dir=data_dir, transform=transform)

    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=True, drop_last=False), \
           ds.class_to_idx, ds


# ======================================================================
# VAE Model
# ======================================================================
class VAEEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, latent_dim, dropout=0.0):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(c, 2*c, 4, 2, 1),
            nn.BatchNorm2d(2*c), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(2*c, 4*c, 4, 2, 1),
            nn.BatchNorm2d(4*c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*c, 4*c, 3, 1, 1),
            nn.BatchNorm2d(4*c), nn.LeakyReLU(0.2, inplace=True),
        )
        dummy = torch.zeros(1, in_channels, CFG["img_size"], CFG["img_size"])
        feat  = self.net(dummy)
        self.feat_shape = feat.shape[1:]
        flat_dim = int(feat.numel())
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, latent_dim, feat_shape, dropout=0.0):
        super().__init__()
        c = base_channels
        self.feat_shape = feat_shape
        self.fc = nn.Linear(latent_dim, int(feat_shape[0]*feat_shape[1]*feat_shape[2]))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4*c, 4*c, 3, 1, 1),
            nn.BatchNorm2d(4*c), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(4*c, 2*c, 4, 2, 1),
            nn.BatchNorm2d(2*c), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(2*c, c, 4, 2, 1),
            nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c, out_channels, 4, 2, 1),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, *self.feat_shape))


class VAE(nn.Module):
    def __init__(self, img_size=256, in_channels=1, base_channels=64,
                 latent_dim=128, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels, base_channels, latent_dim, dropout)
        self.decoder = VAEDecoder(in_channels, base_channels, latent_dim,
                                  self.encoder.feat_shape, dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar, kld_weight=0.001):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld_weight * kld, recon, kld


# ======================================================================
# Inference
# ======================================================================
@torch.no_grad()
def infer_scores_vae(model, loader, device):
    model.eval()
    results = []
    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        x_hat, _, _ = model(x)
        if CFG.get("normalize", False):
            x     = x     * 0.5 + 0.5
            x_hat = x_hat * 0.5 + 0.5
        err_map = (x - x_hat).pow(2).view(x.size(0), -1).cpu()
        if CFG["aggregate"] == "max":
            scores = err_map.max(1).values
        elif CFG["aggregate"] == "mean":
            scores = err_map.mean(1)
        else:
            k = max(int(CFG["topk"] * err_map.size(1)), 1)
            scores = torch.topk(err_map, k, dim=1).values.mean(1)
        for b in range(x.size(0)):
            results.append({"index": bi*CFG["batch_size"]+b,
                             "label": int(y[b]), "score": float(scores[b])})
    return results


@torch.no_grad()
def compute_reconstruction_metrics(model, dataset, device, label_name=""):
    model.eval()
    mse_list = []; psnr_list = []; ssim_list = []
    loader = DataLoader(dataset, batch_size=CFG["batch_size"],
                        shuffle=False, num_workers=CFG["num_workers"])
    for x, _ in loader:
        x = x.to(device)
        x_hat, _, _ = model(x)
        if CFG.get("normalize", False):
            x = x * 0.5 + 0.5; x_hat = x_hat * 0.5 + 0.5
        x = x.clamp(0,1).cpu(); x_hat = x_hat.clamp(0,1).cpu()
        for i in range(x.size(0)):
            orig = x[i].numpy(); recon = x_hat[i].numpy()
            mse  = float(((orig-recon)**2).mean())
            mse_list.append(mse)
            psnr_list.append(10*math.log10(1/mse) if mse > 1e-10 else 100.0)
            if SKIMAGE_OK:
                o = orig[0]  if orig.shape[0]==1  else orig.transpose(1,2,0)
                r = recon[0] if recon.shape[0]==1 else recon.transpose(1,2,0)
                ssim_list.append(float(ssim_fn(o, r, data_range=1.0,
                                               channel_axis=None if orig.shape[0]==1 else -1)))
    res = {"label": label_name, "n": len(mse_list),
           "mse_mean": float(np.mean(mse_list)), "mse_std": float(np.std(mse_list)),
           "psnr_mean": float(np.mean(psnr_list)), "psnr_std": float(np.std(psnr_list))}
    if SKIMAGE_OK:
        res["ssim_mean"] = float(np.mean(ssim_list))
        res["ssim_std"]  = float(np.std(ssim_list))
    return res


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("device:", device)

    # Loaders
    train_loader, train_label_map, train_ds = make_loader(CFG["train_dir"], shuffle=True)
    test_loader,  test_label_map,  test_ds  = make_loader(CFG["test_dir"],  shuffle=False)
    print(f"train: {len(train_ds)}  test: {len(test_ds)}")

    # Model
    model = VAE(CFG["img_size"], CFG["channels"], CFG["base_channels"],
                CFG["latent_dim"], CFG["dropout"]).to(device)
    print(f"VAE params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    loss_hist = {"total":[], "recon":[], "kld":[]}

    for epoch in range(1, CFG["epochs"]+1):
        model.train()
        tot = rec = kldv = nb = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, recon, kld  = vae_loss(x_hat, x, mu, logvar, CFG["kld_weight"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss); rec += float(recon); kldv += float(kld); nb += 1
        t=tot/nb; r=rec/nb; k=kldv/nb
        print(f"Epoch {epoch:03d} | loss={t:.6f} | recon={r:.6f} | kld={k:.6f}")
        loss_hist["total"].append(t); loss_hist["recon"].append(r); loss_hist["kld"].append(k)
        if epoch % CFG["save_every"] == 0:
            torch.save({"epoch":epoch,"model_state":model.state_dict(),"cfg":CFG},
                       out_dir/"last.pt")

    print("Training done.")

    # Inference
    infer_results = infer_scores_vae(model, test_loader, device)
    (out_dir/"infer_scores.json").write_text(json.dumps(infer_results, indent=2))

    scores = np.array([r["score"] for r in infer_results])
    labels = np.array([r["label"] for r in infer_results])

    if SKLEARN_OK and (labels==1).any():
        auroc = roc_auc_score(labels, scores)
        print(f"AUROC = {auroc:.4f} ({auroc*100:.1f}%)")

    # Metrics
    good_idx   = [i for i,(_, l) in enumerate(test_ds) if l==0]
    defect_idx = [i for i,(_, l) in enumerate(test_ds) if l==1]
    gm = compute_reconstruction_metrics(model, torch.utils.data.Subset(test_ds, good_idx),   device, "good")
    dm = compute_reconstruction_metrics(model, torch.utils.data.Subset(test_ds, defect_idx), device, "defect")

    print(f"\n{'Metric':<18} {'Good':>16} {'Defect':>16}")
    print("-"*55)
    print(f"{'MSE (↓)':<18} {gm['mse_mean']:>12.5f}±{gm['mse_std']:.4f} {dm['mse_mean']:>12.5f}±{dm['mse_std']:.4f}")
    print(f"{'PSNR dB (↑)':<18} {gm['psnr_mean']:>12.2f}±{gm['psnr_std']:.2f} {dm['psnr_mean']:>12.2f}±{dm['psnr_std']:.2f}")
    if SKIMAGE_OK:
        print(f"{'SSIM (↑)':<18} {gm['ssim_mean']:>12.4f}±{gm['ssim_std']:.4f} {dm['ssim_mean']:>12.4f}±{dm['ssim_std']:.4f}")

    comparison = {"model":"VAE", "good":gm, "defect":dm,
                  "auroc": float(auroc) if SKLEARN_OK and (labels==1).any() else None}
    (out_dir/"recon_metrics.json").write_text(json.dumps(comparison, indent=2))
    print("Metrics saved:", out_dir/"recon_metrics.json")
