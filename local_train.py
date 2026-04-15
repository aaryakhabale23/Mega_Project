"""
local_train.py — Full local DL pipeline: download, preprocess, train, evaluate.

Runs entirely on CPU (no Colab / GPU needed).
Saves: density_head.pth, train_loss_log.npy, val_loss_log.npy

Usage:
    python local_train.py
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
import urllib.request
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# ── Make sure dl_pipeline is importable ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dl_pipeline.model import DensityHead, YOLOv8Backbone

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT   = PROJECT_ROOT / "data" / "part_B_final"
SAVE_PATH   = PROJECT_ROOT / "density_head.pth"
EPOCHS      = 50        # CPU-friendly
BATCH_SIZE  = 4         # smaller for CPU RAM
LR          = 1e-4
PATIENCE    = 12
COUNT_WEIGHT = 0.01
NUM_WORKERS = 0         # safest on Windows


# ============================================================
# STEP 1: Download ShanghaiTech Part B
# ============================================================

def download_dataset():
    """Download and extract ShanghaiTech dataset if not present."""
    if DATA_ROOT.exists() and (DATA_ROOT / "train_data" / "images").exists():
        n_train = len(list((DATA_ROOT / "train_data" / "images").glob("*.jpg")))
        n_test  = len(list((DATA_ROOT / "test_data" / "images").glob("*.jpg")))
        print(f"Dataset already exists ({n_train} train, {n_test} test images)")
        return

    print("Downloading ShanghaiTech dataset (~344 MB)...")
    print("   This may take a few minutes depending on your connection.\n")

    url = "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
    download_dir = PROJECT_ROOT / "data"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "ShanghaiTech.zip"

    try:
        # Download with progress
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // max(total_size, 1))
            mb  = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r   {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress)
        print("\n   Download done. Extracting...")

        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(str(download_dir))
        zip_path.unlink()

        print("Dataset ready!")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nAlternative: Download manually from Kaggle:")
        print("   https://www.kaggle.com/datasets/tthien/shanghaitech")
        print(f"   Extract 'part_B_final' folder to: {DATA_ROOT}")
        sys.exit(1)

    # Verify
    for d in ["train_data/images", "train_data/ground_truth",
              "test_data/images",  "test_data/ground_truth"]:
        p = DATA_ROOT / d
        count = len(list(p.iterdir())) if p.exists() else 0
        status = "OK" if count > 0 else "MISSING"
        print(f"  {status} {d}/ ({count} files)")


# ============================================================
# STEP 2: Preprocess — generate density maps
# ============================================================

KERNEL_SIZE = 15
SIGMA = 4.0

def generate_density_map(img_shape, points, sigma=SIGMA):
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density
    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1.0
    density = gaussian_filter(density, sigma=sigma,
                               truncate=KERNEL_SIZE / (2 * sigma))
    return density

def preprocess_split(split: str):
    img_dir = DATA_ROOT / split / "images"
    gt_dir  = DATA_ROOT / split / "ground_truth"
    density_dir = DATA_ROOT / split / "density_maps"
    density_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"[{split}] Processing {len(img_files)} images ...")

    for img_path in tqdm(img_files, desc=split):
        save_path = density_dir / (img_path.stem + ".npy")
        if save_path.exists():
            continue  # skip already processed

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        mat_name = "GT_" + img_path.stem + ".mat"
        mat_path = gt_dir / mat_name
        if not mat_path.exists():
            continue

        mat = loadmat(str(mat_path))
        try:
            points = mat["image_info"][0][0][0][0][0]
        except (KeyError, IndexError):
            continue

        density = generate_density_map((h, w), points)
        np.save(str(save_path), density)

    print(f"[{split}] Done.")

def preprocess():
    existing_train = list((DATA_ROOT / "train_data" / "density_maps").glob("*.npy")) if (DATA_ROOT / "train_data" / "density_maps").exists() else []
    existing_test  = list((DATA_ROOT / "test_data"  / "density_maps").glob("*.npy")) if (DATA_ROOT / "test_data"  / "density_maps").exists() else []

    if len(existing_train) > 0 and len(existing_test) > 0:
        print(f"✅ Density maps already exist ({len(existing_train)} train, {len(existing_test)} test)")
        return

    print("🔧 Generating density maps...\n")
    preprocess_split("train_data")
    preprocess_split("test_data")
    print("✅ Preprocessing complete!")


# ============================================================
# STEP 3: Dataset class
# ============================================================

class ShanghaiTechDataset(Dataset):
    INPUT_H = 480
    INPUT_W = 640
    FEAT_H  = 60
    FEAT_W  = 80
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_root, split="train_data", augment=False):
        self.img_dir = data_root / split / "images"
        self.density_dir = data_root / split / "density_maps"
        self.augment = augment

        all_imgs = sorted(self.img_dir.glob("*.jpg"))
        self.img_files = [
            f for f in all_imgs
            if (self.density_dir / (f.stem + ".npy")).exists()
        ]

        if len(self.img_files) == 0:
            raise FileNotFoundError(
                f"No valid image+density pairs in {self.img_dir}. "
                f"Run preprocessing first."
            )
        print(f"  [{split}] Loaded {len(self.img_files)} samples (augment={augment})")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        density_path = self.density_dir / (img_path.stem + ".npy")

        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (self.INPUT_W, self.INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        density = np.load(str(density_path)).astype(np.float32)
        orig_h, orig_w = density.shape
        density_resized = cv2.resize(density, (self.FEAT_W, self.FEAT_H),
                                     interpolation=cv2.INTER_LINEAR)
        scale_factor = (orig_h * orig_w) / (self.FEAT_H * self.FEAT_W)
        density_resized *= scale_factor

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                density_resized = density_resized[:, ::-1].copy()
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta  = np.random.uniform(-0.1, 0.1)
                img = np.clip(alpha * img + beta, 0.0, 1.0)

        img = (img - self.MEAN) / self.STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        density_tensor = torch.from_numpy(density_resized).unsqueeze(0).float()
        return img_tensor, density_tensor


# ============================================================
# STEP 4: Loss
# ============================================================

class DensityLoss(nn.Module):
    def __init__(self, count_weight=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.count_weight = count_weight

    def forward(self, pred, gt):
        mse_loss = self.mse(pred, gt)
        pred_count = pred.sum(dim=(1, 2, 3))
        gt_count   = gt.sum(dim=(1, 2, 3))
        count_loss = torch.abs(pred_count - gt_count).mean()
        return mse_loss + self.count_weight * count_loss


# ============================================================
# STEP 5: Training loop
# ============================================================

def train_model():
    device = torch.device("cpu")
    print(f"\n🖥️  Training on: {device}")
    print(f"   (CPU training — this will take ~30-60 min depending on your machine)\n")

    train_dataset = ShanghaiTechDataset(DATA_ROOT, "train_data", augment=True)
    val_dataset   = ShanghaiTechDataset(DATA_ROOT, "test_data",  augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    print(f"   Training samples:   {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    backbone = YOLOv8Backbone().to(device)
    backbone.eval()
    density_head = DensityHead(in_channels=YOLOv8Backbone.FEATURE_CHANNELS).to(device)

    optimizer = optim.Adam(density_head.parameters(), lr=LR)
    criterion = DensityLoss(count_weight=COUNT_WEIGHT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  Training — max {EPOCHS} epochs, patience={PATIENCE}")
    print(f"  lr={LR}, batch={BATCH_SIZE}, count_weight={COUNT_WEIGHT}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        density_head.train()
        epoch_train_loss = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False)
        for imgs, gt_density in pbar:
            imgs       = imgs.to(device)
            gt_density = gt_density.to(device)

            with torch.no_grad():
                features = backbone(imgs)

            pred = density_head(features, gt_density.shape[2], gt_density.shape[3])
            loss = criterion(pred, gt_density)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train = epoch_train_loss / max(n_train, 1)
        train_losses.append(avg_train)

        # ── Validate ──
        density_head.eval()
        epoch_val_loss = 0.0
        n_val = 0
        val_mae = 0.0

        with torch.no_grad():
            for imgs, gt_density in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False):
                imgs       = imgs.to(device)
                gt_density = gt_density.to(device)

                features = backbone(imgs)
                pred = density_head(features, gt_density.shape[2], gt_density.shape[3])
                loss = criterion(pred, gt_density)
                epoch_val_loss += loss.item()
                n_val += 1

                pred_count = pred.sum(dim=(1, 2, 3))
                gt_count   = gt_density.sum(dim=(1, 2, 3))
                val_mae += torch.abs(pred_count - gt_count).sum().item()

        avg_val     = epoch_val_loss / max(n_val, 1)
        avg_val_mae = val_mae / len(val_dataset)
        val_losses.append(avg_val)

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {avg_train:.6f} | "
            f"Val: {avg_val:.6f} | "
            f"MAE: {avg_val_mae:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(density_head.state_dict(), str(SAVE_PATH))
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}")
                break

    # Save loss logs
    np.save(str(SAVE_PATH.parent / "train_loss_log.npy"), np.array(train_losses))
    np.save(str(SAVE_PATH.parent / "val_loss_log.npy"),   np.array(val_losses))

    print(f"\n✅ Training complete!")
    print(f"   Best model:    {SAVE_PATH}")
    print(f"   Train log:     {SAVE_PATH.parent / 'train_loss_log.npy'}")
    print(f"   Val log:       {SAVE_PATH.parent / 'val_loss_log.npy'}")

    return train_losses, val_losses


# ============================================================
# STEP 6: Evaluate
# ============================================================

INPUT_H, INPUT_W = 480, 640
FEAT_H, FEAT_W   = 60, 80
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def evaluate_model():
    device = torch.device("cpu")

    backbone = YOLOv8Backbone().to(device)
    backbone.eval()

    density_head = DensityHead(in_channels=YOLOv8Backbone.FEATURE_CHANNELS).to(device)
    density_head.load_state_dict(
        torch.load(str(SAVE_PATH), map_location=device, weights_only=True)
    )
    density_head.eval()

    img_dir = DATA_ROOT / "test_data" / "images"
    gt_dir  = DATA_ROOT / "test_data" / "ground_truth"
    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"\n📊 Evaluating on {len(img_files)} test images...")

    errors = []
    with torch.no_grad():
        for img_path in tqdm(img_files, desc="Evaluating"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_norm = (img_rgb - MEAN) / STD
            tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

            features = backbone(tensor)
            pred = density_head(features, FEAT_H, FEAT_W)
            pred_count = pred.sum().item()

            mat_path = gt_dir / f"GT_{img_path.stem}.mat"
            if not mat_path.exists():
                continue
            mat = loadmat(str(mat_path))
            try:
                gt_count = len(mat["image_info"][0][0][0][0][0])
            except (KeyError, IndexError):
                continue

            ae = abs(pred_count - gt_count)
            se = (pred_count - gt_count) ** 2
            errors.append((ae, se))

    mae  = np.mean([e[0] for e in errors])
    rmse = np.sqrt(np.mean([e[1] for e in errors]))

    print(f"\n{'='*50}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  Images evaluated: {len(errors)}")
    print(f"{'='*50}")
    return mae, rmse


# ============================================================
# STEP 7: Plot training curves
# ============================================================

def plot_curves(train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train", linewidth=2, color="#06B6D4")
    ax1.plot(epochs, val_losses,   label="Val",   linewidth=2, color="#F59E0B")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_losses, label="Train", linewidth=2, color="#06B6D4")
    ax2.plot(epochs, val_losses,   label="Val",   linewidth=2, color="#F59E0B")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (log)")
    ax2.set_title("Loss (Log Scale)"); ax2.set_yscale("log"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Density Head Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = SAVE_PATH.parent / "training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"📈 Training curves saved to {plot_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  🚀 LOCAL TRAINING — Smart Environment Energy Optimizer")
    print("  📦 DL Pipeline: YOLOv8-s Backbone + Density Head")
    print("=" * 60)

    # Step 1: Download
    print("\n── Step 1/5: Dataset ──")
    download_dataset()

    # Step 2: Preprocess
    print("\n── Step 2/5: Preprocessing ──")
    preprocess()

    # Step 3: Train
    print("\n── Step 3/5: Training ──")
    train_losses, val_losses = train_model()

    # Step 4: Plot
    print("\n── Step 4/5: Plotting ──")
    plot_curves(train_losses, val_losses)

    # Step 5: Evaluate
    print("\n── Step 5/5: Evaluation ──")
    mae, rmse = evaluate_model()

    print("\n" + "=" * 60)
    print("  🎉 ALL DONE!")
    print(f"  📁 Model weights: {SAVE_PATH}")
    print(f"  📊 MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")
    print(f"  📈 Curves: {SAVE_PATH.parent / 'training_curves.png'}")
    print("  ▶️  Next: python demo.py --simulate")
    print("=" * 60)
