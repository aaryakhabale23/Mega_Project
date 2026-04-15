"""
train_mall.py — Train density head on the Mall Dataset (indoor crowd counting).

The Mall Dataset is 2000 indoor surveillance frames (640x480) with 13-53 people
per frame. This is much more suitable for indoor room occupancy estimation than
ShanghaiTech (outdoor crowds).

Architecture: Frozen YOLOv8-small backbone + 3-layer dilated CNN density head
              (same as local_train.py, different dataset)

Usage:
    python train_mall.py

Saves: density_head.pth, training_curves.png
"""

from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# Force UTF-8 on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

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
DATA_ROOT   = PROJECT_ROOT / "data" / "mall_dataset"
SAVE_PATH   = PROJECT_ROOT / "density_head.pth"
EPOCHS      = 50
BATCH_SIZE  = 4
LR          = 1e-4
PATIENCE    = 12
COUNT_WEIGHT = 0.01
NUM_WORKERS = 0        # safest on Windows
TRAIN_RATIO = 0.8      # 80% train, 20% val from 2000 frames


# ============================================================
# STEP 1: Download Mall Dataset
# ============================================================

def download_dataset():
    """Download and extract Mall Dataset if not present."""
    frames_dir = DATA_ROOT / "frames"
    gt_file = DATA_ROOT / "mall_gt.mat"

    if frames_dir.exists() and gt_file.exists():
        n_frames = len(list(frames_dir.glob("*.jpg")))
        print(f"Mall dataset already exists ({n_frames} frames)")
        return

    print("Downloading Mall Dataset (~130 MB)...")
    print("  Source: CUHK (personal.ie.cuhk.edu.hk)\n")

    url = "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_ROOT / "mall_dataset.zip"

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // max(total_size, 1))
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}% ({downloaded // (1024*1024)} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, str(zip_path), reporthook=_progress)
        print("\n  Download complete!")

    except Exception as e:
        print(f"\n  Download failed: {e}")
        print("  Please download manually from:")
        print(f"    {url}")
        print(f"  and extract to: {DATA_ROOT}")
        sys.exit(1)

    print("  Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(DATA_ROOT))
    zip_path.unlink()

    # The zip may extract into a subfolder — handle common layouts
    # Check if frames are directly in DATA_ROOT or in a subfolder
    if not (DATA_ROOT / "frames").exists():
        # Look for frames in subdirectories
        for sub in DATA_ROOT.iterdir():
            if sub.is_dir():
                if (sub / "frames").exists():
                    # Move contents up
                    for item in sub.iterdir():
                        dest = DATA_ROOT / item.name
                        if not dest.exists():
                            item.rename(dest)
                    sub.rmdir()
                    break

    n_frames = len(list((DATA_ROOT / "frames").glob("*.jpg")))
    print(f"  Extracted {n_frames} frames\n")


# ============================================================
# STEP 2: Parse Mall ground truth + generate density maps
# ============================================================

SIGMA = 4.0
KERNEL_SIZE = 15

def generate_density_map(img_shape, points, sigma=SIGMA):
    """Create Gaussian density map from point annotations."""
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


def preprocess():
    """Parse mall_gt.mat and generate density maps for all frames."""
    density_dir = DATA_ROOT / "density_maps"
    density_dir.mkdir(parents=True, exist_ok=True)

    existing = list(density_dir.glob("*.npy"))
    if len(existing) >= 2000:
        print(f"Density maps already exist ({len(existing)} maps)")
        return

    gt_path = DATA_ROOT / "mall_gt.mat"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found at {gt_path}")
        sys.exit(1)

    print("Generating density maps from mall_gt.mat...")

    mat = loadmat(str(gt_path))

    # mall_gt.mat structure: mat['frame'][0] is array of frame annotations
    # Each frame[i][0] contains the (x,y) head coordinates
    frame_annotations = mat['frame'][0]

    frames_dir = DATA_ROOT / "frames"
    frame_files = sorted(frames_dir.glob("*.jpg"))

    count_stats = []

    for i, frame_path in enumerate(tqdm(frame_files, desc="Generating density maps")):
        save_path = density_dir / (frame_path.stem + ".npy")
        if save_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Get annotations for this frame
        if i < len(frame_annotations):
            try:
                # mall_gt.mat: frame[i][0][0][0] contains Nx2 array of (x,y) coordinates
                points = frame_annotations[i][0][0][0]
                if len(points.shape) == 1:
                    points = points.reshape(-1, 2)
            except (IndexError, AttributeError):
                points = np.array([])
        else:
            points = np.array([])

        count_stats.append(len(points))
        density = generate_density_map((h, w), points)
        np.save(str(save_path), density)

    if count_stats:
        print(f"\n  Frame statistics:")
        print(f"    Min count:  {min(count_stats)}")
        print(f"    Max count:  {max(count_stats)}")
        print(f"    Mean count: {np.mean(count_stats):.1f}")
        print(f"    Total maps: {len(count_stats)}")
    print("Preprocessing complete!\n")


# ============================================================
# STEP 3: Dataset class
# ============================================================

class MallDataset(Dataset):
    """Mall Dataset loader with train/val splitting."""
    INPUT_H = 480
    INPUT_W = 640
    FEAT_H  = 60
    FEAT_W  = 80
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_root, indices, augment=False):
        self.frames_dir = data_root / "frames"
        self.density_dir = data_root / "density_maps"
        self.augment = augment

        # Get all frame files and filter to given indices
        all_frames = sorted(self.frames_dir.glob("*.jpg"))
        self.img_files = [
            all_frames[i] for i in indices
            if i < len(all_frames) and
               (self.density_dir / (all_frames[i].stem + ".npy")).exists()
        ]

        if len(self.img_files) == 0:
            raise FileNotFoundError(
                f"No valid image+density pairs found. Run preprocessing first."
            )
        print(f"    Loaded {len(self.img_files)} samples (augment={augment})")

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
    print(f"\nTraining on: {device}")
    print(f"  (CPU training - this will take ~20-40 min)\n")

    # Split 2000 frames into train/val
    total_frames = len(list((DATA_ROOT / "frames").glob("*.jpg")))
    indices = list(range(total_frames))
    np.random.shuffle(indices)
    split_idx = int(total_frames * TRAIN_RATIO)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print(f"  Train: {len(train_indices)} frames")
    print(f"  Val:   {len(val_indices)} frames")

    train_dataset = MallDataset(DATA_ROOT, train_indices, augment=True)
    val_dataset   = MallDataset(DATA_ROOT, val_indices,   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

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
    print(f"  Training - max {EPOCHS} epochs, patience={PATIENCE}")
    print(f"  lr={LR}, batch={BATCH_SIZE}, count_weight={COUNT_WEIGHT}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        # -- Train --
        density_head.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]",
                    leave=False)
        for imgs, densities in pbar:
            imgs = imgs.to(device)
            densities = densities.to(device)

            with torch.no_grad():
                features = backbone(imgs)

            pred = density_head(features, target_h=MallDataset.FEAT_H, target_w=MallDataset.FEAT_W)
            loss = criterion(pred, densities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # -- Validate --
        density_head.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, densities in val_loader:
                imgs = imgs.to(device)
                densities = densities.to(device)
                features = backbone(imgs)
                pred = density_head(features, target_h=MallDataset.FEAT_H, target_w=MallDataset.FEAT_W)
                loss = criterion(pred, densities)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(density_head.state_dict(), str(SAVE_PATH))
            epochs_no_improve = 0
            improved = " << BEST"
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}  |  train: {avg_train:.6f}  |  "
              f"val: {avg_val:.6f}  |  lr: {current_lr:.1e}{improved}")

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Save loss logs
    np.save(str(PROJECT_ROOT / "train_loss_log.npy"), np.array(train_losses))
    np.save(str(PROJECT_ROOT / "val_loss_log.npy"), np.array(val_losses))

    # Plot curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mall Dataset - Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(PROJECT_ROOT / "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Training curves saved to training_curves.png")


# ============================================================
# STEP 6: Evaluation
# ============================================================

def evaluate():
    """Compute MAE and RMSE on the validation set."""
    device = torch.device("cpu")

    print("\n" + "=" * 50)
    print("  EVALUATION (Mall Dataset)")
    print("=" * 50)

    backbone = YOLOv8Backbone().to(device)
    backbone.eval()
    density_head = DensityHead(in_channels=YOLOv8Backbone.FEATURE_CHANNELS).to(device)

    if not SAVE_PATH.exists():
        print("  No trained model found!")
        return

    density_head.load_state_dict(torch.load(str(SAVE_PATH), map_location=device))
    density_head.eval()

    # Use ALL frames for evaluation stats
    total_frames = len(list((DATA_ROOT / "frames").glob("*.jpg")))
    all_indices = list(range(total_frames))

    eval_dataset = MallDataset(DATA_ROOT, all_indices, augment=False)
    eval_loader  = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS)

    gt_path = DATA_ROOT / "mall_gt.mat"
    mat = loadmat(str(gt_path))
    frame_annotations = mat['frame'][0]

    errors = []
    with torch.no_grad():
        for i, (img, density_gt) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            img = img.to(device)
            features = backbone(img)
            pred = density_head(features, target_h=MallDataset.FEAT_H, target_w=MallDataset.FEAT_W)
            pred_count = pred.sum().item()
            gt_count = density_gt.sum().item()
            errors.append(abs(pred_count - gt_count))

    errors = np.array(errors)
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())

    print(f"\n{'='*50}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  Frames evaluated: {len(errors)}")
    print(f"{'='*50}")

    return mae, rmse


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  Mall Dataset - Indoor Density Head Training")
    print("  Architecture: YOLOv8s (frozen) + Density Head (trainable)")
    print("=" * 60 + "\n")

    download_dataset()
    preprocess()
    train_model()
    mae, rmse = evaluate()

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print(f"  Model weights: {SAVE_PATH}")
    print(f"  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")
    print(f"  Curves: {PROJECT_ROOT / 'training_curves.png'}")
    print(f"  Next: python demo.py --simulate")
    print("=" * 60)


if __name__ == "__main__":
    main()
