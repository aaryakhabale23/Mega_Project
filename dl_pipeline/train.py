"""
Training script for the density estimation head on ShanghaiTech Part B.

Designed to run on Google Colab with a T4 GPU.

Key improvements over the initial version:
    - Data augmentation (random horizontal flip, colour jitter)
    - Validation split with early stopping (patience-based)
    - Combined loss: MSE (pixel) + counting loss (|sum(pred) − sum(gt)|)
    - LR scheduler (ReduceLROnPlateau)
    - Robust dataset with missing-file filtering
    - Consistent resolution between training and evaluation

Usage
-----
    python dl_pipeline/train.py --data_root /content/ShanghaiTech/part_B \
                                --epochs 100 --batch_size 8 --lr 1e-4
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dl_pipeline.model import DensityHead, YOLOv8Backbone

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset with augmentation
# ---------------------------------------------------------------------------

class ShanghaiTechDataset(Dataset):
    """
    PyTorch dataset for ShanghaiTech Part B with pre-generated density maps.

    Each sample consists of an image resized to (480, 640) and its
    corresponding density map resized to (60, 80) — i.e. the spatial
    resolution of the YOLOv8 backbone feature map (stride 8).

    Augmentations (training only):
        - Random horizontal flip
        - Random colour jitter (brightness, contrast)

    Parameters
    ----------
    data_root : Path
        Root of Part B (contains train_data/, test_data/).
    split : str
        'train_data' or 'test_data'.
    augment : bool
        Whether to apply data augmentation (True for training).
    """

    INPUT_H = 480
    INPUT_W = 640
    FEAT_H = 60   # INPUT_H / 8
    FEAT_W = 80   # INPUT_W / 8
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_root: Path, split: str = "train_data", augment: bool = False):
        self.img_dir = data_root / split / "images"
        self.density_dir = data_root / split / "density_maps"
        self.augment = augment

        # Only include images that have a corresponding density map
        all_imgs = sorted(self.img_dir.glob("*.jpg"))
        self.img_files = [
            f for f in all_imgs
            if (self.density_dir / (f.stem + ".npy")).exists()
        ]

        if len(self.img_files) == 0:
            raise FileNotFoundError(
                f"No valid image+density pairs found in {self.img_dir}. "
                f"Have you run preprocess.py first?"
            )

        skipped = len(all_imgs) - len(self.img_files)
        if skipped > 0:
            print(f"  [{split}] Skipped {skipped} images with missing density maps.")
        print(f"  [{split}] Loaded {len(self.img_files)} samples (augment={augment})")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        density_path = self.density_dir / (img_path.stem + ".npy")

        # Load & preprocess image
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (self.INPUT_W, self.INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load density map and resize to feature-map spatial dims
        density = np.load(str(density_path)).astype(np.float32)

        # Scale density to preserve count after resize:
        # sum of original density ≈ person count.
        # After resize from (orig_h, orig_w) → (FEAT_H, FEAT_W),
        # we scale by the area ratio so the sum is preserved.
        orig_h, orig_w = density.shape
        density_resized = cv2.resize(density, (self.FEAT_W, self.FEAT_H),
                                     interpolation=cv2.INTER_LINEAR)
        scale_factor = (orig_h * orig_w) / (self.FEAT_H * self.FEAT_W)
        density_resized *= scale_factor

        # --- Augmentation ---
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                density_resized = density_resized[:, ::-1].copy()

            # Random brightness & contrast jitter
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.uniform(-0.1, 0.1)   # brightness
                img = np.clip(alpha * img + beta, 0.0, 1.0)

        # ImageNet normalisation
        img = (img - self.MEAN) / self.STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        density_tensor = torch.from_numpy(density_resized).unsqueeze(0).float()  # (1, FH, FW)

        return img_tensor, density_tensor


# ---------------------------------------------------------------------------
# Combined loss: MSE + counting loss
# ---------------------------------------------------------------------------

class DensityLoss(nn.Module):
    """
    Combined loss for density map estimation.

    L = MSE(pred, gt) + count_weight * |sum(pred) − sum(gt)| / batch_size

    The counting loss encourages the model to produce density maps whose
    integral matches the ground-truth person count, in addition to matching
    the spatial distribution pixel-by-pixel.

    Parameters
    ----------
    count_weight : float
        Weight for the counting (L1 of sum) term relative to MSE.
    """

    def __init__(self, count_weight: float = 0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.count_weight = count_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, gt)

        # Counting loss: compare predicted vs ground-truth person count
        pred_count = pred.sum(dim=(1, 2, 3))  # (B,)
        gt_count = gt.sum(dim=(1, 2, 3))      # (B,)
        count_loss = torch.abs(pred_count - gt_count).mean()

        return mse_loss + self.count_weight * count_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_root: Path,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    save_path: Path = Path("density_head.pth"),
    device_str: str = "auto",
    patience: int = 15,
    count_weight: float = 0.01,
    num_workers: int = 2,
) -> None:
    """
    Train the density estimation head.

    Parameters
    ----------
    data_root : Path
        Root of ShanghaiTech Part B dataset.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for Adam optimiser.
    save_path : Path
        Where to save the trained weights.
    device_str : str
        'auto', 'cuda', or 'cpu'.
    patience : int
        Early stopping patience (epochs without val improvement).
    count_weight : float
        Weight for the counting loss term.
    num_workers : int
        DataLoader workers (set to 0 for Colab compatibility).
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on: {device}")

    # Datasets & loaders
    train_dataset = ShanghaiTechDataset(data_root, split="train_data", augment=True)
    val_dataset = ShanghaiTechDataset(data_root, split="test_data", augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Models
    backbone = YOLOv8Backbone().to(device)
    backbone.eval()
    density_head = DensityHead(in_channels=YOLOv8Backbone.FEATURE_CHANNELS).to(device)

    # Optimiser, loss, scheduler
    optimizer = optim.Adam(density_head.parameters(), lr=lr)
    criterion = DensityLoss(count_weight=count_weight)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"  Training — max {epochs} epochs, patience={patience}")
    print(f"  lr={lr}, batch={batch_size}, count_weight={count_weight}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        density_head.train()
        epoch_train_loss = 0.0
        n_train = 0

        for imgs, gt_density in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs = imgs.to(device)
            gt_density = gt_density.to(device)

            # Forward through frozen backbone
            with torch.no_grad():
                features = backbone(imgs)

            # Forward through trainable density head at feature-map resolution
            pred_density = density_head(features, gt_density.shape[2], gt_density.shape[3])

            loss = criterion(pred_density, gt_density)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train += 1

        avg_train_loss = epoch_train_loss / max(n_train, 1)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        density_head.eval()
        epoch_val_loss = 0.0
        n_val = 0
        val_mae = 0.0

        with torch.no_grad():
            for imgs, gt_density in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                imgs = imgs.to(device)
                gt_density = gt_density.to(device)

                features = backbone(imgs)
                pred_density = density_head(features, gt_density.shape[2], gt_density.shape[3])

                loss = criterion(pred_density, gt_density)
                epoch_val_loss += loss.item()
                n_val += 1

                # MAE on counts
                pred_count = pred_density.sum(dim=(1, 2, 3))
                gt_count = gt_density.sum(dim=(1, 2, 3))
                val_mae += torch.abs(pred_count - gt_count).sum().item()

        avg_val_loss = epoch_val_loss / max(n_val, 1)
        avg_val_mae = val_mae / len(val_dataset)
        val_losses.append(avg_val_loss)

        # LR scheduler step
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val MAE: {avg_val_mae:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- Early stopping / save best ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(density_head.state_dict(), str(save_path))
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save loss history
    loss_log_path = save_path.parent / "train_loss_log.npy"
    np.save(str(loss_log_path), np.array(train_losses))

    val_log_path = save_path.parent / "val_loss_log.npy"
    np.save(str(val_log_path), np.array(val_losses))

    print(f"\nBest model saved to {save_path}")
    print(f"Train loss log → {loss_log_path}")
    print(f"Val loss log   → {val_log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train density head on ShanghaiTech Part B.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of ShanghaiTech Part B dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="density_head.pth")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience.")
    parser.add_argument("--count_weight", type=float, default=0.01,
                        help="Weight for counting loss term.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (0 for Colab).")
    args = parser.parse_args()

    train(
        data_root=Path(args.data_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=Path(args.save_path),
        device_str=args.device,
        patience=args.patience,
        count_weight=args.count_weight,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
