"""
evaluate_model.py — Detailed evaluation of the trained density head.

Produces:
  1. MAE and RMSE metrics
  2. Predicted vs Actual scatter plot
  3. Per-frame error distribution
  4. Sample predictions on random frames
"""

import sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dl_pipeline.model import DensityHead, YOLOv8Backbone

DATA_ROOT  = PROJECT_ROOT / "data" / "mall_dataset"
WEIGHTS    = PROJECT_ROOT / "density_head.pth"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

FEAT_H, FEAT_W = 60, 80
INPUT_H, INPUT_W = 480, 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    device = torch.device("cpu")
    backbone = YOLOv8Backbone().to(device)
    backbone.eval()
    density_head = DensityHead(in_channels=YOLOv8Backbone.FEATURE_CHANNELS).to(device)
    density_head.load_state_dict(torch.load(str(WEIGHTS), map_location=device))
    density_head.eval()
    return backbone, density_head, device


def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - MEAN) / STD
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)
    return tensor, img


def main():
    print("=" * 60)
    print("  MODEL EVALUATION - Mall Dataset")
    print("  Architecture: YOLOv8s (frozen) + Density Head")
    print("=" * 60 + "\n")

    # Load model
    backbone, density_head, device = load_model()
    print(f"  Model loaded from: {WEIGHTS}")

    # Load ground truth
    gt_path = DATA_ROOT / "mall_gt.mat"
    mat = loadmat(str(gt_path))
    frame_annotations = mat['frame'][0]

    # Get all frames
    frames_dir = DATA_ROOT / "frames"
    frame_files = sorted(frames_dir.glob("*.jpg"))
    print(f"  Total frames: {len(frame_files)}\n")

    # Run predictions
    gt_counts = []
    pred_counts = []
    errors = []

    print("  Running predictions on all frames...")
    with torch.no_grad():
        for i, frame_path in enumerate(tqdm(frame_files, desc="  Evaluating")):
            # Get ground truth count
            try:
                points = frame_annotations[i][0][0][0]
                gt_count = len(points)
            except (IndexError, AttributeError):
                gt_count = 0

            # Get prediction
            tensor, _ = preprocess_image(frame_path)
            tensor = tensor.to(device)
            features = backbone(tensor)
            pred_map = density_head(features, target_h=FEAT_H, target_w=FEAT_W)
            pred_count = pred_map.sum().item()

            gt_counts.append(gt_count)
            pred_counts.append(pred_count)
            errors.append(abs(pred_count - gt_count))

    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)
    errors = np.array(errors)

    # ── Metrics ──
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    mape = np.mean(errors / np.maximum(gt_counts, 1)) * 100

    print(f"\n{'=' * 50}")
    print(f"  RESULTS")
    print(f"{'=' * 50}")
    print(f"  MAE  (Mean Abs Error)  : {mae:.2f}")
    print(f"  RMSE (Root Mean Sq Err): {rmse:.2f}")
    print(f"  MAPE (Mean Abs % Error): {mape:.1f}%")
    print(f"  Min GT: {gt_counts.min()}  Max GT: {gt_counts.max()}")
    print(f"  Min Pred: {pred_counts.min():.1f}  Max Pred: {pred_counts.max():.1f}")
    print(f"{'=' * 50}\n")

    # ── Plot 1: Predicted vs Actual ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("#0F0F1A")
    fig.suptitle("Model Evaluation - Mall Dataset (Indoor Crowd Counting)",
                 fontsize=14, fontweight="bold", color="white")

    # Scatter plot
    ax = axes[0, 0]
    ax.set_facecolor("#1A1A2E")
    ax.scatter(gt_counts, pred_counts, alpha=0.4, s=15, c="#00D4FF", edgecolors="none")
    max_val = max(gt_counts.max(), pred_counts.max()) + 5
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.7, linewidth=1.5, label="Perfect")
    ax.set_xlabel("Ground Truth Count", color="white", fontsize=10)
    ax.set_ylabel("Predicted Count", color="white", fontsize=10)
    ax.set_title(f"Predicted vs Actual  (MAE={mae:.2f})", color="white", fontsize=11)
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2)

    # Error distribution
    ax = axes[0, 1]
    ax.set_facecolor("#1A1A2E")
    ax.hist(errors, bins=30, color="#00E676", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(mae, color="#FF5252", linestyle="--", linewidth=2, label=f"MAE={mae:.2f}")
    ax.set_xlabel("Absolute Error", color="white", fontsize=10)
    ax.set_ylabel("Frequency", color="white", fontsize=10)
    ax.set_title("Error Distribution", color="white", fontsize=11)
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    ax.tick_params(colors="white")

    # Per-frame timeline
    ax = axes[1, 0]
    ax.set_facecolor("#1A1A2E")
    ax.plot(gt_counts, color="#00D4FF", alpha=0.7, linewidth=1, label="Ground Truth")
    ax.plot(pred_counts, color="#FFB300", alpha=0.7, linewidth=1, label="Predicted")
    ax.set_xlabel("Frame Index", color="white", fontsize=10)
    ax.set_ylabel("Count", color="white", fontsize=10)
    ax.set_title("Per-Frame Counts (Timeline)", color="white", fontsize=11)
    ax.legend(facecolor="#1A1A2E", labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2)

    # Residual plot
    ax = axes[1, 1]
    ax.set_facecolor("#1A1A2E")
    residuals = pred_counts - gt_counts
    ax.scatter(gt_counts, residuals, alpha=0.4, s=15, c="#FFB300", edgecolors="none")
    ax.axhline(0, color="#FF5252", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Ground Truth Count", color="white", fontsize=10)
    ax.set_ylabel("Prediction Error (Pred - GT)", color="white", fontsize=10)
    ax.set_title("Residual Plot", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = OUTPUT_DIR / "evaluation_plots.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plots saved to: {plot_path}")

    # ── Plot 2: Sample predictions ──
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.patch.set_facecolor("#0F0F1A")
    fig2.suptitle("Sample Predictions", fontsize=14, fontweight="bold", color="white")

    np.random.seed(42)
    sample_indices = np.random.choice(len(frame_files), 8, replace=False)
    sample_indices.sort()

    for idx, ax in zip(sample_indices, axes2.flat):
        _, img = preprocess_image(frame_files[idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        gt = gt_counts[idx]
        pred = pred_counts[idx]
        err = abs(pred - gt)
        color = "#00E676" if err < 2 else ("#FFB300" if err < 5 else "#FF5252")
        ax.set_title(f"GT:{gt}  Pred:{pred:.0f}", fontsize=9,
                     fontweight="bold", color=color)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    sample_path = OUTPUT_DIR / "sample_predictions.png"
    fig2.savefig(str(sample_path), dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Samples saved to: {sample_path}")

    print(f"\n  All results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
