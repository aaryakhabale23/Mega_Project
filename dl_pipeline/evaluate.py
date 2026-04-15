"""
Evaluation script for the trained density head on ShanghaiTech Part B test set.

Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) between
predicted and ground-truth crowd counts per image.

The density map is predicted at feature-map resolution (60×80) — the same
resolution used during training — so that summed counts are consistent.

Usage
-----
    python dl_pipeline/evaluate.py --data_root /content/ShanghaiTech/part_B \
                                   --weights density_head.pth
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

from dl_pipeline.model import DensityHead, YOLOv8Backbone

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


INPUT_H = 480
INPUT_W = 640
FEAT_H = 60   # INPUT_H / 8
FEAT_W = 80   # INPUT_W / 8
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def evaluate(
    data_root: Path,
    weights: Path,
    device_str: str = "auto",
) -> dict:
    """
    Evaluate density head on the ShanghaiTech Part B test set.

    Parameters
    ----------
    data_root : Path
        Root of Part B dataset.
    weights : Path
        Path to trained density_head.pth.
    device_str : str
        'auto', 'cuda', or 'cpu'.

    Returns
    -------
    dict with keys 'mae', 'rmse', 'per_image' (list of dicts).
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load models
    backbone = YOLOv8Backbone().to(device)
    backbone.eval()

    density_head = DensityHead(in_channels=YOLOv8Backbone.FEATURE_CHANNELS).to(device)
    density_head.load_state_dict(
        torch.load(str(weights), map_location=device, weights_only=True)
    )
    density_head.eval()

    # Test images & ground truth
    img_dir = data_root / "test_data" / "images"
    gt_dir = data_root / "test_data" / "ground-truth"
    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"Evaluating on {len(img_files)} test images ...")

    errors = []
    per_image = []

    with torch.no_grad():
        for img_path in tqdm(img_files, desc="Evaluating"):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_norm = (img_rgb - MEAN) / STD
            tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

            # Predict at feature-map resolution (same as training)
            features = backbone(tensor)
            pred_density = density_head(features, FEAT_H, FEAT_W)

            # The density map at feature-map resolution needs area scaling
            # to get the correct count (same as in the dataset preprocessing)
            scale_factor = (INPUT_H * INPUT_W) / (FEAT_H * FEAT_W)
            # Actually, the model already learned to predict at feature-map
            # resolution with area-scaled targets, so the sum IS the count.
            pred_count = pred_density.sum().item()

            # Ground truth count
            mat_name = "GT_" + img_path.stem + ".mat"
            mat_path = gt_dir / mat_name
            if not mat_path.exists():
                continue
            mat = loadmat(str(mat_path))
            try:
                points = mat["image_info"][0][0][0][0][0]
                gt_count = len(points)
            except (KeyError, IndexError):
                continue

            ae = abs(pred_count - gt_count)
            se = (pred_count - gt_count) ** 2
            errors.append((ae, se))

            per_image.append({
                "image": img_path.name,
                "gt_count": gt_count,
                "pred_count": round(pred_count, 2),
                "ae": round(ae, 2),
            })

    if not errors:
        print("No valid test images found!")
        return {"mae": float("inf"), "rmse": float("inf"), "per_image": []}

    mae = np.mean([e[0] for e in errors])
    rmse = np.sqrt(np.mean([e[1] for e in errors]))

    print(f"\n{'='*50}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  Evaluated on {len(errors)} images")
    print(f"{'='*50}")

    return {"mae": float(mae), "rmse": float(rmse), "per_image": per_image}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate density head on ShanghaiTech Part B.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--weights", type=str, default="density_head.pth")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    evaluate(
        data_root=Path(args.data_root),
        weights=Path(args.weights),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
