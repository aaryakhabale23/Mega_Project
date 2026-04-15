"""
ShanghaiTech Part B dataset preprocessing.

Loads .mat annotation files, generates Gaussian density maps using a fixed
kernel (size 15, sigma 4), and saves as .npy files alongside the source
images.  Density maps are NOT normalised — their integral (sum) equals
the ground-truth person count, which is essential for training.

Usage (Google Colab)
--------------------
    python dl_pipeline/preprocess.py --data_root /content/ShanghaiTech/part_B
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

KERNEL_SIZE = 15
SIGMA = 4.0


def generate_density_map(
    img_shape: tuple[int, int],
    points: np.ndarray,
    kernel_size: int = KERNEL_SIZE,
    sigma: float = SIGMA,
) -> np.ndarray:
    """
    Generate a Gaussian density map from head annotation points.

    Parameters
    ----------
    img_shape : (H, W)
    points : np.ndarray of shape (N, 2) — each row is (x, y) head position
    kernel_size : int   (unused explicitly — sigma controls spread)
    sigma : float       Gaussian kernel standard deviation

    Returns
    -------
    density_map : np.ndarray of shape (H, W), float32
        Unnormalised density map whose sum ≈ len(points) (person count).
    """
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1.0

    # Apply Gaussian filter
    density = gaussian_filter(density, sigma=sigma, truncate=kernel_size / (2 * sigma))

    # NOTE: we intentionally do NOT normalise to [0, 1].
    # The Gaussian filter preserves the integral, so:
    #     density.sum() ≈ len(points)  (the person count)
    # This is critical for the model to learn actual counts.

    return density


def preprocess_split(data_root: Path, split: str) -> None:
    """
    Process all images in a split (train_data or test_data).

    For each image, loads the corresponding .mat annotation, generates a
    density map, and saves it as a .npy file in the same directory.

    Parameters
    ----------
    data_root : Path
        Root directory of ShanghaiTech Part B (contains train_data/, test_data/).
    split : str
        'train_data' or 'test_data'.
    """
    img_dir = data_root / split / "images"
    gt_dir = data_root / split / "ground-truth"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    # Output directory for density maps
    density_dir = data_root / split / "density_maps"
    density_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"[{split}] Processing {len(img_files)} images ...")

    for img_path in tqdm(img_files, desc=split):
        # Load image to get shape
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

        # Corresponding .mat file: IMG_1.jpg → GT_IMG_1.mat
        mat_name = "GT_" + img_path.stem + ".mat"
        mat_path = gt_dir / mat_name

        if not mat_path.exists():
            print(f"  WARNING: Ground-truth not found: {mat_path}, skipping.")
            continue

        # Load annotation points
        mat = loadmat(str(mat_path))
        # ShanghaiTech stores points in image_info[0][0][0][0][0]
        try:
            points = mat["image_info"][0][0][0][0][0]  # (N, 2) — (x, y)
        except (KeyError, IndexError):
            print(f"  WARNING: Unexpected .mat format for {mat_path}, skipping.")
            continue

        # Generate density map
        density = generate_density_map((h, w), points)

        # Save
        save_path = density_dir / (img_path.stem + ".npy")
        np.save(str(save_path), density)

    print(f"[{split}] Done.  Density maps saved to {density_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess ShanghaiTech Part B dataset into density maps."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of ShanghaiTech Part B dataset.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    preprocess_split(data_root, "train_data")
    preprocess_split(data_root, "test_data")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
