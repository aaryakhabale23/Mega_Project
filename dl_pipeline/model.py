"""
Occupancy estimation pipeline — YOLOv8-s backbone + custom density head.

Two-stage architecture:
    1. Frozen YOLOv8-small backbone (feature extractor)
    2. Custom 3-layer dilated CNN density head (trainable)

Also includes MOG2 background subtraction for per-zone activity estimation.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Density estimation head
# ---------------------------------------------------------------------------

class DensityHead(nn.Module):
    """
    3-layer dilated CNN that takes YOLOv8 backbone feature maps and produces
    a single-channel density map.

    Architecture
    ------------
    Conv2d(in_ch → 64,  k=3, dilation=1, pad=1)  + ReLU
    Conv2d(64    → 32,  k=3, dilation=2, pad=2)  + ReLU
    Conv2d(32    → 1,   k=3, dilation=2, pad=2)  + ReLU

    The output is bilinearly upsampled to the requested spatial dimensions.
    Values are non-negative (ReLU) and unnormalised so that
    ``density_map.sum() ≈ person_count``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the backbone feature map.  For YOLOv8-s
        backbone layer-4 output this is 128.
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, target_h: int = 480, target_w: int = 640) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, C, H', W']
            Feature map from the YOLOv8 backbone.
        target_h, target_w : int
            Spatial dimensions to upsample the density map to.

        Returns
        -------
        Tensor [B, 1, target_h, target_w]
            Predicted density map with non-negative values (unnormalised).
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # ReLU: density is non-negative, unbounded
        # Upsample to requested resolution
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# YOLOv8 backbone wrapper
# ---------------------------------------------------------------------------

class YOLOv8Backbone(nn.Module):
    """
    Wraps the YOLOv8-small backbone as a frozen feature extractor.

    Extracts the output of backbone layer 4 (C2f block after stride-8
    downsampling).  For a 640×480 input the feature map shape is
    [B, 128, 60, 80].
    """

    FEATURE_LAYER = 4          # Index into model.model (after 3rd Conv + C2f)
    FEATURE_CHANNELS = 128     # Output channels at this layer for YOLOv8-s

    def __init__(self, weights: str = "yolov8s.pt"):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(weights)
        # Grab the sequential backbone layers (0..9 in YOLOv8)
        # Layers 0–4 all use from=-1 (purely sequential) so nn.Sequential works.
        self.backbone_layers = nn.Sequential(
            *list(yolo.model.model.children())[: self.FEATURE_LAYER + 1]
        )

        # Freeze all parameters
        for param in self.backbone_layers.parameters():
            param.requires_grad = False

        self.eval()

        # Runtime verification: ensure output channels match our expectation.
        # This guards against silent breakage if ultralytics changes internals.
        dummy = torch.randn(1, 3, 480, 640)
        with torch.no_grad():
            out = self.backbone_layers(dummy)
        assert out.shape[1] == self.FEATURE_CHANNELS, (
            f"YOLOv8 backbone layer {self.FEATURE_LAYER} produced "
            f"{out.shape[1]} channels, expected {self.FEATURE_CHANNELS}. "
            f"The ultralytics version may have changed its internal structure."
        )
        del dummy, out

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 3, H, W]  (normalised RGB)

        Returns
        -------
        Tensor [B, 128, H/8, W/8]
        """
        return self.backbone_layers(x)


# ---------------------------------------------------------------------------
# Background subtraction — activity estimation
# ---------------------------------------------------------------------------

class ActivityEstimator:
    """
    Uses OpenCV MOG2 background subtractor to estimate per-zone activity
    levels from the proportion of motion pixels.

    Activity levels:
        0 → idle   (motion ratio < 0.02)
        1 → moderate (0.02 – 0.10)
        2 → active  (> 0.10)
    """

    IDLE_THRESH = 0.02
    ACTIVE_THRESH = 0.10

    def __init__(self, num_zones: int = 3):
        self.num_zones = num_zones
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        frame : np.ndarray  (H, W, 3) BGR

        Returns
        -------
        np.ndarray of shape (num_zones,) with values in {0, 1, 2}
        """
        fg_mask = self.bg_sub.apply(frame)
        # Remove shadows (shadow pixels = 127 in MOG2)
        fg_mask = (fg_mask == 255).astype(np.uint8)

        h = fg_mask.shape[0]
        zone_h = h // self.num_zones
        activities = np.zeros(self.num_zones, dtype=np.float32)

        for i in range(self.num_zones):
            y_start = i * zone_h
            y_end = (i + 1) * zone_h if i < self.num_zones - 1 else h
            zone = fg_mask[y_start:y_end, :]
            ratio = zone.sum() / max(zone.size, 1)

            if ratio > self.ACTIVE_THRESH:
                activities[i] = 2.0
            elif ratio > self.IDLE_THRESH:
                activities[i] = 1.0
            else:
                activities[i] = 0.0

        return activities


# ---------------------------------------------------------------------------
# Full occupancy pipeline
# ---------------------------------------------------------------------------

class OccupancyPipeline:
    """
    End-to-end pipeline: BGR frame → per-zone occupancy + activity.

    Parameters
    ----------
    density_head_weights : Path or str, optional
        Path to trained density_head.pth.  If None, uses random weights
        (for testing / before training).
    num_zones : int
        Number of horizontal zones to divide the density map into.
    device : str
        'cuda' or 'cpu'.
    """

    INPUT_H = 480
    INPUT_W = 640
    # ImageNet normalisation stats
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        density_head_weights: Optional[Path] = None,
        num_zones: int = 3,
        device: str = "cpu",
    ):
        self.num_zones = num_zones
        self.device = torch.device(device)

        # YOLOv8 backbone
        self.backbone = YOLOv8Backbone().to(self.device)
        self.backbone.eval()

        # Density head
        self.density_head = DensityHead(
            in_channels=YOLOv8Backbone.FEATURE_CHANNELS
        ).to(self.device)

        if density_head_weights is not None:
            state = torch.load(
                str(density_head_weights), map_location=self.device, weights_only=True
            )
            self.density_head.load_state_dict(state)
        self.density_head.eval()

        # Activity estimator (runs on CPU / OpenCV)
        self.activity_estimator = ActivityEstimator(num_zones=num_zones)

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Resize, BGR→RGB, normalise, to tensor [1, 3, H, W]."""
        frame = cv2.resize(frame_bgr, (self.INPUT_W, self.INPUT_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame = (frame - self.MEAN) / self.STD
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def __call__(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the full pipeline on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray (H, W, 3) uint8 BGR

        Returns
        -------
        density_map : np.ndarray (INPUT_H, INPUT_W) float32
            Predicted density map ∈ [0, 1].
        zone_occupancy : np.ndarray (num_zones,) float32
            Summed density per zone (proportional to person count).
        zone_activity : np.ndarray (num_zones,) float32
            Activity level per zone {0, 1, 2}.
        """
        # Stage 1 — features from frozen backbone
        tensor = self._preprocess(frame_bgr)
        features = self.backbone(tensor)

        # Stage 2 — density prediction at feature-map resolution
        feat_h, feat_w = features.shape[2], features.shape[3]
        density = self.density_head(features, feat_h, feat_w)
        density_fm = density.squeeze().cpu().numpy()  # (feat_h, feat_w)

        # Upsample density to full resolution for visualisation
        density_map = cv2.resize(density_fm, (self.INPUT_W, self.INPUT_H),
                                 interpolation=cv2.INTER_LINEAR)

        # Zone occupancy: sum density in each horizontal zone of feature map
        # (summing at feature-map resolution avoids the 64× upsampling bias)
        zone_h = feat_h // self.num_zones
        zone_occ = np.zeros(self.num_zones, dtype=np.float32)
        for i in range(self.num_zones):
            y_start = i * zone_h
            y_end = (i + 1) * zone_h if i < self.num_zones - 1 else feat_h
            zone_occ[i] = density_fm[y_start:y_end, :].sum()

        # Activity (from background subtraction)
        frame_resized = cv2.resize(frame_bgr, (self.INPUT_W, self.INPUT_H))
        zone_activity = self.activity_estimator.estimate(frame_resized)

        return density_map, zone_occ, zone_activity


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def visualize_density(
    frame_bgr: np.ndarray,
    density_map: np.ndarray,
    zone_occupancy: np.ndarray,
    num_zones: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay predicted density map on the original frame as a semi-transparent
    jet-coloured heatmap, draw zone boundary lines, and annotate each zone
    with its occupancy count.

    Parameters
    ----------
    frame_bgr : np.ndarray (H, W, 3) uint8
    density_map : np.ndarray (H, W) float32 (unnormalised density)
    zone_occupancy : np.ndarray (num_zones,)
    num_zones : int
    alpha : float  overlay opacity

    Returns
    -------
    np.ndarray (H, W, 3) uint8 — annotated BGR image
    """
    h, w = frame_bgr.shape[:2]

    # Resize density map to frame dimensions
    dm_resized = cv2.resize(density_map, (w, h))
    # Normalise density to [0, 255] for visualisation only
    dm_max = dm_resized.max()
    if dm_max > 0:
        dm_uint8 = (dm_resized / dm_max * 255).astype(np.uint8)
    else:
        dm_uint8 = np.zeros((h, w), dtype=np.uint8)
    heatmap = cv2.applyColorMap(dm_uint8, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap, alpha, 0)

    # Zone lines and labels
    zone_h = h // num_zones
    for i in range(1, num_zones):
        y = i * zone_h
        cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 2)

    for i in range(num_zones):
        y_center = i * zone_h + zone_h // 2
        label = f"Zone {i + 1}: {zone_occupancy[i]:.1f}"
        cv2.putText(
            overlay, label, (10, y_center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return overlay
