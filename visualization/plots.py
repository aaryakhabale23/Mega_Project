"""
Static result plots for project evaluation and presentation.

Each function generates a single publication-quality PNG figure.
All plots use a dark theme consistent with the dashboard.

Functions
---------
    plot_learning_curve
    plot_energy_comparison
    plot_comfort_energy_tradeoff
    plot_room_behavior
    plot_dl_results_panel
"""

from __future__ import annotations

import cv2
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

SEED = 42

# ---------- Dark theme setup ----------
DARK_BG = "#1E1E2E"
CARD_BG = "#2D2D44"
ACCENT_CYAN = "#A5F3FC"
ACCENT_GREEN = "#34D399"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"
ACCENT_PURPLE = "#A78BFA"
TEXT_COLOR = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#4A5568",
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": "#4A5568",
    "font.family": "sans-serif",
    "font.size": 11,
    "grid.color": "#4A5568",
    "grid.alpha": 0.4,
})


# ======================================================================
# 1. Learning curve
# ======================================================================

def plot_learning_curve(
    reward_log_path: str | Path,
    random_reward: float,
    rule_based_reward: float,
    save_path: str | Path = "plots/learning_curve.png",
    window: int = 20,
) -> None:
    """
    Plot episode reward over training timesteps with a smoothed moving
    average and reference lines for baselines.

    Parameters
    ----------
    reward_log_path : path
        .npy file with shape (N, 2) — [timestep, mean_reward].
    random_reward : float
        Mean episode reward from the random baseline.
    rule_based_reward : float
        Mean episode reward from the rule-based baseline.
    save_path : path
        Output PNG path.
    window : int
        Moving average window size.
    """
    data = np.load(str(reward_log_path), allow_pickle=True)
    if data.ndim == 1:
        # list of (step, reward) tuples
        data = np.array([list(d) for d in data])
    steps = data[:, 0]
    rewards = data[:, 1]

    # Smoothed moving average
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_steps = steps[window - 1:]
    else:
        smoothed = rewards
        smooth_steps = steps

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, rewards, alpha=0.25, color=ACCENT_CYAN, linewidth=0.8, label="Raw")
    ax.plot(smooth_steps, smoothed, color=ACCENT_CYAN, linewidth=2.5, label=f"Smoothed (w={window})")

    ax.axhline(random_reward, color=ACCENT_RED, linestyle="--", linewidth=1.5, label=f"Random ({random_reward:.1f})")
    ax.axhline(rule_based_reward, color=ACCENT_AMBER, linestyle="--", linewidth=1.5, label=f"Rule-Based ({rule_based_reward:.1f})")

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("PPO Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ======================================================================
# 2. Energy comparison bar chart
# ======================================================================

def plot_energy_comparison(
    energy_data: Dict[str, float],
    cost_per_kwh: float = 8.0,
    save_path: str | Path = "plots/energy_comparison.png",
) -> None:
    """
    Bar chart comparing average daily energy (kWh) and cost (₹) across
    Always-On, Rule-Based, and PPO policies.

    Parameters
    ----------
    energy_data : dict
        Keys = policy names, values = average daily energy in kWh.
    cost_per_kwh : float
    save_path : path
    """
    names = list(energy_data.keys())
    energies = [energy_data[n] for n in names]
    costs = [e * cost_per_kwh for e in energies]

    max_energy = max(energies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Energy bars
    colors = [ACCENT_RED, ACCENT_AMBER, ACCENT_GREEN][:len(names)]
    bars1 = ax1.bar(names, energies, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Average Daily Energy Consumption", fontweight="bold")
    ax1.grid(axis="y")

    # Annotate savings
    for i, bar in enumerate(bars1):
        saving_pct = (1 - energies[i] / max_energy) * 100 if max_energy > 0 else 0
        label = f"{energies[i]:.1f} kWh"
        if saving_pct > 0:
            label += f"\n(−{saving_pct:.0f}%)"
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max_energy * 0.02,
            label, ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COLOR,
        )

    # Cost bars
    bars2 = ax2.bar(names, costs, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    ax2.set_ylabel("Cost (₹)")
    ax2.set_title("Average Daily Electricity Cost", fontweight="bold")
    ax2.grid(axis="y")

    for i, bar in enumerate(bars2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(costs) * 0.02,
            f"₹{costs[i]:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COLOR,
        )

    fig.suptitle("Energy & Cost Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ======================================================================
# 3. Comfort-energy trade-off scatter
# ======================================================================

def plot_comfort_energy_tradeoff(
    sweep_results: Dict[float, Dict],
    save_path: str | Path = "plots/comfort_energy_tradeoff.png",
) -> None:
    """
    Scatter plot with one point per beta value.  X = energy, Y = comfort.

    Parameters
    ----------
    sweep_results : dict
        beta → {"mean_comfort": (mean, std), "total_energy_kwh": (mean, std)}
    save_path : path
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = [ACCENT_GREEN, ACCENT_CYAN, ACCENT_PURPLE]
    for i, (beta, res) in enumerate(sorted(sweep_results.items())):
        energy_mean = res["total_energy_kwh"][0]
        comfort_mean = res["mean_comfort"][0]
        ax.scatter(
            energy_mean, comfort_mean,
            s=250, c=colors[i % len(colors)], edgecolors="white",
            linewidths=2, zorder=5,
        )
        ax.annotate(
            f"β={beta}",
            (energy_mean, comfort_mean),
            textcoords="offset points", xytext=(12, 12),
            fontsize=12, fontweight="bold", color=colors[i % len(colors)],
            arrowprops=dict(arrowstyle="->", color=colors[i % len(colors)], lw=1.5),
        )

    ax.set_xlabel("Average Daily Energy (kWh)", fontsize=12)
    ax.set_ylabel("Average Comfort Score", fontsize=12)
    ax.set_title("Comfort–Energy Trade-off (Beta Sweep)", fontsize=14, fontweight="bold")
    ax.grid(True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ======================================================================
# 4. Room-level behavior time-series
# ======================================================================

def plot_room_behavior(
    episode_data: Dict[str, Dict[str, List[float]]],
    save_path: str | Path = "plots/room_behavior.png",
) -> None:
    """
    One subplot per room showing occupancy (blue) and lighting level (orange)
    over 96 timesteps with time-of-day x-axis.

    Parameters
    ----------
    episode_data : dict
        room_id → {"occupancy": list[96], "light_level": list[96]}
    save_path : path
    """
    rooms = sorted(episode_data.keys())
    n = len(rooms)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    timesteps = np.arange(96)
    hours = 8.0 + timesteps * (10.0 / 96)

    for ax, room_id in zip(axes, rooms):
        data = episode_data[room_id]
        occ = data.get("occupancy", [0.0] * 96)
        light = data.get("light_level", [0.0] * 96)

        ax.plot(hours, occ, color=ACCENT_CYAN, linewidth=2, label="Occupancy")
        ax.plot(hours, light, color=ACCENT_AMBER, linewidth=2, label="Lighting Level", linestyle="--")
        ax.fill_between(hours, occ, alpha=0.15, color=ACCENT_CYAN)

        ax.set_ylabel("Level (0–1)")
        ax.set_title(f"Room {room_id}", fontweight="bold", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True)

    axes[-1].set_xlabel("Time of Day")
    axes[-1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x)}:00" if x == int(x) else "")
    )
    axes[-1].set_xlim(8, 18)

    fig.suptitle("Room-Level Behavior Over One Day", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ======================================================================
# 5. DL results panel
# ======================================================================

def plot_dl_results_panel(
    image_paths: List[str | Path],
    gt_density_paths: List[str | Path],
    pred_density_paths: List[str | Path],
    save_path: str | Path = "plots/dl_results_panel.png",
) -> None:
    """
    Show up to 5 test frames side-by-side with ground truth and predicted
    density maps.

    Parameters
    ----------
    image_paths : list of paths to test images
    gt_density_paths : list of paths to ground-truth density .npy files
    pred_density_paths : list of paths to predicted density .npy files
    save_path : path
    """
    n = min(len(image_paths), 5)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))

    if n == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["Input Frame", "Ground Truth Density", "Predicted Density"]

    for i in range(n):
        # Input image
        img = cv2.imread(str(image_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title(Path(image_paths[i]).stem, fontsize=9)
        axes[0, i].axis("off")

        # Ground truth
        gt = np.load(str(gt_density_paths[i]))
        axes[1, i].imshow(gt, cmap="jet", vmin=0)
        axes[1, i].axis("off")

        # Prediction
        pred = np.load(str(pred_density_paths[i]))
        axes[2, i].imshow(pred, cmap="jet", vmin=0)
        axes[2, i].axis("off")

    # Row labels
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=11, fontweight="bold", rotation=90, labelpad=12)

    fig.suptitle("Density Estimation — Ground Truth vs Prediction", fontsize=14, fontweight="bold")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ======================================================================
# Convenience: collect one episode of room behavior data
# ======================================================================

def collect_episode_data(env, policy) -> Dict[str, Dict[str, List[float]]]:
    """
    Run one episode with *policy* and record occupancy + lighting per room
    per timestep for ``plot_room_behavior``.
    """
    from rl_environment.env import ROOM_ORDER

    data: Dict[str, Dict[str, List[float]]] = {
        r: {"occupancy": [], "light_level": []} for r in ROOM_ORDER
    }

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        for r in ROOM_ORDER:
            data[r]["occupancy"].append(env.occupancy[r])
            data[r]["light_level"].append(env.device_levels[r]["light"])

    return data
