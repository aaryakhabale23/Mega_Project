"""
Real-time floor dashboard using matplotlib.

Premium dark-themed dashboard showing:
  - Top-down floor plan with colour-coded occupancy
  - Per-room people count, device status, and power draw
  - Real-time metrics panel (energy, cost, savings, comfort)
  - Live occupancy bar chart

Usage
-----
    from visualization.dashboard import FloorDashboard
    dash = FloorDashboard(env)
    dash.update(info)           # call every timestep
    dash.close()
"""

from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from typing import Any, Dict, List, Optional

from rl_environment.env import CollegeFloorEnv, ROOM_CONFIGS, ROOM_ORDER


# ---------------------------------------------------------------------------
# Theme colours
# ---------------------------------------------------------------------------
BG_DARK      = "#0F0F1A"
BG_CARD      = "#1A1A2E"
BG_PANEL     = "#16213E"
ACCENT_CYAN  = "#00D4FF"
ACCENT_GREEN = "#00E676"
ACCENT_AMBER = "#FFB300"
ACCENT_RED   = "#FF5252"
TEXT_PRIMARY  = "#EAEAEA"
TEXT_MUTED    = "#8892B0"
BORDER_COLOR = "#2D3561"

OCC_COLORS = {
    "empty":   "#1B5E20",
    "low":     "#2196F3",
    "medium":  "#FF9800",
    "high":    "#F44336",
}


def _occ_color(occ: float) -> str:
    if occ < 0.05:
        return OCC_COLORS["empty"]
    elif occ < 0.30:
        return OCC_COLORS["low"]
    elif occ < 0.65:
        return OCC_COLORS["medium"]
    else:
        return OCC_COLORS["high"]


def _occ_label(occ: float) -> str:
    if occ < 0.05:
        return "EMPTY"
    elif occ < 0.30:
        return "LOW"
    elif occ < 0.65:
        return "MODERATE"
    else:
        return "HIGH"


def _device_status(level: float) -> tuple:
    """Return (label, color) for a device level."""
    if level >= 0.9:
        return ("ON", ACCENT_GREEN)
    elif level >= 0.4:
        return ("MID", ACCENT_AMBER)
    else:
        return ("OFF", "#555555")


class FloorDashboard:
    """
    Premium matplotlib-based real-time floor dashboard.

    Parameters
    ----------
    env : CollegeFloorEnv
        The environment instance.
    always_on_kwh : float, optional
        Daily always-on energy baseline for savings calculation.
    """

    def __init__(self, env: CollegeFloorEnv, always_on_kwh: Optional[float] = None):
        self.env = env
        self.always_on_kwh = always_on_kwh or env.get_always_on_energy()
        self.energy_history: List[float] = []
        self.comfort_history: List[float] = []

        plt.ion()
        self.fig = plt.figure(figsize=(22, 11))
        self.fig.patch.set_facecolor(BG_DARK)

        # Layout: 2 rows x 6 cols
        # Row 0: 5 rooms across top        (cols 0-4) + metrics (col 5)
        # Row 1: corridor + occ bars        (cols 0-2) + metrics cont (cols 3-5)
        gs = self.fig.add_gridspec(
            2, 6,
            height_ratios=[1.2, 1],
            hspace=0.30, wspace=0.12,
            left=0.03, right=0.97, top=0.88, bottom=0.06,
        )

        # Room axes (top row) - one per room
        self.ax_rooms = []
        for i in range(5):
            ax = self.fig.add_subplot(gs[0, i])
            self.ax_rooms.append(ax)

        # Metrics panel (top-right)
        self.ax_metrics = self.fig.add_subplot(gs[0, 5])

        # Corridor (bottom-left)
        self.ax_corridor = self.fig.add_subplot(gs[1, 0:2])

        # Occupancy bars (bottom-middle)
        self.ax_bars = self.fig.add_subplot(gs[1, 2:4])

        # Summary stats (bottom-right)
        self.ax_summary = self.fig.add_subplot(gs[1, 4:6])

        # Title
        self.fig.suptitle(
            "SMART ENVIRONMENT ENERGY OPTIMIZER",
            fontsize=18, fontweight="bold", color=ACCENT_CYAN,
            fontfamily="monospace", y=0.96,
        )

        self._init_axes()

    def _init_axes(self):
        for ax in self.ax_rooms + [self.ax_metrics, self.ax_corridor, self.ax_summary]:
            ax.set_facecolor(BG_CARD)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_color(BORDER_COLOR)

        self.ax_bars.set_facecolor(BG_DARK)

    def update(self, info: Dict[str, Any]) -> None:
        """Redraw the dashboard with current environment state."""
        for ax in self.ax_rooms:
            ax.clear()
        self.ax_metrics.clear()
        self.ax_bars.clear()
        self.ax_corridor.clear()
        self.ax_summary.clear()
        self._init_axes()

        self._draw_rooms(info)
        self._draw_corridor()
        self._draw_metrics(info)
        self._draw_occ_bars()
        self._draw_summary(info)

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Room cards (one per axis)
    # ------------------------------------------------------------------
    def _draw_rooms(self, info: Dict[str, Any]):
        time_str = info.get("current_time", "--:--")
        step = info.get("timestep", 0)

        for i, room_id in enumerate(ROOM_ORDER):
            ax = self.ax_rooms[i]
            cfg = ROOM_CONFIGS[room_id]
            occ = self.env.occupancy[room_id]
            color = _occ_color(occ)
            people = int(round(occ * cfg["capacity"]))
            levels = self.env.device_levels[room_id]
            power_w = self.env.get_room_power_watts(room_id)

            # Background card
            rect = mpatches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.03",
                facecolor=color, edgecolor=ACCENT_CYAN,
                linewidth=2, alpha=0.85,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)

            # --- Room header ---
            ax.text(0.5, 0.93, f"Room {room_id}",
                    ha="center", va="top", fontsize=11, fontweight="bold",
                    color="white", fontfamily="monospace",
                    transform=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

            ax.text(0.5, 0.83, cfg["name"],
                    ha="center", va="top", fontsize=7, color="#D0D0D0",
                    fontfamily="monospace", transform=ax.transAxes)

            # --- People count (big) ---
            ax.text(0.5, 0.62, f"{people}",
                    ha="center", va="center", fontsize=28, fontweight="bold",
                    color="white", fontfamily="monospace",
                    transform=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])

            ax.text(0.5, 0.48, f"people  |  {occ:.0%}",
                    ha="center", va="center", fontsize=7.5, color="#E0E0E0",
                    fontfamily="monospace", transform=ax.transAxes)

            # --- Occupancy label ---
            label = _occ_label(occ)
            label_color = {
                "EMPTY": "#4CAF50", "LOW": "#64B5F6",
                "MODERATE": "#FFB74D", "HIGH": "#EF5350",
            }.get(label, TEXT_MUTED)
            ax.text(0.5, 0.39, label,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=label_color, fontfamily="monospace",
                    transform=ax.transAxes)

            # --- Device status (spaced out) ---
            # Light
            l_stat, l_col = _device_status(levels["light"])
            ax.text(0.17, 0.24, f"L", ha="center", va="center",
                    fontsize=7, color=TEXT_MUTED, fontfamily="monospace",
                    transform=ax.transAxes)
            ax.text(0.17, 0.15, l_stat, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=l_col,
                    fontfamily="monospace", transform=ax.transAxes)

            # Fan
            if cfg["has_fan"]:
                f_stat, f_col = _device_status(levels["fan"])
            else:
                f_stat, f_col = ("--", "#444")
            ax.text(0.50, 0.24, f"F", ha="center", va="center",
                    fontsize=7, color=TEXT_MUTED, fontfamily="monospace",
                    transform=ax.transAxes)
            ax.text(0.50, 0.15, f_stat, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=f_col,
                    fontfamily="monospace", transform=ax.transAxes)

            # AC
            if cfg["has_ac"]:
                a_stat, a_col = _device_status(levels["ac"])
            else:
                a_stat, a_col = ("--", "#444")
            ax.text(0.83, 0.24, f"AC", ha="center", va="center",
                    fontsize=7, color=TEXT_MUTED, fontfamily="monospace",
                    transform=ax.transAxes)
            ax.text(0.83, 0.15, a_stat, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=a_col,
                    fontfamily="monospace", transform=ax.transAxes)

            # --- Power draw ---
            ax.text(0.5, 0.05, f"{power_w:.0f} W",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=ACCENT_AMBER, fontfamily="monospace",
                    transform=ax.transAxes)

    # ------------------------------------------------------------------
    # Corridor
    # ------------------------------------------------------------------
    def _draw_corridor(self):
        ax = self.ax_corridor
        rect = mpatches.FancyBboxPatch(
            (0.02, 0.05), 0.96, 0.90,
            boxstyle="round,pad=0.03",
            facecolor="#37474F", edgecolor="#546E7A",
            linewidth=1.5, alpha=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        corr_watts = 4 * 40 * 0.5
        ax.text(0.5, 0.65, "CORRIDOR",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="#B0BEC5", fontfamily="monospace",
                transform=ax.transAxes)
        ax.text(0.5, 0.40, f"Always On  |  Lights: ON  |  {corr_watts:.0f} W",
                ha="center", va="center", fontsize=9, color="#78909C",
                fontfamily="monospace", transform=ax.transAxes)

    # ------------------------------------------------------------------
    # Metrics panel
    # ------------------------------------------------------------------
    def _draw_metrics(self, info: Dict[str, Any]):
        ax = self.ax_metrics
        ax.set_title("LIVE METRICS", fontsize=10, fontweight="bold",
                     color=TEXT_PRIMARY, fontfamily="monospace", pad=6)

        corr_watts = 4 * 40 * 0.5
        total_power = sum(self.env.get_room_power_watts(r) for r in ROOM_ORDER) + corr_watts
        cum_energy  = info.get("cumulative_energy_kwh", 0.0)
        cum_cost    = info.get("cumulative_cost_rupees", 0.0)
        saving_kwh  = max(self.always_on_kwh - cum_energy, 0.0)
        saving_pct  = (saving_kwh / max(self.always_on_kwh, 1e-9)) * 100
        comfort     = info.get("mean_comfort", 0.0)
        time_str    = info.get("current_time", "--:--")
        step        = info.get("timestep", 0)

        metrics = [
            ("TIME",        f"{time_str}",            TEXT_PRIMARY),
            ("STEP",        f"{step} / 96",           TEXT_PRIMARY),
            ("POWER",       f"{total_power:,.0f} W",  ACCENT_CYAN),
            ("ENERGY",      f"{cum_energy:.2f} kWh",  ACCENT_AMBER),
            ("COST",        f"Rs {cum_cost:.0f}",     ACCENT_AMBER),
            ("SAVINGS",     f"{saving_pct:.0f}%",     ACCENT_GREEN),
            ("COMFORT",     f"{comfort:.0%}",         ACCENT_CYAN if comfort > 0.6 else ACCENT_RED),
        ]

        y_start = 0.92
        spacing = 0.125
        for i, (label, value, color) in enumerate(metrics):
            y = y_start - i * spacing

            # Card background
            card = mpatches.FancyBboxPatch(
                (0.04, y - 0.04), 0.92, 0.10,
                boxstyle="round,pad=0.02",
                facecolor=BG_DARK, edgecolor=BORDER_COLOR,
                linewidth=0.6, alpha=0.9,
                transform=ax.transAxes,
            )
            ax.add_patch(card)

            ax.text(0.10, y, label,
                    transform=ax.transAxes, fontsize=7,
                    color=TEXT_MUTED, fontfamily="monospace", va="center")
            ax.text(0.90, y, value,
                    transform=ax.transAxes, fontsize=9,
                    fontweight="bold", color=color, fontfamily="monospace",
                    ha="right", va="center")

    # ------------------------------------------------------------------
    # Occupancy bar chart
    # ------------------------------------------------------------------
    def _draw_occ_bars(self):
        ax = self.ax_bars
        ax.set_facecolor(BG_DARK)
        ax.set_title("OCCUPANCY BY ROOM", fontsize=10, fontweight="bold",
                     color=TEXT_PRIMARY, fontfamily="monospace", pad=6)

        rooms = ROOM_ORDER
        occupancies = [self.env.occupancy[r] for r in rooms]
        capacities = [ROOM_CONFIGS[r]["capacity"] for r in rooms]
        people = [int(round(o * c)) for o, c in zip(occupancies, capacities)]
        colors = [_occ_color(o) for o in occupancies]

        x_pos = np.arange(len(rooms))
        bars = ax.bar(x_pos, occupancies, color=colors, edgecolor="white",
                      linewidth=0.8, alpha=0.85, width=0.55)

        for i, (bar, p) in enumerate(zip(bars, people)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
                    f"{p}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="white", fontfamily="monospace")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"R{r}" for r in rooms], color=TEXT_MUTED,
                           fontfamily="monospace", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"],
                           color=TEXT_MUTED, fontsize=7)
        ax.axhline(y=0.30, color=ACCENT_AMBER, linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axhline(y=0.65, color=ACCENT_RED, linestyle="--", alpha=0.3, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER_COLOR)
        ax.spines["bottom"].set_color(BORDER_COLOR)

    # ------------------------------------------------------------------
    # Summary panel
    # ------------------------------------------------------------------
    def _draw_summary(self, info: Dict[str, Any]):
        ax = self.ax_summary
        ax.set_title("SAVINGS SUMMARY", fontsize=10, fontweight="bold",
                     color=TEXT_PRIMARY, fontfamily="monospace", pad=6)

        cum_energy  = info.get("cumulative_energy_kwh", 0.0)
        saving_kwh  = max(self.always_on_kwh - cum_energy, 0.0)
        saving_pct  = (saving_kwh / max(self.always_on_kwh, 1e-9)) * 100

        # Savings donut-like visual
        total_people = sum(
            int(round(self.env.occupancy[r] * ROOM_CONFIGS[r]["capacity"]))
            for r in ROOM_ORDER
        )

        items = [
            ("Total People",    f"{total_people}",              ACCENT_CYAN,  0.82),
            ("Baseline",        f"{self.always_on_kwh:.0f} kWh",TEXT_MUTED,   0.65),
            ("AI Optimized",    f"{cum_energy:.1f} kWh",        ACCENT_GREEN, 0.48),
            ("Energy Saved",    f"{saving_kwh:.1f} kWh",        ACCENT_GREEN, 0.31),
            ("Money Saved",     f"Rs {saving_kwh * 8:.0f}",     ACCENT_GREEN, 0.14),
        ]

        for label, value, color, y in items:
            ax.text(0.10, y, label,
                    transform=ax.transAxes, fontsize=8,
                    color=TEXT_MUTED, fontfamily="monospace", va="center")
            ax.text(0.90, y, value,
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=color, fontfamily="monospace",
                    ha="right", va="center")

    def close(self) -> None:
        """Close the dashboard window."""
        plt.ioff()
        plt.close(self.fig)
