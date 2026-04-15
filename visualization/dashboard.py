"""
Real-time floor dashboard using matplotlib.

Displays a top-down schematic of the college floor with colour-coded
occupancy, device settings, power draw per room, and a sidebar with
aggregate metrics (total power, energy, cost, estimated savings).

Usage
-----
    from visualization.dashboard import FloorDashboard
    dash = FloorDashboard(env)
    dash.update(info)           # call every timestep
    dash.close()
"""

from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")  # or "Agg" for non-interactive / Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Any, Dict, Optional

from rl_environment.env import CollegeFloorEnv, ROOM_CONFIGS, ROOM_ORDER


# ---------------------------------------------------------------------------
# Room layout coordinates (top-down schematic)
# ---------------------------------------------------------------------------

# (x, y, width, height) in figure units
ROOM_LAYOUT = {
    "101": (0.05, 0.55, 0.25, 0.35),
    "102": (0.35, 0.55, 0.25, 0.35),
    "103": (0.65, 0.55, 0.25, 0.35),
    "104": (0.05, 0.10, 0.25, 0.30),
    "105": (0.35, 0.10, 0.25, 0.30),
    "corridor": (0.65, 0.10, 0.25, 0.30),
}


def _occupancy_color(occ: float) -> str:
    """Map normalised occupancy → hex colour (blue→yellow→red)."""
    if occ < 0.3:
        return "#3B82F6"    # blue
    elif occ < 0.7:
        return "#F59E0B"    # amber/yellow
    else:
        return "#EF4444"    # red


class FloorDashboard:
    """
    Matplotlib-based real-time floor dashboard.

    Parameters
    ----------
    env : CollegeFloorEnv
        The environment instance (used for metadata only).
    always_on_kwh : float, optional
        Daily always-on energy baseline for savings calculation.
    """

    def __init__(self, env: CollegeFloorEnv, always_on_kwh: Optional[float] = None):
        self.env = env
        self.always_on_kwh = always_on_kwh or env.get_always_on_energy()

        plt.ion()
        self.fig, (self.ax_floor, self.ax_sidebar) = plt.subplots(
            1, 2, figsize=(16, 8),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        self.fig.patch.set_facecolor("#1E1E2E")
        self.fig.suptitle(
            "Smart Environment Energy Optimizer — Floor Dashboard",
            fontsize=14, fontweight="bold", color="white",
        )
        self.ax_floor.set_facecolor("#1E1E2E")
        self.ax_sidebar.set_facecolor("#1E1E2E")
        self.ax_floor.set_xlim(0, 1)
        self.ax_floor.set_ylim(0, 1)
        self.ax_floor.axis("off")
        self.ax_sidebar.axis("off")

        self._drawn = False

    def update(self, info: Dict[str, Any]) -> None:
        """
        Redraw the dashboard with current environment state.

        Parameters
        ----------
        info : dict
            The info dict returned by ``env.step()``.
        """
        self.ax_floor.clear()
        self.ax_sidebar.clear()
        self.ax_floor.set_facecolor("#1E1E2E")
        self.ax_sidebar.set_facecolor("#1E1E2E")
        self.ax_floor.set_xlim(0, 1)
        self.ax_floor.set_ylim(0, 1)
        self.ax_floor.axis("off")
        self.ax_sidebar.axis("off")

        # ---- Floor plan ----
        for room_id in ROOM_ORDER:
            x, y, w, h = ROOM_LAYOUT[room_id]
            cfg = ROOM_CONFIGS[room_id]
            occ = self.env.occupancy[room_id]
            color = _occupancy_color(occ)

            # Room rectangle
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor="white",
                linewidth=1.5, alpha=0.85,
            )
            self.ax_floor.add_patch(rect)

            # Room label
            levels = self.env.device_levels[room_id]
            l_val = int(levels["light"] * 2)
            f_val = int(levels["fan"] * 2) if cfg["has_fan"] else "-"
            a_val = int(levels["ac"] * 2) if cfg["has_ac"] else "-"
            power_w = self.env.get_room_power_watts(room_id)

            lines = [
                f"Room {room_id}",
                f"{cfg['name']}",
                f"Occ: {occ:.0%}",
                f"L:{l_val}  F:{f_val}  AC:{a_val}",
                f"{power_w:.0f} W",
            ]
            text = "\n".join(lines)
            self.ax_floor.text(
                x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="white", family="monospace",
            )

        # Corridor
        cx, cy, cw, ch = ROOM_LAYOUT["corridor"]
        rect = mpatches.FancyBboxPatch(
            (cx, cy), cw, ch,
            boxstyle="round,pad=0.01",
            facecolor="#6B7280", edgecolor="white",
            linewidth=1.5, alpha=0.7,
        )
        self.ax_floor.add_patch(rect)
        corr_watts = 4 * 40 * 0.5  # minimum lighting
        self.ax_floor.text(
            cx + cw / 2, cy + ch / 2,
            f"Corridor\n(always on)\n{corr_watts:.0f} W",
            ha="center", va="center",
            fontsize=9, color="white", fontweight="bold",
        )

        self.ax_floor.set_title(
            f"Time: {info.get('current_time', '--:--')}  |  "
            f"Step: {info.get('timestep', 0)}/96",
            fontsize=11, color="white", pad=10,
        )

        # ---- Sidebar ----
        total_power = sum(self.env.get_room_power_watts(r) for r in ROOM_ORDER) + corr_watts
        cum_energy = info.get("cumulative_energy_kwh", 0.0)
        cum_cost = info.get("cumulative_cost_rupees", 0.0)
        saving_kwh = max(self.always_on_kwh - cum_energy, 0.0)
        saving_pct = (saving_kwh / max(self.always_on_kwh, 1e-9)) * 100
        comfort = info.get("mean_comfort", 0.0)

        sidebar_text = (
            "── Summary ──\n\n"
            f"Total Power  : {total_power:,.0f} W\n\n"
            f"Energy Today : {cum_energy:.2f} kWh\n\n"
            f"Cost Today   : ₹{cum_cost:.2f}\n\n"
            f"Saving vs\n"
            f"  Always-On  : {saving_kwh:.2f} kWh\n"
            f"               ({saving_pct:.1f}%)\n\n"
            f"Comfort      : {comfort:.2%}\n"
        )
        self.ax_sidebar.text(
            0.1, 0.95, sidebar_text,
            transform=self.ax_sidebar.transAxes,
            fontsize=11, color="#A5F3FC",
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2D2D44", edgecolor="#A5F3FC", alpha=0.8),
        )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor="#3B82F6", label="Low (<30%)"),
            mpatches.Patch(facecolor="#F59E0B", label="Medium (30-70%)"),
            mpatches.Patch(facecolor="#EF4444", label="High (>70%)"),
        ]
        self.ax_sidebar.legend(
            handles=legend_elements, loc="lower center",
            fontsize=9, title="Occupancy", title_fontsize=10,
            facecolor="#2D2D44", edgecolor="white",
            labelcolor="white",
        )
        self.ax_sidebar.get_legend().get_title().set_color("white")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self) -> None:
        """Close the dashboard window."""
        plt.ioff()
        plt.close(self.fig)
