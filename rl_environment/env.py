"""
CollegeFloorEnv — Custom Gymnasium environment modelling a real college floor.

Models 5 controllable rooms (101-105) plus a fixed corridor.
The RL agent controls lighting, fans, and AC per room to minimize energy
consumption and electricity cost (₹8/kWh) while maintaining occupant comfort.

Observation space : Box(27,)  — per-room [occ, activity, light, fan, ac] × 5 rooms + 2 global
Action space      : MultiDiscrete([3]*12) — 12 controllable devices, each 3 levels
Episode length    : 96 timesteps (8 AM – 6 PM, ~6.25 min each)
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

SEED = 42


# ---------------------------------------------------------------------------
# Room & device specification
# ---------------------------------------------------------------------------

ROOM_CONFIGS = {
    "101": {
        "name": "Lecture Hall",
        "capacity": 60,
        "lights": {"count": 8, "wattage": 40},
        "fans": {"count": 2, "wattage": 75},
        "ac": {"count": 1, "wattage": 1500},
        "computers": None,
        "has_light": True,
        "has_fan": True,
        "has_ac": True,
    },
    "102": {
        "name": "Lecture Hall",
        "capacity": 60,
        "lights": {"count": 8, "wattage": 40},
        "fans": {"count": 2, "wattage": 75},
        "ac": {"count": 1, "wattage": 1500},
        "computers": None,
        "has_light": True,
        "has_fan": True,
        "has_ac": True,
    },
    "103": {
        "name": "Computer Lab",
        "capacity": 40,
        "lights": {"count": 6, "wattage": 20},
        "fans": None,
        "ac": {"count": 1, "wattage": 1500},
        "computers": {"count": 40, "idle_wattage": 80},
        "has_light": True,
        "has_fan": False,
        "has_ac": True,
    },
    "104": {
        "name": "Tutorial Room",
        "capacity": 30,
        "lights": {"count": 4, "wattage": 40},
        "fans": {"count": 2, "wattage": 75},
        "ac": None,
        "computers": None,
        "has_light": True,
        "has_fan": True,
        "has_ac": False,
    },
    "105": {
        "name": "Staff Room",
        "capacity": 10,
        "lights": {"count": 4, "wattage": 40},
        "fans": None,
        "ac": {"count": 1, "wattage": 1500},
        "computers": None,
        "has_light": True,
        "has_fan": False,
        "has_ac": True,
    },
}

CORRIDOR_CONFIG = {
    "lights": {"count": 4, "wattage": 40},
}

ROOM_ORDER: List[str] = ["101", "102", "103", "104", "105"]

# Action indices → (room_id, device_type)
ACTION_MAP: List[Tuple[str, str]] = [
    ("101", "light"), ("101", "fan"), ("101", "ac"),       # 0-2
    ("102", "light"), ("102", "fan"), ("102", "ac"),       # 3-5
    ("103", "light"), ("103", "ac"),                       # 6-7
    ("104", "light"), ("104", "fan"),                      # 8-9
    ("105", "light"), ("105", "ac"),                       # 10-11
]

NUM_ACTIONS = len(ACTION_MAP)  # 12

# Discrete level → fractional level
LEVEL_FRACTION = {0: 0.0, 1: 0.5, 2: 1.0}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CollegeFloorEnv(gym.Env):
    """
    Gymnasium environment that simulates energy management for a college floor.

    The agent receives per-room occupancy/activity observations and controls
    lighting, fans, and AC.  The reward balances occupant comfort against
    energy cost and device-switching penalties.

    Parameters
    ----------
    alpha : float
        Weight for comfort in the reward.
    beta : float
        Weight for energy cost in the reward.
    gamma : float
        Weight for device adjustment penalty in the reward.
    cost_per_kwh : float
        Electricity tariff in ₹ per kWh.
    seed : int
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    EPISODE_LENGTH = 96
    HOURS_PER_STEP = 10.0 / 96          # ≈ 0.1042 hours per timestep
    START_HOUR = 8.0                     # 8 AM
    END_HOUR = 18.0                      # 6 PM
    OBS_DIM = len(ROOM_ORDER) * 5 + 2   # 27

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.6,
        gamma: float = 0.1,
        cost_per_kwh: float = 8.0,
        seed: int = SEED,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cost_per_kwh = cost_per_kwh
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * NUM_ACTIONS, seed=seed)

        # Internal state  (initialised properly in reset())
        self.timestep: int = 0
        self.phase_offset: float = 0.0
        self.occupancy: Dict[str, float] = {r: 0.0 for r in ROOM_ORDER}
        self.activity: Dict[str, float] = {r: 0.0 for r in ROOM_ORDER}
        self.device_levels: Dict[str, Dict[str, float]] = {
            r: {"light": 0.0, "fan": 0.0, "ac": 0.0} for r in ROOM_ORDER
        }
        self.prev_action: Optional[np.ndarray] = None
        self.outside_temp: float = 35.0  # °C (default hot Indian summer)

        # Cumulative trackers for info
        self.total_energy_kwh: float = 0.0
        self.total_cost_rupees: float = 0.0

        # Pre-compute base occupancy curves
        self._base_curves = self._build_occupancy_curves()

    # ------------------------------------------------------------------
    # Occupancy simulation
    # ------------------------------------------------------------------

    def _build_occupancy_curves(self) -> Dict[str, np.ndarray]:
        """
        Create realistic base occupancy curves for each room over 96 steps.

        The curves are normalised Gaussian bumps that model typical college
        schedule patterns.  Noise and phase offsets are applied at runtime.
        """
        t = np.linspace(0.0, 1.0, self.EPISODE_LENGTH)
        curves: Dict[str, np.ndarray] = {}

        # 101 — Lecture Hall: peaks 9-11 AM (t≈0.1-0.3) and 1-3 PM (t≈0.5-0.7)
        curves["101"] = np.clip(
            0.85 * np.exp(-0.5 * ((t - 0.15) / 0.08) ** 2)
            + 0.75 * np.exp(-0.5 * ((t - 0.55) / 0.08) ** 2),
            0.0, 1.0,
        )

        # 102 — Lecture Hall: slightly offset schedule
        curves["102"] = np.clip(
            0.75 * np.exp(-0.5 * ((t - 0.20) / 0.08) ** 2)
            + 0.80 * np.exp(-0.5 * ((t - 0.60) / 0.08) ** 2),
            0.0, 1.0,
        )

        # 103 — Computer Lab: peaks 2-5 PM (t≈0.6-0.9)
        curves["103"] = np.clip(
            0.80 * np.exp(-0.5 * ((t - 0.75) / 0.12) ** 2),
            0.0, 1.0,
        )

        # 104 — Tutorial Room: scattered short sessions
        curves["104"] = np.clip(
            0.45 * np.exp(-0.5 * ((t - 0.25) / 0.06) ** 2)
            + 0.35 * np.exp(-0.5 * ((t - 0.50) / 0.06) ** 2)
            + 0.40 * np.exp(-0.5 * ((t - 0.80) / 0.06) ** 2),
            0.0, 1.0,
        )

        # 105 — Staff Room: steady low occupancy 9 AM – 5 PM
        sigmoid = lambda x, c, w: 1.0 / (1.0 + np.exp(-w * (x - c)))
        curves["105"] = np.clip(
            0.50 * (sigmoid(t, 0.10, 30) - sigmoid(t, 0.88, 30)),
            0.0, 1.0,
        )

        return curves

    def _get_occupancy(self, room: str) -> float:
        """Return noisy occupancy for *room* at current timestep."""
        base = self._base_curves[room]
        # Apply phase offset (shifts the curve slightly)
        idx = (self.timestep + int(self.phase_offset * self.EPISODE_LENGTH)) % self.EPISODE_LENGTH
        occ = base[idx]
        # Add Gaussian noise
        occ += self.rng.normal(0.0, 0.1)
        return float(np.clip(occ, 0.0, 1.0))

    def _get_activity(self, room: str) -> float:
        """Derive normalised activity level from occupancy (simulation proxy)."""
        occ = self.occupancy[room]
        if occ > 0.6:
            return 1.0  # active
        elif occ > 0.25:
            return 0.5  # moderate
        return 0.0      # idle

    # ------------------------------------------------------------------
    # Energy model
    # ------------------------------------------------------------------

    def _compute_room_energy(self, room: str) -> float:
        """
        Compute energy consumed by *room* in this timestep (kWh).

        Takes into account the current device levels and room hardware.
        """
        cfg = ROOM_CONFIGS[room]
        levels = self.device_levels[room]
        h = self.HOURS_PER_STEP

        energy = 0.0

        # Lighting
        if cfg["has_light"]:
            light_watts = cfg["lights"]["count"] * cfg["lights"]["wattage"]
            energy += light_watts * levels["light"] * h / 1000.0

        # Fans
        if cfg["has_fan"]:
            fan_watts = cfg["fans"]["count"] * cfg["fans"]["wattage"]
            energy += fan_watts * levels["fan"] * h / 1000.0

        # AC
        if cfg["has_ac"]:
            ac_watts = cfg["ac"]["count"] * cfg["ac"]["wattage"]
            energy += ac_watts * levels["ac"] * h / 1000.0

        # Computers (idle draw when lab is occupied)
        if cfg["computers"] is not None and self.occupancy[room] > 0.05:
            comp_watts = cfg["computers"]["count"] * cfg["computers"]["idle_wattage"]
            energy += comp_watts * h / 1000.0

        return energy

    def _compute_corridor_energy(self) -> float:
        """Fixed corridor energy per timestep (always minimum lighting)."""
        watts = CORRIDOR_CONFIG["lights"]["count"] * CORRIDOR_CONFIG["lights"]["wattage"]
        return watts * 0.5 * self.HOURS_PER_STEP / 1000.0  # half brightness

    # ------------------------------------------------------------------
    # Comfort model
    # ------------------------------------------------------------------

    @staticmethod
    def _required_level(occ: float) -> float:
        """Map normalised occupancy to a required device level (0 / 0.5 / 1.0)."""
        if occ > 0.7:
            return 1.0
        elif occ > 0.3:
            return 0.5
        return 0.0

    def _compute_comfort(self, room: str) -> float:
        """
        Compute comfort score ∈ [0, 1] for *room*.

        Comfort = 1 − mean(max(required − actual, 0)) across devices.
        Only under-provision hurts comfort; over-provision is captured by
        the energy penalty instead.
        """
        cfg = ROOM_CONFIGS[room]
        occ = self.occupancy[room]
        levels = self.device_levels[room]
        req = self._required_level(occ)

        deviations: List[float] = []

        if cfg["has_light"]:
            deviations.append(max(req - levels["light"], 0.0))
        if cfg["has_fan"]:
            deviations.append(max(req - levels["fan"], 0.0))
        if cfg["has_ac"]:
            deviations.append(max(req - levels["ac"], 0.0))

        if not deviations:
            return 1.0

        return float(1.0 - np.mean(deviations))

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """
        Build the 27-dim observation vector.

        Layout:
            [room_101_occ, room_101_act, room_101_light, room_101_fan, room_101_ac,
             room_102_..., room_103_..., room_104_..., room_105_...,
             outside_temp_norm, time_of_day_norm]
        """
        obs = []
        for room in ROOM_ORDER:
            cfg = ROOM_CONFIGS[room]
            obs.append(self.occupancy[room])
            obs.append(self.activity[room])
            obs.append(self.device_levels[room]["light"])
            obs.append(self.device_levels[room]["fan"] if cfg["has_fan"] else 0.0)
            obs.append(self.device_levels[room]["ac"] if cfg["has_ac"] else 0.0)

        # Global features
        temp_norm = np.clip((self.outside_temp - 15.0) / 30.0, 0.0, 1.0)
        time_norm = self.timestep / self.EPISODE_LENGTH
        obs.extend([temp_norm, time_norm])

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> int:
        """Apply discrete action vector → device levels.  Returns num changes."""
        changes = 0
        for i, (room, device) in enumerate(ACTION_MAP):
            new_level = LEVEL_FRACTION[int(action[i])]
            old_level = self.device_levels[room][device]
            if new_level != old_level:
                changes += 1
            self.device_levels[room][device] = new_level
        return changes

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to timestep 0.

        Adds a random phase offset to occupancy curves so each episode is
        slightly different.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep = 0
        self.phase_offset = self.rng.uniform(-0.05, 0.05)
        self.prev_action = None
        self.total_energy_kwh = 0.0
        self.total_cost_rupees = 0.0

        # Randomise outside temperature (Indian climate 28-42 °C)
        self.outside_temp = self.rng.uniform(28.0, 42.0)

        # Reset devices to off
        for room in ROOM_ORDER:
            self.device_levels[room] = {"light": 0.0, "fan": 0.0, "ac": 0.0}

        # Seed initial occupancy
        for room in ROOM_ORDER:
            self.occupancy[room] = self._get_occupancy(room)
            self.activity[room] = self._get_activity(room)

        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the environment by one timestep.

        Parameters
        ----------
        action : np.ndarray of shape (12,) with values in {0, 1, 2}

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.int64)

        # 1. Apply action
        num_changes = self._apply_action(action)

        # 2. Advance time & update occupancy
        self.timestep += 1
        for room in ROOM_ORDER:
            self.occupancy[room] = self._get_occupancy(room)
            self.activity[room] = self._get_activity(room)

        # 3. Compute per-room energy & comfort
        room_energy: Dict[str, float] = {}
        room_comfort: Dict[str, float] = {}
        for room in ROOM_ORDER:
            room_energy[room] = self._compute_room_energy(room)
            room_comfort[room] = self._compute_comfort(room)

        corridor_energy = self._compute_corridor_energy()
        total_energy = sum(room_energy.values()) + corridor_energy
        total_cost = total_energy * self.cost_per_kwh

        self.total_energy_kwh += total_energy
        self.total_cost_rupees += total_cost

        # 4. Reward
        mean_comfort = float(np.mean(list(room_comfort.values())))
        adjustment_penalty = num_changes / NUM_ACTIONS
        reward = (
            self.alpha * mean_comfort
            - self.beta * total_energy * self.cost_per_kwh
            - self.gamma * adjustment_penalty
        )

        # 5. Done?
        terminated = self.timestep >= self.EPISODE_LENGTH
        truncated = False

        # 6. Current time string
        hours_elapsed = self.timestep * self.HOURS_PER_STEP
        current_hour = self.START_HOUR + hours_elapsed
        hour_int = int(current_hour)
        minute_int = int((current_hour - hour_int) * 60)
        time_str = f"{hour_int:02d}:{minute_int:02d}"

        info: Dict[str, Any] = {
            "room_energy_kwh": room_energy,
            "corridor_energy_kwh": corridor_energy,
            "room_comfort": room_comfort,
            "total_energy_kwh": total_energy,
            "total_cost_rupees": total_cost,
            "cumulative_energy_kwh": self.total_energy_kwh,
            "cumulative_cost_rupees": self.total_cost_rupees,
            "mean_comfort": mean_comfort,
            "current_time": time_str,
            "timestep": self.timestep,
        }

        self.prev_action = action.copy()

        return self._build_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Camera integration helper
    # ------------------------------------------------------------------

    @classmethod
    def from_dl_output(
        cls,
        env: "CollegeFloorEnv",
        zone_occupancy: np.ndarray,
        zone_activity: np.ndarray,
        zone_room_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Override simulated occupancy with real camera-derived values.

        Parameters
        ----------
        env : CollegeFloorEnv
            The live environment instance.
        zone_occupancy : np.ndarray
            Per-zone normalised occupancy from the DL pipeline (length N).
        zone_activity : np.ndarray
            Per-zone normalised activity from background subtraction (length N).
        zone_room_mapping : dict, optional
            Maps zone index → room id.  Defaults to {0: '101', 1: '102', 2: '103'}.
        """
        if zone_room_mapping is None:
            zone_room_mapping = {0: "101", 1: "102", 2: "103"}

        for zone_idx, room_id in zone_room_mapping.items():
            if room_id in ROOM_ORDER and zone_idx < len(zone_occupancy):
                env.occupancy[room_id] = float(np.clip(zone_occupancy[zone_idx], 0.0, 1.0))
                env.activity[room_id] = float(np.clip(zone_activity[zone_idx], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_always_on_energy(self) -> float:
        """
        Compute total daily energy if every device were at full power for the
        entire episode (always-on baseline).
        """
        total = 0.0
        for room in ROOM_ORDER:
            cfg = ROOM_CONFIGS[room]
            h = self.HOURS_PER_STEP * self.EPISODE_LENGTH

            if cfg["has_light"]:
                total += cfg["lights"]["count"] * cfg["lights"]["wattage"] * h / 1000.0
            if cfg["has_fan"]:
                total += cfg["fans"]["count"] * cfg["fans"]["wattage"] * h / 1000.0
            if cfg["has_ac"]:
                total += cfg["ac"]["count"] * cfg["ac"]["wattage"] * h / 1000.0
            if cfg["computers"] is not None:
                total += cfg["computers"]["count"] * cfg["computers"]["idle_wattage"] * h / 1000.0

        # Corridor
        total += (
            CORRIDOR_CONFIG["lights"]["count"]
            * CORRIDOR_CONFIG["lights"]["wattage"]
            * 0.5
            * self.HOURS_PER_STEP
            * self.EPISODE_LENGTH
            / 1000.0
        )
        return total

    def get_room_power_watts(self, room: str) -> float:
        """Return instantaneous power draw in watts for a given room."""
        cfg = ROOM_CONFIGS[room]
        levels = self.device_levels[room]
        watts = 0.0
        if cfg["has_light"]:
            watts += cfg["lights"]["count"] * cfg["lights"]["wattage"] * levels["light"]
        if cfg["has_fan"]:
            watts += cfg["fans"]["count"] * cfg["fans"]["wattage"] * levels["fan"]
        if cfg["has_ac"]:
            watts += cfg["ac"]["count"] * cfg["ac"]["wattage"] * levels["ac"]
        if cfg["computers"] is not None and self.occupancy[room] > 0.05:
            watts += cfg["computers"]["count"] * cfg["computers"]["idle_wattage"]
        return watts

    def __repr__(self) -> str:
        return (
            f"CollegeFloorEnv(alpha={self.alpha}, beta={self.beta}, "
            f"gamma={self.gamma}, cost=₹{self.cost_per_kwh}/kWh)"
        )
