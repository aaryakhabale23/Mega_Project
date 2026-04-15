"""
Baseline policies for comparison against the PPO agent.

Provides:
    - RandomPolicy  : samples uniformly random actions each step
    - RuleBasedPolicy : deterministic rule based on current occupancy thresholds
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from rl_environment.env import (
    ACTION_MAP,
    LEVEL_FRACTION,
    NUM_ACTIONS,
    ROOM_CONFIGS,
    ROOM_ORDER,
    CollegeFloorEnv,
)

SEED = 42


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomPolicy:
    """
    Uniformly random action selection from the environment action space.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, **kwargs) -> tuple[np.ndarray, None]:
        """
        Return a random action compatible with CollegeFloorEnv.

        Returns (action, None) to match Stable-Baselines3 API.
        """
        action = self.rng.integers(0, 3, size=NUM_ACTIONS)
        return action, None

    def __repr__(self) -> str:
        return "RandomPolicy()"


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

class RuleBasedPolicy:
    """
    Deterministic rule-based policy.

    For each room:
        - normalised occupancy > 0.6  → all devices at level 2 (full)
        - 0.3 < occupancy ≤ 0.6      → all devices at level 1 (half/economy)
        - occupancy ≤ 0.3             → all devices at level 0 (off)

    Corridor is always at minimum lighting (hardcoded in the environment).
    """

    def predict(self, obs: np.ndarray, **kwargs) -> tuple[np.ndarray, None]:
        """
        Select an action purely from the observation's per-room occupancy.

        Parameters
        ----------
        obs : np.ndarray (27,)
            Current observation vector from CollegeFloorEnv.

        Returns
        -------
        (action, None)
        """
        action = np.zeros(NUM_ACTIONS, dtype=np.int64)

        # Extract per-room occupancy from the observation vector.
        # Layout: for each room i in [0..4], occupancy is at obs[i*5].
        for act_idx, (room_id, device_type) in enumerate(ACTION_MAP):
            room_pos = ROOM_ORDER.index(room_id)
            occ = obs[room_pos * 5]  # first feature per room = occupancy

            if occ > 0.6:
                action[act_idx] = 2
            elif occ > 0.3:
                action[act_idx] = 1
            else:
                action[act_idx] = 0

        return action, None

    def __repr__(self) -> str:
        return "RuleBasedPolicy()"


# ---------------------------------------------------------------------------
# Always-on baseline (for energy comparison only; not a policy per se)
# ---------------------------------------------------------------------------

class AlwaysOnPolicy:
    """Every controllable device at maximum level at every step."""

    def predict(self, obs: np.ndarray, **kwargs) -> tuple[np.ndarray, None]:
        action = np.full(NUM_ACTIONS, 2, dtype=np.int64)
        return action, None

    def __repr__(self) -> str:
        return "AlwaysOnPolicy()"
