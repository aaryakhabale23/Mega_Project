"""
Generate all static plots from existing training and evaluation data.

Run this after PPO training and baseline evaluation to produce
publication-ready figures for the project report.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rl_environment.env import CollegeFloorEnv
from rl_training.baselines import RandomPolicy, RuleBasedPolicy, AlwaysOnPolicy
from rl_training.evaluate_rl import evaluate_policy
from visualization.plots import (
    plot_learning_curve,
    plot_energy_comparison,
    plot_room_behavior,
    collect_episode_data,
)
from stable_baselines3 import PPO

SEED = 42
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  Generating Static Plots")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Evaluate baselines (quick, for chart data)
    # ------------------------------------------------------------------
    print("\n[1/4] Evaluating baselines for chart data ...")
    env = CollegeFloorEnv(seed=SEED)

    random_res = evaluate_policy(RandomPolicy(seed=SEED), env=env, n_episodes=30, seed=SEED)
    rule_res = evaluate_policy(RuleBasedPolicy(), env=env, n_episodes=30, seed=SEED)
    always_on_res = evaluate_policy(AlwaysOnPolicy(), env=env, n_episodes=30, seed=SEED)

    ppo_model = PPO.load("ppo_college_floor")
    ppo_res = evaluate_policy(ppo_model, env=env, n_episodes=30, seed=SEED)

    # ------------------------------------------------------------------
    # 2. Learning Curve
    # ------------------------------------------------------------------
    print("\n[2/4] Plotting learning curve ...")
    reward_log = Path("ppo_college_floor_reward_log.npy")
    if reward_log.exists():
        plot_learning_curve(
            reward_log_path=reward_log,
            random_reward=random_res["mean_reward"][0],
            rule_based_reward=rule_res["mean_reward"][0],
            save_path=PLOTS_DIR / "learning_curve.png",
            window=10,
        )
    else:
        print("  WARNING: reward log not found, skipping learning curve.")

    # ------------------------------------------------------------------
    # 3. Energy Comparison Bar Chart
    # ------------------------------------------------------------------
    print("\n[3/4] Plotting energy comparison ...")
    energy_data = {
        "Always-On": always_on_res["total_energy_kwh"][0],
        "Rule-Based": rule_res["total_energy_kwh"][0],
        "PPO Agent": ppo_res["total_energy_kwh"][0],
    }
    plot_energy_comparison(
        energy_data=energy_data,
        save_path=PLOTS_DIR / "energy_comparison.png",
    )

    # ------------------------------------------------------------------
    # 4. Room Behavior Time-Series
    # ------------------------------------------------------------------
    print("\n[4/4] Plotting room behavior ...")
    env2 = CollegeFloorEnv(seed=SEED)
    episode_data = collect_episode_data(env2, ppo_model)
    plot_room_behavior(
        episode_data=episode_data,
        save_path=PLOTS_DIR / "room_behavior.png",
    )

    print("\n" + "=" * 60)
    print(f"  All plots saved to {PLOTS_DIR.resolve()}/")
    print("=" * 60)
    print("  - learning_curve.png")
    print("  - energy_comparison.png")
    print("  - room_behavior.png")
    print("  (comfort_energy_tradeoff.png will be generated after beta sweep)")
    print("=" * 60)


if __name__ == "__main__":
    main()
