"""
RL evaluation — run any policy (PPO, random, rule-based) for N episodes
and report summary statistics.

Usage
-----
    python rl_training/evaluate_rl.py [--model ppo_college_floor.zip] [--episodes 50]
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

from rl_environment.env import CollegeFloorEnv
from rl_training.baselines import RandomPolicy, RuleBasedPolicy, AlwaysOnPolicy

SEED = 42


def evaluate_policy(
    policy,
    env: CollegeFloorEnv | None = None,
    n_episodes: int = 50,
    beta: float = 0.6,
    seed: int = SEED,
) -> dict:
    """
    Run *policy* for *n_episodes* on CollegeFloorEnv and collect metrics.

    Parameters
    ----------
    policy
        Anything with a `predict(obs)` → (action, _) method.
    env : CollegeFloorEnv, optional
        If None a fresh env is created.
    n_episodes : int
        Number of evaluation episodes.
    beta : float
        Beta for the environment (used if creating a new env).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        total_energy_kwh  : (mean, std)
        total_cost_rupees : (mean, std)
        mean_comfort      : (mean, std)
        mean_reward       : (mean, std)
        raw               : list[dict] per-episode data
    """
    if env is None:
        env = CollegeFloorEnv(beta=beta, seed=seed)

    ep_energy = []
    ep_cost = []
    ep_comfort = []
    ep_reward = []
    raw_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        comfort_scores = []

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            comfort_scores.append(info["mean_comfort"])

        ep_energy.append(info["cumulative_energy_kwh"])
        ep_cost.append(info["cumulative_cost_rupees"])
        ep_comfort.append(np.mean(comfort_scores))
        ep_reward.append(total_reward)

        raw_episodes.append({
            "episode": ep,
            "energy_kwh": info["cumulative_energy_kwh"],
            "cost_rupees": info["cumulative_cost_rupees"],
            "mean_comfort": float(np.mean(comfort_scores)),
            "total_reward": total_reward,
        })

    results = {
        "total_energy_kwh": (float(np.mean(ep_energy)), float(np.std(ep_energy))),
        "total_cost_rupees": (float(np.mean(ep_cost)), float(np.std(ep_cost))),
        "mean_comfort": (float(np.mean(ep_comfort)), float(np.std(ep_comfort))),
        "mean_reward": (float(np.mean(ep_reward)), float(np.std(ep_reward))),
        "raw": raw_episodes,
    }

    return results


def print_results(name: str, results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Energy    : {results['total_energy_kwh'][0]:8.2f} +/- {results['total_energy_kwh'][1]:.2f} kWh")
    print(f"  Cost      : Rs{results['total_cost_rupees'][0]:7.2f} +/- {results['total_cost_rupees'][1]:.2f}")
    print(f"  Comfort   : {results['mean_comfort'][0]:8.4f} +/- {results['mean_comfort'][1]:.4f}")
    print(f"  Reward    : {results['mean_reward'][0]:8.3f} +/- {results['mean_reward'][1]:.3f}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL policies on CollegeFloorEnv.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to PPO model .zip (omit extension).")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compare", action="store_true",
                        help="Run all baselines for comparison.")
    args = parser.parse_args()

    env = CollegeFloorEnv(beta=args.beta, seed=args.seed)

    # Always-on baseline
    if args.compare:
        for name, policy in [
            ("Always-On", AlwaysOnPolicy()),
            ("Random", RandomPolicy(seed=args.seed)),
            ("Rule-Based", RuleBasedPolicy()),
        ]:
            res = evaluate_policy(policy, env=env, n_episodes=args.episodes, seed=args.seed)
            print_results(name, res)

    # PPO agent
    if args.model:
        ppo = PPO.load(args.model)
        res = evaluate_policy(ppo, env=env, n_episodes=args.episodes, seed=args.seed)
        print_results(f"PPO ({args.model})", res)


if __name__ == "__main__":
    main()
