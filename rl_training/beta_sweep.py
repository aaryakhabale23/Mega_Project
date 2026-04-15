"""
Beta sweep — train PPO agents at beta = 0.3, 0.6, 0.9 and compare.

Trains each agent for 100 000 timesteps, evaluates over 50 episodes,
and returns a consolidated results dict suitable for the comfort-energy
tradeoff scatter plot.

Usage
-----
    python rl_training/beta_sweep.py [--timesteps 100000] [--episodes 50]
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

from rl_training.train_ppo import train_ppo
from rl_training.evaluate_rl import evaluate_policy, print_results
from rl_environment.env import CollegeFloorEnv

SEED = 42
BETA_VALUES = [0.3, 0.6, 0.9]


def beta_sweep(
    betas: list[float] | None = None,
    timesteps: int = 100_000,
    n_episodes: int = 50,
    seed: int = SEED,
    output_dir: Path = Path("beta_sweep_results"),
) -> dict:
    """
    Train and evaluate PPO agents at multiple beta values.

    Parameters
    ----------
    betas : list of float
        Energy-cost weights to sweep over.
    timesteps : int
        Training timesteps per agent.
    n_episodes : int
        Evaluation episodes per agent.
    seed : int
        Random seed.
    output_dir : Path
        Directory to save models and results.

    Returns
    -------
    dict mapping beta → evaluation results dict.
    """
    if betas is None:
        betas = BETA_VALUES

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[float, dict] = {}

    for beta in betas:
        print(f"\n{'#'*60}")
        print(f"  Beta Sweep — beta = {beta}")
        print(f"{'#'*60}")

        save_name = str(output_dir / f"ppo_beta_{beta}")

        # Train
        model, callback = train_ppo(
            total_timesteps=timesteps,
            beta=beta,
            save_path=save_name,
            seed=seed,
        )

        # Evaluate
        env = CollegeFloorEnv(beta=beta, seed=seed)
        results = evaluate_policy(model, env=env, n_episodes=n_episodes, seed=seed)
        print_results(f"PPO (beta={beta})", results)

        all_results[beta] = results

    # Save consolidated results
    summary = {}
    for beta, res in all_results.items():
        summary[str(beta)] = {
            "energy_kwh_mean": res["total_energy_kwh"][0],
            "energy_kwh_std": res["total_energy_kwh"][1],
            "cost_mean": res["total_cost_rupees"][0],
            "comfort_mean": res["mean_comfort"][0],
            "comfort_std": res["mean_comfort"][1],
            "reward_mean": res["mean_reward"][0],
        }

    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary saved to {summary_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Beta sweep for PPO on CollegeFloorEnv.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default="beta_sweep_results")
    args = parser.parse_args()

    beta_sweep(
        timesteps=args.timesteps,
        n_episodes=args.episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
