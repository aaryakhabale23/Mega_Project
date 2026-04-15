"""
PPO training script for CollegeFloorEnv.

Uses Stable-Baselines3 PPO with MlpPolicy.

Usage
-----
    python rl_training/train_ppo.py [--timesteps 200000] [--beta 0.6]
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_environment.env import CollegeFloorEnv

SEED = 42


# ---------------------------------------------------------------------------
# Custom logging callback
# ---------------------------------------------------------------------------

class RewardLoggerCallback(BaseCallback):
    """
    Logs mean episode reward every ``log_interval`` steps to a list
    and prints it to stdout.

    Parameters
    ----------
    log_interval : int
        Frequency (in steps) at which to log.
    verbose : int
        Verbosity level.
    """

    def __init__(self, log_interval: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: list[float] = []
        self.step_rewards: list[tuple[int, float]] = []  # (timestep, mean_reward)

    def _on_step(self) -> bool:
        # Collect episode rewards from the Monitor wrapper
        if len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            mean_reward = np.mean(recent_rewards)
            self.episode_rewards.append(mean_reward)

            if self.num_timesteps % self.log_interval < self.model.n_envs:
                self.step_rewards.append((self.num_timesteps, mean_reward))
                if self.verbose:
                    print(
                        f"  Step {self.num_timesteps:>8d} | "
                        f"Mean Ep Reward: {mean_reward:.3f} | "
                        f"Episodes: {len(self.model.ep_info_buffer)}"
                    )
        return True


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_ppo(
    total_timesteps: int = 200_000,
    beta: float = 0.6,
    alpha: float = 1.0,
    gamma_env: float = 0.1,
    save_path: str = "ppo_college_floor",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    seed: int = SEED,
    log_interval: int = 2048,
) -> tuple[PPO, RewardLoggerCallback]:
    """
    Train a PPO agent on CollegeFloorEnv.

    Parameters
    ----------
    total_timesteps : int
        Total environment steps to train for.
    beta : float
        Energy-cost weight in the reward function.
    save_path : str
        Path to save the trained model (without extension).

    Returns
    -------
    (model, callback)
    """
    # Create environment
    env = CollegeFloorEnv(alpha=alpha, beta=beta, gamma=gamma_env, seed=seed)
    env = Monitor(env)

    print(f"\n{'='*60}")
    print(f"  PPO Training — {total_timesteps:,} timesteps")
    print(f"  alpha={alpha}, beta={beta}, gamma={gamma_env}")
    print(f"  lr={learning_rate}, n_steps={n_steps}, batch={batch_size}")
    print(f"{'='*60}\n")

    # PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=seed,
        verbose=0,
    )

    callback = RewardLoggerCallback(log_interval=log_interval)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    # Save reward log
    log_path = Path(save_path).parent / f"{Path(save_path).stem}_reward_log.npy"
    np.save(str(log_path), np.array(callback.step_rewards))
    print(f"Reward log saved to {log_path}")

    return model, callback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on CollegeFloorEnv.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma_env", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="ppo_college_floor")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        beta=args.beta,
        alpha=args.alpha,
        gamma_env=args.gamma_env,
        save_path=args.save_path,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
