"""
demo.py — Full integration demo for professor presentation.

Ties together:
    1. Trained PPO model (rl agent)
    2. CollegeFloorEnv (simulation / camera-override)
    3. DL pipeline (YOLOv8 + density head + MOG2)
    4. Real-time floor dashboard

Usage
-----
    # Webcam:
    python demo.py

    # Video file:
    python demo.py --source path/to/video.mp4

    # Simulation only (no camera):
    python demo.py --simulate

Controls
--------
    Press Ctrl+C to stop and print summary.
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import sys
import time
from pathlib import Path

SEED = 42
np.random.seed(SEED)


def run_demo(
    source: str | int = 0,
    model_path: str = "ppo_college_floor.zip",
    density_weights: str = "density_head.pth",
    simulate: bool = False,
    num_zones: int = 3,
    target_fps: float = 2.0,
) -> None:
    """
    Run the live integration demo.

    Parameters
    ----------
    source : str or int
        Webcam index (0) or path to a video file.
    model_path : str
        Path to the trained PPO model (.zip).
    density_weights : str
        Path to trained density_head.pth.
    simulate : bool
        If True, run purely on simulated occupancy (no camera needed).
    num_zones : int
        Number of horizontal zones for the DL pipeline.
    target_fps : float
        Target frames per second for the demo loop.
    """
    # ------------------------------------------------------------------
    # 1. Load PPO model
    # ------------------------------------------------------------------
    from stable_baselines3 import PPO
    from rl_environment.env import CollegeFloorEnv

    print("Loading PPO model ...")
    try:
        ppo_model = PPO.load(model_path)
        print(f"  [OK] Loaded from {model_path}")
    except FileNotFoundError:
        print(f"  [!!] Model not found at {model_path}, using rule-based fallback.")
        ppo_model = None

    # ------------------------------------------------------------------
    # 2. Initialise environment
    # ------------------------------------------------------------------
    env = CollegeFloorEnv(seed=SEED)
    obs, _ = env.reset()
    always_on_kwh = env.get_always_on_energy()
    print(f"  [OK] Environment ready (always-on baseline: {always_on_kwh:.2f} kWh)")

    # ------------------------------------------------------------------
    # 3. Initialise DL pipeline (if using camera)
    # ------------------------------------------------------------------
    pipeline = None
    cap = None

    if not simulate:
        try:
            from dl_pipeline.model import OccupancyPipeline

            weights_path = Path(density_weights) if Path(density_weights).exists() else None
            device = "cuda" if _cuda_available() else "cpu"

            pipeline = OccupancyPipeline(
                density_head_weights=weights_path,
                num_zones=num_zones,
                device=device,
            )
            print(f"  [OK] DL pipeline loaded (device={device})")
        except Exception as e:
            print(f"  [!!] DL pipeline failed to load: {e}")
            print("    Falling back to simulation mode.")
            simulate = True

    if not simulate:
        cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
        if not cap.isOpened():
            print(f"  [!!] Cannot open video source: {source}")
            print("    Falling back to simulation mode.")
            simulate = True
        else:
            print(f"  [OK] Video source opened: {source}")

    # ------------------------------------------------------------------
    # 4. Initialise dashboard
    # ------------------------------------------------------------------
    from visualization.dashboard import FloorDashboard

    dashboard = FloorDashboard(env, always_on_kwh=always_on_kwh)
    print("  [OK] Dashboard initialised\n")

    # Fallback policy if no PPO model
    if ppo_model is None:
        from rl_training.baselines import RuleBasedPolicy
        fallback_policy = RuleBasedPolicy()

    # ------------------------------------------------------------------
    # 5. Demo loop
    # ------------------------------------------------------------------
    frame_interval = 1.0 / target_fps
    start_time = time.time()
    step_count = 0

    print("=" * 60)
    print("  DEMO RUNNING  |  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            loop_start = time.time()

            # --- Read frame (if camera mode) ---
            if not simulate and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    # Video ended -- restart or switch to simulation
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print("Video source exhausted. Switching to simulation.")
                        simulate = True
                        frame = None
                else:
                    # Run DL pipeline
                    density_map, zone_occ, zone_activity = pipeline(frame)

                    # Normalise zone occupancy to [0, 1] range
                    # (raw sums can be large; scale by room capacity proxy)
                    max_expected = density_map.shape[0] * density_map.shape[1] / num_zones * 0.5
                    zone_occ_norm = np.clip(zone_occ / max(max_expected, 1.0), 0.0, 1.0)
                    zone_act_norm = zone_activity / 2.0  # {0,1,2} -> {0, 0.5, 1}

                    # Inject into environment
                    CollegeFloorEnv.from_dl_output(env, zone_occ_norm, zone_act_norm)

                    # Show camera feed with overlay
                    from dl_pipeline.model import visualize_density
                    overlay = visualize_density(frame, density_map, zone_occ, num_zones)
                    cv2.imshow("Camera - Density Overlay", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # --- Get action ---
            if ppo_model is not None:
                action, _ = ppo_model.predict(obs, deterministic=True)
            else:
                action, _ = fallback_policy.predict(obs)

            # --- Step environment ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Update dashboard ---
            dashboard.update(info)

            step_count += 1

            # --- Check episode end ---
            if terminated or truncated:
                print(f"\n  Episode complete at step {step_count}")
                obs, _ = env.reset()
                step_count = 0

            # --- Frame rate control ---
            elapsed = time.time() - loop_start
            sleep_time = max(frame_interval - elapsed, 0)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    total_runtime = time.time() - start_time

    print("\n")
    print("=" * 60)
    print("  DEMO SUMMARY")
    print("=" * 60)
    print(f"  Total Runtime     : {total_runtime:.1f} seconds")
    print(f"  Steps Completed   : {step_count}")
    print(f"  Total Energy      : {env.total_energy_kwh:.3f} kWh")
    print(f"  Total Cost        : Rs {env.total_cost_rupees:.2f}")
    print(f"  Always-On Baseline: {always_on_kwh:.3f} kWh (Rs {always_on_kwh * env.cost_per_kwh:.2f})")
    saving = always_on_kwh - env.total_energy_kwh
    saving_pct = (saving / max(always_on_kwh, 1e-9)) * 100
    print(f"  Estimated Saving  : {saving:.3f} kWh ({saving_pct:.1f}%)")
    print(f"                      Rs {saving * env.cost_per_kwh:.2f}")
    print("=" * 60)

    # Cleanup
    dashboard.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def _cuda_available() -> bool:
    """Check if CUDA is available without importing torch at module level."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart Environment Energy Optimizer - Live Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Webcam index (default: 0) or path to video file.",
    )
    parser.add_argument(
        "--model", type=str, default="ppo_college_floor.zip",
        help="Path to trained PPO model.",
    )
    parser.add_argument(
        "--density_weights", type=str, default="density_head.pth",
        help="Path to trained density head weights.",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in simulation-only mode (no camera).",
    )
    parser.add_argument(
        "--num_zones", type=int, default=3,
        help="Number of horizontal zones.",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Target frames per second.",
    )
    args = parser.parse_args()

    # Parse source — int for webcam, str for file
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run_demo(
        source=source,
        model_path=args.model,
        density_weights=args.density_weights,
        simulate=args.simulate,
        num_zones=args.num_zones,
        target_fps=args.fps,
    )


if __name__ == "__main__":
    main()
