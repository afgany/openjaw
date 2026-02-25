"""Step 18: Run Ablation Experiments.

Runs all 4 standard ablation conditions:
  1. combined:      w_a > 0, w_v > 0 (full model)
  2. audio_only:    w_a > 0, w_v = 0
  3. visual_only:   w_a = 0, w_v > 0
  4. no_curriculum:  combined reward, no curriculum phasing

Usage:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --num-episodes 500 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from openjaw.evaluation.ablations import STANDARD_ABLATIONS, run_all_ablations
from openjaw.evaluation.visualization import plot_ablation_comparison, plot_reward_curves
from openjaw.training.trainer import TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Episodes per condition (use 5000+ for paper results)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-real-sparc", action="store_true")
    parser.add_argument("--use-real-sylber", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs/ablations/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ablations/")
    parser.add_argument("--output-dir", type=str, default="results/ablations/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainerConfig(
        num_envs=1,
        episode_length=50,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name="ablation",
        use_real_sparc=args.use_real_sparc,
        use_real_sylber=args.use_real_sylber,
        device=args.device,
    )

    logger.info(f"Running {len(STANDARD_ABLATIONS)} ablation conditions, {args.num_episodes} episodes each")
    start_time = time.time()

    results = run_all_ablations(config, num_episodes=args.num_episodes)

    elapsed = time.time() - start_time
    logger.info(f"\nAll ablations complete in {elapsed / 3600:.2f} hours")

    # Save raw results
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "rewards"}
        sr["rewards"] = [float(x) for x in r["rewards"]]
        serializable.append(sr)

    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Condition':<20} {'Mean Reward':>12} {'Final Reward':>13} {'Max Reward':>11}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['condition']:<20} {r['mean_reward']:>12.4f} "
            f"{r['final_reward']:>13.4f} {r['max_reward']:>11.4f}"
        )
    print("=" * 70)

    # Generate figures
    fig_dir = str(output_dir / "figures")

    # Reward curves
    reward_data = {r["condition"]: r["rewards"] for r in results}
    plot_reward_curves(
        reward_data,
        output_path=f"{fig_dir}/ablation_reward_curves.pdf",
        title="Ablation Study: Reward Curves",
        window=max(1, args.num_episodes // 20),
    )

    # Bar chart
    plot_ablation_comparison(
        results,
        output_path=f"{fig_dir}/ablation_comparison.pdf",
    )

    logger.info(f"Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
