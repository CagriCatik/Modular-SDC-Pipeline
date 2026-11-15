"""CLI for tuning Modular-SDC-Pipeline parameters with Particle Swarm Optimization (PSO)."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0] / "src"))

from configuration import AppConfig, DEFAULT_CONFIG_PATH
from pso_tuner import (
    DEFAULT_CONTROL_PLANNING_SPECS,
    DEFAULT_MPC_CONTROL_PLANNING_SPECS,
    DEFAULT_PERCEPTION_SPECS,
    PSOConfig,
    evaluate_config,
    optimize_pipeline_with_pso,
    save_config_yaml,
    save_pso_dataset_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize pipeline parameters using PSO.")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG_PATH, help="Base YAML config")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("configs/pso_best.yaml"),
        help="Output YAML path for the best parameters",
    )
    parser.add_argument(
        "--dataset-output",
        type=pathlib.Path,
        default=pathlib.Path("configs/pso_dataset.json"),
        help="Output JSON path for the evaluated parameter dataset",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Also write the optimized parameters back to --config",
    )
    parser.add_argument("--particles", type=int, default=12, help="Number of PSO particles")
    parser.add_argument("--iterations", type=int, default=12, help="Number of PSO iterations")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--max-steps", type=int, default=None, help="Override runtime.max_steps during tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Evaluation seeds")
    parser.add_argument(
        "--include-perception",
        action="store_true",
        help="Also tune lane detection parameters (spline_smoothness, gradient_threshold)",
    )
    parser.add_argument(
        "--controller",
        choices=("stanley", "mpc"),
        default=None,
        help="Override lateral controller type during tuning (defaults to config.yml)",
    )
    parser.add_argument("--seed", type=int, default=42, help="PSO random seed")
    args = parser.parse_args()

    base_config = AppConfig.from_file(args.config)
    if args.controller is not None:
        base_config.control.lateral.controller_type = args.controller
    seeds: List[int] = list(args.seeds) if args.seeds else list(base_config.evaluation.score_seeds[:3])

    specs = list(
        DEFAULT_MPC_CONTROL_PLANNING_SPECS
        if base_config.control.lateral.controller_type == "mpc"
        else DEFAULT_CONTROL_PLANNING_SPECS
    )
    if args.include_perception:
        specs.extend(DEFAULT_PERCEPTION_SPECS)

    baseline_score = evaluate_config(base_config, seeds=seeds, max_steps=args.max_steps)
    print(f"Baseline mean reward ({len(seeds)} seeds): {baseline_score:.2f}")

    pso_cfg = PSOConfig(
        num_particles=args.particles,
        iterations=args.iterations,
        seed=args.seed,
    )

    def _on_iter(iteration: int, mean_score: float, best_score: float) -> None:
        print(f"iter {iteration + 1:02d}/{args.iterations:02d} | mean={mean_score:8.2f} | best={best_score:8.2f}")

    result, best_config = optimize_pipeline_with_pso(
        base_config,
        specs,
        seeds=seeds,
        max_steps=args.max_steps,
        pso_config=pso_cfg,
        workers=args.workers,
        progress_callback=_on_iter,
    )

    save_config_yaml(best_config, args.output)
    save_pso_dataset_json(result, args.dataset_output)
    if args.apply:
        save_config_yaml(best_config, args.config)

    print("\nBest parameters:")
    for key, val in result.best_params.items():
        print(f"  {key}: {val}")
    print(f"\nBest mean reward: {result.best_score:.2f} (baseline: {baseline_score:.2f})")
    print(f"Saved optimized configuration to: {args.output}")
    print(f"Saved parameter dataset ({len(result.evaluations)} samples) to: {args.dataset_output}")
    if args.apply:
        print(f"Updated base configuration: {args.config}")


if __name__ == "__main__":
    main()
