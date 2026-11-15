"""Particle Swarm Optimization (PSO) tuner for Modular-SDC-Pipeline parameters."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import yaml

from configuration import AppConfig
from modular_pipeline import build_pipeline
from sdc_wrapper import SDC_Wrapper


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of a single tunable parameter in :class:`AppConfig`."""

    path: str
    low: float
    high: float
    is_int: bool = False

    def clamp(self, value: float) -> float:
        clamped = float(np.clip(value, self.low, self.high))
        return int(round(clamped)) if self.is_int else clamped


DEFAULT_CONTROL_PLANNING_SPECS: Tuple[ParameterSpec, ...] = (
    ParameterSpec("control.lateral.gain_constant", 0.005, 0.25),
    ParameterSpec("control.lateral.damping_constant", 0.0, 0.9),
    ParameterSpec("control.longitudinal.KP", 0.02, 0.30),
    ParameterSpec("control.longitudinal.KI", 0.0, 0.05),
    ParameterSpec("control.longitudinal.KD", 0.0, 0.01),
    ParameterSpec("control.longitudinal.max_gas", 0.4, 1.0),
    ParameterSpec("control.longitudinal.max_brake", 0.2, 1.0),
    ParameterSpec("planning.waypoints.smoothing_beta", 5.0, 60.0),
    ParameterSpec("planning.target_speed.max_speed", 25.0, 75.0),
    ParameterSpec("planning.target_speed.min_speed", 10.0, 35.0),
    ParameterSpec("planning.target_speed.curvature_gain", 0.5, 6.0),
)

DEFAULT_MPC_CONTROL_PLANNING_SPECS: Tuple[ParameterSpec, ...] = (
    ParameterSpec("control.lateral.wheelbase", 10.0, 60.0),
    ParameterSpec("control.lateral.cte_weight", 0.1, 5.0),
    ParameterSpec("control.lateral.heading_weight", 0.1, 6.0),
    ParameterSpec("control.lateral.steer_weight", 0.0, 2.0),
    ParameterSpec("control.lateral.steer_rate_weight", 0.1, 8.0),
    ParameterSpec("control.longitudinal.KP", 0.02, 0.30),
    ParameterSpec("control.longitudinal.KI", 0.0, 0.05),
    ParameterSpec("control.longitudinal.KD", 0.0, 0.01),
    ParameterSpec("control.longitudinal.max_gas", 0.4, 1.0),
    ParameterSpec("control.longitudinal.max_brake", 0.2, 1.0),
    ParameterSpec("planning.waypoints.smoothing_beta", 5.0, 60.0),
    ParameterSpec("planning.target_speed.max_speed", 25.0, 75.0),
    ParameterSpec("planning.target_speed.min_speed", 10.0, 35.0),
    ParameterSpec("planning.target_speed.curvature_gain", 0.5, 6.0),
)

DEFAULT_PERCEPTION_SPECS: Tuple[ParameterSpec, ...] = (
    ParameterSpec("perception.lane_detection.cut_size", 56, 74, is_int=True),
    ParameterSpec("perception.lane_detection.spline_smoothness", 2.0, 30.0),
    ParameterSpec("perception.lane_detection.gradient_threshold", 8.0, 28.0),
    ParameterSpec("perception.lane_detection.distance_maxima_gradient", 2, 6, is_int=True),
)


def _get_nested(obj: object, dotted_path: str) -> float:
    cur = obj
    for part in dotted_path.split("."):
        cur = getattr(cur, part)
    return float(cur)


def _set_nested(obj: object, dotted_path: str, value: float) -> None:
    parts = dotted_path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def extract_vector(config: AppConfig, specs: Sequence[ParameterSpec]) -> np.ndarray:
    """Extract parameter values from ``config`` in the order of ``specs``."""
    return np.array([_get_nested(config, spec.path) for spec in specs], dtype=float)


def apply_params_to_config(
    base_config: AppConfig,
    specs: Sequence[ParameterSpec],
    vector: Sequence[float],
) -> AppConfig:
    """Return a copy of ``base_config`` with ``vector`` applied to ``specs``."""
    cfg = copy.deepcopy(base_config)
    for spec, raw_val in zip(specs, vector):
        _set_nested(cfg, spec.path, spec.clamp(float(raw_val)))

    # Keep min_speed <= max_speed consistent
    if cfg.planning.target_speed.min_speed > cfg.planning.target_speed.max_speed:
        cfg.planning.target_speed.min_speed = cfg.planning.target_speed.max_speed
    return cfg


def evaluate_config(
    config: AppConfig,
    seeds: Sequence[int],
    max_steps: Optional[int] = None,
) -> float:
    """Run headless CarRacing-v3 episodes for ``seeds`` and return mean reward."""
    cfg = copy.deepcopy(config)
    if max_steps is not None:
        cfg.runtime.max_steps = int(max_steps)

    env_kwargs = dict(cfg.environment.kwargs)
    env_kwargs["render_mode"] = "rgb_array"

    base_env = gym.make(cfg.environment.id, **env_kwargs)
    env = SDC_Wrapper(base_env, **cfg.environment.wrapper.to_kwargs())
    pipeline = build_pipeline(cfg, env, enable_dashboard=False)
    try:
        rewards = [pipeline.run_episode(seed=int(s)) for s in seeds]
    finally:
        pipeline.close()

    return float(np.mean(rewards))


def _eval_worker(args: Tuple[AppConfig, Sequence[ParameterSpec], np.ndarray, Sequence[int], Optional[int]]) -> float:
    base_config, specs, vector, seeds, max_steps = args
    cfg = apply_params_to_config(base_config, specs, vector)
    return evaluate_config(cfg, seeds=seeds, max_steps=max_steps)


def save_config_yaml(config: AppConfig, output_path: Path) -> None:
    """Serialize ``AppConfig`` to a YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["evaluation"]["score_seeds"] = list(config.evaluation.score_seeds)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_pso_dataset_json(result: "PSOResult", output_path: Path) -> None:
    """Save all evaluated PSO parameter samples and iteration history to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_score": result.best_score,
        "best_params": result.best_params,
        "history": result.history,
        "evaluations": result.evaluations,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass
class PSOConfig:
    """Hyperparameters for Particle Swarm Optimization."""

    num_particles: int = 12
    iterations: int = 15
    w_start: float = 0.8
    w_end: float = 0.4
    c1: float = 1.5
    c2: float = 1.5
    velocity_clamp: float = 0.25
    seed: int = 42


@dataclass
class PSOResult:
    """Optimization result returned by :class:`ParticleSwarmOptimizer`."""

    best_vector: np.ndarray
    best_score: float
    best_params: Dict[str, float]
    history: List[Dict[str, float]]
    evaluations: List[Dict[str, Any]] = field(default_factory=list)


class ParticleSwarmOptimizer:
    """Maximize a bounded continuous objective function using PSO."""

    def __init__(
        self,
        specs: Sequence[ParameterSpec],
        pso_config: Optional[PSOConfig] = None,
    ) -> None:
        self.specs = tuple(specs)
        self.pso_config = pso_config or PSOConfig()
        self.lows = np.array([s.low for s in self.specs], dtype=float)
        self.highs = np.array([s.high for s in self.specs], dtype=float)
        self.span = self.highs - self.lows

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        *,
        initial_vector: Optional[np.ndarray] = None,
        batch_evaluator: Optional[Callable[[np.ndarray], List[float]]] = None,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> PSOResult:
        rng = np.random.default_rng(self.pso_config.seed)
        n_particles = self.pso_config.num_particles
        dim = len(self.specs)

        positions = rng.uniform(self.lows, self.highs, size=(n_particles, dim))
        if initial_vector is not None:
            positions[0] = np.clip(np.asarray(initial_vector, dtype=float), self.lows, self.highs)

        max_vel = self.pso_config.velocity_clamp * self.span
        velocities = rng.uniform(-max_vel, max_vel, size=(n_particles, dim)) * 0.5

        pbest_pos = positions.copy()
        pbest_scores = np.full(n_particles, -np.inf, dtype=float)
        gbest_pos = positions[0].copy()
        gbest_score = -np.inf
        history: List[Dict[str, float]] = []
        evaluations: List[Dict[str, Any]] = []

        for iteration in range(self.pso_config.iterations):
            if batch_evaluator is not None:
                scores = np.asarray(batch_evaluator(positions), dtype=float)
            else:
                scores = np.array([objective(p) for p in positions], dtype=float)

            for particle_idx, (pos, score) in enumerate(zip(positions, scores)):
                evaluations.append(
                    {
                        "iteration": int(iteration),
                        "particle": int(particle_idx),
                        "score": float(score),
                        "params": {
                            spec.path: spec.clamp(float(val))
                            for spec, val in zip(self.specs, pos)
                        },
                    }
                )

            improved = scores > pbest_scores
            pbest_scores[improved] = scores[improved]
            pbest_pos[improved] = positions[improved]

            best_idx = int(np.argmax(pbest_scores))
            if pbest_scores[best_idx] > gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest_pos = pbest_pos[best_idx].copy()

            mean_score = float(np.mean(scores))
            history.append(
                {
                    "iteration": float(iteration),
                    "mean_score": mean_score,
                    "best_iter_score": float(np.max(scores)),
                    "global_best_score": gbest_score,
                }
            )
            if progress_callback is not None:
                progress_callback(iteration, mean_score, gbest_score)

            if iteration + 1 >= self.pso_config.iterations:
                break

            frac = iteration / max(self.pso_config.iterations - 1, 1)
            w = self.pso_config.w_start + frac * (self.pso_config.w_end - self.pso_config.w_start)

            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))
            velocities = (
                w * velocities
                + self.pso_config.c1 * r1 * (pbest_pos - positions)
                + self.pso_config.c2 * r2 * (gbest_pos - positions)
            )
            velocities = np.clip(velocities, -max_vel, max_vel)
            positions = np.clip(positions + velocities, self.lows, self.highs)

        best_params = {
            spec.path: spec.clamp(float(val)) for spec, val in zip(self.specs, gbest_pos)
        }
        return PSOResult(
            best_vector=gbest_pos,
            best_score=gbest_score,
            best_params=best_params,
            history=history,
            evaluations=evaluations,
        )


def optimize_pipeline_with_pso(
    base_config: AppConfig,
    specs: Sequence[ParameterSpec],
    *,
    seeds: Sequence[int],
    max_steps: Optional[int] = None,
    pso_config: Optional[PSOConfig] = None,
    workers: int = 1,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Tuple[PSOResult, AppConfig]:
    """Run PSO on the CarRacing-v3 pipeline and return ``(result, best_config)``."""
    optimizer = ParticleSwarmOptimizer(specs, pso_config=pso_config)
    initial_vec = extract_vector(base_config, specs)

    def _single_eval(vec: np.ndarray) -> float:
        return evaluate_config(
            apply_params_to_config(base_config, specs, vec),
            seeds=seeds,
            max_steps=max_steps,
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            def _batch_eval(pop: np.ndarray) -> List[float]:
                tasks = [(base_config, specs, row, seeds, max_steps) for row in pop]
                return list(pool.map(_eval_worker, tasks))

            result = optimizer.optimize(
                _single_eval,
                initial_vector=initial_vec,
                batch_evaluator=_batch_eval,
                progress_callback=progress_callback,
            )
    else:
        result = optimizer.optimize(
            _single_eval,
            initial_vector=initial_vec,
            progress_callback=progress_callback,
        )

    best_config = apply_params_to_config(base_config, specs, result.best_vector)
    return result, best_config
