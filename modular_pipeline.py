"""Command-line orchestrator for the Modular SDC pipeline."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0] / "src"))

from configuration import AppConfig, DEFAULT_CONFIG_PATH, TargetSpeedConfig
from lane_detection import LaneDetection
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
from sdc_wrapper import SDC_Wrapper
from waypoint_prediction import target_speed_prediction, waypoint_prediction


def _action_contract(action: Sequence[float], action_space) -> np.ndarray:
    """Clip an action vector to the environment bounds."""

    act = np.asarray(action, dtype=np.float32).reshape(3)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return np.clip(act, low, high).astype(action_space.dtype, copy=False)


def _reset_with_speed(env, **kwargs):
    obs, info = env.reset(**kwargs)
    speed = float(info.get("speed", 0.0)) if isinstance(info, dict) else 0.0
    return obs, speed


@dataclass
class ModularPipeline:
    """High-level orchestrator for the perception-planning-control loop."""

    env: gym.Env
    num_waypoints: int = 6
    num_waypoints_speed: int = 4
    max_steps: int = 600
    timestep_seconds: float = 1.0 / 50.0
    waypoint_type: str = "smooth"
    waypoint_smoothing_beta: float = 30.0
    target_speed_settings: TargetSpeedConfig = field(default_factory=TargetSpeedConfig)
    lane_detection_factory: Callable[[], LaneDetection] = LaneDetection
    lateral_controller_factory: Callable[[], LateralController] = LateralController
    longitudinal_controller_factory: Callable[[], LongitudinalController] = LongitudinalController

    def __post_init__(self) -> None:
        self._lane_detector = self.lane_detection_factory()
        self._lateral_controller = self.lateral_controller_factory()
        self._longitudinal_controller = self.longitudinal_controller_factory()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _reset_modules(self) -> None:
        self._lane_detector = self.lane_detection_factory()

        if hasattr(self._lateral_controller, "reset"):
            self._lateral_controller.reset()
        if hasattr(self._longitudinal_controller, "reset"):
            self._longitudinal_controller.reset()

    def run_episode(self, seed: Optional[int] = None) -> float:
        """Execute one episode and return the accumulated reward."""

        try:
            obs, speed = _reset_with_speed(self.env, seed=seed)
        except Exception:  # pragma: no cover - environment initialisation errors
            print("Please note that you can't use the window on the cluster")
            raise

        self._reset_modules()

        total_reward = 0.0
        for step in range(self.max_steps):
            lane_left, lane_right = self._lane_detector.lane_detection(obs)
            waypoints = waypoint_prediction(
                lane_left,
                lane_right,
                num_waypoints=self.num_waypoints,
                way_type=self.waypoint_type,
                smoothing_beta=self.waypoint_smoothing_beta,
            )
            target_speed = target_speed_prediction(
                waypoints,
                num_waypoints_used=self.target_speed_settings.num_waypoints_used,
                max_speed=self.target_speed_settings.max_speed,
                min_speed=self.target_speed_settings.min_speed,
                K_v=self.target_speed_settings.curvature_gain,
            )

            steer = self._lateral_controller.stanley(waypoints, speed)
            gas, brake = self._longitudinal_controller.control(
                speed,
                target_speed,
                dt=self.timestep_seconds,
            )

            action = _action_contract([steer, gas, brake], self.env.action_space)
            obs, reward, terminated, truncated, info = self.env.step(action)

            if isinstance(info, dict) and "speed" in info:
                speed = float(info["speed"])

            total_reward += float(reward)

            if terminated or truncated:
                break

        return float(total_reward)


def evaluate(pipeline: ModularPipeline, episodes: int = 5) -> None:
    for episode in range(episodes):
        reward = pipeline.run_episode()
        print(f"episode {episode}\t reward {reward:.6f}")


def calculate_score_for_leaderboard(pipeline: ModularPipeline, seeds: Sequence[int]) -> None:
    # DO NOT CHANGE THE EVALUATION PROTOCOL
    total_reward = 0.0
    for episode, seed in enumerate(seeds):
        reward = pipeline.run_episode(seed=seed)
        print(f"episode {episode}\t reward {reward:.6f}")
        total_reward += float(np.clip(reward, 0.0, np.inf))

    print("---------------------------")
    print(" total score: %f" % (total_reward / len(seeds)))
    print("---------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="evaluate for the leaderboard")
    parser.add_argument("--no_display", action="store_true", default=False, help="headless mode")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML configuration file (defaults to config.yml)",
    )
    args = parser.parse_args()

    config = AppConfig.from_file(args.config)

    env_kwargs = dict(config.environment.kwargs)
    if args.no_display:
        env_kwargs["render_mode"] = "rgb_array"

    base_env = gym.make(config.environment.id, **env_kwargs)
    env = SDC_Wrapper(
        base_env,
        **config.environment.wrapper.to_kwargs(),
    )

    target_speed_settings = TargetSpeedConfig(
        num_waypoints_used=config.planning.target_speed.num_waypoints_used,
        max_speed=config.planning.target_speed.max_speed,
        min_speed=config.planning.target_speed.min_speed,
        curvature_gain=config.planning.target_speed.curvature_gain,
    )

    pipeline = ModularPipeline(
        env=env,
        num_waypoints=config.planning.waypoints.num_waypoints,
        num_waypoints_speed=config.planning.target_speed.num_waypoints_used,
        max_steps=config.runtime.max_steps,
        timestep_seconds=config.runtime.timestep_seconds,
        waypoint_type=config.planning.waypoints.way_type,
        waypoint_smoothing_beta=config.planning.waypoints.smoothing_beta,
        target_speed_settings=target_speed_settings,
        lane_detection_factory=lambda: LaneDetection(
            **config.perception.lane_detection.to_kwargs()
        ),
        lateral_controller_factory=lambda: LateralController(
            **config.control.lateral.to_kwargs()
        ),
        longitudinal_controller_factory=lambda: LongitudinalController(
            **config.control.longitudinal.to_kwargs()
        ),
    )

    try:
        if args.score:
            calculate_score_for_leaderboard(pipeline, config.evaluation.score_seeds)
        else:
            evaluate(pipeline, config.evaluation.episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()
