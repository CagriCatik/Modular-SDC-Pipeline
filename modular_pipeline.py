"""Command-line orchestrator for the Modular SDC pipeline."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0] / "src"))

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
            )
            target_speed = target_speed_prediction(
                waypoints,
                num_waypoints_used=self.num_waypoints_speed,
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


def evaluate(env, episodes: int = 5) -> None:
    pipeline = ModularPipeline(env)
    for episode in range(episodes):
        reward = pipeline.run_episode()
        print(f"episode {episode}\t reward {reward:.6f}")


def calculate_score_for_leaderboard(env) -> None:
    # DO NOT CHANGE
    seeds = [97657630, 47460391, 22619914, 76925063, 84647422,
             83470445, 77482096, 94017676, 99341122, 58134947]

    pipeline = ModularPipeline(env)

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
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_display else "human"

    # Keep v2 to match the original wrapper usage
    env = SDC_Wrapper(
        gym.make("CarRacing-v3", render_mode=render_mode),
        remove_score=True,
        return_linear_velocity=True,
    )

    try:
        if args.score:
            calculate_score_for_leaderboard(env)
        else:
            evaluate(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
