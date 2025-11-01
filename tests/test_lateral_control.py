"""Lateral controller test harness with live dashboard support."""

from __future__ import annotations

import argparse
import logging

import numpy as np

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from lane_detection import LaneDetection
from lateral_control import LateralController
from waypoint_prediction import target_speed_prediction, waypoint_prediction

try:  # pragma: no cover
    from .dashboard_utils import (
        create_dashboard,
        create_env,
        extract_speed,
        load_config,
        reset_dashboard,
        update_dashboard,
    )
except ImportError:  # pragma: no cover
    from dashboard_utils import (  # type: ignore
        create_dashboard,
        create_env,
        extract_speed,
        load_config,
        reset_dashboard,
        update_dashboard,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_episode(*, render_mode: str, throttle: float, show_dashboard: bool, max_steps: int) -> None:
    config = load_config()
    env = create_env(config, render_mode=render_mode)

    lane_detector = LaneDetection(**config.perception.lane_detection.to_kwargs())
    lateral = LateralController(**config.control.lateral.to_kwargs())

    dashboard = create_dashboard(config, enabled=show_dashboard)

    observation, info = env.reset()
    reset_dashboard(dashboard, observation, info)

    action = np.zeros(3, dtype=np.float32)
    action[1] = throttle

    timestep = config.runtime.timestep_seconds

    total_reward = 0.0
    for step in range(max_steps):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        left_lane, right_lane = lane_detector.lane_detection(observation)
        waypoints = waypoint_prediction(left_lane, right_lane)
        target_speed = target_speed_prediction(waypoints)

        speed = extract_speed(info)
        action[0] = lateral.stanley(waypoints, speed)

        update_dashboard(
            dashboard,
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            step_index=step,
            speed=speed,
            timestep=timestep,
            lanes=(left_lane, right_lane),
            waypoints=waypoints,
            target_speed=target_speed,
        )

        if step % 10 == 0 or terminated or truncated:
            logging.info(
                "step %03d steer=%+.2f speed=%+.2f target=%+.2f reward=%+.2f",
                step,
                action[0],
                speed,
                target_speed,
                reward,
            )

        env.render()
        if terminated or truncated:
            break

    logging.info("Episode finished at step %d with total reward %.2f", step, total_reward)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--render_mode", default="human")
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    run_episode(
        render_mode=args.render_mode,
        throttle=args.throttle,
        show_dashboard=not args.no_dashboard,
        max_steps=args.max_steps,
    )
