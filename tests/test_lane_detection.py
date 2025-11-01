"""Interactive lane-detection smoke test with optional live dashboard."""

from __future__ import annotations

import argparse
import logging
from typing import Dict

import numpy as np
import pygame

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lane_detection import LaneDetection
from waypoint_prediction import target_speed_prediction, waypoint_prediction

try:  # pragma: no cover - convenience for running as a script
    from .dashboard_utils import (
        create_dashboard,
        create_env,
        extract_speed,
        load_config,
        reset_dashboard,
        update_dashboard,
    )
except ImportError:  # pragma: no cover - executed when run directly via python path/to/script
    from dashboard_utils import (  # type: ignore
        create_dashboard,
        create_env,
        extract_speed,
        load_config,
        reset_dashboard,
        update_dashboard,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def initialize_pygame(fps: int) -> tuple[pygame.time.Clock, int]:
    pygame.init()
    pygame.display.set_caption("Lane detection test - focus here for keyboard input")
    pygame.display.set_mode((640, 480), pygame.HIDDEN)
    return pygame.time.Clock(), fps


def process_input(
    key_bindings: Dict[str, int],
    *,
    steering_sensitivity: float,
    action_intensity: float,
) -> tuple[np.ndarray, bool]:
    """Process keyboard input into a CarRacing action vector."""

    action = np.zeros(3, dtype=np.float32)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return action, True

    keys = pygame.key.get_pressed()
    if keys[key_bindings["quit"]]:
        return action, True
    if keys[key_bindings["left"]]:
        action[0] = -steering_sensitivity
    if keys[key_bindings["right"]]:
        action[0] = +steering_sensitivity
    if keys[key_bindings["gas"]]:
        action[1] = +action_intensity
    if keys[key_bindings["brake"]]:
        action[2] = +action_intensity

    return action, False


def drive(
    *,
    render_mode: str,
    steering_sensitivity: float,
    action_intensity: float,
    show_dashboard: bool,
    fps: int,
    key_bindings: Dict[str, int],
) -> None:
    config = load_config()
    env = create_env(config, render_mode=render_mode)

    lane_detector = LaneDetection(**config.perception.lane_detection.to_kwargs())

    dashboard = create_dashboard(config, enabled=show_dashboard)

    observation, info = env.reset()
    reset_dashboard(dashboard, observation, info)

    clock, fps = initialize_pygame(fps)

    total_reward = 0.0
    steps = 0
    done = False

    timestep = config.runtime.timestep_seconds

    while not done:
        action, quit_signal = process_input(
            key_bindings,
            steering_sensitivity=steering_sensitivity,
            action_intensity=action_intensity,
        )
        if quit_signal:
            break

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        left_lane, right_lane = lane_detector.lane_detection(observation)
        waypoints = waypoint_prediction(left_lane, right_lane)
        target_speed = target_speed_prediction(waypoints)

        speed = extract_speed(info)

        update_dashboard(
            dashboard,
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            step_index=steps,
            speed=speed,
            timestep=timestep,
            lanes=(left_lane, right_lane),
            waypoints=waypoints,
            target_speed=target_speed,
        )

        if steps % 10 == 0 or terminated or truncated:
            logging.info(
                "step %03d action=%s reward=%+.2f speed=%+.2f target=%+.2f",  # noqa: G004
                steps,
                ", ".join(f"{val:+0.2f}" for val in action.tolist()),
                reward,
                speed,
                target_speed,
            )

        env.render()
        clock.tick(fps)

        steps += 1
        done = terminated or truncated

    logging.info("Episode finished after %d steps. Total reward: %.2f", steps, total_reward)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dashboard", action="store_true", help="disable live dashboard")
    parser.add_argument("--render_mode", default="human")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--steering", type=float, default=1.0)
    parser.add_argument("--throttle", type=float, default=1.0)
    args = parser.parse_args()

    bindings = {
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "gas": pygame.K_UP,
        "brake": pygame.K_DOWN,
        "quit": pygame.K_ESCAPE,
    }

    drive(
        render_mode=args.render_mode,
        steering_sensitivity=args.steering,
        action_intensity=args.throttle,
        show_dashboard=not args.no_dashboard,
        fps=args.fps,
        key_bindings=bindings,
    )
