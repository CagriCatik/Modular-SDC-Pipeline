"""Keyboard-controlled waypoint planner harness with live dashboard."""

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


def process_events(action: np.ndarray, key_bindings: Dict[str, int]) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == key_bindings["quit"]:
            return True
    return False


def update_action(action: np.ndarray, key_bindings: Dict[str, int]) -> None:
    keys = pygame.key.get_pressed()
    action[0] = 0.0
    if keys[key_bindings["left"]]:
        action[0] = -1.0
    if keys[key_bindings["right"]]:
        action[0] = +1.0
    action[1] = float(keys[key_bindings["gas"]])
    action[2] = float(keys[key_bindings["brake"]]) * 0.8


def run_episode(
    *,
    render_mode: str,
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

    pygame.init()
    pygame.display.set_caption("Waypoint planner harness - focus window for keyboard input")
    pygame.display.set_mode((640, 480), pygame.HIDDEN)
    clock = pygame.time.Clock()

    action = np.zeros(3, dtype=np.float32)
    timestep = config.runtime.timestep_seconds

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        if process_events(action, key_bindings):
            break
        update_action(action, key_bindings)

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
                "step %03d reward=%+.2f speed=%+.2f target=%+.2f",  # noqa: G004
                steps,
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
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--render_mode", default="human")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    bindings = {
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "gas": pygame.K_UP,
        "brake": pygame.K_DOWN,
        "quit": pygame.K_ESCAPE,
    }

    run_episode(
        render_mode=args.render_mode,
        show_dashboard=not args.no_dashboard,
        fps=args.fps,
        key_bindings=bindings,
    )
