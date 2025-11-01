import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0] / "src"))

import argparse
import numpy as np
import gymnasium as gym

from sdc_wrapper import SDC_Wrapper
from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController


def _action_contract(a, action_space):
    a = np.asarray(a, dtype=np.float32).reshape(3)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return np.clip(a, low, high).astype(action_space.dtype, copy=False)


def _reset_with_speed(env, **kwargs):
    s, info = env.reset(**kwargs)
    speed = float(info["speed"]) if isinstance(info, dict) and "speed" in info else 0.0
    return s, speed


def evaluate(env):
    for episode in range(5):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        try:
            s, speed = _reset_with_speed(env)
        except Exception:
            print("Please note that you can't use the window on the cluster")
            raise

        LD_module = LaneDetection()
        LatC_module = LateralController()
        LongC_module = LongitudinalController()

        reward_per_episode = 0.0

        for t in range(600):
            lane1, lane2 = LD_module.lane_detection(s)

            waypoints = waypoint_prediction(lane1, lane2, t)
            target_speed = target_speed_prediction(waypoints, t)

            steer = LatC_module.stanley(waypoints, speed)
            gas, brake = LongC_module.control(speed, target_speed)

            a = _action_contract([steer, gas, brake], env.action_space)

            s, r, done, trunc, info = env.step(a)
            speed = float(info["speed"]) if isinstance(info, dict) and "speed" in info else speed
            reward_per_episode += float(r)

            if done or trunc:
                break

        print("episode %d \t reward %f" % (episode, reward_per_episode))


def calculate_score_for_leaderboard(env):
    # DO NOT CHANGE
    seeds = [97657630, 47460391, 22619914, 76925063, 84647422,
             83470445, 77482096, 94017676, 99341122, 58134947]

    total_reward = 0.0
    for episode, seed in enumerate(seeds):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        try:
            s, speed = _reset_with_speed(env, seed=seed)
        except Exception:
            print("Please note that you can't use the window on the cluster")
            raise

        LD_module = LaneDetection()
        LatC_module = LateralController()
        LongC_module = LongitudinalController()

        reward_per_episode = 0.0
        for t in range(600):
            lane1, lane2 = LD_module.lane_detection(s)

            waypoints = waypoint_prediction(lane1, lane2, t)
            target_speed = target_speed_prediction(waypoints, t)

            steer = LatC_module.stanley(waypoints, speed)
            gas, brake = LongC_module.control(speed, target_speed)

            a = _action_contract([steer, gas, brake], env.action_space)

            s, r, done, trunc, info = env.step(a)
            speed = float(info["speed"]) if isinstance(info, dict) and "speed" in info else speed
            reward_per_episode += float(r)

            if done or trunc:
                break

        print("episode %d \t reward %f" % (episode, reward_per_episode))
        total_reward += float(np.clip(reward_per_episode, 0.0, np.inf))

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
