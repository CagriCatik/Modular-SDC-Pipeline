import numpy as np
import gymnasium as gym
import pygame
import logging
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lane_detection import LaneDetection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def initialize_environment(env_name="CarRacing-v3", render_mode="human"):
    """Initialize the Gymnasium environment and return it."""
    env = gym.make(env_name, render_mode=render_mode)
    return env

def initialize_pygame(fps=60):
    """Initialize Pygame, create a hidden display to enable event processing, and return a clock and fps."""
    pygame.init()
    # Hidden 1x1 window to ensure event pump and key state work on all platforms
    pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
    return pygame.time.Clock(), fps

def process_input(key_bindings, steering_sensitivity=1.0, action_intensity=1.0, action_space=None):
    """Process Pygame input and return (action: np.ndarray[shape=(3,), dtype=float32], quit: bool)."""
    steer, gas, brake = 0.0, 0.0, 0.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return np.zeros(3, dtype=np.float32), True

    keys = pygame.key.get_pressed()
    if keys[key_bindings["quit"]]:
        return np.zeros(3, dtype=np.float32), True
    if keys[key_bindings["left"]]:
        steer = -steering_sensitivity
    if keys[key_bindings["right"]]:
        steer = +steering_sensitivity
    if keys[key_bindings["gas"]]:
        gas = +action_intensity
    if keys[key_bindings["brake"]]:
        brake = +action_intensity

    action = np.array([steer, gas, brake], dtype=np.float32)

    if action_space is not None:
        # Conform strictly to env action contract
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        action = np.clip(action, low, high).astype(action_space.dtype, copy=False)

    return action, False

def handle_step(env, action):
    """Perform a step in the environment and return observation, reward, terminated, truncated."""
    observation, reward, terminated, truncated, _info = env.step(action)
    return observation, reward, terminated, truncated

def game_loop(env, clock, fps, key_bindings, steering_sensitivity, action_intensity):
    """Main game loop."""
    total_reward = 0.0
    observation, _ = env.reset()
    done = False

    # Initialize lane detection module
    LD_module = LaneDetection()

    # Initialize non-blocking plot
    fig = plt.figure()
    plt.ion()
    plt.show()

    steps = 0
    while not done:
        action, quit_signal = process_input(
            key_bindings,
            steering_sensitivity,
            action_intensity,
            action_space=env.action_space,
        )
        if quit_signal:
            done = True
            break

        observation, reward, terminated, truncated = handle_step(env, action)
        total_reward += reward

        # Perform lane detection
        _splines = LD_module.lane_detection(observation)

        # Plot lane detection results periodically
        if steps % 2 == 0 or terminated:
            logging.info(
                "Action: [%s], Step: %d, Total Reward: %.2f",
                ", ".join("{:+0.2f}".format(x) for x in action.tolist()),
                steps,
                total_reward,
            )
            LD_module.plot_state_lane(observation, steps, fig)
            plt.pause(0.001)  # ensure GUI refreshes

        clock.tick(fps)
        steps += 1

        if terminated or truncated:
            logging.info("Episode ended. Total reward: %s", total_reward)
            observation, _ = env.reset()

    return total_reward

def close_environment(env):
    """Close the environment and quit Pygame."""
    env.close()
    pygame.quit()

def drive(
    env_name="CarRacing-v3",
    render_mode="human",
    fps=60,
    steering_sensitivity=1.0,
    action_intensity=1.0,
    key_bindings=None,
):
    """Run the car driving simulation with customizable parameters."""
    if key_bindings is None:
        key_bindings = {
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
            "gas": pygame.K_UP,
            "brake": pygame.K_DOWN,
            "quit": pygame.K_ESCAPE,
        }

    logging.info("Initializing environment...")
    env = initialize_environment(env_name, render_mode)

    logging.info("Initialized environment: %s", env_name)
    clock, fps = initialize_pygame(fps)

    logging.info("Starting game loop...")
    total_reward = game_loop(env, clock, fps, key_bindings, steering_sensitivity, action_intensity)

    logging.info("Total reward for this session: %s", total_reward)
    logging.info("Closing environment...")
    close_environment(env)

if __name__ == "__main__":
    custom_key_bindings = {
        "left": pygame.K_a,    # A
        "right": pygame.K_d,   # D
        "gas": pygame.K_w,     # W
        "brake": pygame.K_s,   # S
        "quit": pygame.K_q,    # Q
    }

    drive(
        env_name="CarRacing-v3",
        render_mode="human",
        fps=60,
        steering_sensitivity=1.0,
        action_intensity=1.0,
        key_bindings=custom_key_bindings,
    )
