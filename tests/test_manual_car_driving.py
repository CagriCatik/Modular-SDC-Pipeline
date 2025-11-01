import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import numpy as np
import gymnasium as gym
import pygame
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def normalize_env_name(env_name: str) -> str:
    # Map deprecated CarRacing-v2 to the current v3
    if env_name.strip().lower() == "carracing-v2":
        logging.info("Environment CarRacing-v2 is deprecated. Switching to CarRacing-v3.")
        return "CarRacing-v3"
    return env_name

def initialize_environment(env_name="CarRacing-v3", render_mode="human"):
    """Initialize the Gymnasium environment and return it."""
    env_name = normalize_env_name(env_name)
    env = gym.make(env_name, render_mode=render_mode)
    return env

def initialize_pygame(fps=60):
    """Initialize Pygame and return a clock object."""
    pygame.init()
    return pygame.time.Clock(), fps

def process_input(key_bindings, steering_sensitivity=1.0, action_intensity=1.0):
    """Process Pygame input and return an action array with customized key bindings."""
    action = [0.0, 0.0, 0.0]  # [steer, gas, brake]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return np.array(action, dtype=np.float32), True

    keys = pygame.key.get_pressed()
    if keys[key_bindings["quit"]]:
        return np.array(action, dtype=np.float32), True
    if keys[key_bindings["left"]]:
        action[0] = -steering_sensitivity
    if keys[key_bindings["right"]]:
        action[0] = +steering_sensitivity
    if keys[key_bindings["gas"]]:
        action[1] = +action_intensity
    if keys[key_bindings["brake"]]:
        action[2] = +action_intensity

    # CarRacing expects float32: steer in [-1,1], gas/brake in [0,1]
    action = np.array(action, dtype=np.float32)
    action[0] = np.clip(action[0], -1.0, 1.0)
    action[1] = np.clip(action[1], 0.0, 1.0)
    action[2] = np.clip(action[2], 0.0, 1.0)
    return action, False

def handle_step(env, action):
    """Perform a step in the environment and return updated information (without info)."""
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated

def game_loop(env, clock, fps, key_bindings, steering_sensitivity, action_intensity):
    """Main game loop where the action and environment are updated."""
    total_reward = 0.0
    observation, info = env.reset()
    done = False

    while not done:
        action, quit_signal = process_input(key_bindings, steering_sensitivity, action_intensity)
        if quit_signal:
            done = True
            break

        observation, reward, terminated, truncated = handle_step(env, action)
        total_reward += reward

        env.render()
        clock.tick(fps)

        if terminated or truncated:
            logging.info(f"Episode ended. Total reward: {total_reward}")
            observation, info = env.reset()

    return total_reward

def close_environment(env):
    """Close the environment and quit Pygame."""
    env.close()
    pygame.quit()

def drive(env_name="CarRacing-v3", render_mode="human", fps=60, steering_sensitivity=1.0, action_intensity=1.0, key_bindings=None):
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

    logging.info(f"Initialized environment: {env.unwrapped.spec.id}")
    clock, fps = initialize_pygame(fps)

    logging.info("Starting game loop...")
    total_reward = game_loop(env, clock, fps, key_bindings, steering_sensitivity, action_intensity)

    logging.info(f"Total reward for this session: {total_reward}")
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
