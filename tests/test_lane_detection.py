# test_lane_detection.py
import numpy as np
import gymnasium as gym
import pygame
import logging
import matplotlib.pyplot as plt
from typing import Dict, Tuple

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lane_detection import LaneDetection  # expects debug-enabled class

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# -----------------------------------------------------------------------------
# Debug window manager: one Matplotlib window per pipeline step
# -----------------------------------------------------------------------------
class DebugWindows:
    def __init__(self):
        plt.ion()
        plt.show()
        self.fig, self.axes = plt.subplots(2, 2, num="debug_2x2", figsize=(8, 8))
        # slot -> dict with keys: name, im
        self.slots = {i: {"name": None, "im": None} for i in range(4)}
        # explicit order: replace or extend as you add frames in lane_detection
        self.order = ["10_gray", "30_edges", "50_maxima_mask", "90_overlay"]

    def _is_grayscale(self, img: np.ndarray) -> bool:
        return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)

    def _prep_data(self, img: np.ndarray):
        if self._is_grayscale(img):
            data = img.squeeze()
            vmin = float(np.min(data))
            vmax = float(np.max(data))
            if vmin == vmax:
                vmax = vmin + 1.0
            return data, dict(cmap="gray", vmin=vmin, vmax=vmax)
        return img, {}

    def update(self, frames: dict):
        # Select images by preferred order, keep only those present
        selected = [(k, frames[k]) for k in self.order if k in frames]
        # Render into 4 slots
        for idx in range(4):
            ax = self.axes.ravel()[idx]
            if idx >= len(selected):
                # clear empty slots
                ax.cla()
                ax.axis("off")
                self.slots[idx] = {"name": None, "im": None}
                continue

            name, img = selected[idx]
            data, kwargs = self._prep_data(img)

            slot = self.slots[idx]
            # If slot name changed or artist missing/detached, recreate
            recreate = (
                slot["im"] is None
                or slot["name"] != name
                or slot["im"].axes is None
                or slot["im"].axes is not ax
            )
            if recreate:
                ax.cla()
                ax.set_title(name)
                im = ax.imshow(data, interpolation="nearest", **kwargs)
                ax.axis("off")
                self.slots[idx] = {"name": name, "im": im}
            else:
                im = slot["im"]
                im.set_data(data)
                # update clim only for grayscale
                if "vmin" in kwargs and "vmax" in kwargs:
                    im.set_clim(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
                # keep title stable if same name
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def tick(self):
        plt.pause(0.001)





# -----------------------------------------------------------------------------
# Environment / input helpers
# -----------------------------------------------------------------------------
def initialize_environment(env_name="CarRacing-v3", render_mode="human"):
    env = gym.make(env_name, render_mode=render_mode)
    return env


def initialize_pygame(fps=60):
    pygame.init()
    pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
    return pygame.time.Clock(), fps


def process_input(key_bindings, steering_sensitivity=1.0, action_intensity=1.0, action_space=None):
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
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        action = np.clip(action, low, high).astype(action_space.dtype, copy=False)

    return action, False


def handle_step(env, action):
    observation, reward, terminated, truncated, _info = env.step(action)
    return observation, reward, terminated, truncated


# -----------------------------------------------------------------------------
# Main loop with per-step debug windows
# -----------------------------------------------------------------------------
def game_loop(env, clock, fps, key_bindings, steering_sensitivity, action_intensity):
    total_reward = 0.0
    observation, _ = env.reset()
    done = False

    # Enable debug collection inside LaneDetection
    LD_module = LaneDetection(debug=True)

    # Non-blocking Matplotlib and window manager
    plt.ion()
    plt.show()
    dbg = DebugWindows()

    # Keep your overlay figure for spline/waypoint summary
    overlay_fig = plt.figure("overlay", clear=True)

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

        # Run lane detection; intermediate frames are captured internally
        _splines = LD_module.lane_detection(observation)

        if steps % 2 == 0 or terminated or truncated:
            logging.info(
                "Action: [%s], Step: %d, Total Reward: %.2f",
                ", ".join("{:+0.2f}".format(x) for x in action.tolist()),
                steps,
                total_reward,
            )

            # Render all intermediate steps into persistent windows
            frames = LD_module.get_debug_frames(reset=True)
            if frames:
                dbg.update(frames)
                dbg.tick()

            # Optional overlay of detected lanes on the RGB frame
            LD_module.plot_state_lane(observation, steps, overlay_fig)
            plt.pause(0.001)

        clock.tick(fps)
        steps += 1

        if terminated or truncated:
            logging.info("Episode ended. Total reward: %s", total_reward)
            observation, _ = env.reset()

    return total_reward


def close_environment(env):
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
