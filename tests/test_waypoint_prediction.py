# test_waypoint_prediction.py
import sys
import pathlib
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
from typing import Dict, Tuple, Sequence

# Local src imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction

# -----------------------------
# 2x2 debug window (slot-safe)
# -----------------------------
class DebugWindows:
    def __init__(self, order: Sequence[str] = ("10_gray", "30_edges", "50_maxima_mask", "90_overlay")):
        plt.ion()
        plt.show()
        self.fig, self.axes = plt.subplots(2, 2, num="debug_2x2", figsize=(8, 8))
        self.slots: Dict[int, Dict[str, object]] = {i: {"name": None, "im": None} for i in range(4)}
        self.order = list(order)

    @staticmethod
    def _is_grayscale(img: np.ndarray) -> bool:
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

    def update(self, frames: Dict[str, np.ndarray]):
        selected = [(k, frames[k]) for k in self.order if k in frames]
        for idx in range(4):
            ax = self.axes.ravel()[idx]
            if idx >= len(selected):
                ax.cla()
                ax.axis("off")
                self.slots[idx] = {"name": None, "im": None}
                continue

            name, img = selected[idx]
            data, kwargs = self._prep_data(img)

            slot = self.slots[idx]
            recreate = (
                slot["im"] is None
                or slot["name"] != name
                or getattr(slot["im"], "axes", None) is None
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
                if "vmin" in kwargs and "vmax" in kwargs:
                    im.set_clim(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    @staticmethod
    def tick():
        plt.pause(0.001)


# Config
FPS = 60
STEER_LEFT = -1.0
STEER_RIGHT = +1.0
THROTTLE = +1.0
BRAKE = 0.8  # in [0,1]

# Env
env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset()

# Pygame init (required to receive keyboard events)
pygame.init()
pygame.display.set_caption("Keyboard Control - Focus here")
pygame.display.set_mode((640, 480), pygame.RESIZABLE)
clock = pygame.time.Clock()

# Pipeline modules
LD_module = LaneDetection(debug=True)  # enable debug frames

# Live plot windows
plt.ion()
overlay_fig = plt.figure("overlay", figsize=(6, 6))
plt.show(block=False)

# 2x2 debug windows
dbg = DebugWindows(order=("10_gray", "30_edges", "50_maxima_mask", "90_overlay"))

# Action vector: [steering (-1..1), gas (0..1), brake (0..1)]
action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

total_reward = 0.0
steps = 0

running = True
restart = False

def soft_center_steer(a, k=0.90):
    a[0] *= k

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                restart = True
            elif event.key == pygame.K_LEFT:
                action[0] = STEER_LEFT
            elif event.key == pygame.K_RIGHT:
                action[0] = STEER_RIGHT
            elif event.key == pygame.K_UP:
                action[1] = THROTTLE
            elif event.key == pygame.K_DOWN:
                action[2] = BRAKE

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT and action[0] == STEER_LEFT:
                action[0] = 0.0
            elif event.key == pygame.K_RIGHT and action[0] == STEER_RIGHT:
                action[0] = 0.0
            elif event.key == pygame.K_UP:
                action[1] = 0.0
            elif event.key == pygame.K_DOWN:
                action[2] = 0.0

    soft_center_steer(action, k=0.90)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    # Perception and planning pipeline
    lane1, lane2 = LD_module.lane_detection(obs)
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # Accounting
    total_reward += reward

    if steps % 2 == 0:
        print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        # Update 2x2 debug windows from LaneDetection debug frames
        frames = LD_module.get_debug_frames(reset=True)
        if frames:
            dbg.update(frames)
            dbg.tick()

        # Overlay window
        LD_module.plot_state_lane(obs, steps, overlay_fig, waypoints=waypoints)
        plt.pause(0.001)

    steps += 1

    # Episode end or manual restart
    done = terminated or truncated
    if done or restart or steps >= 600:
        print("episode_end step {} total_reward {:+0.2f}".format(steps, total_reward))
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        action[:] = 0.0

    # Frame rate control
    clock.tick(FPS)

# Cleanup
pygame.quit()
env.close()
