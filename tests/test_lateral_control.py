# test_lateral_control.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Sequence

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController


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


# Initialize environment using gym.make()
env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset()

# Define variables
total_reward = 0.0
steps = 0
restart = False

# Initialize modules of the pipeline
LD_module = LaneDetection(debug=True)  # enable debug frames
LatC_module = LateralController()

# Overlay window and 2x2 debug window
plt.ion()
overlay_fig = plt.figure("overlay", figsize=(6, 6))
plt.show(block=False)
dbg = DebugWindows(order=("10_gray", "30_edges", "50_maxima_mask", "90_overlay"))

# Action variables: [steering, gas, brake]
a = np.array([0.0, 0.0, 0.0], dtype=np.float32)

while True:
    # Perform step
    obs, reward, terminated, truncated, info = env.step(a)
    done = terminated or truncated

    # Lane detection
    lane1, lane2 = LD_module.lane_detection(obs)

    # Waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # Speed from info or fallback
    speed = float(info["speed"]) if isinstance(info, dict) and "speed" in info else 0.1

    # Lateral control, constant gas, no brake
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1] = 0.5
    a[2] = 0.0

    # Update total reward
    total_reward += float(reward)

    # Outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("target_speed {:+0.2f}".format(target_speed))

        # Update 2x2 debug windows from LaneDetection debug frames
        frames = LD_module.get_debug_frames(reset=True)
        if frames:
            dbg.update(frames)
            dbg.tick()

        # Overlay window
        LD_module.plot_state_lane(obs, steps, overlay_fig, waypoints=waypoints)
        plt.pause(0.001)

    steps += 1
    env.render()

    # Check if stop
    if done or restart or steps >= 600:
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        break

env.close()
