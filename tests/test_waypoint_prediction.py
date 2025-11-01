import sys
import pathlib
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame

# Local src imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction

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
LD_module = LaneDetection()

# Live plot
plt.ion()
fig = plt.figure()
plt.show(block=False)

# Action vector: [steering (-1..1), gas (0..1), brake (0..1)]
action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

total_reward = 0.0
steps = 0

running = True
restart = False

def soft_center_steer(a, k=0.90):
    # Optional steering recentring for smoother driving
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

    # Optional steering recentering for smoother control
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
        LD_module.plot_state_lane(obs, steps, fig, waypoints=waypoints)
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
