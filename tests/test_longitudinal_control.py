import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController

# Initialize environment using gym.make()
env = gym.make('CarRacing-v3', render_mode='human')
env.reset()

# Define variables
total_reward = 0.0
steps = 0
restart = False

# Initialize modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()
LongC_module = LongitudinalController()

# Initialize extra plot
fig = plt.figure()
plt.ion()
plt.show()

# Action variables
a = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

while True:
    # Perform step
    observation, reward, terminated, truncated, info = env.step(a)
    done = terminated or truncated

    # Lane detection
    lane1, lane2 = LD_module.lane_detection(observation)

    # Waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints, max_speed=60, K_v=4.5)

    # Obtain the speed from the info dictionary or estimate it
    if 'speed' in info:
        speed = info['speed']
    else:
        # Alternatively, estimate speed or set a default value
        # Here, we estimate speed based on the car's linear velocity
        car = env.unwrapped.car
        speed = np.linalg.norm([car.hull.linearVelocity.x, car.hull.linearVelocity.y])

    # Control
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1], a[2] = LongC_module.control(speed, target_speed)

    # Update total reward
    total_reward += reward

    # Outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

        # Uncomment the following line if you want to plot lane detection
        # LD_module.plot_state_lane(observation, steps, fig, waypoints=waypoints)
        LongC_module.plot_speed(speed, target_speed, steps, fig)

    steps += 1
    env.render()

    # Check if stop
    if done or restart or steps >= 600:
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        break

env.close()
