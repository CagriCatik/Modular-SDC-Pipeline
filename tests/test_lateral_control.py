import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))


from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController


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

# Initialize extra plot
fig = plt.figure()
plt.ion()
plt.show()

# Action variables: [steering, gas, brake]
a = np.array([0.0, 0.0, 0.0])

while True:
    # Perform step
    observation, reward, terminated, truncated, info = env.step(a)
    done = terminated or truncated

    # Lane detection
    lane1, lane2 = LD_module.lane_detection(observation)

    # Waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # Control with constant gas and no braking
    # Obtain the speed from the info dictionary if available
    if 'speed' in info:
        speed = info['speed']
    else:
        # Alternatively, estimate speed or set a default value
        speed = 0.1  # Replace with appropriate speed estimation if necessary

    # Update steering angle using the lateral controller
    a[0] = LatC_module.stanley(waypoints, speed)

    # Update total reward
    total_reward += reward

    # Outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("target_speed {:+0.2f}".format(target_speed))
        LD_module.plot_state_lane(observation, steps, fig, waypoints=waypoints)

    steps += 1
    env.render()

    # Check if stop
    if done or restart or steps >= 600:
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        break

env.close()
