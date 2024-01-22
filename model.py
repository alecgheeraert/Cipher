from controller import Controller
from math import pi
import time

controller = Controller(20.0, 0.2)

for i in range(2):
    print("[position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, time]")
    observation = controller.reset()
    terminated = truncated = False
    reward = 0.0
    counter = 0
    steering = 0.0
    angle = observation[1]
    while not terminated and not truncated:
        counter += 1
        if abs(angle) < pi:
            steering = -0.1
        else:
            steering = 0.1
        observation, reward, terminated, truncated = controller.step(0, 0, steering)
        print(observation, steering)
        controller.render()
        angle = observation[controller._observation_space.position_z]
    print("Episode finished after {} timesteps".format(counter))