from random import *
from pyray import *
from math import *

class _ObservationSpace:
    def __init__(self):
        self.position_x = 0
        self.position_y = 1
        self.position_z = 2
        self.velocity_x = 3
        self.velocity_y = 4
        self.velocity_z = 5
        self.time = 6

class Controller:
    def __init__(self, cruise_velocity, dt, render=True, fps=20):
        """
        Parameters
        ----------
        cruise_velocity : float
            The cruising velocity of the vehicle
        dt : float
            The time delta between each step
        render : bool, optional
            If rendering should be enabled (default is True)
        fps : int, optional
            The rendering fps (default is 20)
        """
        self._cruise_velocity = cruise_velocity
        self._dt = dt
        self._render = render
        # init renderer
        self.__screen_width = 800
        self.__screen_height = 400
        self.__screen_title = 'cipher'
        self.__screen_fps = fps
        # raylib
        if render:
            init_window(self.__screen_width, self.__screen_height, self.__screen_title)
            set_target_fps(self.__screen_fps)
        # set wc
        self._observation_space = _ObservationSpace()
        self._action_space = self.__action_space()
        self.reset()

    def reset(self):
        resistance_drag = 5.0
        resistance_roll = 30.0
        force_engine = 8000.0
        # calculate cruise forces
        drag_x = resistance_drag * self._cruise_velocity * abs(self._cruise_velocity)
        roll_x = resistance_roll * self._cruise_velocity
        # time (s)
        self.__wc_time = 0.0
        # controls
        self.__wc_throttle = (drag_x + roll_x) / force_engine
        self.__wc_brake = 0.0
        self.__wc_steering = 0.0
        # position (m, m, rads)
        self.__wc_position_x = 0.0
        self.__wc_position_y = 0.0
        self.__wc_position_z = 0.0
        # velocity (m/s, m/s, rads/s)
        self.__wc_velocity_x = self._cruise_velocity
        self.__wc_velocity_y = 0.0
        self.__wc_velocity_z = 0.0
        # acceleration (m/s^2, m/s^2)
        self.__wc_acceleration_x = 0.0
        self.__wc_acceleration_y = 0.0
        # run physics
        self.__slip()
        for _ in range(3):
            self.__physics(True)
        
        # clear renderer
        return self.__observation()


    def step(self, throttle, brake, steering):
        # filter controls
        # TODO smoother controls
        self.__wc_throttle = max(min(throttle, 1.0), 0.0)
        self.__wc_brake = max(min(brake, 1.0), 0.0)
        self.__wc_steering = max(min(steering, 0.65), -0.65) * -1.0
        # run physics
        self.__physics()
        # filter observation
        # return observation, reward, terminated, truncated
        observation = self.__observation()
        terminated = self.__terminated(observation)
        truncated = self.__truncated(observation)
        reward = self.__reward(observation, terminated, truncated)
        return observation, reward, terminated, truncated

    def render(self):
        if self._render:
            if window_should_close():
                close_window()
                return False
            self.__render()

    def __physics(self, slip=False):
        # constants
        mass = 1500
        inertia = 1500
        cg_front = 1.25
        cg_rear = 1.25
        grip_front = 2.0
        grip_rear = 2.0
        stiffness_front = 5.0
        stiffness_rear = 5.2
        resistance_drag = 5.0
        resistance_roll = 30.0
        force_engine = 8000.0
        force_brake = 10000.0
        wheel_base = cg_front + cg_rear
        cg_front_ratio = cg_front / wheel_base
        cg_rear_ratio = cg_rear / wheel_base
        # calculate position sin and cos
        position_sn = sin(self.__wc_position_z)
        position_cs = cos(self.__wc_position_z)
        # transform into local velocity
        velocity_x = position_cs * self.__wc_velocity_x + position_sn * self.__wc_velocity_y
        velocity_y = position_cs * self.__wc_velocity_y - position_sn * self.__wc_velocity_x
        
        # calculate angular velocity on each axle
        angular_front = cg_front * self.__wc_velocity_z
        angular_rear = -cg_rear * self.__wc_velocity_z
        # calculate slip angles on each axle
        slip_front = atan2(velocity_y + angular_front, abs(velocity_x)) - self.__wc_steering * copysign(1.0, velocity_x)
        slip_rear = atan2(velocity_y + angular_rear, abs(velocity_x))
        # calculate lateral forces on each axle
        lateral_front = min(max(stiffness_front * slip_front, -grip_front), grip_front) * (mass * cg_front_ratio)
        lateral_rear = min(max(stiffness_rear * slip_rear, -grip_rear), grip_rear) * (mass * cg_rear_ratio)

        # calculate drag resistance for each axes
        drag_x = -resistance_drag * velocity_x * abs(velocity_x)
        drag_y = -resistance_drag * velocity_y * abs(velocity_y)
        # calculate rolling resistance for each axes
        roll_x = -resistance_roll * velocity_x
        roll_y = -resistance_roll * velocity_y
        # calculate traction force for each axes
        traction_x = (force_engine * self.__wc_throttle) - (force_brake * self.__wc_brake) * copysign(1.0, velocity_x)
        traction_y = 0.0

        if slip:
            lateral_front *= self.__slip_lateral_front
            lateral_rear *= self.__slip_lateral_rear
            traction_x *= self.__slip_traction_x
            print(lateral_front, lateral_rear, traction_x, self.__wc_steering)

        # calculate total force on each axes
        force_x = drag_x + roll_x + traction_x
        force_y = drag_y + roll_y + traction_y + cos(self.__wc_steering) * lateral_front + lateral_rear
        # calculate acceleration on each axes
        acceleration_x = force_x / mass
        acceleration_y = force_y / mass
        # calculate angular torque
        angular_torque = cos(self.__wc_steering) * lateral_front * cg_front - lateral_rear * cg_rear

        self.__wc_acceleration_x = position_cs * acceleration_x - position_sn * acceleration_y
        self.__wc_acceleration_y = position_sn * acceleration_x + position_cs * acceleration_y
        self.__wc_velocity_x += self._dt * self.__wc_acceleration_x
        self.__wc_velocity_y += self._dt * self.__wc_acceleration_y
        self.__wc_velocity_z += (angular_torque / inertia) * self._dt
        self.__wc_position_x += (velocity_x * self._dt) + (self.__wc_acceleration_x * self._dt * self._dt) * 0.5
        self.__wc_position_y += (velocity_y * self._dt) + (self.__wc_acceleration_y * self._dt * self._dt) * 0.5        
        self.__wc_position_z = (self.__wc_position_z + (self._dt * self.__wc_velocity_z * self._dt) * 0.5) % (2 * pi)
        self.__wc_time += self._dt

    def __slip(self):
        self.__slip_lateral_front = round(uniform(0.8, 1.0), 3)
        self.__slip_lateral_rear = round(uniform(0.6, 0.8), 3)
        self.__slip_traction_x = round(uniform(0.6, 0.8), 3)
        if randint(0, 1) == 0:
            self.__wc_steering = round(uniform(0.2, 0.3), 2)
        else:
            self.__wc_steering = round(uniform(-0.3, -0.2), 2)

    def __action_space(self):
        return [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]

    # observations
    def __observation(self):
        # nc_position_y = round((self.__wc_position_y / 1.6), 3)
        # nc_position_z = round(copysign(cos(self.__wc_position_z), self.__wc_velocity_z), 5)
        position_x = round(self.__wc_position_x, 3)
        position_y = round(self.__wc_position_y, 3)
        position_z = round(copysign(self.__wc_position_z, self.__wc_velocity_z), 3)
        velocity_x = round(self.__wc_velocity_x, 3)
        velocity_y = round(self.__wc_velocity_y, 3)
        velocity_z = round(self.__wc_velocity_z, 3)
        time = round(self.__wc_time, 3)
        return [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, time]
    
    def __reward(self, observation, terminated, truncated):
        reward = 0.0
        if terminated:
            reward += -100.0
        if truncated:
            reward += -20.0
        if abs(observation[1]) > 0.5:
            reward += 10.0
        return

    def __terminated(self, observation):
        if observation[self._observation_space.position_y] > 4.8 or observation[self._observation_space.position_y] < -4.8:
            return True
        elif observation[self._observation_space.time] >= 30.0:
            return True
        return False
    
    def __truncated(self, observation):
        if observation[self._observation_space.position_x] >= 300.0:
            return True
        return False
    
    def __render(self):
        begin_drawing()
        clear_background(RAYWHITE)
        
        # world coordinates
        draw_text("pos:", 20, 20, 20, BLACK)
        draw_text("vel:", 20, 50, 20, BLACK)
        draw_text("acc:", 20, 80, 20, BLACK)
        draw_text("rad:", 20, 110, 20, BLACK)

        draw_text(str(round(self.__wc_position_x, 2)), 80, 20, 20, BLACK)
        draw_text(str(round(self.__wc_position_y, 2)), 200, 20, 20, BLACK)

        draw_text(str(round(self.__wc_velocity_x, 2)), 80, 50, 20, BLACK)
        draw_text(str(round(self.__wc_velocity_y, 2)), 200, 50, 20, BLACK)

        draw_text(str(round(self.__wc_acceleration_x, 2)), 80, 80, 20, BLACK)
        draw_text(str(round(self.__wc_acceleration_y, 2)), 200, 80, 20, BLACK)

        draw_text(str(round(self.__wc_position_z, 2)), 80, 110, 20, BLACK)
        draw_text(str(round(self.__wc_velocity_z, 2)), 200, 110, 20, BLACK)

        # timing
        draw_text("time:", 20, self.__screen_height - 40, 20, BLACK)
        draw_text(str(round(self.__wc_time, 2)), 80, self.__screen_height - 40, 20, BLACK)

        # compass
        compass_x = self.__screen_width // 4 * 3
        compass_y = self.__screen_height // 2
        radius = self.__screen_height // 10
        angle = self.__wc_position_z - 1.5708
        draw_line(compass_x, compass_y - radius * 3, compass_x, compass_y + radius * 3, BLACK)
        draw_line(compass_x - radius * 3, compass_y, compass_x + radius * 3, compass_y, BLACK)
        draw_triangle(
            Vector2(compass_x + radius * 2 * cos(angle), compass_y + radius * 2 * sin(angle)), 
            Vector2(compass_x + radius * cos(angle + 4 * pi / 3), compass_y + radius * sin(angle + 4 * pi / 3)),
            Vector2(compass_x + radius * cos(angle + 2 * pi / 3), compass_y + radius * sin(angle + 2 * pi / 3)), 
            BLACK
        )
        end_drawing()