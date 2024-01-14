from math import (sin, cos, pi, atan2, copysign)
from pyray import *
import time

# --- Physics ---
class Physics:
    def __init__(self):
        # time (s)
        self.wc_time = 0.0
        # position (m, m, rads)
        self.wc_position_x = 0.0
        self.wc_position_y = 0.0
        self.wc_position_z = 0.0
        # velocity (m/s, m/s, rads/s)
        self.wc_velocity_x = 0.0
        self.wc_velocity_y = 0.0
        self.wc_velocity_z = 0.0
        # acceleration (m/s^2, m/s^2)
        self.wc_acceleration_x = 0.0
        self.wc_acceleration_y = 0.0
        # constants
        self.mass = 1500
        self.inertia = 1500
        self.cg_front = 1.25
        self.cg_rear = 1.25
        self.grip_front = 2.0
        self.grip_rear = 2.0
        self.stiffness_front = 5.0
        self.stiffness_rear = 5.2
        self.resistance_drag = 5.0
        self.resistance_roll = 30.0
        self.force_engine = 8000.0
        self.force_brake = 10000.0
        self.wheel_base = self.cg_front + self.cg_rear
        self.cg_front_ratio = self.cg_front / self.wheel_base
        self.cg_rear_ratio = self.cg_rear / self.wheel_base


    def update(self, dt, throttle, brake, steering):
        # calculate steering angle
        steering = max(min(steering, 0.65), -0.65) * -1.0
        # calculate position sin and cos
        position_sn = sin(self.wc_position_z)
        position_cs = cos(self.wc_position_z)
        # transform into local velocity
        velocity_x = position_cs * self.wc_velocity_x + position_sn * self.wc_velocity_y
        velocity_y = position_cs * self.wc_velocity_y - position_sn * self.wc_velocity_x
        
        # calculate angular velocity on each axle
        angular_front = self.cg_front * self.wc_velocity_z
        angular_rear = -self.cg_rear * self.wc_velocity_z
        # calculate slip angles on each axle
        slip_front = atan2(velocity_y + angular_front, abs(velocity_x)) - steering * copysign(1.0, velocity_x)
        slip_rear = atan2(velocity_y + angular_rear, abs(velocity_x))
        # calculate lateral forces on each axle
        lateral_front = min(max(self.stiffness_front * slip_front, -self.grip_front), self.grip_front) * (self.mass * self.cg_front_ratio)
        lateral_rear = min(max(self.stiffness_rear * slip_rear, -self.grip_rear), self.grip_rear) * (self.mass * self.cg_rear_ratio)

        # calculate drag resistance for each axes
        drag_x = -self.resistance_drag * velocity_x * abs(velocity_x)
        drag_y = -self.resistance_drag * velocity_y * abs(velocity_y)
        # calculate rolling resistance for each axes
        roll_x = -self.resistance_roll * velocity_x
        roll_y = -self.resistance_roll * velocity_y
        # calculate traction force for each axes
        traction_x = (self.force_engine * throttle) - (self.force_brake * brake) * copysign(1.0, velocity_x)
        traction_y = 0.0

        # calculate total force on each axes
        force_x = drag_x + roll_x + traction_x
        force_y = drag_y + roll_y + traction_y + cos(steering) * lateral_front + lateral_rear
        # calculate acceleration on each axes
        acceleration_x = force_x / self.mass
        acceleration_y = force_y / self.mass
        # calculate angular torque
        angular_torque = cos(steering) * lateral_front * self.cg_front - lateral_rear * self.cg_rear

        self.wc_acceleration_x = position_cs * acceleration_x - position_sn * acceleration_y
        self.wc_acceleration_y = position_sn * acceleration_x + position_cs * acceleration_y
        self.wc_velocity_x += dt * self.wc_acceleration_x
        self.wc_velocity_y += dt * self.wc_acceleration_y
        self.wc_velocity_z += (angular_torque / self.inertia) * dt
        self.wc_position_x += (velocity_x * dt) + (self.wc_acceleration_x * dt * dt) * 0.5
        self.wc_position_y += (velocity_y * dt) + (self.wc_acceleration_y * dt * dt) * 0.5        
        self.wc_position_z = (self.wc_position_z + (dt * self.wc_velocity_z * dt) * 0.5) % (2 * pi)
        self.wc_time += dt

        package = {
            'wc_position_x': self.wc_position_x,
            'wc_position_y': self.wc_position_y,
            'wc_position_z': self.wc_position_z,
            'wc_velocity_x': self.wc_velocity_x,
            'wc_velocity_y': self.wc_velocity_y,
            'wc_velocity_z': self.wc_velocity_z,
            'wc_acceleration_x': self.wc_acceleration_x,
            'wc_acceleration_y': self.wc_acceleration_y,
            'wc_time': self.wc_time
        }

        print(self.wc_velocity_x, self.wc_velocity_y, self.wc_position_x, self.wc_position_y, self.wc_position_z, self.wc_velocity_z)
        self.output_control()
        return package

    # cruise control (m/s)
    def cruise_control(self, velocity_x):
        # set target velocity, position
        self.wc_velocity_x = velocity_x
        # self.wc_position_x = -10.0 * velocity_x
        # calculate throttle for velocity
        drag_x = self.resistance_drag * velocity_x * abs(velocity_x)
        roll_x = self.resistance_roll * velocity_x
        return ((drag_x + roll_x) / self.force_engine)

    # output control
    def output_control(self):
        # normalised position_y 
        n_position_y = (self.wc_position_y / 4.6)
        # normalised position_z
        n_position_z = self.cg_front * round(cos(self.wc_position_z), 1)
        # 
        print(n_position_y, n_position_z)








# --- Renderer ---
class Renderer:
    def __init__(self):
        # screen
        self.screen_width = 800
        self.screen_height = 400
        self.screen_title = 'cipher'
        self.screen_fps = 10
        # raylib
        init_window(self.screen_width, self.screen_height, self.screen_title)
        set_target_fps(self.screen_fps)
        begin_drawing()
        clear_background(RAYWHITE)
        end_drawing()

    def render(self, package):
        begin_drawing()
        clear_background(RAYWHITE)
        
        # --- world coordinates ---
        draw_text("pos:", 20, 20, 20, BLACK)
        draw_text("vel:", 20, 50, 20, BLACK)
        draw_text("acc:", 20, 80, 20, BLACK)
        draw_text("rad:", 20, 110, 20, BLACK)

        draw_text(str(round(package['wc_position_x'], 2)), 80, 20, 20, BLACK)
        draw_text(str(round(package['wc_position_y'], 2)), 200, 20, 20, BLACK)

        draw_text(str(round(package['wc_velocity_x'], 2)), 80, 50, 20, BLACK)
        draw_text(str(round(package['wc_velocity_y'], 2)), 200, 50, 20, BLACK)

        draw_text(str(round(package['wc_acceleration_x'], 2)), 80, 80, 20, BLACK)
        draw_text(str(round(package['wc_acceleration_y'], 2)), 200, 80, 20, BLACK)

        draw_text(str(round(package['wc_position_z'], 2)), 80, 110, 20, BLACK)
        draw_text(str(round(package['wc_velocity_z'], 2)), 200, 110, 20, BLACK)

        # --- timing ---
        draw_text("time:", 20, self.screen_height - 40, 20, BLACK)
        draw_text(str(round(package['wc_time'], 2)), 80, self.screen_height - 40, 20, BLACK)

        # --- compass ---
        compass_x = self.screen_width // 4 * 3
        compass_y = self.screen_height // 2
        radius = self.screen_height // 10
        angle = package['wc_position_z'] - 1.5708

        draw_line(compass_x, compass_y - radius * 3, compass_x, compass_y + radius * 3, BLACK)
        draw_line(compass_x - radius * 3, compass_y, compass_x + radius * 3, compass_y, BLACK)
        draw_triangle(
            Vector2(compass_x + radius * 2 * cos(angle), compass_y + radius * 2 * sin(angle)), 
            Vector2(compass_x + radius * cos(angle + 4 * pi / 3), compass_y + radius * sin(angle + 4 * pi / 3)),
            Vector2(compass_x + radius * cos(angle + 2 * pi / 3), compass_y + radius * sin(angle + 2 * pi / 3)), 
            BLACK
        )

        end_drawing()












# --- cipher engine ---
class Cipher:
    def __init__(self, render=True):
        # physics
        self.physics = Physics()
        # renderer
        self.render = render
        if render:
            self.renderer = Renderer()

    def update(self, loop, dt, throttle, brake, steering):
        # update physics
        package = None
        for _ in range(loop):    
            package = self.physics.update(dt, throttle, brake, steering)
        # update renderer
        if self.render:
            self.renderer.render(package)
            # self.physics.output_control()


cipher = Cipher()
throttle = cipher.physics.cruise_control(30.0)
for _ in range(5):
    cipher.update(5, 0.2, throttle, 0.0, 0.8)
    time.sleep(0.4)