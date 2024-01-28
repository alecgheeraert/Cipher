from random import uniform
import numpy as np
from pyray import *
from math import *
from gymnasium import spaces, Env

class Controller(Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, cruise_velocity, dt, fps=20, render_mode=None):
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
        super(Controller, self).__init__()
        self._cruise_velocity = cruise_velocity
        self._dt = dt
        self.render_mode = render_mode
        self.__screen_width = 800
        self.__screen_height = 400
        self.__screen_title = 'research'
        self.__screen_fps = fps

        if self.render_mode == "human":
            init_window(self.__screen_width, self.__screen_height, self.__screen_title)
            set_target_fps(self.__screen_fps)

        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self, seed=None):
        """
        Returns
        -------
        observation : array
            The initial observation
        info : dict
            Additional information
        """
        super().reset(seed=seed)
        resistance = - 5.0 * self._cruise_velocity * abs(self._cruise_velocity) - 30.0 * self._cruise_velocity

        self._position_x_wc = 0.0
        self._position_y_wc = 0.0
        self._velocity_x = self._cruise_velocity
        self._velocity_y = 0.0
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._time = 0.0

        self._throttle = 0.0
        self._brake = 0.0
        self._delta = 0.0
        self._theta = 0.0
        self._omega = 0.0

        self.__rewards = 0.0

        self._slip()
        for _ in range(4):
            self._physics()
        self._throttle = 0.0

        return self._observation(), self._get_info()
    
    def step(self, action):
        """
        Parameters
        ----------
        action : int
            The action to take
        
        Returns
        -------
        observation : array
            The observation after the action is taken
        reward : float
            The reward after the action is taken
        done : bool
            If the episode is done
        info : dict
            Additional information
        """
        self._action(action)
        self._physics()

        observation = self._observation()
        terminated = self._terminated(observation)
        truncated = self._truncated(observation)
        self._reward(observation, terminated)
        return observation, self.__rewards, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            if window_should_close():
                close_window()
                return False
            else:
                self._render()

    def _action(self, action):
        chosen = self._get_action_space(action)
        if chosen[0] == 0.0:
            if self._throttle >= 0.2:
                self._throttle -= 0.2
        if chosen[0] == 1.0:
            if self._throttle <= 0.8:
                self._throttle += 0.2
        if chosen[1] == 0.0:
            if self._brake >= 0.2:
                self._brake -= 0.2
        if chosen[1] == 1.0:
            if self._brake <= 0.8:
                self._brake += 0.2
        if chosen[2] == 1.0:
            if self._delta <= 0.5:
                self._delta += 0.1
        if chosen[2] == -1.0:
            if self._delta >= -0.5:
                self._delta -= 0.1

    def _physics(self):
        self.__f_x = 0.0
        self.__f_y = 0.0
        self.__a_y = 0.0

        drag_x = - 5.0 * self._velocity_x * abs(self._velocity_x)
        drag_y = - 5.0 * self._velocity_y * abs(self._velocity_y)
        roll_x = - 30.0 * self._velocity_x
        roll_y = - 30.0 * self._velocity_y

        beta = atan(tan(self._delta) * 0.5)
        self.__f_x = self._throttle * 4000 - self._brake * 10000
        if self._delta:
            r = 2.6 / tan(beta)
            self.__a_y = self._velocity_x ** 2 / r
            self.__f_y = self.__a_y * 1500.0
            self._omega = self._velocity_x / r
        else:
            self._omega /= 1.1

        if self.__f_y > 15000:
            self._omega *= 2.5

        self._acceleration_x = (drag_x + roll_x + self.__f_x) / 1500.0
        self._acceleration_y = (drag_y + roll_y + self.__f_y * sin(beta)) / 1500.0
        self._velocity_x += self._acceleration_x * self._dt
        self._velocity_y += self._acceleration_y * self._dt

        self._theta = (self._theta + self._omega * self._dt) % (2 * pi)
        self._time += self._dt     
        
        self._position_x_wc += (cos(self._theta) * self._velocity_x - sin(self._theta) * self._velocity_y) * self._dt
        self._position_y_wc += (sin(self._theta) * self._velocity_x + cos(self._theta) * self._velocity_y) * self._dt

    def _slip(self):
        self._omega = uniform(-3, 3)

    def _render(self):
        begin_drawing()
        clear_background(RAYWHITE)
        
        # world coordinates
        draw_text("pos:", 20, 20, 20, BLACK)
        draw_text("vel:", 20, 50, 20, BLACK)
        draw_text("acc:", 20, 80, 20, BLACK)
        draw_text("rad:", 20, 110, 20, BLACK)

        draw_text(str(round(self._position_x_wc, 2)), 80, 20, 20, BLACK)
        draw_text(str(round(self._position_y_wc, 2)), 200, 20, 20, BLACK)

        draw_text(str(round(self._velocity_x, 2)), 80, 50, 20, BLACK)
        draw_text(str(round(self._velocity_y, 2)), 200, 50, 20, BLACK)
        draw_text(str(round(self._acceleration_x, 2)), 80, 80, 20, BLACK)
        draw_text(str(round(self._acceleration_y, 2)), 200, 80, 20, BLACK)
        draw_text(str(round(self._theta, 2)), 80, 110, 20, BLACK)
        draw_text(str(round(self._omega, 2)), 200, 110, 20, BLACK)

        draw_text(str(round(self._throttle, 2)), 200, 160, 20, BLACK)
        draw_text(str(round(self._brake, 2)), 200, 180, 20, BLACK)
        draw_text(str(round(self._delta, 2)), 200, 200, 20, BLACK)

        # timing
        draw_text("time:", 20, self.__screen_height - 40, 20, BLACK)
        draw_text(str(round(self._time, 2)), 80, self.__screen_height - 40, 20, BLACK)

        # compass
        compass_x = self.__screen_width // 4 * 3
        compass_y = self.__screen_height // 2
        radius = self.__screen_height // 10
        angle = self._theta - 1.5708
        draw_line(compass_x, compass_y - radius * 3, compass_x, compass_y + radius * 3, BLACK)
        draw_line(compass_x - radius * 3, compass_y, compass_x + radius * 3, compass_y, BLACK)
        draw_triangle(
            Vector2(compass_x + radius * 2 * cos(angle), compass_y + radius * 2 * sin(angle)), 
            Vector2(compass_x + radius * cos(angle + 4 * pi / 3), compass_y + radius * sin(angle + 4 * pi / 3)),
            Vector2(compass_x + radius * cos(angle + 2 * pi / 3), compass_y + radius * sin(angle + 2 * pi / 3)), 
            BLACK
        )
        end_drawing()

    def _observation(self):
        return {
            'position_x': np.array([round(self._position_x_wc, 3)]),
            'position_y': np.array([round(self._position_y_wc, 3)]),
            'velocity_x': np.array([round(self._velocity_x, 3)]),
            'velocity_y': np.array([round(self._velocity_y, 3)]),
            'angular_velocity': np.array([round(self._omega, 3)]),
            'angle': np.array([round(self._theta, 3)])
        }

    def _terminated(self, observation):
        if observation['position_y'] > 4.8 or observation['position_y'] < -4.8:
            return True
        elif self._time >= 30.0:
            return True
        return False

    def _truncated(self, observation):
        if observation['position_x'] >= 300.0:
            return True
        return False

    def _reward(self, observation, terminated):
        self.__rewards = self.__rewards - 10.0
        if terminated:
            self.__rewards -= 100.0
        # rewards position y
        if -1.6 < abs(observation['position_y']) < 1.6:
            self.__rewards += 60.0
        elif -3.2 < abs(observation['position_y']) < 3.2:
            self.__rewards += 40.0
        elif -4.8 < abs(observation['position_y']) < 4.8:
            self.__rewards += 20.0
        # rewards position z
        if -1.0 < abs(observation['angle']) < 1.0:
            self.__rewards += 60.0
        elif -2.0 < abs(observation['angle']) < 2.0:
            self.__rewards += 30.0
        # rewards velocity z
        if -0.1 < abs(observation['angular_velocity']) < 0.1:
            self.__rewards += 60.0
        elif -0.2 < abs(observation['angular_velocity']) < 0.2:
            self.__rewards += 40.0
        elif -0.4 < abs(observation['angular_velocity']) < 0.4:
            self.__rewards += 20.0
        elif -0.6 < abs(observation['angular_velocity']) < 0.6:
            self.__rewards += 10.0
        # rewards direction
        if observation['position_x'] > 0.0:
            self.__rewards += 10.0
        else:
            self.__rewards -= 50.0

    def _get_observation_space(self):
        return spaces.Dict({
            'position_x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype='float64'),
            'position_y': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype='float64'),
            'velocity_x': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype='float64'),
            'velocity_y': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype='float64'),
            'angular_velocity': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype='float64'),
            'angle': spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype='float64')
        })

    def _get_action_space(self, action):
        actions = [
            # [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            # [1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
        ]
        return actions[action]

    def _get_info(self):
        return {
            "distance": 300 - self._position_x_wc
        }