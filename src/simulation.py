from config import *
import numpy as np
from vehicle_model import Vehicle
import matplotlib.pyplot as plt
import gameplotlib as gplt
from typing import Callable, Iterable
import time
import copy
import os

class Box():

    def __init__(self, low: np.ndarray, high: np.ndarray, dtype: str = 'float32') -> None:

        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape")

        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        self.dtype = self.low.dtype

        self.shape = low.shape


class Discrete():

    def __init__(self, n: int, shape, dtype: str = 'float32') -> None: # <- the shape is the shape of each discrete element of the space

        self.shape = shape
        self.n = n
        self.dtype = np.zeros(1).astype(dtype).dtype


class Obstacle():

    def __init__(self, x_1: float, y_1: float, x_2: float, y_2: float) -> None:
        """
        a simple line segment obstacle defined by two endpoints (x_1, y_1) and (x_2, y_2)
        """
        self.x_1 = x_1
        self.y_1 = y_1

        self.x_2 = x_2
        self.y_2 = y_2

        self._update()


    def _update(self) -> None:

        self.low_x  = min(self.x_1, self.x_2)
        self.high_x = max(self.x_1, self.x_2)


    def y(self, x: float) -> None:

        self._update()

        if x < self.low_x:
            return np.nan

        if x > self.high_x:
            return np.nan

        if self.high_x == self.low_x:
            return np.nan

        return (self.y_2 - self.y_1)/(self.x_2 - self.x_1) * (x - self.x_1) + self.y_1

    def get_line(self) -> None:
        
        self._update()

        return {"x": [self.low_x, self.high_x], "y": [self.y(self.low_x), self.y(self.high_x)]}

class Environment:


    def __init__(self, dt: float, x_bounds: list[float], y_bounds: list[float],
                 memory_size: int = 10, sensor_names: list[str] = None,
                 obstacle_count: int = 2, obstacle_size: float = 0.5,
                 seed: int = None, render_type: str = None, nn_control: Callable = None) -> None:
        """
        :param dt: step size of the simulation
        :param x_bounds: a list [x_min, x_max]
        :param y_bounds: a list [y_min, y_max]
        :param memory_size: how large the model's memory is (NOT THE PPO EPISODE BUFFER).
        :param sensor_names: a list with the names ('front', 'left', 'back', 'right') of the sensors to be active in the simulation.
        :param obstacle_count: how many obstacles
        :param obstacle size: how big the obstacles are in relation to the space.
        :param seed: a random initial seed for the environment generation
        :param render type: whether to render with pygame ("human") or not at all (None). Use None for training.
        :param nn_control: if "human" render_type is used, but actor-critic control is desired, pass the actor critic model here to give it control. Otherwise, human control will be used
        """
        self.sensor_names = sensor_names

        if sensor_names is None:
            self.observation_count = 1
        else:
            self.observation_count = 1 + len(self.sensor_names)

        self.memory_size = memory_size
        self.obstacle_count = obstacle_count
        self.obstacle_size = obstacle_size

        self.observation_space = Box(low=np.array([0] * self.memory_size * self.observation_count),
                                     high=np.array([np.inf] * self.memory_size * self.observation_count))
        self.action_space = Box(low=np.array([-1,-1]), high=np.array([1,1]))

        self.dt = dt
        self.x_min = x_bounds[0]
        self.x_max = x_bounds[1]

        self.y_min = y_bounds[0]
        self.y_max = y_bounds[1]

        self.render_type = render_type
        self.nn_control = nn_control

        self.sensor_colors = {"front": (255, 0, 255), "left": (0, 0, 255), "right": (255, 0, 0), "back": (0, 255, 0)}
        
        self.reset(seed)


    def reset(self, seed: int = None) -> np.ndarray:
        """
        reset the simulation environment
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()


        # initial_x = rng.uniform(self.x_min, self.x_max)
        # initial_y = rng.uniform(self.y_min, self.y_max)

        # target_x = rng.uniform(self.x_min, self.x_max)
        # target_y = rng.uniform(self.y_min, self.y_max)
        initial_x = (self.x_max - self.x_min) * 0.1 + self.x_min
        initial_y = (self.y_max - self.y_min) * 0.1 + self.y_min
        target_x = (self.x_max - self.x_min) * 0.9 + self.x_min
        target_y = (self.x_max - self.x_min) * 0.9 + self.y_min

        self.car = Vehicle(r = 0.15, P=[initial_x, initial_y], F=(target_x, target_y), theta=0, throttle=0, steer=0)

        self.current_state = np.zeros(self.memory_size * self.observation_count, dtype=np_dtype)
        self.memory_empty = True
        self.time_elapsed = 0

        # self.obstacles = self._create_obstacles(seed)
        self.obstacles = self._create_obstacle_course()
        self.closest_distance = {}
        self.closest_obstacle = {}
        self.intersections_x = {}
        self.intersections_y = {}

        self.time_elapsed = 0
        self.episode_return = 0
        self.step_count = 0
        self.steer_history = []

        return self._update_state()


    def step(self, action: np.ndarray) -> tuple:
        """
        take an euler step with the given actions
        """

        old_state = copy.deepcopy(self.current_state)

        self.car.throttle = action[0]
        self.car.steer = action[1]
        self.car.step(dt=self.dt)

        new_state = self._update_state()

        reward = self._get_reward(new_state, old_state, action)

        self.step_count += 1
        self.episode_return += reward
        self.steer_history.append(action[1])

        terminal_flag = self._update_terminal_flag(new_state)

        self.time_elapsed += self.dt


        return new_state, reward, terminal_flag, 0 # <- PPO assumes step returns a 4th thing but ignores it


    def render(self) -> None:

        if self.render_type == "human":

            self._set_human_render()

            self.fig.show()


    def _set_human_render(self) -> None:

        self.reset()

        # setup figure
        self.fig = gplt.Figure(fps = 1/self.dt)    
    
        # setup box bounds
        self.fig.plot([self.x_min, self.x_min, self.x_max, self.x_max, self.x_min],
                      [self.y_max, self.y_min, self.y_min, self.y_max, self.y_max],
                      name='box', color=(255,255,255))

        # setup console
        self.fig.text("REWARD", self.x_min, self.y_min, name="reward text", alignment="topleft", font_size=20)
        self.fig.text("AVG_STEER", self.x_min, self.y_max, name="steer text", alignment="topleft", font_size=20)

        # start plot
        lines = self.car.get_lines()

        for i, line in enumerate(lines):
            x = line["x"]
            y = line["y"]

            if i == 0:
                color = (255, 255, 255)

            elif i == len(lines) - 1 or i == len(lines) - 2:
                color = (240, 235, 69)

            else:
                color = (150, 150, 150)

            self.fig.plot(x, y, name=f'carline_{i}', color=color)

        for i, obstacle in enumerate(self.obstacles):

            line = obstacle.get_line()
            x = line["x"]
            y = line["y"]

            color = (255, 255, 255)
            for name, closest_obstacle in self.closest_obstacle.items():

                if obstacle == closest_obstacle:

                    color = self.sensor_colors[name]

            self.fig.plot(x, y, name=f"obstacle_{i}", color=color)
        
        if self.sensor_names is not None:
            for name in self.sensor_names:
                self.fig.scatter(self.intersections_x[name], self.intersections_y[name], name=f"intersections_{name}", color=(0,0,0), follow=False)

        def _on_loop_f(figure, user_variables: dict, pressed_keys: list) -> None:


            if self.nn_control is None:
                action = self._get_human_action(pressed_keys)
            else:
                action = self.nn_control(self.current_state)

            # step
            new_state, reward, terminal_flag, _ = self.step(action)

            if terminal_flag:
                time.sleep(0.3)
                self.fig.clear_variables()
                self.fig.text(f"REACHED TERMINAL STATE WITH {self.episode_return:.5f} | {self.step_count}", 0, 0, font_size=30)
                self.fig.re_render()
                self.fig.set_hang()

            # update text
            user_variables['reward text'].text = f"REWARD: {reward:+.5f} | TOTAL RETURN: {self.episode_return:.5f}"
            user_variables['steer text'].text  = f"AVG STEER: {np.array(self.steer_history).mean()}"

            # update lines
            lines = self.car.get_lines()

            for i, line in enumerate(lines):
                x = line["x"]
                y = line["y"]

                if i == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                user_variables[f'carline_{i}'].x = x
                user_variables[f'carline_{i}'].y = y

            for i, obstacle in enumerate(self.obstacles):

                color = (255, 255, 255)
                for name, closest_obstacle in self.closest_obstacle.items():

                    if obstacle == closest_obstacle:

                        color = self.sensor_colors[name]

                user_variables[f'obstacle_{i}'].color = color

            if self.sensor_names is not None:
                for name in self.sensor_names:
                    user_variables[f'intersections_{name}'].x = self.intersections_x[name]
                    user_variables[f'intersections_{name}'].y = self.intersections_y[name]

        self.fig.set_on_loop(_on_loop_f)


    def _get_human_action(self, pressed_keys) -> np.ndarray:

        steer = 0.0
        throttle = 0.0

        if pressed_keys[gplt.K_LEFT]:
            steer = -1.0

        if pressed_keys[gplt.K_RIGHT]:
            steer = 1.0

        if pressed_keys[gplt.K_UP]:
            throttle = 1.0

        if pressed_keys[gplt.K_DOWN]:
            throttle = -1.0

        action = np.array([throttle, steer], dtype=np_dtype)

        return action

    def _create_obstacles(self, seed: int = None) -> list[Obstacle]:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        self.x_span = self.x_max - self.x_min
        self.y_span = self.y_max - self.y_min

        obstacles = []

        for i in range(self.obstacle_count):

            x_1 = rng.uniform(self.x_min, self.x_max)
            y_1 = rng.uniform(self.y_min, self.y_max)

            x_2 = x_1 + rng.uniform(-self.obstacle_size*self.x_span, self.obstacle_size*self.x_span)
            y_2 = y_1 + rng.uniform(-self.obstacle_size*self.y_span, self.obstacle_size*self.y_span)

            obstacles.append(Obstacle(x_1, y_1, x_2, y_2))

        return obstacles

    def _create_obstacle_course(self) -> list[Obstacle]:
        obstacles = []
        x1 = (self.x_max - self.x_min) * 0.25 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.5 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.75 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.5 + self.y_max
        obstacles.append(Obstacle(x1, y1, x2, y2))
        
        x1 = (self.x_max - self.x_min) * 0.75 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.5 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.75 + self.x_min
        y2 = (self.y_max - self.y_min) * 1 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        
        return obstacles

    def _get_reward(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

        """
        the state is a memory buffer of size self.memory_size * self.observation_count that holds the past 'self.memory_size' measurements for
        each of the 'self.observation_count' observations.

        Example: Let self.observation_count = 2 so we can use 1 target distance measurement and 1 proximity sensor measurement.
                 Then, let self.memory_size = 3, which means at step n, we have a state array of the form:

                 state = [target_distance_n, target_distance_(n-1), target_distance_(n-2), sensor_measurement_n, sensor_measurement_(n-1), sensor_measurement_(n-2)]

                 The latest measurement (measurement_n) for each type of observation is always at the beginning of its respective section and can be accessed through its respective 'base index' ('bi')
        """

        # proximity reward
        w_p = 2
        r_p = 0

        for i in reversed(range(self.d_bi + 1, self.d_bi + self.memory_size)):
            r_p += (self.current_state[i] - self.current_state[i-1])

        self._update_terminal_flag(new_state)

        # NOTE: r_p is used in some of the remaning rewards as a 'scaling factor' to make sure all rewards are of approximately the same scale

        # crash penalty
        w_c = -30
        r_c = int(self.bad_terminal)

        # goal reward
        w_g = 30
        r_g = int(self.good_terminal)

        # throttle reward
        w_t = 1.1
        if action[0] < 0:
            r_t = action[0] * abs(r_p) # <- penalize backward motion,
        else:
            r_t = 0 # <- nothing for forward motion

        # steer reward
        if len(self.steer_history) > 1:
            avg_steer = np.array(self.steer_history).mean() # <- calculate average steer
        else:
            avg_steer = 0

        w_s = 0.5
        r_s = (-abs(avg_steer)) * abs(r_p) # <- penalize non-zero average steer


        return r_p*w_p + r_g*w_g + r_c*w_c + r_t*w_t + r_s*w_s


    def _update_state(self) -> np.ndarray:
        """
        calculate the distance to the target and the sensor data
        """

        # assign base index for distance measurement
        self.d_bi = 0 # <- distance measurement base index 'bi' (for use in the 'self.current_state' memory buffer

        # get target distance measurement
        d = self.car.F - self.car.P
        d = np.sqrt(d.T @ d)

        if self.sensor_names is not None:

            # assign base indices
            self.s_bi = {} # <- sensor base indices 'bi's (for use in the 'self.current_state' memory buffer)

            for i, name in enumerate(self.sensor_names):
                self.s_bi[name] = self.memory_size * (i+1)

            # get sensor measurements
            s = {}
            for name in self.sensor_names:
                s[name] = self._get_sensor_reading(sensor_name=name)


        # update state buffer
        if self.memory_empty: # <- intialize buffer to the first observations

            for i in range(self.memory_size):
                self.current_state[self.d_bi + i] = d

            if self.sensor_names is not None:
                for name in self.sensor_names:
                    for i in range(self.memory_size):
                        self.current_state[self.s_bi[name] + i] = s[name]

            self.memory_empty = False

        else: # <- update with just the newest observation

            self.current_state = np.roll(self.current_state, 1) # <- shift elements 1 index over to the right

            self.current_state[self.d_bi] = d  # <- insert new distance

            if self.sensor_names is not None:
                for name in self.sensor_names: # <- for each sensor
                    self.current_state[self.s_bi[name]] = s[name] # <- insert new sensor measurement

        return self.current_state

    def _update_terminal_flag(self, state: np.ndarray) -> bool:
        """
        calculate whether the state is terminal or not, and if terminal, whether it is good terminal or bad terminal
        """

        collision_tolerance = 0.1
        target_tolerance = 0.1
        
        # check for collision
        if self.sensor_names is not None:
            self.bad_terminal = False
            for name in self.sensor_names:
                self.bad_terminal += state[self.s_bi[name]] < collision_tolerance # <- this is basically an OR operation, checking if any of the sensors report a collision
        else:
            self.bad_terminal = False # <- if no sensors, no collisions can be detected

        # check for target reached
        self.good_terminal = state[self.d_bi] < target_tolerance

        self.is_terminal = 1 * ((self.good_terminal) or (self.bad_terminal)) 

        return self.is_terminal

    def _get_sensor_reading(self, sensor_name: str) -> float:

        """
        basic 'line of sight' intersection calculations
        """
        
        # select correct sensor by getting its coordinates
        if sensor_name == "front":
            x_s = self.car.P_f[0]
            y_s = self.car.P_f[1]
        elif sensor_name == "left":
            x_s = self.car.P_l[0]
            y_s = self.car.P_l[1]
        elif sensor_name == "back":
            x_s = self.car.P_b[0]
            y_s = self.car.P_b[1]
        elif sensor_name == "right":
            x_s = self.car.P_r[0]
            y_s = self.car.P_r[1]
        else:
            raise ValueError("invalid sensor name")

        # get car coordinates
        x_P = self.car.P[0]
        y_P = self.car.P[1]


        # reset measurements
        self.closest_distance[sensor_name] = MAX_SENSOR_MEASUREMENT
        self.closest_obstacle[sensor_name] = None

        self.intersections_x[sensor_name] = []
        self.intersections_y[sensor_name] = []

        for obstacle in self.obstacles:

            x_1 = obstacle.x_1
            y_1 = obstacle.y_1

            x_2 = obstacle.x_2
            y_2 = obstacle.y_2


            m_o = (y_2 - y_1)/(x_2 - x_1 + np.finfo(float).eps) # <- the slope of the obstacle line
            m_s = (y_s - y_P)/(x_s - x_P + np.finfo(float).eps) # <- the slope of the sensor 'line of sight'

            # calculate point of intersection between sensor 'line of sight' and obstacle line
            x_intersection = (m_o * x_1 - m_s * x_P + y_P - y_1)/(m_o - m_s)
            y_intersection = obstacle.y(x_intersection)

            self.intersections_x[sensor_name].append(x_intersection)
            self.intersections_y[sensor_name].append(y_intersection)

            # check if it is a valid intersection
            intersects_obstacle =  (x_intersection > min(x_1, x_2)) and (x_intersection < max(x_1, x_2))
            on_correct_side = ((min(x_P, x_s) == x_P) and (x_intersection >= x_s)) or ((min(x_P, x_s) == x_s) and (x_intersection <= x_s))

            # calculate sensor measurement
            if intersects_obstacle and on_correct_side:

                distance = np.sqrt( (x_s - x_intersection)**2 + (y_s - y_intersection)**2 )

                if distance < self.closest_distance[sensor_name]:
                    self.closest_distance[sensor_name] = distance
                    self.closest_obstacle[sensor_name] = obstacle

        return self.closest_distance[sensor_name] # <- return sensor measurement


    def take_snapshot(self, snapshot_id: int = None) -> None:
        """
        take a snapshot of the current state and save it to a snapshots directory
        """

        fig, ax = plt.subplots()    

        ax.plot([self.x_min, self.x_min, self.x_max, self.x_max, self.x_min],
                [self.y_max, self.y_min, self.y_min, self.y_max, self.y_max], color="b")

        lines = self.car.get_lines()

        for i, line in enumerate(lines):
            x = line["x"]
            y = line["y"]

            if i == 0:
                color = "r"
            else:
                color = "b"

            ax.plot(x, y, color=color)

        if snapshot_id is None:
            snapshot_id = int(time.time())
            

        os.makedirs("snapshots", exist_ok=True)
        plt.axis("equal")
        plt.xlim(self.x_min-1, self.x_max+1)
        plt.ylim(self.y_min-1, self.y_max+1)
        plt.title(f"{self.time_elapsed:.2f}")
        plt.savefig(f"snapshots/snapshot_{snapshot_id}.jpg")

        plt.close("all")


if __name__ == '__main__':

    while True:
        env = Environment(dt=dt, x_bounds=x_bounds, y_bounds=y_bounds, memory_size=memory_size, sensor_names=sensor_names,
                          obstacle_count=obstacle_count, obstacle_size=obstacle_size, seed=seed, render_type="human")
        env.render()
