from config import *
import numpy as np

from vehicle_model import Vehicle
import matplotlib.pyplot as plt
import gameplotlib as gplt
from typing import Callable, Iterable
import time
import copy
import os


def normal_2d(x_1: float, y_1: float, x_2: float, y_2: float) -> tuple[float]:
    """
    returns the components x_n, y_n of the vector normal to the given line segment
    """

    dx = x_2 - x_1
    dy = y_2 - y_1

    normal_direction = np.array([-dy, dx])

    norm = np.linalg.norm(normal_direction)
    if norm == 0:
        raise ValueError("Zero-length segment; normal is undefined.")
    normal = normal_direction / norm

    return normal[0], normal[1]



class Box():
    """
    a class necessary to satisfy the OpenAI SpinningUp API
    """

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:

        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape")

        self.low = low.astype(np_dtype)
        self.high = high.astype(np_dtype)
        self.dtype = self.low.dtype

        self.shape = low.shape



class Discrete():
    """
    a class necessary to satisfy the OpenAI SpinningUp API
    """

    def __init__(self, n: int, shape) -> None: # <- the shape is the shape of each discrete element of the space

        self.shape = shape
        self.n = n
        self.dtype = np.zeros(1).astype(np.dtype).dtype



class Obstacle():
    """
    a simple line segment obstacle defined by two endpoints (x_1, y_1) and (x_2, y_2)
    """

    def __init__(self, x_1: float, y_1: float, x_2: float, y_2: float) -> None:

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
                 memory_size: int = 10, sensor_names: list[str] = None, reward_function: int = 5,
                 obstacle_types: list[str] = None, obstacle_proportions: list[float] = None, obstacle_configs: list[dict] = None,
                 seed: int = None, render_type: str = None, nn_control: Callable = None) -> None:
        """
        :param dt: step size of the simulation
        :param x_bounds: a list [x_min, x_max] from which the bounds of the domain will be randomly generated
        :param y_bounds: a list [y_min, y_max] from which the bounds of the domain will be randomly generated
        :param memory_size: how large the model's memory is (NOT THE PPO EPISODE BUFFER).
        :param sensor_names: a list with the names ('front', 'left', 'back', 'right') of the sensors to be active in the simulation.
        :param obstacle_types: what obstacle types you want to use (ex: ["random", "track", "spiral", "school"])
        :param obstacle_proportions: what proportions do each of the obstacle types get when randomly generating (must be of same length as obstacle_types). Defaults to equal proportions.
        :param obstacle_configs: a list of dicts containing any options to pass to the selected obstacle generators
        :param seed: a random initial seed for the environment generation
        :param render type: whether to render with pygame ("human") or not at all (None). Use None for training.
        :param nn_control: if "human" render_type is used, but actor-critic control is desired, pass the actor critic model here to give it control. Otherwise, human control will be used
        """

        self.reward_function = reward_function

        self.sensor_names = sensor_names

        if sensor_names is None:
            self.observation_count = 1
        else:
            self.observation_count = 1 + len(self.sensor_names)

        self.memory_size = memory_size

        self.obstacle_types = obstacle_types
        self.obstacle_proportions = obstacle_proportions
        self.obstacle_configs = obstacle_configs

        if self.obstacle_types is not None:
            if self.obstacle_proportions is None:
                self.obstacle_proportions = [1/len(self.obstacle_types)] * len(self.obstacle_types)# <- default to equal proportions
            else:
                assert(len(self.obstacle_proportions) == len(self.obstacle_types))

        assert(len(self.obstacle_configs) == len(self.obstacle_types))

        self.observation_space = Box(low=np.array([0] * self.memory_size * self.observation_count),
                                     high=np.array([np.inf] * self.memory_size * self.observation_count))
        self.action_space = Box(low=np.array([-1,-1]), high=np.array([1,1]))

        self.dt = dt

        self.render_type = render_type
        self.nn_control = nn_control

        self.sensor_colors = {"front": (255, 0, 255), "left": (0, 0, 255), "right": (255, 0, 0), "back": (0, 255, 0)}
        
        self.seed = seed
        self.reset(seed)


    def reset(self, seed: int = None) -> np.ndarray:
        """
        reset the simulation environment
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        if self.obstacle_types is not None:
            index = rng.choice(len(self.obstacle_types), p=self.obstacle_proportions)
            self.obstacle_type = self.obstacle_types[index]
            self.obstacle_config = self.obstacle_configs[index]
        else:
            self.obstacle_type = None
            self.obstacle_config = None

        self.bounce_guard = 0

        self.x_min = -rng.uniform(x_bounds[0], x_bounds[1]) 
        self.x_max =  rng.uniform(x_bounds[0], x_bounds[1])

        self.y_min = -rng.uniform(y_bounds[0], y_bounds[1]) 
        self.y_max =  rng.uniform(y_bounds[0], y_bounds[1])

        if self.obstacle_type == "track":
            initial_x = self.x_min
            initial_y = rng.uniform(self.y_min, self.y_max)
            target_x  = self.x_max
            target_y  = rng.uniform(self.y_min, self.y_max)

        elif self.obstacle_type == "spiral":
            target_x  = 0
            target_y  = 0
            initial_x = rng.uniform(self.x_min, self.x_max)
            initial_y = rng.uniform(self.y_min, self.y_max)

        elif self.obstacle_type == "school":
            self.x_min = -x_bounds[0]
            self.x_max = x_bounds[1]
            self.y_min = -y_bounds[0]
            self.y_max = y_bounds[1]
            
            if self.obstacle_config["version"] == "turn":

                initial_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                initial_y = (self.y_max - self.y_min) * 0.05 + self.y_min
                target_x = (self.x_max - self.x_min) * 0.85 + self.x_min
                target_y = (self.x_max - self.x_min) * 0.95 + self.y_min

            elif self.obstacle_config["version"] == "setups":
            
                # Hallway
                setups = []
                initial_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                initial_y = (self.y_max - self.y_min) * 0.05 + self.y_min
                target_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                target_y = (self.y_max - self.y_min) * 0.95 + self.y_min
                setups.append([initial_x, initial_y, target_x, target_y])
                
                initial_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                initial_y = (self.y_max - self.y_min) * 0.95 + self.y_min
                target_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                target_y = (self.y_max - self.y_min) * 0.05 + self.y_min
                setups.append([initial_x, initial_y, target_x, target_y])
                
                initial_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                initial_y = (self.y_max - self.y_min) * 0.85 + self.y_min
                target_x = (self.x_max - self.x_min) * 0.95 + self.x_min
                target_y = (self.y_max - self.y_min) * 0.85 + self.y_min
                setups.append([initial_x, initial_y, target_x, target_y])
                
                initial_x = (self.x_max - self.x_min) * 0.95 + self.x_min
                initial_y = (self.y_max - self.y_min) * 0.85 + self.y_min
                target_x = (self.x_max - self.x_min) * 0.05 + self.x_min
                target_y = (self.y_max - self.y_min) * 0.85 + self.y_min
                setups.append([initial_x, initial_y, target_x, target_y])
                
                random_setup = rng.choice(setups)
                initial_x, initial_y, target_x, target_y = random_setup

            elif self.obstacle_config["version"] == "area":
                
                initial_x = (self.x_max - self.x_min) * 0.8 + self.x_min
                initial_y = (self.y_max - self.y_min) * 0.2 + self.y_min
                target_x = (self.x_max - self.x_min) * 0.95 + self.x_min
                target_y = (self.x_max - self.x_min) * 0.35 + self.y_min

            else:

                raise ValueError("Invalid school obstacles version")

        elif self.obstacle_type == "random" or self.obstacle_type == None:

            initial_x = rng.uniform(self.x_min, self.x_max)
            initial_y = rng.uniform(self.y_min, self.y_max)
            target_x  = rng.uniform(self.x_min, self.x_max)
            target_y  = rng.uniform(self.y_min, self.y_max)

        else:
            
            raise ValueError(f"invalid obstacle type: {self.obstacle_type}")

        # save for future use
        self.target_x = target_x
        self.target_y = target_y
        self.initial_x = initial_x
        self.initial_y = initial_y

        # instantiate car
        self.car = Vehicle(r = 0.15, P=[initial_x, initial_y], F=(target_x, target_y), theta=0, throttle=0, steer=0)

        # further setup
        self.collision_tolerance = self.car.r
        self.target_tolerance = 0.1
        self.collided = False

        self.current_state = np.zeros(self.memory_size * self.observation_count, dtype=np_dtype)
        self.memory_empty = True
        self.time_elapsed = 0

        self.obstacles = self._create_obstacles(seed)
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

        self.collided = False
        self.bounce_guard += 1
        if self.bounce_guard > 10:
            self._check_for_collision()

        new_state = self._update_state()

        self._update_terminal_flag(new_state)

        reward = self._get_reward(new_state, old_state, action)

        self.step_count += 1
        self.episode_return += reward
        self.steer_history.append(action[1])


        self.time_elapsed += self.dt


        return new_state, reward, self.is_terminal, 0 # <- PPO assumes step returns a 4th thing but ignores it


    def _create_obstacles(self, seed: int = None) -> list[Obstacle]:

        if self.obstacle_type is None:
            return None
        elif self.obstacle_type == "track":
            return self._track_obstacles(seed)
        elif self.obstacle_type == "spiral":
            return self._spiral_obstacles()
        elif self.obstacle_type == "random":
            return self._random_obstacles(seed)
        elif self.obstacle_type == "school":
            return self._school_obstacles()
        else:
            raise ValueError(f"unknown obstacle type {self.obstacle_type}")


    def _school_obstacles(self) -> list[Obstacle]:

        obstacles = []

        # environment bounds
        x1 = self.x_min
        y1 = self.y_max
        x2 = self.x_max
        y2 = self.y_max
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = self.x_max
        y1 = self.y_min
        x2 = self.x_max + 0.000001
        y2 = self.y_max
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = self.x_min
        y1 = self.y_min
        x2 = self.x_max
        y2 = self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = self.x_min
        y1 = self.y_min
        x2 = self.x_min + 0.000001
        y2 = self.y_max
        obstacles.append(Obstacle(x1, y1, x2, y2))

        # Hallway obstacle 1 - Another classroom
        x1 = (self.x_max - self.x_min) * 0.1 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.4 + self.y_min
        x2 = (self.x_max - self.x_min) * 1 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.4 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.1 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.4 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.1000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.8 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.1 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.8 + self.y_min
        x2 = (self.x_max - self.x_min) * 1 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.8 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))

        # Hallway obstacle 2 - More offices
        x1 = (self.x_max - self.x_min) * 0.1 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.9 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.1000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 1 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.1 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.9 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.8 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.9 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.8 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.9 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.8000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 1 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))

        # Hallway obstacle 3 - More offices
        x1 = (self.x_max - self.x_min) * 0.9 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.9 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.9000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 1 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.9 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.9 + self.y_min
        x2 = (self.x_max - self.x_min) * 1 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.9 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))

        # Classroom obstacle 1 - Walls
        x1 = (self.x_max - self.x_min) * 0.1 + self.x_min
        y1 = (self.y_max - self.y_min) * 0 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.1000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.32 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.6 + self.x_min
        y1 = (self.y_max - self.y_min) * 0 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.6000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.4 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))

        # Classroom obstacle 2 - Professor's Desk
        x1 = (self.x_max - self.x_min) * 0.45 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.375 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.375 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.325 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.375 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.45 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.325 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.325 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.45 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.325 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.45000001 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.375 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))

        # Classroom obstacle 3 - Tables
        x1 = (self.x_max - self.x_min) * 0.15 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.2875 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.2875 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.15 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.2125 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.2125 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.15 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.1375 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.1375 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))
        x1 = (self.x_max - self.x_min) * 0.15 + self.x_min
        y1 = (self.y_max - self.y_min) * 0.0625 + self.y_min
        x2 = (self.x_max - self.x_min) * 0.55 + self.x_min
        y2 = (self.y_max - self.y_min) * 0.0625 + self.y_min
        obstacles.append(Obstacle(x1, y1, x2, y2))

        return obstacles


    def _track_obstacles(self, seed: int = None) -> list[Obstacle]:

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        x_car = self.car.P[0] 
        y_car = self.car.P[1]

        x_target = self.car.F[0]
        y_target = self.car.F[1]

        segment_count = self.obstacle_config["segment_count"]

        obstacles = []

        path_span = abs(x_car - x_target)
        segment_size = path_span/(segment_count + 1)

        xs = np.linspace(x_car, x_target, segment_count+2)
        noise = np.hstack([np.zeros(1), rng.normal(loc=0, scale=segment_size/10, size=segment_count), np.zeros(1)]) # <- no noise at the ends
        xs = xs + noise
        ys = np.linspace(y_car, y_target, segment_count+2)
        noise = np.hstack([np.zeros(1), rng.normal(loc=0, scale=path_span/20, size=segment_count), np.zeros(1)]) # <- no noise at the ends
        ys = ys + noise

        xs_m = []
        ys_m = []

        for i in range(0, segment_count):

            x_1 = xs[i]
            x_2 = xs[i+1]
            
            y_1 = ys[i]
            y_2 = ys[i+1]
            
            x_n, y_n = normal_2d(x_1, y_1, x_2, y_2)
            x_n, y_n = 2*x_n*self.car.r, 2*y_n*self.car.r

            obstacles.append(Obstacle(x_1+x_n, y_1+y_n, x_2+x_n, y_2+y_n))
            obstacles.append(Obstacle(x_1-x_n, y_1-y_n, x_2-x_n, y_2-y_n))

        return obstacles

    def _spiral_obstacles(self)  -> list[Obstacle]:
        
        self.x_span = self.x_max - self.x_min
        self.y_span = self.y_max - self.y_min

        R = min(self.y_span, self.x_span)/3

        ring_count = self.obstacle_config["ring_count"]
        segment_count = self.obstacle_config["segment_count"]

        obstacles = []
        for i in range(1,ring_count+1):
        
            r = R*(0.3 + i/ring_count)

            theta = (np.linspace(0, 2*np.pi, segment_count*2+1)[:-1] + (i/ring_count)/2*np.pi)
            xs = (r * np.cos(theta)) + self.car.F[0]
            ys = (r * np.sin(theta)) + self.car.F[1]

            xs_1 = xs[:-1:2]
            ys_1 = ys[:-1:2]
            xs_2 = xs[1::2]
            ys_2 = ys[1::2]

            for j in range(segment_count):

                x_1 = xs_1[j]
                x_2 = xs_2[j]
                y_1 = ys_1[j]
                y_2 = ys_2[j]
                
                obstacle = Obstacle(x_1, y_1, x_2, y_2)
                if self._check_distance(obstacle):
                    obstacles.append(obstacle)

        return obstacles


    def _random_obstacles(self, seed: int = None) -> list[Obstacle]:


        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        self.x_span = self.x_max - self.x_min
        self.y_span = self.y_max - self.y_min

        obstacles = []
        obstacle_count = self.obstacle_config["obstacle_count"]
        obstacle_size = self.obstacle_config["obstacle_size"]

        for i in range(obstacle_count):

            x_1 = rng.uniform(self.x_min, self.x_max)
            y_1 = rng.uniform(self.y_min, self.y_max)

            x_2 = x_1 + rng.uniform(-obstacle_size*self.x_span, obstacle_size*self.x_span)
            y_2 = y_1 + rng.uniform(-obstacle_size*self.y_span, obstacle_size*self.y_span)

            obstacle = Obstacle(x_1, y_1, x_2, y_2)
            if self._check_distance(obstacle):
                obstacles.append(obstacle)

        return obstacles

    def _check_distance(self, obstacle) -> bool:
        """
        checks the distance to an obstacle to make sure it does not overlap the car.
        this is used to avoid invalid initial obstacle-car setups.
        """

        x_P = self.car.P[0]
        y_P = self.car.P[1]

        tolerance = self.collision_tolerance*2

        x_1 = obstacle.x_1
        y_1 = obstacle.y_1

        x_2 = obstacle.x_2
        y_2 = obstacle.y_2

        m_o = (y_2 - y_1)/(x_2 - x_1 + np.finfo(float).eps) # <- the slope of the obstacle line
        m_s = -1/(m_o + np.finfo(float).eps) # <- slope of perpendicular line

        x_intersection = (m_o * x_1 - m_s * x_P + y_P - y_1)/(m_o - m_s + np.finfo(float).eps)
        y_intersection = obstacle.y(x_intersection)

        intersects_obstacle = (x_intersection > min(x_1, x_2)) and (x_intersection < max(x_1, x_2))

        is_good = False

        if not intersects_obstacle:

            is_good = True

        elif intersects_obstacle:

            distance = np.sqrt( (x_P - x_intersection)**2 + (y_P - y_intersection)**2 )

            if distance > tolerance:

                is_good = True

        return is_good


    def _get_reward(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

        if self.reward_function == 1:
            return self._get_reward_1(new_state, old_state, action)

        elif self.reward_function == 2:
            return self._get_reward_2(new_state, old_state, action)
        
        elif self.reward_function == 3:
            return self._get_reward_3(new_state, old_state, action)

        elif self.reward_function == 4:
            return self._get_reward_4(new_state, old_state, action)

        elif self.reward_function == 5:
            return self._get_reward_5(new_state, old_state, action)

        elif self.reward_function == 6:
            return self._get_reward_6(new_state, old_state, action)

        else:
            raise ValueError("invalid reward function")


    def _get_reward_1(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

        """
        the state is a memory buffer of size self.memory_size * self.observation_count that holds the past 'self.memory_size' measurements for
        each of the 'self.observation_count' observations.

        Example: Let self.observation_count = 2 so we can use 1 target distance measurement and 1 proximity sensor measurement.
                 Then, let self.memory_size = 3, which means at step n, we have a state array of the form:

                 state = [target_distance_n, target_distance_(n-1), target_distance_(n-2), sensor_measurement_n, sensor_measurement_(n-1), sensor_measurement_(n-2)]

                 The latest measurement (measurement_n) for each type of observation is always at the beginning of its respective section and can be accessed through its respective 'base index' ('bi')
        """

        # proximity reward
        w_p = 5
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

    def _get_reward_2(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

        """
        the state is a memory buffer of size self.memory_size * self.observation_count that holds the past 'self.memory_size' measurements for
        each of the 'self.observation_count' observations.

        Example: Let self.observation_count = 2 so we can use 1 target distance measurement and 1 proximity sensor measurement.
                 Then, let self.memory_size = 3, which means at step n, we have a state array of the form:

                 state = [target_distance_n, target_distance_(n-1), target_distance_(n-2), sensor_measurement_n, sensor_measurement_(n-1), sensor_measurement_(n-2)]

                 The latest measurement (measurement_n) for each type of observation is always at the beginning of its respective section and can be accessed through its respective 'base index' ('bi')
        """

        # proximity reward
        w_p = 1
        r_p = 0

        for i in reversed(range(self.d_bi + 1, self.d_bi + self.memory_size)):
            r_p += (self.current_state[i] - self.current_state[i-1])

        self._update_terminal_flag(new_state)

        # NOTE: r_p is used in some of the remaning rewards as a 'scaling factor' to make sure all rewards are of approximately the same scale

        # crash penalty
        w_c = -2
        r_c = int(self.collided)

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

        w_s = 0.50
        r_s = (-abs(avg_steer)) * abs(r_p) # <- penalize non-zero average steer

        return r_p*w_p + r_g*w_g + r_c*w_c + r_t*w_t + r_s*w_s

    def _get_reward_3(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

        """
        the state is a memory buffer of size self.memory_size * self.observation_count that holds the past 'self.memory_size' measurements for
        each of the 'self.observation_count' observations.

        Example: Let self.observation_count = 2 so we can use 1 target distance measurement and 1 proximity sensor measurement.
                 Then, let self.memory_size = 3, which means at step n, we have a state array of the form:

                 state = [target_distance_n, target_distance_(n-1), target_distance_(n-2), sensor_measurement_n, sensor_measurement_(n-1), sensor_measurement_(n-2)]

                 The latest measurement (measurement_n) for each type of observation is always at the beginning of its respective section and can be accessed through its respective 'base index' ('bi')
        """

        # proximity reward
        w_p = 1
        r_p = 0

        for i in reversed(range(self.d_bi + 1, self.d_bi + self.memory_size)):
            r_p += (self.current_state[i] - self.current_state[i-1])

        self._update_terminal_flag(new_state)

        # NOTE: r_p is used in some of the remaining rewards as a 'scaling factor' to make sure all rewards are of approximately the same scale

        initial_distance = np.sqrt((self.target_x - self.initial_x)**2 + (self.target_y - self.initial_y)**2)
        final_distance = np.sqrt((self.target_x - self.car.P_f[0])**2 + (self.target_y - self.car.P_f[1])**2)
        
        # crash penalty
        w_c = -2
        r_c = int(self.bad_terminal) * abs(final_distance - initial_distance)
        
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


    def _get_reward_4(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

        """
        the state is a memory buffer of size self.memory_size * self.observation_count that holds the past 'self.memory_size' measurements for
        each of the 'self.observation_count' observations.

        Example: Let self.observation_count = 2 so we can use 1 target distance measurement and 1 proximity sensor measurement.
                 Then, let self.memory_size = 3, which means at step n, we have a state array of the form:

                 state = [target_distance_n, target_distance_(n-1), target_distance_(n-2), sensor_measurement_n, sensor_measurement_(n-1), sensor_measurement_(n-2)]

                 The latest measurement (measurement_n) for each type of observation is always at the beginning of its respective section and can be accessed through its respective 'base index' ('bi')
        """

        # proximity reward
        w_p = 1
        r_p = 0

        for i in reversed(range(self.d_bi + 1, self.d_bi + self.memory_size)):
            r_p += (self.current_state[i] - self.current_state[i-1])

        self._update_terminal_flag(new_state)

        # NOTE: r_p is used in some of the remaining rewards as a 'scaling factor' to make sure all rewards are of approximately the same scale

        initial_distance = np.sqrt((self.target_x - self.initial_x)**2 + (self.target_y - self.initial_y)**2)
        final_distance = np.sqrt((self.target_x - self.car.P_f[0])**2 + (self.target_y - self.car.P_f[1])**2)

        # crash penalty
        w_c = -1
        r_c = int(self.bad_terminal) * abs(final_distance - initial_distance)

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


    def _get_reward_5(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

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
        w_c = -2
        r_c = int(self.collided)

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

        w_s = 0.50
        r_s = (-abs(avg_steer)) * abs(r_p) # <- penalize non-zero average steer


        return r_p*w_p + r_g*w_g + r_c*w_c + r_t*w_t + r_s*w_s


    def _get_reward_6(self, new_state: np.ndarray, old_state: np.ndarray, action: np.ndarray) -> float:

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
        r_c = int(self.collided)

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

        w_s = 0.50
        r_s = (-abs(avg_steer)) * abs(r_p) # <- penalize non-zero average steer


        return r_p*w_p + r_g*w_g + r_c*w_c + r_t*w_t + r_s*w_s


    def _check_for_collision(self) -> None:

        x_P = self.car.P[0]
        y_P = self.car.P[1]

        for obstacle in self.obstacles:

            x_1 = obstacle.x_1
            y_1 = obstacle.y_1

            x_2 = obstacle.x_2
            y_2 = obstacle.y_2

            m_o = (y_2 - y_1)/(x_2 - x_1 + np.finfo(float).eps) # <- the slope of the obstacle line
            m_s = -1/(m_o + np.finfo(float).eps) # <- slope of perpendicular line
            gamma = np.arctan(m_o)

            # calculate point of intersection obstacle line and perpendicular line that goes through center of car
            x_intersection = (m_o * x_1 - m_s * x_P + y_P - y_1)/(m_o - m_s)
            y_intersection = obstacle.y(x_intersection)

            # check if it is a valid intersection
            intersects_obstacle =  (x_intersection > min(x_1, x_2)) and (x_intersection < max(x_1, x_2))

            # calculate sensor measurement
            if intersects_obstacle:

                distance = np.sqrt( (x_P - x_intersection)**2 + (y_P - y_intersection)**2 )

                if distance <= self.collision_tolerance:
                    self.car.reorient(gamma)
                    self.collided = True
                    self.bounce_guard = 0 # <- reset guard (won't be able to bounce again for 4 steps)

    def _update_state(self) -> np.ndarray:
        """
        calculate the distance to the target and the sensor data
        """

        # assign base index for distance measurement
        self.d_bi = 0 # <- distance measurement base index 'bi' (for use in the 'self.current_state' memory buffer

        # get target distance measurement
        d = self.car.F - self.car.P
        d = np.sqrt(d.T @ d)
        
        # check sensors
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

        # check for collision
        if self.collided:
            self.bad_terminal = True
        else:
            self.bad_terminal = False

        # check for target reached
        self.good_terminal = state[self.d_bi] < self.target_tolerance

        self.is_terminal = 1 * ((self.good_terminal) or (self.bad_terminal)) 

        return self.is_terminal

    def _get_sensor_reading(self, sensor_name: str) -> float:

        """
        basic 'line of sight' intersection calculations
        """
        if self.obstacles is None:
            return max_sensor_measurement
        
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
        self.closest_distance[sensor_name] = max_sensor_measurement
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
            x_intersection = (m_o * x_1 - m_s * x_P + y_P - y_1)/(m_o - m_s + np.finfo(float).eps)
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

    def render(self) -> None:

        if self.render_type == "human":

            self._set_human_render()

            self.fig.show()


    def _set_human_render(self) -> None:

        self.reset(self.seed)

        # setup figure
        self.fig = gplt.Figure(fps = 1/self.dt)    
    
        # setup box bounds
        self.fig.plot([self.x_min, self.x_min, self.x_max, self.x_max, self.x_min],
                      [self.y_max, self.y_min, self.y_min, self.y_max, self.y_max],
                      name='box', color=(255,255,255))

        # setup console
        self.fig.text("REWARD", self.x_min, self.y_min, name="reward text", alignment="topleft", font_size=20)

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

        if self.obstacles is not None:
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

            if self.obstacles is not None:
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
        env = Environment(dt=dt, x_bounds=x_bounds, y_bounds=y_bounds,
                          memory_size=memory_size, sensor_names=sensor_names, reward_function=reward_function,
                          obstacle_types=obstacle_types, obstacle_proportions=obstacle_proportions, 
                          obstacle_configs=obstacle_configs, seed=seed, render_type="human")
        env.render()
