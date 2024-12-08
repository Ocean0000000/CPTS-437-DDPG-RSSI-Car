"""
script to observe the behavior of the model in human-rendered simulation
"""
from config import *
import numpy as np
import torch
torch.set_default_dtype(torch_dtype)

import simulation as sim
from core import MLPActorCritic as actor_critic
import time
import os


# LOAD MODULE
checkpoint = torch.load("checkpoints/checkpoint_200.tar") # <- pick checkpoint to load
ac = checkpoint["model"][0]


# SETUP ENVIRONMENT
def nn_control(state: np.ndarray) -> float:

    a, v, logp = ac.step(torch.as_tensor(state, dtype=torch_dtype))

    return a

env = sim.Environment(dt=dt, x_bounds=x_bounds, y_bounds=y_bounds,
                      memory_size=memory_size, sensor_names=sensor_names, reward_function=reward_function,
                      obstacle_types=obstacle_types, obstacle_proportions=obstacle_proportions, 
                      obstacle_configs=obstacle_configs, seed=seed, render_type="human", nn_control=nn_control)

# RENDER
while True:

    env.render()




