from config import *
import numpy as np
import torch
torch.set_default_dtype(torch_dtype)

import simulation as sim
from core import MLPActorCritic as actor_critic
import time
import os


# LOAD MODULE
checkpoint = torch.load("tests/trial_5/checkpoints/checkpoint_999.tar", map_location=torch_device)
ac = checkpoint["model"][0]
ac.to(torch_device)

def nn_control(state: np.ndarray) -> float:

    a, v, logp = ac.step(torch.as_tensor(state, dtype=torch_dtype, device=torch_device))

    return a

# CREATE ENVIRONMENT
# env = sim.Environment(dt=dt, x_bounds=[-2,2], y_bounds=[-2, 2], memory_size=memory_size, sensor_names=sensor_names,
#                       obstacle_count=obstacle_count, obstacle_size=obstacle_size, seed=seed, render_type="human", nn_control=nn_control,
#                       obstacle_types=obstacle_types, obstacle_proportions=obstacle_proportions)
env = sim.Environment(dt=dt, x_bounds=x_bounds, y_bounds=y_bounds, memory_size=memory_size, sensor_names=sensor_names,
                      obstacle_count=obstacle_count, obstacle_size=obstacle_size, seed=seed, render_type="human", nn_control=nn_control,
                      obstacle_types=obstacle_types, obstacle_proportions=obstacle_proportions)

while True:
    env.render()




