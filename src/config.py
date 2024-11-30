import torch
import numpy as np

# torch/numpy configuration
torch_dtype = torch.float32
np_dtype = np.float32
torch_device = torch.device("cuda")

# environment configuration
MAX_SENSOR_MEASUREMENT = 0.5
dt = 1/30
x_bounds = [-2,2]
y_bounds = [-2,2]
memory_size = 10
sensor_names = ["front"]
obstacle_count = 10
obstacle_size = 0.1
seed = None

# training configuration
epoch_count = 1000
hidden_neurons = 128
hidden_layers = 3
save_freq = 20
