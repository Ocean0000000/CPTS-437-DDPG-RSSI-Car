import torch
import numpy as np

# torch/numpy configuration
torch_dtype = torch.float32
np_dtype = np.float32
torch_device = torch.device("cpu")

# environment configuration
MAX_SENSOR_MEASUREMENT = 4
dt = 1/30
x_bounds = [2, 5]
y_bounds = [2, 5]
memory_size = 10
sensor_names = ["front", "left", "right"]
obstacle_proportions = None
obstacle_types = ["track"]
obstacle_count = 6
obstacle_size = 0.3
seed = None

# training configuration
epoch_count = 1000
hidden_neurons = 64
hidden_layers = 2
save_freq = 20
