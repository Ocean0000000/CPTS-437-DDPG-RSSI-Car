import torch
import numpy as np

# torch/numpy configuration
torch_dtype = torch.float32
np_dtype = np.float32
torch_device = torch.device("cpu")

# environment configuration
MAX_SENSOR_MEASUREMENT = 2
dt = 1/30
x_bounds = [5, 5]
y_bounds = [5, 5]
memory_size = 10
sensor_names = ["front", "right", "left"]
obstacle_proportions = None
obstacle_types = ["wall", "random"]
obstacle_count = 3
obstacle_size = 0.2
seed = None

# training configuration
epoch_count = 500
hidden_neurons = 64
hidden_layers = 2
save_freq = 20
