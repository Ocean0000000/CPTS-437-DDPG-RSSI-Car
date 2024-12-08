"""
training session configuration file
"""
import torch
import numpy as np

# TORCH/NUMPY CONFIGURATION
torch_dtype = torch.float32
np_dtype = np.float32
torch_device = torch.device("cpu")

# ENVIRONMENT CONFIGURATION
proportional = False
bounce = False
max_sensor_measurement = 0.5
dt = 1/30
x_bounds = [2, 5]
y_bounds = [2, 5]
memory_size = 10
sensor_names = ["front"]
obstacle_proportions = None
obstacle_types   = ["random", "school"]
obstacle_configs = [{"obstacle_count": 10, "obstacle_size": 0.1}, {"version":"setups"}]
reward_function = 6 # 1, 2, 3, 4, or 5

# TRAINING CONFIGURATION
hidden_neurons = 64
hidden_layers = 2
epoch_count = 500
steps_per_epoch=4000
gamma = 0.99
clip_ratio=0.2
pi_lr=3e-4
vf_lr=1e-3
train_pi_iters=80
train_v_iters=80
lam=0.97
max_ep_len=1000
target_kl=0.01
save_freq = 20

seed = None # keep this at None or else it will always train on only one map!
