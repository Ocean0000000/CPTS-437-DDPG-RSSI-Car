# CPTS 437 PPO RSSI Car
The goal is to train a car using the ~~Deep Deterministic Policy Gradient~~ Proximal Policy Optimization algorithm that can navigate its surroundings in order to reach the signal source, using proximity sensors for obstacle avoidance and RSSI localization for determining proximity to signal.

Code created by rangelDi
https://github.com/rangelDi

The gitignore ignores every file with extension .tar (meaning all checkpoints created during training are ignored). When committing a checkpoint, please rename the checkpoint file extension from .tar to .tarcopy. Consequently, to use a checkpoint in src/ppo.py or src/observe.py, rename the file extension from .tarcopy to .tar.