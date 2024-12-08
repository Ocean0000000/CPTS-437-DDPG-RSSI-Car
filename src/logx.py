"""
Adapted from OpenAI's SpinningUp PPO implementation
"""
from config import *
import numpy as np
import torch
torch.set_default_dtype(torch_dtype)

import os

class Logger():

    def __init__(self) -> None:
        
        os.makedirs("logs", exist_ok=True)
        self.location = 'logs'

        self.epoch_dict = {}


    def log(self, message: str) -> None:
        """
        append to a log file
        """

        with open(self.location + "/log.txt", "a") as f:
            f.write(message + "\n")
        print(message)


    def store(self, **kwargs) -> None:
        """
        Save something into the logger's current dict

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """

        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
