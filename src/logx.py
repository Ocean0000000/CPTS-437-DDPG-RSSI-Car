import torch
torch.set_default_dtype(torch.float64)

import os

class Logger():

    def __init__(self, location: str = None) -> None:
        
        if location == None:
            os.system('mkdir logs')
            self.location = 'logs'

        self.logger_dict = {}


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
            if not(k in self.logger_dict.keys()):
                self.logger_dict[k] = []
            self.logger_dict[k].append(v)

    def save_state(self, filename: str) -> None:
        """
        Save the current log dict
        """

        torch.save(self.logger_dict, filename)

