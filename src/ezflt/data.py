import os
import json
import itertools
import torch
from dataclasses import dataclass

class Generator():
    '''
    Generate data based on configuration file, or 
    '''
    def __init__(self, config):
        self.seed = config["seed"]
        torch.manual_seed(self.seed)
        self.device = config["device"]
        self.experiment_id = config["experiment_id"]
        self.repeats = config["repeats"]
        self.data_config = config["data"]

        self.data_type = self.data_config["data_type"]
        self.train_size = self.data_config["train_size"]
        self.test_size = self.data_config["test_size"]
        self.batch_size = self.data_config["batch_size"]
        self.shuffle = self.data_config["shuffle"]
        self.parameters = self.data_config["parameters"]
        self.parameters_combinations = list(itertools.product(*self.parameters.values()))

        self.check_config(self.data_config)
        self._feature = None

    def check_config(self, data_config):
        pass

    def process_parameters(self, parameters):
        pass

    def generate_feature(self):
        pass

    def get_feature(self):
        if self._feature is None:
            raise ValueError("self._feature not defined")
        else:
            return self._feature
        
    def get_supported_data_type(self):
        return ["tensor, sequence"]