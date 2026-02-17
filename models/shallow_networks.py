'''
Implementation of shallow (i.e., 2-layer) neural network architectures
'''

import json
from torch import nn

class ShallowNetwork(nn.Module):
    '''
    Initialize a shallow neural network, load parameters from config.json and perform shape check.
    Users need to define the forward function themselves.
    '''
    def __init__(self):
        super.__init__()

    def load_config(self, config_file: str = "../config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.task = self.config["task"]
        self.arch = self.config["model"]["arch"]        
        self.input_dim = self.config["model"]["input_dim"]
        if self.task == "classification":
            self.num_classes = self.config["model"]["num_classes"]

        self.check_params()

    def check_params(self):
        '''
        Check the validity of the parameters loaded from config.json
        '''
        assert self.task in self.config["model"]["supported_tasks"], \
            f"Task {self.task} not supported. Supported tasks: {self.config['model']['supported_tasks']}"
        assert self.arch in self.config["model"]["supported_archs"], \
            f"Architecture {self.arch} not supported. Supported archs: {self.config['model']['supported_archs']}"
        assert self.input_dim == self.config["data"]["input_shape"], \
            f"Input dimension {self.input_dim} does not match expected input shape {self.config['data']['input_shape']}"

    def forward(self):
        pass

def shallow_mlp():
    pass

