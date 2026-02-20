'''
Implementation of shallow (i.e., 2-layer) neural network architectures
'''

from dataclasses import dataclass

@dataclass
class ModelConfig:
    task: str = "classification"
    arch: str = "shallow"
    num_classes: int = 10
    input_dim: tuple = (3, 32, 32)
    output_dim: int = 10
    num_classes: int = 10

class FLTNetwork():
    '''
    Initialize a shallow neural network, load parameters from config.json and perform shape check.
    Users need to define the forward function themselves.
    '''
    def __init__(self, config):

        self.model_config = config['model']
        self.data_config = config['data']

        self.task = self.model_config["task"]
        self.arch = self.model_config["arch"]
        if self.task == "classification":
            try:
               self.num_classes = self.model_config["num_classes"]
            except:
                raise ValueError("num_classes not found in model_config")

        self.check_params(model_config=self.model_config, data_config=self.data_config)

    def check_params(self, model_config: dict, data_config: dict):
        '''
        Check the validity of the parameters loaded from config.json
        '''
        pass

    def track_feature(self, target_module: dict, tracked_weights: dict = None):
        """
        Track the learned features (e.g., network weights or other related 
        statistics) through training and store them in a dictionary.
        """
        if tracked_weights is None:
            tracked_weights = {}
        
        # the target module (function) and the weights share the same key
        for key in target_module.keys():
            tracked_weights[key] = []

        for key in target_module.keys():
            weight = target_module[key].weight.detach().clone()
            tracked_weights[key].append(weight)

        return tracked_weights

    def get_supported_tasks(self):
        return ["classification", "generation"]

    def get_supported_archs(self):
        return ["shallow", "transformer", "diffusion"]
