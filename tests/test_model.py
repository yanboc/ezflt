"""
测试网络（模型）的共用接口
"""

import pytest
import torch
import torch.nn as nn
from ezflt.model import FLTNetwork


class TestModelInterface:
    """测试模型的共用接口"""
    
    def test_flt_network_initialization(self):
        """测试FLTNetwork初始化"""
        config = {
            "model": {
                "task": "classification",
                "arch": "shallow",
                "num_classes": 10
            },
            "data": {
                "train_size": 1000,
                "test_size": 200
            }
        }
        
        network = FLTNetwork(config)
        assert network.task == "classification"
        assert network.arch == "shallow"
    
    def test_flt_network_supported_tasks(self):
        """测试支持的任务类型"""
        config = {
            "model": {"task": "classification", "arch": "shallow", "num_classes": 10},
            "data": {"train_size": 100, "test_size": 20}
        }
        network = FLTNetwork(config)
        supported_tasks = network.get_supported_tasks()
        assert "classification" in supported_tasks
        assert "generation" in supported_tasks
    
    def test_flt_network_supported_archs(self):
        """测试支持的架构类型"""
        config = {
            "model": {"task": "classification", "arch": "shallow", "num_classes": 10},
            "data": {"train_size": 100, "test_size": 20}
        }
        network = FLTNetwork(config)
        supported_archs = network.get_supported_archs()
        assert "shallow" in supported_archs
        assert "transformer" in supported_archs


class TestModelCompatibility:
    """测试PyTorch原生兼容性"""
    
    def test_standard_pytorch_model(self):
        """测试标准PyTorch模型"""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # 测试前向传播
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)
    
    def test_model_parameters(self):
        """测试模型参数访问"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # 测试参数访问
        params = list(model.parameters())
        assert len(params) == 4  # 2个weight + 2个bias
        
        # 测试named_parameters
        named_params = dict(model.named_parameters())
        assert "0.weight" in named_params
        assert "2.weight" in named_params
    
    def test_model_named_modules(self):
        """测试named_modules访问"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        modules = dict(model.named_modules())
        assert "" in modules  # 根模块
        assert "0" in modules
        assert "2" in modules

