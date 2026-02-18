"""
pytest配置文件

提供测试用的fixtures和配置
"""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_model():
    """提供简单的测试模型"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )


@pytest.fixture
def sample_data():
    """提供示例数据"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    return X, y


@pytest.fixture
def sample_config():
    """提供示例配置"""
    return {
        "seed": 42,
        "device": "cpu",
        "experiment_id": "test_experiment",
        "repeats": 1,
        "data": {
            "generate_data": True,
            "data_type": "tensor",
            "store_data": False,
            "parameters": {
                "P": [20],
                "d_P_ratio": [1.2],
            },
            "train_size": 100,
            "test_size": 20,
            "batch_size": 32,
            "shuffle": True
        },
        "model": {
            "task": "classification",
            "arch": "shallow",
            "num_classes": 5
        }
    }

