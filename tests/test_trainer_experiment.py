"""
Trainer与ExperimentManager集成的测试
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ezflt import Trainer, ExperimentManager


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    """创建简单模型"""
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model):
    """创建优化器"""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def simple_dataloader():
    """创建数据加载器"""
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=4)


class TestTrainerWithExperimentManager:
    """Trainer与ExperimentManager集成的测试"""
    
    def test_trainer_without_experiment_manager(self, simple_model, simple_optimizer, simple_dataloader):
        """测试Trainer不使用实验管理器"""
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            use_wandb=False,
            use_hydra=False,
        )
        trainer.fit(train_loader=simple_dataloader, num_epochs=2)
        assert trainer.current_epoch == 1
    
    def test_trainer_with_experiment_manager_disabled(self, simple_model, simple_optimizer, simple_dataloader):
        """测试Trainer禁用实验管理"""
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            experiment_manager=None,
            use_wandb=False,
            use_hydra=False,
        )
        assert trainer.experiment_manager is None
        trainer.fit(train_loader=simple_dataloader, num_epochs=2)
    
    def test_trainer_auto_create_experiment_manager(self, simple_model, simple_optimizer, simple_dataloader):
        """测试Trainer自动创建实验管理器"""
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            use_wandb=True,  # 启用wandb会自动创建ExperimentManager
            use_hydra=False,
        )
        assert trainer.experiment_manager is not None
        assert trainer.experiment_manager.use_wandb is True
        assert trainer.experiment_manager.use_hydra is False
    
    def test_trainer_with_custom_experiment_manager(self, simple_model, simple_optimizer, simple_dataloader):
        """测试Trainer使用自定义实验管理器"""
        exp_manager = ExperimentManager(
            project_name="test-project",
            use_wandb=False,
            use_hydra=False,
        )
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            experiment_manager=exp_manager,
        )
        assert trainer.experiment_manager is exp_manager
        trainer.fit(train_loader=simple_dataloader, num_epochs=2)
    
    def test_trainer_auto_add_wandb_callback(self, simple_model, simple_optimizer):
        """测试Trainer自动添加Wandb回调"""
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            use_wandb=True,
            use_hydra=False,
        )
        # 检查是否有WandbCallback
        from ezflt.callbacks import WandbCallback
        has_wandb = any(isinstance(cb, WandbCallback) for cb in trainer.callbacks)
        assert has_wandb is True
    
    def test_trainer_no_duplicate_wandb_callback(self, simple_model, simple_optimizer):
        """测试Trainer不会重复添加Wandb回调"""
        from ezflt.callbacks import WandbCallback
        existing_callback = WandbCallback(project="test")
        
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            callbacks=[existing_callback],
            use_wandb=True,
            use_hydra=False,
        )
        # 应该只有一个WandbCallback
        wandb_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, WandbCallback)]
        assert len(wandb_callbacks) == 1

