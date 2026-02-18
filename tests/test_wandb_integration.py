"""
Wandb集成功能测试

需要wandb API key已配置才能运行。
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ezflt import Trainer, FeatureTracker, FeatureTrackingCallback, ExperimentManager


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def simple_dataloader():
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=4)


@pytest.mark.skipif(
    True,  # 默认跳过，需要手动启用（pytest -m "not skip"）
    reason="需要wandb API key和网络连接"
)
class TestWandbIntegration:
    """Wandb集成测试"""
    
    def test_experiment_manager_wandb(self, simple_model, simple_optimizer, simple_dataloader):
        """测试ExperimentManager的Wandb集成"""
        manager = ExperimentManager(
            project_name="ezflt-test",
            experiment_name="test-exp-manager",
            use_wandb=True,
            use_hydra=False,
        )
        
        # 初始化wandb
        manager._init_wandb()
        assert manager.wandb_run is not None
        
        # 记录一些指标
        manager.log_metrics({"test_metric": 1.0}, step=0)
        manager.log_metrics({"test_metric": 2.0}, step=1)
        
        # 完成实验
        manager.finish()
    
    def test_trainer_with_wandb(self, simple_model, simple_optimizer, simple_dataloader):
        """测试Trainer的Wandb集成"""
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            use_wandb=True,
            use_hydra=False,
        )
        
        # 训练（应该自动记录到wandb）
        trainer.fit(train_loader=simple_dataloader, num_epochs=2)
        
        # 检查实验管理器已初始化
        assert trainer.experiment_manager is not None
        assert trainer.experiment_manager.wandb_run is not None
    
    def test_trainer_with_tracker_and_wandb(self, simple_model, simple_optimizer, simple_dataloader):
        """测试Trainer、FeatureTracker和Wandb的完整集成"""
        # 创建特征追踪器
        tracker = FeatureTracker(
            model=simple_model,
            layers=["fc"],
            track_weights=True,
        )
        
        # 创建训练器（自动集成wandb）
        trainer = Trainer(
            model=simple_model,
            optimizer=simple_optimizer,
            callbacks=[FeatureTrackingCallback(tracker)],
            use_wandb=True,
            use_hydra=False,
        )
        
        # 训练
        trainer.fit(train_loader=simple_dataloader, num_epochs=2)
        
        # 记录特征信息到wandb
        features = tracker.get_features()
        if trainer.experiment_manager:
            trainer.experiment_manager.log_features(features, epoch=1)
        
        # 完成实验
        if trainer.experiment_manager:
            trainer.experiment_manager.finish()


# 手动测试脚本（可以直接运行）
if __name__ == "__main__":
    print("测试Wandb集成...")
    
    # 创建模型和数据
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)
    
    # 测试ExperimentManager
    print("1. 测试ExperimentManager...")
    manager = ExperimentManager(
        project_name="ezflt-test",
        experiment_name="manual-test",
        use_wandb=True,
        use_hydra=False,
    )
    manager._init_wandb()
    manager.log_metrics({"test": 1.0}, step=0)
    manager.finish()
    print("   ✓ ExperimentManager测试通过")
    
    # 测试Trainer
    print("2. 测试Trainer集成...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        use_wandb=True,
        use_hydra=False,
    )
    trainer.fit(train_loader=dataloader, num_epochs=2)
    print("   ✓ Trainer测试通过")
    
    print("\n所有测试通过！请检查wandb网站确认实验已记录。")

