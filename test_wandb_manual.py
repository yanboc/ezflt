"""
手动测试Wandb功能的脚本

运行方式：python test_wandb_manual.py
"""

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


def main():
    print("=" * 60)
    print("Wandb集成功能测试")
    print("=" * 60)
    
    # 创建模型和数据
    print("\n1. 创建模型和数据...")
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)
    print("   ✓ 完成")
    
    # 测试1: ExperimentManager基本功能
    print("\n2. 测试ExperimentManager基本功能...")
    try:
        manager = ExperimentManager(
            project_name="ezflt-test",
            experiment_name="test-exp-manager",
            use_wandb=True,
            use_hydra=False,
        )
        manager._init_wandb()
        assert manager.wandb_run is not None, "Wandb未初始化"
        
        # 记录一些指标
        manager.log_metrics({"test_metric": 1.0}, step=0)
        manager.log_metrics({"test_metric": 2.0}, step=1)
        
        manager.finish()
        print("   ✓ ExperimentManager测试通过")
    except Exception as e:
        print(f"   ✗ ExperimentManager测试失败: {e}")
        return
    
    # 测试2: Trainer自动集成Wandb
    print("\n3. 测试Trainer自动集成Wandb...")
    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            use_wandb=True,
            use_hydra=False,
        )
        
        # 检查是否自动创建了ExperimentManager
        assert trainer.experiment_manager is not None, "ExperimentManager未创建"
        assert trainer.experiment_manager.use_wandb is True, "Wandb未启用"
        
        # 训练（应该自动记录到wandb）
        trainer.fit(train_loader=dataloader, num_epochs=2)
        
        print("   ✓ Trainer集成测试通过")
    except Exception as e:
        print(f"   ✗ Trainer集成测试失败: {e}")
        return
    
    # 测试3: 完整集成（Trainer + FeatureTracker + Wandb）
    print("\n4. 测试完整集成（Trainer + FeatureTracker + Wandb）...")
    try:
        # 创建新的模型实例
        model2 = SimpleModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        
        # 创建特征追踪器
        tracker = FeatureTracker(
            model=model2,
            layers=["fc"],
            track_weights=True,
        )
        
        # 创建训练器
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            callbacks=[FeatureTrackingCallback(tracker)],
            use_wandb=True,
            use_hydra=False,
        )
        
        # 训练
        trainer2.fit(train_loader=dataloader, num_epochs=2)
        
        # 记录特征信息到wandb
        features = tracker.get_features()
        if trainer2.experiment_manager:
            trainer2.experiment_manager.log_features(features, epoch=1)
        
        print("   ✓ 完整集成测试通过")
    except Exception as e:
        print(f"   ✗ 完整集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    print("\n请访问 https://wandb.ai 查看实验记录。")
    print("项目名称: ezflt-test")


if __name__ == "__main__":
    main()

