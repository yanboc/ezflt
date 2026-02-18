"""
使用Hydra和Wandb管理多组实验的示例

展示如何使用ezflt的默认实验管理功能。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ezflt import Trainer, FeatureTracker, FeatureTrackingCallback, ExperimentManager


# 1. 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 28 * 28, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 2. 创建实验管理器（默认启用Hydra和Wandb）
with ExperimentManager(
    config_path="configs",
    project_name="ezflt-demo",
    use_wandb=True,
    use_hydra=True,
) as exp_manager:
    
    # 获取配置（来自Hydra）
    config = exp_manager.get_config()
    
    # 3. 创建模型和数据
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get("training", {}).get("learning_rate", 0.01))
    
    # 生成示例数据
    X = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 4. 创建特征追踪器
    tracker = FeatureTracker(
        model=model,
        layers=["conv1", "conv2", "fc"],
        track_weights=True,
    )
    
    # 5. 创建训练器（自动集成Wandb）
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        callbacks=[FeatureTrackingCallback(tracker)],
        use_wandb=True,  # 默认启用
        use_hydra=True,  # 默认启用
    )
    
    # 6. 训练（自动记录到Wandb）
    trainer.fit(train_loader=dataloader, num_epochs=5)
    
    # 7. 记录特征信息到Wandb
    features = tracker.get_features()
    exp_manager.log_features(features, epoch=4)
    
    print("训练完成！实验已自动记录到Wandb。")

