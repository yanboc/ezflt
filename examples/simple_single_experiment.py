"""
最简单的单次实验示例

展示ezflt最基础的使用方式，适合快速上手。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ezflt import Trainer, FeatureTracker, FeatureTrackingCallback


# 1. 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


# 2. 准备数据
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 创建模型和优化器
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 4. 创建特征追踪器（可选）
tracker = FeatureTracker(model=model, layers=["fc"], track_weights=True)

# 5. 创建训练器（自动集成Wandb）
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[FeatureTrackingCallback(tracker)],
    use_wandb=True,  # 默认启用
    use_hydra=False,  # 单次实验不使用Hydra
)

# 6. 训练（自动记录到Wandb）
tracker.start()
trainer.fit(train_loader=dataloader, num_epochs=5)
tracker.stop()

# 7. 完成实验
if trainer.experiment_manager:
    trainer.experiment_manager.finish()

print("训练完成！实验已自动记录到Wandb。")

