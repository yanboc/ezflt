"""
ezflt基本使用示例

展示如何使用ezflt进行特征追踪和可视化。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ezflt import FeatureTracker, FeatureTrackingCallback, Trainer
from ezflt.plot import visualize_feature_evolution, visualize_features_heatmap

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


# 2. 创建模型和数据
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 生成示例数据
X = torch.randn(100, 1, 28, 28)
y = torch.randint(0, 10, (100,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 创建特征追踪器
tracker = FeatureTracker(
    model=model,
    layers=["conv1", "conv2", "fc"],  # 追踪这些层
    track_weights=True,
    track_frequency=1,  # 每个batch都记录
)

# 4. 创建回调
callbacks = [FeatureTrackingCallback(tracker)]

# 5. 创建训练器并训练
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    callbacks=callbacks,
)

trainer.fit(train_loader=dataloader, num_epochs=5)

import ipdb; ipdb.set_trace()

# 6. 可视化结果
# 可视化单个特征的演化
conv1_weight = tracker.get_feature("conv1.weight")
if conv1_weight:
    visualize_feature_evolution(conv1_weight, metric="norm", save_path="conv1_evolution.png")

# 可视化所有特征的热力图
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
visualize_features_heatmap(tracker, epoch=4, save_path=figure_dir / "features_heatmap.pdf")

print("训练完成！特征追踪结果已保存。")

