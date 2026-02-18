# 快速开始

本指南将帮助你快速开始使用 ezflt。

## 安装

### 前置要求

- Python 3.9+
- uv（推荐）或 pip

### 使用 uv 安装（推荐）

```bash
# 1. 创建虚拟环境
uv venv

# 2. 激活虚拟环境
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux:
source .venv/bin/activate

# 3. 安装项目（开发模式，包含测试依赖）
uv pip install -e ".[dev]"
```

### 使用 pip 安装

```bash
pip install -e ".[dev]"
```

## 基本使用

### 1. 特征追踪

```python
import torch
import torch.nn as nn
from ezflt import FeatureTracker

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 创建特征追踪器
tracker = FeatureTracker(
    model=model,
    layers=["0", "2"],  # 追踪第一层和第三层
    track_weights=True,
)

# 使用上下文管理器自动管理追踪生命周期
with tracker:
    # 你的训练循环
    for epoch in range(10):
        tracker.on_epoch_start(epoch)
        # ... 训练代码 ...
        tracker.on_batch_end(batch_idx)

# 获取追踪的特征
features = tracker.get_features()
```

### 2. 集成训练器

```python
from ezflt import FeatureTracker, FeatureTrackingCallback, Trainer
from torch.utils.data import DataLoader

# 创建特征追踪器
tracker = FeatureTracker(model=model, layers=["conv1", "fc"])

# 创建回调
callbacks = [FeatureTrackingCallback(tracker)]

# 创建训练器
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    callbacks=callbacks,
)

# 训练模型
trainer.fit(train_loader=dataloader, num_epochs=10)
```

### 3. 可视化

```python
from ezflt.plot import visualize_feature_evolution, visualize_features_heatmap

# 可视化单个特征的演化
feature = tracker.get_feature("0.weight")
visualize_feature_evolution(feature, metric="norm")

# 可视化所有特征的热力图
visualize_features_heatmap(tracker, epoch=9)
```

## 下一步

- 查看 [API 参考](api/tracker.md) 了解完整的 API
- 查看 [示例](examples.md) 了解更多使用场景

