# 使用示例

## 基础用法：特征追踪

```python
import torch
import torch.nn as nn
from ezflt import FeatureTracker
from ezflt.plot import visualize_feature_evolution

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 创建特征追踪器
tracker = FeatureTracker(
    model=model,
    layers=["0", "2"],
    track_weights=True,
)

# 使用上下文管理器
with tracker:
    for epoch in range(10):
        tracker.on_epoch_start(epoch)
        for batch_idx in range(100):
            # ... 训练代码 ...
            tracker.on_batch_end(batch_idx)

# 可视化
feature = tracker.get_feature("0.weight")
if feature:
    visualize_feature_evolution(feature, metric="norm")
```

## 高级用法：集成训练器和回调

```python
from ezflt import (
    FeatureTracker,
    FeatureTrackingCallback,
    WandbCallback,
    Trainer
)
from torch.utils.data import DataLoader

# 创建特征追踪器
tracker = FeatureTracker(
    model=model,
    layers=["conv1", "fc"],
    track_weights=True,
    track_frequency=10,
)

# 创建回调
callbacks = [
    FeatureTrackingCallback(tracker),
    WandbCallback(project="flt-research"),
]

# 创建训练器
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    callbacks=callbacks,
)

# 训练
trainer.fit(train_loader=dataloader, num_epochs=10)

# 可视化结果
from ezflt.plot import visualize_features_heatmap
visualize_features_heatmap(tracker, epoch=9)
```

## 完整示例

查看 `examples/basic_usage.py` 获取完整的使用示例。

