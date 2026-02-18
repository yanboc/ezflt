# tracker 模块

特征追踪核心模块，实现非侵入式的参数追踪功能。

## Feature

保存网络参数（即"网络所提取的特征"）的类。

::: ezflt.tracker.Feature
    options:
      show_source: true
      heading_level: 3

## FeatureTracker

特征追踪器，非侵入式地追踪模型在训练过程中参数的变化。

::: ezflt.tracker.FeatureTracker
    options:
      show_source: true
      heading_level: 3

## 使用示例

```python
from ezflt import FeatureTracker, Feature
import torch.nn as nn

# 创建模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# 创建追踪器
tracker = FeatureTracker(
    model=model,
    layers=["0", "2"],
    track_weights=True,
    track_frequency=1
)

# 开始追踪
tracker.start()

# 在训练循环中使用
for epoch in range(10):
    tracker.on_epoch_start(epoch)
    for batch_idx in range(100):
        # ... 训练代码 ...
        tracker.on_batch_end(batch_idx)

# 获取特征
features = tracker.get_features()
feature = tracker.get_feature("0.weight")

# 停止追踪
tracker.stop()
```

