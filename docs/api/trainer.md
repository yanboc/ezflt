# trainer 模块

训练器模块，提供统一的训练接口，支持回调系统和特征追踪。

## Trainer

训练器，提供标准的训练循环，支持回调系统和特征追踪。

::: ezflt.trainer.Trainer
    options:
      show_source: true
      heading_level: 3

## 使用示例

```python
from ezflt import Trainer, FeatureTrackingCallback
from torch.utils.data import DataLoader

# 创建训练器
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    callbacks=[FeatureTrackingCallback(tracker)],
)

# 训练模型
trainer.fit(
    train_loader=train_loader,
    num_epochs=10,
    val_loader=val_loader  # 可选
)

# 获取训练指标
metrics = trainer.get_metrics()
```

