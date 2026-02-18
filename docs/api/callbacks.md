# callbacks 模块

训练回调系统，实现生命周期钩子，支持在训练过程的特定时刻执行自定义逻辑。

## Callback

回调基类，定义训练生命周期的钩子接口。

::: ezflt.callbacks.Callback
    options:
      show_source: true
      heading_level: 3

## FeatureTrackingCallback

特征追踪回调，在训练过程中自动追踪模型参数的变化。

::: ezflt.callbacks.FeatureTrackingCallback
    options:
      show_source: true
      heading_level: 3

## WandbCallback

Wandb集成回调，自动将训练指标、超参数和特征信息记录到wandb。

::: ezflt.callbacks.WandbCallback
    options:
      show_source: true
      heading_level: 3

## 使用示例

```python
from ezflt import FeatureTracker, FeatureTrackingCallback, WandbCallback, Trainer

# 创建追踪器和回调
tracker = FeatureTracker(model=model, layers=["conv1", "fc"])
callbacks = [
    FeatureTrackingCallback(tracker),
    WandbCallback(project="flt-research"),
]

# 在训练器中使用
trainer = Trainer(model=model, optimizer=optimizer, callbacks=callbacks)
trainer.fit(train_loader=dataloader, num_epochs=10)
```

