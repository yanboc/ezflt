# model 模块

模型架构模块，提供FLT网络基类。

## FLTNetwork

FLT网络基类，支持浅层网络等架构。

::: ezflt.model.FLTNetwork
    options:
      show_source: true
      heading_level: 3

## 使用示例

```python
from ezflt import FLTNetwork

config = {
    "model": {
        "task": "classification",
        "arch": "shallow",
        "num_classes": 10
    },
    "data": {
        "train_size": 1000,
        "test_size": 200
    }
}

# 创建网络
network = FLTNetwork(config)

# 获取支持的任务和架构
tasks = network.get_supported_tasks()
archs = network.get_supported_archs()
```

