# plot 模块

特征可视化模块，提供参数变化过程的可视化功能。

## 可视化函数

::: ezflt.plot.visualize_feature_evolution
    options:
      show_source: true
      heading_level: 3

::: ezflt.plot.visualize_features_heatmap
    options:
      show_source: true
      heading_level: 3

::: ezflt.plot.visualize_parameter_comparison
    options:
      show_source: true
      heading_level: 3

::: ezflt.plot.visualize
    options:
      show_source: true
      heading_level: 3

## 使用示例

```python
from ezflt.plot import (
    visualize_feature_evolution,
    visualize_features_heatmap,
    visualize_parameter_comparison
)

# 可视化单个特征的演化
feature = tracker.get_feature("conv1.weight")
visualize_feature_evolution(feature, metric="norm", save_path="evolution.png")

# 可视化所有特征的热力图
visualize_features_heatmap(tracker, epoch=9, save_path="heatmap.png")

# 比较不同epoch的参数
visualize_parameter_comparison(
    feature,
    epochs=[0, 5, 9],
    save_path="comparison.png"
)
```

