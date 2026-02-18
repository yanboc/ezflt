# task 模块

学习任务模块，实现FLT中的"学习任务四要素"框架。

## LearningTask

学习任务四要素，用于刻画和配置一个完整的学习任务。

::: ezflt.task.LearningTask
    options:
      show_source: true
      heading_level: 3

## 创建函数

::: ezflt.task.create_learning_task_from_config
    options:
      show_source: true
      heading_level: 3

::: ezflt.task.create_synthetic_task
    options:
      show_source: true
      heading_level: 3

::: ezflt.task.create_real_data_task
    options:
      show_source: true
      heading_level: 3

## 学习任务四要素

根据FLT理论，一个学习任务由四个要素组成：

1. **数据（Data）**：生成数据（信号-噪音模型）或真实数据
2. **模型（Model）**：浅层模型或标准模型
3. **评价标准（Evaluation）**：损失函数与经验风险的定义
4. **优化算法（Optimizer）**：SGD、Adam、AdamW等

