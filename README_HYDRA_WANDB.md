# Hydra 和 Wandb 集成说明

ezflt 默认使用 **Hydra** 和 **Wandb** 管理多组实验。

## 快速开始

### 1. 安装依赖

```bash
uv pip install -e ".[dev]"
```

Hydra 和 Wandb 已包含在主依赖中，无需额外安装。

### 2. 基本使用

```python
from ezflt import Trainer, FeatureTracker, FeatureTrackingCallback, ExperimentManager

# 创建实验管理器（默认启用Hydra和Wandb）
with ExperimentManager(
    config_path="configs",
    project_name="my-project",
    use_wandb=True,  # 默认True
    use_hydra=True,  # 默认True
) as exp_manager:
    
    # 获取配置（来自Hydra）
    config = exp_manager.get_config()
    
    # 创建模型和训练器
    model = ...
    optimizer = ...
    
    # Trainer会自动集成Wandb
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        use_wandb=True,  # 默认启用
        use_hydra=True,  # 默认启用
    )
    
    # 训练（自动记录到Wandb）
    trainer.fit(train_loader=dataloader, num_epochs=10)
```

### 3. 使用 Hydra 配置

配置文件位于 `configs/config.yaml`。可以通过命令行覆盖参数：

```bash
python train.py training.learning_rate=0.01 training.epochs=50
```

### 4. 多组实验（参数扫描）

使用 Hydra 的多组实验功能：

```bash
# 扫描学习率
python train.py -m training.learning_rate=0.001,0.01,0.1

# 扫描多个参数
python train.py -m training.learning_rate=0.001,0.01 training.epochs=10,20,30
```

每次运行会自动创建新的 Wandb 实验记录。

## 配置说明

### Hydra 配置

配置文件结构：
```
configs/
├── config.yaml          # 主配置文件
└── experiment/
    ├── default.yaml     # 默认实验配置
    └── sweep.yaml       # 扫描实验配置
```

### Wandb 配置

在 `config.yaml` 中配置：

```yaml
wandb:
  project: ezflt-experiments
  enabled: true
```

## 禁用默认行为

如果不想使用 Hydra 或 Wandb：

```python
# 只使用 Wandb，不使用 Hydra
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    use_wandb=True,
    use_hydra=False,
)

# 完全禁用
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    use_wandb=False,
    use_hydra=False,
)
```

## 更多示例

查看 `examples/hydra_wandb_example.py` 获取完整示例。

