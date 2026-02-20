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
# 使用@hydra.main装饰器（推荐）
python examples/train_with_hydra_simple.py training.learning_rate=0.01 training.epochs=50

# 或手动初始化Hydra
python examples/train_with_hydra.py training.learning_rate=0.01
```

### 4. 多组实验（参数扫描）

使用 Hydra 的多组实验功能（multirun）：

```bash
# 扫描学习率（会运行3次实验）
python examples/train_with_hydra_simple.py -m training.learning_rate=0.001,0.01,0.1

# 扫描多个参数（会运行3x3=9次实验）
python examples/train_with_hydra_simple.py -m training.learning_rate=0.001,0.01,0.1 training.epochs=10,20,30

# 使用网格搜索
python examples/train_with_hydra_simple.py -m training.learning_rate=0.001,0.01 training.epochs=10,20
```

每次运行会自动创建新的 Wandb 实验记录。

### 5. 使用@hydra.main装饰器（推荐方式）

最简单的方式是使用 `@hydra.main` 装饰器：

```python
from hydra import main
from omegaconf import DictConfig

@main(config_path="configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    # cfg包含所有配置
    print(f"学习率: {cfg.training.learning_rate}")
    print(f"Epochs: {cfg.training.epochs}")
    
    # 创建实验管理器
    exp_manager = ExperimentManager(
        use_wandb=True,
        use_hydra=True,
    )
    exp_manager.hydra_cfg = cfg
    exp_manager._hydra_initialized = True
    
    # ... 训练代码 ...

if __name__ == "__main__":
    train()
```

查看 `examples/train_with_hydra_simple.py` 获取完整示例。

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

