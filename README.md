# EasyFLT: 特征学习理论友好库

*Easy Feature Learning Theory (`ezflt`)* 是一个轻量级库，旨在促进特征学习理论研究与实验。它提供了特征学习理论研究的各个阶段的简单实现，从数值评估到结果可视化。

## 核心定位

**重要**：这是一个关注模型训练和微调过程中**网络参数如何从数据中提取有用特征**的库。ezflt 的核心目标是帮助研究者理解和可视化神经网络在训练过程中如何学习特征。

## 核心功能

### 1. 特征提取过程追踪与可视化

ezflt 的核心功能是在训练过程中自动记录某一层或几层的参数变化过程，并将之可视化。这使得研究者能够：

- 自动追踪指定层的网络参数在训练过程中的变化
- 可视化参数如何从初始状态演化到最终状态
- 分析网络如何从数据中提取和优化特征

### 2. PyTorch 原生兼容性

ezflt 与 PyTorch 完全兼容，采用最 Pythonic、最 PyTorch 原生的方式：

- **网络初始化**：使用标准的 PyTorch `nn.Module` 定义方式
- **网络定义**：完全兼容 PyTorch 的模型定义规范
- **训练与微调**：使用标准的 PyTorch 训练循环，无需修改现有代码
- **特征保存**：采用一个专门的类来保存网络参数（即"网络所提取的特征"），便于后续分析和可视化

### 3. 训练流程管理（默认使用Hydra和Wandb）

ezflt **默认使用 Hydra 和 Wandb** 管理多组实验：

- **Wandb**：自动记录训练指标、超参数和特征信息
- **Hydra**：配置管理和参数扫描，支持多组实验
- **无缝集成**：Trainer 自动集成实验管理，无需额外配置
- **灵活控制**：可通过参数禁用任一功能

> 详细使用说明请查看 [README_HYDRA_WANDB.md](README_HYDRA_WANDB.md)

## 快速开始

`ezflt` 仅使用 `torch` 和 `seaborn` 等基础工具，易于设置和运行。

### 环境配置（推荐使用uv）

本项目使用 `uv` + `pyproject.toml` 进行环境管理。

#### 安装uv

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

或者使用pip安装：
```bash
pip install uv
```

#### 使用uv安装项目

```bash
# 创建虚拟环境并安装依赖
uv venv

# 激活虚拟环境
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 安装项目（开发模式，包含dev依赖）
uv pip install -e ".[dev]"

# 或者只安装基础依赖
uv pip install -e .
```

#### 传统方式（conda + pip）

如果你偏好使用conda：

**Linux:**
```bash
conda create -n flt python=3.9 -y
conda activate flt
pip install -r requirements.txt
```

**Windows:**
```powershell
conda create -n flt python=3.9 -y
conda activate flt
pip install -r requirements.txt
```

### 运行测试

```bash
# 使用uv运行
uv run pytest tests/ -v

# 或激活环境后运行
pytest tests/ -v
```

安装完成后，你可以在 Linux/Windows 设备上（支持或不支持 GPU）运行示例脚本。例如，我们复现了经典特征学习理论论文的结果：[Feature Purification: How Adversarial Training Performs Robust Deep Learning., Allen-Zhu & Li (2020)](https://arxiv.org/abs/2005.10190)。

### uv常用命令

```bash
# 创建虚拟环境
uv venv

# 安装项目（开发模式，包含dev依赖）
uv pip install -e ".[dev]"

# 安装可选依赖（如wandb）
uv pip install -e ".[wandb]"

# 运行Python脚本
uv run python script.py

# 运行测试
uv run pytest tests/ -v

# 更新依赖
uv pip install --upgrade-package package-name

# 查看已安装的包
uv pip list
```

### 运行示例

```bash
# 使用uv运行
uv run python example.py

# 或激活环境后运行
python example.py
```

你可以在 `config.py` 中配置可选参数。

### 参数配置

主要配置项在 `config.py` 中，包括：
- 数据生成参数
- 模型架构参数
- 训练超参数
- 评估指标
- 可视化选项

## 使用示例

### 基础用法：特征追踪

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

# 创建特征追踪器（追踪指定层）
tracker = FeatureTracker(
    model=model,
    layers=["0", "2"],  # 追踪第一层和第三层
    track_weights=True,
)

# 使用上下文管理器自动管理追踪生命周期
with tracker:
    # 你的训练循环
    for epoch in range(10):
        # ... 训练代码 ...
        tracker.on_epoch_start(epoch)
        # ... batch循环 ...
        tracker.on_batch_end(batch_idx)

# 获取追踪的特征
features = tracker.get_features()
conv1_weight = tracker.get_feature("0.weight")

# 可视化特征演化
if conv1_weight:
    visualize_feature_evolution(conv1_weight, metric="norm")
```

### 高级用法：集成训练器和回调系统

```python
from ezflt import FeatureTracker, FeatureTrackingCallback, WandbCallback, Trainer
from torch.utils.data import DataLoader

# 创建特征追踪器
tracker = FeatureTracker(
    model=model,
    layers=["conv1", "fc"],
    track_weights=True,
    track_frequency=10,  # 每10个batch记录一次
)

# 创建回调列表
callbacks = [
    FeatureTrackingCallback(tracker),  # 自动追踪特征
    WandbCallback(project="flt-research"),  # 集成wandb（可选）
]

# 创建训练器
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    callbacks=callbacks,
)

# 训练模型（自动追踪特征）
trainer.fit(train_loader=dataloader, num_epochs=10)

# 可视化结果
from ezflt.plot import visualize_features_heatmap, visualize_parameter_comparison

# 热力图可视化
visualize_features_heatmap(tracker, epoch=9)

# 参数对比（不同epoch）
visualize_parameter_comparison(
    tracker.get_feature("conv1.weight"),
    epochs=[0, 5, 9]
)
```

## 核心API

### FeatureTracker

特征追踪器，非侵入式地追踪模型参数变化。

```python
tracker = FeatureTracker(
    model: nn.Module,              # PyTorch模型
    layers: Optional[List[str]],   # 要追踪的层名称列表
    track_weights: bool = True,    # 是否追踪权重
    track_gradients: bool = False, # 是否追踪梯度
    track_frequency: int = 1,      # 追踪频率（每N个batch）
)
```

**主要方法**：
- `start()`: 开始追踪
- `stop()`: 停止追踪
- `get_features()`: 获取所有追踪的特征
- `get_feature(name)`: 获取指定特征
- `on_epoch_start(epoch)`: 生命周期钩子
- `on_batch_end(batch_idx)`: 生命周期钩子

### Feature

保存网络参数（特征）的类。

```python
feature = Feature(name="conv1.weight", layer_name="conv1")
feature.append(value, epoch, batch_idx)  # 添加参数值
latest = feature.get_latest()            # 获取最新值
at_epoch = feature.get_at_epoch(5)       # 获取指定epoch的值
```

### Trainer

统一的训练接口，支持回调系统。

```python
trainer = Trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    callbacks: Optional[List[Callback]],
)
trainer.fit(train_loader, num_epochs, val_loader=None)
```

### Callbacks

#### FeatureTrackingCallback

自动在训练过程中追踪特征。

```python
callback = FeatureTrackingCallback(tracker)
```

#### WandbCallback

集成wandb，自动记录训练指标和特征信息。

```python
callback = WandbCallback(project="my-project", config=hyperparams)
```

### 可视化函数

- `visualize_feature_evolution(feature, metric="norm")`: 可视化单个特征的演化
- `visualize_features_heatmap(tracker, epoch=None)`: 多特征热力图
- `visualize_parameter_comparison(feature, epochs)`: 不同epoch的参数对比

## 设计特点

- **非侵入式**：不需要修改模型定义，通过Hook自动追踪
- **渐进式复杂度**：简单场景默认参数，复杂场景可深度定制
- **PyTorch原生兼容**：完全兼容标准PyTorch训练流程
- **统一接口**：参考优秀库的设计，接口清晰统一
- **默认实验管理**：自动集成Wandb和Hydra，开箱即用

## 更多示例

查看 `examples/basic_usage.py` 获取完整的使用示例。

## API 文档

完整的 API 文档请查看 [docs/](docs/) 目录。

### 本地查看文档

```bash
# 安装文档依赖
uv pip install -e ".[docs]"

# 启动文档服务器
mkdocs serve

# 浏览器访问 http://127.0.0.1:8000
```

### 构建文档

```bash
mkdocs build
```

## 作者

**Yanbo Chen** - 项目主要开发者

本项目在开发过程中得到了 Cursor Agent 的协助。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。