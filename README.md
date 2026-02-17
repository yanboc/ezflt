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

### 3. 训练流程管理

集成高效的管理工具来管理训练流程：

- 支持 [wandb](https://wandb.ai/) 等实验管理工具
- 自动记录训练指标、超参数和模型检查点
- 便于实验对比和结果复现

## 快速开始

`ezflt` 仅使用 `torch` 和 `seaborn` 等基础工具，易于设置和运行。

### 安装

#### Linux

```bash
conda create -n flt python=3.9 -y
conda activate flt
pip install -r requirements.txt
```

#### Windows

```powershell
conda create -n flt python=3.9 -y
conda activate flt
pip install -r requirements.txt
```

安装完成后，你可以在 Linux/Windows 设备上（支持或不支持 GPU）运行示例脚本。例如，我们复现了经典特征学习理论论文的结果：[Feature Purification: How Adversarial Training Performs Robust Deep Learning., Allen-Zhu & Li (2020)](https://arxiv.org/abs/2005.10190)。

### 运行示例

```bash
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