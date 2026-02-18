# ezflt API 文档

欢迎使用 **ezflt** (Easy Feature Learning Theory) API 文档。

## 简介

ezflt 是一个轻量级库，旨在促进特征学习理论研究与实验。它提供了特征学习理论研究的各个阶段的简单实现，从数值评估到结果可视化。

## 核心定位

**重要**：这是一个关注模型训练和微调过程中**网络参数如何从数据中提取有用特征**的库。ezflt 的核心目标是帮助研究者理解和可视化神经网络在训练过程中如何学习特征。

## 主要功能

### 1. 特征提取过程追踪与可视化

- 自动追踪指定层的网络参数在训练过程中的变化
- 可视化参数如何从初始状态演化到最终状态
- 分析网络如何从数据中提取和优化特征

### 2. PyTorch 原生兼容性

- 完全兼容标准 PyTorch 训练流程
- 非侵入式设计，不需要修改模型定义
- 使用 Hook 机制自动追踪参数变化

### 3. 训练流程管理

- 支持 wandb 等实验管理工具
- 自动记录训练指标、超参数和模型检查点
- 便于实验对比和结果复现

## 快速链接

- [快速开始](getting_started.md) - 安装和基本使用
- [API 参考](api/tracker.md) - 完整的 API 文档
- [示例](examples.md) - 使用示例

## 安装

```bash
# 使用uv（推荐）
uv venv
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate    # Linux/Mac
uv pip install -e ".[dev]"
```

## 作者

**Yanbo Chen** - 项目主要开发者

本项目在开发过程中得到了 Cursor Agent 的协助。

## 许可证

本项目采用 MIT 许可证。

