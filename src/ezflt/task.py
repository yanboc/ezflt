"""
学习任务模块

实现FLT中的"学习任务四要素"框架，用于刻画和配置一个完整的学习任务。
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class LearningTask:
    """
    学习任务四要素
    
    根据FLT理论，一个学习任务由四个要素组成：
    1. 数据（Data）
    2. 模型（Model）
    3. 评价标准（Evaluation）
    4. 优化算法（Optimizer）
    """
    
    # 1. 数据（必需字段）
    data: Union[Dataset, DataLoader, Dict[str, Any]]
    data_type: str  # "synthetic" (信号-噪音模型) 或 "real" (真实数据)
    
    # 2. 模型（必需字段）
    model: nn.Module
    model_type: str  # "shallow" (浅层模型) 或 "standard" (标准模型)
    
    # 3. 评价标准（必需字段）
    loss_function: Union[str, nn.Module, callable]
    
    # 4. 优化算法（必需字段）
    optimizer: torch.optim.Optimizer
    optimizer_type: str  # "sgd", "adam", "adamw"等
    
    # 可选字段（有默认值，必须放在最后）
    data_name: Optional[str] = None  # 数据名称（如"MNIST", "CIFAR-10"等）
    model_name: Optional[str] = None  # 模型名称（如"ResNet-18", "GPT-2-200M"等）
    loss_name: Optional[str] = None  # 损失函数名称（如"cross_entropy", "mse"等）
    optimizer_config: Optional[Dict[str, Any]] = None  # 优化器配置（如lr, weight_decay等）
    task_name: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """验证四要素的完整性"""
        if self.data is None:
            raise ValueError("数据（Data）不能为空")
        if self.model is None:
            raise ValueError("模型（Model）不能为空")
        if self.loss_function is None:
            raise ValueError("评价标准（Loss Function）不能为空")
        if self.optimizer is None:
            raise ValueError("优化算法（Optimizer）不能为空")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取任务摘要"""
        return {
            "task_name": self.task_name,
            "description": self.description,
            "data": {
                "type": self.data_type,
                "name": self.data_name,
            },
            "model": {
                "type": self.model_type,
                "name": self.model_name,
            },
            "loss": {
                "name": self.loss_name or str(self.loss_function),
            },
            "optimizer": {
                "type": self.optimizer_type,
                "config": self.optimizer_config,
            },
        }


def create_learning_task_from_config(config: Dict[str, Any]) -> LearningTask:
    """
    从配置文件创建学习任务
    
    支持从config.py格式的配置创建LearningTask实例。
    """
    # 这里需要根据实际的config结构来实现
    # 暂时提供接口框架
    raise NotImplementedError("待实现：根据config创建LearningTask")


def create_synthetic_task(
    signal_dim: int,
    noise_level: float,
    model: nn.Module,
    loss_fn: Union[str, nn.Module],
    optimizer_type: str = "sgd",
    lr: float = 0.01,
) -> LearningTask:
    """
    创建生成数据（信号-噪音模型）的学习任务
    
    Args:
        signal_dim: 信号维度（d维正交基）
        noise_level: 噪音水平
        model: 浅层模型
        loss_fn: 损失函数
        optimizer_type: 优化器类型
        lr: 学习率
    """
    # 这里需要实现信号-噪音数据生成
    # 暂时提供接口框架
    raise NotImplementedError("待实现：生成数据任务创建")


def create_real_data_task(
    dataset_name: str,
    model: nn.Module,
    loss_fn: Union[str, nn.Module],
    optimizer_type: str = "adam",
    lr: float = 0.001,
) -> LearningTask:
    """
    创建真实数据的学习任务
    
    Args:
        dataset_name: 数据集名称（"MNIST", "CIFAR-10"等）
        model: 模型（ResNet-18, GPT-2-200M等）
        loss_fn: 损失函数
        optimizer_type: 优化器类型
        lr: 学习率
    """
    # 这里需要实现真实数据加载
    # 暂时提供接口框架
    raise NotImplementedError("待实现：真实数据任务创建")

