"""
EasyFLT: 特征学习理论友好库

支持FLT研究的工具库，提供特征追踪、可视化和训练管理功能。
"""

from .tracker import FeatureTracker, Feature
from .callbacks import Callback, FeatureTrackingCallback, WandbCallback
from .trainer import Trainer
from .task import LearningTask, create_learning_task_from_config
from .data import Generator
from .model import FLTNetwork
from .plot import (
    visualize_feature_evolution,
    visualize_features_heatmap,
    visualize_parameter_comparison,
    visualize,
)

__version__ = "0.1.0"

__all__ = [
    "FeatureTracker",
    "Feature",
    "Callback",
    "FeatureTrackingCallback",
    "WandbCallback",
    "Trainer",
    "LearningTask",
    "create_learning_task_from_config",
    "Generator",
    "FLTNetwork",
    "visualize_feature_evolution",
    "visualize_features_heatmap",
    "visualize_parameter_comparison",
    "visualize",
]

