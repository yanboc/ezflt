"""
特征追踪核心模块

实现非侵入式的参数追踪功能，支持在训练过程中自动记录网络参数的变化。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable
from collections import defaultdict


class Feature:
    """
    保存网络参数（即"网络所提取的特征"）的类
    
    这个类用于存储和访问训练过程中记录的网络参数。
    """
    
    def __init__(self, name: str, layer_name: str):
        """
        Args:
            name: 特征名称（如"conv1.weight"）
            layer_name: 层名称（如"conv1"）
        """
        self.name = name
        self.layer_name = layer_name
        self.values: List[torch.Tensor] = []  # 存储每个时间步的参数值
        self.epochs: List[int] = []  # 记录对应的epoch
        self.batch_indices: List[int] = []  # 记录对应的batch索引
    
    def append(self, value: torch.Tensor, epoch: int, batch_idx: int = 0):
        """添加一个时间步的参数值"""
        self.values.append(value.detach().clone())
        self.epochs.append(epoch)
        self.batch_indices.append(batch_idx)
    
    def get_at_epoch(self, epoch: int) -> Optional[torch.Tensor]:
        """获取指定epoch的参数值"""
        if epoch in self.epochs:
            idx = self.epochs.index(epoch)
            return self.values[idx]
        return None
    
    def get_latest(self) -> Optional[torch.Tensor]:
        """获取最新的参数值"""
        return self.values[-1] if self.values else None
    
    def __len__(self):
        return len(self.values)


class FeatureTracker:
    """
    特征追踪器
    
    非侵入式地追踪模型在训练过程中参数的变化。
    参考Captum的统一接口设计和渐进式复杂度原则。
    """
    
    def __init__(
        self,
        model: nn.Module,
        layers: Optional[Union[str, List[str]]] = None,
        track_weights: bool = True,
        track_gradients: bool = False,
        track_frequency: int = 1,  # 每N个batch记录一次
    ):
        """
        Args:
            model: 要追踪的PyTorch模型
            layers: 要追踪的层名称列表，如果为None则追踪所有层
            track_weights: 是否追踪权重
            track_gradients: 是否追踪梯度
            track_frequency: 追踪频率（每N个batch记录一次）
        """
        self.model = model
        self.track_weights = track_weights
        self.track_gradients = track_gradients
        self.track_frequency = track_frequency
        
        # 确定要追踪的层
        if layers is None:
            self.layers = [name for name, _ in model.named_modules() if name]
        elif isinstance(layers, str):
            self.layers = [layers]
        else:
            self.layers = layers
        
        # 存储特征
        self.features: Dict[str, Feature] = {}
        self.hooks: List = []  # 存储hook句柄，用于清理
        
        # 训练状态
        self.current_epoch = 0
        self.current_batch_idx = 0
        self.is_tracking = False
    
    def _register_hooks(self):
        """注册hook以追踪参数变化"""
        for name, module in self.model.named_modules():
            if name in self.layers or not self.layers:
                # 追踪权重
                if self.track_weights:
                    for param_name, param in module.named_parameters(recurse=False):
                        full_name = f"{name}.{param_name}" if name else param_name
                        feature = Feature(full_name, name)
                        self.features[full_name] = feature
                        
                        # 注册forward hook
                        hook_handle = module.register_forward_hook(
                            self._create_weight_hook(full_name)
                        )
                        self.hooks.append(hook_handle)
                
                # 追踪梯度
                if self.track_gradients:
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            full_name = f"{name}.{param_name}.grad" if name else f"{param_name}.grad"
                            feature = Feature(full_name, name)
                            self.features[full_name] = feature
                            
                            # 注册backward hook
                            hook_handle = param.register_hook(
                                self._create_gradient_hook(full_name)
                            )
                            self.hooks.append(hook_handle)
    
    def _create_weight_hook(self, feature_name: str) -> Callable:
        """创建权重追踪hook"""
        def hook_fn(module, input, output):
            if self.is_tracking and self.current_batch_idx % self.track_frequency == 0:
                # 获取权重
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight
                    self.features[feature_name].append(
                        weight, self.current_epoch, self.current_batch_idx
                    )
        return hook_fn
    
    def _create_gradient_hook(self, feature_name: str) -> Callable:
        """创建梯度追踪hook"""
        def hook_fn(grad):
            if self.is_tracking and self.current_batch_idx % self.track_frequency == 0:
                if grad is not None:
                    self.features[feature_name].append(
                        grad, self.current_epoch, self.current_batch_idx
                    )
            return grad
        return hook_fn
    
    def start(self):
        """开始追踪"""
        if not self.is_tracking:
            self._register_hooks()
            self.is_tracking = True
    
    def stop(self):
        """停止追踪并清理hooks"""
        if self.is_tracking:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            self.is_tracking = False
    
    def on_epoch_start(self, epoch: int):
        """生命周期钩子：epoch开始"""
        self.current_epoch = epoch
        self.current_batch_idx = 0
    
    def on_batch_end(self, batch_idx: int):
        """生命周期钩子：batch结束"""
        self.current_batch_idx = batch_idx
    
    def get_features(self) -> Dict[str, Feature]:
        """获取所有追踪的特征"""
        return self.features
    
    def get_feature(self, name: str) -> Optional[Feature]:
        """获取指定名称的特征"""
        return self.features.get(name)
    
    def __enter__(self):
        """上下文管理器：进入时自动开始追踪"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：退出时自动停止追踪"""
        self.stop()

