"""
训练器模块

提供统一的训练接口，支持回调系统和特征追踪。
参考PyTorch Lightning的设计，但保持PyTorch原生兼容。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any
from .callbacks import Callback, FeatureTrackingCallback, WandbCallback
from .tracker import FeatureTracker
from .experiment import ExperimentManager


class Trainer:
    """
    训练器
    
    提供标准的训练循环，支持回调系统和特征追踪。
    保持PyTorch原生兼容，不强制继承。
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
        experiment_manager: Optional[ExperimentManager] = None,
        use_wandb: bool = True,
        use_hydra: bool = True,
    ):
        """
        Args:
            model: PyTorch模型
            optimizer: 优化器
            criterion: 损失函数（如果为None，会自动推断）
            callbacks: 回调列表
            device: 设备（CPU/GPU）
            experiment_manager: 实验管理器（如果为None，会自动创建）
            use_wandb: 是否使用Wandb（默认True）
            use_hydra: 是否使用Hydra（默认True）
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion  # 自定义损失函数
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 实验管理（默认启用）
        if experiment_manager is None and (use_wandb or use_hydra):
            self.experiment_manager = ExperimentManager(
                use_wandb=use_wandb,
                use_hydra=use_hydra,
            )
        else:
            self.experiment_manager = experiment_manager
        
        # 自动添加Wandb回调（如果启用）
        self.callbacks = callbacks or []
        if use_wandb and self.experiment_manager and self.experiment_manager.use_wandb:
            # 检查是否已有WandbCallback
            has_wandb = any(isinstance(cb, WandbCallback) for cb in self.callbacks)
            if not has_wandb:
                wandb_callback = WandbCallback(
                    project=self.experiment_manager.project_name,
                    config=self.experiment_manager.get_config(),
                )
                self.callbacks.append(wandb_callback)
        
        self.model.to(self.device)
        self.current_epoch = 0
        self.metrics: Dict[str, float] = {}
    
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            val_loader: 验证数据加载器（可选）
        """
        # 初始化实验管理器
        if self.experiment_manager:
            self.experiment_manager._init_hydra()
            self.experiment_manager._init_wandb(
                config=self.experiment_manager.get_config()
            )
        
        # 训练开始回调
        for callback in self.callbacks:
            callback.on_train_start(self, self.model)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Epoch开始回调
            for callback in self.callbacks:
                callback.on_epoch_start(self, self.model, epoch)
            
            # 训练一个epoch
            self._train_epoch(train_loader)
            
            # 验证（如果有）
            if val_loader is not None:
                self._validate(val_loader)
            
            # Epoch结束回调
            for callback in self.callbacks:
                callback.on_epoch_end(self, self.model, epoch)
        
        # 训练结束回调
        for callback in self.callbacks:
            callback.on_train_end(self, self.model)
        
        # 完成实验管理
        if self.experiment_manager:
            self.experiment_manager.finish()
    
    def _train_epoch(self, train_loader: DataLoader):
        """训练一个epoch"""
        self.model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            # Batch开始回调
            for callback in self.callbacks:
                callback.on_batch_start(self, self.model, batch, batch_idx)
            
            # 前向传播
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch.get("input"), batch.get("target")
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            
            # 计算损失
            if self.criterion is not None:
                # 使用自定义损失函数
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    # 自编码器等任务（无targets）
                    loss = self.criterion(outputs, inputs)
            else:
                # 自动推断损失函数
                if targets is not None:
                    criterion = nn.CrossEntropyLoss() if len(outputs.shape) > 1 else nn.MSELoss()
                    loss = criterion(outputs, targets)
                else:
                    # 自编码器等任务
                    loss = nn.MSELoss()(outputs, inputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Batch结束回调
            for callback in self.callbacks:
                callback.on_batch_end(self, self.model, batch, batch_idx, loss)
            
            # 更新指标
            self.metrics["loss"] = loss.item()
    
    def _validate(self, val_loader: DataLoader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch.get("input"), batch.get("target")
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # 使用相同的损失函数
                if self.criterion is not None:
                    if targets is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = self.criterion(outputs, inputs)
                else:
                    # 自动推断
                    if targets is not None:
                        criterion = nn.CrossEntropyLoss() if len(outputs.shape) > 1 else nn.MSELoss()
                        loss = criterion(outputs, targets)
                    else:
                        loss = nn.MSELoss()(outputs, inputs)
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            self.metrics["val_loss"] = total_loss / num_batches
        
        self.model.train()
    
    def get_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        return self.metrics.copy()

