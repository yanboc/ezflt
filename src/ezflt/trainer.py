"""
训练器模块

提供统一的训练接口，支持回调系统和特征追踪。
参考PyTorch Lightning的设计，但保持PyTorch原生兼容。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any
from .callbacks import Callback, FeatureTrackingCallback
from .tracker import FeatureTracker


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
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: PyTorch模型
            optimizer: 优化器
            callbacks: 回调列表
            device: 设备（CPU/GPU）
        """
        self.model = model
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            
            # 计算损失（简化处理，实际应该根据任务类型）
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

