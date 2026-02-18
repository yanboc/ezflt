"""
训练回调系统

实现生命周期钩子，支持在训练过程的特定时刻执行自定义逻辑。
参考PyTorch Lightning的Callback系统设计。
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class Callback(ABC):
    """
    回调基类
    
    定义训练生命周期的钩子接口。
    """
    
    def on_train_start(self, trainer, model):
        """训练开始"""
        pass
    
    def on_train_end(self, trainer, model):
        """训练结束"""
        pass
    
    def on_epoch_start(self, trainer, model, epoch: int):
        """Epoch开始"""
        pass
    
    def on_epoch_end(self, trainer, model, epoch: int):
        """Epoch结束"""
        pass
    
    def on_batch_start(self, trainer, model, batch, batch_idx: int):
        """Batch开始"""
        pass
    
    def on_batch_end(self, trainer, model, batch, batch_idx: int, loss):
        """Batch结束"""
        pass


class FeatureTrackingCallback(Callback):
    """
    特征追踪回调
    
    在训练过程中自动追踪模型参数的变化。
    """
    
    def __init__(self, tracker):
        """
        Args:
            tracker: FeatureTracker实例
        """
        self.tracker = tracker
    
    def on_train_start(self, trainer, model):
        """训练开始时启动追踪"""
        self.tracker.start()
    
    def on_train_end(self, trainer, model):
        """训练结束时停止追踪"""
        self.tracker.stop()
    
    def on_epoch_start(self, trainer, model, epoch: int):
        """Epoch开始时更新状态"""
        self.tracker.on_epoch_start(epoch)
    
    def on_batch_end(self, trainer, model, batch, batch_idx: int, loss):
        """Batch结束时更新状态"""
        self.tracker.on_batch_end(batch_idx)


class WandbCallback(Callback):
    """
    Wandb集成回调
    
    自动将训练指标、超参数和特征信息记录到wandb。
    """
    
    def __init__(self, project: Optional[str] = None, config: Optional[Dict] = None):
        """
        Args:
            project: wandb项目名称
            config: 实验配置（超参数等）
        """
        self.project = project
        self.config = config or {}
        self.wandb = None
        self._initialized = False
    
    def _init_wandb(self):
        """延迟初始化wandb（避免导入错误）"""
        if not self._initialized:
            try:
                import wandb
                self.wandb = wandb
                if not wandb.run:
                    wandb.init(project=self.project, config=self.config)
                self._initialized = True
            except ImportError:
                raise ImportError(
                    "wandb未安装。请使用 'pip install wandb' 安装。"
                )
    
    def on_train_start(self, trainer, model):
        """训练开始时初始化wandb"""
        self._init_wandb()
        if self.wandb:
            # 记录模型结构
            self.wandb.watch(model, log='all', log_freq=100)
    
    def on_epoch_end(self, trainer, model, epoch: int):
        """Epoch结束时记录指标"""
        if self.wandb:
            # 记录训练指标
            metrics = trainer.get_metrics()
            if metrics:
                self.wandb.log(metrics, step=epoch)
    
    def on_batch_end(self, trainer, model, batch, batch_idx: int, loss):
        """Batch结束时记录loss"""
        if self.wandb and batch_idx % 100 == 0:  # 每100个batch记录一次
            self.wandb.log({"batch_loss": loss.item()}, step=batch_idx)
    
    def log_features(self, features: Dict[str, Any], epoch: int):
        """记录特征信息到wandb"""
        if self.wandb:
            # 记录特征统计信息（如参数范数等）
            feature_metrics = {}
            for name, feature in features.items():
                if hasattr(feature, 'get_latest'):
                    latest = feature.get_latest()
                    if latest is not None:
                        feature_metrics[f"{name}/norm"] = latest.norm().item()
                        feature_metrics[f"{name}/mean"] = latest.mean().item()
                        feature_metrics[f"{name}/std"] = latest.std().item()
            
            if feature_metrics:
                self.wandb.log(feature_metrics, step=epoch)

