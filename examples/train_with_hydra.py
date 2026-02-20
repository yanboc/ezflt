"""
使用Hydra管理配置的训练脚本示例

展示如何使用Hydra进行配置管理和多组实验。
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ezflt import Trainer, FeatureTracker, FeatureTrackingCallback, ExperimentManager

# 配置目录：相对本脚本所在目录，指向项目根下的 configs（任意 CWD 均可运行）
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


def train_with_config(cfg: DictConfig):
    """
    使用Hydra配置进行训练
    
    Args:
        cfg: Hydra配置对象
    """
    # 设置随机种子
    if hasattr(cfg.experiment, 'seed'):
        torch.manual_seed(cfg.experiment.seed)
    
    # 创建实验管理器
    exp_manager = ExperimentManager(
        config_path=str(_CONFIG_DIR),
        project_name=cfg.wandb.project if cfg.wandb.enabled else None,
        experiment_name=cfg.experiment.name,
        use_wandb=cfg.wandb.enabled,
        use_hydra=True,  # 使用Hydra
    )
    
    # 初始化Hydra（会从cfg中读取）
    exp_manager.hydra_cfg = cfg
    exp_manager._hydra_initialized = True
    
    # 初始化Wandb
    if cfg.wandb.enabled:
        exp_manager._init_wandb(config=OmegaConf.to_container(cfg, resolve=True))
    
    # 准备数据
    train_size = cfg.data.train_size
    test_size = cfg.data.test_size
    batch_size = cfg.data.batch_size
    
    X_train = torch.randn(train_size, 10)
    y_train = torch.randint(0, 2, (train_size,))
    X_val = torch.randn(test_size, 10)
    y_val = torch.randint(0, 2, (test_size,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.data.shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = SimpleModel(input_dim=10, output_dim=2)
    
    # 创建优化器（从配置读取学习率）
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建特征追踪器（如果启用）
    tracker = None
    callbacks = []
    if cfg.tracking.enabled:
        layers = cfg.tracking.layers if cfg.tracking.layers else None
        tracker = FeatureTracker(
            model=model,
            layers=layers,
            track_weights=cfg.tracking.track_weights,
            track_gradients=cfg.tracking.track_gradients,
            track_frequency=cfg.tracking.track_frequency,
        )
        callbacks.append(FeatureTrackingCallback(tracker))
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=callbacks,
        use_wandb=cfg.wandb.enabled,
        use_hydra=False,  # 已经通过ExperimentManager管理
        experiment_manager=exp_manager,
    )
    
    # 训练
    if tracker:
        tracker.start()
    
    trainer.fit(
        train_loader=train_loader,
        num_epochs=cfg.training.epochs,
        val_loader=val_loader,
    )
    
    if tracker:
        tracker.stop()
        
        # 记录特征信息到Wandb
        if cfg.wandb.enabled:
            features = tracker.get_features()
            exp_manager.log_features(features, epoch=cfg.training.epochs - 1)
    
    # 完成实验
    exp_manager.finish()
    
    print(f"训练完成！实验: {cfg.experiment.name}")


def main_with_hydra(cfg: DictConfig):
    """
    Hydra主函数
    
    使用@hydra.main装饰器，支持：
    - 命令行参数覆盖：python train_with_hydra.py training.learning_rate=0.01
    - 多组实验：python train_with_hydra.py -m training.learning_rate=0.001,0.01,0.1
    """
    print("=" * 60)
    print("使用Hydra配置进行训练")
    print("=" * 60)
    print(f"\n配置摘要:")
    print(f"  实验名称: {cfg.experiment.name}")
    print(f"  学习率: {cfg.training.learning_rate}")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch Size: {cfg.data.batch_size}")
    print(f"  Wandb启用: {cfg.wandb.enabled}")
    print()
    
    train_with_config(cfg)


if __name__ == "__main__":
    try:
        from hydra import main as hydra_main

        @hydra_main(config_path=str(_CONFIG_DIR), config_name="config", version_base="1.1")
        def main(cfg: DictConfig):
            main_with_hydra(cfg)

        main()
    except ImportError:
        print("错误: 需要安装hydra-core")
        print("安装命令: pip install hydra-core")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

