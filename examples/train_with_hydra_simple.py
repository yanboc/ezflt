"""
简单的Hydra训练脚本

使用@hydra.main装饰器，支持命令行参数覆盖和多组实验。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hydra import compose, initialize, main
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ezflt import Trainer, ExperimentManager


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


@main(config_path="configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    """训练函数，使用Hydra配置"""
    print(f"实验: {cfg.experiment.name}")
    print(f"学习率: {cfg.training.learning_rate}")
    print(f"Epochs: {cfg.training.epochs}")
    
    # 创建实验管理器
    exp_manager = ExperimentManager(
        project_name=cfg.wandb.project if cfg.wandb.enabled else None,
        experiment_name=cfg.experiment.name,
        use_wandb=cfg.wandb.enabled,
        use_hydra=True,
    )
    
    # 设置Hydra配置
    exp_manager.hydra_cfg = cfg
    exp_manager._hydra_initialized = True
    
    # 初始化Wandb
    if cfg.wandb.enabled:
        from omegaconf import OmegaConf
        exp_manager._init_wandb(config=OmegaConf.to_container(cfg, resolve=True))
    
    # 准备数据
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True)
    
    # 创建模型和训练器
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        use_wandb=cfg.wandb.enabled,
        use_hydra=False,  # 已通过ExperimentManager管理
        experiment_manager=exp_manager,
    )
    
    # 训练
    trainer.fit(train_loader=dataloader, num_epochs=cfg.training.epochs)
    
    # 完成
    exp_manager.finish()
    print("训练完成！")


if __name__ == "__main__":
    train()

