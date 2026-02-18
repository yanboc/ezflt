"""
单次实验完整示例

展示如何使用ezflt进行单次实验，包括：
1. 模型定义
2. 数据准备
3. 特征追踪
4. Wandb自动记录
5. 训练和可视化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ezflt import (
    Trainer,
    FeatureTracker,
    FeatureTrackingCallback,
    ExperimentManager,
    visualize_feature_evolution,
    visualize_features_heatmap,
)


class SimpleCNN(nn.Module):
    """简单的CNN模型用于演示"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("=" * 60)
    print("ezflt 单次实验示例")
    print("=" * 60)
    
    # ========== 1. 创建实验管理器 ==========
    print("\n1. 创建实验管理器...")
    exp_manager = ExperimentManager(
        project_name="ezflt-single-experiment",
        experiment_name="simple-cnn-demo",
        use_wandb=True,  # 启用Wandb
        use_hydra=False,  # 单次实验不使用Hydra
    )
    exp_manager._init_wandb()
    print(f"   ✓ Wandb项目: {exp_manager.project_name}")
    print(f"   ✓ 实验名称: {exp_manager.experiment_name}")
    
    # ========== 2. 准备数据 ==========
    print("\n2. 准备数据...")
    # 生成模拟数据（实际使用时替换为真实数据）
    X_train = torch.randn(200, 1, 28, 28)
    y_train = torch.randint(0, 10, (200,))
    X_val = torch.randn(50, 1, 28, 28)
    y_val = torch.randint(0, 10, (50,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   ✓ 训练样本: {len(train_dataset)}")
    print(f"   ✓ 验证样本: {len(val_dataset)}")
    print(f"   ✓ Batch大小: 32")
    
    # ========== 3. 创建模型和优化器 ==========
    print("\n3. 创建模型和优化器...")
    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 记录模型信息到Wandb
    if exp_manager.wandb_run:
        exp_manager.wandb_run.config.update({
            "model": "SimpleCNN",
            "num_classes": 10,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 5,
        })
    
    print(f"   ✓ 模型: SimpleCNN")
    print(f"   ✓ 优化器: Adam (lr=0.001)")
    print(f"   ✓ 损失函数: CrossEntropyLoss")
    
    # ========== 4. 创建特征追踪器 ==========
    print("\n4. 创建特征追踪器...")
    tracker = FeatureTracker(
        model=model,
        layers=["conv1", "conv2", "fc1", "fc2"],  # 追踪这些层
        track_weights=True,
        track_frequency=1,  # 每个batch都记录
    )
    
    print(f"   ✓ 追踪层: {tracker.layers}")
    print(f"   ✓ 追踪权重: {tracker.track_weights}")
    
    # ========== 5. 创建训练器 ==========
    print("\n5. 创建训练器...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,  # 指定损失函数
        callbacks=[
            FeatureTrackingCallback(tracker),  # 自动追踪特征
        ],
        use_wandb=True,  # 自动集成Wandb
        use_hydra=False,
    )
    
    print(f"   ✓ 训练器已创建")
    print(f"   ✓ Wandb自动集成: 是")
    
    # ========== 6. 开始训练 ==========
    print("\n6. 开始训练...")
    print("-" * 60)
    
    num_epochs = 5
    tracker.start()  # 开始追踪
    
    # 注意：Trainer会自动初始化实验管理器
    
    for epoch in range(num_epochs):
        tracker.on_epoch_start(epoch)
        
        # 训练一个epoch
        trainer.fit(train_loader=train_loader, num_epochs=1, val_loader=val_loader)
        
        # 记录特征信息到Wandb
        features = tracker.get_features()
        if trainer.experiment_manager:
            trainer.experiment_manager.log_features(features, epoch=epoch)
        
        # 打印训练指标
        metrics = trainer.get_metrics()
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {metrics.get('loss', 'N/A'):.4f}")
        if 'val_loss' in metrics:
            print(f"  Val Loss: {metrics['val_loss']:.4f}")
        print()
    
    tracker.stop()  # 停止追踪
    
    print("-" * 60)
    print("   ✓ 训练完成")
    
    # ========== 7. 可视化特征演化 ==========
    print("\n7. 可视化特征演化...")
    try:
        # 可视化conv1权重的演化
        conv1_weight = tracker.get_feature("conv1.weight")
        if conv1_weight:
            visualize_feature_evolution(
                conv1_weight,
                metric="norm",
                save_path="conv1_evolution.png"
            )
            print("   ✓ conv1权重演化图已保存: conv1_evolution.png")
        
        # 可视化所有特征的热力图
        visualize_features_heatmap(
            tracker,
            epoch=num_epochs - 1,
            save_path="features_heatmap.png"
        )
        print("   ✓ 特征热力图已保存: features_heatmap.png")
    except Exception as e:
        print(f"   ⚠ 可视化失败: {e}")
    
    # ========== 8. 完成实验 ==========
    print("\n8. 完成实验...")
    # 完成实验管理器（如果Trainer使用了独立的实验管理器，也需要完成）
    if trainer.experiment_manager and trainer.experiment_manager != exp_manager:
        trainer.experiment_manager.finish()
    exp_manager.finish()
    print("   ✓ 实验已记录到Wandb")
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print("\n请访问 https://wandb.ai 查看实验详情。")
    print(f"项目: {exp_manager.project_name}")
    print(f"实验: {exp_manager.experiment_name}")


if __name__ == "__main__":
    main()

