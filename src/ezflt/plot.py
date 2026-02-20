"""
特征可视化模块

提供参数变化过程的可视化功能，支持多种可视化方式。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
from .tracker import Feature, FeatureTracker

def visualize_feature_evolution(
    feature: Feature,
    save_path: Optional[str] = None,
    metric: str = "norm",
    figsize: tuple = (10, 6)
):
    """
    可视化单个特征的演化过程
    
    Args:
        feature: Feature对象
        save_path: 保存路径，如果为None则显示
        metric: 可视化指标（"norm", "mean", "std"等）
        figsize: 图像大小
    """
    if len(feature.values) == 0:
        print(f"特征 {feature.name} 没有数据")
        return
    
    # 计算指标
    metrics = []
    for value in feature.values:
        if metric == "norm":
            metrics.append(value.norm().item())
        elif metric == "mean":
            metrics.append(value.mean().item())
        elif metric == "std":
            metrics.append(value.std().item())
        else:
            raise ValueError(f"不支持的指标: {metric}")
    
    # 绘制
    plt.figure(figsize=figsize)
    plt.plot(feature.epochs, metrics)
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric.capitalize()}")
    plt.title(f"特征演化: {feature.name} ({metric})")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_features_heatmap(
    tracker: FeatureTracker,
    epoch: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    可视化多个特征的热力图
    
    Args:
        tracker: FeatureTracker对象
        epoch: 指定epoch，如果为None则使用最新
        save_path: 保存路径
        figsize: 图像大小
    """
    features = tracker.get_features()
    if not features:
        print("没有可用的特征")
        return
    
    # 收集数据
    feature_names = []
    feature_values = []
    
    for name, feature in features.items():
        if epoch is not None:
            value = feature.get_at_epoch(epoch)
        else:
            value = feature.get_latest()
        
        if value is not None:
            feature_names.append(name)
            # 展平为1D用于热力图
            feature_values.append(value.flatten().cpu().numpy())
    
    if not feature_values:
        print("没有可用的特征值")
        return
    
    # 创建热力图数据
    max_len = max(len(v) for v in feature_values)
    heatmap_data = []
    for v in feature_values:
        if len(v) < max_len:
            # 填充或截断
            padded = np.pad(v, (0, max_len - len(v)), mode='constant')
            heatmap_data.append(padded[:max_len])
        else:
            heatmap_data.append(v[:max_len])
    
    heatmap_data = np.array(heatmap_data)
    
    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        xticklabels=False,
        yticklabels=feature_names,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Feature Value"}
    )
    plt.title(f"特征热力图 (Epoch: {epoch if epoch is not None else 'Latest'})")
    plt.ylabel("Feature")
    plt.xlabel("Parameter Index")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_parameter_comparison(
    feature: Feature,
    epochs: List[int],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    比较不同epoch的参数
    
    Args:
        feature: Feature对象
        epochs: 要比较的epoch列表
        save_path: 保存路径
        figsize: 图像大小
    """
    values = []
    valid_epochs = []
    
    for epoch in epochs:
        value = feature.get_at_epoch(epoch)
        if value is not None:
            values.append(value.flatten().cpu().numpy())
            valid_epochs.append(epoch)
    
    if not values:
        print(f"特征 {feature.name} 在指定epoch没有数据")
        return
    
    # 绘制对比图
    fig, axes = plt.subplots(1, len(values), figsize=figsize)
    if len(values) == 1:
        axes = [axes]
    
    for idx, (value, epoch) in enumerate(zip(values, valid_epochs)):
        ax = axes[idx]
        im = ax.imshow(
            value.reshape(-1, 1) if len(value.shape) == 1 else value,
            cmap="coolwarm",
            aspect="auto"
        )
        ax.set_title(f"Epoch {epoch}")
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f"参数演化对比: {feature.name}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize(feature, colored=True, output_size=[1, 0]):
    """
    原有的可视化函数（保持向后兼容）
    
    Args:
        feature: 特征tensor
        colored: 是否使用彩色
        output_size: 输出大小
    """
    # 简化实现，保持接口兼容
    if isinstance(feature, torch.Tensor):
        plt.figure(figsize=output_size if len(output_size) == 2 else (10, 6))
        if colored:
            plt.imshow(feature.cpu().numpy(), cmap="viridis")
        else:
            plt.imshow(feature.cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.show()
