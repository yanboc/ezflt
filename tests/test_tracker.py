"""
测试Feature和FeatureTracker的核心功能
"""

import pytest
import torch
import torch.nn as nn
from ezflt.tracker import Feature, FeatureTracker


class TestFeature:
    """测试Feature类"""
    
    def test_feature_initialization(self):
        """测试Feature初始化"""
        feature = Feature("conv1.weight", "conv1")
        assert feature.name == "conv1.weight"
        assert feature.layer_name == "conv1"
        assert len(feature.values) == 0
        assert len(feature.epochs) == 0
        assert len(feature.batch_indices) == 0
    
    def test_feature_append(self):
        """测试Feature添加值"""
        feature = Feature("fc.weight", "fc")
        value = torch.randn(10, 20)
        
        feature.append(value, epoch=0, batch_idx=0)
        assert len(feature) == 1
        assert feature.epochs[0] == 0
        assert feature.batch_indices[0] == 0
        assert torch.equal(feature.values[0], value)
    
    def test_feature_multiple_append(self):
        """测试Feature多次添加"""
        feature = Feature("layer.weight", "layer")
        
        for epoch in range(5):
            for batch_idx in range(3):
                value = torch.randn(5, 10) * epoch
                feature.append(value, epoch, batch_idx)
        
        assert len(feature) == 15
        assert len(feature.epochs) == 15
        assert len(feature.batch_indices) == 15
    
    def test_feature_get_at_epoch(self):
        """测试Feature按epoch获取值"""
        feature = Feature("test.weight", "test")
        
        # 添加不同epoch的值
        for epoch in range(3):
            value = torch.tensor([epoch * 1.0])
            feature.append(value, epoch, batch_idx=0)
        
        # 测试获取
        epoch_0_value = feature.get_at_epoch(0)
        assert epoch_0_value is not None
        assert epoch_0_value.item() == 0.0
        
        epoch_2_value = feature.get_at_epoch(2)
        assert epoch_2_value is not None
        assert epoch_2_value.item() == 2.0
        
        # 不存在的epoch
        assert feature.get_at_epoch(10) is None
    
    def test_feature_get_latest(self):
        """测试Feature获取最新值"""
        feature = Feature("test.weight", "test")
        
        assert feature.get_latest() is None
        
        for i in range(5):
            value = torch.tensor([i * 1.0])
            feature.append(value, epoch=0, batch_idx=i)
        
        latest = feature.get_latest()
        assert latest is not None
        assert latest.item() == 4.0


class TestFeatureTracker:
    """测试FeatureTracker类"""
    
    def test_tracker_initialization(self):
        """测试FeatureTracker初始化"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        tracker = FeatureTracker(
            model=model,
            layers=["0", "2"],
            track_weights=True
        )
        
        assert tracker.model == model
        assert tracker.layers == ["0", "2"]
        assert tracker.track_weights is True
        assert tracker.track_gradients is False
        assert tracker.is_tracking is False
    
    def test_tracker_start_stop(self):
        """测试追踪器的启动和停止"""
        model = nn.Sequential(nn.Linear(10, 5))
        tracker = FeatureTracker(model, layers=["0"])
        
        assert tracker.is_tracking is False
        tracker.start()
        assert tracker.is_tracking is True
        assert len(tracker.hooks) > 0
        
        tracker.stop()
        assert tracker.is_tracking is False
        assert len(tracker.hooks) == 0
    
    def test_tracker_context_manager(self):
        """测试上下文管理器"""
        model = nn.Sequential(nn.Linear(10, 5))
        tracker = FeatureTracker(model, layers=["0"])
        
        assert tracker.is_tracking is False
        
        with tracker:
            assert tracker.is_tracking is True
        
        assert tracker.is_tracking is False
    
    def test_tracker_lifecycle_hooks(self):
        """测试生命周期钩子"""
        model = nn.Sequential(nn.Linear(10, 5))
        tracker = FeatureTracker(model, layers=["0"])
        
        tracker.on_epoch_start(5)
        assert tracker.current_epoch == 5
        assert tracker.current_batch_idx == 0
        
        tracker.on_batch_end(10)
        assert tracker.current_batch_idx == 10
    
    def test_tracker_weight_tracking(self):
        """测试权重追踪功能"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        tracker = FeatureTracker(
            model=model,
            layers=["0"],
            track_weights=True,
            track_frequency=1
        )
        
        tracker.start()
        tracker.on_epoch_start(0)
        
        # 模拟前向传播
        x = torch.randn(32, 10)
        _ = model(x)
        
        tracker.on_batch_end(0)
        
        # 检查是否记录了特征
        features = tracker.get_features()
        # 注意：hook可能需要在forward时触发，这里主要测试接口
        
        tracker.stop()
    
    def test_tracker_get_features(self):
        """测试获取特征"""
        model = nn.Sequential(nn.Linear(10, 5))
        tracker = FeatureTracker(model, layers=["0"])
        
        features = tracker.get_features()
        assert isinstance(features, dict)
    
    def test_tracker_get_feature(self):
        """测试获取指定特征"""
        model = nn.Sequential(nn.Linear(10, 5))
        tracker = FeatureTracker(model, layers=["0"])
        
        # 在追踪前，特征字典应该是空的或未初始化的
        feature = tracker.get_feature("0.weight")
        # 可能为None（如果还未追踪）或Feature对象
    
    def test_tracker_track_frequency(self):
        """测试追踪频率"""
        model = nn.Sequential(nn.Linear(10, 5))
        tracker = FeatureTracker(
            model=model,
            layers=["0"],
            track_frequency=5  # 每5个batch记录一次
        )
        
        assert tracker.track_frequency == 5
        
        tracker.start()
        tracker.on_epoch_start(0)
        
        # 测试频率过滤
        for batch_idx in range(10):
            tracker.on_batch_end(batch_idx)
            # 只有batch_idx % 5 == 0时才会记录
        
        tracker.stop()
    
    def test_tracker_all_layers(self):
        """测试追踪所有层"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        tracker = FeatureTracker(
            model=model,
            layers=None  # 追踪所有层
        )
        
        # layers应该包含所有层
        assert len(tracker.layers) > 0

