"""
实验管理模块的单元测试
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from ezflt.experiment import ExperimentManager, create_experiment_manager


class TestExperimentManager:
    """ExperimentManager的测试"""
    
    def test_init_with_defaults(self):
        """测试默认初始化"""
        manager = ExperimentManager()
        assert manager.use_wandb is True
        assert manager.use_hydra is True
        assert manager.project_name == "ezflt-experiments"
    
    def test_init_with_custom_params(self):
        """测试自定义参数初始化"""
        manager = ExperimentManager(
            project_name="test-project",
            experiment_name="test-exp",
            use_wandb=False,
            use_hydra=False,
        )
        assert manager.project_name == "test-project"
        assert manager.experiment_name == "test-exp"
        assert manager.use_wandb is False
        assert manager.use_hydra is False
    
    def test_init_hydra_without_config_dir(self):
        """测试Hydra初始化（配置文件不存在时）"""
        manager = ExperimentManager(
            config_path="non_existent_configs",
            use_hydra=True,
            use_wandb=False,
        )
        # 应该不会抛出异常，而是禁用hydra
        manager._init_hydra()
        assert manager.use_hydra is False
    
    def test_get_config_without_hydra(self):
        """测试获取配置（不使用Hydra时）"""
        manager = ExperimentManager(use_hydra=False, use_wandb=False)
        config = manager.get_config()
        assert config == {}
    
    def test_context_manager(self):
        """测试上下文管理器"""
        manager = ExperimentManager(use_wandb=False, use_hydra=False)
        with manager:
            assert manager is not None
        # 退出上下文后应该正常清理
    
    def test_log_metrics_without_wandb(self):
        """测试记录指标（不使用Wandb时）"""
        manager = ExperimentManager(use_wandb=False, use_hydra=False)
        # 应该不会抛出异常
        manager.log_metrics({"loss": 0.5}, step=0)
    
    def test_log_features_without_wandb(self):
        """测试记录特征（不使用Wandb时）"""
        manager = ExperimentManager(use_wandb=False, use_hydra=False)
        # 应该不会抛出异常
        manager.log_features({}, epoch=0)
    
    def test_finish_without_wandb(self):
        """测试完成实验（不使用Wandb时）"""
        manager = ExperimentManager(use_wandb=False, use_hydra=False)
        # 应该不会抛出异常
        manager.finish()


class TestCreateExperimentManager:
    """create_experiment_manager便捷函数的测试"""
    
    def test_create_with_defaults(self):
        """测试默认创建"""
        manager = create_experiment_manager()
        assert isinstance(manager, ExperimentManager)
        assert manager.use_wandb is True
        assert manager.use_hydra is True
    
    def test_create_with_custom_params(self):
        """测试自定义参数创建"""
        manager = create_experiment_manager(
            project_name="custom-project",
            use_wandb=False,
            use_hydra=False,
        )
        assert manager.project_name == "custom-project"
        assert manager.use_wandb is False
        assert manager.use_hydra is False


class TestExperimentManagerIntegration:
    """ExperimentManager集成测试（需要实际环境）"""
    
    @pytest.mark.skipif(
        not Path("configs/config.yaml").exists(),
        reason="需要configs/config.yaml文件"
    )
    def test_hydra_config_loading(self):
        """测试Hydra配置加载（需要配置文件）"""
        manager = ExperimentManager(
            config_path="configs",
            use_hydra=True,
            use_wandb=False,
        )
        manager._init_hydra()
        if manager.use_hydra:  # 如果成功初始化
            config = manager.get_config()
            assert isinstance(config, dict)
    
    @pytest.mark.skipif(
        True,  # 默认跳过，需要手动启用
        reason="需要wandb API key和网络连接"
    )
    def test_wandb_integration(self):
        """测试Wandb集成（需要API key）"""
        manager = ExperimentManager(
            project_name="ezflt-test",
            use_wandb=True,
            use_hydra=False,
        )
        manager._init_wandb()
        if manager.wandb_run:
            manager.log_metrics({"test_metric": 1.0}, step=0)
            manager.finish()

