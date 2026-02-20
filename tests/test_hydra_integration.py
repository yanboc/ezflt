"""
Hydra集成功能测试

测试Hydra配置管理、命令行参数覆盖、多组实验等功能。
"""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from ezflt.experiment import ExperimentManager


class TestHydraIntegration:
    """Hydra集成测试"""
    
    @pytest.mark.skipif(
        not Path("configs/config.yaml").exists(),
        reason="需要configs/config.yaml文件"
    )
    def test_hydra_config_loading(self):
        """测试Hydra配置加载"""
        manager = ExperimentManager(
            config_path="configs",
            use_hydra=True,
            use_wandb=False,
        )
        manager._init_hydra()
        
        assert manager.use_hydra is True
        assert manager.hydra_cfg is not None
        
        config = manager.get_config()
        assert isinstance(config, dict)
        assert "experiment" in config
        assert "training" in config
        assert "data" in config
    
    @pytest.mark.skipif(
        not Path("configs/config.yaml").exists(),
        reason="需要configs/config.yaml文件"
    )
    def test_hydra_config_structure(self):
        """测试配置结构"""
        manager = ExperimentManager(
            config_path="configs",
            use_hydra=True,
            use_wandb=False,
        )
        manager._init_hydra()
        
        if manager.hydra_cfg:
            cfg = manager.hydra_cfg
            
            # 检查必需配置项
            assert hasattr(cfg, 'experiment')
            assert hasattr(cfg, 'training')
            assert hasattr(cfg, 'data')
            assert hasattr(cfg, 'model')
            
            # 检查实验配置
            assert hasattr(cfg.experiment, 'name')
            assert hasattr(cfg.experiment, 'seed')
            
            # 检查训练配置
            assert hasattr(cfg.training, 'epochs')
            assert hasattr(cfg.training, 'learning_rate')
    
    @pytest.mark.skipif(
        not Path("configs/config.yaml").exists(),
        reason="需要configs/config.yaml文件"
    )
    def test_hydra_config_to_dict(self):
        """测试配置转换为字典"""
        manager = ExperimentManager(
            config_path="configs",
            use_hydra=True,
            use_wandb=False,
        )
        manager._init_hydra()
        
        if manager.hydra_cfg:
            config = manager.get_config()
            
            # 检查字典结构
            assert isinstance(config, dict)
            assert "experiment" in config
            assert isinstance(config["experiment"], dict)
            
            # 检查嵌套结构
            assert "training" in config
            assert "learning_rate" in config["training"]
    
    def test_hydra_without_config_dir(self):
        """测试Hydra在配置文件不存在时的行为"""
        manager = ExperimentManager(
            config_path="non_existent_configs",
            use_hydra=True,
            use_wandb=False,
        )
        manager._init_hydra()
        
        # 应该禁用hydra而不是抛出异常
        assert manager.use_hydra is False
        assert manager.hydra_cfg is None


class TestHydraCommandLineOverride:
    """测试命令行参数覆盖（需要实际运行）"""
    
    @pytest.mark.skipif(
        True,  # 默认跳过，需要手动测试
        reason="需要实际运行脚本测试命令行参数"
    )
    def test_command_line_override(self):
        """测试命令行参数覆盖"""
        # 这个测试需要实际运行脚本：
        # python examples/train_with_hydra_simple.py training.learning_rate=0.01
        pass
    
    @pytest.mark.skipif(
        True,  # 默认跳过，需要手动测试
        reason="需要实际运行脚本测试多组实验"
    )
    def test_multirun(self):
        """测试多组实验（multirun）"""
        # 这个测试需要实际运行脚本：
        # python examples/train_with_hydra_simple.py -m training.learning_rate=0.001,0.01,0.1
        pass


class TestHydraWandbIntegration:
    """测试Hydra与Wandb的集成"""
    
    @pytest.mark.skipif(
        not Path("configs/config.yaml").exists(),
        reason="需要configs/config.yaml文件"
    )
    def test_hydra_config_to_wandb(self):
        """测试将Hydra配置传递给Wandb"""
        manager = ExperimentManager(
            config_path="configs",
            use_hydra=True,
            use_wandb=False,  # 不实际初始化wandb
        )
        manager._init_hydra()
        
        if manager.hydra_cfg:
            # 测试配置转换
            config = manager.get_config()
            
            # 应该能成功转换为字典
            assert isinstance(config, dict)
            
            # 检查wandb相关配置
            if "wandb" in config:
                assert isinstance(config["wandb"], dict)

