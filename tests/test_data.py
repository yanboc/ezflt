"""
测试数据生成、下载和预处理功能
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from ezflt.data import Generator
from config import CONFIG


class TestDataGeneration:
    """测试数据生成功能"""
    
    def test_generator_initialization(self):
        """测试Generator初始化"""
        generator = Generator(CONFIG)
        assert generator.seed == CONFIG["seed"]
        assert generator.device == CONFIG["device"]
        assert generator.data_type == CONFIG["data"]["data_type"]
        assert generator.train_size == CONFIG["data"]["train_size"]
        assert generator.test_size == CONFIG["data"]["test_size"]
        assert generator.batch_size == CONFIG["data"]["batch_size"]
    
    def test_generator_parameters_combinations(self):
        """测试参数组合生成"""
        generator = Generator(CONFIG)
        assert len(generator.parameters_combinations) > 0
        assert isinstance(generator.parameters_combinations, list)
    
    def test_generator_supported_data_types(self):
        """测试支持的数据类型"""
        generator = Generator(CONFIG)
        supported = generator.get_supported_data_type()
        assert isinstance(supported, list)
        assert len(supported) > 0


class TestDataPreprocessing:
    """测试数据预处理（DataLoader）"""
    
    def test_dataloader_creation(self):
        """测试DataLoader创建"""
        # 创建示例数据
        X = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(X, y)
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        
        assert len(dataloader) > 0
        assert dataloader.batch_size == 32
        
        # 测试数据加载
        for batch in dataloader:
            inputs, targets = batch
            assert inputs.shape[0] <= 32
            assert targets.shape[0] <= 32
            break
    
    def test_dataloader_batch_size(self):
        """测试DataLoader的batch size"""
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            inputs, _ = batch
            if batch_count < len(dataloader):
                assert inputs.shape[0] == 16
        
        assert batch_count == len(dataloader)
    
    def test_dataloader_shuffle(self):
        """测试DataLoader的shuffle功能"""
        X = torch.arange(100).float().unsqueeze(1)
        y = torch.zeros(100)
        dataset = TensorDataset(X, y)
        
        # 不shuffle
        dataloader_no_shuffle = DataLoader(dataset, batch_size=10, shuffle=False)
        first_batch_no_shuffle = next(iter(dataloader_no_shuffle))[0]
        
        # shuffle
        dataloader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)
        first_batch_shuffle = next(iter(dataloader_shuffle))[0]
        
        # shuffle后第一个batch应该不同（概率很高）
        # 注意：由于随机性，这个测试可能偶尔失败，但概率很低
        assert not torch.equal(first_batch_no_shuffle, first_batch_shuffle)


class TestDataDownload:
    """测试数据下载功能（占位，待实现真实数据下载）"""
    
    def test_data_download_placeholder(self):
        """占位测试，待实现真实数据下载功能"""
        # TODO: 实现真实数据下载测试
        # 测试MNIST、CIFAR-10等数据集的下载和加载
        pass

