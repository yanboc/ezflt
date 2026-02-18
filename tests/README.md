# 单元测试

## 测试内容

### 1. 数据测试 (`test_data.py`)
- **数据生成**：测试Generator类的初始化和参数组合
- **数据预处理**：测试DataLoader的创建、batch size、shuffle等功能
- **数据下载**：占位测试（待实现真实数据下载功能）

### 2. 模型测试 (`test_model.py`)
- **共用接口**：测试FLTNetwork的初始化和接口
- **PyTorch兼容性**：测试标准PyTorch模型的兼容性
- **参数访问**：测试模型参数的访问方式

### 3. 特征追踪测试 (`test_tracker.py`)
- **Feature类**：测试特征存储、添加、获取等功能
- **FeatureTracker类**：测试追踪器的初始化、启动/停止、生命周期钩子等
- **权重追踪**：测试参数追踪的核心功能

## 运行测试

### 方式1：使用pytest
```bash
pytest tests/ -v
```

### 方式2：运行特定测试文件
```bash
pytest tests/test_tracker.py -v
```

### 方式3：运行特定测试类或函数
```bash
pytest tests/test_tracker.py::TestFeature -v
pytest tests/test_tracker.py::TestFeature::test_feature_append -v
```

### 方式4：使用便捷脚本
```bash
python run_tests.py
```

## Git Hook配置

已配置pre-commit hook，每次commit前会自动运行测试。

### Windows用户
如果使用Git Bash，pre-commit会自动工作。如果使用其他方式，可以手动运行：
```bash
.git/hooks/pre-commit.bat
```

### Linux/Mac用户
pre-commit hook会自动工作。如果权限问题，运行：
```bash
chmod +x .git/hooks/pre-commit
```

## 跳过测试（不推荐）

如果确实需要跳过测试（不推荐），可以使用：
```bash
git commit --no-verify
```

