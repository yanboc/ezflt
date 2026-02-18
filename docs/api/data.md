# data 模块

数据生成和处理模块。

## Generator

基于配置文件生成数据的类。

::: ezflt.data.Generator
    options:
      show_source: true
      heading_level: 3

## 使用示例

```python
from ezflt import Generator
from config import CONFIG

# 创建数据生成器
generator = Generator(CONFIG)

# 获取支持的数据类型
supported_types = generator.get_supported_data_type()
```

