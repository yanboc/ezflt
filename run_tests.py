#!/usr/bin/env python
"""
运行单元测试的便捷脚本
"""

import sys
import subprocess

if __name__ == "__main__":
    # 运行pytest
    result = subprocess.run(
        ["pytest", "tests/", "-v"],
        cwd=sys.path[0]
    )
    sys.exit(result.returncode)

