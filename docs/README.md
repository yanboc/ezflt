# API 文档

本目录包含 ezflt 项目的 API 文档，使用 [MkDocs](https://www.mkdocs.org/) 和 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 生成。

## 生成文档

### 安装文档依赖

```bash
uv pip install -e ".[docs]"
```

### 本地预览

```bash
# 启动开发服务器
mkdocs serve

# 浏览器访问 http://127.0.0.1:8000
```

### 构建文档

```bash
# 构建静态网站
mkdocs build

# 构建的文件在 site/ 目录
```

### 部署

```bash
# 部署到 GitHub Pages
mkdocs gh-deploy
```

## 文档结构

- `index.md` - 首页
- `getting_started.md` - 快速开始指南
- `api/` - API 参考文档
- `examples.md` - 使用示例
- `contributing.md` - 贡献指南

