# Qwen3.5 Docker 构建指南

本文档介绍如何使用 `Dockerfile.Qwen3.5` 构建和部署支持 Qwen3.5 模型推理的 Docker 镜像。

## 目录

- [概述](#概述)
- [文件说明](#文件说明)
- [本地构建](#本地构建)
- [通过 GitHub Actions 构建](#通过-github-actions-构建)
- [使用 Docker 镜像](#使用-docker-镜像)
- [配置说明](#配置说明)

## 概述

Qwen3.5 Docker 镜像基于 NVIDIA CUDA 镜像构建，包含以下组件：

- **KT-Kernel** (qwen3.5 分支): CPU-GPU 异构推理内核
- **SGLang** (qwen3.5 分支): 高性能推理框架
- **CUDA Toolkit**: CUDA 12.8+ (推荐)
- **Python 3.12**: Conda 环境

## 文件说明

### 新增文件

1. **`docker/Dockerfile.Qwen3.5`**
   - 专门为 Qwen3.5 模型推理设计的 Dockerfile
   - 基于 Ubuntu 24.04 + CUDA 12.8.1
   - 包含 KT-Kernel 和 SGLang (qwen3.5 分支)

2. **`.github/workflows/docker-qwen35.yml`**
   - GitHub Actions workflow 配置文件
   - 自动化构建和推送 Docker 镜像到 DockerHub
   - 支持多种触发方式和参数配置

## 本地构建

### 基础构建

```bash
cd docker
docker build -t ktransformers:qwen3.5 -f Dockerfile.Qwen3.5 ..
```

### 自定义 CUDA 版本

```bash
docker build \
  --build-arg CUDA_VERSION=12.8.1 \
  -t ktransformers:qwen3.5-cu128 \
  -f Dockerfile.Qwen3.5 \
  ..
```

### 使用代理构建

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy.example.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.example.com:8080 \
  -t ktransformers:qwen3.5 \
  -f Dockerfile.Qwen3.5 \
  ..
```

### 支持的构建参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CUDA_VERSION` | `12.8.1` | CUDA 版本 |
| `UBUNTU_MIRROR` | (空) | Ubuntu 镜像源 (保留用于国内加速) |
| `HTTP_PROXY` | (空) | HTTP 代理 |
| `HTTPS_PROXY` | (空) | HTTPS 代理 |
| `GITHUB_ARTIFACTORY` | `github.com` | GitHub 域名 (可用于镜像加速) |
| `BUILD_AND_DOWNLOAD_PARALLEL` | `8` | 并行构建和下载线程数 |

## 通过 GitHub Actions 构建

### 方式 1: 手动触发 (Workflow Dispatch)

1. 访问 GitHub 仓库的 Actions 页面
2. 选择 "Qwen3.5 Docker CI" workflow
3. 点击 "Run workflow"
4. 配置参数：
   - **push_to_dockerhub**: 是否推送到 DockerHub (`true`/`false`)
   - **cuda_version**: CUDA 版本 (默认 `12.8.1`)
   - **ubuntu_mirror**: 是否使用清华镜像 (`0`/`1`)
   - **image_tag**: 自定义镜像标签 (可选)

### 方式 2: Release 自动触发

当发布新版本 (Release) 时，workflow 会自动触发：
- 自动构建镜像
- 使用 release 标签命名 (例如: `qwen3.5-v1.0.0-cu128`)
- 自动推送到 DockerHub

### GitHub Secrets 配置

需要在仓库中配置以下 Secrets：

| Secret 名称 | 必需 | 说明 |
|------------|------|------|
| `DOCKERHUB_USERNAME` | ✅ | DockerHub 用户名 |
| `DOCKERHUB_TOKEN` | ✅ | DockerHub 访问令牌 |
| `HTTP_PROXY` | ❌ | HTTP 代理 (可选) |
| `HTTPS_PROXY` | ❌ | HTTPS 代理 (可选) |

## 使用 Docker 镜像

### 拉取镜像

```bash
# 拉取最新版本
docker pull <username>/ktransformers:qwen3.5-latest

# 拉取特定版本
docker pull <username>/ktransformers:qwen3.5-v20260223-cu128
```

### 启动容器

```bash
docker run -it --gpus all \
  -v /path/to/models:/workspace/models \
  -p 30005:30005 \
  <username>/ktransformers:qwen3.5-latest \
  /bin/bash
```

### 在容器中运行 Qwen3.5

1. **激活环境**：

```bash
conda activate qwen35
# 或使用别名
qwen35
```

2. **下载模型** (如果还没有)：

```bash
cd /workspace/models
huggingface-cli download Qwen/Qwen3.5 \
  --local-dir /workspace/models/qwen3.5
```

3. **启动 SGLang 服务器** (4x RTX 4090 示例)：

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30005 \
  --model /workspace/models/qwen3.5 \
  --kt-weight-path /workspace/models/qwen3.5 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 1 \
  --kt-method BF16 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 32 \
  --max-total-tokens 32000 \
  --served-model-name qwen3.5 \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce
```

4. **发送推理请求**：

```bash
curl -s http://localhost:30005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "stream": false,
    "messages": [
      {"role": "user", "content": "hi, who are you?"}
    ]
  }'
```

## 配置说明

### 硬件要求

- **GPU**: NVIDIA 4x RTX 4090 或同等配置 (至少 96GB 总显存)
- **CPU**: 支持 AVX512F 的 x86 CPU (例如 Intel Sapphire Rapids)
- **RAM**: 至少 800GB 系统内存
- **存储**: ~800GB 用于模型权重 (BF16)

### 环境变量

镜像内置以下环境变量：

- `CUDA_HOME=/usr/local/cuda`
- `CONDA_DEFAULT_ENV=qwen35`
- `PATH` 包含 CUDA 和 conda 路径

### Conda 环境

镜像包含一个名为 `qwen35` 的 conda 环境，预装：

- PyTorch (with CUDA support)
- SGLang (qwen3.5 分支)
- KT-Kernel (qwen3.5 分支)
- nvidia-cudnn-cu12==9.16.0.29
- huggingface-hub

### 版本信息

构建完成后，版本信息保存在 `/workspace/versions.env`：

```bash
cat /workspace/versions.env
```

输出示例：
```
SGLANG_VERSION=0.5.6
KTRANSFORMERS_VERSION=0.4.2
```

## 故障排除

### 问题 1: CUDNN 相关错误

如果启动 SGLang 时遇到 CUDNN 错误，尝试重新安装：

```bash
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall
```

### 问题 2: 内存不足

- 调整 `--kt-cpuinfer` 参数以控制 CPU 推理专家数量
- 调整 `--mem-fraction-static` 降低 GPU 内存使用
- 确保系统有足够的 RAM (推荐 800GB+)

### 问题 3: 子模块未初始化

如果遇到子模块相关错误，在容器内运行：

```bash
cd /workspace/ktransformers
git submodule update --init --recursive
```

## 参考文档

- [Qwen3.5 完整教程](../doc/en/Qwen3.5.md)
- [KT-Kernel 参数说明](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters)
- [SGLang 文档](https://github.com/kvcache-ai/sglang/tree/qwen3.5)

## 许可证

遵循 KTransformers 项目许可证。

