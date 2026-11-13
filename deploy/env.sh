#!/usr/bin/env bash
# 部署环境变量：所有路径都落在项目内，模型指向外部 GGUF 目录。
# 用法：source deploy/env.sh
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"

# 模型目录（只读，不入库）
export VLX_GGUF_DIR="${VLX_GGUF_DIR:-/data/www/models/VLX-Seek-1.5-10B-GGUF}"

# 项目内环境
export VLX_BUILD_DIR="${VLX_BUILD_DIR:-$PROJECT_ROOT/cpp_ggml/build-cuda}"
export VLX_CLI="$VLX_BUILD_DIR/vlx-cli"
export VLX_IMAGES="$DEPLOY_DIR/test_images"
export VLX_OUT="$DEPLOY_DIR/out"
export VLX_LOGS="$DEPLOY_DIR/logs"

# CUDA 工具链：构建/运行使用 toolkit 13.0（nvcc 13.0.88，产物链接 libcudart.so.13）。
# nvidia-smi 报的 "CUDA Version: 13.0" 是驱动（580.173.02）支持上限，与本机 toolkit 一致。
# 12.8.93 也构建/对比过：同图推理逐 token 一致、坐标差 0.0000 px；13.0 构建通过 65 项验收。
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 目标 GPU 架构：RTX 5880 Ada = sm_89
export VLX_CUDA_ARCH="${VLX_CUDA_ARCH:-89}"
