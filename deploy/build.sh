#!/usr/bin/env bash
# 构建 VLX-Seek C++/GGML 推理后端（CUDA 版）
# 产物：cpp_ggml/build-cuda/vlx-cli（全部在项目内）
#
# 失败预演：依赖缺失 / 子模块缺失 / 架构不支持时立即退出，不产生半成品覆盖。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
# shellcheck source=/dev/null
source "$HERE/env.sh"

LOG="$VLX_LOGS/build-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$VLX_LOGS"

die() { echo "build.sh: $*" >&2; exit 1; }

command -v cmake >/dev/null || die "cmake 未安装"
command -v nvcc  >/dev/null || die "nvcc 未安装（需要 CUDA toolkit）"
[ -f "$PROJECT_ROOT/cpp_ggml/third_party/llama.cpp/CMakeLists.txt" ] \
    || die "llama.cpp 子模块缺失，先执行: git submodule update --init --recursive"
[ -d "$VLX_GGUF_DIR" ] || die "模型目录不存在: $VLX_GGUF_DIR"

echo "== 配置 ($VLX_BUILD_DIR, CUDA arch=$VLX_CUDA_ARCH) ==" | tee -a "$LOG"
cmake -S "$PROJECT_ROOT/cpp_ggml" -B "$VLX_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVLX_GGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$VLX_CUDA_ARCH" \
    2>&1 | tee -a "$LOG"

echo "== 编译 ==" | tee -a "$LOG"
cmake --build "$VLX_BUILD_DIR" --parallel "$(nproc)" 2>&1 | tee -a "$LOG"

[ -x "$VLX_CLI" ] || die "构建完成但未找到 $VLX_CLI"
echo "== 完成: $VLX_CLI ==" | tee -a "$LOG"
"$VLX_CLI" run --help 2>&1 | head -5
