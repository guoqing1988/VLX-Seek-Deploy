#!/usr/bin/env bash
# 构建 VLX-Seek 常驻 HTTP 服务（原生异步接口 + 多实例并发）
# 产物：deploy/server/build/vlx-seek-server
#
# 独立 CMake 工程：把 ../../cpp_ggml 作为子目录引入（上游代码零改动），
# 链接 vlxvision（C API）+ llama + llama.cpp vendor 里的 cpp-httplib。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../env.sh"

BUILD_DIR="$HERE/build"
LOG_DIR="$HERE/../logs"
LOG="$LOG_DIR/build-server-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOG_DIR"

die() { echo "build-server.sh: $*" >&2; exit 1; }

command -v cmake >/dev/null || die "cmake 未安装"
[ -f "$PROJECT_ROOT/cpp_ggml/third_party/llama.cpp/CMakeLists.txt" ] \
    || die "llama.cpp 子模块缺失: git submodule update --init --recursive"

echo "== 配置 ($BUILD_DIR, CUDA arch=$VLX_CUDA_ARCH) ==" | tee -a "$LOG"
cmake -S "$HERE" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVLX_GGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$VLX_CUDA_ARCH" \
    2>&1 | tee -a "$LOG"

echo "== 编译 ==" | tee -a "$LOG"
cmake --build "$BUILD_DIR" --target vlx-seek-server --parallel "$(nproc)" 2>&1 | tee -a "$LOG"

[ -x "$BUILD_DIR/vlx-seek-server" ] || die "构建完成但未找到 $BUILD_DIR/vlx-seek-server"
echo "== 完成: $BUILD_DIR/vlx-seek-server ==" | tee -a "$LOG"
"$BUILD_DIR/vlx-seek-server" --help 2>&1 | head -6
