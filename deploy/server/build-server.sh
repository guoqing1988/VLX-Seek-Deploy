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

# 可用 VLX_SERVER_BUILD_DIR 指定独立构建目录（例如并行保留 12.8 / 13.0 两套）
BUILD_DIR="${VLX_SERVER_BUILD_DIR:-$HERE/build}"
LOG_DIR="$HERE/../logs"
LOG="$LOG_DIR/build-server-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOG_DIR"

die() { echo "build-server.sh: $*" >&2; exit 1; }

command -v cmake >/dev/null || die "cmake 未安装"
[ -f "$PROJECT_ROOT/cpp_ggml/third_party/llama.cpp/CMakeLists.txt" ] \
    || die "llama.cpp 子模块缺失: git submodule update --init --recursive"

# 应用 deploy/patches/*.patch 到上游源码（幂等：已应用则跳过；不适用则退出）
APPLIED_PATCHES=()
# 构建结束（含失败）时把 patch 还原，保持上游源码干净
cleanup_patches() {
    local rc=$?
    local p
    for p in "${APPLIED_PATCHES[@]}"; do
        if git -C "$PROJECT_ROOT" apply --reverse "$p" 2>/dev/null; then
            echo "== 已还原 patch: $(basename "$p") ==" | tee -a "$LOG"
        else
            echo "WARN: 还原 patch 失败（源码被改动过？）: $p" | tee -a "$LOG"
        fi
    done
    return $rc
}
trap cleanup_patches EXIT

apply_vlx_patches() {
    local pdir="$HERE/../patches"
    [ -d "$pdir" ] || return 0
    shopt -s nullglob
    local p
    for p in "$pdir"/*.patch; do
        if git -C "$PROJECT_ROOT" apply --reverse --check "$p" 2>/dev/null; then
            echo "== patch 已应用: $(basename "$p") ==" | tee -a "$LOG"
        elif git -C "$PROJECT_ROOT" apply --check "$p" 2>/dev/null; then
            git -C "$PROJECT_ROOT" apply "$p" || die "应用 patch 失败: $p"
            APPLIED_PATCHES+=("$p")
            echo "== 已应用 patch: $(basename "$p")（构建后自动还原）==" | tee -a "$LOG"
        else
            die "patch 不适用于当前源码（上游更新了？请重新生成）: $p"
        fi
    done
    shopt -u nullglob
}
apply_vlx_patches

echo "== 配置 ($BUILD_DIR, CUDA arch=$VLX_CUDA_ARCH) ==" | tee -a "$LOG"
cmake -S "$HERE" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVLX_GGML_CUDA=ON \
    -DVLX_ROOT="$PROJECT_ROOT" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES="$VLX_CUDA_ARCH" \
    2>&1 | tee -a "$LOG"

echo "== 编译 ==" | tee -a "$LOG"
cmake --build "$BUILD_DIR" --target vlx-seek-server --parallel "$(nproc)" 2>&1 | tee -a "$LOG"

[ -x "$BUILD_DIR/vlx-seek-server" ] || die "构建完成但未找到 $BUILD_DIR/vlx-seek-server"
echo "== 完成: $BUILD_DIR/vlx-seek-server ==" | tee -a "$LOG"
"$BUILD_DIR/vlx-seek-server" --help 2>&1 | head -6
