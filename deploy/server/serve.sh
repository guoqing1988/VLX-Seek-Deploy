#!/usr/bin/env bash
# 启动 VLX-Seek 常驻服务（原生异步接口）
#
#   ./deploy/server/serve.sh                       # 127.0.0.1:8100, 2 实例, 空闲 300s 卸载
#   PORT=8200 INSTANCES=1 ./deploy/server/serve.sh
#   IDLE_UNLOAD_S=60 ./deploy/server/serve.sh --preload --log-level 0
#
# 环境变量：PORT / HOST / INSTANCES / IDLE_UNLOAD_S / PRECISION / DEVICE
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

BIN="$HERE/build/vlx-seek-server"
# 多卡选择：GPU=1 表示只用物理卡 1（等价于 CUDA_VISIBLE_DEVICES=1）
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
[ -x "$BIN" ] || { echo "服务未构建：先运行 deploy/server/build-server.sh" >&2; exit 1; }
[ -d "$VLX_GGUF_DIR" ] || { echo "模型目录不存在: $VLX_GGUF_DIR" >&2; exit 1; }

mkdir -p "$HERE/tmp"

# 可选：图像 URL 代理 / 追加参数（systemd 的 deploy/.env 会注入）
PROXY_ARGS=()
[ -n "${IMAGE_PROXY:-}" ] && PROXY_ARGS=(--image-proxy "$IMAGE_PROXY")
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a _extra <<< "$EXTRA_ARGS"
    set -- "${_extra[@]}" "$@"
fi

exec "$BIN" \
    --gguf-dir "$VLX_GGUF_DIR" \
    --tmp-dir "$HERE/tmp" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8100}" \
    --instances "${INSTANCES:-2}" \
    --idle-unload-s "${IDLE_UNLOAD_S:-300}" \
    --precision "${PRECISION:-q8_0}" \
    --device "${DEVICE:-cuda}" \
    "${PROXY_ARGS[@]}" \
    "$@"
