#!/usr/bin/env bash
# 启动包装：把模型目录 / 构建目录固定成部署配置，其余参数原样透传给 vlx-cli。
#
# 用法：
#   ./deploy/vlx.sh --task detection --text "orange; apple" --lang zh
#   ./deploy/vlx.sh --image-path deploy/test_images/street_traffic.jpg \
#                   --task counting --text "person" --lang en --precision q8_0
#   ./deploy/vlx.sh --precision f16 --warm-runs 3 --task detection --text "orange; apple"
#
# 精度组合说明：视觉链有 f16 / q8_0 两套合并 GGUF，LLM 只有 q8_0。
# 所以 --precision f16（视觉 f16）时自动补 --llm .../vlx-10b-llm-q8_0.gguf；
# 调用方显式传 --llm 时以调用方为准。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

[ -x "$VLX_CLI" ] || { echo "vlx-cli 未构建，先运行 deploy/build.sh" >&2; exit 1; }
[ -d "$VLX_GGUF_DIR" ] || { echo "模型目录不存在: $VLX_GGUF_DIR" >&2; exit 1; }

cd "$(cd "$HERE/.." && pwd)"

PRECISION="f16"
HAS_LLM=0
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --precision) PRECISION="${ARGS[i+1]:-}" ;;
        --llm)       HAS_LLM=1 ;;
    esac
done
# f32 在 CLI 里映射到 f16 LLM 文件（LLM 权重本身是 bf16），同样适用回退
[ "$PRECISION" = "f32" ] && LLM_PREC="f16" || LLM_PREC="$PRECISION"

EXTRA=()
if [ "$HAS_LLM" -eq 0 ] && [ ! -f "$VLX_GGUF_DIR/vlx-10b-llm-$LLM_PREC.gguf" ]; then
    FALLBACK="$VLX_GGUF_DIR/vlx-10b-llm-q8_0.gguf"
    [ -f "$FALLBACK" ] || { echo "缺少 LLM GGUF: $VLX_GGUF_DIR/vlx-10b-llm-$LLM_PREC.gguf" >&2; exit 1; }
    echo "[deploy] 未找到 vlx-10b-llm-$LLM_PREC.gguf，回退 --llm $(basename "$FALLBACK")" >&2
    EXTRA=(--llm "$FALLBACK")
fi

exec "$VLX_CLI" run --gguf-dir "$VLX_GGUF_DIR" "${EXTRA[@]}" "$@"
