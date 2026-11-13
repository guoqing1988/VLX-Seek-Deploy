#!/usr/bin/env bash
# llama.cpp 原生 llama-server：纯文本 OpenAI 兼容服务（/v1/models、/v1/chat/completions）
#
# ⚠️ 只加载 vlx-10b-llm-q8_0.gguf 这个 LLM，没有 WeDetect 候选框、没有 RADIO/HFRE 区域编码，
#    因此不具备 VLX-Seek 的视觉/定位能力（vision GGUF 不是 llama.cpp 的 clip 架构，
#    llama-server 加载会直接报 "Failed to load CLIP model"）。
#    需要 VLX 能力请用 deploy/vlx.sh（CLI）。
#
# 用法：
#   ./deploy/llama-server.sh                 # 127.0.0.1:8099
#   PORT=8090 ./deploy/llama-server.sh -c 8192 -t 24
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

BIN="$PROJECT_ROOT/cpp_ggml/build-server/bin/llama-server"
LLM="$VLX_GGUF_DIR/vlx-10b-llm-q8_0.gguf"
[ -x "$BIN" ] || { echo "llama-server 未构建：cmake -S cpp_ggml/third_party/llama.cpp -B cpp_ggml/build-server -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON && cmake --build cpp_ggml/build-server --target llama-server" >&2; exit 1; }
[ -f "$LLM" ] || { echo "缺少 LLM GGUF: $LLM" >&2; exit 1; }

exec "$BIN" -m "$LLM" -c 8192 -ngl 999 -t "$(nproc)" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8099}" "$@"
