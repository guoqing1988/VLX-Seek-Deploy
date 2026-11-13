#!/usr/bin/env bash
# 部署冒烟测试：7 个任务模板各跑一次，校验退出码与 JSON 输出。
# 用法：./deploy/smoke.sh [q8_0|f16]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRECISION="${1:-q8_0}"
IMG="$HERE/test_images/water_bottles_table.jpg"
LOG="$HERE/logs/smoke-$PRECISION-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$HERE/logs"
: > "$LOG"

PASS=0
FAIL=0
run() {
    local task="$1" text="$2" extra="${3:-}"
    local out rc
    # shellcheck disable=SC2086
    out="$("$HERE/vlx.sh" --precision "$PRECISION" --device cuda \
        --image-path "$IMG" --task "$task" --text "$text" --lang en \
        --no-visualize $extra 2>>"$LOG")" || rc=$?
    rc="${rc:-0}"
    if [ "$rc" -eq 0 ] && python3 -c 'import json,sys; json.loads(sys.argv[1])' "$out" 2>/dev/null; then
        local n
        n="$(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);print(len(d.get("result_bbox_list",[])))' "$out")"
        echo "PASS  $task   boxes=$n" | tee -a "$LOG"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $task   rc=$rc" | tee -a "$LOG"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== 冒烟测试 precision=$PRECISION image=$(basename "$IMG") $(date) ===" | tee -a "$LOG"
run vqa                    "what is in the image?"
run detection              "bottle; bowl"
run grounding_single       "the bottle on the left"
run counting               "bottle"
run reasoning_detection    "which object is being used to drink?"
run brief_region_caption   "describe" "--target-region-indexes [0]"
run region_ocr             "read the text" "--target-region-indexes [0]"

echo "--- 通过 $PASS / 失败 $FAIL，日志 $LOG" | tee -a "$LOG"
[ "$FAIL" -eq 0 ]
