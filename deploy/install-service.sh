#!/usr/bin/env bash
# 用 systemd 管理 VLX-Seek 常驻服务（软链方式，不复制 unit 文件）
#
#   ./deploy/install-service.sh install     # ln -s + daemon-reload + enable + restart
#   ./deploy/install-service.sh uninstall   # 停服务 + 删除软链
#   ./deploy/install-service.sh status      # 软链状态 + systemd 状态 + /healthz
#   ./deploy/install-service.sh restart     # 重启（改完 deploy/.env 后）
#
# 需要 sudo 权限写入 /etc/systemd/system（脚本内部调用 sudo）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="vlx-seek.service"
SRC="$HERE/$NAME"
DST="/etc/systemd/system/$NAME"
ENV_FILE="$HERE/.env"
BIN="$HERE/server/build/vlx-seek-server"

die() { echo "install-service.sh: $*" >&2; exit 1; }

env_get() {  # env_get KEY DEFAULT
    local key="$1" def="$2"
    [ -f "$ENV_FILE" ] || { echo "$def"; return; }
    local v
    v="$(sed -n "s/^${key}=//p" "$ENV_FILE" | tail -1)"
    [ -n "$v" ] && echo "$v" || echo "$def"
}

require_bin() {
    [ -x "$BIN" ] || die "服务未构建：先运行 deploy/server/build-server.sh"
}

case "${1:-install}" in
install)
    require_bin
    if [ ! -f "$ENV_FILE" ]; then
        cp "$HERE/.env.example" "$ENV_FILE"
        echo "已从 .env.example 生成 $ENV_FILE"
    fi
    echo "== 软链 $SRC -> $DST =="
    sudo ln -sfn "$SRC" "$DST"
    sudo systemctl daemon-reload
    sudo systemctl enable "$NAME"
    sudo systemctl restart "$NAME"
    sleep 2
    systemctl --no-pager --full status "$NAME" || true
    HOST_="$(env_get HOST 0.0.0.0)"; PORT_="$(env_get PORT 8100)"
    case "$HOST_" in 0.0.0.0|::|"") HOST_=127.0.0.1 ;; esac
    echo
    echo "健康检查：curl -s http://$HOST_:$PORT_/healthz"
    echo "日志：    journalctl -u vlx-seek -f"
    ;;
uninstall)
    sudo systemctl disable --now "$NAME" 2>/dev/null || true
    [ -L "$DST" ] && sudo rm -f "$DST" || sudo rm -f "$DST" 2>/dev/null || true
    sudo systemctl daemon-reload
    echo "已移除 $DST（软链目标 $SRC 保留）"
    ;;
status)
    echo "== 软链 =="
    ls -l "$DST" 2>/dev/null || echo "未安装（$DST 不存在）"
    echo
    systemctl --no-pager --full status "$NAME" 2>/dev/null || true
    HOST_="$(env_get HOST 0.0.0.0)"; PORT_="$(env_get PORT 8100)"
    case "$HOST_" in 0.0.0.0|::|"") HOST_=127.0.0.1 ;; esac
    echo
    echo "== /healthz =="
    curl -s --max-time 3 "http://$HOST_:$PORT_/healthz" || echo "（无响应）"
    ;;
restart)
    sudo systemctl restart "$NAME"
    sleep 2
    systemctl --no-pager --full status "$NAME" || true
    ;;
*)
    die "用法: $0 [install|uninstall|status|restart]"
    ;;
esac
