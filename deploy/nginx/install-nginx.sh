#!/usr/bin/env bash
# 用软链把 deploy/nginx/vlx-seek.conf 挂进 nginx，并重载
#
#   ./deploy/nginx/install-nginx.sh install     # ln -sfn + nginx -t + reload
#   ./deploy/nginx/install-nginx.sh status      # 软链 / 语法 / 端口 / HTTP 检查
#   ./deploy/nginx/install-nginx.sh uninstall   # 删除软链 + reload
#
# 需要 sudo（写 /etc/nginx/conf.d）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="vlx-seek.conf"
SRC="$HERE/$NAME"
DST="/etc/nginx/conf.d/$NAME"
WEB="$HERE/../web/index.html"

die() { echo "install-nginx.sh: $*" >&2; exit 1; }

port_of() { grep -m1 -oP 'listen\s+\K[0-9]+' "$SRC" 2>/dev/null || echo 8188; }
PORT="$(port_of)"

check_http() {
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/" || true)"
    echo "  http://127.0.0.1:$PORT/  -> HTTP ${code:-000}"
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/healthz" || true)"
    echo "  http://127.0.0.1:$PORT/healthz -> HTTP ${code:-000}"
}

case "${1:-install}" in
install)
    command -v nginx >/dev/null || die "nginx 未安装"
    [ -f "$WEB" ] || die "缺少控制台页面：$WEB"
    [ -f "$SRC" ] || die "缺少配置：$SRC"
    echo "== 软链 $SRC -> $DST =="
    sudo ln -sfn "$SRC" "$DST"
    echo "== nginx -t =="
    sudo nginx -t
    echo "== 重载 =="
    sudo systemctl reload nginx
    sleep 1
    check_http
    echo
    echo "浏览器访问： http://<本机IP>:$PORT/"
    echo "登录密码   ： deploy/.env 里的 API_KEY"
    ;;
uninstall)
    sudo rm -f "$DST"
    sudo nginx -t && sudo systemctl reload nginx || true
    echo "已移除 $DST（软链目标 $SRC 保留）"
    ;;
status)
    echo "== 软链 =="
    ls -l "$DST" 2>/dev/null || echo "未安装（$DST 不存在）"
    echo "== 语法 =="
    sudo nginx -t 2>&1 | tail -2
    echo "== 监听 =="
    ss -ltn | grep ":$PORT " || echo "  端口 $PORT 未监听"
    echo "== HTTP =="
    check_http
    ;;
*)
    die "用法: $0 [install|uninstall|status]"
    ;;
esac
