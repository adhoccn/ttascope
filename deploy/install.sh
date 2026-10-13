#!/usr/bin/env bash
# 在服务器上安装 TTAScope（venv + systemd + nginx）。
#
#   sudo bash deploy/install.sh                 # 挂到已有站点 8501 的 /tta/ 子路径
#   sudo bash deploy/install.sh --port 8510     # 独立端口（需安全组放行）
#
# 默认模式不改安全组：复用已放行的 8501，在 /etc/nginx/conf.d/hybridscope.conf
# 里插入 location /tta/，把请求转发到 127.0.0.1:8511。
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ttascope}"
SERVICE=ttascope
UPSTREAM_PORT=8511
PUBLIC_PORT=""
NGINX_CONF=/etc/nginx/conf.d/hybridscope.conf
PY_MIRROR="https://mirrors.cloud.tencent.com/pypi/simple"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PUBLIC_PORT="$2"; shift 2;;
    *) echo "未知参数 $1"; exit 1;;
  esac
done

RUN_USER="$(logname 2>/dev/null || echo ubuntu)"
echo "==> 应用目录 $APP_DIR，运行用户 $RUN_USER"

cd "$APP_DIR"

# 1. 虚拟环境
if [ ! -d .venv ]; then
  echo "==> 创建虚拟环境"
  sudo -u "$RUN_USER" python3 -m venv .venv
fi
echo "==> 安装依赖（腾讯 PyPI 镜像）"
sudo -u "$RUN_USER" .venv/bin/pip install -q --upgrade pip
sudo -u "$RUN_USER" .venv/bin/pip install -q -i "$PY_MIRROR" -r requirements.txt

# 2. systemd
echo "==> 写入 systemd 服务"
sed -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__USER__|$RUN_USER|g" \
    -e "s|__PORT__|$UPSTREAM_PORT|g" \
    deploy/ttascope.service > /etc/systemd/system/$SERVICE.service

if [ -n "$PUBLIC_PORT" ]; then
  # 独立端口模式：直接监听公网，需自行放行安全组
  sed -i "s|--host 127.0.0.1|--host 0.0.0.0|; s|--port $UPSTREAM_PORT|--port $PUBLIC_PORT|" \
      /etc/systemd/system/$SERVICE.service
fi

systemctl daemon-reload
systemctl enable --now $SERVICE
sleep 6
systemctl is-active --quiet $SERVICE && echo "==> 服务已启动" || {
  journalctl -u $SERVICE -n 30 --no-pager; exit 1; }

# 3. nginx（仅子路径模式）
if [ -z "$PUBLIC_PORT" ]; then
  if [ -f "$NGINX_CONF" ]; then
    # 3a. 限流区（http 级）+ location 片段，幂等写入，改前自动备份
    python3 deploy/nginx_sync.py --port "$UPSTREAM_PORT" || exit 1
    nginx -t && systemctl reload nginx && echo "==> nginx 已重载"
  else
    echo "!! 未找到 $NGINX_CONF，改用独立端口：sudo bash deploy/install.sh --port 8510"
  fi
fi

echo
if [ -n "$PUBLIC_PORT" ]; then
  echo "访问：http://<公网IP>:$PUBLIC_PORT/"
else
  echo "访问：http://<公网IP>:8501/tta/"
fi
echo "运维：systemctl status|restart $SERVICE    journalctl -u $SERVICE -f"
