#!/usr/bin/env bash
# 把最小运行集送到服务器（tar over SSH，需已配免密登录）。
#
#   bash deploy/upload.sh ubuntu@106.54.192.87
#
# 只上传 detector/ app/ models/ requirements.txt README.md。
# 训练语料、实验脚本、论文草稿一律不上传。
set -euo pipefail

TARGET="${1:-ubuntu@106.54.192.87}"
REMOTE_DIR="${REMOTE_DIR:-/opt/ttascope}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$HERE"

echo "==> 目标 $TARGET:$REMOTE_DIR"
ssh -o ConnectTimeout=10 "$TARGET" \
  "sudo mkdir -p $REMOTE_DIR && sudo chown -R \$(id -u):\$(id -g) $REMOTE_DIR"

tar czf - detector app models deploy requirements.txt README.md \
  | ssh "$TARGET" "tar xzf - -C $REMOTE_DIR"

echo "==> 上传完成，共 $(ssh "$TARGET" "du -sh $REMOTE_DIR | cut -f1")"
echo "==> 接下来在服务器上执行："
echo "    sudo bash $REMOTE_DIR/deploy/install.sh"
