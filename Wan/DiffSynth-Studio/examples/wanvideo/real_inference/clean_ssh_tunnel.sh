#!/bin/bash
#
# 清理SSH隧道缓存脚本
# 用于解决SSH端口转发失效的问题
#
# 使用方法:
#   ./clean_ssh_tunnel.sh          # 清理5555端口（默认）
#   ./clean_ssh_tunnel.sh 5556     # 清理指定端口
#

PORT=${1:-5555}

echo "=============================================="
echo "SSH隧道缓存清理工具"
echo "=============================================="
echo ""

# 1. 检查并杀掉占用端口的进程
echo "[1/3] 检查端口 $PORT 占用情况..."
PIDS=$(lsof -t -i :$PORT 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "      发现占用进程: $PIDS"
    echo "      正在终止..."
    kill -9 $PIDS 2>/dev/null
    echo "      ✓ 已终止"
else
    echo "      端口 $PORT 未被占用"
fi
echo ""

# 2. 杀掉所有SSH端口转发进程
echo "[2/3] 检查SSH端口转发进程..."
SSH_TUNNEL_PIDS=$(ps aux | grep "ssh -L" | grep -v grep | awk '{print $2}')
if [ -n "$SSH_TUNNEL_PIDS" ]; then
    echo "      发现SSH隧道进程: $SSH_TUNNEL_PIDS"
    echo "      正在终止..."
    echo $SSH_TUNNEL_PIDS | xargs kill -9 2>/dev/null
    echo "      ✓ 已终止"
else
    echo "      无活跃的SSH隧道进程"
fi
echo ""

# 3. 清理SSH多路复用socket文件
echo "[3/3] 清理SSH多路复用缓存..."
SSH_SOCKETS=$(ls ~/.ssh/cm-* 2>/dev/null)
if [ -n "$SSH_SOCKETS" ]; then
    echo "      发现socket文件:"
    for sock in $SSH_SOCKETS; do
        echo "        - $(basename $sock)"
    done
    rm -f ~/.ssh/cm-* 2>/dev/null
    echo "      ✓ 已清理"
else
    echo "      无缓存的socket文件"
fi
echo ""

# 4. 验证清理结果
echo "=============================================="
echo "清理完成！验证结果："
echo "=============================================="
echo ""

# 检查端口
if lsof -i :$PORT >/dev/null 2>&1; then
    echo "⚠ 警告: 端口 $PORT 仍被占用"
    lsof -i :$PORT
else
    echo "✓ 端口 $PORT 已释放"
fi
echo ""

# 检查SSH隧道
if ps aux | grep "ssh -L" | grep -v grep >/dev/null 2>&1; then
    echo "⚠ 警告: 仍有SSH隧道进程运行"
    ps aux | grep "ssh -L" | grep -v grep
else
    echo "✓ 无SSH隧道进程"
fi
echo ""

# 检查socket
if ls ~/.ssh/cm-* >/dev/null 2>&1; then
    echo "⚠ 警告: 仍有SSH socket文件"
    ls ~/.ssh/cm-*
else
    echo "✓ SSH socket已清理"
fi
echo ""

echo "=============================================="
echo "现在可以重新建立SSH隧道了"
echo "=============================================="
