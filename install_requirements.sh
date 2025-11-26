#!/bin/bash

# 激活环境
source /home/yw/anaconda3/bin/activate BridgeVLA_DM

# 读取 requirements.txt 并逐个安装
requirements_file="/DATA/disk1/lpy/BridgeVLA_dev/requirements.txt"

echo "========================================="
echo "开始安装依赖包..."
echo "========================================="

# 计数器
total=0
success=0
skipped=0

# 创建失败列表文件
failed_packages="/DATA/disk1/lpy/BridgeVLA_dev/failed_packages.txt"
> "$failed_packages"

# 读取文件并处理每一行
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行和注释行
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    total=$((total + 1))

    # 获取包名（处理 == 格式）
    package=$(echo "$line" | tr -d '[:space:]')
    package_name=$(echo "$package" | cut -d'=' -f1)

    echo ""
    echo "[$total] 正在安装: $package"
    echo "-----------------------------------------"

    # 尝试安装，设置超时时间为10分钟
    timeout 600 pip install "$package" --no-cache-dir 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ 成功: $package"
        success=$((success + 1))
    elif [ $? -eq 124 ]; then
        echo "✗ 超时跳过: $package (超过10分钟)"
        echo "$package (超时)" >> "$failed_packages"
        skipped=$((skipped + 1))
    else
        echo "✗ 失败跳过: $package"
        echo "$package (安装失败)" >> "$failed_packages"
        skipped=$((skipped + 1))
    fi

done < "$requirements_file"

echo ""
echo "========================================="
echo "安装完成！"
echo "========================================="
echo "总计: $total 个包"
echo "成功: $success 个"
echo "跳过/失败: $skipped 个"
echo "========================================="

if [ $skipped -gt 0 ]; then
    echo ""
    echo "以下包未能成功安装："
    echo "-----------------------------------------"
    cat "$failed_packages"
    echo "-----------------------------------------"
    echo "详细列表已保存到: $failed_packages"
fi

echo ""
echo "完整日志已保存到: /DATA/disk1/lpy/BridgeVLA_dev/install_log.txt"
