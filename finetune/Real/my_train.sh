#!/bin/bash

# =============================================================================
# BridgeVLA 自定义训练脚本
# 修改自原始的train.sh，适配您的环境
# =============================================================================

echo "🚀 开始BridgeVLA训练..."
echo "当前时间: $(date)"
echo "当前用户: $(whoami)"
echo "当前目录: $(pwd)"

# =============================================================================
# 环境配置部分 - 请根据您的实际环境修改
# =============================================================================

# 设置 Hugging Face 缓存路径 (根据您的存储空间调整)
export HF_HOME="/home/lpy/BridgeVLA_dev/huggingface_cache"

# 切换到项目目录
cd /home/lpy/BridgeVLA_dev/finetune

# CoppeliaSim 配置 (如果不使用仿真可以注释掉)
# export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# export DISPLAY=:1.0

# 切换到Real目录
cd /home/lpy/BridgeVLA_dev/finetune/Real

# 创建必要的目录
mkdir -p logs
mkdir -p /home/lpy/BridgeVLA_dev/huggingface_cache

echo "📋 所有传入的参数：$@"

# =============================================================================
# GPU 和分布式训练配置
# =============================================================================

# 设置可见的GPU (根据您的GPU数量和编号调整)
# 示例：使用GPU 0 (单卡训练)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 示例：使用多张GPU (多卡训练)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 调试选项 (生产环境可以注释掉)
export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 只在调试时启用

# 检查GPU可用性
echo "🔍 检查GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# =============================================================================
# 训练参数配置
# =============================================================================

# 获取GPU数量
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
echo "📊 使用GPU数量: $GPU_COUNT"

# 设置随机端口避免冲突
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "🌐 Master Port: $MASTER_PORT"

# =============================================================================
# 启动训练
# =============================================================================

if [ $GPU_COUNT -eq 1 ]; then
    echo "🚀 启动单GPU训练..."
    python3 train.py $@
else
    echo "🚀 启动多GPU分布式训练..."
    torchrun --nnodes=1 \
             --node_rank=0 \
             --master_port=$MASTER_PORT \
             --nproc_per_node=$GPU_COUNT \
             train.py $@
fi

# =============================================================================
# 预定义的训练配置示例
# =============================================================================

# 您可以取消注释以下任一配置来快速开始训练：

# 1. 调试模式 (最小配置，快速测试)
: '
bash my_train.sh \
    --debug \
    --exp_cfg_path configs/real.yaml \
    --exp_note debug_test \
    --cameras 3rd \
    --ep_per_task 1 \
    --data_folder /home/lpy/BridgeVLA_dev/finetune/Real/data/put_code_can_on_top_shelf_with_different_rotation
'

# 2. 完整训练模式
: '
bash my_train.sh \
    --exp_cfg_path configs/real.yaml \
    --exp_note my_full_training \
    --cameras 3rd \
    --ep_per_task 10 \
    --data_folder /home/lpy/BridgeVLA_dev/finetune/Real/data/put_code_can_on_top_shelf_with_different_rotation \
    --test_split_ratio 0.1 \
    --freeze_vision_tower \
    --load_pretrain \
    --pretrain_path /path/to/your/pretrained/model
'

# 3. 多数据集训练
: '
bash my_train.sh \
    --exp_cfg_path configs/real.yaml \
    --exp_note multi_dataset_training \
    --cameras 3rd \
    --ep_per_task 15 \
    --data_folder /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
    --test_split_ratio 0.1
'

echo "✅ 训练脚本执行完成!"
echo "📝 日志文件位置: logs/"
echo "💾 模型保存位置: logs/train/*/models/"
