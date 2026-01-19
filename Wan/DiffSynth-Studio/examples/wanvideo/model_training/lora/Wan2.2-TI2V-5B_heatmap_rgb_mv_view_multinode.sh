#!/bin/bash

# ==============================================
# 多机分布式训练启动脚本
# View Concatenation Mode (6 views: 3 RGB + 3 Heatmap)
# ==============================================
# 使用说明：
# 集群调度系统会自动设置以下环境变量：
#   WORLD_SIZE = 总机器数（节点数）
#   RANK = 当前机器的rank（0到WORLD_SIZE-1）
#   MASTER_ADDR = 主节点IP地址
#   MASTER_PORT = 主节点端口
#
# 脚本会自动从这些环境变量读取配置，无需手动设置
# ==============================================

# ==============================================
# 多机分布式训练配置
# ==============================================
# 从集群环境变量获取分布式训练参数
# 集群调度系统设置的环境变量：
#   WORLD_SIZE = 总机器数（节点数）
#   RANK = 当前机器的rank
#   MASTER_ADDR = 主节点IP
#   MASTER_PORT = 主节点端口

# 检查必需的环境变量
if [ -z "${WORLD_SIZE}" ]; then
    echo "错误: 环境变量 WORLD_SIZE 未设置（应该等于总机器数）"
    exit 1
fi

if [ -z "${RANK}" ]; then
    echo "错误: 环境变量 RANK 未设置（应该是当前机器的rank）"
    exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
    echo "错误: 环境变量 MASTER_ADDR 未设置"
    exit 1
fi

if [ -z "${MASTER_PORT}" ]; then
    echo "错误: 环境变量 MASTER_PORT 未设置"
    exit 1
fi

# 设置accelerate需要的变量名
NUM_MACHINES=${WORLD_SIZE}
MACHINE_RANK=${RANK}
MAIN_PROCESS_IP="${MASTER_ADDR}"
MAIN_PROCESS_PORT=${MASTER_PORT}

echo "检测到集群环境变量:"
echo "  WORLD_SIZE (总机器数): ${WORLD_SIZE}"
echo "  RANK (当前机器rank): ${RANK}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"

# 自动检测当前节点的GPU数量
NUM_GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "  当前节点GPU数量: ${NUM_GPUS_PER_NODE}"

# 计算总GPU数量
NUM_GPUS=$((NUM_MACHINES * NUM_GPUS_PER_NODE))
echo "  总GPU数量: ${NUM_GPUS}"

# 网络配置（阿里云推荐配置）
# 优先使用环境变量，如果不存在则使用默认值
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

echo "================================================================"
echo "多机分布式训练配置"
echo "================================================================"
echo "主节点地址: ${MAIN_PROCESS_IP}"
echo "主节点端口: ${MAIN_PROCESS_PORT}"
echo "总机器数: ${NUM_MACHINES}"
echo "当前机器Rank: ${MACHINE_RANK}"
echo "每节点GPU数: ${NUM_GPUS_PER_NODE}"
echo "总GPU数: ${NUM_GPUS}"
echo "================================================================"

# ==============================================
# 机器环境配置（CoppeliaSim配置）
# ==============================================
# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置（待填写）
MACHINE2_COPPELIASIM_ROOT="/home/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE2_DISPLAY=":1.0"
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

# 机器3的CoppeliaSim配置
MACHINE3_COPPELIASIM_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE3_DISPLAY=":1.0"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# 机器4的CoppeliaSim配置
MACHINE4_COPPELIASIM_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE4_DISPLAY=":1.0"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"

# 机器5的CoppeliaSim配置
MACHINE5_COPPELIASIM_ROOT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE5_DISPLAY=":1.0"
MACHINE5_CONDA_PATH="/DATA/disk1/yaoliang/miniconda3/etc/profile.d/conda.sh"
MACHINE5_CONDA_ENV="BridgeVLA_DM"

# 机器6的CoppeliaSim配置
MACHINE6_COPPELIASIM_ROOT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE6_DISPLAY=":1.0"
MACHINE6_CONDA_PATH="/root/miniconda3/etc/profile.d/conda.sh"
MACHINE6_CONDA_ENV="BridgeVLA_DM"

# 通过COPPELIASIM_ROOT检测机器并设置环境
if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器1（基于COPPELIASIM_ROOT），使用配置1"
    CURRENT_MACHINE="machine1"
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE1_DISPLAY}"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    echo "已设置机器1的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE2_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器2（基于COPPELIASIM_ROOT），使用配置2"
    CURRENT_MACHINE="machine2"
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE2_DISPLAY}"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    echo "已设置机器2的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE3_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器3（基于COPPELIASIM_ROOT），使用配置3"
    CURRENT_MACHINE="machine3"
    export COPPELIASIM_ROOT="${MACHINE3_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE3_DISPLAY}"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
    echo "已设置机器3的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE4_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器4（基于COPPELIASIM_ROOT），使用配置4"
    CURRENT_MACHINE="machine4"
    export COPPELIASIM_ROOT="${MACHINE4_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE4_DISPLAY}"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
    echo "已设置机器4的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE5_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器5（基于COPPELIASIM_ROOT），使用配置5"
    CURRENT_MACHINE="machine5"
    export COPPELIASIM_ROOT="${MACHINE5_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE5_DISPLAY}"
    source "${MACHINE5_CONDA_PATH}"
    conda activate "${MACHINE5_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
    echo "已设置机器5的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE6_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器6（基于COPPELIASIM_ROOT），使用配置6"
    CURRENT_MACHINE="machine6"
    export COPPELIASIM_ROOT="${MACHINE6_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE6_DISPLAY}"
    source "${MACHINE6_CONDA_PATH}"
    conda activate "${MACHINE6_CONDA_ENV}"
    echo "已设置机器6的CoppeliaSim环境变量和conda环境"
else
    echo "错误：未找到COPPELIASIM_ROOT路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  机器2: ${MACHINE2_COPPELIASIM_ROOT}"
    echo "  机器3: ${MACHINE3_COPPELIASIM_ROOT}"
    echo "  机器4: ${MACHINE4_COPPELIASIM_ROOT}"
    echo "  机器5: ${MACHINE5_COPPELIASIM_ROOT}"
    echo "  机器6: ${MACHINE6_COPPELIASIM_ROOT}"
    exit 1
fi

# 补丁，在服务器上执行时似乎swanlab得重新安装，不知道为什么
# 强制重新安装
pip install --force-reinstall --no-deps swanlab



# 内存优化环境变量 - 保守的显存优化（与单机版本一致）
# 注意：expandable_segments=True 会增加显存占用，在单机训练时建议不使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2  # 进一步限制CPU线程以节省内存
export CUDA_LAUNCH_BLOCKING=0  # 多机训练建议设为0以提高性能

echo ""
echo "================================================================"
echo "CUDA 内存配置"
echo "================================================================"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "CUDA_LAUNCH_BLOCKING: ${CUDA_LAUNCH_BLOCKING}"
echo "================================================================"
echo ""

# CUDA驱动优化设置
export CUDA_MODULE_LOADING=LAZY
export CUDA_VISIBLE_DEVICES_ORDER=PCI_BUS_ID
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# 额外的稳定性设置
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 启用详细的错误追踪
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json

# ==============================================
# 项目路径配置
# ==============================================

# 机器1的项目路径配置
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE1_DEEPSPEED_CONFIG_DIR="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器2的项目路径配置
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE2_DEEPSPEED_CONFIG_DIR="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器3的项目路径配置
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE3_DEEPSPEED_CONFIG_DIR="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器4的项目路径配置
MACHINE4_PROJECT_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE4_DEEPSPEED_CONFIG_DIR="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器5的项目路径配置
MACHINE5_PROJECT_ROOT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE5_DEEPSPEED_CONFIG_DIR="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器6的项目路径配置
MACHINE6_PROJECT_ROOT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE6_DEEPSPEED_CONFIG_DIR="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 根据机器类型设置项目路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PROJECT_ROOT="${MACHINE1_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE1_DEEPSPEED_CONFIG_DIR}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PROJECT_ROOT="${MACHINE2_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE2_DEEPSPEED_CONFIG_DIR}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    PROJECT_ROOT="${MACHINE3_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE3_DEEPSPEED_CONFIG_DIR}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    PROJECT_ROOT="${MACHINE4_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE4_DEEPSPEED_CONFIG_DIR}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    PROJECT_ROOT="${MACHINE5_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE5_DEEPSPEED_CONFIG_DIR}"
elif [ "${CURRENT_MACHINE}" = "machine6" ]; then
    PROJECT_ROOT="${MACHINE6_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE6_DEEPSPEED_CONFIG_DIR}"
fi

# 检查项目根目录是否存在
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "错误：项目根目录不存在: ${PROJECT_ROOT}"
    exit 1
fi

# 多进程启动方法设置
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# 设置工作目录
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_HEATMAP_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器2的路径配置（待填写）
MACHINE2_HEATMAP_DATA_ROOT="/data/Franka_data_3zed_5/"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B"

# 机器3的路径配置
MACHINE3_HEATMAP_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器4的路径配置
MACHINE4_HEATMAP_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B"

# 机器5的路径配置
MACHINE5_HEATMAP_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/pour_filter"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B"

# 机器6的路径配置
MACHINE6_HEATMAP_DATA_ROOT="/mnt/robot-rfm/user/lpy/data/Franka_data_3zed_5/cook_6"
MACHINE6_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE6_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B"

task="cook_6"

# 根据机器类型设置路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    HEATMAP_DATA_ROOT="${MACHINE1_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    HEATMAP_DATA_ROOT="${MACHINE2_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    HEATMAP_DATA_ROOT="${MACHINE3_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    HEATMAP_DATA_ROOT="${MACHINE4_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    HEATMAP_DATA_ROOT="${MACHINE5_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE5_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine6" ]; then
    HEATMAP_DATA_ROOT="${MACHINE6_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE6_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE6_MODEL_BASE_PATH}"
fi

# ==============================================
# 多任务训练和Trail范围过滤配置
# ==============================================
TRAIL_START="1"
TRAIL_END="8"

# 指定Wan模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV"
# 热力图专用参数
SEQUENCE_LENGTH=24
STEP_INTERVAL=1
MIN_TRAIL_LENGTH=10
HEATMAP_SIGMA=1.5
COLORMAP_NAME="jet"

# 历史帧配置
NUM_HISTORY_FRAMES=1

# 验证 NUM_HISTORY_FRAMES 的合法性
is_valid_history_frames=false
if [ ${NUM_HISTORY_FRAMES} -eq 1 ]; then
    is_valid_history_frames=true
elif [ ${NUM_HISTORY_FRAMES} -eq 2 ]; then
    is_valid_history_frames=true
elif [ ${NUM_HISTORY_FRAMES} -gt 2 ]; then
    remainder=$(( (${NUM_HISTORY_FRAMES} - 1) % 4 ))
    if [ ${remainder} -eq 0 ]; then
        is_valid_history_frames=true
    fi
fi

if [ "${is_valid_history_frames}" != "true" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} is invalid!"
    echo "       Allowed values: 1, 2, or 1+4N (5, 9, 13, 17, ...)"
    exit 1
fi
# 验证 WAN_TYPE 和 NUM_HISTORY_FRAMES 的一致性
if [ ${NUM_HISTORY_FRAMES} -gt 1 ] && [ "${WAN_TYPE}" != "5B_TI2V_RGB_HEATMAP_MV_HISTORY" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} > 1, but WAN_TYPE=${WAN_TYPE}"
    echo "       When using multi-frame history, you MUST set WAN_TYPE=\"5B_TI2V_RGB_HEATMAP_MV_HISTORY\""
    exit 1
fi
if [ ${NUM_HISTORY_FRAMES} -eq 1 ] && [ "${WAN_TYPE}" == "5B_TI2V_RGB_HEATMAP_MV_HISTORY" ]; then
    echo "ERROR: WAN_TYPE=5B_TI2V_RGB_HEATMAP_MV_HISTORY, but NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES}"
    echo "       When using 5B_TI2V_RGB_HEATMAP_MV_HISTORY, you MUST set NUM_HISTORY_FRAMES > 1"
    exit 1
fi

# 图像和训练参数
HEIGHT=256
WIDTH=256
NUM_FRAMES=${SEQUENCE_LENGTH}
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=100
SAVE_EPOCHS_INTERVAL=10

# 多机训练参数调整
TRAIN_BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=1
# 计算总的有效批次大小（所有GPU的总和）
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))
echo "Total GPUs across all nodes: ${NUM_GPUS}"
echo "Total effective batch size: ${EFFECTIVE_BATCH_SIZE}"

# 显存优化参数
DATASET_NUM_WORKERS=0
USE_GRADIENT_CHECKPOINTING=false
MIXED_PRECISION="bf16"
DATALOADER_PIN_MEMORY=false
PREFETCH_FACTOR=2

# LoRA参数
LORA_RANK=32
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# Dual Head模式
USE_DUAL_HEAD=true

# 损失权重配置
RGB_LOSS_WEIGHT=0.08

# Modulation 和 Norm 解冻控制
UNFREEZE_MODULATION_AND_NORMS=true

# 预训练模型加载配置
LOAD_PRETRAINED_CHECKPOINT=False
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NAME="multinode_${NUM_MACHINES}nodes_${NUM_GPUS}cards_${LOAD_PRETRAINED_CHECKPOINT}_history_${NUM_HISTORY_FRAMES}_seq_${SEQUENCE_LENGTH}_new_projection"
OUTPUT_PATH="${OUTPUT_BASE}/Wan2.2-TI2V-5B_heatmap_rgb_lora_view/5_trajectory_${task}_3camera_${NAME}_rgb_loss_${RGB_LOSS_WEIGHT}/${TIMESTAMP}"

# 预训练checkpoint路径配置
MACHINE1_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"
MACHINE2_PRETRAINED_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true/20251208_045615/epoch-15.safetensors"
MACHINE3_PRETRAINED_CHECKPOINT="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_cook_5_3camera_pour_filter_pretrain_False_history_1_seq_48_new_projection_rgb_loss_1/20260105_203050/epoch-99.safetensors"
MACHINE4_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"
MACHINE5_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"
MACHINE6_PRETRAINED_CHECKPOINT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"

# 根据机器类型设置预训练checkpoint路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE1_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE2_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE3_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE4_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE5_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine6" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE6_PRETRAINED_CHECKPOINT}"
fi

# 点云配置
USE_MERGED_POINTCLOUD=false

# 投影模式配置
USE_DIFFERENT_PROJECTION=true

# 数据增强参数
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.1,0.1,0.1"
TRANSFORM_AUG_RPY="5.0,5.0,5.0"

# SwanLab配置参数
ENABLE_SWANLAB=true
SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"
SWANLAB_PROJECT="Wan2.2-TI2V-5B_heatmap_rgb_lora_${task}"
SWANLAB_EXPERIMENT="${NAME}-$(date +%Y%m%d-%H%M%S)"
DEBUG_MODE=false

echo "================================================================"
echo "HEATMAP SEQUENCE TRAINING FOR Wan2.2-TI2V-5B_heatmap_rgb_lora"
echo "================================================================"
echo "当前使用机器: ${CURRENT_MACHINE}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据根目录: ${HEATMAP_DATA_ROOT}"
echo "模型路径: ${MODEL_BASE_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
echo "预训练配置:"
echo "  加载预训练权重: ${LOAD_PRETRAINED_CHECKPOINT}"
if [ "${LOAD_PRETRAINED_CHECKPOINT}" = "true" ]; then
    echo "  预训练checkpoint: ${PRETRAINED_CHECKPOINT}"
fi
echo "----------------------------------------------------------------"
echo "训练参数:"
echo "  序列长度: ${SEQUENCE_LENGTH}"
echo "  图像尺寸: ${HEIGHT}x${WIDTH}"
echo "  学习率: ${LEARNING_RATE}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  LoRA目标模块: ${LORA_TARGET_MODULES}"
echo "  双Head模式: ${USE_DUAL_HEAD}"
echo "  RGB Loss权重: ${RGB_LOSS_WEIGHT}"
echo "  解冻Modulation和Norms: ${UNFREEZE_MODULATION_AND_NORMS}"
echo "  点云融合模式: ${USE_MERGED_POINTCLOUD}"
echo "  不同投影模式: ${USE_DIFFERENT_PROJECTION}"
echo "  每节点GPU数量: ${NUM_GPUS_PER_NODE}"
echo "  总GPU数量: ${NUM_GPUS}"
echo "  每GPU批次大小: ${TRAIN_BATCH_SIZE_PER_GPU}"
echo "  有效批次大小: ${EFFECTIVE_BATCH_SIZE}"
echo "  数据加载线程数: ${DATASET_NUM_WORKERS}"
echo "  混合精度: ${MIXED_PRECISION}"
echo "  梯度检查点: ${USE_GRADIENT_CHECKPOINTING}"
echo "  保存间隔: 每${SAVE_EPOCHS_INTERVAL}个epoch保存一次"
echo "  历史帧数量: ${NUM_HISTORY_FRAMES}"
echo "  SwanLab启用: ${ENABLE_SWANLAB}"
echo "  调试模式: ${DEBUG_MODE}"
echo "================================================================"

# ==============================================
# 路径验证
# ==============================================

# 检查数据目录
if [ ! -d "${HEATMAP_DATA_ROOT}" ]; then
    echo "错误：数据目录不存在: ${HEATMAP_DATA_ROOT}"
    echo "请检查数据路径并重试。"
    exit 1
fi

# 检查模型目录
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "错误：模型目录不存在: ${MODEL_BASE_PATH}"
    echo "请检查模型路径并重试。"
    exit 1
fi

# 检查预训练checkpoint路径（如果启用）
if [ "${LOAD_PRETRAINED_CHECKPOINT}" = "true" ]; then
    if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
        echo "错误：预训练checkpoint文件不存在: ${PRETRAINED_CHECKPOINT}"
        echo "请检查预训练checkpoint路径并重试，或设置 LOAD_PRETRAINED_CHECKPOINT=false"
        exit 1
    fi
    echo "✓ 预训练checkpoint验证通过: ${PRETRAINED_CHECKPOINT}"
fi

# 创建输出目录（只在主节点创建）
if [ "${MACHINE_RANK}" -eq 0 ]; then
    mkdir -p "${OUTPUT_PATH}"
    echo "输出目录已创建: ${OUTPUT_PATH}"
fi

# ================================================================
# 多机分布式训练
# ================================================================
echo "================================================================"
echo "STARTING MULTI-NODE DISTRIBUTED TRAINING"
echo "================================================================"
echo ""
echo "Current configuration:"
echo "   - This machine rank: ${MACHINE_RANK}"
echo "   - Total machines: ${NUM_MACHINES}"
echo "   - GPUs per machine: ${NUM_GPUS_PER_NODE}"
echo "   - Total GPUs across all machines: ${NUM_GPUS}"
echo "   - Master address: ${MAIN_PROCESS_IP}"
echo "   - Master port: ${MAIN_PROCESS_PORT}"
echo ""
if [ "${MACHINE_RANK}" -eq 0 ]; then
    echo "📍 This is the MASTER machine (rank 0)"
    echo "   Other machines will connect to this machine at ${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
    echo ""
    echo "⚠️  IMPORTANT: The training will HANG if not all machines connect!"
    echo ""
else
    echo "📍 This is WORKER machine (rank ${MACHINE_RANK})"
    echo "   Connecting to master at ${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
    echo ""
    echo "⚠️  This machine will WAIT for the master machine to initialize."
    echo "   If training hangs, check network connectivity to ${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
    echo ""
fi
echo "================================================================"
echo ""

# 使用accelerate launch启动多机训练
echo "🚀 Launching multi-node training on machine ${MACHINE_RANK}..."
echo "   If you don't see all ${NUM_GPUS} GPUs being used, check that you've started"
echo "   the script on all ${NUM_MACHINES} machines with correct RANK values."
echo ""

# 打印环境变量以便调试
echo "Environment variables for debugging:"
echo "  WORLD_SIZE=${WORLD_SIZE}"
echo "  RANK=${RANK}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NUM_MACHINES=${NUM_MACHINES}"
echo "  MACHINE_RANK=${MACHINE_RANK}"
echo "  NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo ""

# 设置 accelerate 需要的环境变量（确保传递给训练脚本）
export MASTER_ADDR="${MAIN_PROCESS_IP}"
export MASTER_PORT="${MAIN_PROCESS_PORT}"

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=${NUM_MACHINES} \
  --machine_rank=${MACHINE_RANK} \
  --main_process_ip="${MAIN_PROCESS_IP}" \
  --main_process_port=${MAIN_PROCESS_PORT} \
  --mixed_precision=${MIXED_PRECISION} \
  --same_network \
  examples/wanvideo/model_training/heatmap_train_mv_view.py \
  --heatmap_data_root ${HEATMAP_DATA_ROOT} \
  $(if [ -n "${TRAIL_START}" ]; then echo "--trail_start ${TRAIL_START}"; fi) \
  $(if [ -n "${TRAIL_END}" ]; then echo "--trail_end ${TRAIL_END}"; fi) \
  --sequence_length ${SEQUENCE_LENGTH} \
  --step_interval ${STEP_INTERVAL} \
  --min_trail_length ${MIN_TRAIL_LENGTH} \
  --heatmap_sigma ${HEATMAP_SIGMA} \
  --colormap_name "${COLORMAP_NAME}" \
  --scene_bounds="${SCENE_BOUNDS}" \
  --transform_augmentation_xyz="${TRANSFORM_AUG_XYZ}" \
  --transform_augmentation_rpy="${TRANSFORM_AUG_RPY}" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --num_frames ${NUM_FRAMES} \
  --dataset_repeat ${DATASET_REPEAT} \
  --wan_type ${WAN_TYPE} \
  --model_paths '[
    [
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00001-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00002-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "'${MODEL_BASE_PATH}'/models_t5_umt5-xxl-enc-bf16.pth",
    "'${MODEL_BASE_PATH}'/Wan2.2_VAE.pth"
    ]' \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --save_epochs_interval ${SAVE_EPOCHS_INTERVAL} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank ${LORA_RANK} \
  $(if [ "${LOAD_PRETRAINED_CHECKPOINT}" = "true" ]; then echo "--lora_checkpoint ${PRETRAINED_CHECKPOINT}"; fi) \
  --extra_inputs "input_image,input_image_rgb,input_video_rgb" \
  --train_batch_size ${TRAIN_BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} \
  $(if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then echo "--use_gradient_checkpointing"; fi) \
  $(if [ "${USE_DUAL_HEAD}" = "true" ]; then echo "--use_dual_head"; fi) \
  --rgb_loss_weight ${RGB_LOSS_WEIGHT} \
  $(if [ "${UNFREEZE_MODULATION_AND_NORMS}" = "true" ]; then echo "--unfreeze_modulation_and_norms"; fi) \
  $(if [ "${USE_MERGED_POINTCLOUD}" = "true" ]; then echo "--use_merged_pointcloud"; fi) \
  $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi) \
  --num_history_frames ${NUM_HISTORY_FRAMES} \
  --dataloader_pin_memory false \
  --save_steps 0 \
  --logging_steps 10 \
  --max_grad_norm 1.0 \
  --warmup_steps 100 \
  $(if [ "${DEBUG_MODE}" = "true" ]; then echo "--debug_mode"; fi) \
  $(if [ "${ENABLE_SWANLAB}" = "true" ]; then echo "--enable_swanlab"; fi) \
  --swanlab_api_key "${SWANLAB_API_KEY}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment "${SWANLAB_EXPERIMENT}"

echo "================================================================"
echo "Training completed on node ${MACHINE_RANK}!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"
sleep 360000000