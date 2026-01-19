#!/bin/bash

# ==============================================
# 多机分布式训练启动脚本
# View Concatenation Version of mv_rot_grip_decode_feature_4.sh
# ==============================================
# KEY DIFFERENCES:
# 1. Uses VIEW concatenation instead of CHANNEL concatenation
# 2. Direct latent input (not VAE decoder intermediate features)
# 3. Supports multi-node distributed training
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

# 内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8,expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=0  # 多机训练建议设为0以提高性能

# CUDA驱动优化设置
export CUDA_MODULE_LOADING=LAZY
export CUDA_VISIBLE_DEVICES_ORDER=PCI_BUS_ID
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 分布式训练稳定性设置
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 启用详细的错误追踪
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json

# ==============================================
# 机器环境配置（CoppeliaSim配置）
# ==============================================
# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置
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


# ==============================================
# 项目路径配置
# ==============================================

# 机器1的项目路径配置
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器2的项目路径配置
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器3的项目路径配置
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器4的项目路径配置
MACHINE4_PROJECT_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器5的项目路径配置
MACHINE5_PROJECT_ROOT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器6的项目路径配置
MACHINE6_PROJECT_ROOT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 根据机器类型设置项目路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PROJECT_ROOT="${MACHINE1_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PROJECT_ROOT="${MACHINE2_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    PROJECT_ROOT="${MACHINE3_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    PROJECT_ROOT="${MACHINE4_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    PROJECT_ROOT="${MACHINE5_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine6" ]; then
    PROJECT_ROOT="${MACHINE6_PROJECT_ROOT}"
fi

# 检查项目根目录是否存在
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "错误：项目根目录不存在: ${PROJECT_ROOT}"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"

# 设置工作目录
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"


# GPU信息已在上面显示，这里不再重复
# echo "使用GPU: ${CUDA_VISIBLE_DEVICES}"
# echo "GPU数量: ${NUM_GPUS}"

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器2的路径配置
MACHINE2_DATA_ROOT="/data/Franka_data_3zed_5/"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器3的路径配置
MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/pour_filter"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器4的路径配置
MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/pour_filter"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B"

# 机器5的路径配置
MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B"

# 机器6的路径配置
MACHINE6_DATA_ROOT="/mnt/robot-rfm/user/lpy/data/Franka_data_3zed_5/cook_6"
MACHINE6_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE6_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B"




# 根据机器类型设置路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    DATA_ROOT="${MACHINE2_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    DATA_ROOT="${MACHINE3_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    DATA_ROOT="${MACHINE4_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    DATA_ROOT="${MACHINE5_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE5_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine6" ]; then
    DATA_ROOT="${MACHINE6_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE6_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE6_MODEL_BASE_PATH}"
fi

# Trail范围过滤（可选，默认使用所有trails）
TRAIL_START="1"          # 起始trail编号
TRAIL_END="8"            # 结束trail编号

# ==============================================
# 训练参数配置
# ==============================================

# 数据参数
SEQUENCE_LENGTH=24           # 序列长度（不包括初始帧）
IMAGE_SIZE=256              # 图像尺寸
NUM_WORKERS=0               # DataLoader工作线程数（必须为0，因为collate_fn中使用CUDA）

# 数据集参数
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.1 0.1 0.1"
TRANSFORM_AUG_RPY="5.0 5.0 5.0"

# 指定Wan模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"
DEBUG=false

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
if [ ${NUM_HISTORY_FRAMES} -gt 1 ] && [ "${WAN_TYPE}" != "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} > 1, but WAN_TYPE=${WAN_TYPE}"
    echo "       When using multi-frame history, you MUST set WAN_TYPE=\"5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY\""
    exit 1
fi
if [ ${NUM_HISTORY_FRAMES} -eq 1 ] && [ "${WAN_TYPE}" == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY" ]; then
    echo "ERROR: WAN_TYPE=5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, but NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES}"
    echo "       When using 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, you MUST set NUM_HISTORY_FRAMES > 1"
    exit 1
fi

# 模型参数
HIDDEN_DIM=512              # 隐藏层维度
NUM_ROTATION_BINS=72        # 旋转bins数量（对应5度分辨率）
DROPOUT=0.1                 # Dropout率

# 局部特征提取参数（V2新增）
USE_ACCURATE_PEAK_DETECTION=true  # 使用精确的峰值检测
LOCAL_FEATURE_SIZE=5              # 局部特征提取窗口大小

# 点云配置
USE_MERGED_POINTCLOUD=false

# 初始夹爪状态输入配置
USE_INITIAL_GRIPPER_STATE=false

# 投影模式配置
USE_DIFFERENT_PROJECTION=true

# 输出路径（带时间戳）
task="cook_6"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NAME="multinode_${NUM_MACHINES}nodes_${NUM_GPUS}cards_view_concat_${task}_local_feat_size_${LOCAL_FEATURE_SIZE}_seq_${SEQUENCE_LENGTH}_history_${NUM_HISTORY_FRAMES}_3zed_different_projection_${USE_DIFFERENT_PROJECTION}_new_projection_with_gripper_${USE_INITIAL_GRIPPER_STATE}"
OUTPUT_PATH="${OUTPUT_BASE}/mv_rot_grip_v2_view/${NAME}/${TIMESTAMP}"

# 训练参数
NUM_EPOCHS=100               # 训练轮数
LEARNING_RATE=1e-4          # 学习率
WEIGHT_DECAY=1e-5           # 权重衰减
MAX_GRAD_NORM=1.0           # 最大梯度范数
GRADIENT_ACCUMULATION_STEPS=1  # 梯度累积步数
BATCH_SIZE=1                # 每个GPU的batch size

# 计算总的有效批次大小（所有GPU的总和）
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

# 保存和日志参数
SAVE_EPOCH_INTERVAL=1        # 每N个epoch保存一次checkpoint
LOGGING_STEPS=10            # 日志记录步数
SWANLAB_PROJECT="mv_rot_grip"
SWANLAB_EXPERIMENT=$NAME

# VAE参数
HEATMAP_LATENT_SCALE=1.0    # Heatmap latent缩放因子
LATENT_NOISE_STD=0.1        # Latent噪声标准差

# 编码模式选择
USE_ONLINE_ENCODING=true   # 是否使用在线编码

# 缓存参数（仅在USE_ONLINE_ENCODING=false时使用）
NUM_AUGMENTATIONS=1        # 每个样本预编码的增强版本数量

echo "================================================================"
echo "Multi-View Rotation and Gripper Prediction Training (V2 - Multi-Node)"
echo "================================================================"
echo "当前使用机器: ${CURRENT_MACHINE}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据根目录: ${DATA_ROOT}"
echo "模型路径: ${MODEL_BASE_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
echo "分布式训练配置:"
echo "  总机器数: ${NUM_MACHINES}"
echo "  当前机器Rank: ${MACHINE_RANK}"
echo "  每机器GPU数: ${NUM_GPUS_PER_NODE}"
echo "  总GPU数: ${NUM_GPUS}"
echo "  有效批次大小: ${EFFECTIVE_BATCH_SIZE}"
echo "----------------------------------------------------------------"
echo "训练参数:"
echo "  序列长度: ${SEQUENCE_LENGTH}"
echo "  图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  隐藏层维度: ${HIDDEN_DIM}"
echo "  旋转Bins: ${NUM_ROTATION_BINS}"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  点云融合模式: ${USE_MERGED_POINTCLOUD}"
echo "  不同投影模式: ${USE_DIFFERENT_PROJECTION}"
echo "  初始夹爪状态输入: ${USE_INITIAL_GRIPPER_STATE}"
echo "  历史帧数量: ${NUM_HISTORY_FRAMES}"
echo "  WAN_TYPE: ${WAN_TYPE}"
echo "----------------------------------------------------------------"
echo "局部特征参数 (V2新增):"
echo "  精确峰值检测: ${USE_ACCURATE_PEAK_DETECTION}"
echo "  局部特征窗口: ${LOCAL_FEATURE_SIZE}x${LOCAL_FEATURE_SIZE}"
echo "----------------------------------------------------------------"
echo "VAE编码模式:"
if [ "${USE_ONLINE_ENCODING}" = "true" ]; then
    echo "  模式: 在线编码 (slower, no cache needed)"
else
    echo "  模式: 预编码缓存 (faster, requires disk space)"
    echo "  增强版本数: ${NUM_AUGMENTATIONS}"
fi
echo "================================================================"

# ==============================================
# 路径验证
# ==============================================

# 检查数据目录
if [ ! -d "${DATA_ROOT}" ]; then
    echo "错误：数据目录不存在: ${DATA_ROOT}"
    echo "请检查数据路径并重试。"
    exit 1
fi

# 检查模型目录
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "错误：模型目录不存在: ${MODEL_BASE_PATH}"
    echo "请检查模型路径并重试。"
    exit 1
fi

# 创建输出目录（只在主机器创建）
if [ "${MACHINE_RANK}" -eq 0 ]; then
    mkdir -p "${OUTPUT_PATH}"
    echo "输出目录已创建: ${OUTPUT_PATH}"
fi

# ==============================================
# 启动多机分布式训练
# ==============================================

echo "================================================================"
echo "STARTING MULTI-NODE DISTRIBUTED TRAINING"
echo "================================================================"

# 使用accelerate launch启动多机训练
echo "🚀 Launching multi-node training on machine ${MACHINE_RANK}..."

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --num_machines=${NUM_MACHINES} \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip="${MAIN_PROCESS_IP}" \
    --main_process_port=${MAIN_PROCESS_PORT} \
    --mixed_precision=bf16 \
    --same_network \
    examples/wanvideo/model_training/mv_rot_grip_vae_decode_feature_3_view.py \
    --data_root ${DATA_ROOT} \
    --output_path "${OUTPUT_PATH}" \
    $(if [ -n "${TRAIL_START}" ]; then echo "--trail_start ${TRAIL_START}"; fi) \
    $(if [ -n "${TRAIL_END}" ]; then echo "--trail_end ${TRAIL_END}"; fi) \
    --sequence_length ${SEQUENCE_LENGTH} \
    --image_size ${IMAGE_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --scene_bounds="${SCENE_BOUNDS}" \
    --transform_augmentation_xyz ${TRANSFORM_AUG_XYZ} \
    --transform_augmentation_rpy ${TRANSFORM_AUG_RPY} \
    --wan_type "${WAN_TYPE}" \
    $(if [ "${DEBUG}" = "true" ]; then echo "--debug"; fi) \
    --hidden_dim ${HIDDEN_DIM} \
    --num_rotation_bins ${NUM_ROTATION_BINS} \
    --dropout "${DROPOUT}" \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --save_epoch_interval ${SAVE_EPOCH_INTERVAL} \
    --logging_steps ${LOGGING_STEPS} \
    --swanlab_project ${SWANLAB_PROJECT} \
    --swanlab_experiment ${SWANLAB_EXPERIMENT} \
    --model_base_path "${MODEL_BASE_PATH}" \
    --heatmap_latent_scale "${HEATMAP_LATENT_SCALE}" \
    --latent_noise_std "${LATENT_NOISE_STD}" \
    --num_augmentations ${NUM_AUGMENTATIONS} \
    $(if [ "${USE_ACCURATE_PEAK_DETECTION}" = "true" ]; then echo "--use_accurate_peak_detection"; fi) \
    --local_feature_size ${LOCAL_FEATURE_SIZE} \
    $(if [ "${USE_ONLINE_ENCODING}" = "true" ]; then echo "--use_online_encoding"; fi) \
    $(if [ "${USE_MERGED_POINTCLOUD}" = "true" ]; then echo "--use_merged_pointcloud"; fi) \
    $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi) \
    $(if [ "${USE_INITIAL_GRIPPER_STATE}" = "true" ]; then echo "--use_initial_gripper_state"; fi) \
    --num_history_frames ${NUM_HISTORY_FRAMES}

echo "================================================================"
echo "Training completed on machine ${MACHINE_RANK}!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"
