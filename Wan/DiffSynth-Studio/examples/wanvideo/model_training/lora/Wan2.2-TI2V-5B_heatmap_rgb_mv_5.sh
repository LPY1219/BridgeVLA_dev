#!/bin/bash

# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置（待填写）
MACHINE2_COPPELIASIM_ROOT="/home/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"  # TODO: 填写当前机器的CoppeliaSim路径
MACHINE2_DISPLAY=":1.0"           # TODO: 填写当前机器的DISPLAY配置
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"  # TODO: 填写当前机器的conda路径
MACHINE2_CONDA_ENV="BridgeVLA_DM"   # TODO: 填写当前机器的conda环境名

# 机器3的CoppeliaSim配置
MACHINE3_COPPELIASIM_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE3_DISPLAY=":1.0"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"  # 和machine1一致
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# 机器4的CoppeliaSim配置
MACHINE4_COPPELIASIM_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE4_DISPLAY=":1.0"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"  # 和machine1一致
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

# 内存优化环境变量 - 激进的40GB显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,roundup_power2_divisions:32,garbage_collection_threshold:0.6,expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2  # 进一步限制CPU线程以节省内存
export CUDA_LAUNCH_BLOCKING=1  # 启用同步启动以获得更好的错误追踪
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # 暂时注释掉，可能影响DeepSpeed
export NCCL_DEBUG=WARN  # 减少NCCL调试输出
export NCCL_TIMEOUT=3600  # 增加NCCL超时时间到1小时
export NCCL_IB_DISABLE=1  # 禁用InfiniBand，使用以太网
export NCCL_P2P_DISABLE=1  # 禁用P2P通信，避免某些通信问题
export NCCL_TREE_THRESHOLD=0  # 强制使用树算法
export NCCL_ALGO=Tree  # 使用更稳定的树算法
# export NCCL_SOCKET_IFNAME=eth0  # 注释掉，让NCCL自动检测网络接口

# CUDA驱动优化设置（解决CUDA driver error问题）
export CUDA_MODULE_LOADING=LAZY  # 延迟加载CUDA模块
export CUDA_VISIBLE_DEVICES_ORDER=PCI_BUS_ID  # 使用PCI总线ID顺序
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 确保设备顺序一致
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  # 设置MPS管道目录
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log   # 设置MPS日志目录

# 额外的稳定性设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8  # 更保守的内存分配
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理
export TORCH_NCCL_BLOCKING_WAIT=1  # 启用阻塞等待以提高稳定性
export NCCL_ASYNC_ERROR_HANDLING=1  # NCCL异步错误处理

# 启用详细的错误追踪
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 显示详细的分布式训练错误信息
export TORCH_SHOW_CPP_STACKTRACES=1    # 显示C++堆栈跟踪
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json  # 保存详细错误到文件

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

# 多进程启动方法设置（解决CUDA多进程问题）
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# 设置工作目录
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"

# ==============================================
# GPU配置 - 支持单机多卡训练
# ==============================================
# 8张A100训练配置 - 最大化分散模型参数
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NUM_GPUS=8

# 其他常用配置示例（备用）：
export CUDA_VISIBLE_DEVICES=7; NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1; NUM_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; NUM_GPUS=7
# 测试用4个GPU（推荐先用这个排查问题）
# export CUDA_VISIBLE_DEVICES=1,2,3,4; NUM_GPUS=4
# 如果4个GPU正常，再改回7个GPU：
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; NUM_GPUS=8

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_HEATMAP_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/pour_filter"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器2的路径配置（待填写）
MACHINE2_HEATMAP_DATA_ROOT=""    # TODO: 填写当前机器的热力图数据根目录
# MACHINE2_HEATMAP_DATA_ROOT="/data/Franka_data_3zed_2/put_red_bull_in_pink_plate"    # TODO: 填写当前机器的热力图数据根目录
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"          # TODO: 填写当前机器的输出基础目录
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"      # TODO: 填写当前机器的模型基础路径

# 机器3的路径配置
MACHINE3_HEATMAP_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/pour_filter"  # TODO: 根据实际任务修改
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器4的路径配置
MACHINE4_HEATMAP_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/pour_filter"  # TODO: 根据实际任务修改
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused"

# 机器5的路径配置
MACHINE5_HEATMAP_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/pour_filter"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B-fused"

# 机器6的路径配置
MACHINE6_HEATMAP_DATA_ROOT="/mnt/robot-rfm/user/lpy/data/Franka_data_3zed_5/pour_filter"
MACHINE6_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE6_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B-fused"

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

# 单任务训练示例（保留原有配置）
# HEATMAP_DATA_ROOT已在上面设置

# 多任务训练示例（用户可以修改为多个任务路径，用空格分隔）
# HEATMAP_DATA_ROOT="/data/Franka_data/task1 /data/Franka_data/task2 /data/Franka_data/task3 /data/Franka_data/task4"

# Trail范围过滤（可选，默认使用所有trails）
TRAIL_START="1"          # 起始trail编号，如1表示从trail_1开始。留空表示不限制
TRAIL_END="5"            # 结束trail编号，如50表示到trail_50结束。留空表示不限制
# 示例：只使用trail_1到trail_50
# TRAIL_START=1
# TRAIL_END=50


# 指定Wan模型类型
# 可选值:
#   - 5B_TI2V_RGB_HEATMAP_MV: 标准多视角模式（NUM_HISTORY_FRAMES=1时使用）
#   - 5B_TI2V_RGB_HEATMAP_MV_HISTORY: 多帧历史模式（NUM_HISTORY_FRAMES>1时使用）
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV"
# 热力图专用参数
SEQUENCE_LENGTH=24       # 未来帧序列长度 # 热力图序列加上原始图像的得到的总帧数，必须除以4余一
STEP_INTERVAL=1           # 轨迹步长间隔
MIN_TRAIL_LENGTH=10       # 最小轨迹长度
HEATMAP_SIGMA=1.5         # 高斯热力图标准差
COLORMAP_NAME="jet"   # colormap名称（统一使用cv2 JET）

# 历史帧配置 - 控制输入条件帧数量
# 历史帧按照VAE原始方式编码：第一帧单独编码，后续每4帧一组
#
# 允许的历史帧数量：
#   - 1: 单帧条件（向后兼容）→ 1个条件latent
#   - 2: 两帧分别单独编码 → 2个条件latent
#   - 1+4N (5,9,13...): 第一帧单独 + 后续每4帧一组 → (1+N)个条件latent
#
# 示例：
#   NUM_HISTORY_FRAMES=1  → num_condition_latents=1
#   NUM_HISTORY_FRAMES=2  → num_condition_latents=2  (每帧单独编码)
#   NUM_HISTORY_FRAMES=5  → num_condition_latents=2  (1帧 + 4帧)
#   NUM_HISTORY_FRAMES=9  → num_condition_latents=3  (1帧 + 4帧 + 4帧)
#
# ⚠️ 重要：当 NUM_HISTORY_FRAMES>1 时，必须将 WAN_TYPE 设置为 5B_TI2V_RGB_HEATMAP_MV_HISTORY
NUM_HISTORY_FRAMES=1     # 历史帧数量，必须为 1, 2, 或 1+4N (5,9,13...)

# 验证 NUM_HISTORY_FRAMES 的合法性
# 必须是 1, 2, 或 1+4N (即 (n-1) % 4 == 0 当 n > 2)
is_valid_history_frames=false
if [ ${NUM_HISTORY_FRAMES} -eq 1 ]; then
    is_valid_history_frames=true
elif [ ${NUM_HISTORY_FRAMES} -eq 2 ]; then
    is_valid_history_frames=true
elif [ ${NUM_HISTORY_FRAMES} -gt 2 ]; then
    # 检查是否为 1+4N 形式
    remainder=$(( (${NUM_HISTORY_FRAMES} - 1) % 4 ))
    if [ ${remainder} -eq 0 ]; then
        is_valid_history_frames=true
    fi
fi

if [ "${is_valid_history_frames}" != "true" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} is invalid!"
    echo "       Allowed values: 1, 2, or 1+4N (5, 9, 13, 17, ...)"
    echo "       This ensures proper VAE encoding: first frame alone, then groups of 4"
    exit 1
fi

# 验证 WAN_TYPE 和 NUM_HISTORY_FRAMES 的一致性（双向检测）
# 规则1: NUM_HISTORY_FRAMES > 1 必须使用 5B_TI2V_RGB_HEATMAP_MV_HISTORY
if [ ${NUM_HISTORY_FRAMES} -gt 1 ] && [ "${WAN_TYPE}" != "5B_TI2V_RGB_HEATMAP_MV_HISTORY" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} > 1, but WAN_TYPE=${WAN_TYPE}"
    echo "       When using multi-frame history (NUM_HISTORY_FRAMES > 1), you MUST set:"
    echo "       WAN_TYPE=\"5B_TI2V_RGB_HEATMAP_MV_HISTORY\""
    exit 1
fi
# 规则2: 使用 5B_TI2V_RGB_HEATMAP_MV_HISTORY 必须设置 NUM_HISTORY_FRAMES > 1
if [ ${NUM_HISTORY_FRAMES} -eq 1 ] && [ "${WAN_TYPE}" == "5B_TI2V_RGB_HEATMAP_MV_HISTORY" ]; then
    echo "ERROR: WAN_TYPE=5B_TI2V_RGB_HEATMAP_MV_HISTORY, but NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES}"
    echo "       When using 5B_TI2V_RGB_HEATMAP_MV_HISTORY, you MUST set NUM_HISTORY_FRAMES > 1"
    echo "       If you want single-frame mode, use WAN_TYPE=\"5B_TI2V_RGB_HEATMAP_MV\""
    exit 1
fi

# 图像和训练参数 - 256x256分辨率以节省显存
HEIGHT=256
WIDTH=256
NUM_FRAMES=${SEQUENCE_LENGTH}
# 优化方案：减少DATASET_REPEAT，增加EPOCHS获得更精细的控制
DATASET_REPEAT=1                     # 不重复数据集
LEARNING_RATE=1e-4
NUM_EPOCHS=100                       # 最大训练epoch数
SAVE_EPOCHS_INTERVAL=10               # 每隔多少个epoch保存一次模型（0表示只在最后保存）

# 多GPU训练参数调整 - 2张GPU配置
TRAIN_BATCH_SIZE_PER_GPU=1            # 每张GPU的批次大小保持为1，Wan只支持这样
GRADIENT_ACCUMULATION_STEPS=4           # 增加梯度累积步数以保持相同的有效批次大小
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))
echo "Total effective batch size: ${EFFECTIVE_BATCH_SIZE}"

# 显存优化参数 - 目标40GB以下
DATASET_NUM_WORKERS=0                   # 数据加载线程数（设为0避免CUDA多进程问题）
USE_GRADIENT_CHECKPOINTING=false        # 启用梯度检查点（节省显存）
# 显存优化已移除，由DeepSpeed处理
MIXED_PRECISION="bf16"                  # 使用bf16混合精度
DATALOADER_PIN_MEMORY=false             # 关闭pin memory以节省显存
PREFETCH_FACTOR=2                       # 减少数据预取因子以节省显存
# LoRA参数
LORA_RANK=32
# 移除patch_embedding和head.head，改为全量训练
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# Dual Head模式 - 是否使用双head（RGB和Heatmap各自独立的head）
USE_DUAL_HEAD=true  # 设置为true启用双head模式，false使用单head模式

# ==============================================
# 损失权重配置 - RGB Loss Weight
# ==============================================
# 控制RGB loss和Heatmap loss的权重分配
# 总loss = RGB_LOSS_WEIGHT * loss_rgb + (1 - RGB_LOSS_WEIGHT) * loss_heatmap
# 取值范围: 0.0 ~ 1.0
#   - 1.0: 只使用RGB loss
#   - 0.5: RGB和Heatmap loss各占一半（默认）
#   - 0.0: 只使用Heatmap loss
RGB_LOSS_WEIGHT=0.08

# ==============================================
# 高级训练参数 - Modulation 和 Norm 解冻控制
# ==============================================
# ⚠️  IMPORTANT: 控制是否解冻 modulation 和 norm 参数以获得更好的适应性
#
# 什么时候设置为 true：
# - 训练全新模型时（推荐）
# - 想要最大化任务适应性时
# - 不在意与旧 checkpoint 的兼容性时
#
# 什么时候设置为 false（默认）：
# - 测试已有的旧 checkpoint 时（向后兼容）
# - 从旧的预训练模型继续训练时（保持一致性）
#
# 影响的参数：
# - modulation: AdaLN 调制参数 (~0.55M，影响大）
# - mvs_attn.norm_q/norm_k: 多视角 RMSNorm (~0.18M)
#
UNFREEZE_MODULATION_AND_NORMS=true  # 设置为 true 训练新模型，false 保持向后兼容

# ==============================================
# 预训练模型加载配置 (用于从预训练checkpoint继续finetune)
# ==============================================
# 是否加载预训练的checkpoint进行finetune
LOAD_PRETRAINED_CHECKPOINT=False # 设置为true启用预训练权重加载
# 生成时间戳目录名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NAME="test_h20e_${LOAD_PRETRAINED_CHECKPOINT}_history_${NUM_HISTORY_FRAMES}_seq_${SEQUENCE_LENGTH}_new_projection"
OUTPUT_PATH="${OUTPUT_BASE}/Wan2.2-TI2V-5B_heatmap_rgb_lora/5_trajectory_pour_3camera_${NAME}_rgb_loss_${RGB_LOSS_WEIGHT}/${TIMESTAMP}"

# 预训练checkpoint路径配置（各机器的路径）
# 机器1的预训练checkpoint路径
MACHINE1_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"
# 机器2的预训练checkpoint路径
MACHINE2_PRETRAINED_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true/20251208_045615/epoch-15.safetensors"
# 机器3的预训练checkpoint路径
ACHINE3_PRETRAINED_CHECKPOINT="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_cook_5_3camera_pour_filter_pretrain_False_history_1_seq_48_new_projection_rgb_loss_1/20260105_203050/epoch-99.safetensors"
# 机器4的预训练checkpoint路径
MACHINE4_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"

# 机器5的预训练checkpoint路径
MACHINE5_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true_epoch-15.safetensors"

# 机器6的预训练checkpoint路径
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
# true: 使用三个相机拼接的点云
# false: 只使用相机1的点云
USE_MERGED_POINTCLOUD=false

# 投影模式配置
# true: 使用不同的投影方式（base_multi_view_dataset_with_rot_grip_3cam_different_projection.py）
# false: 使用默认投影方式（base_multi_view_dataset_with_rot_grip_3cam.py）
USE_DIFFERENT_PROJECTION=true

# 数据增强参数
# SCENE_BOUNDS="0,-0.7,-0.05,0.8,0.7,0.65"
# SCENE_BOUNDS="0,-0.55,-0.05,0.8,0.45,0.6"
# SCENE_BOUNDS="0,-0.65,-0.05,0.8,0.55,0.75"
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.1,0.1,0.1"
TRANSFORM_AUG_RPY="5.0,5.0,5.0"  

# SwanLab配置参数
ENABLE_SWANLAB=true                         # 是否启用SwanLab记录
SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"    # SwanLab API密钥
SWANLAB_PROJECT="Wan2.2-TI2V-5B_heatmap_rgb_lora_pour"  # SwanLab项目名称
SWANLAB_EXPERIMENT="${NAME}-$(date +%Y%m%d-%H%M%S)"  # SwanLab实验名称（添加时间戳）
DEBUG_MODE=false                           # 调试模式（为true时禁用SwanLab）

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
echo "  GPU数量: ${NUM_GPUS}"
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

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"
echo "输出目录已创建: ${OUTPUT_PATH}"

# ================================================================
# 单阶段训练
# ================================================================
echo "================================================================"
echo "STARTING HEATMAP TRAINING"
echo "================================================================"

# 标准多GPU训练
echo "🔧 Using standard multi-GPU training"

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --mixed_precision=${MIXED_PRECISION} \
  --main_process_port=29500 \
  examples/wanvideo/model_training/heatmap_train_mv.py \
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
  --find_unused_parameters \
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

#   --model_paths '[
#     [
#         "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00001-of-00003.safetensors",
#         "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00002-of-00003.safetensors",
#         "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00003-of-00003.safetensors"
#     ],
#     "'${MODEL_BASE_PATH}'/models_t5_umt5-xxl-enc-bf16.pth",
#     "'${MODEL_BASE_PATH}'/Wan2.2_VAE.pth"
#     ]' \


#   --model_paths '[
#     [
#         "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
#         "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
#         "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00003-of-00003-bf16.safetensors"
#     ],
#     "'${MODEL_BASE_PATH}'/models_t5_umt5-xxl-enc-bf16.pth",
#     "'${MODEL_BASE_PATH}'/Wan2.2_VAE.pth"
#     ]' \

echo "================================================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"