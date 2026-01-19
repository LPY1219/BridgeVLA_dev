#!/bin/bash

# Heatmap Sequence Training Script for Wan2.1-I2V-14B-480P
# 基于Wan2.2-I2V-A14B_heatmap.sh，适配Wan2.1-I2V-14B-480P模型进行热力图序列生成训练
# 支持多台机器运行，通过COPPELIASIM_ROOT路径检测自动选择相应配置

# ==============================================
# TODO: 机器2/3/4配置需要填写的项目
# ==============================================
# 在使用当前机器之前，请填写以下配置项：
# 机器2:
# 1. MACHINE2_COPPELIASIM_ROOT: CoppeliaSim安装路径（主要机器识别依据）
# 2. MACHINE2_DISPLAY: DISPLAY环境变量
# 3. MACHINE2_CONDA_PATH: conda安装路径
# 4. MACHINE2_CONDA_ENV: conda环境名称
# 5. MACHINE2_PROJECT_ROOT: 项目根目录路径
# 6. MACHINE2_DEEPSPEED_CONFIG_DIR: DeepSpeed配置目录
# 7. MACHINE2_HEATMAP_DATA_ROOT: 热力图数据根目录
# 8. MACHINE2_OUTPUT_BASE: 输出基础目录
# 9. MACHINE2_MODEL_BASE_PATH: 模型基础路径
#
# 机器3 和 机器4 的配置已经按照 disk0 和 disk2 的路径模式填写，
# 请根据实际情况检查和修改数据路径（HEATMAP_DATA_ROOT）
# ==============================================

# ==============================================
# 机器检测和环境配置
# ==============================================

# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/mnt/data/cyx/miniconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置（待填写）
MACHINE2_COPPELIASIM_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"  # TODO: 填写当前机器的CoppeliaSim路径
MACHINE2_DISPLAY=":1.0"           # TODO: 填写当前机器的DISPLAY配置
MACHINE2_CONDA_PATH="/root/miniconda3/etc/profile.d/conda.sh"  # TODO: 填写当前机器的conda路径
MACHINE2_CONDA_ENV="metaworld"   # TODO: 填写当前机器的conda环境名

# 机器3的CoppeliaSim配置
MACHINE3_COPPELIASIM_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE3_DISPLAY=":1.0"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"  # 和machine1一致
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# 机器4的CoppeliaSim配置
MACHINE4_COPPELIASIM_ROOT="/DATA/disk2/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE4_DISPLAY=":1.0"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"  # 和machine1一致
MACHINE4_CONDA_ENV="BridgeVLA_DM"

# 辅助函数：从conda路径推断conda根目录并直接设置环境变量（适用于tmux等场景）
setup_conda_env_directly() {
    local conda_path="$1"
    local env_name="$2"
    
    # 从conda.sh路径推断conda根目录（去掉 /etc/profile.d/conda.sh）
    local conda_base=""
    if [[ "${conda_path}" == *"/etc/profile.d/conda.sh" ]]; then
        conda_base="$(dirname "$(dirname "$(dirname "${conda_path}")")")"
    fi
    
    # 构建conda环境路径
    local env_path="${conda_base}/envs/${env_name}"
    
    # 检查环境是否存在
    if [ ! -d "${env_path}" ]; then
        echo "警告: conda环境路径不存在: ${env_path}"
        echo "尝试从conda路径激活..."
        # 如果直接路径不存在，尝试使用conda命令
        if [ -f "${conda_path}" ]; then
            source "${conda_path}" 2>/dev/null || true
            eval "$(conda shell.bash hook 2>/dev/null)" || true
            conda activate "${env_name}" 2>/dev/null || true
        fi
    else
        # 直接设置环境变量，不依赖conda activate
        export CONDA_PREFIX="${env_path}"
        export CONDA_DEFAULT_ENV="${env_name}"
        export PATH="${env_path}/bin:${PATH}"
        export CONDA_PROMPT_MODIFIER="(${env_name}) "
        
        # 设置其他必要的conda相关路径
        if [ -d "${conda_base}/bin" ]; then
            export PATH="${conda_base}/bin:${PATH}"
        fi
        
        echo "✓ 直接设置conda环境: ${env_name}"
        echo "  CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

# 通过COPPELIASIM_ROOT检测机器并设置环境
if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器1（基于COPPELIASIM_ROOT），使用配置1"
    CURRENT_MACHINE="machine1"
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE1_DISPLAY}"
    setup_conda_env_directly "${MACHINE1_CONDA_PATH}" "${MACHINE1_CONDA_ENV}"
    conda activate "${MACHINE1_CONDA_ENV}"
    echo "已设置机器1的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE2_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器2（基于COPPELIASIM_ROOT），使用配置2"
    CURRENT_MACHINE="machine2"
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE2_DISPLAY}"
    setup_conda_env_directly "${MACHINE2_CONDA_PATH}" "${MACHINE2_CONDA_ENV}"
    conda activate "${MACHINE2_CONDA_ENV}"
    echo "已设置机器2的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE3_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器3（基于COPPELIASIM_ROOT），使用配置3"
    CURRENT_MACHINE="machine3"
    export COPPELIASIM_ROOT="${MACHINE3_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE3_DISPLAY}"
    setup_conda_env_directly "${MACHINE3_CONDA_PATH}" "${MACHINE3_CONDA_ENV}"
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
    setup_conda_env_directly "${MACHINE4_CONDA_PATH}" "${MACHINE4_CONDA_ENV}"
    conda activate "${MACHINE4_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
    echo "已设置机器4的CoppeliaSim环境变量和conda环境"
else
    echo "错误：未找到COPPELIASIM_ROOT路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  机器2: ${MACHINE2_COPPELIASIM_ROOT}"
    echo "  机器3: ${MACHINE3_COPPELIASIM_ROOT}"
    echo "  机器4: ${MACHINE4_COPPELIASIM_ROOT}"
    exit 1
fi

# 确保PATH正确设置（双重保险）
if [ -n "${CONDA_PREFIX}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    # 设置Python相关路径
    if [ -d "${CONDA_PREFIX}/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    fi
    if [ -d "${CONDA_PREFIX}/lib/python3.9/site-packages" ]; then
        export PYTHONPATH="${CONDA_PREFIX}/lib/python3.9/site-packages:${PYTHONPATH}"
    fi
    echo "✓ Conda环境路径已设置: ${CONDA_PREFIX}"
    echo "✓ Python路径: $(which python 2>/dev/null || echo '未找到')"
else
    echo "警告: CONDA_PREFIX未设置，conda环境可能未正确激活"
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
pip install --force-reinstall --no-deps swanlab

# 机器1的项目路径配置
MACHINE1_PROJECT_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE1_DEEPSPEED_CONFIG_DIR="/mnt/data/cyx/workspace/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器2的项目路径配置（待填写）
MACHINE2_PROJECT_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/Wan/DiffSynth-Studio"         # TODO: 填写当前机器的项目根目录
MACHINE2_DEEPSPEED_CONFIG_DIR="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config" # TODO: 填写当前机器的DeepSpeed配置目录

# 机器3的项目路径配置
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE3_DEEPSPEED_CONFIG_DIR="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器4的项目路径配置
MACHINE4_PROJECT_ROOT="/DATA/disk2/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE4_DEEPSPEED_CONFIG_DIR="/DATA/disk2/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

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
# export CUDA_VISIBLE_DEVICES=2; NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1; NUM_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; NUM_GPUS=7
# 测试用4个GPU（推荐先用这个排查问题）
# export CUDA_VISIBLE_DEVICES=0,1,2,3; NUM_GPUS=4
# 如果4个GPU正常，再改回7个GPU：
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6; NUM_GPUS=7
NUM_GPUS=64
# 检查环境变量，如果存在则自动配置多机模式
MULTI_NODE_AUTO_CONFIG=false
if [ -n "${WORLD_SIZE}" ] && [ -n "${RANK}" ] && [ -n "${MASTER_ADDR}" ] && [ -n "${MASTER_PORT}" ]; then
    # 从环境变量自动获取多机配置
    NUM_MACHINES=${WORLD_SIZE}
    MACHINE_RANK=${RANK}
    MAIN_PROCESS_IP="${MASTER_ADDR}"
    MAIN_PROCESS_PORT=${MASTER_PORT}
    MULTI_NODE_AUTO_CONFIG=true
else
    # 单机训练配置（默认）
    NUM_MACHINES=1
    MACHINE_RANK=0
    MAIN_PROCESS_IP=""
    MAIN_PROCESS_PORT=29500
    
    # 如果需要手动配置多机训练，取消注释并修改以下参数
    # NUM_MACHINES=2                    # 总机器数
    # MACHINE_RANK=0                    # 当前机器rank（主节点为0，其他节点为1,2,3...）
    # MAIN_PROCESS_IP="192.168.1.100"  # 主节点IP地址（所有节点都需要能访问）
    # MAIN_PROCESS_PORT=29500           # 通信端口（确保防火墙允许）
fi

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_HEATMAP_DATA_ROOT_LIST=(
"/mnt/workspace/cyx/datasets/RLBench/data/train/reach_and_drag"
)
MACHINE1_HEATMAP_DATA_ROOT="${MACHINE1_HEATMAP_DATA_ROOT_LIST[*]}"
MACHINE1_OUTPUT_BASE="/mnt/data/cyx/workspace/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/mnt/data/cyx/huggingface/Wan2.2-TI2V-5B-fused"


# 机器2的路径配置（待填写）
MACHINE2_HEATMAP_DATA_ROOT_LIST=(
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/reach_and_drag"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/close_jar"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/insert_onto_square_peg"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/light_bulb_in"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/meat_off_grill"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/open_drawer"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/place_cups"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/place_shape_in_shape_sorter"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/place_wine_at_rack_location"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/push_buttons"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/put_groceries_in_cupboard"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/put_item_in_drawer"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/put_money_in_safe"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/slide_block_to_color_target"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/stack_blocks"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/stack_cups"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/sweep_to_dustpan_of_size"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/turn_tap"
"/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev_backup/RLBench_Data/phone_on_base"
)
MACHINE2_HEATMAP_DATA_ROOT="${MACHINE2_HEATMAP_DATA_ROOT_LIST[*]}"
MACHINE2_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/logs/Wan/train"
MACHINE2_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B"


# 机器3的路径配置
MACHINE3_HEATMAP_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_2/put_red_bull_in_pink_plate"  # TODO: 根据实际任务修改
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器4的路径配置
MACHINE4_HEATMAP_DATA_ROOT="/DATA/disk2/lpy/data/Franka_data_3zed_2/put_red_bull_in_pink_plate"  # TODO: 根据实际任务修改
MACHINE4_OUTPUT_BASE="/DATA/disk2/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE4_MODEL_BASE_PATH="/DATA/disk2/lpy/huggingface/Wan2.2-TI2V-5B-fused"

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
fi

# ==============================================
# 多任务训练和Trail范围过滤配置
# ==============================================

# 单任务训练示例（保留原有配置）
# HEATMAP_DATA_ROOT已在上面设置

# 多任务训练示例（用户可以修改为多个任务路径，用空格分隔）
# HEATMAP_DATA_ROOT="/data/Franka_data/task1 /data/Franka_data/task2 /data/Franka_data/task3 /data/Franka_data/task4"

# Trail范围过1滤（可选，默认使用所有trails）
TRAIL_START=""          # 起始trail编号，如1表示从trail_1开始。留空表示不限制
TRAIL_END="99"            # 结束trail编号，如50表示到trail_50结束。留空表示不限制
# 示例：只使用trail_1到trail_50
# TRAIL_START=1
# TRAIL_END=50

# 生成时间戳目录名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PATH="${OUTPUT_BASE}/Wan2.2-TI2V-5B_heatmap_rgb_lora/rlbench_phone/${TIMESTAMP}"
# 指定Wan模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV"

# 热力图专用参数
SEQUENCE_LENGTH=24        # 热力图序列长度 # 热力图序列加上原始图像的得到的总帧数，必须除以4余一
STEP_INTERVAL=1           # 轨迹步长间隔
MIN_TRAIL_LENGTH=10       # 最小轨迹长度
HEATMAP_SIGMA=1.5         # 高斯热力图标准差
COLORMAP_NAME="jet"   # colormap名称（统一使用cv2 JET）

# 图像和训练参数 - 256x256分辨率以节省显存
HEIGHT=256
WIDTH=256
NUM_FRAMES=${SEQUENCE_LENGTH}
# 优化方案：减少DATASET_REPEAT，增加EPOCHS获得更精细的控制
DATASET_REPEAT=1                     # 不重复数据集
LEARNING_RATE=1e-4
NUM_EPOCHS=100                       # 最大训练epoch数\
SAVE_EPOCHS_INTERVAL=10               # 每隔多少个epoch保存一次模型（0表示只在最后保存）

# 多GPU训练参数调整 - 2张GPU配置
TRAIN_BATCH_SIZE_PER_GPU=1            # 每张GPU的批次大小保持为1，Wan只支持这样
GRADIENT_ACCUMULATION_STEPS=1           # 增加梯度累积步数以保持相同的有效批次大小
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

UNFREEZE_MODULATION_AND_NORMS=true  # 设置为 true 训练新模型，false 保持向后兼容
# ==============================================
# 预训练模型加载配置 (用于从预训练checkpoint继续finetune)
# ==============================================
# 是否加载预训练的checkpoint进行finetune
LOAD_PRETRAINED_CHECKPOINT=false  # 设置为true启用预训练权重加载

# 预训练checkpoint路径配置（各机器的路径）
# 机器1的预训练checkpoint路径
MACHINE1_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train/pretrain_ckpt/pretrain_9_epoch.safetensors"
# 机器2的预训练checkpoint路径
MACHINE2_PRETRAINED_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain/20251206_075139/epoch-9.safetensors"
# 机器3的预训练checkpoint路径
MACHINE3_PRETRAINED_CHECKPOINT="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain/20251206_075139/epoch-9.safetensors"
# 机器4的预训练checkpoint路径
MACHINE4_PRETRAINED_CHECKPOINT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/pretrain/Wan2.2-TI2V-5B_mvtrack_pretrain/20251206_075139/epoch-9.safetensors"

# 根据机器类型设置预训练checkpoint路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE1_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE2_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE3_PRETRAINED_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    PRETRAINED_CHECKPOINT="${MACHINE4_PRETRAINED_CHECKPOINT}"
fi


# Dual Head模式 - 是否使用双head（RGB和Heatmap各自独立的head）
USE_DUAL_HEAD=true  # 设置为true启用双head模式，false使用单head模式

# 点云配置
# true: 使用三个相机拼接的点云
# false: 只使用相机1的点云
USE_MERGED_POINTCLOUD=true

# 投影模式配置
# true: 使用不同的投影方式（base_multi_view_dataset_with_rot_grip_3cam_different_projection.py）
# false: 使用默认投影方式（base_multi_view_dataset_with_rot_grip_3cam.py）
USE_DIFFERENT_PROJECTION=false

# 数据增强参数
# SCENE_BOUNDS="0,-0.7,-0.05,0.8,0.7,0.65"
# SCENE_BOUNDS="0,-0.55,-0.05,0.8,0.45,0.6"
# SCENE_BOUNDS="0,-0.65,-0.05,0.8,0.55,0.75"
SCENE_BOUNDS="-0.3,-0.5,0.6,0.7,0.5,1.6"
TRANSFORM_AUG_XYZ="0.1,0.1,0.1"
TRANSFORM_AUG_RPY="0.0,0.0,20.0"  

# SwanLab配置参数
ENABLE_SWANLAB=true                         # 是否启用SwanLab记录
SWANLAB_API_KEY="8iNh1p4ePH7cbYlN9rC0j"    # SwanLab API密钥
SWANLAB_PROJECT="Wan2.2-TI2V-5B_heatmap_rgb_lora_RLBench"  # SwanLab项目名称
SWANLAB_EXPERIMENT="heatmap-lora-TI2V-5B_heatmap_rgb-$(date +%Y%m%d-%H%M%S)"  # SwanLab实验名称（添加时间戳）
DEBUG_MODE=false                           # 调试模式（为true时禁用SwanLab）

pip install --force-reinstall --no-deps swanlab

echo "================================================================"
echo "HEATMAP SEQUENCE TRAINING FOR Wan2.2-TI2V-5B_heatmap_rgb_lora"
echo "================================================================"
echo "当前使用机器: ${CURRENT_MACHINE}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据根目录: ${HEATMAP_DATA_ROOT}"
echo "模型路径: ${MODEL_BASE_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
echo "训练参数:"
echo "  序列长度: ${SEQUENCE_LENGTH}"
echo "  图像尺寸: ${HEIGHT}x${WIDTH}"
echo "  学习率: ${LEARNING_RATE}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  LoRA目标模块: ${LORA_TARGET_MODULES}"
echo "  双Head模式: ${USE_DUAL_HEAD}"
echo "  点云融合模式: ${USE_MERGED_POINTCLOUD}"
echo "  不同投影模式: ${USE_DIFFERENT_PROJECTION}"
echo "  GPU数量: ${NUM_GPUS}"
echo "  每GPU批次大小: ${TRAIN_BATCH_SIZE_PER_GPU}"
echo "  有效批次大小: ${EFFECTIVE_BATCH_SIZE}"
echo "  数据加载线程数: ${DATASET_NUM_WORKERS}"
echo "  混合精度: ${MIXED_PRECISION}"
echo "  梯度检查点: ${USE_GRADIENT_CHECKPOINTING}"
echo "  SwanLab启用: ${ENABLE_SWANLAB}"
echo "  调试模式: ${DEBUG_MODE}"
echo "================================================================"

# ==============================================
# 路径验证
# ==============================================

# 检查数据目录
missing_paths=()
IFS=' ' read -ra __heatmap_paths <<< "${HEATMAP_DATA_ROOT}"
for p in "${__heatmap_paths[@]}"; do
    if [ ! -d "${p}" ]; then
        missing_paths+=("${p}")
    fi
done
if [ "${#missing_paths[@]}" -ne 0 ]; then
    echo "错误：以下数据目录不存在:"
    for mp in "${missing_paths[@]}"; do
        echo "  ${mp}"
    done
    echo "请检查数据路径并重试。"
    exit 1
fi


# 检查模型目录
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "错误：模型目录不存在: ${MODEL_BASE_PATH}"
    echo "请检查模型路径并重试。"
    exit 1
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

# 使用accelerate launch进行多卡训练
# 多机多卡训练时，需要添加额外的参数
if [ "${NUM_MACHINES}" -gt 1 ]; then
    echo "启动多机多卡训练..."
    echo "  机器总数: ${NUM_MACHINES}"
    echo "  当前机器rank: ${MACHINE_RANK}"
    echo "  主节点IP: ${MAIN_PROCESS_IP}"
    echo "  通信端口: ${MAIN_PROCESS_PORT}"
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --num_machines=${NUM_MACHINES} \
        --machine_rank=${MACHINE_RANK} \
        --main_process_ip="${MAIN_PROCESS_IP}" \
        --main_process_port=${MAIN_PROCESS_PORT} \
        --mixed_precision=${MIXED_PRECISION} \
        examples/wanvideo/model_training/heatmap_train_mv_view_rlbench.py \
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
        $(if [ "${UNFREEZE_MODULATION_AND_NORMS}" = "true" ]; then echo "--unfreeze_modulation_and_norms"; fi) \
        $(if [ "${USE_MERGED_POINTCLOUD}" = "true" ]; then echo "--use_merged_pointcloud"; fi) \
        $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi) \
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
else
    echo "启动单机多卡训练..."
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --num_machines=1 \
        --mixed_precision=${MIXED_PRECISION} \
        --main_process_port=29500 \
        examples/wanvideo/model_training/heatmap_train_mv_view_rlbench.py \
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
        $(if [ "${UNFREEZE_MODULATION_AND_NORMS}" = "true" ]; then echo "--unfreeze_modulation_and_norms"; fi) \
        $(if [ "${USE_MERGED_POINTCLOUD}" = "true" ]; then echo "--use_merged_pointcloud"; fi) \
        $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi) \
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
fi

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