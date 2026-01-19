#!/bin/bash

# Heatmap Sequence Training Script for Wan2.1-I2V-14B-480P
# 基于Wan2.2-I2V-A14B_heatmap.sh，适配Wan2.1-I2V-14B-480P模型进行热力图序列生成训练
# 支持多台机器运行，通过COPPELIASIM_ROOT路径检测自动选择相应配置

# ==============================================
# TODO: 机器2配置需要填写的项目
# ==============================================
# 在使用当前机器之前，请填写以下配置项：
# 1. MACHINE2_COPPELIASIM_ROOT: CoppeliaSim安装路径（主要机器识别依据）
# 2. MACHINE2_DISPLAY: DISPLAY环境变量
# 3. MACHINE2_CONDA_PATH: conda安装路径
# 4. MACHINE2_CONDA_ENV: conda环境名称
# 5. MACHINE2_PROJECT_ROOT: 项目根目录路径
# 6. MACHINE2_DEEPSPEED_CONFIG_DIR: DeepSpeed配置目录
# 7. MACHINE2_HEATMAP_DATA_ROOT: 热力图数据根目录
# 8. MACHINE2_OUTPUT_BASE: 输出基础目录
# 9. MACHINE2_MODEL_BASE_PATH: 模型基础路径
# ==============================================

# ==============================================
# 机器检测和环境配置
# ==============================================

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
else
    echo "错误：未找到COPPELIASIM_ROOT路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  机器2: ${MACHINE2_COPPELIASIM_ROOT}"
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

# ==============================================
# 项目路径配置
# ==============================================

# 机器1的项目路径配置
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE1_DEEPSPEED_CONFIG_DIR="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# 机器2的项目路径配置（待填写）
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"         # TODO: 填写当前机器的项目根目录
MACHINE2_DEEPSPEED_CONFIG_DIR="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config" # TODO: 填写当前机器的DeepSpeed配置目录

# 根据机器类型设置项目路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PROJECT_ROOT="${MACHINE1_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE1_DEEPSPEED_CONFIG_DIR}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PROJECT_ROOT="${MACHINE2_PROJECT_ROOT}"
    DEEPSPEED_CONFIG_DIR="${MACHINE2_DEEPSPEED_CONFIG_DIR}"
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
# export CUDA_VISIBLE_DEVICES=0; NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1; NUM_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; NUM_GPUS=7
# export CUDA_VISIBLE_DEVICES=0,1,2,3; NUM_GPUS=4
# 四GPU: 
export CUDA_VISIBLE_DEVICES=1,2,6,7; NUM_GPUS=4

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_HEATMAP_DATA_ROOT="/DATA/disk1/lpy/Franka_data/put_the_lion_on_the_top_shelf"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器2的路径配置（待填写）
MACHINE2_HEATMAP_DATA_ROOT="/data/Franka_data/put_the_lion_on_the_top_shelf"    # TODO: 填写当前机器的热力图数据根目录
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"          # TODO: 填写当前机器的输出基础目录
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"      # TODO: 填写当前机器的模型基础路径

# 根据机器类型设置路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    HEATMAP_DATA_ROOT="${MACHINE1_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    HEATMAP_DATA_ROOT="${MACHINE2_HEATMAP_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
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
TRAIL_END="16"            # 结束trail编号，如50表示到trail_50结束。留空表示不限制
# 示例：只使用trail_1到trail_50
# TRAIL_START=1
# TRAIL_END=50

# 生成时间戳目录名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PATH="${OUTPUT_BASE}/Wan2.2-TI2V-5B_heatmap_rgb_lora/10_trajectory/${TIMESTAMP}"
# 指定Wan模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV"

# 热力图专用参数
SEQUENCE_LENGTH=12        # 热力图序列长度 # 热力图序列加上原始图像的得到的总帧数，必须除以4余一
STEP_INTERVAL=1           # 轨迹步长间隔
MIN_TRAIL_LENGTH=10       # 最小轨迹长度
HEATMAP_SIGMA=1.5         # 高斯热力图标准差
COLORMAP_NAME="jet"   # colormap名称（统一使用cv2 JET）

# 图像和训练参数 - 256x256分辨率以节省显存
HEIGHT=384
WIDTH=384
NUM_FRAMES=${SEQUENCE_LENGTH}
# 优化方案：减少DATASET_REPEAT，增加EPOCHS获得更精细的控制
DATASET_REPEAT=1                     # 不重复数据集
LEARNING_RATE=1e-4
NUM_EPOCHS=20                       # 最大训练epoch数

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

# 数据增强参数
# SCENE_BOUNDS="0,-0.7,-0.05,0.8,0.7,0.65"
# SCENE_BOUNDS="0,-0.55,-0.05,0.8,0.45,0.6"
SCENE_BOUNDS="0,-0.65,-0.05,0.8,0.55,0.75"
TRANSFORM_AUG_XYZ="0.1,0.1,0.1"
TRANSFORM_AUG_RPY="10.0,10.0,20.0"  

# SwanLab配置参数
ENABLE_SWANLAB=true                         # 是否启用SwanLab记录
SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"    # SwanLab API密钥
SWANLAB_PROJECT="Wan2.2-TI2V-5B_heatmap_rgb_lora_A100_2"  # SwanLab项目名称
SWANLAB_EXPERIMENT="heatmap-lora-TI2V-5B_heatmap_rgb-$(date +%Y%m%d-%H%M%S)"  # SwanLab实验名称（添加时间戳）
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
echo "训练参数:"
echo "  序列长度: ${SEQUENCE_LENGTH}"
echo "  图像尺寸: ${HEIGHT}x${WIDTH}"
echo "  学习率: ${LEARNING_RATE}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  LoRA目标模块: ${LORA_TARGET_MODULES}"
echo "  双Head模式: ${USE_DUAL_HEAD}"
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
  --scene_bounds "${SCENE_BOUNDS}" \
  --transform_augmentation_xyz "${TRANSFORM_AUG_XYZ}" \
  --transform_augmentation_rpy "${TRANSFORM_AUG_RPY}" \
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
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_rank ${LORA_RANK} \
  --extra_inputs "input_image,input_image_rgb,input_video_rgb" \
  --train_batch_size ${TRAIN_BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} \
  $(if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then echo "--use_gradient_checkpointing"; fi) \
  $(if [ "${USE_DUAL_HEAD}" = "true" ]; then echo "--use_dual_head"; fi) \
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