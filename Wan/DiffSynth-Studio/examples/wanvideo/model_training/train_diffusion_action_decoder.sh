#!/bin/bash

# ==============================================
# Multi-Machine Environment Configuration
# ==============================================

# 机器1的conda配置
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的conda配置
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

# 机器3的conda配置
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# 机器4的conda配置
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"

# 机器5的conda配置
MACHINE5_CONDA_PATH="/DATA/disk1/yaoliang/miniconda3/etc/profile.d/conda.sh"
MACHINE5_CONDA_ENV="BridgeVLA_DM"

# ==============================================
# 项目路径配置（用于机器检测）
# ==============================================

# 机器1的项目路径
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器2的项目路径
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器3的项目路径
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器4的项目路径
MACHINE4_PROJECT_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器5的项目路径
MACHINE5_PROJECT_ROOT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 通过PROJECT_ROOT检测机器并设置环境
if [ -d "${MACHINE1_PROJECT_ROOT}" ]; then
    echo "检测到机器1（基于PROJECT_ROOT），使用配置1"
    CURRENT_MACHINE="machine1"
    PROJECT_ROOT="${MACHINE1_PROJECT_ROOT}"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    echo "已设置机器1的conda环境"
elif [ -d "${MACHINE2_PROJECT_ROOT}" ]; then
    echo "检测到机器2（基于PROJECT_ROOT），使用配置2"
    CURRENT_MACHINE="machine2"
    PROJECT_ROOT="${MACHINE2_PROJECT_ROOT}"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    echo "已设置机器2的conda环境"
elif [ -d "${MACHINE3_PROJECT_ROOT}" ]; then
    echo "检测到机器3（基于PROJECT_ROOT），使用配置3"
    CURRENT_MACHINE="machine3"
    PROJECT_ROOT="${MACHINE3_PROJECT_ROOT}"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "已设置机器3的conda环境"
elif [ -d "${MACHINE4_PROJECT_ROOT}" ]; then
    echo "检测到机器4（基于PROJECT_ROOT），使用配置4"
    CURRENT_MACHINE="machine4"
    PROJECT_ROOT="${MACHINE4_PROJECT_ROOT}"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "已设置机器4的conda环境"
elif [ -d "${MACHINE5_PROJECT_ROOT}" ]; then
    echo "检测到机器5（基于PROJECT_ROOT），使用配置5"
    CURRENT_MACHINE="machine5"
    PROJECT_ROOT="${MACHINE5_PROJECT_ROOT}"
    source "${MACHINE5_CONDA_PATH}"
    conda activate "${MACHINE5_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "已设置机器5的conda环境"
else
    echo "错误：未找到PROJECT_ROOT路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_PROJECT_ROOT}"
    echo "  机器2: ${MACHINE2_PROJECT_ROOT}"
    echo "  机器3: ${MACHINE3_PROJECT_ROOT}"
    echo "  机器4: ${MACHINE4_PROJECT_ROOT}"
    echo "  机器5: ${MACHINE5_PROJECT_ROOT}"
    exit 1
fi

# ==============================================
# 环境变量配置
# ==============================================

# 内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8,expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree

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

# 设置PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# 设置工作目录
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"

# ==============================================
# GPU配置 - 支持单机多卡训练
# ==============================================
# 8张GPU训练配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# 其他常用配置示例（备用）：
# export CUDA_VISIBLE_DEVICES=7; NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1; NUM_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1,2,3; NUM_GPUS=4

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train/action_decoder"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE1_LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_4/8_trajectory_cook_5_3camera_pour_filter_pretrain_False_history_1_seq_48_new_projection_rgb_loss_1_epoch-99.safetensors"

# 机器2的路径配置
MACHINE2_DATA_ROOT=""  # TODO: 填写当前机器的数据根目录
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train/action_decoder"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE2_LORA_CHECKPOINT=""

# 机器3的路径配置
MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/" 
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train/action_decoder"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE3_LORA_CHECKPOINT="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/6_trajectory_cook_4_3camera_cook_4_pretrain_true_history_1_new_projection_rgb_loss_0.08/20251227_230119/epoch-99.safetensors"

# 机器4的路径配置
MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train/action_decoder"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE4_LORA_CHECKPOINT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_3/8_trajectory_cook_5_3camera_pour_filter_pretrain_False_history_1_seq_48_new_projection_rgb_loss_1_epoch-99.safetensors"

# 机器5的路径配置
MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/pour_filter"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/action_decoder"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE5_LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_4/8_trajectory_cook_5_3camera_pour_filter_pretrain_False_history_1_seq_48_new_projection_rgb_loss_1_epoch-99.safetensors"

# 根据机器类型设置路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
    LORA_CHECKPOINT="${MACHINE1_LORA_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    DATA_ROOT="${MACHINE2_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
    LORA_CHECKPOINT="${MACHINE2_LORA_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    DATA_ROOT="${MACHINE3_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
    LORA_CHECKPOINT="${MACHINE3_LORA_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    DATA_ROOT="${MACHINE4_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
    LORA_CHECKPOINT="${MACHINE4_LORA_CHECKPOINT}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    DATA_ROOT="${MACHINE5_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE5_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
    LORA_CHECKPOINT="${MACHINE5_LORA_CHECKPOINT}"
fi

# ==============================================
# 数据集配置
# ==============================================

# 多任务训练和Trail范围过滤配置
TRAIL_START="1"          # 起始trail编号
TRAIL_END="5"            # 结束trail编号

# 数据集参数
SEQUENCE_LENGTH=48        # 未来帧数量（不包括初始帧）。总poses = 1(历史) + 24(未来) = 25
STEP_INTERVAL=1           # 轨迹步长间隔
MIN_TRAIL_LENGTH=10       # 最小轨迹长度
HEATMAP_SIGMA=1.5         # 高斯热力图标准差
COLORMAP_NAME="jet"       # colormap名称

# 历史帧配置
NUM_HISTORY_FRAMES=1      # 历史帧数量（默认1帧）

# 投影模式配置
USE_DIFFERENT_PROJECTION=true  # 使用不同的投影方式

# 数据增强参数
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.1,0.1,0.1"  # 逗号分隔
TRANSFORM_AUG_RPY="5.0,5.0,5.0"  # 逗号分隔

# ==============================================
# Action Decoder模型配置
# ==============================================

# Pipeline配置
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"
USE_DUAL_HEAD=true        # 使用dual head架构

# DiT特征提取配置
EXTRACT_BLOCK_ID=15       # 从DiT的第20个block提取特征
DIT_FEATURE_DIM=3072      # DiT隐藏维度（Wan-5B）
FREEZE_DIT=true           # 冻结DiT权重

# Action Decoder架构配置
HIDDEN_DIM=512            # Decoder隐藏层维度
NUM_VIEWS=3               # 相机视角数量
NUM_ROTATION_BINS=72      # 旋转分类的bin数量（360度/5度=72）
NUM_FUTURE_FRAMES=48      # 预测未来帧数量（与SEQUENCE_LENGTH一致）
DROPOUT=0.1               # Dropout率
DENOISING_TIMESTEP_ID="0"  # 特征提取的固定timestep ID（留空则随机采样）

# 损失权重配置
HEATMAP_LOSS_WEIGHT=1.0   # 热力图损失权重
ROTATION_LOSS_WEIGHT=1.0  # 旋转损失权重
GRIPPER_LOSS_WEIGHT=0.5   # 夹爪损失权重

# ==============================================
# 训练配置
# ==============================================

# 基础训练参数
HEIGHT=256
WIDTH=256
NUM_FRAMES=${SEQUENCE_LENGTH}
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=500
SAVE_EPOCHS_INTERVAL=5

# 多GPU训练参数
TRAIN_BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))

# 显存优化参数
DATASET_NUM_WORKERS=0
MIXED_PRECISION="bf16"
DATALOADER_PIN_MEMORY=false
PREFETCH_FACTOR=2

# 生成时间戳目录名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NAME="action_decoder_block${EXTRACT_BLOCK_ID}_hidden${HIDDEN_DIM}_rgb_only_diffusion_fixed_train_noise"
OUTPUT_PATH="${OUTPUT_BASE}/${NAME}/${TIMESTAMP}"

# SwanLab配置参数
ENABLE_SWANLAB=true
SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"
SWANLAB_PROJECT="Wan-ActionDecoder"
SWANLAB_EXPERIMENT="${NAME}-$(date +%Y%m%d-%H%M%S)"
DEBUG_MODE=false

echo "================================================================"
echo "ACTION DECODER TRAINING FOR Wan2.2-TI2V-5B"
echo "================================================================"
echo "当前使用机器: ${CURRENT_MACHINE}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据根目录: ${DATA_ROOT}"
echo "模型路径: ${MODEL_BASE_PATH}"
echo "LoRA Checkpoint: ${LORA_CHECKPOINT}"
echo "输出路径: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
echo "数据集配置:"
echo "  序列长度: ${SEQUENCE_LENGTH}"
echo "  Trail范围: ${TRAIL_START} - ${TRAIL_END}"
echo "  图像尺寸: ${HEIGHT}x${WIDTH}"
echo "----------------------------------------------------------------"
echo "Action Decoder配置:"
echo "  Pipeline类型: ${WAN_TYPE}"
echo "  使用Dual Head: ${USE_DUAL_HEAD}"
echo "  提取Block ID: ${EXTRACT_BLOCK_ID}"
echo "  DiT特征维度: ${DIT_FEATURE_DIM}"
echo "  隐藏层维度: ${HIDDEN_DIM}"
echo "  视角数量: ${NUM_VIEWS}"
echo "  旋转Bin数: ${NUM_ROTATION_BINS}"
echo "  未来帧数: ${NUM_FUTURE_FRAMES}"
echo "  Dropout率: ${DROPOUT}"
echo "  冻结DiT: ${FREEZE_DIT}"
echo "  Denoising Timestep ID: ${DENOISING_TIMESTEP_ID:-随机采样}"
echo "----------------------------------------------------------------"
echo "损失权重:"
echo "  Heatmap: ${HEATMAP_LOSS_WEIGHT}"
echo "  Rotation: ${ROTATION_LOSS_WEIGHT}"
echo "  Gripper: ${GRIPPER_LOSS_WEIGHT}"
echo "----------------------------------------------------------------"
echo "训练参数:"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  保存间隔: 每${SAVE_EPOCHS_INTERVAL}个epoch"
echo "  GPU数量: ${NUM_GPUS}"
echo "  每GPU批次大小: ${TRAIN_BATCH_SIZE_PER_GPU}"
echo "  梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  有效批次大小: ${EFFECTIVE_BATCH_SIZE}"
echo "  混合精度: ${MIXED_PRECISION}"
echo "  SwanLab启用: ${ENABLE_SWANLAB}"
echo "================================================================"

# ==============================================
# 路径验证
# ==============================================

# 检查数据目录
if [ ! -d "${DATA_ROOT}" ]; then
    echo "错误：数据目录不存在: ${DATA_ROOT}"
    exit 1
fi

# 检查模型目录
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "错误：模型目录不存在: ${MODEL_BASE_PATH}"
    exit 1
fi

# 检查LoRA checkpoint
if [ ! -f "${LORA_CHECKPOINT}" ]; then
    echo "错误：LoRA checkpoint文件不存在: ${LORA_CHECKPOINT}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"
echo "输出目录已创建: ${OUTPUT_PATH}"

# ==============================================
# 开始训练
# ==============================================
echo "================================================================"
echo "STARTING ACTION DECODER TRAINING"
echo "================================================================"

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --mixed_precision=${MIXED_PRECISION} \
  --main_process_port=29500 \
  examples/wanvideo/model_training/train_diffusion_action_decoder.py \
  --heatmap_data_root ${DATA_ROOT} \
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
  --num_history_frames ${NUM_HISTORY_FRAMES} \
  $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi) \
  --model_base_path "${MODEL_BASE_PATH}" \
  --lora_checkpoint "${LORA_CHECKPOINT}" \
  --wan_type "${WAN_TYPE}" \
  $(if [ "${USE_DUAL_HEAD}" = "true" ]; then echo "--use_dual_head"; fi) \
  --extract_block_id ${EXTRACT_BLOCK_ID} \
  --dit_feature_dim ${DIT_FEATURE_DIM} \
  --hidden_dim ${HIDDEN_DIM} \
  --num_views ${NUM_VIEWS} \
  --num_rotation_bins ${NUM_ROTATION_BINS} \
  --num_future_frames ${NUM_FUTURE_FRAMES} \
  --dropout ${DROPOUT} \
  $(if [ "${FREEZE_DIT}" = "true" ]; then echo "--freeze_dit"; fi) \
  $(if [ -n "${DENOISING_TIMESTEP_ID}" ]; then echo "--denoising_timestep_id ${DENOISING_TIMESTEP_ID}"; fi) \
  --heatmap_loss_weight ${HEATMAP_LOSS_WEIGHT} \
  --rotation_loss_weight ${ROTATION_LOSS_WEIGHT} \
  --gripper_loss_weight ${GRIPPER_LOSS_WEIGHT} \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --save_epochs_interval ${SAVE_EPOCHS_INTERVAL} \
  --output_path "${OUTPUT_PATH}" \
  --train_batch_size ${TRAIN_BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} \
  --logging_steps 10 \
  --max_grad_norm 1.0 \
  --warmup_steps 100 \
  $(if [ "${DEBUG_MODE}" = "true" ]; then echo "--debug_mode"; fi) \
  $(if [ "${ENABLE_SWANLAB}" = "true" ]; then echo "--enable_swanlab"; fi) \
  --swanlab_api_key "${SWANLAB_API_KEY}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment "${SWANLAB_EXPERIMENT}"

echo "================================================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"
