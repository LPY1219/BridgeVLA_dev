#!/bin/bash

# MVTrack Pretraining Script for Wan2.2-TI2V-5B Multi-View Video Diffusion Model
# Based on Wan2.2-TI2V-5B_heatmap_rgb_mv_3.sh, adapted for MVTrack dataset

# ==============================================
# Machine Detection and Environment Configuration
# ==============================================

# Machine 1 CoppeliaSim configuration
MACHINE1_COPPELIASIM_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# Machine 2 CoppeliaSim configuration
MACHINE2_COPPELIASIM_ROOT="/home/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE2_DISPLAY=":1.0"
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

# Machine 3 CoppeliaSim configuration
MACHINE3_COPPELIASIM_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE3_DISPLAY=":1.0"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# Machine 4 CoppeliaSim configuration
MACHINE4_COPPELIASIM_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE4_DISPLAY=":1.0"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"

# Detect machine and set environment
if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    echo "Detected Machine 1"
    CURRENT_MACHINE="machine1"
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE1_DISPLAY}"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
elif [ -d "${MACHINE2_COPPELIASIM_ROOT}" ]; then
    echo "Detected Machine 2"
    CURRENT_MACHINE="machine2"
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE2_DISPLAY}"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
elif [ -d "${MACHINE3_COPPELIASIM_ROOT}" ]; then
    echo "Detected Machine 3"
    CURRENT_MACHINE="machine3"
    export COPPELIASIM_ROOT="${MACHINE3_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE3_DISPLAY}"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
elif [ -d "${MACHINE4_COPPELIASIM_ROOT}" ]; then
    echo "Detected Machine 4"
    CURRENT_MACHINE="machine4"
    export COPPELIASIM_ROOT="${MACHINE4_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE4_DISPLAY}"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
else
    echo "Error: No known COPPELIASIM_ROOT path found"
    exit 1
fi

# ==============================================
# Memory Optimization Environment Variables
# ==============================================
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
# export CUDA_LAUNCH_BLOCKING=1  # 关闭同步启动以提高性能（调试时可开启）
export NCCL_DEBUG=INFO  # 改为INFO以获取更多调试信息
export NCCL_TIMEOUT=7200  # 增加超时时间到2小时
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree

# CUDA Driver Optimization
export CUDA_MODULE_LOADING=LAZY
export CUDA_VISIBLE_DEVICES_ORDER=PCI_BUS_ID
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Stability settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# ==============================================
# Project Path Configuration
# ==============================================

# Machine 1 paths
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE1_DEEPSPEED_CONFIG_DIR="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# Machine 2 paths
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE2_DEEPSPEED_CONFIG_DIR="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# Machine 3 paths
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE3_DEEPSPEED_CONFIG_DIR="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# Machine 4 paths
MACHINE4_PROJECT_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE4_DEEPSPEED_CONFIG_DIR="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_training/config"

# Set paths based on machine
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

# Check project root exists
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "Error: Project root not found: ${PROJECT_ROOT}"
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
cd "${PROJECT_ROOT}"
echo "Working directory: $(pwd)"

# ==============================================
# GPU Configuration
# ==============================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=0; NUM_GPUS=1
# ==============================================
# Data and Model Path Configuration
# ==============================================

# Machine 1 data paths
MACHINE1_MVTRACK_DATA_ROOT="/DATA/disk1/lpy/pretrain_data/MVTrack/MVTrack"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/pretrain"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# Machine 2 data paths
MACHINE2_MVTRACK_DATA_ROOT="/data/lpy/pretrain_data/Dataset/MVTrack"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/pretrain"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# Machine 3 data paths
MACHINE3_MVTRACK_DATA_ROOT="/DATA/disk0/lpy/data/MVTrack"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/pretrain"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# Machine 4 data paths
MACHINE4_MVTRACK_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/pretrain_data/MVTrack"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/pretrain"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused"

# Set paths based on machine
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    MVTRACK_DATA_ROOT="${MACHINE1_MVTRACK_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    MVTRACK_DATA_ROOT="${MACHINE2_MVTRACK_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    MVTRACK_DATA_ROOT="${MACHINE3_MVTRACK_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    MVTRACK_DATA_ROOT="${MACHINE4_MVTRACK_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
fi

# ==============================================
# Training Parameters
# ==============================================



# Wan model type
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV"

# MVTrack specific parameters
SEQUENCE_LENGTH=12        # Number of future frames to predict (total frames = sequence_length + 1)
STEP_INTERVAL=2           # Frame sampling interval
MIN_SEQUENCE_LENGTH=20    # Minimum sequence length requirement
HEATMAP_SIGMA=2         # Gaussian sigma for heatmap generation
NUM_VIEWS=3               # Number of views to sample
COLORMAP_NAME="jet"       # Colormap name

# Image and training parameters
HEIGHT=256
WIDTH=256
NUM_FRAMES=$((SEQUENCE_LENGTH + 1))  # Including first frame
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=30
SAVE_EPOCHS_INTERVAL=2

# Multi-GPU training parameters
TRAIN_BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))
echo "Total effective batch size: ${EFFECTIVE_BATCH_SIZE}"

# Other training parameters
DATASET_NUM_WORKERS=0
USE_GRADIENT_CHECKPOINTING=false
MIXED_PRECISION="bf16"
DATALOADER_PIN_MEMORY=false

# LoRA parameters
LORA_RANK=32
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# Dual Head mode
USE_DUAL_HEAD=true

# Advanced training parameters
# ⚠️  IMPORTANT: Unfreeze modulation and norms for better adaptation
# Set to true for NEW training runs, false for backward compatibility with old checkpoints
# - modulation: AdaLN parameters (~0.55M params, critical for task adaptation)
# - mvs_attn.norm_q/norm_k: RMSNorm in multi-view attention (~0.18M params)
UNFREEZE_MODULATION_AND_NORMS=true  # Set to true when training NEW models

# Generate timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PATH="${OUTPUT_BASE}/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_${UNFREEZE_MODULATION_AND_NORMS}/${TIMESTAMP}"
# SwanLab configuration
ENABLE_SWANLAB=true
SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"
SWANLAB_PROJECT="Wan2.2-TI2V-5B_mvtrack_pretrain"
SWANLAB_EXPERIMENT="mvtrack-pretrain_unfreeze_modulation-$(date +%Y%m%d-%H%M%S)"
DEBUG_MODE=false

echo "================================================================"
echo "MVTRACK PRETRAINING FOR Wan2.2-TI2V-5B"
echo "================================================================"
echo "Current machine: ${CURRENT_MACHINE}"
echo "Project root: ${PROJECT_ROOT}"
echo "Data root: ${MVTRACK_DATA_ROOT}"
echo "Model path: ${MODEL_BASE_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
echo "Training parameters:"
echo "  Sequence length: ${SEQUENCE_LENGTH}"
echo "  Step interval: ${STEP_INTERVAL}"
echo "  Image size: ${HEIGHT}x${WIDTH}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  LoRA rank: ${LORA_RANK}"
echo "  LoRA target modules: ${LORA_TARGET_MODULES}"
echo "  Dual Head mode: ${USE_DUAL_HEAD}"
echo "  GPU count: ${NUM_GPUS}"
echo "  Batch size per GPU: ${TRAIN_BATCH_SIZE_PER_GPU}"
echo "  Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Heatmap sigma: ${HEATMAP_SIGMA}"
echo "  Number of views: ${NUM_VIEWS}"
echo "  Unfreeze modulation & norms: ${UNFREEZE_MODULATION_AND_NORMS}"
echo "  SwanLab enabled: ${ENABLE_SWANLAB}"
echo "  Debug mode: ${DEBUG_MODE}"
echo "================================================================"

# ==============================================
# Path Validation
# ==============================================

# Check data directory
if [ ! -d "${MVTRACK_DATA_ROOT}" ]; then
    echo "Error: Data directory not found: ${MVTRACK_DATA_ROOT}"
    exit 1
fi

# Check model directory
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "Error: Model directory not found: ${MODEL_BASE_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_PATH}"
echo "Output directory created: ${OUTPUT_PATH}"

# ================================================================
# Launch Training
# ================================================================
echo "================================================================"
echo "STARTING MVTRACK PRETRAINING"
echo "================================================================"

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --mixed_precision=${MIXED_PRECISION} \
  --main_process_port=29500 \
  examples/wanvideo/model_training/heatmap_train_mvtrack.py \
  --mvtrack_data_root ${MVTRACK_DATA_ROOT} \
  --sequence_length ${SEQUENCE_LENGTH} \
  --step_interval ${STEP_INTERVAL} \
  --min_sequence_length ${MIN_SEQUENCE_LENGTH} \
  --heatmap_sigma ${HEATMAP_SIGMA} \
  --num_views ${NUM_VIEWS} \
  --colormap_name "${COLORMAP_NAME}" \
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
  --extra_inputs "input_image,input_image_rgb,input_video_rgb" \
  --train_batch_size ${TRAIN_BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} \
  $(if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then echo "--use_gradient_checkpointing"; fi) \
  $(if [ "${USE_DUAL_HEAD}" = "true" ]; then echo "--use_dual_head"; fi) \
  $(if [ "${UNFREEZE_MODULATION_AND_NORMS}" = "true" ]; then echo "--unfreeze_modulation_and_norms"; fi) \
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

echo "================================================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"
