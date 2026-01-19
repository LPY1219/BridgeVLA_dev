#!/bin/bash

# MVTrack Inference Script for Wan2.2-TI2V-5B Multi-View Video Diffusion Model

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
MACHINE4_COPPELIASIM_ROOT="/DATA/disk2/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
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
    echo "Warning: No known COPPELIASIM_ROOT path found"
    # Try to continue anyway
    CURRENT_MACHINE="unknown"
fi

# ==============================================
# Project Path Configuration
# ==============================================

# Machine 1 paths
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE1_MVTRACK_DATA_ROOT="/DATA/disk1/lpy/pretrain_data/MVTrack/MVTrack"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/pretrain"

# Machine 2 paths
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE2_MVTRACK_DATA_ROOT="/data/lpy/pretrain_data/Dataset/MVTrack"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/pretrain"

# Machine 3 paths
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE3_MVTRACK_DATA_ROOT="/DATA/disk0/lpy/data/MVTrack"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/pretrain"

# Machine 4 paths
MACHINE4_PROJECT_ROOT="/DATA/disk2/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE4_MVTRACK_DATA_ROOT="/DATA/disk2/lpy/data/MVTrack"
MACHINE4_MODEL_BASE_PATH="/DATA/disk2/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE4_OUTPUT_BASE="/DATA/disk2/lpy/BridgeVLA_dev/logs/Wan/pretrain"

# Set paths based on machine
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PROJECT_ROOT="${MACHINE1_PROJECT_ROOT}"
    MVTRACK_DATA_ROOT="${MACHINE1_MVTRACK_DATA_ROOT}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PROJECT_ROOT="${MACHINE2_PROJECT_ROOT}"
    MVTRACK_DATA_ROOT="${MACHINE2_MVTRACK_DATA_ROOT}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    PROJECT_ROOT="${MACHINE3_PROJECT_ROOT}"
    MVTRACK_DATA_ROOT="${MACHINE3_MVTRACK_DATA_ROOT}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    PROJECT_ROOT="${MACHINE4_PROJECT_ROOT}"
    MVTRACK_DATA_ROOT="${MACHINE4_MVTRACK_DATA_ROOT}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
else
    # Default for unknown machine
    PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
    MVTRACK_DATA_ROOT="/data/lpy/pretrain_data/Dataset/MVTrack"
    MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"
    OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/pretrain"
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
export CUDA_VISIBLE_DEVICES=0

# ==============================================
# Model and Training Configuration
# ==============================================

# LoRA checkpoint path - 使用最新的训练checkpoint
# 你可以在这里修改为你想测试的checkpoint
# 注意：如果使用了 --unfreeze_modulation_and_norms 训练，路径中会包含 _unfreeze_modulation_true
CHECKPOINT_DIR="${OUTPUT_BASE}/Wan2.2-TI2V-5B_mvtrack_pretrain_unfreeze_modulation_true/20251208_045615"
LORA_CHECKPOINT="${CHECKPOINT_DIR}/epoch-15.safetensors"

# 检查指定的checkpoint是否存在，不存在则报错退出
if [ ! -f "${LORA_CHECKPOINT}" ]; then
    echo "Error: Specified checkpoint not found: ${LORA_CHECKPOINT}"
    echo "Please check the CHECKPOINT_DIR and LORA_CHECKPOINT paths."
    exit 1
fi

# 模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV"

# 是否使用双head模式（必须与训练时设置一致）
USE_DUAL_HEAD=true

# ==============================================
# Dataset Configuration
# ==============================================

# 使用测试集
SPLIT_FILE="train_split.txt"

# 序列参数（应与训练时一致）
SEQUENCE_LENGTH=12
STEP_INTERVAL=2
NUM_VIEWS=3
HEATMAP_SIGMA=2.0
IMG_SIZE="256,256"

# 测试样本数量
NUM_SAMPLES=5

# 可以指定特定的测试索引（逗号分隔），留空则均匀采样
TEST_INDICES="50,200,350,450,550"

# 推理参数
NUM_INFERENCE_STEPS=50
CFG_SCALE=1.0
SEED=42

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_ROOT}/examples/wanvideo/model_inference/mvtrack_inference_results/${TIMESTAMP}"

# ==============================================
# Print Configuration
# ==============================================
echo "================================================================"
echo "MVTRACK INFERENCE TEST"
echo "================================================================"
echo "Current machine: ${CURRENT_MACHINE}"
echo "Project root: ${PROJECT_ROOT}"
echo "Data root: ${MVTRACK_DATA_ROOT}"
echo "Model path: ${MODEL_BASE_PATH}"
echo "LoRA Checkpoint: ${LORA_CHECKPOINT}"
echo "----------------------------------------------------------------"
echo "Model Configuration:"
echo "  WAN Type: ${WAN_TYPE}"
echo "  Dual Head Mode: ${USE_DUAL_HEAD}"
echo "----------------------------------------------------------------"
echo "Dataset Configuration:"
echo "  Split File: ${SPLIT_FILE}"
echo "  Sequence Length: ${SEQUENCE_LENGTH}"
echo "  Step Interval: ${STEP_INTERVAL}"
echo "  Num Views: ${NUM_VIEWS}"
echo "  Heatmap Sigma: ${HEATMAP_SIGMA}"
echo "  Image Size: ${IMG_SIZE}"
echo "----------------------------------------------------------------"
echo "Inference Configuration:"
echo "  Num Samples: ${NUM_SAMPLES}"
echo "  Inference Steps: ${NUM_INFERENCE_STEPS}"
echo "  CFG Scale: ${CFG_SCALE}"
echo "  Seed: ${SEED}"
echo "----------------------------------------------------------------"
echo "Output Directory: ${OUTPUT_DIR}"
echo "================================================================"

# ==============================================
# Validate Paths
# ==============================================

# Check data directory
if [ ! -d "${MVTRACK_DATA_ROOT}" ]; then
    echo "Error: Data directory not found: ${MVTRACK_DATA_ROOT}"
    exit 1
fi

# Check split file
if [ ! -f "${MVTRACK_DATA_ROOT}/${SPLIT_FILE}" ]; then
    echo "Error: Split file not found: ${MVTRACK_DATA_ROOT}/${SPLIT_FILE}"
    echo "Available split files:"
    ls "${MVTRACK_DATA_ROOT}"/*.txt 2>/dev/null
    exit 1
fi

# Check model directory
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "Error: Model directory not found: ${MODEL_BASE_PATH}"
    exit 1
fi

# Check LoRA checkpoint
if [ ! -f "${LORA_CHECKPOINT}" ]; then
    echo "Error: LoRA checkpoint not found: ${LORA_CHECKPOINT}"
    echo "Available checkpoints in ${OUTPUT_BASE}/Wan2.2-TI2V-5B_mvtrack_pretrain:"
    find "${OUTPUT_BASE}/Wan2.2-TI2V-5B_mvtrack_pretrain" -name "*.safetensors" 2>/dev/null | head -10
    exit 1
fi

echo "✓ All paths validated"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo "Output directory created: ${OUTPUT_DIR}"

# ==============================================
# Run Inference
# ==============================================
echo "================================================================"
echo "STARTING MVTRACK INFERENCE"
echo "================================================================"

# Build Python command arguments
PYTHON_ARGS=(
    --lora_checkpoint "${LORA_CHECKPOINT}"
    --model_base_path "${MODEL_BASE_PATH}"
    --wan_type "${WAN_TYPE}"
    --data_root "${MVTRACK_DATA_ROOT}"
    --split_file "${SPLIT_FILE}"
    --sequence_length ${SEQUENCE_LENGTH}
    --step_interval ${STEP_INTERVAL}
    --num_views ${NUM_VIEWS}
    --heatmap_sigma ${HEATMAP_SIGMA}
    --img_size "${IMG_SIZE}"
    --num_inference_steps ${NUM_INFERENCE_STEPS}
    --cfg_scale ${CFG_SCALE}
    --seed ${SEED}
    --num_samples ${NUM_SAMPLES}
    --output_dir "${OUTPUT_DIR}"
    --device "cuda"
)

# Add dual head parameter
if [ "${USE_DUAL_HEAD}" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# Add test indices if specified
if [ -n "${TEST_INDICES}" ]; then
    PYTHON_ARGS+=(--test_indices "${TEST_INDICES}")
fi

# Print command
echo "Running command:"
echo "python3 ${PROJECT_ROOT}/examples/wanvideo/model_inference/heatmap_inference_mvtrack.py ${PYTHON_ARGS[@]}"
echo ""

# Execute inference
python3 "${PROJECT_ROOT}/examples/wanvideo/model_inference/heatmap_inference_mvtrack.py" "${PYTHON_ARGS[@]}"

echo ""
echo "================================================================"
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "================================================================"
