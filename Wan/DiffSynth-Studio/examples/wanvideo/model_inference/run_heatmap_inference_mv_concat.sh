#!/bin/bash

# ==============================================
# Multi-View Token Concatenation Model Inference Script
# ==============================================
# This script runs inference using WanModel_mv_concat which concatenates
# different view tokens along the sequence dimension instead of using
# multi-view attention modules.
# Supports multiple machines with automatic detection.
# ==============================================

# ==============================================
# Machine Configurations
# ==============================================
MACHINE1_COPPELIASIM_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

MACHINE2_COPPELIASIM_ROOT="/home/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE2_DISPLAY=":1.0"
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

MACHINE3_COPPELIASIM_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE3_DISPLAY=":1.0"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"

MACHINE4_COPPELIASIM_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE4_DISPLAY=":1.0"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"

MACHINE5_COPPELIASIM_ROOT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE5_DISPLAY=":1.0"
MACHINE5_CONDA_PATH="/DATA/disk1/yaoliang/miniconda3/etc/profile.d/conda.sh"
MACHINE5_CONDA_ENV="BridgeVLA_DM"

# ==============================================
# Detect machine and setup environment
# ==============================================
if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    echo "Detected machine1"
    CURRENT_MACHINE="machine1"
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE1_DISPLAY}"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
elif [ -d "${MACHINE2_COPPELIASIM_ROOT}" ]; then
    echo "Detected machine2"
    CURRENT_MACHINE="machine2"
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE2_DISPLAY}"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
elif [ -d "${MACHINE3_COPPELIASIM_ROOT}" ]; then
    echo "Detected machine3"
    CURRENT_MACHINE="machine3"
    export COPPELIASIM_ROOT="${MACHINE3_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE3_DISPLAY}"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
elif [ -d "${MACHINE4_COPPELIASIM_ROOT}" ]; then
    echo "Detected machine4"
    CURRENT_MACHINE="machine4"
    export COPPELIASIM_ROOT="${MACHINE4_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE4_DISPLAY}"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
elif [ -d "${MACHINE5_COPPELIASIM_ROOT}" ]; then
    echo "Detected machine5"
    CURRENT_MACHINE="machine5"
    export COPPELIASIM_ROOT="${MACHINE5_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE5_DISPLAY}"
    source "${MACHINE5_CONDA_PATH}"
    conda activate "${MACHINE5_CONDA_ENV}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
else
    echo "Error: No matching COPPELIASIM_ROOT found"
    echo "Checked paths:"
    echo "  Machine1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  Machine2: ${MACHINE2_COPPELIASIM_ROOT}"
    echo "  Machine3: ${MACHINE3_COPPELIASIM_ROOT}"
    echo "  Machine4: ${MACHINE4_COPPELIASIM_ROOT}"
    echo "  Machine5: ${MACHINE5_COPPELIASIM_ROOT}"
    exit 1
fi

# ==============================================
# Project Paths
# ==============================================
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE4_PROJECT_ROOT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/Wan/DiffSynth-Studio"
MACHINE5_PROJECT_ROOT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/Wan/DiffSynth-Studio"

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
fi

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
cd "${PROJECT_ROOT}/examples/wanvideo/model_inference"
echo "Working directory: $(pwd)"

# ==============================================
# Data and Model Paths
# ==============================================
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE1_LORA_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE2_DATA_ROOT="/data/Franka_data_3zed_5/push_T_5"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE2_LORA_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE3_LORA_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/inference"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE4_LORA_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train"

MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/inference"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE5_LORA_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train"

if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE1_LORA_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    DATA_ROOT="${MACHINE2_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE2_LORA_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    DATA_ROOT="${MACHINE3_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE3_LORA_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    DATA_ROOT="${MACHINE4_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE4_LORA_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    DATA_ROOT="${MACHINE5_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE5_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE5_LORA_BASE}"
fi

# Print environment info
echo "=============================================="
echo "Environment Configuration"
echo "=============================================="
echo "Current machine: ${CURRENT_MACHINE}"
echo "COPPELIASIM_ROOT: $COPPELIASIM_ROOT"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "QT_QPA_PLATFORM_PLUGIN_PATH: $QT_QPA_PLATFORM_PLUGIN_PATH"
echo "DISPLAY: $DISPLAY"
echo "=============================================="

# Check CoppeliaSim library
if [ -f "$COPPELIASIM_ROOT/libcoppeliaSim.so.1" ]; then
    echo "CoppeliaSim library found"
else
    echo "Warning: CoppeliaSim library not found at $COPPELIASIM_ROOT/libcoppeliaSim.so.1"
fi

# ==============================================
# MV Concat Model Configuration
# ==============================================

# Model type - Token concatenation model
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_CONCAT"

# Use dual head mode (RGB + Heatmap)
USE_DUAL_HEAD=true

# Number of views
NUM_VIEWS=3

# Is full finetune checkpoint (vs LoRA)
IS_FULL_FINETUNE=false

# Inference parameters
NUM_FRAMES=25
HEIGHT=256
WIDTH=256
NUM_INFERENCE_STEPS=50
CFG_SCALE=1.0

# LoRA checkpoint path (relative to LORA_BASE, or set full path)
# You can override this with an absolute path if needed
LORA_CHECKPOINT="${LORA_BASE}/Wan2.2-TI2V-5B_heatmap_rgb_mv_concat/mv_concat_3views_rgb_loss_0.08/20260105_194029/epoch-99.safetensors"

# Output directory
OUTPUT_DIR="${OUTPUT_BASE}/mv_concat"

# Test data (optional)
# DATA_ROOT is already set based on machine
TEST_INDICES="0,100,200,300,400"

# Dataset parameters
SEQUENCE_LENGTH=24
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.0,0.0,0.0"
TRANSFORM_AUG_RPY="0.0,0.0,0.0"
USE_DIFFERENT_PROJECTION=true

# GPU device
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Running MV Concat Model Inference"
echo "=============================================="
echo "WAN Type: $WAN_TYPE"
echo "Dual Head Mode: $USE_DUAL_HEAD"
echo "Num Views: $NUM_VIEWS"
echo "Is Full Finetune: $IS_FULL_FINETUNE"
echo "Num Frames: $NUM_FRAMES"
echo "Resolution: ${HEIGHT}x${WIDTH}"
echo "Inference Steps: $NUM_INFERENCE_STEPS"
echo "CFG Scale: $CFG_SCALE"
echo "Model Base: $MODEL_BASE_PATH"
echo "LoRA Checkpoint: $LORA_CHECKPOINT"
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Validate paths
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "Warning: Model directory not found: ${MODEL_BASE_PATH}"
fi

if [ -n "${LORA_CHECKPOINT}" ] && [ ! -f "${LORA_CHECKPOINT}" ]; then
    echo "Warning: LoRA checkpoint not found: ${LORA_CHECKPOINT}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command arguments using array (better handling of special characters)
PYTHON_ARGS=(
    --model_base_path "$MODEL_BASE_PATH"
    --lora_checkpoint "$LORA_CHECKPOINT"
    --wan_type "$WAN_TYPE"
    --num_views $NUM_VIEWS
    --output_dir "$OUTPUT_DIR"
    --num_frames $NUM_FRAMES
    --height $HEIGHT
    --width $WIDTH
    --num_inference_steps $NUM_INFERENCE_STEPS
    --cfg_scale $CFG_SCALE
    --test_indices "$TEST_INDICES"
)

# Add dual head flag
if [ "$USE_DUAL_HEAD" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# Add full finetune flag
if [ "$IS_FULL_FINETUNE" = "true" ]; then
    PYTHON_ARGS+=(--is_full_finetune)
fi

# Add data root and dataset parameters if specified
if [ -n "$DATA_ROOT" ]; then
    PYTHON_ARGS+=(
        --data_root "$DATA_ROOT"
        --sequence_length $SEQUENCE_LENGTH
        --scene_bounds="$SCENE_BOUNDS"
        --transform_augmentation_xyz="$TRANSFORM_AUG_XYZ"
        --transform_augmentation_rpy="$TRANSFORM_AUG_RPY"
    )
fi

# Add different projection flag
if [ "$USE_DIFFERENT_PROJECTION" = "true" ]; then
    PYTHON_ARGS+=(--use_different_projection)
fi

# Run inference
echo "Running command:"
echo "python3 heatmap_inference_mv_concat.py ${PYTHON_ARGS[*]}"
echo ""

python3 "${PROJECT_ROOT}/examples/wanvideo/model_inference/heatmap_inference_mv_concat.py" "${PYTHON_ARGS[@]}"

echo ""
echo "=============================================="
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
