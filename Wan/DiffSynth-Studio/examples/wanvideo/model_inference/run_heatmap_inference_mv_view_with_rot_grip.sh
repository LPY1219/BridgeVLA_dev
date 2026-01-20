#!/bin/bash

# ==============================================
# Multi-View Heatmap + Rotation & Gripper Inference Script
# ==============================================
# This script runs inference using WanModel with view-concatenation mode,
# including both heatmap prediction and rotation/gripper prediction.
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

MACHINE6_COPPELIASIM_ROOT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE6_DISPLAY=":1.0"
MACHINE6_CONDA_PATH="/root/miniconda3/etc/profile.d/conda.sh"
MACHINE6_CONDA_ENV="BridgeVLA_DM"

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
elif [ -d "${MACHINE6_COPPELIASIM_ROOT}" ]; then
    echo "Detected machine6"
    CURRENT_MACHINE="machine6"
    export COPPELIASIM_ROOT="${MACHINE6_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE6_DISPLAY}"
    source "${MACHINE6_CONDA_PATH}"
    conda activate "${MACHINE6_CONDA_ENV}"
else
    echo "Error: No matching COPPELIASIM_ROOT found"
    echo "Checked paths:"
    echo "  Machine1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  Machine2: ${MACHINE2_COPPELIASIM_ROOT}"
    echo "  Machine3: ${MACHINE3_COPPELIASIM_ROOT}"
    echo "  Machine4: ${MACHINE4_COPPELIASIM_ROOT}"
    echo "  Machine5: ${MACHINE5_COPPELIASIM_ROOT}"
    echo "  Machine6: ${MACHINE6_COPPELIASIM_ROOT}"
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
MACHINE6_PROJECT_ROOT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

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

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
cd "${PROJECT_ROOT}/examples/wanvideo/model_inference"
echo "Working directory: $(pwd)"

# ==============================================
# Data and Model Paths
# ==============================================
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B"
MACHINE1_LORA_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE2_DATA_ROOT="/data/Franka_data_3zed_5/"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B"
MACHINE2_LORA_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B"
MACHINE3_LORA_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/inference"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B"
MACHINE4_LORA_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train"

MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/inference"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B"
MACHINE5_LORA_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train"

MACHINE6_DATA_ROOT="/mnt/robot-rfm/user/lpy/data/Franka_data_3zed_5/put_lion_8"
MACHINE6_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE6_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B"
MACHINE6_LORA_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train"

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
elif [ "${CURRENT_MACHINE}" = "machine6" ]; then
    DATA_ROOT="${MACHINE6_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE6_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE6_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE6_LORA_BASE}"
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
# MV View Model Configuration (Heatmap + Rot/Grip)
# ==============================================

# Model type - View concatenation with rotation and gripper prediction
# 可选值:
#   - 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP: 标准多视角模式（单帧历史）
#   - 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY: 多帧历史模式（需设置NUM_HISTORY_FRAMES>1）
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"

# Use dual head mode (RGB + Heatmap)
USE_DUAL_HEAD=true

# Number of views (for reference only, NOT passed to Python)
# 6 views total: 3 RGB cameras + 3 Heatmap overlays
# Views are concatenated along batch dimension
NUM_VIEWS=6  # Note: This is for documentation only, not used in Python args

# Use only heatmap views (last 3 views) as input condition
# true: Use only the last 3 heatmap views (skip first 3 RGB views)
# false: Use all 6 views (3 RGB + 3 Heatmap) - default behavior
USE_HEATMAP_VIEWS_ONLY=false

# Is full finetune checkpoint (vs LoRA)
IS_FULL_FINETUNE=false

# Inference parameters
NUM_FRAMES=25
HEIGHT=256
WIDTH=256
NUM_INFERENCE_STEPS=50
CFG_SCALE=1.0

# ==============================================
# Checkpoint Paths
# ==============================================

# LoRA checkpoint path (for heatmap diffusion model)
# Default: Use the specified checkpoint from machine3
LORA_CHECKPOINT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora_view/5_trajectory_cook6_pourfilter6trail_putlion8_multitask_3camera_multinode_4nodes_32cards_False_history_1_seq_24_new_projection_rgb_loss_0.08/20260115_121502/epoch-99.safetensors"
# # Rotation and Gripper checkpoint path (leave empty if not available yet)
ROT_GRIP_CHECKPOINT="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train/mv_rot_grip_v2_view/multinode_4nodes_32cards_view_concat_cook6_pourfilter6trail_putlion8_multitask_local_feat_size_5_seq_24_history_1_3zed_different_projection_true_new_projection_with_gripper_false/20260115_121334/epoch-100.pth"
#LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/from_xm/cook_6_heatmap.safetensors"
# Rotation and Gripper checkpoint path (leave empty if not available yet)
#ROT_GRIP_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/from_xm/cook_6_rot_grip.pth/cook_6_rot_grip.pth"



# ==============================================
# Rotation and Gripper Prediction Parameters
# ==============================================

# Rotation prediction parameters
ROTATION_RESOLUTION=5.0      # Rotation angle resolution (degrees)
HIDDEN_DIM=512              # Hidden layer dimension
LOCAL_FEAT_SIZE=5           # Local feature extraction neighborhood size

# Automatically calculate number of rotation bins (360 / resolution)
# Use bash arithmetic instead of bc for better compatibility
NUM_ROTATION_BINS=$(awk "BEGIN {print int(360 / $ROTATION_RESOLUTION)}")

# History frames configuration (must match training configuration)
# Allowed values: 1 (single frame), 2 (two frames), or 1+4N (5, 9, 13, ...)
# When using 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, must set NUM_HISTORY_FRAMES > 1
NUM_HISTORY_FRAMES=1

# Point cloud configuration
# true: Use merged point cloud from 3 cameras
# false: Use only camera 3's point cloud
USE_MERGED_POINTCLOUD=false

# Projection mode configuration (must match training configuration)
# true: Use different projection mode (each camera projects separately)
# false: Use default projection mode
USE_DIFFERENT_PROJECTION=true

# Initial gripper state input configuration (must match training configuration)
# true: Use initial gripper state as model input
# false: Do not use initial gripper state as input
USE_INITIAL_GRIPPER_STATE=false

# Output directory
OUTPUT_DIR="${OUTPUT_BASE}/mv_view_rot_grip"

# Test data (optional)
# DATA_ROOT is already set based on machine
TEST_INDICES="0,100,200,300,400"

# Dataset parameters
SEQUENCE_LENGTH=24
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.0,0.0,0.0"
TRANSFORM_AUG_RPY="0.0,0.0,0.0"

# Additional configuration flags
USE_MERGED_POINTCLOUD=false  # Whether to use merged pointcloud from 3 cameras
USE_DIFFERENT_PROJECTION=true  # Whether to use different projection mode

# Trail filtering (optional - leave empty or comment out to use all trails)
TRAIL_START=1  # Starting trail number (e.g., 1 for trail_1)
TRAIL_END=8    # Ending trail number (e.g., 5 for trail_5)

# GPU device
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Running MV View Model Inference (Heatmap + Rot/Grip)"
echo "=============================================="
echo "WAN Type: $WAN_TYPE"
echo "Dual Head Mode: $USE_DUAL_HEAD"
echo "Num Views (reference): $NUM_VIEWS (3 RGB + 3 Heatmap)"
echo "Use Heatmap Views Only: $USE_HEATMAP_VIEWS_ONLY"
echo "Is Full Finetune: $IS_FULL_FINETUNE"
echo "Num Frames: $NUM_FRAMES"
echo "Resolution: ${HEIGHT}x${WIDTH}"
echo "Inference Steps: $NUM_INFERENCE_STEPS"
echo "CFG Scale: $CFG_SCALE"
echo "Model Base: $MODEL_BASE_PATH"
echo "LoRA Checkpoint: $LORA_CHECKPOINT"
echo "Rot/Grip Checkpoint: ${ROT_GRIP_CHECKPOINT:-'(Not set - will skip rot/grip prediction)'}"
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Use Merged Pointcloud: $USE_MERGED_POINTCLOUD"
echo "Use Different Projection: $USE_DIFFERENT_PROJECTION"
echo "Use Initial Gripper State: $USE_INITIAL_GRIPPER_STATE"
echo "Num History Frames: $NUM_HISTORY_FRAMES"
echo "Rotation Resolution: ${ROTATION_RESOLUTION}°"
echo "Num Rotation Bins: $NUM_ROTATION_BINS (auto-calculated: 360° / ${ROTATION_RESOLUTION}°)"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Local Feat Size: $LOCAL_FEAT_SIZE"
echo "=============================================="

# Validate paths
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "Warning: Model directory not found: ${MODEL_BASE_PATH}"
fi

if [ -n "${LORA_CHECKPOINT}" ] && [ ! -f "${LORA_CHECKPOINT}" ]; then
    echo "Warning: LoRA checkpoint not found: ${LORA_CHECKPOINT}"
    echo "Please verify the checkpoint path or update LORA_CHECKPOINT variable"
fi

if [ -n "${ROT_GRIP_CHECKPOINT}" ] && [ ! -f "${ROT_GRIP_CHECKPOINT}" ]; then
    echo "Warning: Rot/Grip checkpoint not found: ${ROT_GRIP_CHECKPOINT}"
    echo "Please verify the checkpoint path or update ROT_GRIP_CHECKPOINT variable"
    echo "Note: Inference will proceed with heatmap only if checkpoint is not available"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command arguments using array (better handling of special characters)
PYTHON_ARGS=(
    --model_base_path "$MODEL_BASE_PATH"
    --lora_checkpoint "$LORA_CHECKPOINT"
    --wan_type "$WAN_TYPE"
    --output_dir "$OUTPUT_DIR"
    --num_frames $NUM_FRAMES
    --height $HEIGHT
    --width $WIDTH
    --num_inference_steps $NUM_INFERENCE_STEPS
    --cfg_scale $CFG_SCALE
    --test_indices "$TEST_INDICES"
    --rotation_resolution $ROTATION_RESOLUTION
    --hidden_dim $HIDDEN_DIM
    --num_rotation_bins $NUM_ROTATION_BINS
    --num_history_frames $NUM_HISTORY_FRAMES
    --local_feat_size $LOCAL_FEAT_SIZE
)

# Add rotation/gripper checkpoint if provided
if [ -n "${ROT_GRIP_CHECKPOINT}" ]; then
    PYTHON_ARGS+=(--rot_grip_checkpoint "$ROT_GRIP_CHECKPOINT")
fi

# Add dual head flag
if [ "$USE_DUAL_HEAD" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# Add full finetune flag
if [ "$IS_FULL_FINETUNE" = "true" ]; then
    PYTHON_ARGS+=(--is_full_finetune)
fi

# Add merged pointcloud flag
if [ "$USE_MERGED_POINTCLOUD" = "true" ]; then
    PYTHON_ARGS+=(--use_merged_pointcloud)
fi

# Add different projection flag
if [ "$USE_DIFFERENT_PROJECTION" = "true" ]; then
    PYTHON_ARGS+=(--use_different_projection)
fi

# Add initial gripper state flag
if [ "$USE_INITIAL_GRIPPER_STATE" = "true" ]; then
    PYTHON_ARGS+=(--use_initial_gripper_state)
fi

# Add heatmap views only flag
if [ "$USE_HEATMAP_VIEWS_ONLY" = "true" ]; then
    PYTHON_ARGS+=(--use_heatmap_views_only)
fi

# Add trail filtering if specified
if [ -n "${TRAIL_START}" ]; then
    PYTHON_ARGS+=(--trail_start $TRAIL_START)
fi
if [ -n "${TRAIL_END}" ]; then
    PYTHON_ARGS+=(--trail_end $TRAIL_END)
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

# Run inference
echo "Running command:"
echo "python3 heatmap_inference_mv_view_rot_grip.py ${PYTHON_ARGS[*]}"
echo ""

python3 "${PROJECT_ROOT}/examples/wanvideo/model_inference/heatmap_inference_mv_view_rot_grip.py" "${PYTHON_ARGS[@]}"

echo ""
echo "=============================================="
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
