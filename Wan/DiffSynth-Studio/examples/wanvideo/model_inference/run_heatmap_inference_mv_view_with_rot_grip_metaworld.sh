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

# 机器2的CoppeliaSim配置（待填写）
MACHINE2_COPPELIASIM_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"  # TODO: 填写当前机器的CoppeliaSim路径
MACHINE2_DISPLAY=":1.0"           # TODO: 填写当前机器的DISPLAY配置
MACHINE2_CONDA_PATH="/root/miniconda3/etc/profile.d/conda.sh"  # TODO: 填写当前机器的conda路径
MACHINE2_CONDA_ENV="metaworld"   # TODO: 填写当前机器的conda环境名


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
MACHINE2_PROJECT_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/Wan/DiffSynth-Studio"
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

MACHINE2_DATA_ROOT_LIST=(
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_door-open_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_door-close_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_shelf-place_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_button-press_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_button-press-topdown_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_faucet-close_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_faucet-open_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_handle-press_expert.zarr"
"/mnt/robot-rfm/user/lpy/data/MetaWorld/metaworld_hammer_expert.zarr"
)
MACHINE2_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/logs/Wan/train"          # TODO: 填写当前机器的输出基础目录
MACHINE2_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B"

MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B"
MACHINE3_LORA_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"

MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/inference"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B"
MACHINE4_LORA_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train"

MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/cook_6"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/inference"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B"
MACHINE5_LORA_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train"

MACHINE6_DATA_ROOT="/mnt/robot-rfm/user/lpy/data/Franka_data_3zed_5/"
MACHINE6_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/inference"
MACHINE6_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B"
MACHINE6_LORA_BASE="/mnt/robot-rfm/user/lpy/BridgeVLA_dev/logs/Wan/train"

if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
    LORA_BASE="${MACHINE1_LORA_BASE}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    # For machine2, DATA_ROOT will be set in the loop for each task
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
# LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora_view/10_trajectory_pour_3camera_view_concat_False_history_1_seq_24_new_projection_rgb_loss_0.08/20260112_051949/epoch-99.safetensors"
# # Rotation and Gripper checkpoint path (leave empty if not available yet)
# ROT_GRIP_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/mv_rot_grip_v2_view/from_a100_3/pour_filter_5_14_trail.pth"
LORA_CHECKPOINT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/metaworld_9task_5traj/20260112_005256/epoch-99.safetensors"
# Rotation and Gripper checkpoint path (leave empty if not available yet)
ROT_GRIP_CHECKPOINT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/logs/Wan/train/mv_rot_grip_v2/metaworld_9task_5traj/20260112_012618/epoch-100.pth"



# ==============================================
# Rotation and Gripper Prediction Parameters
# ==============================================

# Rotation prediction parameters
ROTATION_RESOLUTION=5.0      # Rotation angle resolution (degrees)
HIDDEN_DIM=512              # Hidden layer dimension
LOCAL_FEAT_SIZE=5           # Local feature extraction neighborhood size
NUM_ROTATION_BINS=72

# Automatically calculate number of rotation bins (360 / resolution)
NUM_ROTATION_BINS=$(awk "BEGIN {print int(360 / $ROTATION_RESOLUTION)}")

# History frames configuration (must match training configuration)
# Allowed values: 1 (single frame), 2 (two frames), or 1+4N (5, 9, 13, ...)
# When using 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, must set NUM_HISTORY_FRAMES > 1
NUM_HISTORY_FRAMES=1

# Point cloud configuration
# true: Use merged point cloud from 3 cameras
# false: Use only camera 3's point cloud
USE_MERGED_POINTCLOUD=true

# Projection mode configuration (must match training configuration)
# true: Use different projection mode (each camera projects separately)
# false: Use default projection mode
USE_DIFFERENT_PROJECTION=false

# Initial gripper state input configuration (must match training configuration)
# true: Use initial gripper state as model input
# false: Do not use initial gripper state as input
USE_INITIAL_GRIPPER_STATE=false

# Output directory
OUTPUT_DIR="${OUTPUT_BASE}/inference_debug_metaworld"

# Test data (optional)
# DATA_ROOT is already set based on machine
TEST_INDICES="0,50,100,150,200"

# Dataset parameters-
SCENE_BOUNDS="-0.5, 0.2, 0, 0.5, 1, 0.5"
TRANSFORM_AUG_XYZ="0.0,0.0,0.0"
TRANSFORM_AUG_RPY="0.0,0.0,0.0"

# Trail filtering (optional - leave empty or comment out to use all trails)
TRAIL_START=0  # Starting trail number (e.g., 1 for trail_1)
TRAIL_END=4    # Ending trail number (e.g., 5 for trail_5)

# GPU device
export CUDA_VISIBLE_DEVICES=3

# ==============================================
# Helper function: Extract task name from data path
# ==============================================
extract_task_name() {
    local data_path="$1"
    # Extract filename without extension
    # Example: /path/to/metaworld_door-open_expert.zarr -> door-open
    local filename=$(basename "$data_path" .zarr)
    # Remove "metaworld_" prefix and "_expert" suffix
    local task_name=$(echo "$filename" | sed 's/^metaworld_//;s/_expert$//')
    echo "$task_name"
}

# ==============================================
# Function to run inference for a single task
# ==============================================
run_inference_for_task() {
    local TASK_DATA_ROOT="$1"
    local TASK_NAME="$2"

    # Set output directory for this task
    if [ "${CURRENT_MACHINE}" = "machine2" ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/inference_debug_metaworld/${TASK_NAME}"
    else
        OUTPUT_DIR="${OUTPUT_BASE}/inference_debug_metaworld"
    fi

    echo "=============================================="
    echo "Running MV View Model Inference (Heatmap + Rot/Grip)"
    if [ -n "$TASK_NAME" ]; then
        echo "Task: ${TASK_NAME}"
    fi
    echo "=============================================="
    echo "WAN Type: $WAN_TYPE"
    echo "Dual Head Mode: $USE_DUAL_HEAD"
    echo "Num Views (reference): $NUM_VIEWS (3 RGB + 3 Heatmap)"
    echo "Is Full Finetune: $IS_FULL_FINETUNE"
    echo "Num Frames: $NUM_FRAMES"
    echo "Resolution: ${HEIGHT}x${WIDTH}"
    echo "Inference Steps: $NUM_INFERENCE_STEPS"
    echo "CFG Scale: $CFG_SCALE"
    echo "Model Base: $MODEL_BASE_PATH"
    echo "LoRA Checkpoint: $LORA_CHECKPOINT"
    echo "Rot/Grip Checkpoint: ${ROT_GRIP_CHECKPOINT:-'(Not set - will skip rot/grip prediction)'}"
    echo "Data Root: $TASK_DATA_ROOT"
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

    # Add trail filtering if specified
    if [ -n "${TRAIL_START}" ]; then
        PYTHON_ARGS+=(--trail_start $TRAIL_START)
    fi
    if [ -n "${TRAIL_END}" ]; then
        PYTHON_ARGS+=(--trail_end $TRAIL_END)
    fi

    # Add data root and dataset parameters if specified
    if [ -n "$TASK_DATA_ROOT" ]; then
        PYTHON_ARGS+=(
            --data_root "$TASK_DATA_ROOT"
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

    python3 "${PROJECT_ROOT}/examples/wanvideo/model_inference/heatmap_inference_mv_view_rot_grip_metaworld.py" "${PYTHON_ARGS[@]}"

    echo ""
    echo "=============================================="
    echo "Inference completed!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=============================================="
}

# ==============================================
# Main execution logic
# ==============================================

# Check if we're on machine2 and need to loop through tasks
if [ "${CURRENT_MACHINE}" = "machine2" ]; then
    echo "=============================================="
    echo "Machine2 detected - Running inference for all tasks"
    echo "Total tasks: ${#MACHINE2_DATA_ROOT_LIST[@]}"
    echo "=============================================="

    # Loop through all tasks
    for TASK_PATH in "${MACHINE2_DATA_ROOT_LIST[@]}"; do
        TASK_NAME=$(extract_task_name "$TASK_PATH")
        echo ""
        echo "##############################################"
        echo "# Processing task: ${TASK_NAME}"
        echo "##############################################"

        # Run inference for this task
        run_inference_for_task "$TASK_PATH" "$TASK_NAME"
    done

    echo ""
    echo "=============================================="
    echo "All tasks completed!"
    echo "=============================================="
else
    # For other machines, run inference once with DATA_ROOT
    run_inference_for_task "$DATA_ROOT" ""
fi
