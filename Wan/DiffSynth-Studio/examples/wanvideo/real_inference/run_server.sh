#!/bin/bash

# RoboWan Server 启动脚本
# 支持多台机器运行，通过路径检测自动选择相应配置

# ==============================================
# 机器检测和环境配置
# ==============================================

# 机器2的配置
MACHINE2_ROOT_PATH="/home/lpy/BridgeVLA_dev"
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器3的配置
MACHINE3_ROOT_PATH="/DATA/disk0/lpy/BridgeVLA_dev"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B"


# 机器4的配置
MACHINE4_ROOT_PATH="/DATA/disk1/lpy_a100_4/BridgeVLA_dev"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B"

# 机器1（disk1/lpy）的配置
MACHINE1_DISK1_ROOT_PATH="/DATA/disk1/lpy_a100_1/BridgeVLA_dev"
MACHINE1_DISK1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_DISK1_CONDA_ENV="BridgeVLA_DM"
MACHINE1_DISK1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B"

# 机器5的配置（lpy_a100_1）
MACHINE5_ROOT_PATH="/DATA/disk1/lpy_a100_1/BridgeVLA_dev"
MACHINE5_CONDA_PATH="/DATA/disk1/yaoliang/miniconda3/etc/profile.d/conda.sh"
MACHINE5_CONDA_ENV="BridgeVLA_DM"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B"

# 检测机器并设置环境
if [ -d "${MACHINE1_ROOT_PATH}" ]; then
    echo "检测到机器1（share/project），使用配置1"
    CURRENT_MACHINE="machine1"
    ROOT_PATH="${MACHINE1_ROOT_PATH}"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ -d "${MACHINE2_ROOT_PATH}" ]; then
    echo "检测到机器2，使用配置2"
    CURRENT_MACHINE="machine2"
    ROOT_PATH="${MACHINE2_ROOT_PATH}"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
elif [ -d "${MACHINE3_ROOT_PATH}" ]; then
    echo "检测到机器3（disk0），使用配置3"
    CURRENT_MACHINE="machine3"
    ROOT_PATH="${MACHINE3_ROOT_PATH}"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
    # 修复CUDA库冲突
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
elif [ -d "${MACHINE4_ROOT_PATH}" ]; then
    echo "检测到机器4（disk1/lpy_a100_4），使用配置4"
    CURRENT_MACHINE="machine4"
    ROOT_PATH="${MACHINE4_ROOT_PATH}"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
    # 修复CUDA库冲突
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
elif [ -d "${MACHINE5_ROOT_PATH}" ]; then
    echo "检测到机器5（disk1/lpy_a100_1），使用配置5"
    CURRENT_MACHINE="machine5"
    ROOT_PATH="${MACHINE5_ROOT_PATH}"
    source "${MACHINE5_CONDA_PATH}"
    conda activate "${MACHINE5_CONDA_ENV}"
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
elif [ -d "${MACHINE1_DISK1_ROOT_PATH}" ]; then
    echo "检测到机器1（disk1/lpy），使用配置1_disk1"
    CURRENT_MACHINE="machine1_disk1"
    ROOT_PATH="${MACHINE1_DISK1_ROOT_PATH}"
    source "${MACHINE1_DISK1_CONDA_PATH}"
    conda activate "${MACHINE1_DISK1_CONDA_ENV}"
    MODEL_BASE_PATH="${MACHINE1_DISK1_MODEL_BASE_PATH}"
else
    echo "错误：未找到BridgeVLA根目录"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_ROOT_PATH}"
    echo "  机器2: ${MACHINE2_ROOT_PATH}"
    echo "  机器3: ${MACHINE3_ROOT_PATH}"
    echo "  机器4: ${MACHINE4_ROOT_PATH}"
    echo "  机器5: ${MACHINE5_ROOT_PATH}"
    echo "  机器1_disk1: ${MACHINE1_DISK1_ROOT_PATH}"
    exit 1
fi

echo "Using ROOT_PATH: $ROOT_PATH"
echo "Using MODEL_BASE_PATH: $MODEL_BASE_PATH"
echo "✓ Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# ==============================================
# 模型checkpoint配置（各机器共用相对路径）
# ==============================================

# LoRA checkpoint - 使用相对于ROOT_PATH的路径
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_2/pretrain_false_unfreeze_modulate_true_diffprojection_true_epoch_99.safetensors"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_plate_1camera_pretrain_false_unfreeze_modulate_true/20251207_230054/epoch-99.safetensors"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_2/pretrain_true_modulate_false_epoch_89.safetensors"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_1/pretrain_false_modulate_false_epoch_99.safetensors"
# Rotation/Gripper checkpoint

# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/1_9_8gpu_no_cache_1zed_put_plate/20251203_022054/epoch-100.pth"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/local_feat_size_5_seq_12_1zed_different_projection_false/20251208_164415/epoch-100.pth"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/3zed_different_projection_gripper_rot_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_1/pretrain_true_unfreeze_modulate_true_diff_projection_true_epoch_99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/3zed_different_projection_gripper_rot_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_1/3camera_pretrain_false_history_5_epoch_99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/local_feat_size_5_seq_12_history_5_1zed_different_projection_true_epoch_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_lion_3camera_pretrain_true_history_1/20251214_051437/epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/local_feat_size_5_seq_12_history_1_1zed_different_projection_true_epoch-71.pth"


# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_3/8_trajectory_put_plate_3camera_pretrain_false_history_2_epoch_99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/put_redbull_local_feat_size_5_seq_12_history_2_1zed_different_projection_true/20251214_192716/epoch-99.pth"



# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_lion_3camera_pretrain_true_history_1_new_projection/20251215_145034/epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/local_feat_size_5_seq_12_history_1_1zed_different_projection_true_new_projection_epoch-100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/push_T_3camera_pretrain_true_history_1_new_projection_epoch-89.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/put_lion_local_feat_size_5_seq_12_history_1_1zed_different_projection_true_new_projection_with_gripper_false_epoch_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_4/8_trajectory_3camera_seq_48_put_lion_pretrain_true_history_1_new_projection_epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/put_lion_local_feat_size_5_seq_48_history_1_3zed_different_projection_true_new_projection_with_gripper_false_epoch-30.pth"



# Old checkpoints (channel concat, seq_48):
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/5_trajectory_pour_3camera_pour_pretrain_False_history_1_seq_48_new_projection_rgb_loss_0.08/20260105_165443/epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_1/pour_5_seq_48_history_1_3zed_different_projection_true_new_projection_with_gripper_false_epoch-79.pth"

# LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora_view/10_trajectory_pour_3camera_view_concat_False_history_1_seq_24_new_projection_rgb_loss_0.08/20260112_051949/epoch-99.safetensors"
# # # Rotation and Gripper checkpoint path (leave empty if not available yet)
# ROT_GRIP_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/mv_rot_grip_v2_view/from_a100_3/pour_filter_5_14_trail.pth"

# New checkpoints (view concat, seq_24):
#LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/from_xm/cook_6_heatmap.safetensors"
## Rotation and Gripper checkpoint path (leave empty if not available yet)
#ROT_GRIP_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/from_xm/cook_6_rot_grip.pth/cook_6_rot_grip.pth"


LORA_CHECKPOINT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora_view/from_xm/push_T_30_epoch_99.safetensors"
# # Rotation and Gripper checkpoint path (leave empty if not available yet)
ROT_GRIP_CHECKPOINT="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora_view/from_xm/3_tasks_rot_grip.pth"

# ==============================================
# 服务器配置
# ==============================================

# 注意：当 NUM_HISTORY_FRAMES > 1 时，必须使用 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"  # 单帧模式
# WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY"  # 多帧历史模式（配合 NUM_HISTORY_FRAMES > 1）
# WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"  # 多帧历史模式（配合 NUM_HISTORY_FRAMES > 1）
USE_DIFFERENT_PROJECTION=true  # 与训练时保持一致
NUM_HISTORY_FRAMES=1  # 历史帧数量（必须与训练时一致：1, 2, 或 1+4N）
USE_INITIAL_GRIPPER_STATE=false  # 是否使用初始夹爪状态作为输入（必须与训练时保持一致）
USE_DUAL_HEAD=true
ROTATION_RESOLUTION=5.0
HIDDEN_DIM=512
NUM_ROTATION_BINS=72
LOCAL_FEAT_SIZE=5
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
IMG_SIZE=256  # 图像尺寸，用于投影接口
NUM_FRAMES=25  # 总帧数（包括初始帧），必须与训练时的sequence_length+1一致 (seq_24 + 1 = 25)
DEVICE="cuda:1"
HOST="0.0.0.0"
PORT=5555


echo "================================"
echo "Starting RoboWan Server"
echo "================================"
echo "Machine: ${CURRENT_MACHINE}"
echo ""
echo "Model Configuration:"
echo "  LoRA Checkpoint: $LORA_CHECKPOINT"
echo "  Rot/Grip Checkpoint: $ROT_GRIP_CHECKPOINT"
echo "  Model Base Path: $MODEL_BASE_PATH"
echo "  WAN Type: $WAN_TYPE"
echo "  Dual Head Mode: $USE_DUAL_HEAD"
echo "  Different Projection: $USE_DIFFERENT_PROJECTION"
echo "  Num History Frames: $NUM_HISTORY_FRAMES"
echo "  Use Initial Gripper State: $USE_INITIAL_GRIPPER_STATE"
echo "  Rotation Resolution: ${ROTATION_RESOLUTION}°"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Num Rotation Bins: $NUM_ROTATION_BINS"
echo "  Local Feat Size: $LOCAL_FEAT_SIZE"
echo "  Scene Bounds: $SCENE_BOUNDS"
echo "  Image Size: $IMG_SIZE"
echo "  Num Frames: $NUM_FRAMES"
echo ""
echo "Server Configuration:"
echo "  Device: $DEVICE"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "================================"

# 检查checkpoint文件是否存在
if [ ! -f "$LORA_CHECKPOINT" ]; then
    echo "✗ Error: LoRA checkpoint not found: $LORA_CHECKPOINT"
    exit 1
fi

if [ ! -f "$ROT_GRIP_CHECKPOINT" ]; then
    echo "✗ Error: Rotation/Gripper checkpoint not found: $ROT_GRIP_CHECKPOINT"
    exit 1
fi

echo "✓ All checkpoint files found"
echo ""

# 构建Python命令参数
PYTHON_ARGS=(
    --lora_checkpoint "$LORA_CHECKPOINT"
    --rot_grip_checkpoint "$ROT_GRIP_CHECKPOINT"
    --model_base_path "$MODEL_BASE_PATH"
    --wan_type "$WAN_TYPE"
    --rotation_resolution $ROTATION_RESOLUTION
    --hidden_dim $HIDDEN_DIM
    --num_rotation_bins $NUM_ROTATION_BINS
    --local_feat_size $LOCAL_FEAT_SIZE
    --num_history_frames $NUM_HISTORY_FRAMES
    --scene_bounds="$SCENE_BOUNDS"
    --img_size $IMG_SIZE
    --num_frames $NUM_FRAMES
    --device "$DEVICE"
    --host "$HOST"
    --port $PORT
)

# 添加dual head参数
if [ "$USE_DUAL_HEAD" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# 添加different projection参数
if [ "$USE_DIFFERENT_PROJECTION" = "true" ]; then
    PYTHON_ARGS+=(--use_different_projection)
fi

# 添加初始夹爪状态参数
if [ "$USE_INITIAL_GRIPPER_STATE" = "true" ]; then
    PYTHON_ARGS+=(--use_initial_gripper_state)
fi

# 启动服务器
python3 "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/real_inference/RoboWan_server.py" "${PYTHON_ARGS[@]}"

echo ""
echo "================================"
echo "Server stopped"
echo "================================"
