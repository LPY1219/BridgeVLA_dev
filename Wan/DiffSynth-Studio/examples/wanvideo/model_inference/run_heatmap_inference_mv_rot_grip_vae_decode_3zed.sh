#!/bin/bash

# 自动检测根路径
if [ -d "/DATA/disk1/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH_1="/DATA/disk1/lpy/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_1
elif [ -d "/home/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH_2="/home/lpy/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_2
elif [ -d "/DATA/disk0/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH_3="/DATA/disk0/lpy/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_3
elif [ -d "/DATA/disk1/lpy_a100_4/BridgeVLA_dev" ]; then
    ROOT_PATH_4="/DATA/disk1/lpy_a100_4/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_4
elif [ -d "/DATA/disk1/lpy_a100_1/BridgeVLA_dev" ]; then
    ROOT_PATH_5="/DATA/disk1/lpy_a100_1/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_5
else
    echo "Error: Cannot find BridgeVLA root directory"
    exit 1
fi

# 机器1配置
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2配置
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

# 机器3配置
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# 机器4配置
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"

# 机器5配置
MACHINE5_CONDA_PATH="/DATA/disk1/yaoliang/miniconda3/etc/profile.d/conda.sh"
MACHINE5_CONDA_ENV="BridgeVLA_DM"

# 检测机器并设置conda环境
if [ "${ROOT_PATH}" = "${ROOT_PATH_1}" ]; then
    echo "检测到机器1，使用配置1"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    CURRENT_MACHINE="machine1"
elif [ "${ROOT_PATH}" = "${ROOT_PATH_2}" ]; then
    echo "检测到机器2，使用配置2"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    CURRENT_MACHINE="machine2"
elif [ "${ROOT_PATH}" = "${ROOT_PATH_3}" ]; then
    echo "检测到机器3，使用配置3"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    CURRENT_MACHINE="machine3"
elif [ "${ROOT_PATH}" = "${ROOT_PATH_4}" ]; then
    echo "检测到机器4，使用配置4"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    CURRENT_MACHINE="machine4"
elif [ "${ROOT_PATH}" = "${ROOT_PATH_5}" ]; then
    echo "检测到机器5，使用配置5"
    source "${MACHINE5_CONDA_PATH}"
    conda activate "${MACHINE5_CONDA_ENV}"
    CURRENT_MACHINE="machine5"
fi

cd "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference"

# 设置CoppeliaSim环境变量
export COPPELIASIM_ROOT="${ROOT_PATH}/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export DISPLAY=":1.0"

# 打印环境变量确认
echo "Environment variables set:"
echo "COPPELIASIM_ROOT: $COPPELIASIM_ROOT"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "QT_QPA_PLATFORM_PLUGIN_PATH: $QT_QPA_PLATFORM_PLUGIN_PATH"
echo "DISPLAY: $DISPLAY"

# 检查CoppeliaSim库文件是否存在
if [ -f "$COPPELIASIM_ROOT/libcoppeliaSim.so.1" ]; then
    echo "✓ CoppeliaSim library found"
else
    echo "✗ CoppeliaSim library not found at $COPPELIASIM_ROOT/libcoppeliaSim.so.1"
    find "$COPPELIASIM_ROOT" -name "*coppelia*" -type f 2>/dev/null | head -5
fi

# ==============================================
# 模型和数据配置
# ==============================================

# 各机器的模型路径配置
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B-fused"

# 根据机器类型设置模型路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
fi

# LoRA checkpoint路径（diffusion model）
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251017_170901/epoch-56.safetensors"
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251115_170416/epoch-10.safetensors"
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/from_a100_2/Wan2.2-TI2V-5B_heatmap_rgb_lora/epoch-18.safetensors"
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/from_a100_2/Wan2.2-TI2V-5B_heatmap_rgb_lora/10_trajectory_epoch-19.safetensors"
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/100_trajectory/20251121_152901/epoch-19.safetensors"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/15_trajectory/20251126_023541/epoch-19.safetensors"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/15_trajectory/20251126_185709/epoch-19.safetensors"
# # 旋转和夹爪预测器checkpoint路径
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/1_16_5gpu_no_cache_3zed/20251126_104032/epoch-19.pth"
# ROT_GRIP_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/mv_rot_grip_v2/1_16_7gpu_no_cache/20251123_190601/epoch-20.pth"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_plate/20251128_140256/epoch-90.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/from_a100_2/Wan2.2-TI2V-5B_heatmap_rgb_lora/put_plate_100_epoch.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_plate_1camera/20251202_183407/epoch-31.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/from_a100_4/3zed_different_projection_gripper_rot_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_plate_1camera_pretrain_true/20251207_173841/epoch-19.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/1_9_8gpu_no_cache_1zed_put_plate/20251129_201828/epoch-100.pth"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_plate_1camera_pretrain_false_unfreeze_modulate_true/20251207_230054/epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/1_9_8gpu_no_cache_1zed_put_plate/20251203_022054/epoch-100.pth"
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_2/pretrain_false_unfreeze_modulate_true_diffprojection_true_epoch_99.safetensors"

# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/3zed_different_projection_gripper_rot_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_1/pretrain_true_unfreeze_modulate_true_diff_projection_true_epoch_99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/3zed_different_projection_gripper_rot_100.pth"

# # 多帧历史模式（NUM_HISTORY_FRAMES=5）的checkpoints
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_1/3camera_pretrain_false_history_5_epoch_99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/local_feat_size_5_seq_12_history_5_1zed_different_projection_true_epoch_100.pth"



# 单帧模式（NUM_HISTORY_FRAMES=1）的checkpoints（已注释）
# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_1/pretrain_true_unfreeze_modulate_true_diff_projection_true_epoch_99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/3zed_different_projection_gripper_rot_100.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_lion_3camera_pretrain_true_history_1/20251214_051437/epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/local_feat_size_5_seq_12_history_1_1zed_different_projection_true_epoch-100.pth"


# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/8_trajectory_put_lion_3camera_pretrain_true_history_1_new_projection/20251215_145034/epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/from_a100_4/local_feat_size_5_seq_12_history_1_1zed_different_projection_true_new_projection_epoch-100.pth"


# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_4/8_trajectory_3camera_seq_48_put_lion_pretrain_true_history_1_new_projection_epoch-99.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/put_lion_local_feat_size_5_seq_48_history_1_3zed_different_projection_true_new_projection_with_gripper_false_epoch-30.pth"

# LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/from_a100_4/8_trajectory_push_T_3camera_seq_48_push_T_2_pretrain_true_history_1_new_projection_epoch-89.safetensors"
# ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/put_lion_local_feat_size_5_seq_48_history_1_3zed_different_projection_true_new_projection_with_gripper_false/20251218_175815/epoch-90.pth"




LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/5_trajectory_pour_3camera_pour_rgb_only_pretrain_True_history_1_seq_48_new_projection_rgb_loss_0.08/20260107_013956/epoch-99.safetensors"
ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/cook_4_seq_24_epoch-100.pth"




# 模型类型
# 可选值:
#   - 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP: 标准多视角模式（单帧历史）
#   - 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY: 多帧历史模式（需设置NUM_HISTORY_FRAMES>1）
# WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY"  
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"
# 历史帧数量配置（必须与训练时保持一致）
# 允许的值：1（单帧）, 2（两帧）, 或 1+4N（5, 9, 13, ...）
# 当使用 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY 时，必须设置为 >1
NUM_HISTORY_FRAMES=1
# 是否使用双head模式（必须与训练时设置一致）
USE_DUAL_HEAD=true

# 是否为全量微调checkpoint（必须与训练时设置一致）
# true: 使用全量微调的checkpoint（所有模块都被训练）
# false: 使用LoRA checkpoint（只有LoRA层被训练）
IS_FULL_FINETUNE=false



# 旋转预测相关参数
ROTATION_RESOLUTION=5.0      # 旋转角度分辨率（度）
HIDDEN_DIM=512              # 隐藏层维度
LOCAL_FEAT_SIZE=5           # 局部特征提取的邻域大小

# 点云配置
# true: 使用三个相机拼接的点云
# false: 只使用相机1的点云
USE_MERGED_POINTCLOUD=false

# 投影模式配置
# true: 使用不同的投影方式（每个相机单独投影，必须与训练时保持一致）
# false: 使用默认投影方式
USE_DIFFERENT_PROJECTION=true

# 初始夹爪状态输入配置（必须与训练时保持一致）
# true: 将初始夹爪状态作为模型输入
# false: 不使用初始夹爪状态作为输入
USE_INITIAL_GRIPPER_STATE=false

# 自动计算旋转bins数量（360度 / 分辨率）
NUM_ROTATION_BINS=$(echo "360 / $ROTATION_RESOLUTION" | bc)

# ==============================================
# 数据集配置
# ==============================================

# 各机器的数据集路径配置
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/pour_filter"
MACHINE2_DATA_ROOT="/data/Franka_data_3zed_4/pour_filter"
MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/pour_filter"
MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/pour_filter"
MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/pour_filter"

# 根据机器类型设置数据集路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    DATA_ROOT="${MACHINE2_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    DATA_ROOT="${MACHINE3_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    DATA_ROOT="${MACHINE4_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    DATA_ROOT="${MACHINE5_DATA_ROOT}"
fi

# 检查数据集路径是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "✗ Error: Cannot find dataset at: $DATA_ROOT"
    echo "Current machine: ${CURRENT_MACHINE}"
    exit 1
else
    echo "✓ Found dataset at: $DATA_ROOT"
fi

# 数据集参数（使用逗号分隔）
# SCENE_BOUNDS="0,-0.45,-0.05,0.8,0.55,0.6"  # 场景边界
# SCENE_BOUNDS="0,-0.55,-0.05,0.8,0.45,0.6"
# SCENE_BOUNDS="0,-0.65,-0.05,0.8,0.55,0.75"
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.0,0.0,0.0"            # xyz变换增强（测试时通常为0）
TRANSFORM_AUG_RPY="0.0,0.0,0.0"            # rpy变换增强（测试时通常为0）
SEQUENCE_LENGTH=48                          # 序列长度（不包括初始帧）
IMG_SIZE="256,256"                           # 图像尺寸（height,width）

# 测试样本索引（逗号分隔）
# TEST_INDICES="100,200,300,400,500"
# TEST_INDICES="250,550,700,800,900"
# TEST_INDICES="255,555,705,805,905"
TEST_INDICES="50,150,255,355,405"
# 输出目录x
OUTPUT_DIR="${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/5B_TI2V_MV_ROT_GRIP"

# ==============================================
# GPU配置
# ==============================================
export CUDA_VISIBLE_DEVICES=1

# ==============================================
# 打印配置信息
# ==============================================
echo "================================"
echo "Running Multi-View Rotation/Gripper Inference..."
echo "================================"
echo "Machine Configuration:"
echo "  Current Machine: ${CURRENT_MACHINE}"
echo "  ROOT_PATH: ${ROOT_PATH}"
echo ""
echo "Model Configuration:"
echo "  LoRA Checkpoint: $LORA_CHECKPOINT"
echo "  Rot/Grip Checkpoint: $ROT_GRIP_CHECKPOINT"
echo "  Model Base Path: $MODEL_BASE_PATH"
echo "  WAN Type: $WAN_TYPE"
echo "  Dual Head Mode: $USE_DUAL_HEAD"
echo "  Is Full Finetune: $IS_FULL_FINETUNE"
echo "  Num History Frames: $NUM_HISTORY_FRAMES"
echo "  Rotation Resolution: ${ROTATION_RESOLUTION}°"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Local Feat Size: $LOCAL_FEAT_SIZE"
echo "  Num Rotation Bins: $NUM_ROTATION_BINS (auto-calculated: 360° / ${ROTATION_RESOLUTION}°)"
echo "  Use Merged Pointcloud: $USE_MERGED_POINTCLOUD"
echo "  Use Different Projection: $USE_DIFFERENT_PROJECTION"
echo "  Use Initial Gripper State: $USE_INITIAL_GRIPPER_STATE"
echo ""
echo "Dataset Configuration:"
echo "  Data Root: $DATA_ROOT"
echo "  Scene Bounds: $SCENE_BOUNDS"
echo "  Transform Aug XYZ: $TRANSFORM_AUG_XYZ"
echo "  Transform Aug RPY: $TRANSFORM_AUG_RPY"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Image Size: $IMG_SIZE"
echo "  Test Indices: $TEST_INDICES"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "================================"

# ==============================================
# 检查checkpoint文件是否存在
# ==============================================
if [ ! -f "$LORA_CHECKPOINT" ]; then
    echo "✗ Error: LoRA checkpoint not found: $LORA_CHECKPOINT"
    echo "Please check the path or update LORA_CHECKPOINT variable"
    exit 1
fi

if [ ! -f "$ROT_GRIP_CHECKPOINT" ]; then
    echo "✗ Error: Rotation/Gripper checkpoint not found: $ROT_GRIP_CHECKPOINT"
    echo "Please check the path or update ROT_GRIP_CHECKPOINT variable"
    exit 1
fi

echo "✓ All checkpoint files found"
echo ""

# ==============================================
# 运行推理脚本
# ==============================================

# 构建Python命令参数
PYTHON_ARGS=(
    --lora_checkpoint "$LORA_CHECKPOINT"
    --rot_grip_checkpoint "$ROT_GRIP_CHECKPOINT"
    --model_base_path "$MODEL_BASE_PATH"
    --wan_type "$WAN_TYPE"
    --output_dir "$OUTPUT_DIR"
    --data_root "$DATA_ROOT"
    --scene_bounds="$SCENE_BOUNDS"
    --transform_augmentation_xyz="$TRANSFORM_AUG_XYZ"
    --transform_augmentation_rpy="$TRANSFORM_AUG_RPY"
    --sequence_length $SEQUENCE_LENGTH
    --img_size "$IMG_SIZE"
    --test_indices "$TEST_INDICES"
    --rotation_resolution $ROTATION_RESOLUTION
    --hidden_dim $HIDDEN_DIM
    --num_rotation_bins $NUM_ROTATION_BINS
    --num_history_frames $NUM_HISTORY_FRAMES
    --local_feat_size $LOCAL_FEAT_SIZE
    --device "cuda"
)

# 添加dual head参数
if [ "$USE_DUAL_HEAD" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# 添加全量微调参数
if [ "$IS_FULL_FINETUNE" = "true" ]; then
    PYTHON_ARGS+=(--is_full_finetune)
fi

# 添加点云配置参数
if [ "$USE_MERGED_POINTCLOUD" = "true" ]; then
    PYTHON_ARGS+=(--use_merged_pointcloud)
fi

# 添加投影模式参数
if [ "$USE_DIFFERENT_PROJECTION" = "true" ]; then
    PYTHON_ARGS+=(--use_different_projection)
fi

# 添加初始夹爪状态参数
if [ "$USE_INITIAL_GRIPPER_STATE" = "true" ]; then
    PYTHON_ARGS+=(--use_initial_gripper_state)
fi

# # 打印完整的命令用于调试
# echo "Executing command with arguments:"
# printf '%s\n' "${PYTHON_ARGS[@]}"
# echo ""

# 执行推理（使用VAE decode feature版本）
python3 "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed.py" "${PYTHON_ARGS[@]}"

echo ""
echo "================================"
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
