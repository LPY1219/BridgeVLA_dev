#!/bin/bash

# 自动检测根路径
if [ -d "/DATA/disk0/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH_1="/DATA/disk0/lpy/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_1
elif [ -d "/home/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH_2="/home/lpy/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_2
else
    echo "Error: Cannot find BridgeVLA root directory"
    exit 1
fi

# 机器1配置（原始机器）
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="metaworld_eval"


# 机器2配置（当前机器）
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="metaworld_eval"

# 检测机器并设置conda环境
if [ "${ROOT_PATH}" = "${ROOT_PATH_1}" ]; then
    echo "检测到机器1，使用配置1"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    CURRENT_MACHINE="machine1"
else
    echo "检测到机器2，使用配置2"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    CURRENT_MACHINE="machine2"
fi

echo "Using ROOT_PATH: $ROOT_PATH"

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
LORA_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/metaworld_pretrain_false_unfreeze_modulate_true/20251209_095730/epoch-49.safetensors"
ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip_v2/metaworld/20251210_232605/epoch-10.pth"

# 输出目录
OUTPUT_DIR="${ROOT_PATH}/metaworld_results"

# 模型基础路径
MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"

# 是否使用双head模式（必须与训练时设置一致）
USE_DUAL_HEAD=true

# 旋转预测相关参数
ROTATION_RESOLUTION=5.0      # 旋转角度分辨率（度）
HIDDEN_DIM=512              # 隐藏层维度

# 点云配置
# true: 使用三个相机拼接的点云
# false: 只使用相机1的点云
USE_MERGED_POINTCLOUD=true

# 自动计算旋转bins数量（360度 / 分辨率）
NUM_ROTATION_BINS=$(echo "360 / $ROTATION_RESOLUTION" | bc)

# ==============================================
# 数据集配置
# ==============================================

# 数据集路径（自动检测）
DATA_ROOT_OPTIONS=(
    # "/data/Franka_data_3zed/put_lion_on_top_shelf"
    "/DATA/disk0/lpy/rlbench_data/data/test/"
)

# 查找第一个存在的数据集路径
DATA_ROOT=""
for path in "${DATA_ROOT_OPTIONS[@]}"; do
    if [ -d "$path" ]; then
        DATA_ROOT="$path"
        echo "✓ Found dataset at: $DATA_ROOT"
        break
    fi
done

if [ -z "$DATA_ROOT" ]; then
    echo "✗ Error: Cannot find dataset in any of the following paths:"
    for path in "${DATA_ROOT_OPTIONS[@]}"; do
        echo "  - $path"
    done
    exit 1
fi

# 数据集参数（使用逗号分隔）
SCENE_BOUNDS="-0.5, 0.2, 0, 0.5, 1, 0.5"
SEQUENCE_LENGTH=12                           # 序列长度（不包括初始帧）
IMG_SIZE="256,256"                           # 图像尺寸（height,width）

# ==============================================
# GPU配置
# ==============================================
export CUDA_VISIBLE_DEVICES=0

# ==============================================
# RLBench测试配置
# ==============================================
# put_item_in_drawer reach_and_drag put_groceries_in_cupboard put_money_in_safe close_jar place_cups place_wine_at_rack_location light_bulb_in sweep_to_dustpan_of_size
# turn_tap slide_block_to_color_target open_drawer place_shape_in_shape_sorter push_buttons stack_blocks insert_onto_square_peg meat_off_grill stack_cups
TASK="door-open"  # 任务名称，使用"all"表示所有任务
EVAL_EPISODES=25  # 每个任务的评估次数
EPISODE_LENGTH=25  # 每个任务的最大步数




# ==============================================
# 打印配置信息
# ==============================================
echo "================================"
echo "Running Multi-View Rotation/Gripper Inference..."
echo "================================"
echo "Model Configuration:"
echo "  LoRA Checkpoint: $LORA_CHECKPOINT"
echo "  Rot/Grip Checkpoint: $ROT_GRIP_CHECKPOINT"
echo "  Model Base Path: $MODEL_BASE_PATH"
echo "  WAN Type: $WAN_TYPE"
echo "  Dual Head Mode: $USE_DUAL_HEAD"
echo "  Rotation Resolution: ${ROTATION_RESOLUTION}°"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Num Rotation Bins: $NUM_ROTATION_BINS (auto-calculated: 360° / ${ROTATION_RESOLUTION}°)"
echo "  Use Merged Pointcloud: $USE_MERGED_POINTCLOUD"
echo ""
echo "Dataset Configuration:"
echo "  Data Root: $DATA_ROOT"
echo "  Scene Bounds: $SCENE_BOUNDS"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Image Size: $IMG_SIZE"
echo ""
echo "Evaluation Configuration:"
echo "  Task: $TASK"
echo "  Eval Episodes per Task: $EVAL_EPISODES"
echo "  Episode Length: $EPISODE_LENGTH"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"

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
    --scene_bounds "$SCENE_BOUNDS"
    --sequence_length $SEQUENCE_LENGTH
    --img_size "$IMG_SIZE"
    --rotation_resolution $ROTATION_RESOLUTION
    --hidden_dim $HIDDEN_DIM
    --num_rotation_bins $NUM_ROTATION_BINS
    --device "cuda:0"
    --output_dir "$OUTPUT_DIR"
    --task "$TASK"
)

# 添加dual head参数
if [ "$USE_DUAL_HEAD" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# 添加点云配置参数
if [ "$USE_MERGED_POINTCLOUD" = "true" ]; then
    PYTHON_ARGS+=(--use_merged_pointcloud)
fi

# # 打印完整的命令用于调试
# echo "Executing command with arguments:"
# printf '%s\n' "${PYTHON_ARGS[@]}"
# echo ""

# 执行推理（使用VAE decode feature版本）
xvfb-run -a python3 "${ROOT_PATH}/finetune/MetaWorld/eval_dm.py" "${PYTHON_ARGS[@]}"

echo ""
echo "================================"
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
