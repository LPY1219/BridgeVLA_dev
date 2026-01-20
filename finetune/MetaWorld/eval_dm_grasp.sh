#!/bin/bash
# 自动检测根路径
if [ -d "/mnt/data/cyx/workspace/BridgeVLA_dev" ]; then
    ROOT_PATH_1="/mnt/data/cyx/workspace/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_1
elif [ -d "/DATA/disk1/cyx/BridgeVLA_dev" ]; then
    ROOT_PATH_2="/DATA/disk1/cyx/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_2
elif [ -d "/DATA/disk0/lpy/cyx/BridgeVLA_dev" ]; then
    ROOT_PATH_3="/DATA/disk0/lpy/cyx/BridgeVLA_dev"
    ROOT_PATH=$ROOT_PATH_3
else
    echo "Error: Cannot find BridgeVLA root directory"
    exit 1
fi

# 机器1配置（原始机器）
MACHINE1_CONDA_PATH="/mnt/data/cyx/miniconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="metaworld_eval"


# 机器2配置（当前机器）
MACHINE2_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="metaworld"

# 机器2配置（当前机器）
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="metaworld"

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
else
    echo "检测到机器3，使用配置3"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    CURRENT_MACHINE="machine3"
fi

echo "Using ROOT_PATH: $ROOT_PATH"

cd "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference"

# ==============================================
# 模型和数据配置
# ==============================================
LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/heatmap/9tasks_5traj/heatmap.safetensors"
ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/rot_grip/9tasks_5traj/rotgrip.pth"

# 输出目录
OUTPUT_DIR="${ROOT_PATH}/metaworld_results_grasp"

# 模型基础路径
MODEL_BASE_PATH="/DATA/disk0/lpy/cyx/huggingface/Wan2.2-TI2V-5B"

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
NUM_ROTATION_BINS=$(awk "BEGIN {printf \"%.0f\", 360 / $ROTATION_RESOLUTION}")

# 验证计算结果
if [ -z "$NUM_ROTATION_BINS" ] || [ "$NUM_ROTATION_BINS" = "0" ]; then
    echo "Error: Failed to calculate NUM_ROTATION_BINS. Using default value 72."
    NUM_ROTATION_BINS=72
fi

# ==============================================
# 数据集配置
# ==============================================


# 数据集参数（使用逗号分隔）
SCENE_BOUNDS="-0.5, 0.2, 0, 0.5, 1, 0.5"
SEQUENCE_LENGTH=24                           # 序列长度（不包括初始帧）
IMG_SIZE="256,256"                           # 图像尺寸（height,width）

# ==============================================
# GPU配置
# ==============================================
export CUDA_VISIBLE_DEVICES=1

# ==============================================
# RLBench测试配置
# ==============================================
# 任务列表（可以添加多个任务）
# TASKS=(
#     # "reach"
#     # "push-wall"
#     "soccer"
#     # "bin-picking"
#     # "dial-turn"

# )

TASKS=(
#    "door-open"
#    "door-close"
   "button-press"
   "button-press-topdown"
   "faucet-close"
   "faucet-open"
   "handle-press"
   "shelf-place"
   "hammer"
)
CONSTANT_GRIPPER_NUM=1.0
# 如果只想测试单个任务，可以这样设置：
# TASKS=("push-wall")




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
echo "  Scene Bounds: $SCENE_BOUNDS"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Image Size: $IMG_SIZE"
echo ""
echo "Evaluation Configuration:"
echo "  Tasks: ${TASKS[*]}"
echo "  Total Tasks: ${#TASKS[@]}"
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
# 运行推理脚本（循环测试多个任务）
# ==============================================

# 记录开始时间
START_TIME=$(date +%s)
TOTAL_TASKS=${#TASKS[@]}
CURRENT_TASK=0

echo ""
echo "================================"
echo "Starting evaluation for $TOTAL_TASKS task(s)"
echo "================================"
echo ""

# 循环遍历所有任务
for TASK in "${TASKS[@]}"; do
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    echo ""
    echo "================================"
    echo "[$CURRENT_TASK/$TOTAL_TASKS] Evaluating task: $TASK"
    echo "================================"
    echo ""
    
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
        --constant_gripper_num "$CONSTANT_GRIPPER_NUM"
    )
    
    # 添加dual head参数
    if [ "$USE_DUAL_HEAD" = "true" ]; then
        PYTHON_ARGS+=(--use_dual_head)
    fi
    
    # 添加点云配置参数
    if [ "$USE_MERGED_POINTCLOUD" = "true" ]; then
        PYTHON_ARGS+=(--use_merged_pointcloud)
    fi
    
    # 执行推理（使用VAE decode feature版本）
    TASK_START_TIME=$(date +%s)
    if xvfb-run -a python "${ROOT_PATH}/finetune/MetaWorld/eval_dm_grasp.py" "${PYTHON_ARGS[@]}"; then
        TASK_END_TIME=$(date +%s)
        TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
        echo ""
        echo "✓ Task '$TASK' completed successfully (took ${TASK_DURATION}s)"
        echo "  Results saved to: $OUTPUT_DIR/$TASK"
    else
        TASK_END_TIME=$(date +%s)
        TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
        echo ""
        echo "✗ Task '$TASK' failed (took ${TASK_DURATION}s)"
        echo "  Continuing with next task..."
    fi
    echo ""
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "================================"
echo "All tasks evaluation completed!"
echo "================================"
echo "Total tasks: $TOTAL_TASKS"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"