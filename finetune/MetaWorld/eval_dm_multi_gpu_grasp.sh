#!/bin/bash
# 多GPU并行推理脚本
# 自动将任务分配到不同GPU上并行执行

# ==============================================
# GPU配置 - 在这里指定要使用的GPU
# ==============================================
# 可以指定多个GPU，例如: GPU_IDS=(0 1 2 3)
GPU_IDS=(4 5 6 7)  # 修改这里来指定要使用的GPU

# ==============================================
# 自动检测根路径
# ==============================================
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

# 机器3配置（当前机器）
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
# LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/heatmap/9tasks_5traj/heatmap.safetensors"
# ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/rot_grip/9tasks_5traj/rotgrip.pth"
# LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/heatmap/2new_tasks/heatmap_2tasks_new.safetensors"
# ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/rot_grip/2new_tasks/rotgrip_2tasks_new.pth"
LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/heatmap/9tasks_new/heatmap_new.safetensors"
ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/ckpt/rot_grip/9tasks_new/rotgrip_new.pth"


# 输出目录
OUTPUT_DIR="${ROOT_PATH}/metaworld_results_25_grasp_multi_gpu"

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
# 任务配置
# ==============================================
TASKS=(
    "door-open"
    "door-close"
    "button-press"
    "button-press-topdown"
    "faucet-close"
    "faucet-open"
    "handle-press"
    "shelf-place"
    "hammer"
    # "basketball"
    # "assembly"
)

# ==============================================
# 打印配置信息
# ==============================================
echo "================================"
echo "Multi-GPU Parallel Inference"
echo "================================"
echo "GPU Configuration:"
echo "  Available GPUs: ${GPU_IDS[*]}"
echo "  Number of GPUs: ${#GPU_IDS[@]}"
echo ""
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
# 定义单任务执行函数
# ==============================================
run_task_on_gpu() {
    local TASK=$1
    local GPU_ID=$2
    local TASK_INDEX=$3
    local TOTAL_TASKS=$4

    # 创建任务专属的日志文件
    local LOG_FILE="${OUTPUT_DIR}/logs/${TASK}_gpu${GPU_ID}.log"
    mkdir -p "${OUTPUT_DIR}/logs"

    echo "[GPU $GPU_ID] [$TASK_INDEX/$TOTAL_TASKS] Starting task: $TASK" | tee -a "$LOG_FILE"

    # 构建Python命令参数
    local PYTHON_ARGS=(
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
        --device "cuda:0"  # 注意：这里使用cuda:0，因为CUDA_VISIBLE_DEVICES已经设置
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

    # 执行推理
    local TASK_START_TIME=$(date +%s)

    # 设置当前任务使用的GPU
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    if xvfb-run -a python "${ROOT_PATH}/finetune/MetaWorld/eval_dm_grasp.py" "${PYTHON_ARGS[@]}" >> "$LOG_FILE" 2>&1; then
        local TASK_END_TIME=$(date +%s)
        local TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
        echo "[GPU $GPU_ID] ✓ Task '$TASK' completed successfully (took ${TASK_DURATION}s)" | tee -a "$LOG_FILE"
        return 0
    else
        local TASK_END_TIME=$(date +%s)
        local TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
        echo "[GPU $GPU_ID] ✗ Task '$TASK' failed (took ${TASK_DURATION}s)" | tee -a "$LOG_FILE"
        return 1
    fi
}

# 导出函数和变量，使其在子shell中可用
export -f run_task_on_gpu
export ROOT_PATH LORA_CHECKPOINT ROT_GRIP_CHECKPOINT MODEL_BASE_PATH WAN_TYPE
export SCENE_BOUNDS SEQUENCE_LENGTH IMG_SIZE ROTATION_RESOLUTION HIDDEN_DIM NUM_ROTATION_BINS
export OUTPUT_DIR USE_DUAL_HEAD USE_MERGED_POINTCLOUD

# ==============================================
# 并行执行任务 - 使用任务队列确保每个GPU同时只运行一个任务
# ==============================================
echo ""
echo "================================"
echo "Starting parallel evaluation"
echo "================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 获取GPU和任务数量
NUM_GPUS=${#GPU_IDS[@]}
TOTAL_TASKS=${#TASKS[@]}

echo "Task distribution:"
echo "  Total GPUs: $NUM_GPUS"
echo "  Total Tasks: $TOTAL_TASKS"
echo "  Strategy: Each GPU runs one task at a time (queue-based)"
echo ""

# 为每个GPU创建任务队列
declare -A GPU_TASK_QUEUES
for i in "${!TASKS[@]}"; do
    TASK=${TASKS[$i]}
    GPU_INDEX=$((i % NUM_GPUS))
    GPU_ID=${GPU_IDS[$GPU_INDEX]}

    if [ -z "${GPU_TASK_QUEUES[$GPU_ID]}" ]; then
        GPU_TASK_QUEUES[$GPU_ID]="$TASK"
    else
        GPU_TASK_QUEUES[$GPU_ID]="${GPU_TASK_QUEUES[$GPU_ID]} $TASK"
    fi
done

# 打印每个GPU的任务队列
echo "GPU Task Queues:"
for GPU_ID in "${GPU_IDS[@]}"; do
    TASKS_FOR_GPU=(${GPU_TASK_QUEUES[$GPU_ID]})
    echo "  GPU $GPU_ID: ${TASKS_FOR_GPU[*]} (${#TASKS_FOR_GPU[@]} tasks)"
done
echo ""

# 定义GPU工作函数：串行执行该GPU上的所有任务
run_gpu_queue() {
    local GPU_ID=$1
    local TASKS_STRING=$2
    local TASKS_ARRAY=($TASKS_STRING)
    local NUM_GPU_TASKS=${#TASKS_ARRAY[@]}

    local GPU_SUCCESS=0
    local GPU_FAIL=0

    for i in "${!TASKS_ARRAY[@]}"; do
        local TASK=${TASKS_ARRAY[$i]}
        local TASK_NUM=$((i + 1))

        echo "[GPU $GPU_ID] [$TASK_NUM/$NUM_GPU_TASKS] Starting task: $TASK"

        if run_task_on_gpu "$TASK" "$GPU_ID" "$TASK_NUM" "$NUM_GPU_TASKS"; then
            GPU_SUCCESS=$((GPU_SUCCESS + 1))
        else
            GPU_FAIL=$((GPU_FAIL + 1))
        fi
    done

    echo "[GPU $GPU_ID] Completed all tasks: $GPU_SUCCESS succeeded, $GPU_FAIL failed"
    return $GPU_FAIL
}

# 导出GPU队列函数
export -f run_gpu_queue

# 启动每个GPU的工作进程（并行）
declare -a GPU_PIDS
declare -a GPU_IDS_RUNNING

for GPU_ID in "${GPU_IDS[@]}"; do
    TASKS_FOR_GPU="${GPU_TASK_QUEUES[$GPU_ID]}"

    if [ -n "$TASKS_FOR_GPU" ]; then
        echo "Starting GPU $GPU_ID worker (will process tasks sequentially)..."
        run_gpu_queue "$GPU_ID" "$TASKS_FOR_GPU" &
        GPU_PIDS+=($!)
        GPU_IDS_RUNNING+=($GPU_ID)
    fi
done

echo ""
echo "All GPU workers started (${#GPU_PIDS[@]} GPUs)"
echo "Waiting for all GPUs to complete their task queues..."
echo ""

# ==============================================
# 等待所有GPU完成并收集结果
# ==============================================
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!GPU_PIDS[@]}"; do
    PID=${GPU_PIDS[$i]}
    GPU_ID=${GPU_IDS_RUNNING[$i]}

    # 等待GPU进程完成
    wait $PID
    GPU_EXIT_CODE=$?

    if [ $GPU_EXIT_CODE -eq 0 ]; then
        echo "GPU $GPU_ID completed all tasks successfully"
    else
        echo "GPU $GPU_ID had $GPU_EXIT_CODE failed task(s)"
        FAIL_COUNT=$((FAIL_COUNT + GPU_EXIT_CODE))
    fi
done

# 计算成功任务数
SUCCESS_COUNT=$((TOTAL_TASKS - FAIL_COUNT))

# ==============================================
# 打印最终统计
# ==============================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "================================"
echo "Parallel evaluation completed!"
echo "================================"
echo "Results Summary:"
echo "  Total tasks: $TOTAL_TASKS"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: ${OUTPUT_DIR}/logs/"
echo "================================"

# 如果有失败的任务，返回错误码
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi