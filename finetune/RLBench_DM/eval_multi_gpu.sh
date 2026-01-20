#!/bin/bash
# RLBench多GPU并行评估脚本
# 自动将任务分配到不同GPU上并行执行

# ==============================================
# GPU配置 - 在这里指定要使用的GPU
# ==============================================
# 可以指定多个GPU，例如: GPU_IDS=(0 1 2)
# GPU_IDS=(2 3 4 5 6 7)  # 修改这里来指定要使用的GPU
GPU_IDS=(1 2)  # 修改这里来指定要使用的GPU

# ==============================================
# CoppeliaSim和环境配置
# ==============================================
# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/mnt/data/cyx/miniconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置
MACHINE2_COPPELIASIM_ROOT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE2_DISPLAY=":1.0"
MACHINE2_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="metaworld"

# 辅助函数：从conda路径推断conda根目录并直接设置环境变量
setup_conda_env_directly() {
    local conda_path="$1"
    local env_name="$2"

    # 从conda.sh路径推断conda根目录（去掉 /etc/profile.d/conda.sh）
    local conda_base=""
    if [[ "${conda_path}" == *"/etc/profile.d/conda.sh" ]]; then
        conda_base="$(dirname "$(dirname "$(dirname "${conda_path}")")")"
    fi

    # 构建conda环境路径
    local env_path="${conda_base}/envs/${env_name}"

    # 检查环境是否存在
    if [ ! -d "${env_path}" ]; then
        echo "警告: conda环境路径不存在: ${env_path}"
        echo "尝试从conda路径激活..."
        # 如果直接路径不存在，尝试使用conda命令
        if [ -f "${conda_path}" ]; then
            source "${conda_path}" 2>/dev/null || true
            eval "$(conda shell.bash hook 2>/dev/null)" || true
            conda activate "${env_name}" 2>/dev/null || true
        fi
    else
        # 直接设置环境变量，不依赖conda activate
        export CONDA_PREFIX="${env_path}"
        export CONDA_DEFAULT_ENV="${env_name}"
        export PATH="${env_path}/bin:${PATH}"
        export CONDA_PROMPT_MODIFIER="(${env_name}) "

        # 设置其他必要的conda相关路径
        if [ -d "${conda_base}/bin" ]; then
            export PATH="${conda_base}/bin:${PATH}"
        fi

        echo "✓ 直接设置conda环境: ${env_name}"
        echo "  CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

# 通过COPPELIASIM_ROOT检测机器并设置环境
if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器1（基于COPPELIASIM_ROOT），使用配置1"
    CURRENT_MACHINE="machine1"
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE1_DISPLAY}"
    setup_conda_env_directly "${MACHINE1_CONDA_PATH}" "${MACHINE1_CONDA_ENV}"
    conda activate "${MACHINE1_CONDA_ENV}"
    echo "已设置机器1的CoppeliaSim环境变量和conda环境"
    ROOT_PATH="/mnt/data/cyx/workspace/BridgeVLA_dev"
elif [ -d "${MACHINE2_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器2（基于COPPELIASIM_ROOT），使用配置2"
    CURRENT_MACHINE="machine2"
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE2_DISPLAY}"
    setup_conda_env_directly "${MACHINE2_CONDA_PATH}" "${MACHINE2_CONDA_ENV}"
    conda activate "${MACHINE2_CONDA_ENV}"
    echo "已设置机器2的CoppeliaSim环境变量和conda环境"
    ROOT_PATH="/DATA/disk0/lpy/cyx/BridgeVLA_dev"
else
    echo "错误：未找到COPPELIASIM_ROOT路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  机器2: ${MACHINE2_COPPELIASIM_ROOT}"
    exit 1
fi

# 确保PATH正确设置（双重保险）
if [ -n "${CONDA_PREFIX}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    # 设置Python相关路径
    if [ -d "${CONDA_PREFIX}/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    fi
    if [ -d "${CONDA_PREFIX}/lib/python3.9/site-packages" ]; then
        export PYTHONPATH="${CONDA_PREFIX}/lib/python3.9/site-packages:${PYTHONPATH}"
    fi
    echo "✓ Conda环境路径已设置: ${CONDA_PREFIX}"
    echo "✓ Python路径: $(which python 2>/dev/null || echo '未找到')"
else
    echo "警告: CONDA_PREFIX未设置，conda环境可能未正确激活"
fi

# 切换到RLBench_DM目录
cd "${ROOT_PATH}/finetune/RLBench_DM"

# ==============================================
# 模型和数据配置
# ==============================================
LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/heatmap_new_2.safetensors"
ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/rot_grip_new_2.pth"

# LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/heatmap_phone.safetensors"
# ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/rotgrip_phone.pth"

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
# true: 使用多个相机拼接的点云 (front, left_shoulder, right_shoulder, wrist)
# false: 只使用相机1的点云 (front)
USE_MERGED_POINTCLOUD=true

# 自动计算旋转bins数量（360度 / 分辨率）
NUM_ROTATION_BINS=$(awk "BEGIN {printf \"%.0f\", 360 / $ROTATION_RESOLUTION}")

# 验证计算结果
if [ -z "$NUM_ROTATION_BINS" ] || [ "$NUM_ROTATION_BINS" = "0" ]; then
    echo "Error: Failed to calculate NUM_ROTATION_BINS. Using default value 72."
    NUM_ROTATION_BINS=72
fi

# ==============================================
# RLBench数据集配置
# ==============================================
# 数据集路径
EVAL_DATAFOLDER="/DATA/disk0/lpy/cyx/BridgeVLA_dev/rlbench_data"

# 场景边界 (RLBench默认: x_min y_min z_min x_max y_max z_max)
SCENE_BOUNDS="-0.3 -0.5 0.6 0.7 0.5 1.6"

# 序列长度（不包括初始帧）
SEQUENCE_LENGTH=24

# 图像尺寸 (height width)
IMG_SIZE="256 256"

# ==============================================
# 评估配置
# ==============================================
# 每个任务评估的episode数量
EVAL_EPISODES=20

# 每个episode的最大步数
EPISODE_LENGTH=25

# 起始episode索引
START_EPISODE=0

# 是否使用headless模式（无GUI）
HEADLESS=true

# ==============================================
# RLBench任务配置
# ==============================================
# 任务列表（可以添加多个任务）
TASKS=(
    # "phone_on_base"
    # "close_jar"
    # "reach_and_drag"
    # "insert_onto_square_peg"
    "meat_off_grill"
    # "open_drawer"
    # "place_cups"
    # "place_wine_at_rack_location"
    "push_buttons"
    # "put_groceries_in_cupboard"
    # "put_item_in_drawer"
    # "put_money_in_safe"
    # "light_bulb_in"
    # "slide_block_to_color_target"
    # "place_shape_in_shape_sorter"
    # "stack_blocks"
    # "stack_cups"
    # "sweep_to_dustpan_of_size"
    # "turn_tap"
)

# ==============================================
# 环境变量设置
# ==============================================
# Suppress TensorFlow and other warnings
export TF_CPP_MIN_LOG_LEVEL=3
export BITSANDBYTES_NOWELCOME=1

# ==============================================
# 打印配置信息
# ==============================================
echo "================================"
echo "RLBench Multi-GPU Parallel Evaluation"
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
echo "  Dataset Folder: $EVAL_DATAFOLDER"
echo "  Scene Bounds: $SCENE_BOUNDS"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Image Size: $IMG_SIZE"
echo ""
echo "Evaluation Configuration:"
echo "  Tasks: ${TASKS[*]}"
echo "  Total Tasks: ${#TASKS[@]}"
echo "  Episodes per task: $EVAL_EPISODES"
echo "  Episode length: $EPISODE_LENGTH"
echo "  Start episode: $START_EPISODE"
echo "  Headless mode: $HEADLESS"
echo ""
echo "Environment:"
echo "  CoppeliaSim Root: $COPPELIASIM_ROOT"
echo "  Conda Env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""
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

if [ ! -d "$EVAL_DATAFOLDER" ]; then
    echo "✗ Error: Evaluation dataset not found: $EVAL_DATAFOLDER"
    echo "Please check the path or update EVAL_DATAFOLDER variable"
    exit 1
fi

echo "✓ All checkpoint files and dataset found"
echo ""

# ==============================================
# 定义单任务执行函数
# ==============================================
run_task_on_gpu() {
    local TASK=$1
    local GPU_ID=$2
    local TASK_INDEX=$3
    local TOTAL_TASKS=$4

    # 创建任务专属的日志目录
    local LOG_DIR="${ROOT_PATH}/finetune/RLBench_DM/logs"
    mkdir -p "$LOG_DIR"
    local LOG_FILE="${LOG_DIR}/${TASK}_gpu${GPU_ID}.log"

    echo "[GPU $GPU_ID] [$TASK_INDEX/$TOTAL_TASKS] Starting task: $TASK" | tee -a "$LOG_FILE"

    # 生成日志名称
    LOG_NAME="dm_eval_${TASK}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)"

    # 构建Python命令参数
    local PYTHON_ARGS=(
        --lora_checkpoint "$LORA_CHECKPOINT"
        --rot_grip_checkpoint "$ROT_GRIP_CHECKPOINT"
        --model_base_path "$MODEL_BASE_PATH"
        --eval_datafolder "$EVAL_DATAFOLDER"
        --tasks "$TASK"
        --device 0  # 注意：这里使用0，因为CUDA_VISIBLE_DEVICES已经设置
        --eval_episodes $EVAL_EPISODES
        --episode_length $EPISODE_LENGTH
        --start_episode $START_EPISODE
        --sequence_length $SEQUENCE_LENGTH
        --img_size $IMG_SIZE
        --scene_bounds $SCENE_BOUNDS
        --wan_type "$WAN_TYPE"
        --rotation_resolution $ROTATION_RESOLUTION
        --hidden_dim $HIDDEN_DIM
        --num_rotation_bins $NUM_ROTATION_BINS
        --log_name "$LOG_NAME"
        --save_video
    )

    # 添加dual head参数
    if [ "$USE_DUAL_HEAD" = "true" ]; then
        PYTHON_ARGS+=(--use_dual_head)
    fi

    # 添加点云配置参数
    if [ "$USE_MERGED_POINTCLOUD" = "true" ]; then
        PYTHON_ARGS+=(--use_merged_pointcloud)
    fi

    # 添加headless参数
    if [ "$HEADLESS" = "true" ]; then
        PYTHON_ARGS+=(--headless)
    fi

    # 执行评估
    local TASK_START_TIME=$(date +%s)

    # 设置当前任务使用的GPU
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    if xvfb-run -a python eval.py "${PYTHON_ARGS[@]}" >> "$LOG_FILE" 2>&1; then
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
export EVAL_DATAFOLDER SCENE_BOUNDS SEQUENCE_LENGTH IMG_SIZE
export ROTATION_RESOLUTION HIDDEN_DIM NUM_ROTATION_BINS
export EVAL_EPISODES EPISODE_LENGTH START_EPISODE HEADLESS
export USE_DUAL_HEAD USE_MERGED_POINTCLOUD
export COPPELIASIM_ROOT LD_LIBRARY_PATH QT_QPA_PLATFORM_PLUGIN_PATH DISPLAY
export CONDA_PREFIX CONDA_DEFAULT_ENV PATH PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL BITSANDBYTES_NOWELCOME

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
echo "Logs saved to: ${ROOT_PATH}/finetune/RLBench_DM/logs/"
echo "================================"

# 如果有失败的任务，返回错误码
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
