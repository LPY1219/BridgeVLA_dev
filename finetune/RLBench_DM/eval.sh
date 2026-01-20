#!/bin/bash

# ==============================================
# RLBench DM Agent Evaluation Script
# ==============================================
# 自动检测根路径
# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/mnt/data/cyx/miniconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置（待填写）
MACHINE2_COPPELIASIM_ROOT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"  # TODO: 填写当前机器的CoppeliaSim路径
MACHINE2_DISPLAY=":1.0"           # TODO: 填写当前机器的DISPLAY配置
MACHINE2_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"  # TODO: 填写当前机器的conda路径
MACHINE2_CONDA_ENV="metaworld"   # TODO: 填写当前机器的conda环境名
GDK_BACKEND=x11
# 辅助函数：从conda路径推断conda根目录并直接设置环境变量（适用于tmux等场景）
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
    echo "  机器3: ${MACHINE3_COPPELIASIM_ROOT}"
    echo "  机器4: ${MACHINE4_COPPELIASIM_ROOT}"
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
# LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/heatmap_new_2.safetensors"
# ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/rot_grip_new_2.pth"

LORA_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/heatmap_phone.safetensors"
ROT_GRIP_CHECKPOINT="/DATA/disk0/lpy/cyx/BridgeVLA_dev/finetune/RLBench_DM/ckpt/rotgrip_phone.pth"

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
EVAL_EPISODES=25

# 每个episode的最大步数
EPISODE_LENGTH=25

# 起始episode索引
START_EPISODE=0

# 是否使用headless模式（无GUI）
HEADLESS=true

# ==============================================
# GPU配置
# ==============================================
export CUDA_VISIBLE_DEVICES=7
DEVICE=0  # 使用的GPU设备ID

# ==============================================
# RLBench任务配置
# ==============================================
# 任务列表（可以添加多个任务）
TASKS=(
    "phone_on_base"
    # "close_jar"
    # "reach_and_drag"
    # "insert_onto_square_peg"
    # "meat_off_grill"
    # "open_drawer"
    # "place_cups"
    # "place_wine_at_rack_location"
    # "push_buttons"
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

# 如果只想测试单个任务，可以这样设置：
# TASKS=("reach_target")

# 或者测试所有任务：
# TASKS=("all")

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
echo "RLBench DM Agent Evaluation"
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
echo "  Device: cuda:$DEVICE"
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
# 运行评估（循环测试多个任务）
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

    # 生成日志名称
    LOG_NAME="dm_eval_${TASK}_$(date +%Y%m%d_%H%M%S)"

    # 构建Python命令参数
    PYTHON_ARGS=(
        --lora_checkpoint "$LORA_CHECKPOINT"
        --rot_grip_checkpoint "$ROT_GRIP_CHECKPOINT"
        --model_base_path "$MODEL_BASE_PATH"
        --eval_datafolder "$EVAL_DATAFOLDER"
        --tasks "$TASK"
        --device $DEVICE
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
    TASK_START_TIME=$(date +%s)
    if xvfb-run -a python eval.py "${PYTHON_ARGS[@]}"; then
        TASK_END_TIME=$(date +%s)
        TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
        echo ""
        echo "✓ Task '$TASK' completed successfully (took ${TASK_DURATION}s)"
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
echo "================================"
