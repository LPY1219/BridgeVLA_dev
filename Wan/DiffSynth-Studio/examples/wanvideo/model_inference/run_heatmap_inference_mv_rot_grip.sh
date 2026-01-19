#!/bin/bash

# 自动检测根路径
if [ -d "/DATA/disk1/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH_1="/DATA/disk1/lpy/BridgeVLA_dev"
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
MACHINE1_CONDA_ENV="BridgeVLA_DM"


# 机器2配置（当前机器）
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

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

# LoRA checkpoint路径（diffusion model）
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251017_170901/epoch-56.safetensors"
LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251115_170416/epoch-10.safetensors"
# LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/from_a100_2/Wan2.2-TI2V-5B_heatmap_rgb_lora/epoch-18.safetensors"
# 旋转和夹爪预测器checkpoint路径
ROT_GRIP_CHECKPOINT="${ROOT_PATH}/logs/Wan/train/mv_rot_grip/epoch-15.pth"

# 模型基础路径
MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 模型类型
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"

# 是否使用双head模式（必须与训练时设置一致）
USE_DUAL_HEAD=true

# 旋转预测相关参数
ROTATION_RESOLUTION=5.0      # 旋转角度分辨率（度）
HIDDEN_DIM=512              # 隐藏层维度

# 自动计算旋转bins数量（360度 / 分辨率）
NUM_ROTATION_BINS=$(echo "360 / $ROTATION_RESOLUTION" | bc)

# ==============================================
# 数据集配置
# ==============================================

# 数据集路径（自动检测）
DATA_ROOT_OPTIONS=(
    "/data/Franka_data/put_the_lion_on_the_top_shelf"
    # "/data/wxn/V2W_Real/put_the_lion_on_the_top_shelf_eval"
    "/DATA/disk1/lpy/Franka_data/put_the_lion_on_the_top_shelf"
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

# 数据集参数
# SCENE_BOUNDS="0 -0.45 -0.05 0.8 0.55 0.6"  # 场景边界
# SCENE_BOUNDS="0 -0.55 -0.05 0.8 0.45 0.6"
SCENE_BOUNDS="0 -0.7 -0.05 0.8 0.7 0.65"
TRANSFORM_AUG_XYZ="0.0 0.0 0.0"            # xyz变换增强（测试时通常为0）
TRANSFORM_AUG_RPY="0.0 0.0 0.0"            # rpy变换增强（测试时通常为0）
SEQUENCE_LENGTH=12                           # 序列长度（不包括初始帧）

# 测试样本索引（逗号分隔）
# TEST_INDICES="100,200,300,400,500"
# TEST_INDICES="250,550,700,800,900"
TEST_INDICES="255,555,705,805,905"
# 输出目录
OUTPUT_DIR="${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/5B_TI2V_MV_ROT_GRIP"

# ==============================================
# GPU配置
# ==============================================
export CUDA_VISIBLE_DEVICES=3

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
echo ""
echo "Dataset Configuration:"
echo "  Data Root: $DATA_ROOT"
echo "  Scene Bounds: $SCENE_BOUNDS"
echo "  Transform Aug XYZ: $TRANSFORM_AUG_XYZ"
echo "  Transform Aug RPY: $TRANSFORM_AUG_RPY"
echo "  Sequence Length: $SEQUENCE_LENGTH"
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
    --scene_bounds $SCENE_BOUNDS
    --transform_augmentation_xyz $TRANSFORM_AUG_XYZ
    --transform_augmentation_rpy $TRANSFORM_AUG_RPY
    --sequence_length $SEQUENCE_LENGTH
    --test_indices "$TEST_INDICES"
    --rotation_resolution $ROTATION_RESOLUTION
    --hidden_dim $HIDDEN_DIM
    --num_rotation_bins $NUM_ROTATION_BINS
    --device "cuda"
)

# 添加dual head参数
if [ "$USE_DUAL_HEAD" = "true" ]; then
    PYTHON_ARGS+=(--use_dual_head)
fi

# 执行推理
python3 "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_TI2V_5B_fused_mv_rot_grip.py" "${PYTHON_ARGS[@]}"

echo ""
echo "================================"
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"
