#!/bin/bash

# ==============================================
# Multi-Machine Environment Configuration
# ==============================================

# 自动检测根路径
if [ -d "/DATA/disk1/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH="/DATA/disk1/lpy/BridgeVLA_dev"
    CURRENT_MACHINE="machine1"
elif [ -d "/home/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH="/home/lpy/BridgeVLA_dev"
    CURRENT_MACHINE="machine2"
elif [ -d "/DATA/disk0/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH="/DATA/disk0/lpy/BridgeVLA_dev"
    CURRENT_MACHINE="machine3"
elif [ -d "/DATA/disk1/lpy_a100_4/BridgeVLA_dev" ]; then
    ROOT_PATH="/DATA/disk1/lpy_a100_4/BridgeVLA_dev"
    CURRENT_MACHINE="machine4"
elif [ -d "/DATA/disk1/lpy_a100_1/BridgeVLA_dev" ]; then
    ROOT_PATH="/DATA/disk1/lpy_a100_1/BridgeVLA_dev"
    CURRENT_MACHINE="machine5"
else
    echo "Error: Cannot find BridgeVLA root directory"
    exit 1
fi

echo "Detected machine: ${CURRENT_MACHINE}"
echo "ROOT_PATH: ${ROOT_PATH}"

# Conda配置
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE5_CONDA_PATH="/DATA/disk1/yaoliang/miniconda3/etc/profile.d/conda.sh"

CONDA_ENV="BridgeVLA_DM"

# 激活conda环境
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    source "${MACHINE1_CONDA_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    source "${MACHINE2_CONDA_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    source "${MACHINE3_CONDA_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    source "${MACHINE4_CONDA_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    source "${MACHINE5_CONDA_PATH}"
fi

conda activate "${CONDA_ENV}"
echo "Conda environment activated: ${CONDA_ENV}"

# ==============================================
# 路径配置
# ==============================================

# 各机器的模型路径配置
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/inference/action_decoder_with_denoising"
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_5/"

MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/inference/action_decoder_with_denoising"
MACHINE2_DATA_ROOT=""  # TODO: 填写测试数据路径

MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/inference/action_decoder_with_denoising"
MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_5/"

MACHINE4_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE4_OUTPUT_BASE="/DATA/disk1/lpy_a100_4/BridgeVLA_dev/logs/Wan/inference/action_decoder_with_denoising"
MACHINE4_DATA_ROOT="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_5/pour_filter"

MACHINE5_MODEL_BASE_PATH="/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B-fused"
MACHINE5_OUTPUT_BASE="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/inference/action_decoder_with_denoising"
MACHINE5_DATA_ROOT="/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/pour_filter"

# 根据机器设置路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    DATA_ROOT="${MACHINE2_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    DATA_ROOT="${MACHINE3_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    DATA_ROOT="${MACHINE4_DATA_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine5" ]; then
    MODEL_BASE_PATH="${MACHINE5_MODEL_BASE_PATH}"
    OUTPUT_BASE="${MACHINE5_OUTPUT_BASE}"
    DATA_ROOT="${MACHINE5_DATA_ROOT}"
fi

# ==============================================
# Checkpoint路径配置
# ==============================================

# LoRA checkpoint (视频扩散模型)
LORA_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/from_a100_4/8_trajectory_cook_5_3camera_pour_filter_pretrain_False_history_1_seq_48_new_projection_rgb_loss_1_epoch-99.safetensors"

# Action Decoder checkpoint (需要指定具体的训练好的模型)
DECODER_CHECKPOINT="/DATA/disk1/lpy_a100_1/BridgeVLA_dev/logs/Wan/train/action_decoder/action_decoder_block15_hidden512_rgb_only_diffusion_fixed_train_noise/20260107_031221/epoch-415.pth"  # TODO: 修改为实际checkpoint路径

# ==============================================
# 数据集配置
# ==============================================

TRAIL_START="1"
TRAIL_END="5"
SEQUENCE_LENGTH="48"
STEP_INTERVAL="1"
MIN_TRAIL_LENGTH="10"
HEATMAP_SIGMA="1.5"
COLORMAP_NAME="jet"
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.0,0.0,0.0"  # 测试时不使用数据增强
TRANSFORM_AUG_RPY="0.0,0.0,0.0"
USE_DIFFERENT_PROJECTION=true

# ==============================================
# 模型配置
# ==============================================

WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"
USE_DUAL_HEAD=true
EXTRACT_BLOCK_ID="15"
DIT_FEATURE_DIM="3072"
HIDDEN_DIM="512"
NUM_VIEWS="3"
NUM_ROTATION_BINS="72"
NUM_FUTURE_FRAMES="48"
DROPOUT="0.1"

# ==============================================
# 降噪推断配置 (NEW)
# ==============================================

# 降噪步数控制 (关键参数!)
# NUM_DENOISING_STEPS: 在特征提取前进行多少步降噪
#   - 0: 不降噪，直接从噪声中提取特征（最快，依赖第一帧条件）
#   - 5-10: 轻度降噪（快速推理，平衡速度和质量）
#   - 15-20: 中度降噪（推荐，较好的语义信息）
#   - 25-30: 重度降噪（高质量，但更慢）
NUM_DENOISING_STEPS="0"

# NUM_INFERENCE_STEPS: 总推理步数（调度器设置）
#   - 标准值：50（视频生成常用设置）
#   - 快速推理：25-30
#   - 高质量推理：75-100
NUM_INFERENCE_STEPS="50"

# ==============================================
# 对比测试配置 (Ground Truth Latents vs Denoising)
# ==============================================

# USE_GT_LATENTS: 是否启用对比测试模式
#   - false: 只运行 denoising 模式（首帧+噪声）
#   - true: 同时运行两种模式并对比结果：
#           1) GT latents: 使用所有帧的ground truth编码
#           2) Denoising: 使用首帧+噪声
#   启用后会对每个样本运行两次推断，耗时翻倍，但可以看到两种方法的性能差距
USE_GT_LATENTS=true

NUM_SAMPLES="10"  # 评估多少个样本（None=全部）
HEIGHT="256"
WIDTH="256"
SAVE_PREDICTIONS=true  # 是否保存每个样本的预测结果

# Device配置
DEVICE="cuda"
TORCH_DTYPE="bfloat16"

# 生成输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "${USE_GT_LATENTS}" = "true" ]; then
    OUTPUT_DIR="${OUTPUT_BASE}/comparison_gt_vs_denoise${NUM_DENOISING_STEPS}of${NUM_INFERENCE_STEPS}_${TIMESTAMP}"
else
    OUTPUT_DIR="${OUTPUT_BASE}/denoise${NUM_DENOISING_STEPS}of${NUM_INFERENCE_STEPS}_${TIMESTAMP}"
fi

echo "================================================================"
echo "ACTION DECODER INFERENCE WITH DENOISING"
echo "================================================================"
echo "Machine: ${CURRENT_MACHINE}"
echo "Model base: ${MODEL_BASE_PATH}"
echo "LoRA checkpoint: ${LORA_CHECKPOINT}"
echo "Decoder checkpoint: ${DECODER_CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "----------------------------------------------------------------"
if [ "${USE_GT_LATENTS}" = "true" ]; then
    echo "Mode: COMPARISON (GT Latents vs Denoising)"
    echo "  - GT Latents: Encode all frames from ground truth"
    echo "  - Denoising: First frame + noise"
else
    echo "Mode: Denoising only (First frame + noise)"
fi
echo "Denoising steps: ${NUM_DENOISING_STEPS} / ${NUM_INFERENCE_STEPS}"
echo "Extract block ID: ${EXTRACT_BLOCK_ID}"
echo "Num samples: ${NUM_SAMPLES}"
echo "================================================================"

# 路径验证
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "ERROR: Model base path not found: ${MODEL_BASE_PATH}"
    exit 1
fi

if [ ! -f "${LORA_CHECKPOINT}" ]; then
    echo "ERROR: LoRA checkpoint not found: ${LORA_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${DECODER_CHECKPOINT}" ]; then
    echo "ERROR: Decoder checkpoint not found: ${DECODER_CHECKPOINT}"
    echo "Please specify the correct decoder checkpoint path"
    exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: Data root not found: ${DATA_ROOT}"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
echo "✓ Output directory created"

# 切换到inference目录
cd "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference"

# ==============================================
# 运行推断（使用降噪模式）
# ==============================================

python3 inference_diffusion_action_decoder_with_denoising.py \
    --model_base_path "${MODEL_BASE_PATH}" \
    --lora_checkpoint "${LORA_CHECKPOINT}" \
    --decoder_checkpoint "${DECODER_CHECKPOINT}" \
    --data_root "${DATA_ROOT}" \
    --trail_start "${TRAIL_START}" \
    --trail_end "${TRAIL_END}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --step_interval "${STEP_INTERVAL}" \
    --min_trail_length "${MIN_TRAIL_LENGTH}" \
    --heatmap_sigma="${HEATMAP_SIGMA}" \
    --colormap_name="${COLORMAP_NAME}" \
    --scene_bounds="${SCENE_BOUNDS}" \
    --transform_augmentation_xyz="${TRANSFORM_AUG_XYZ}" \
    --transform_augmentation_rpy="${TRANSFORM_AUG_RPY}" \
    $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi) \
    --wan_type "${WAN_TYPE}" \
    $(if [ "${USE_DUAL_HEAD}" = "true" ]; then echo "--use_dual_head"; fi) \
    --extract_block_id "${EXTRACT_BLOCK_ID}" \
    --dit_feature_dim "${DIT_FEATURE_DIM}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --num_views "${NUM_VIEWS}" \
    --num_rotation_bins "${NUM_ROTATION_BINS}" \
    --num_future_frames "${NUM_FUTURE_FRAMES}" \
    --dropout "${DROPOUT}" \
    --num_denoising_steps "${NUM_DENOISING_STEPS}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    $(if [ "${USE_GT_LATENTS}" = "true" ]; then echo "--use_gt_latents"; fi) \
    --num_samples "${NUM_SAMPLES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    $(if [ "${SAVE_PREDICTIONS}" = "true" ]; then echo "--save_predictions"; fi) \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}"

echo ""
echo "================================================================"
if [ "${USE_GT_LATENTS}" = "true" ]; then
    echo "Comparison inference completed!"
    echo "Results include both GT latents and denoising modes"
else
    echo "Inference with denoising completed!"
fi
echo "Results saved to: ${OUTPUT_DIR}"
echo "================================================================"
