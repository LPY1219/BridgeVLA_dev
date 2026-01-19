# 和 mv_rot_grip_decode_feature_3.sh 的区别在于，此脚本用于支持新版本多相机数据的训练。

# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置
MACHINE2_COPPELIASIM_ROOT="/home/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE2_DISPLAY=":1.0"
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
MACHINE2_CONDA_ENV="BridgeVLA_DM"

# 机器3的CoppeliaSim配置
MACHINE3_COPPELIASIM_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE3_DISPLAY=":1.0"
MACHINE3_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE3_CONDA_ENV="BridgeVLA_DM"

# 机器4的CoppeliaSim配置
MACHINE4_COPPELIASIM_ROOT="/DATA/disk2/lpy/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE4_DISPLAY=":1.0"
MACHINE4_CONDA_PATH="/home/yw/anaconda3/etc/profile.d/conda.sh"
MACHINE4_CONDA_ENV="BridgeVLA_DM"

# 通过COPPELIASIM_ROOT检测机器并设置环境
if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器1（基于COPPELIASIM_ROOT），使用配置1"
    CURRENT_MACHINE="machine1"
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE1_DISPLAY}"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    echo "已设置机器1的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE2_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器2（基于COPPELIASIM_ROOT），使用配置2"
    CURRENT_MACHINE="machine2"
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE2_DISPLAY}"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    echo "已设置机器2的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE3_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器3（基于COPPELIASIM_ROOT），使用配置3"
    CURRENT_MACHINE="machine3"
    export COPPELIASIM_ROOT="${MACHINE3_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE3_DISPLAY}"
    source "${MACHINE3_CONDA_PATH}"
    conda activate "${MACHINE3_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
    echo "已设置机器3的CoppeliaSim环境变量和conda环境"
elif [ -d "${MACHINE4_COPPELIASIM_ROOT}" ]; then
    echo "检测到机器4（基于COPPELIASIM_ROOT），使用配置4"
    CURRENT_MACHINE="machine4"
    export COPPELIASIM_ROOT="${MACHINE4_COPPELIASIM_ROOT}"
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY="${MACHINE4_DISPLAY}"
    source "${MACHINE4_CONDA_PATH}"
    conda activate "${MACHINE4_CONDA_ENV}"
    # 修复CUDA库冲突：优先使用PyTorch自带的NVIDIA库
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}"
    echo "已设置机器4的CoppeliaSim环境变量和conda环境"
else
    echo "错误：未找到COPPELIASIM_ROOT路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_COPPELIASIM_ROOT}"
    echo "  机器2: ${MACHINE2_COPPELIASIM_ROOT}"
    echo "  机器3: ${MACHINE3_COPPELIASIM_ROOT}"
    echo "  机器4: ${MACHINE4_COPPELIASIM_ROOT}"
    exit 1
fi

# ==============================================
# 项目路径配置
# ==============================================

# 机器1的项目路径配置
MACHINE1_PROJECT_ROOT="/DATA/disk1/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器2的项目路径配置
MACHINE2_PROJECT_ROOT="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器3的项目路径配置
MACHINE3_PROJECT_ROOT="/DATA/disk0/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器4的项目路径配置
MACHINE4_PROJECT_ROOT="/DATA/disk2/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 根据机器类型设置项目路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    PROJECT_ROOT="${MACHINE1_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    PROJECT_ROOT="${MACHINE2_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    PROJECT_ROOT="${MACHINE3_PROJECT_ROOT}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    PROJECT_ROOT="${MACHINE4_PROJECT_ROOT}"
fi

# 检查项目根目录是否存在
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "错误：项目根目录不存在: ${PROJECT_ROOT}"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export SWANLAB_API_KEY="h1x6LOLp5qGLTfsPuB7Qw"

# 设置工作目录
cd "${PROJECT_ROOT}"
echo "当前工作目录: $(pwd)"

# ==============================================
# GPU配置
# ==============================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=1,2,5,6
# NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NUM_GPUS=8

# NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
# export CUDA_VISIBLE_DEVICES=0
# NUM_GPUS=1
echo "使用GPU: ${CUDA_VISIBLE_DEVICES}"
echo "GPU数量: ${NUM_GPUS}"

# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
# MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed/put_lion_on_top_shelf"
MACHINE1_DATA_ROOT="/DATA/disk1/lpy/Franka_data_3zed_keypoint/put_red_bull_in_pink_plate"
MACHINE1_OUTPUT_BASE="/DATA/disk1/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/DATA/disk1/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器2的路径配置
MACHINE2_DATA_ROOT="/data/Franka_data_3zed_keypoint/put_lion_on_top_shelf"
MACHINE2_OUTPUT_BASE="/home/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE2_MODEL_BASE_PATH="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器3的路径配置
MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed_keypoint/put_red_bull_in_pink_plate"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器4的路径配置
MACHINE4_DATA_ROOT="/DATA/disk2/lpy/data/Franka_data_3zed_2_keypoint/put_red_bull_in_pink_plate"
MACHINE4_OUTPUT_BASE="/DATA/disk2/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE4_MODEL_BASE_PATH="/DATA/disk2/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 根据机器类型设置路径
if [ "${CURRENT_MACHINE}" = "machine1" ]; then
    DATA_ROOT="${MACHINE1_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE1_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE1_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine2" ]; then
    DATA_ROOT="${MACHINE2_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE2_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE2_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine3" ]; then
    DATA_ROOT="${MACHINE3_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE3_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE3_MODEL_BASE_PATH}"
elif [ "${CURRENT_MACHINE}" = "machine4" ]; then
    DATA_ROOT="${MACHINE4_DATA_ROOT}"
    OUTPUT_BASE="${MACHINE4_OUTPUT_BASE}"
    MODEL_BASE_PATH="${MACHINE4_MODEL_BASE_PATH}"
fi

# 多任务训练示例（用户可以修改为多个任务路径，用空格分隔）
# DATA_ROOT="/data/Franka_data/task1 /data/Franka_data/task2 /data/Franka_data/task3 /data/Franka_data/task4"

# Trail范围过滤（可选，默认使用所有trails）
TRAIL_START="1"          # 起始trail编号，如1表示从trail_1开始。留空表示不限制
TRAIL_END=""            # 结束trail编号，如50表示到trail_50结束。留空表示不限制
# 示例：只使用trail_1到trail_50
# TRAIL_START=1
# TRAIL_END=50

# 输出路径（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NAME="1_9_8gpu_no_cache_1zed_put_plate"
OUTPUT_PATH="${OUTPUT_BASE}/mv_rot_grip_v2/${NAME}/${TIMESTAMP}"

# ==============================================
# 训练参数配置
# ==============================================

# 数据参数
SEQUENCE_LENGTH=4           # 序列长度（不包括初始帧）
IMAGE_SIZE=256              # 图像尺寸
NUM_WORKERS=0               # DataLoader工作线程数（必须为0，因为collate_fn中使用CUDA）

# 数据集参数
# SCENE_BOUNDS="0,-0.7,-0.05,0.8,0.7,0.65"  # 场景边界 [x_min, y_min, z_min, x_max, y_max,z_max]
# SCENE_BOUNDS="0,-0.55,-0.05,0.8,0.45,0.6"
SCENE_BOUNDS="-0.1,-0.5,-0.1,0.9,0.5,0.9"
TRANSFORM_AUG_XYZ="0.1 0.1 0.1"
TRANSFORM_AUG_RPY="10.0 10.0 20.0"  
WAN_TYPE="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP" # Wan模型类型
DEBUG=false                                  # 调试模式

# 模型参数
HIDDEN_DIM=512              # 隐藏层维度
NUM_ROTATION_BINS=72        # 旋转bins数量（对应5度分辨率）
DROPOUT=0.1                 # Dropout率

# 局部特征提取参数（V2新增）
USE_ACCURATE_PEAK_DETECTION=true  # 使用精确的峰值检测（从colormap提取heatmap）
LOCAL_FEATURE_SIZE=5              # 局部特征提取窗口大小

# 点云配置
# true: 使用三个相机拼接的点云
# false: 只使用相机1的点云
USE_MERGED_POINTCLOUD=false

# 投影模式配置
# true: 使用不同的投影方式（base_multi_view_dataset_with_rot_grip_3cam_different_projection.py）
# false: 使用默认投影方式（base_multi_view_dataset_with_rot_grip_3cam.py）
USE_DIFFERENT_PROJECTION=false

# 训练参数
NUM_EPOCHS=1500               # 训练轮数
LEARNING_RATE=5e-5          # 学习率
WEIGHT_DECAY=1e-5           # 权重衰减
MAX_GRAD_NORM=1.0           # 最大梯度范数
GRADIENT_ACCUMULATION_STEPS=2  # 梯度累积步数
BATCH_SIZE=1                # 每个GPU的batch size（根据显存调整）

# 保存和日志参数
SAVE_EPOCH_INTERVAL=100        # 每N个epoch保存一次checkpoint
LOGGING_STEPS=10            # 日志记录步数
SWANLAB_PROJECT="mv_rot_grip"
SWANLAB_EXPERIMENT="local_feat_size_${LOCAL_FEATURE_SIZE}_seq_${SEQUENCE_LENGTH}_3zed"

# VAE参数
HEATMAP_LATENT_SCALE=1.0    # Heatmap latent缩放因子
LATENT_NOISE_STD=0.1        # Latent噪声标准差（用于训练时数据增强，提升鲁棒性）

# 编码模式选择
USE_ONLINE_ENCODING=true   # 是否使用在线编码（true=在线编码，false=使用预编码缓存）
                            # 在线编码: 慢2-3倍，但无需预先缓存，适合小规模实验
                            # 预编码缓存: 快2-3倍，但需要磁盘空间，适合大规模训练

# 缓存参数（仅在USE_ONLINE_ENCODING=false时使用）
NUM_AUGMENTATIONS=1        # 每个样本预编码的增强版本数量（保持数据增强多样性）

echo "================================================================"
echo "Multi-View Rotation and Gripper Prediction Training (V2)"
echo "================================================================"
echo "当前使用机器: ${CURRENT_MACHINE}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据根目录: ${DATA_ROOT}"
echo "模型路径: ${MODEL_BASE_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"
echo "训练参数:"
echo "  序列长度: ${SEQUENCE_LENGTH}"
echo "  图像尺寸: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  隐藏层维度: ${HIDDEN_DIM}"
echo "  旋转Bins: ${NUM_ROTATION_BINS}"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  GPU数量: ${NUM_GPUS}"
echo "  点云融合模式: ${USE_MERGED_POINTCLOUD}"
echo "  不同投影模式: ${USE_DIFFERENT_PROJECTION}"
echo "----------------------------------------------------------------"
echo "局部特征参数 (V2新增):"
echo "  精确峰值检测: ${USE_ACCURATE_PEAK_DETECTION}"
echo "  局部特征窗口: ${LOCAL_FEATURE_SIZE}x${LOCAL_FEATURE_SIZE}"
echo "----------------------------------------------------------------"
echo "VAE编码模式:"
if [ "${USE_ONLINE_ENCODING}" = "true" ]; then
    echo "  模式: 在线编码 (slower, no cache needed)"
else
    echo "  模式: 预编码缓存 (faster, requires disk space)"
    echo "  增强版本数: ${NUM_AUGMENTATIONS}"
fi
echo "================================================================"

# ==============================================
# 路径验证
# ==============================================

# 检查数据目录
if [ ! -d "${DATA_ROOT}" ]; then
    echo "错误：数据目录不存在: ${DATA_ROOT}"
    echo "请检查数据路径并重试。"
    exit 1
fi

# 检查模型目录
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "错误：模型目录不存在: ${MODEL_BASE_PATH}"
    echo "请检查模型路径并重试。"
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"
echo "输出目录已创建: ${OUTPUT_PATH}"

# ==============================================
# 启动训练
# ==============================================

echo "================================================================"
echo "STARTING TRAINING"
echo "================================================================"

# 使用accelerate launch进行多卡训练
# 主要修改了加载的dataset，所以没有单独新开一个文件
accelerate launch --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    examples/wanvideo/model_training/mv_rot_grip_vae_decode_feature_3.py \
    --data_root ${DATA_ROOT} \
    --output_path "${OUTPUT_PATH}" \
    $(if [ -n "${TRAIL_START}" ]; then echo "--trail_start ${TRAIL_START}"; fi) \
    $(if [ -n "${TRAIL_END}" ]; then echo "--trail_end ${TRAIL_END}"; fi) \
    --sequence_length ${SEQUENCE_LENGTH} \
    --image_size ${IMAGE_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --scene_bounds="${SCENE_BOUNDS}" \
    --transform_augmentation_xyz ${TRANSFORM_AUG_XYZ} \
    --transform_augmentation_rpy ${TRANSFORM_AUG_RPY} \
    --wan_type "${WAN_TYPE}" \
    $(if [ "${DEBUG}" = "true" ]; then echo "--debug"; fi) \
    --hidden_dim ${HIDDEN_DIM} \
    --num_rotation_bins ${NUM_ROTATION_BINS} \
    --dropout "${DROPOUT}" \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --save_epoch_interval ${SAVE_EPOCH_INTERVAL} \
    --logging_steps ${LOGGING_STEPS} \
    --swanlab_project ${SWANLAB_PROJECT} \
    --swanlab_experiment ${SWANLAB_EXPERIMENT} \
    --model_base_path "${MODEL_BASE_PATH}" \
    --heatmap_latent_scale "${HEATMAP_LATENT_SCALE}" \
    --latent_noise_std "${LATENT_NOISE_STD}" \
    --num_augmentations ${NUM_AUGMENTATIONS} \
    $(if [ "${USE_ACCURATE_PEAK_DETECTION}" = "true" ]; then echo "--use_accurate_peak_detection"; fi) \
    --local_feature_size ${LOCAL_FEATURE_SIZE} \
    $(if [ "${USE_ONLINE_ENCODING}" = "true" ]; then echo "--use_online_encoding"; fi) \
    $(if [ "${USE_MERGED_POINTCLOUD}" = "true" ]; then echo "--use_merged_pointcloud"; fi) \
    $(if [ "${USE_DIFFERENT_PROJECTION}" = "true" ]; then echo "--use_different_projection"; fi)

echo "================================================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"
