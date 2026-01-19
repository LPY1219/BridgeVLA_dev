# 和 mv_rot_grip_decode_feature_3.sh 的区别在于，此脚本用于支持新版本多相机数据的训练。

# 机器1的CoppeliaSim配置
MACHINE1_COPPELIASIM_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE1_DISPLAY=":1.0"
MACHINE1_CONDA_PATH="/mnt/data/cyx/miniconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="BridgeVLA_DM"

# 机器2的CoppeliaSim配置（待填写）
MACHINE2_COPPELIASIM_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"  # TODO: 填写当前机器的CoppeliaSim路径
MACHINE2_DISPLAY=":1.0"           # TODO: 填写当前机器的DISPLAY配置
MACHINE2_CONDA_PATH="/root/miniconda3/etc/profile.d/conda.sh"  # TODO: 填写当前机器的conda路径
MACHINE2_CONDA_ENV="metaworld"   # TODO: 填写当前机器的conda环境名

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
MACHINE1_PROJECT_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/Wan/DiffSynth-Studio"

# 机器2的项目路径配置
MACHINE2_PROJECT_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/Wan/DiffSynth-Studio"         # TODO: 填写当前机器的项目根目录


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
export SWANLAB_API_KEY="8iNh1p4ePH7cbYlN9rC0j"

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
# NUM_GPUS=24

# NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# NUM_GPUS=7
export CUDA_VISIBLE_DEVICES=3
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1
# NUM_GPUS=2
# NUM_GPUS=64
echo "使用GPU: ${CUDA_VISIBLE_DEVICES}"
echo "GPU数量: ${NUM_GPUS}"

# 检查环境变量，如果存在则自动配置多机模式
MULTI_NODE_AUTO_CONFIG=false
if [ -n "${WORLD_SIZE}" ] && [ -n "${RANK}" ] && [ -n "${MASTER_ADDR}" ] && [ -n "${MASTER_PORT}" ]; then
    # 从环境变量自动获取多机配置
    NUM_MACHINES=${WORLD_SIZE}
    MACHINE_RANK=${RANK}
    MAIN_PROCESS_IP="${MASTER_ADDR}"
    MAIN_PROCESS_PORT=${MASTER_PORT}
    MULTI_NODE_AUTO_CONFIG=true
else
    # 单机训练配置（默认）
    NUM_MACHINES=1
    MACHINE_RANK=0
    MAIN_PROCESS_IP=""
    MAIN_PROCESS_PORT=29500
    
    # 如果需要手动配置多机训练，取消注释并修改以下参数
    # NUM_MACHINES=2                    # 总机器数
    # MACHINE_RANK=0                    # 当前机器rank（主节点为0，其他节点为1,2,3...）
    # MAIN_PROCESS_IP="192.168.1.100"  # 主节点IP地址（所有节点都需要能访问）
    # MAIN_PROCESS_PORT=29500           # 通信端口（确保防火墙允许）
fi
# 注意：
# 1. NUM_GPUS 应该是当前机器的GPU数量，不是所有机器的总和
# 2. 多机训练时，所有机器上的 NUM_MACHINES、MAIN_PROCESS_IP、MAIN_PROCESS_PORT 必须相同
# 3. 每个机器上的 MACHINE_RANK 必须不同（0, 1, 2, ...）
# 4. 所有机器需要能互相访问，建议配置SSH免密登录
# 5. 所有机器上的代码、数据路径、环境应该保持一致
# 6. 云服务器通常会自动设置 WORLD_SIZE、RANK、MASTER_ADDR、MASTER_PORT 环境变量


# ==============================================
# 数据和模型路径配置
# ==============================================

# 机器1的路径配置
MACHINE1_DATA_ROOT_LIST=(
"/mnt/workspace/cyx/datasets/RLBench/data/train/reach_and_drag"
)
MACHINE1_DATA_ROOT="${MACHINE1_DATA_ROOT_LIST[*]}"
MACHINE1_OUTPUT_BASE="/mnt/data/cyx/workspace/BridgeVLA_dev/logs/Wan/train"
MACHINE1_MODEL_BASE_PATH="/mnt/data/cyx/huggingface/Wan2.2-TI2V-5B-fused"

# 机器2的路径配置（待填写）
# MACHINE2_HEATMAP_DATA_ROOT="/data/Franka_data_3zed/put_lion_on_top_shelf"    # TODO: 填写当前机器的热力图数据根目录
MACHINE2_DATA_ROOT_LIST=(
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/reach_and_drag"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/close_jar"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/insert_onto_square_peg"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/light_bulb_in"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/meat_off_grill"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/open_drawer"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/place_cups"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/place_shape_in_shape_sorter"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/place_wine_at_rack_location"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/push_buttons"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/put_groceries_in_cupboard"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/put_item_in_drawer"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/put_money_in_safe"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/slide_block_to_color_target"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/stack_blocks"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/stack_cups"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/sweep_to_dustpan_of_size"
# "/mnt/robot-rfm/user/lpy/data/RLBench/train/turn_tap"
"/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev_backup/RLBench_Data/phone_on_base"
)
MACHINE2_DATA_ROOT="${MACHINE2_DATA_ROOT_LIST[*]}"
MACHINE2_OUTPUT_BASE="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/logs/Wan/train"          # TODO: 填写当前机器的输出基础目录
MACHINE2_MODEL_BASE_PATH="/mnt/robot-rfm/user/lpy/huggingface/Wan2.2-TI2V-5B-fused"      # TODO: 填写当前机器的模型基础路径

# 机器3的路径配置
MACHINE3_DATA_ROOT="/DATA/disk0/lpy/data/Franka_data_3zed/put_red_bull_in_pink_plate"
MACHINE3_OUTPUT_BASE="/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train"
MACHINE3_MODEL_BASE_PATH="/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"

# 机器4的路径配置
MACHINE4_DATA_ROOT="/DATA/disk2/lpy/data/Franka_data_3zed/put_red_bull_in_pink_plate"
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
TRAIL_START=""          # 起始trail编号，如1表示从trail_1开始。留空表示不限制
TRAIL_END="1"            # 结束trail编号，如50表示到trail_50结束。留空表示不限制
# 示例：只使用trail_1到trail_50
# TRAIL_START=1
# TRAIL_END=50

# 输出路径（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NAME="rlbench_phone"
OUTPUT_PATH="${OUTPUT_BASE}/mv_rot_grip_v2/${NAME}/${TIMESTAMP}"

# ==============================================
# 训练参数配置
# ==============================================

# 数据参数
SEQUENCE_LENGTH=24           # 序列长度（不包括初始帧）
IMAGE_SIZE=256              # 图像尺寸
NUM_WORKERS=0               # DataLoader工作线程数（必须为0，因为collate_fn中使用CUDA）

# 数据集参数
# SCENE_BOUNDS="0,-0.7,-0.05,0.8,0.7,0.65"  # 场景边界 [x_min, y_min, z_min, x_max, y_max,z_max]
# SCENE_BOUNDS="0,-0.55,-0.05,0.8,0.45,0.6"
SCENE_BOUNDS="-0.3,-0.5,0.6,0.7,0.5,1.6"
TRANSFORM_AUG_XYZ="0.1 0.1 0.1"
TRANSFORM_AUG_RPY="0.0 0.0 20.0"  
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
USE_MERGED_POINTCLOUD=true

# 投影模式配置
# true: 使用不同的投影方式（base_multi_view_dataset_with_rot_grip_3cam_different_projection.py）
# false: 使用默认投影方式（base_multi_view_dataset_with_rot_grip_3cam.py）
USE_DIFFERENT_PROJECTION=false

# 训练参数
NUM_EPOCHS=100               # 训练轮数
LEARNING_RATE=1e-4          # 学习率
WEIGHT_DECAY=1e-5           # 权重衰减
MAX_GRAD_NORM=1.0           # 最大梯度范数
GRADIENT_ACCUMULATION_STEPS=1 # 梯度累积步数
BATCH_SIZE=1                # 每个GPU的batch size（根据显存调整）

# 保存和日志参数
SAVE_EPOCH_INTERVAL=10        # 每N个epoch保存一次checkpoint
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

# 历史帧配置 - 控制输入条件帧数量
# 历史帧按照VAE原始方式编码：第一帧单独编码，后续每4帧一组
#
# 允许的历史帧数量：
#   - 1: 单帧条件（向后兼容）→ 1个条件latent
#   - 2: 两帧分别单独编码 → 2个条件latent
#   - 1+4N (5,9,13...): 第一帧单独 + 后续每4帧一组 → (1+N)个条件latent
#
# 示例：
#   NUM_HISTORY_FRAMES=1  → num_condition_latents=1
#   NUM_HISTORY_FRAMES=2  → num_condition_latents=2  (每帧单独编码)
#   NUM_HISTORY_FRAMES=5  → num_condition_latents=2  (1帧 + 4帧)
#   NUM_HISTORY_FRAMES=9  → num_condition_latents=3  (1帧 + 4帧 + 4帧)
#
# ⚠️ 重要：当 NUM_HISTORY_FRAMES>1 时，必须将 WAN_TYPE 设置为 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY
NUM_HISTORY_FRAMES=1     # 历史帧数量，必须为 1, 2, 或 1+4N (5,9,13...)

# 验证 NUM_HISTORY_FRAMES 的合法性
# 必须是 1, 2, 或 1+4N (即 (n-1) % 4 == 0 当 n > 2)
is_valid_history_frames=false
if [ ${NUM_HISTORY_FRAMES} -eq 1 ]; then
    is_valid_history_frames=true
elif [ ${NUM_HISTORY_FRAMES} -eq 2 ]; then
    is_valid_history_frames=true
elif [ ${NUM_HISTORY_FRAMES} -gt 2 ]; then
    # 检查是否为 1+4N 形式
    remainder=$(( (${NUM_HISTORY_FRAMES} - 1) % 4 ))
    if [ ${remainder} -eq 0 ]; then
        is_valid_history_frames=true
    fi
fi

if [ "${is_valid_history_frames}" != "true" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} is invalid!"
    echo "       Allowed values: 1, 2, or 1+4N (5, 9, 13, 17, ...)"
    echo "       This ensures proper VAE encoding: first frame alone, then groups of 4"
    exit 1
fi

# 验证 WAN_TYPE 和 NUM_HISTORY_FRAMES 的一致性（双向检测）
# 规则1: NUM_HISTORY_FRAMES > 1 必须使用 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY
if [ ${NUM_HISTORY_FRAMES} -gt 1 ] && [ "${WAN_TYPE}" != "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY" ]; then
    echo "ERROR: NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES} > 1, but WAN_TYPE=${WAN_TYPE}"
    echo "       When using multi-frame history (NUM_HISTORY_FRAMES > 1), you MUST set:"
    echo "       WAN_TYPE=\"5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY\""
    exit 1
fi
# 规则2: 使用 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY 必须设置 NUM_HISTORY_FRAMES > 1
if [ ${NUM_HISTORY_FRAMES} -eq 1 ] && [ "${WAN_TYPE}" == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY" ]; then
    echo "ERROR: WAN_TYPE=5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, but NUM_HISTORY_FRAMES=${NUM_HISTORY_FRAMES}"
    echo "       When using 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, you MUST set NUM_HISTORY_FRAMES > 1"
    echo "       If you want single-frame mode, use WAN_TYPE=\"5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP\""
    exit 1
fi
pip install --force-reinstall --no-deps swanlab

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
missing_paths=()
IFS=' ' read -ra __paths <<< "${DATA_ROOT}"
for p in "${__paths[@]}"; do
    if [ ! -d "${p}" ]; then
        missing_paths+=("${p}")
    fi
done
if [ "${#missing_paths[@]}" -ne 0 ]; then
    echo "错误：以下数据目录不存在:"
    for mp in "${missing_paths[@]}"; do
        echo "  ${mp}"
    done
    echo "请检查数据路径并重试。"
    exit 1
fi

# 检查模型目录
if [ ! -d "${MODEL_BASE_PATH}" ]; then
    echo "错误：模型目录不存在: ${MODEL_BASE_PATH}"
    echo "请检查模型路径并重试。"
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
# 多机多卡训练时，需要添加额外的参数
if [ "${NUM_MACHINES}" -gt 1 ]; then
    echo "启动多机多卡训练..."
    echo "  机器总数: ${NUM_MACHINES}"
    echo "  当前机器rank: ${MACHINE_RANK}"
    echo "  主节点IP: ${MAIN_PROCESS_IP}"
    echo "  通信端口: ${MAIN_PROCESS_PORT}"
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --num_machines=${NUM_MACHINES} \
        --machine_rank=${MACHINE_RANK} \
        --main_process_ip="${MAIN_PROCESS_IP}" \
        --main_process_port=${MAIN_PROCESS_PORT} \
        --mixed_precision=bf16 \
        examples/wanvideo/model_training/mv_rot_grip_vae_decode_feature_3_view_rlbench.py \
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
else
    echo "启动单机多卡训练..."
    accelerate launch --num_processes=${NUM_GPUS} \
        --mixed_precision=bf16 \
        examples/wanvideo/model_training/mv_rot_grip_vae_decode_feature_3_view_rlbench.py \
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
fi

echo "================================================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "================================================================"