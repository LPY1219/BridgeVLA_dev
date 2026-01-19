#!/bin/bash

# 自动检测根路径
if [ -d "/share/project/lpy/BridgeVLA" ]; then
    ROOT_PATH="/share/project/lpy/BridgeVLA"
elif [ -d "/home/lpy/BridgeVLA_dev" ]; then
    ROOT_PATH="/home/lpy/BridgeVLA_dev"
else
    echo "Error: Cannot find BridgeVLA root directory"
    exit 1
fi

# 机器1配置（原始机器）
MACHINE1_CONDA_PATH="/share/project/lpy/miniconda3/etc/profile.d/conda.sh"
MACHINE1_CONDA_ENV="test"

# 机器2配置（当前机器 - 待填写）
MACHINE2_CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"  # TODO: 填写当前机器的conda路径
MACHINE2_CONDA_ENV="BridgeVLA_DM"   # TODO: 填写当前机器的conda环境名

# 检测机器并设置conda环境
if [ -f "${MACHINE1_CONDA_PATH}" ]; then
    echo "检测到机器1，使用配置1"
    source "${MACHINE1_CONDA_PATH}"
    conda activate "${MACHINE1_CONDA_ENV}"
    CURRENT_MACHINE="machine1"
elif [ -f "${MACHINE2_CONDA_PATH}" ]; then
    echo "检测到机器2，使用配置2"
    source "${MACHINE2_CONDA_PATH}"
    conda activate "${MACHINE2_CONDA_ENV}"
    CURRENT_MACHINE="machine2"
else
    echo "错误：未找到conda配置路径"
    echo "请检查以下路径是否存在："
    echo "  机器1: ${MACHINE1_CONDA_PATH}"
    echo "  机器2: ${MACHINE2_CONDA_PATH}"
    exit 1
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
# Dual Head模式配置
# ==============================================
# 设置为true启用双head模式，必须与训练时的设置一致
USE_DUAL_HEAD=true

echo "================================"
echo "Running heatmap inference..."
echo "================================"
echo "Dual Head Mode: $USE_DUAL_HEAD"
echo "================================"


export CUDA_VISIBLE_DEVICES=0
# 运行Python脚本
# python "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference.py"
# python "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_14BI2V_480P.py"

# 传递dual head参数到Python脚本
if [ "$USE_DUAL_HEAD" = "true" ]; then
    python3 "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_TI2V_5B_fused_mv.py" --use_dual_head
else
    python3 "${ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_TI2V_5B_fused_mv.py"
fi




# python /home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/test_vae_channel_stats.py