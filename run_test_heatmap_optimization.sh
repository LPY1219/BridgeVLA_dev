#!/bin/bash

# 激活conda环境
CONDA_PATH="/home/lpy/anaconda3/etc/profile.d/conda.sh"
if [ -f "${CONDA_PATH}" ]; then
    source "${CONDA_PATH}"
    conda activate BridgeVLA_DM
    echo "Activated conda environment: BridgeVLA_DM"
else
    echo "Warning: Could not find conda at ${CONDA_PATH}"
fi

# 运行测试
cd /home/lpy/BridgeVLA_dev
python3 test_heatmap_optimization.py
