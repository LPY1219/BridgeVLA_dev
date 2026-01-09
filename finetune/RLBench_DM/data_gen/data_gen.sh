#! /bin/bash

MACHINE1_COPPELIASIM_ROOT="/mnt/data/cyx/workspace/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
MACHINE2_COPPELIASIM_ROOT="/mnt/robot-rfm/user/lpy/simulation/BridgeVLA_dev/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

if [ -d "${MACHINE1_COPPELIASIM_ROOT}" ]; then
    export COPPELIASIM_ROOT="${MACHINE1_COPPELIASIM_ROOT}"
else
    export COPPELIASIM_ROOT="${MACHINE2_COPPELIASIM_ROOT}"
fi
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=":1.0"

xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac' python data_gen.py \
    --save_path /mnt/data/cyx/workspace/BridgeVLA_dev/RLBench_Data \
    --image_size 256,256 \
    --tasks close_box \
    --episodes_per_task 100 \