#!/bin/bash

# Activate conda environment
source /share/project/lpy/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /share/project/lpy/BridgeVLA/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

cd /share/project/lpy/BridgeVLA/finetune/RLBench

# Fix tokenizers parallelism warning
# export TOKENIZERS_PARALLELISM=false

# DeepSpeed configuration
GPUS_PER_NODE=8
NNODES=1
DEEPSPEED_CONFIG="configs/deepspeed_config_simple.json"

# Launch with DeepSpeed
deepspeed --num_gpus=$GPUS_PER_NODE \
          --num_nodes=$NNODES \
          train_deepspeed.py \
          --deepspeed_config $DEEPSPEED_CONFIG \
          $@