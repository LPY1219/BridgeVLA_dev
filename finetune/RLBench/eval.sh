cd /share/project/lpy/BridgeVLA/finetune
source /share/project/lpy/miniconda3/etc/profile.d/conda.sh
conda activate test

export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd RLBench

# Auto-convert DeepSpeed checkpoints if needed
echo "Checking for DeepSpeed checkpoints that need conversion..."
MODEL_FOLDER="/share/project/lpy/BridgeVLA/finetune/RLBench/logs/train/debug/20_local_points/08_25_18_16"

# You can specify the target epoch as the first argument, default to epoch 0
TARGET_EPOCH=99

if [ -d "$MODEL_FOLDER" ]; then
    echo "Running auto-conversion for epoch $TARGET_EPOCH in: $MODEL_FOLDER"
    python3 auto_convert_checkpoints.py "$MODEL_FOLDER" "$TARGET_EPOCH"
    if [ $? -eq 0 ]; then
        echo "✓ Checkpoint conversion completed successfully"
    else
        echo "✗ Checkpoint conversion failed"
        exit 1
    fi
else
    echo "Model folder not found: $MODEL_FOLDER"
    exit 1
fi


# apt-get install libxcb-xinerama0
# apt-get install libxcb1 libxcb-render0 libxcb-shm0
# apt-get install libx11-xcb1
# apt-get install libgl1-mesa-glx libgl1-mesa-dri
# apt-get install mesa-utils
# apt-get install qt5-qmake qtbase5-dev qtchooser qtbase5-dev-tools qt5-qmake

# pip uninstall -y opencv-python opencv-contrib-python
# pip install  opencv-python-headless  
# pip uninstall  -y opencv-python-headless      
# pip install  opencv-python-headless   # in my machine , i have to repeat the installation process to avoid the error: "Could not find the Qt platform plugin 'xcb'"   
# xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac'  python3 eval.py --model-folder $MODEL_FOLDER --eval-datafolder   /share/project/lpy/BridgeVLA/data/RLBench/raw_data \
#  --tasks "place_shape_in_shape_sorter"  --eval-episodes 25 --log-name "debug" --device 0 --headless --model-name "model_${TARGET_EPOCH}.pth" 
