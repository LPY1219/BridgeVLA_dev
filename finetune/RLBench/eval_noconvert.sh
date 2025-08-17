
cd /data/lpy/BridgeVLA_dev/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd RLBench

MODEL_FOLDER=/data/lpy/BridgeVLA_dev/finetune/RLBench/logs/ckpts
TARGET_EPOCH=60
# apt-get install libxcb-xinerama0
# apt-get install libxcb1 libxcb-render0 libxcb-shm0
# apt-get install libx11-xcb1
# apt-get install libgl1-mesa-glx libgl1-mesa-dri
# apt-get install mesa-utils
# apt-get install qt5-qmake qtbase5-dev qtchooser qtbase5-dev-tools qt5-qmake


# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org opencv-python-headless
# pip uninstall -y opencv-python opencv-contrib-python
# pip install  opencv-python-headless  
# pip uninstall  -y opencv-python-headless      
# pip install  opencv-python-headless   # in my machine , i have to repeat the installation process to avoid the error: "Could not find the Qt platform plugin 'xcb'"   
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac'  python3 eval.py --model-folder $MODEL_FOLDER --eval-datafolder   /data/lpy/BridgeVLA_dev/finetune/data/RLBench/eval_data \
 --tasks "place_shape_in_shape_sorter"  --eval-episodes 25 --log-name "debug" --device 2 --headless --model-name "model_${TARGET_EPOCH}.pth" 
