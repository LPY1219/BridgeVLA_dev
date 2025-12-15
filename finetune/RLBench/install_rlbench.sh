pip install wheel ninja pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /data/lpy/BridgeVLA_dev/finetune/bridgevla/libs
git clone git@github.com:buttomnutstoast/RLBench.git
cd RLBench
git checkout 587a6a0e6dc8cd36612a208724eb275fe8cb4470
cd ..
git clone git@github.com:stepjam/PyRep.git
cd PyRep
git checkout 231a1ac6b0a179cff53c1d403d379260b9f05f2f
cd  /share/project/lpy/BridgeVLA/finetune
# wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
# tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
# export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
pip3 install pip==25.0.1
pip3 install setuptools==76.1.0
pip3 install open3d

cd  /share/project/lpy/BridgeVLA/finetune
pip install -e .
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121 #
pip install torchaudio==2.5.1 torchvision==0.20.1
pip install 'accelerate>=0.26.0'
pip3 install transformers==4.51.3
pip install git+https://github.com/openai/CLIP.git
apt-get update
apt-get install -y libffi-dev
apt-get install -y xvfb
apt-get install -y libfontconfig1
apt install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
apt install libxcb-cursor0
apt install libxcb-xinerama0
pip install pyqt6
pip3 install yacs
pip3 install wandb
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
# pip install 'git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable'

apt-get update
apt-get install -y  libxcb-xinput0  libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 libxcb-shape0 libxcb-randr0 libxcb-image0 libxcb-keysyms1 libxcb-icccm4 libxcb-sync1 libxcb-xinerama0 libxcb-util1
apt-get install -y libxcb-glx0 libxcb-xkb1 libxkbcommon-x11-0
apt install -y ffmpeg
pip install ffmpeg-python


pip uninstall -y opencv-python opencv-contrib-python
pip install  opencv-python-headless        
pip uninstall  -y opencv-python-headless      
pip install  opencv-python-headless   

pip install -e bridgevla/libs/PyRep 
pip install -e bridgevla/libs/RLBench 
pip install -e bridgevla/libs/YARR 
pip install -e bridgevla/libs/peract_colab
pip install -e bridgevla/libs/point-renderer    

