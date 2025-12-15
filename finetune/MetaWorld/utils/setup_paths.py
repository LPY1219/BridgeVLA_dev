from pathlib import Path
import sys



def setup_project_paths():
    """设置项目路径，确保能够导入 diffsynth 和 examples 模块"""

    # 获取当前文件的目录
    current_file_dir = Path(__file__).parent.absolute()
    
    # 计算项目根目录 (BridgeVLA_dev)
    project_root = current_file_dir.parent.parent.parent  # ../../../
    
    # 设置要添加的路径
    added_path = project_root / "Wan" / "DiffSynth-Studio"
    
    # 添加到 Python 路径
    path_str = str(added_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"Added to path: {path_str}")
    
    return project_root