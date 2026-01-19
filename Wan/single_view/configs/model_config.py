"""
模型配置文件
包含Wan2.2模型路径、输入尺寸、序列长度等配置参数
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class ModelConfig:
    """
    模型相关配置参数
    """
    # Wan2.2模型路径
    wan_model_path: str = "/share/project/lpy/huggingface/Wan_2_2_TI2V_5B_Diffusers"

    # 模型基本参数
    input_image_size: Tuple[int, int] = (256, 256)  # 输入RGB图像尺寸 (H, W)
    output_image_size: Tuple[int, int] = (256, 256)  # 输出heatmap尺寸 (H, W)
    sequence_length: int = 10  # 预测的heatmap序列长度

    # VAE相关配置
    vae_subfolder: str = "vae"
    vae_torch_dtype: str = "float32"  # "float16" or "float32"

    # Colormap编码配置
    colormap_name: str = "jet"  # heatmap转colormap使用的颜色映射
    colormap_resolution: int = 256  # colormap分辨率

    # 推理相关配置
    num_inference_steps: int = 50  # DDIM采样步数
    guidance_scale: float = 7.5  # CFG引导尺度

    # 内存和性能优化
    enable_attention_slicing: bool = True  # 启用注意力切片以节省内存
    enable_cpu_offload: bool = False  # 启用CPU卸载
    enable_model_cpu_offload: bool = False  # 启用模型CPU卸载
    use_torch_compile: bool = False  # 是否使用torch.compile优化

    # 设备配置
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = True  # 混合精度训练

    # 模型检查点
    checkpoint_path: Optional[str] = None  # 预训练检查点路径

    def __post_init__(self):
        """后处理验证配置参数"""
        # 验证模型路径
        if not os.path.exists(self.wan_model_path):
            print(f"Warning: Wan model path does not exist: {self.wan_model_path}")

        # 自动设备检测
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 验证图像尺寸
        assert len(self.input_image_size) == 2, "input_image_size must be (H, W)"
        assert len(self.output_image_size) == 2, "output_image_size must be (H, W)"
        assert self.sequence_length > 0, "sequence_length must be positive"

        # 验证dtype
        assert self.vae_torch_dtype in ["float16", "float32"], "Invalid torch dtype"

        # 验证colormap
        import matplotlib.pyplot as plt
        try:
            plt.get_cmap(self.colormap_name)
        except ValueError:
            raise ValueError(f"Invalid colormap name: {self.colormap_name}")

    def get_torch_dtype(self):
        """获取torch数据类型"""
        import torch
        if self.vae_torch_dtype == "float16":
            return torch.float16
        else:
            return torch.float32

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        import torch
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置对象"""
        return cls(**config_dict)

    def save(self, path: str):
        """保存配置到文件"""
        import json
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Model config saved to: {path}")

    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DataConfig:
    """
    数据相关配置参数
    """
    # 数据集路径
    train_data_root: str = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"
    val_data_root: Optional[str] = None  # 如果为None，从训练集分割

    # 数据集参数
    sequence_length: int = 10  # 序列长度
    step_interval: int = 1  # step采样间隔
    min_trail_length: int = 15  # 最小轨迹长度
    image_size: Tuple[int, int] = (256, 256)  # 图像尺寸
    sigma: float = 1.5  # heatmap高斯分布标准差

    # 数据加载参数
    batch_size: int = 8
    num_workers: int = 4
    shuffle_train: bool = True
    pin_memory: bool = False  # 设为False避免CUDA张量pin_memory冲突
    drop_last: bool = True
    val_split: float = 0.2  # 验证集分割比例

    # 数据增强
    augmentation: bool = True
    scene_bounds: list = field(default_factory=lambda: [0, -0.45, -0.05, 0.8, 0.55, 0.6])
    transform_augmentation_xyz: list = field(default_factory=lambda: [0.1, 0.1, 0.1])
    transform_augmentation_rpy: list = field(default_factory=lambda: [0.0, 0.0, 20.0])

    # 投影接口配置
    img_size: int = 256
    rend_three_views: bool = True
    add_depth: bool = False

    def __post_init__(self):
        """后处理验证配置参数"""
        # 验证数据路径
        if not os.path.exists(self.train_data_root):
            print(f"Warning: Train data root does not exist: {self.train_data_root}")

        if self.val_data_root and not os.path.exists(self.val_data_root):
            print(f"Warning: Val data root does not exist: {self.val_data_root}")

        # 验证参数范围
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert 0 < self.val_split < 1, "val_split must be between 0 and 1"
        assert self.sigma > 0, "sigma must be positive"
        assert len(self.scene_bounds) == 6, "scene_bounds must have 6 elements"


@dataclass
class ExperimentConfig:
    """
    完整的实验配置，包含模型和数据配置
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # 实验元信息
    experiment_name: str = "heatmap_prediction"
    project_name: str = "single_view_heatmap"
    description: str = "Single view RGB to heatmap sequence prediction"

    # 输出路径
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self):
        """后处理创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 确保模型和数据配置的序列长度一致
        if self.model.sequence_length != self.data.sequence_length:
            print(f"Warning: Model sequence_length ({self.model.sequence_length}) != "
                  f"Data sequence_length ({self.data.sequence_length})")
            # 使用数据配置的序列长度
            self.model.sequence_length = self.data.sequence_length

        # 确保图像尺寸一致
        if self.model.input_image_size != self.data.image_size:
            print(f"Warning: Model image size ({self.model.input_image_size}) != "
                  f"Data image size ({self.data.image_size})")
            # 使用数据配置的图像尺寸
            self.model.input_image_size = self.data.image_size
            self.model.output_image_size = self.data.image_size

    def save(self, path: str):
        """保存完整配置到文件"""
        import json

        config_dict = {
            'model': self.model.to_dict(),
            'data': self.data.__dict__,
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'description': self.description,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Experiment config saved to: {path}")

    @classmethod
    def load(cls, path: str):
        """从文件加载完整配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)

        model_config = ModelConfig.from_dict(config_dict['model'])
        data_config = DataConfig(**config_dict['data'])

        experiment_config = cls(
            model=model_config,
            data=data_config,
            experiment_name=config_dict.get('experiment_name', 'heatmap_prediction'),
            project_name=config_dict.get('project_name', 'single_view_heatmap'),
            description=config_dict.get('description', ''),
            output_dir=config_dict.get('output_dir', './outputs'),
            checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
            log_dir=config_dict.get('log_dir', './logs')
        )

        return experiment_config


# 预定义配置模板

def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()


def get_debug_config() -> ExperimentConfig:
    """获取调试配置（小规模快速测试）"""
    config = ExperimentConfig()

    # 模型配置调整
    config.model.sequence_length = 5
    config.model.input_image_size = (256, 256)
    config.model.output_image_size = (256, 256)
    config.model.num_inference_steps = 20

    # 数据配置调整
    config.data.sequence_length = 5
    config.data.image_size = (256, 256)
    config.data.batch_size = 2
    config.data.num_workers = 0
    config.data.min_trail_length = 8

    config.experiment_name = "debug_heatmap_prediction"

    return config


def get_high_quality_config() -> ExperimentConfig:
    """获取高质量配置（更大模型和更多步数）"""
    config = ExperimentConfig()

    # 模型配置调整
    config.model.sequence_length = 15
    config.model.input_image_size = (512, 512)
    config.model.output_image_size = (512, 512)
    config.model.num_inference_steps = 100
    config.model.guidance_scale = 10.0
    config.model.vae_torch_dtype = "float32"

    # 数据配置调整
    config.data.sequence_length = 15
    config.data.image_size = (512, 512)
    config.data.batch_size = 4  # 减小batch size以适应更大图像
    config.data.sigma = 2.0

    config.experiment_name = "high_quality_heatmap_prediction"

    return config


def test_config():
    """测试配置功能"""
    print("Testing model configuration...")

    # 测试默认配置
    config = get_default_config()
    print(f"Default config: {config.experiment_name}")
    print(f"Model device: {config.model.device}")
    print(f"Image size: {config.model.input_image_size}")
    print(f"Sequence length: {config.model.sequence_length}")

    # 测试保存和加载
    test_path = "/tmp/test_config.json"
    config.save(test_path)
    loaded_config = ExperimentConfig.load(test_path)
    print(f"Loaded config: {loaded_config.experiment_name}")

    # 测试调试配置
    debug_config = get_debug_config()
    print(f"Debug config batch size: {debug_config.data.batch_size}")

    # 测试高质量配置
    hq_config = get_high_quality_config()
    print(f"HQ config image size: {hq_config.model.input_image_size}")

    print("Config test completed!")
    return True


if __name__ == "__main__":
    test_config()