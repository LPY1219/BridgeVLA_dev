"""
Heatmap Dataset for Wan2.2 Training with Rotation and Gripper States
适配RobotTrajectoryDataset到UnifiedDataset格式的热力图数据集 (支持rotation和gripper)
"""

import sys
import os
import torch
from typing import Dict, Any, Optional
from .unified_dataset import UnifiedDataset
from .heatmap_utils import prepare_heatmap_data_for_wan_5B_TI2V, prepare_heatmap_data_for_wan_14B_I2V,prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP,prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW,prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_ROT_GRIP

# 从base_multi_view_dataset_with_rot_grip_3cam导入（支持3个第三视角相机）
# 注意：RobotTrajectoryDataset和ProjectionInterface将根据use_different_projection参数动态导入
RobotTrajectoryDataset = None
ProjectionInterface = None


class HeatmapUnifiedDataset(UnifiedDataset):
    """
    热力图专用的UnifiedDataset实现 (支持rotation和gripper states)
    包装RobotTrajectoryDataset并转换为Wan2.2训练所需格式
    """

    def __init__(self,
                 robot_dataset_config: Dict[str, Any],
                 colormap_name: str = 'jet',
                 repeat: int = 1,
                 wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
                 rotation_resolution: float = 5.0,
                 use_different_projection: bool = False,
                 **kwargs):
        """
        初始化热力图数据集

        Args:
            robot_dataset_config: RobotTrajectoryDataset的配置参数
            colormap_name: 使用的colormap名称（统一使用cv2 JET）
            repeat: 数据重复次数
            wan_type: Wan模型类型
            rotation_resolution: 旋转角度离散化分辨率（度），默认5度
            use_different_projection: 是否使用不同的投影方式
            **kwargs: 其他UnifiedDataset参数
        """
        self.colormap_name = colormap_name
        self.robot_dataset_config = robot_dataset_config
        self.wan_type = wan_type
        self.rotation_resolution = rotation_resolution
        self.use_different_projection = use_different_projection

        # 根据use_different_projection参数动态导入对应的模块
        global RobotTrajectoryDataset, ProjectionInterface
        if use_different_projection:
            # 使用不同投影方式的模块
            raise NotImplementedError("Not implemented yet")
        else:
            # 使用默认投影方式的模块
            from .base_multi_view_dataset_with_rot_grip_3cam_rlbench import RobotTrajectoryDataset, ProjectionInterface
            print("使用 RLBench base_multi_view_dataset_with_rot_grip_3cam 模块")

        # 检查依赖
        if RobotTrajectoryDataset is None:
            raise ImportError("RobotTrajectoryDataset not available. Please check single_view project setup.")

        # 创建机器人轨迹数据集
        self.robot_dataset = self._create_robot_dataset()

        # 初始化父类（使用虚拟参数，因为我们会重写关键方法）
        super().__init__(
            base_path="",  # 使用空字符串而不是None
            metadata_path=None,
            repeat=repeat,
            data_file_keys=(),
            main_data_operator=lambda x: x,
            **kwargs
        )

        # 重置数据
        self.data = []
        self.cached_data = []
        self.load_from_cache = False

        print(f"HeatmapUnifiedDataset initialized with {len(self.robot_dataset)} samples")

    def load_metadata(self, metadata_path):
        """
        重写父类的load_metadata方法，跳过文件搜索
        """
        # 我们不需要加载metadata，因为我们使用RobotTrajectoryDataset
        pass

    def _create_robot_dataset(self) -> RobotTrajectoryDataset:
        """
        创建RobotTrajectoryDataset实例
        """
        # 确保必要的参数存在
        required_params = ['data_root']
        for param in required_params:
            if param not in self.robot_dataset_config:
                raise ValueError(f"Missing required parameter: {param}")

        # 创建投影接口
        if 'projection_interface' not in self.robot_dataset_config:
            self.robot_dataset_config['projection_interface'] = ProjectionInterface()

        return RobotTrajectoryDataset(**self.robot_dataset_config)

    def __len__(self) -> int:
        """
        返回数据集长度
        """
        return len(self.robot_dataset) * self.repeat

    def __getitem__(self, data_id: int) -> Dict[str, Any]:
        """
        获取训练样本，转换为Wan2.2训练格式

        Args:
            data_id: 样本索引

        Returns:
            转换后的数据字典，包含 'prompt', 'video', 'input_image' 等
        """
        # 获取原始样本
        robot_sample = self.robot_dataset[data_id % len(self.robot_dataset)]

        # 提取数据
        rgb_image = robot_sample['rgb_image']  # (num_views, 3, H, W)
        heatmap_sequence = robot_sample['heatmap_sequence']  # (T, num_views, H, W)
        instruction = robot_sample['instruction']
        heatmap_start = robot_sample["heatmap_start"]  # (1, num_views, H, W)
        rgb_sequence = robot_sample["rgb_sequence"]  # (T, num_views, 3, H, W)

        # 提取新增的rotation和gripper数据
        start_pose = robot_sample.get('start_pose')  # (7,) - [x, y, z, qx, qy, qz, qw]
        future_poses = robot_sample.get('future_poses')  # (T, 7)
        start_gripper_state = robot_sample.get('start_gripper_state')  # bool
        future_gripper_states = robot_sample.get('future_gripper_states')  # (T,)


        # 转换为Wan格式
        if self.wan_type == "14B_I2V":
            wan_data = prepare_heatmap_data_for_wan_14B_I2V(
                rgb_image=rgb_image,
                heatmap_sequence=heatmap_sequence,
                heatmap_start=heatmap_start,
                instruction=instruction,
                colormap_name=self.colormap_name
            )
        elif self.wan_type == "WAN_2_1_14B_I2V":
            # 使用与14B_I2V相同的数据准备方式，但保留独立分支便于后续迭代
            wan_data = prepare_heatmap_data_for_wan_14B_I2V(
                rgb_image=rgb_image,
                heatmap_sequence=heatmap_sequence,
                heatmap_start=heatmap_start,
                instruction=instruction,
                colormap_name=self.colormap_name
            )
        elif self.wan_type == "5B_TI2V":
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V(
                rgb_image=rgb_image,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name
            )
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP":
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP(
                rgb_image=rgb_image,
                rgb_sequence=rgb_sequence,
                heatmap_start=heatmap_start,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
            )
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP_MV":
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW(
                rgb_image=rgb_image,
                rgb_sequence=rgb_sequence,
                heatmap_start=heatmap_start,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
            )
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP":
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_ROT_GRIP(
                rgb_image=rgb_image,
                rgb_sequence=rgb_sequence,
                heatmap_start=heatmap_start,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
                start_pose=start_pose,
                future_poses=future_poses,
                start_gripper_state=start_gripper_state,
                future_gripper_states=future_gripper_states,
                rotation_resolution=self.rotation_resolution,
            )
        else:
            assert False, f"Unsupported wan_type: {self.wan_type}"

        wan_data["rev_trans"]=robot_sample["rev_trans"]
        return wan_data

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        获取样本的详细信息
        """
        robot_sample = self.robot_dataset[idx % len(self.robot_dataset)]
        return {
            'robot_sample_info': robot_sample,
            'colormap_name': self.colormap_name,
            'dataset_type': 'heatmap_with_rot_grip'
        }


class HeatmapDatasetFactory:
    """
    热力图数据集工厂类，用于创建不同配置的数据集
    """

    @staticmethod
    def create_robot_trajectory_dataset(
        data_root: str,
        sequence_length: int = 10,
        step_interval: int = 1,
        min_trail_length: int = 15,
        image_size: tuple = (256, 256),
        sigma: float = 1.5,
        augmentation: bool = True,
        mode: str = "train",
        scene_bounds: list = [0, -0.45, -0.05, 0.8, 0.55, 0.6],
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        debug: bool = False,
        colormap_name: str = 'jet',
        repeat: int = 1,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
        rotation_resolution: float = 5.0,
        trail_start: int = None,
        trail_end: int = None,
        use_merged_pointcloud: bool = True,  # 是否使用拼接后的点云（True）或只使用相机1的点云（False）
        use_different_projection: bool = False,  # 是否使用不同的投影方式
        **kwargs
    ) -> HeatmapUnifiedDataset:
        """
        创建机器人轨迹热力图数据集

        Args:
            data_root: 数据根目录
            sequence_length: 热力图序列长度
            step_interval: step采样间隔
            min_trail_length: 最小轨迹长度
            image_size: 图像尺寸
            sigma: 热力图高斯标准差
            augmentation: 是否使用数据增强
            mode: 训练模式
            scene_bounds: 场景边界
            transform_augmentation_xyz: xyz变换增强范围
            transform_augmentation_rpy: rpy变换增强范围
            debug: 调试模式
            colormap_name: colormap名称
            repeat: 数据重复次数
            wan_type: Wan模型类型
            rotation_resolution: 旋转角度离散化分辨率（度），默认5度
            trail_start: 起始trail编号（如1表示从trail_1开始），None表示不限制
            trail_end: 结束trail编号（如50表示到trail_50结束），None表示不限制
            use_different_projection: 是否使用不同的投影方式
            **kwargs: 其他参数

        Returns:
            HeatmapUnifiedDataset实例
        """
        # 检查数据目录
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root directory not found: {data_root}")

        # 根据use_different_projection参数动态导入对应的模块
        if use_different_projection:
            from .base_multi_view_dataset_with_rot_grip_3cam_different_projection import ProjectionInterface
        else:
            from .base_multi_view_dataset_with_rot_grip_3cam import ProjectionInterface

        # 创建投影接口
        projection_interface = ProjectionInterface(
            img_size=image_size[0],
            rend_three_views=True,
            add_depth=False
        )

        # 机器人数据集配置
        robot_config = {
            'data_root': data_root,
            'projection_interface': projection_interface,
            'sequence_length': sequence_length,
            'step_interval': step_interval,
            'min_trail_length': min_trail_length,
            'image_size': image_size,
            'sigma': sigma,
            'augmentation': augmentation,
            'mode': mode,
            'scene_bounds': scene_bounds,
            'transform_augmentation_xyz': transform_augmentation_xyz,
            'transform_augmentation_rpy': transform_augmentation_rpy,
            'debug': debug,
            'trail_start': trail_start,
            'trail_end': trail_end,
            'use_merged_pointcloud': use_merged_pointcloud,
        }

        return HeatmapUnifiedDataset(
            robot_dataset_config=robot_config,
            colormap_name=colormap_name,
            repeat=repeat,
            wan_type=wan_type,
            rotation_resolution=rotation_resolution,
            use_different_projection=use_different_projection,
            **kwargs
        )

    @staticmethod
    def create_from_config_file(config_path: str) -> HeatmapUnifiedDataset:
        """
        从配置文件创建数据集
        """
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        return HeatmapDatasetFactory.create_robot_trajectory_dataset(**config)


# 测试代码
if __name__ == "__main__":
    # 测试数据集创建
    try:
        print("Testing HeatmapUnifiedDataset creation with rotation and gripper support...")

        # 使用示例数据路径（需要根据实际情况调整）
        test_data_root = "/data/Franka_data_3zed/put_lion_on_top_shelf"

        if os.path.exists(test_data_root):
            dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
                data_root=test_data_root,
                sequence_length=12,  # 不包括第一帧
                debug=True,  # 调试模式，只使用少量数据
                repeat=1,
                wan_type="5B_TI2V_RGB_HEATMAP_MV"
            )

            print(f"Dataset created successfully with {len(dataset)} samples")

            # 测试获取样本
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample keys: {list(sample.keys())}")
                print(f"Prompt: {sample['prompt']}")
                print(f"Video frames: {len(sample['video'])}")
                print(f"Input image type: {type(sample['input_image'])}")
                if 'start_pose' in sample:
                    print(f"Start pose shape: {sample['start_pose'].shape}")
                if 'future_poses' in sample:
                    print(f"Future poses shape: {sample['future_poses'].shape}")
                if 'start_gripper_state' in sample:
                    print(f"Start gripper state: {sample['start_gripper_state']}")
                if 'future_gripper_states' in sample:
                    print(f"Future gripper states shape: {sample['future_gripper_states'].shape}")

        else:
            print(f"Test data directory not found: {test_data_root}")
            print("Skipping test...")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    print("Test completed.")
