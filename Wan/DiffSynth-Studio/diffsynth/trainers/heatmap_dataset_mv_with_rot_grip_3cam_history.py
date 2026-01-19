"""
Heatmap Dataset for Wan2.2 Training with Multi-Frame History Support
支持多历史帧作为条件输入的热力图数据集

基于 heatmap_dataset_mv_with_rot_grip_3cam.py 修改，新增：
- num_history_frames 参数：控制使用多少帧历史作为条件（1-4）
- 修改采样逻辑确保有足够的历史帧
- 修改数据准备函数返回多帧历史数据
"""

import sys
import os
import torch
from typing import Dict, Any, Optional, List
from .unified_dataset import UnifiedDataset
from .heatmap_utils import (
    prepare_heatmap_data_for_wan_5B_TI2V,
    prepare_heatmap_data_for_wan_14B_I2V,
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP,
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW,
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_ROT_GRIP,
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_HISTORY,
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY,
)

# 动态导入
RobotTrajectoryDatasetHistory = None
ProjectionInterface = None


class RobotTrajectoryDatasetWithHistory:
    """
    支持多历史帧的机器人轨迹数据集包装器

    通过继承底层数据集并修改采样逻辑来支持多帧历史
    """

    def __init__(self, base_dataset, num_history_frames: int = 1):
        """
        Args:
            base_dataset: 底层的RobotTrajectoryDataset实例
            num_history_frames: 历史帧数量（1-4）
        """
        self.base_dataset = base_dataset
        self.num_history_frames = num_history_frames

        # 验证参数：必须是 1, 2, 或 1+4N (5, 9, 13, ...)
        if not self._is_valid_history_frames(num_history_frames):
            raise ValueError(
                f"num_history_frames must be 1, 2, or 1+4N (5, 9, 13, ...), got {num_history_frames}"
            )

        # 重新生成有效样本（考虑历史帧需求）
        self.valid_samples = self._generate_valid_samples_with_history()

        print(f"RobotTrajectoryDatasetWithHistory: {len(self.valid_samples)} valid samples "
              f"(num_history_frames={num_history_frames})")

    @staticmethod
    def _is_valid_history_frames(n: int) -> bool:
        """检查历史帧数量是否合法：必须是 1, 2, 或 1+4N"""
        if n == 1 or n == 2:
            return True
        if n > 2 and (n - 1) % 4 == 0:
            return True
        return False

    def _generate_valid_samples_with_history(self) -> List[Dict]:
        """
        生成有效样本，从轨迹的第一帧开始构造索引

        历史帧不足时的处理策略：
        - 对于轨迹的第一帧（start_step=0），重复第一帧自己来构造历史
        - 对于其他帧，如果前面没有足够的历史帧，重复最前面那一帧
        """
        valid_samples = []

        for trail_info in self.base_dataset.trail_data:
            num_steps = trail_info['num_steps']

            # 从第一帧开始构造索引（修改：原来是 self.num_history_frames - 1）
            min_start_step = 0
            max_start_step = num_steps - 1

            for start_step in range(min_start_step, max_start_step, self.base_dataset.step_interval):
                # 计算历史帧索引（从当前帧往回数）
                history_steps = []
                for i in range(self.num_history_frames - 1, -1, -1):
                    hist_step = start_step - i * self.base_dataset.step_interval
                    # 如果索引为负（历史帧不足），则重复使用第一帧（step 0）
                    if hist_step < 0:
                        hist_step = 0  # padding: 重复第一帧
                    history_steps.append(hist_step)

                # 计算未来帧索引
                future_steps = []
                for i in range(1, self.base_dataset.sequence_length + 1):
                    future_step = start_step + i * self.base_dataset.step_interval
                    if future_step < num_steps:
                        future_steps.append(future_step)
                    else:
                        future_steps.append(num_steps - 1)  # padding

                sample = {
                    'trail_info': trail_info,
                    'start_step': start_step,
                    'history_steps': history_steps,  # 新增：历史帧步骤列表
                    'future_steps': future_steps
                }
                valid_samples.append(sample)

        return valid_samples

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取训练样本，返回多帧历史数据
        """
        sample = self.valid_samples[idx]
        trail_info = sample['trail_info']
        start_step = sample['start_step']
        history_steps = sample['history_steps']
        future_steps = sample['future_steps']

        # 1. 加载所有历史帧数据
        history_poses = []
        history_pcds = []
        history_rgbs = []
        history_gripper_states = []

        for hist_step in history_steps:
            pose, pcd, rgb = self.base_dataset._load_step_data(trail_info, hist_step)
            if pose is None or pcd is None:
                raise ValueError(f"Failed to load history data for sample {idx}, step {hist_step}")
            history_poses.append(pose)
            history_pcds.append(pcd)
            history_rgbs.append(rgb)

            # 加载gripper state
            if trail_info['gripper_states_files']:
                gripper_file = trail_info['gripper_states_files'][hist_step]
                gripper_state = self.base_dataset._load_pickle_file(gripper_file)
                history_gripper_states.append(gripper_state)

        # 2. 加载未来帧数据
        future_poses = []
        future_pcds = []
        future_rgbs = []
        future_gripper_states = []

        for future_step in future_steps:
            future_pose, future_pcd, future_rgb = self.base_dataset._load_step_data(trail_info, future_step)
            if future_pose is None:
                raise ValueError(f"Failed to load future data for sample {idx}, step {future_step}")
            future_poses.append(future_pose)
            future_pcds.append(future_pcd)
            future_rgbs.append(future_rgb)

            if trail_info['gripper_states_files']:
                gripper_file = trail_info['gripper_states_files'][future_step]
                gripper_state = self.base_dataset._load_pickle_file(gripper_file)
                future_gripper_states.append(gripper_state)

        # 3. 拼接所有数据一起处理（历史帧 + 未来帧）
        all_poses = history_poses + future_poses
        all_pcds = history_pcds + future_pcds
        all_rgbs = history_rgbs + future_rgbs

        # 使用preprocess函数处理序列
        processed_pcd_list, processed_rgb_list, processed_pos, processed_rot_xyzw, rev_trans = \
            self.base_dataset.preprocess(all_pcds, all_rgbs, all_poses, trail_info)

        # 分离处理后的数据
        num_history = len(history_steps)
        processed_history_pcds = processed_pcd_list[:num_history]
        processed_history_rgbs = processed_rgb_list[:num_history]
        processed_future_pcds = processed_pcd_list[num_history:]
        processed_future_rgbs = processed_rgb_list[num_history:]

        processed_poses = torch.cat((processed_pos, processed_rot_xyzw), dim=1)  # num, 7
        processed_history_poses = processed_poses[:num_history]
        processed_future_poses = processed_poses[num_history:]

        # 4. 生成多帧历史RGB图像
        history_rgb_images = []
        for hist_pcd, hist_rgb in zip(processed_history_pcds, processed_history_rgbs):
            rgb_img = self.base_dataset.projection_interface.project_pointcloud_to_rgb(
                hist_pcd, hist_rgb
            )  # (1, num_views, H, W, 6)
            rgb_img = rgb_img[0, :, :, :, 3:]  # (num_views, H, W, 3)
            if isinstance(rgb_img, torch.Tensor):
                rgb_img = rgb_img.cpu().numpy()
            rgb_img = (rgb_img * 255).astype('uint8')
            history_rgb_images.append(rgb_img)

        # 5. 生成未来帧RGB图像序列
        rgb_future_list = []
        for future_pcd, future_rgb in zip(processed_future_pcds, processed_future_rgbs):
            future_rgb_image = self.base_dataset.projection_interface.project_pointcloud_to_rgb(
                future_pcd, future_rgb
            )
            future_rgb_image = future_rgb_image[0, :, :, :, 3:]
            if isinstance(future_rgb_image, torch.Tensor):
                future_rgb_image = future_rgb_image.cpu().numpy()
            future_rgb_image = (future_rgb_image * 255).astype('uint8')
            rgb_future_list.append(future_rgb_image)

        # 6. 生成多帧历史热力图
        # 获取渲染器设备
        renderer_device = self.base_dataset.projection_interface.renderer_device

        history_heatmaps = []
        for i, (hist_pcd, hist_pose) in enumerate(zip(processed_history_pcds, processed_history_poses[:, :3])):
            hist_pt_img = self.base_dataset.projection_interface.project_pose_to_pixel(
                hist_pose.unsqueeze(0).unsqueeze(0).to(renderer_device)
            )  # (1, 1, num_views, 2)
            hist_heatmap = self.base_dataset.projection_interface.generate_heatmap_from_img_locations(
                hist_pt_img,
                width=self.base_dataset.image_size[1],
                height=self.base_dataset.image_size[0],
                sigma=self.base_dataset.sigma
            )  # (1, 1, num_views, H, W)
            hist_heatmap = hist_heatmap[0, 0]  # (num_views, H, W)
            history_heatmaps.append(hist_heatmap)

        # 7. 生成未来帧热力图序列
        future_positions = processed_future_poses[:, :3]
        future_pt_img = self.base_dataset.projection_interface.project_pose_to_pixel(
            future_positions.unsqueeze(0).to(renderer_device)
        )  # (1, T, num_views, 2)
        heatmap_sequence = self.base_dataset.projection_interface.generate_heatmap_from_img_locations(
            future_pt_img,
            width=self.base_dataset.image_size[1],
            height=self.base_dataset.image_size[0],
            sigma=self.base_dataset.sigma
        )  # (1, T, num_views, H, W)
        heatmap_sequence = heatmap_sequence[0]  # (T, num_views, H, W)

        # 8. 组织返回数据
        import numpy as np

        # 将历史RGB图像堆叠: (num_history, num_views, H, W, 3) -> (num_history, num_views, 3, H, W)
        history_rgb_images_tensor = torch.from_numpy(
            np.stack(history_rgb_images, axis=0)
        ).permute(0, 1, 4, 2, 3).float()  # (num_history, num_views, 3, H, W)

        # 将未来RGB图像堆叠
        rgb_sequence_tensor = torch.from_numpy(
            np.stack(rgb_future_list, axis=0)
        ).permute(0, 1, 4, 2, 3).float()  # (T, num_views, 3, H, W)

        # 将历史热力图堆叠
        history_heatmaps_tensor = torch.stack(history_heatmaps, dim=0)  # (num_history, num_views, H, W)

        # 构建返回字典
        result = {
            # 多帧历史数据（新增）
            'history_rgb_images': history_rgb_images_tensor,  # (num_history, num_views, 3, H, W)
            'history_heatmaps': history_heatmaps_tensor,  # (num_history, num_views, H, W)
            'history_poses': processed_history_poses,  # (num_history, 7)
            'history_gripper_states': history_gripper_states,  # List[bool]

            # 兼容原有接口（使用最后一个历史帧作为"起始帧"）
            'rgb_image': history_rgb_images_tensor[-1].permute(0, 2, 3, 1).numpy().astype('uint8'),  # (num_views, H, W, 3)
            'heatmap_start': history_heatmaps_tensor[-1:],  # (1, num_views, H, W)
            'start_pose': processed_history_poses[-1],  # (7,)
            'start_gripper_state': history_gripper_states[-1] if history_gripper_states else None,

            # 未来帧数据
            'rgb_sequence': rgb_sequence_tensor,  # (T, num_views, 3, H, W)
            'heatmap_sequence': heatmap_sequence,  # (T, num_views, H, W)
            'future_poses': processed_future_poses,  # (T, 7)
            'future_gripper_states': torch.tensor(future_gripper_states) if future_gripper_states else None,

            # 元数据
            'instruction': trail_info['instruction'],
            'trail_name': trail_info['trail_name'],
            'start_step': start_step,
            'rev_trans': rev_trans,
            'num_history_frames': self.num_history_frames,
        }

        return result


class HeatmapUnifiedDatasetWithHistory(UnifiedDataset):
    """
    支持多历史帧的热力图数据集
    """

    def __init__(self,
                 robot_dataset_config: Dict[str, Any],
                 colormap_name: str = 'jet',
                 repeat: int = 1,
                 wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_HISTORY",
                 rotation_resolution: float = 5.0,
                 use_different_projection: bool = False,
                 num_history_frames: int = 1,
                 **kwargs):
        """
        初始化热力图数据集

        Args:
            robot_dataset_config: RobotTrajectoryDataset的配置参数
            colormap_name: 使用的colormap名称
            repeat: 数据重复次数
            wan_type: Wan模型类型
            rotation_resolution: 旋转角度离散化分辨率
            use_different_projection: 是否使用不同的投影方式
            num_history_frames: 历史帧数量（1-4）
        """
        self.colormap_name = colormap_name
        self.robot_dataset_config = robot_dataset_config
        self.wan_type = wan_type
        self.rotation_resolution = rotation_resolution
        self.use_different_projection = use_different_projection
        self.num_history_frames = num_history_frames

        # 动态导入
        global RobotTrajectoryDatasetHistory, ProjectionInterface
        if use_different_projection:
            from .base_multi_view_dataset_with_rot_grip_3cam_different_projection import RobotTrajectoryDataset, ProjectionInterface
            print("使用 base_multi_view_dataset_with_rot_grip_3cam_different_projection 模块")
        else:
            from .base_multi_view_dataset_with_rot_grip_3cam import RobotTrajectoryDataset, ProjectionInterface
            print("使用 base_multi_view_dataset_with_rot_grip_3cam 模块")

        RobotTrajectoryDatasetHistory = RobotTrajectoryDataset

        # 创建机器人轨迹数据集
        self.robot_dataset = self._create_robot_dataset()

        # 初始化父类
        super().__init__(
            base_path="",
            metadata_path=None,
            repeat=repeat,
            data_file_keys=(),
            main_data_operator=lambda x: x,
            **kwargs
        )

        self.data = []
        self.cached_data = []
        self.load_from_cache = False

        print(f"HeatmapUnifiedDatasetWithHistory initialized with {len(self.robot_dataset)} samples, "
              f"num_history_frames={num_history_frames}")

    def load_metadata(self, metadata_path):
        pass

    def _create_robot_dataset(self):
        """创建支持多历史帧的数据集"""
        required_params = ['data_root']
        for param in required_params:
            if param not in self.robot_dataset_config:
                raise ValueError(f"Missing required parameter: {param}")

        # 创建投影接口
        if 'projection_interface' not in self.robot_dataset_config:
            self.robot_dataset_config['projection_interface'] = ProjectionInterface()

        # 创建底层数据集
        base_dataset = RobotTrajectoryDatasetHistory(**self.robot_dataset_config)

        # 如果是单帧历史，直接使用底层数据集（向后兼容）
        if self.num_history_frames == 1:
            return base_dataset

        # 否则使用支持多历史帧的包装器
        return RobotTrajectoryDatasetWithHistory(base_dataset, self.num_history_frames)

    def __len__(self) -> int:
        return len(self.robot_dataset) * self.repeat

    def __getitem__(self, data_id: int) -> Dict[str, Any]:
        """获取训练样本"""
        robot_sample = self.robot_dataset[data_id % len(self.robot_dataset)]

        # 提取数据
        rgb_image = robot_sample['rgb_image']
        heatmap_sequence = robot_sample['heatmap_sequence']
        instruction = robot_sample['instruction']
        heatmap_start = robot_sample["heatmap_start"]
        rgb_sequence = robot_sample["rgb_sequence"]

        # 提取rotation和gripper数据
        start_pose = robot_sample.get('start_pose')
        future_poses = robot_sample.get('future_poses')
        start_gripper_state = robot_sample.get('start_gripper_state')
        future_gripper_states = robot_sample.get('future_gripper_states')

        # 提取多帧历史数据（如果可用）
        history_rgb_images = robot_sample.get('history_rgb_images')
        history_heatmaps = robot_sample.get('history_heatmaps')
        history_poses = robot_sample.get('history_poses')
        num_history = robot_sample.get('num_history_frames', 1)

        # 根据wan_type和历史帧数量选择数据准备函数
        if self.wan_type == "5B_TI2V_RGB_HEATMAP_MV_HISTORY" and num_history > 1:
            # 使用支持多历史帧的准备函数（不含rot_grip）
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_HISTORY(
                history_rgb_images=history_rgb_images,
                history_heatmaps=history_heatmaps,
                rgb_sequence=rgb_sequence,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
                start_pose=start_pose,
                future_poses=future_poses,
                history_poses=history_poses,
                start_gripper_state=start_gripper_state,
                future_gripper_states=future_gripper_states,
                rotation_resolution=self.rotation_resolution,
            )
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY" and num_history > 1:
            # 使用支持多历史帧的准备函数（含rot_grip）
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY(
                history_rgb_images=history_rgb_images,
                history_heatmaps=history_heatmaps,
                rgb_sequence=rgb_sequence,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
                start_pose=start_pose,
                future_poses=future_poses,
                history_poses=history_poses,
                start_gripper_state=start_gripper_state,
                future_gripper_states=future_gripper_states,
                rotation_resolution=self.rotation_resolution,
            )
        elif self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV", "5B_TI2V_RGB_HEATMAP_MV_HISTORY"]:
            # 单帧历史，使用原有函数
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW(
                rgb_image=rgb_image,
                rgb_sequence=rgb_sequence,
                heatmap_start=heatmap_start,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
            )
        elif self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP", "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY"]:
            # 单帧历史或者多历史帧模式下num_history==1的fallback
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
            raise ValueError(f"Unsupported wan_type: {self.wan_type}")

        wan_data["rev_trans"] = robot_sample["rev_trans"]
        wan_data["num_history_frames"] = num_history
        return wan_data


class HeatmapDatasetFactoryWithHistory:
    """
    支持多历史帧的热力图数据集工厂类
    """

    @staticmethod
    def create_robot_trajectory_dataset(
        data_root: str,
        sequence_length: int = 10,
        step_interval: int = 1,
        min_trail_length: int = 5,
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
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_HISTORY",
        rotation_resolution: float = 5.0,
        trail_start: int = None,
        trail_end: int = None,
        use_merged_pointcloud: bool = True,
        use_different_projection: bool = False,
        num_history_frames: int = 1,
        **kwargs
    ) -> HeatmapUnifiedDatasetWithHistory:
        """
        创建支持多历史帧的机器人轨迹热力图数据集
        """
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root directory not found: {data_root}")

        # 动态导入
        if use_different_projection:
            from .base_multi_view_dataset_with_rot_grip_3cam_different_projection import ProjectionInterface
        else:
            from .base_multi_view_dataset_with_rot_grip_3cam import ProjectionInterface

        projection_interface = ProjectionInterface(
            img_size=image_size[0],
            rend_three_views=True,
            add_depth=False
        )

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

        return HeatmapUnifiedDatasetWithHistory(
            robot_dataset_config=robot_config,
            colormap_name=colormap_name,
            repeat=repeat,
            wan_type=wan_type,
            rotation_resolution=rotation_resolution,
            use_different_projection=use_different_projection,
            num_history_frames=num_history_frames,
            **kwargs
        )
