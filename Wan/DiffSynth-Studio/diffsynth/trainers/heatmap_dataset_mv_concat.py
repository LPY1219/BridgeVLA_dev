"""
Heatmap Dataset for Multi-View Token Concatenation Model

This dataset is compatible with WanModel_mv_concat which uses token concatenation
instead of multi-view attention. The data format is identical to the standard
multi-view dataset since token concatenation happens inside the model.

Supported wan_types:
- 5B_TI2V_RGB_HEATMAP_MV_CONCAT: Token concatenation with dual head (RGB + Heatmap)
- 5B_TI2V_RGB_HEATMAP_MV_CONCAT_ROT_GRIP: Token concatenation with rotation and gripper prediction
"""

import sys
import os
import torch
from typing import Dict, Any, Optional
from .unified_dataset import UnifiedDataset
from .heatmap_utils import (
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW,
    prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_ROT_GRIP
)

# Dynamic imports based on projection type
RobotTrajectoryDataset = None
ProjectionInterface = None


class HeatmapUnifiedDatasetMVConcat(UnifiedDataset):
    """
    Heatmap dataset for multi-view token concatenation model.

    The data format is identical to the standard multi-view dataset
    since token concatenation happens inside the model (WanModel_mv_concat).
    """

    def __init__(self,
                 robot_dataset_config: Dict[str, Any],
                 colormap_name: str = 'jet',
                 repeat: int = 1,
                 wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_CONCAT",
                 rotation_resolution: float = 5.0,
                 use_different_projection: bool = False,
                 num_views: int = 3,
                 **kwargs):
        """
        Initialize heatmap dataset for token concatenation model.

        Args:
            robot_dataset_config: Configuration for RobotTrajectoryDataset
            colormap_name: Colormap name (using cv2 JET)
            repeat: Data repeat count
            wan_type: Model type (5B_TI2V_RGB_HEATMAP_MV_CONCAT or 5B_TI2V_RGB_HEATMAP_MV_CONCAT_ROT_GRIP)
            rotation_resolution: Rotation discretization resolution (degrees)
            use_different_projection: Whether to use different projection methods
            num_views: Number of views (default 3)
            **kwargs: Other UnifiedDataset parameters
        """
        self.colormap_name = colormap_name
        self.robot_dataset_config = robot_dataset_config
        self.wan_type = wan_type
        self.rotation_resolution = rotation_resolution
        self.use_different_projection = use_different_projection
        self.num_views = num_views

        # Dynamic import based on projection type
        global RobotTrajectoryDataset, ProjectionInterface
        if use_different_projection:
            from .base_multi_view_dataset_with_rot_grip_3cam_different_projection import (
                RobotTrajectoryDataset, ProjectionInterface
            )
            print("Using base_multi_view_dataset_with_rot_grip_3cam_different_projection module")
        else:
            from .base_multi_view_dataset_with_rot_grip_3cam import (
                RobotTrajectoryDataset, ProjectionInterface
            )
            print("Using base_multi_view_dataset_with_rot_grip_3cam module")

        if RobotTrajectoryDataset is None:
            raise ImportError("RobotTrajectoryDataset not available.")

        # Create robot trajectory dataset
        self.robot_dataset = self._create_robot_dataset()

        # Initialize parent class
        super().__init__(
            base_path="",
            metadata_path=None,
            repeat=repeat,
            data_file_keys=(),
            main_data_operator=lambda x: x,
            **kwargs
        )

        # Reset data
        self.data = []
        self.cached_data = []
        self.load_from_cache = False

        print(f"HeatmapUnifiedDatasetMVConcat initialized with {len(self.robot_dataset)} samples")
        print(f"  wan_type: {wan_type}, num_views: {num_views}")

    def load_metadata(self, metadata_path):
        """Override parent's load_metadata method."""
        pass

    def _create_robot_dataset(self):
        """Create RobotTrajectoryDataset instance."""
        required_params = ['data_root']
        for param in required_params:
            if param not in self.robot_dataset_config:
                raise ValueError(f"Missing required parameter: {param}")

        if 'projection_interface' not in self.robot_dataset_config:
            self.robot_dataset_config['projection_interface'] = ProjectionInterface()

        return RobotTrajectoryDataset(**self.robot_dataset_config)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.robot_dataset) * self.repeat

    def __getitem__(self, data_id: int) -> Dict[str, Any]:
        """
        Get training sample.

        Args:
            data_id: Sample index

        Returns:
            Data dictionary containing 'prompt', 'video', 'input_image', etc.
        """
        # Get original sample
        robot_sample = self.robot_dataset[data_id % len(self.robot_dataset)]

        # Extract data
        rgb_image = robot_sample['rgb_image']  # (num_views, 3, H, W)
        heatmap_sequence = robot_sample['heatmap_sequence']  # (T, num_views, H, W)
        instruction = robot_sample['instruction']
        heatmap_start = robot_sample["heatmap_start"]  # (1, num_views, H, W)
        rgb_sequence = robot_sample["rgb_sequence"]  # (T, num_views, 3, H, W)

        # Extract rotation and gripper data
        start_pose = robot_sample.get('start_pose')
        future_poses = robot_sample.get('future_poses')
        start_gripper_state = robot_sample.get('start_gripper_state')
        future_gripper_states = robot_sample.get('future_gripper_states')
        img_locations = robot_sample.get('img_locations')

        # Convert to Wan format based on wan_type
        if self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV_CONCAT"]:
            # Token concatenation without rotation/gripper prediction
            # Uses same data format as 5B_TI2V_RGB_HEATMAP_MV
            wan_data = prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW(
                rgb_image=rgb_image,
                rgb_sequence=rgb_sequence,
                heatmap_start=heatmap_start,
                heatmap_sequence=heatmap_sequence,
                instruction=instruction,
                colormap_name=self.colormap_name,
            )
        elif self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV_CONCAT_ROT_GRIP"]:
            # Token concatenation with rotation and gripper prediction
            # Uses same data format as 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP
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
            raise ValueError(f"Unsupported wan_type for MV Concat: {self.wan_type}")

        # Add reverse transformation info
        wan_data["rev_trans"] = robot_sample.get("rev_trans")

        # Add img_locations if available
        if img_locations is not None:
            wan_data["img_locations"] = img_locations

        # Add num_views metadata for model
        wan_data["num_views"] = self.num_views

        return wan_data

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed sample info."""
        robot_sample = self.robot_dataset[idx % len(self.robot_dataset)]
        return {
            'robot_sample_info': robot_sample,
            'colormap_name': self.colormap_name,
            'dataset_type': 'heatmap_mv_concat',
            'num_views': self.num_views
        }


class HeatmapDatasetFactoryMVConcat:
    """
    Factory class for creating multi-view token concatenation datasets.
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
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_CONCAT",
        rotation_resolution: float = 5.0,
        trail_start: int = None,
        trail_end: int = None,
        use_merged_pointcloud: bool = True,
        use_different_projection: bool = False,
        num_views: int = 3,
        **kwargs
    ) -> HeatmapUnifiedDatasetMVConcat:
        """
        Create robot trajectory heatmap dataset for token concatenation model.

        Args:
            data_root: Data root directory
            sequence_length: Heatmap sequence length
            step_interval: Step sampling interval
            min_trail_length: Minimum trail length
            image_size: Image size
            sigma: Gaussian sigma for heatmap
            augmentation: Whether to use data augmentation
            mode: Training mode
            scene_bounds: Scene bounds
            transform_augmentation_xyz: XYZ transform augmentation range
            transform_augmentation_rpy: RPY transform augmentation range
            debug: Debug mode
            colormap_name: Colormap name
            repeat: Data repeat count
            wan_type: Wan model type
            rotation_resolution: Rotation discretization resolution (degrees)
            trail_start: Start trail number
            trail_end: End trail number
            use_different_projection: Whether to use different projection
            num_views: Number of views (default 3)
            **kwargs: Other parameters

        Returns:
            HeatmapUnifiedDatasetMVConcat instance
        """
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root directory not found: {data_root}")

        # Dynamic import based on projection type
        if use_different_projection:
            from .base_multi_view_dataset_with_rot_grip_3cam_different_projection import ProjectionInterface
        else:
            from .base_multi_view_dataset_with_rot_grip_3cam import ProjectionInterface

        # Create projection interface
        projection_interface = ProjectionInterface(
            img_size=image_size[0],
            rend_three_views=True,
            add_depth=False
        )

        # Robot dataset config
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
            'use_merged_pointcloud': use_merged_pointcloud,
        }

        # Add trail range if specified
        if trail_start is not None:
            robot_config['trail_start'] = trail_start
        if trail_end is not None:
            robot_config['trail_end'] = trail_end

        return HeatmapUnifiedDatasetMVConcat(
            robot_dataset_config=robot_config,
            colormap_name=colormap_name,
            repeat=repeat,
            wan_type=wan_type,
            rotation_resolution=rotation_resolution,
            use_different_projection=use_different_projection,
            num_views=num_views,
            **kwargs
        )


# Alias for compatibility
HeatmapUnifiedDataset = HeatmapUnifiedDatasetMVConcat
HeatmapDatasetFactory = HeatmapDatasetFactoryMVConcat
