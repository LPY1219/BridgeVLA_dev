#!/usr/bin/env python3
"""
检查训练数据中gripper_change_targets的分布，特别是第一帧
"""
import sys
import torch
import numpy as np

sys.path.insert(0, '/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio')

from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip import HeatmapDatasetFactory

# 数据集配置
data_root = "/data/Franka_data/put_the_lion_on_the_top_shelf"
sequence_length = 12
image_size = (384, 384)
scene_bounds = [0, -0.65, -0.05, 0.8, 0.55, 0.75]
wan_type = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"
rotation_resolution = 5.0

# 创建数据集
dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
    data_root=data_root,
    sequence_length=sequence_length,
    image_size=image_size,
    scene_bounds=scene_bounds,
    wan_type=wan_type,
    rotation_resolution=rotation_resolution,
    trail_start=1,
    trail_end=16,
    mode='train'
)

print(f"Dataset size: {len(dataset)}")

# 统计第一帧的gripper_change分布
first_frame_changes = []
all_changes = []

for i in range(min(100, len(dataset))):
    sample = dataset[i]

    if 'gripper_change_targets' in sample:
        change_targets = sample['gripper_change_targets']

        if isinstance(change_targets, torch.Tensor):
            change_targets = change_targets.cpu().numpy()

        # 第一帧
        first_frame_changes.append(change_targets[0])

        # 所有帧
        all_changes.extend(change_targets)

first_frame_changes = np.array(first_frame_changes)
all_changes = np.array(all_changes)

print(f"\n=== 前100个样本的Gripper Change统计 ===")
print(f"第一帧 change=0 (不变): {np.sum(first_frame_changes == 0)} / {len(first_frame_changes)} = {np.mean(first_frame_changes == 0)*100:.1f}%")
print(f"第一帧 change=1 (改变): {np.sum(first_frame_changes == 1)} / {len(first_frame_changes)} = {np.mean(first_frame_changes == 1)*100:.1f}%")
print(f"\n所有帧 change=0 (不变): {np.sum(all_changes == 0)} / {len(all_changes)} = {np.mean(all_changes == 0)*100:.1f}%")
print(f"所有帧 change=1 (改变): {np.sum(all_changes == 1)} / {len(all_changes)} = {np.mean(all_changes == 1)*100:.1f}%")
