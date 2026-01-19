"""
Dataset可视化工具
可视化RobotTrajectoryDataset类返回的数据
包括原图、投影图、热力图序列等
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# 设置环境变量
os.environ['COPPELIASIM_ROOT'] = '/share/project/lpy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04'
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + os.environ['COPPELIASIM_ROOT']
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']
os.environ['DISPLAY'] = ':1.0'

sys.path.append("/share/project/lpy/BridgeVLA/Wan/single_view")
from data.dataset import RobotTrajectoryDataset, ProjectionInterface


def visualize_dataset_sample(dataset, sample_idx=0, save_dir=None,viewpoint_idx=0):
    """
    可视化dataset中的一个样本

    Args:
        dataset: RobotTrajectoryDataset实例
        sample_idx: 样本索引
        save_dir: 保存图片的目录，如果为None则不保存
    """
    # 获取样本数据
    sample = dataset[sample_idx]

    # 提取数据
    raw_rgb_image = sample['raw_rgb_image']  # 原图
    rgb_image = sample['rgb_image']          # 投影图
    heatmap_sequence = sample['heatmap_sequence']  # 热力图序列
    img_locations = sample['img_locations']  # 图像位置
    future_poses = sample['future_poses']    # 未来poses
    instruction = sample['instruction']      # 指令
    trail_name = sample['trail_name']        # 轨迹名称
    start_step = sample['start_step']        # 起始步骤

    # 打印信息
    print("=" * 60)
    print(f"Trail Name: {trail_name}")
    print(f"Start Step: {start_step}")
    print(f"Instruction: {instruction}")
    print(f"Raw RGB Image Shape: {raw_rgb_image.shape}")
    print(f"RGB Image Shape: {rgb_image.shape}")
    print(f"Heatmap Sequence Shape: {heatmap_sequence.shape}")
    print(f"Future Poses Shape: {future_poses.shape}")
    print(f"Img Locations Shape: {img_locations.shape}")
    print(f"Future Poses:")
    if isinstance(future_poses, torch.Tensor):
        future_poses_numpy = future_poses.cpu().numpy()
        for i, pose in enumerate(future_poses_numpy):
            print(f"  Step {i+1}: [{float(pose[0]):.4f}, {float(pose[1]):.4f}, {float(pose[2]):.4f}]")
    else:
        for i, pose in enumerate(future_poses):
            print(f"  Step {i+1}: [{float(pose[0]):.4f}, {float(pose[1]):.4f}, {float(pose[2]):.4f}]")
    print(f"Img Locations:")
    if isinstance(img_locations, torch.Tensor):
        img_locs_numpy = img_locations.cpu().numpy()
        for i, loc in enumerate(img_locs_numpy[0]):  # [0]因为batch维度
            print(f"  Step {i+1}: [{float(loc[viewpoint_idx][0]):.2f}, {float(loc[viewpoint_idx][1]):.2f}]")
    else:
        for i, loc in enumerate(img_locations[0]):  # [0]因为batch维度
            print(f"  Step {i+1}: [{float(loc[viewpoint_idx][0]):.2f}, {float(loc[viewpoint_idx][1]):.2f}]")
    print("=" * 60)

    # 转换数据格式用于显示
    # 原图处理 (raw_rgb_image是processed_rgb，格式为(h*w, 3))
    if isinstance(raw_rgb_image, torch.Tensor):
        raw_rgb_display = raw_rgb_image.cpu().numpy()
        # 检查数据形状，如果是扁平化的，需要重新调整
        if len(raw_rgb_display.shape) == 2 and raw_rgb_display.shape[1] == 3:
            # 假设图像是256x256
            img_size = int(np.sqrt(raw_rgb_display.shape[0]))
            raw_rgb_display = raw_rgb_display.reshape(img_size, img_size, 3)
        elif raw_rgb_display.shape[0] == 3:  # CHW -> HWC
            raw_rgb_display = raw_rgb_display.transpose(1, 2, 0)
        # 数据已经在[0,1]范围内了
        raw_rgb_display = np.clip(raw_rgb_display, 0, 1)
    else:
        raw_rgb_display = raw_rgb_image

    # 投影图处理
    if isinstance(rgb_image, torch.Tensor):
        rgb_display = rgb_image.cpu().numpy()
        if rgb_display.shape[0] == 3:  # CHW -> HWC
            rgb_display = rgb_display.transpose(1, 2, 0)
        # 检查数据范围
        if rgb_display.max() <= 1.0 and rgb_display.min() >= -1.0:
            # 假设在[-1,1]范围内，转换到[0,1]
            rgb_display = (rgb_display + 1) / 2
        rgb_display = np.clip(rgb_display, 0, 1)
    else:
        rgb_display = rgb_image

    # 热力图序列处理
    if isinstance(heatmap_sequence, torch.Tensor):
        heatmap_display = heatmap_sequence.cpu().numpy()
    else:
        heatmap_display = heatmap_sequence

    sequence_length = heatmap_display.shape[0]

    # 创建图形
    plt.figure(figsize=(16, 8))

    # 第一行：原图和投影图
    # 原图
    plt.subplot(2, max(sequence_length, 2), 1)
    plt.imshow(raw_rgb_display)
    plt.title('Raw RGB Image', fontsize=12)
    plt.axis('off')

    # 投影图
    plt.subplot(2, max(sequence_length, 2), 2)
    plt.imshow(rgb_display)
    plt.title('Projected RGB Image', fontsize=12)
    plt.axis('off')

    # 第二行：热力图序列
    for i in range(sequence_length):
        ax = plt.subplot(2, max(sequence_length, 2), max(sequence_length, 2) + 1 + i)
        heatmap = heatmap_display[i]

        # 使用jet colormap显示热力图
        im = plt.imshow(heatmap, cmap='jet', vmin=0, vmax=heatmap.max())
        plt.title(f'Heatmap {i+1}', fontsize=10)
        plt.axis('off')

        # 添加colorbar
        plt.colorbar(im, ax=ax, shrink=0.6)

    # 设置整体标题
    plt.suptitle(f'{trail_name} - Step {start_step}\nInstruction: {instruction}',
                 fontsize=14, y=0.95)

    plt.tight_layout()

    # 保存图片
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{trail_name}_step_{start_step}_visualization.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {filepath}")

    plt.show()


def visualize_multiple_samples(dataset, num_samples=3, save_dir=None):
    """
    可视化多个样本

    Args:
        dataset: RobotTrajectoryDataset实例
        num_samples: 要可视化的样本数量
        save_dir: 保存图片的目录
    """
    for i in range(min(num_samples, len(dataset))):
        print(f"\nVisualizing sample {i+1}/{num_samples}...")
        visualize_dataset_sample(dataset, i, save_dir)


def test_visualization():
    """
    测试可视化功能
    """
    # 数据路径 - 根据实际情况修改
    data_root = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"

    # 检查数据路径是否存在
    if not os.path.exists(data_root):
        print(f"数据路径不存在: {data_root}")
        print("请修改data_root变量为正确的数据路径")
        return

    try:
        # 创建投影接口
        projection_interface = ProjectionInterface()

        # 创建数据集
        dataset = RobotTrajectoryDataset(
            data_root=data_root,
            projection_interface=projection_interface,
            sequence_length=5,
            min_trail_length=10,
            debug=True  # 使用debug模式减少数据量
        )

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            # 创建保存目录
            save_dir = "/share/project/lpy/BridgeVLA/Wan/single_view/visualization/outputs"

            # 可视化前几个样本
            visualize_multiple_samples(dataset, num_samples=3, save_dir=save_dir)
        else:
            print("Dataset is empty!")

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visualization()