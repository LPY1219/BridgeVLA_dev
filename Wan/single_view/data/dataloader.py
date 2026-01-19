"""
数据加载器实现
RGB-to-heatmap序列的数据加载和批处理
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import random
try:
    from .dataset import RobotTrajectoryDataset, ProjectionInterface
except ImportError:
    from dataset import RobotTrajectoryDataset, ProjectionInterface



class HeatmapDataLoader:
    """
    Heatmap序列预测的数据加载器封装类
    提供训练和验证数据的加载、批处理和数据增强功能
    """

    def __init__(self,
                 train_data_root: str,
                 val_data_root: Optional[str] = None,
                 projection_interface: ProjectionInterface = None,
                 batch_size: int = 8,
                 sequence_length: int = 10,
                 image_size: Tuple[int, int] = (256, 256),
                 sigma: float = 1.5,
                 num_workers: int = 4,
                 shuffle_train: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 val_split: float = 0.2,
                 augmentation: bool = True,
                 **dataset_kwargs):
        """
        初始化数据加载器

        Args:
            train_data_root: 训练数据根目录
            val_data_root: 验证数据根目录，如果为None则从训练集分割
            projection_interface: 点云投影接口
            batch_size: 批次大小
            sequence_length: 预测序列长度
            image_size: 图像尺寸
            sigma: heatmap高斯分布标准差
            num_workers: 数据加载工作进程数
            shuffle_train: 是否打乱训练数据
            pin_memory: 是否固定内存
            drop_last: 是否丢弃最后不完整的批次
            val_split: 验证集分割比例（当val_data_root为None时使用）
            augmentation: 是否使用数据增强
            **dataset_kwargs: 数据集额外参数
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()

        if projection_interface is None:
            projection_interface = ProjectionInterface()

        # 创建训练数据集
        self.train_dataset = RobotTrajectoryDataset(
            data_root=train_data_root,
            projection_interface=projection_interface,
            sequence_length=sequence_length,
            image_size=image_size,
            sigma=sigma,
            augmentation=augmentation,
            mode="train",
            **dataset_kwargs
        )

        # 创建验证数据集
        if val_data_root is not None:
            # 使用独立的验证数据集
            self.val_dataset = RobotTrajectoryDataset(
                data_root=val_data_root,
                projection_interface=projection_interface,
                sequence_length=sequence_length,
                image_size=image_size,
                sigma=sigma,
                augmentation=False,  # 验证时不使用数据增强
                mode="val",
                **dataset_kwargs
            )
        else:
            # 从训练集分割验证集
            train_size = len(self.train_dataset)
            val_size = int(train_size * val_split)
            train_size = train_size - val_size

            # 随机分割数据集
            indices = list(range(len(self.train_dataset)))
            random.shuffle(indices)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # 创建子数据集
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)

            # 创建验证数据集（使用相同的配置但不用数据增强）
            val_full_dataset = RobotTrajectoryDataset(
                data_root=train_data_root,
                projection_interface=projection_interface,
                sequence_length=sequence_length,
                image_size=image_size,
                sigma=sigma,
                augmentation=False,
                mode="val",
                **dataset_kwargs
            )
            self.val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证时不打乱
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # 验证时不丢弃数据
            collate_fn=self.collate_fn
        )

        print(f"Data loading setup complete:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        自定义批处理函数

        Args:
            batch: 样本列表

        Returns:
            批处理后的数据字典
        """
        # 提取各个字段
        rgb_images = torch.stack([item['rgb_image'] for item in batch])
        heatmap_sequences = torch.stack([item['heatmap_sequence'] for item in batch])

        # 处理可选字段
        instructions = [item['instruction'] for item in batch]
        trail_names = [item['trail_name'] for item in batch]
        start_steps = [item['start_step'] for item in batch]

        # 处理可能存在的其他张量字段
        batch_dict = {
            'rgb_image': rgb_images,
            'heatmap_sequence': heatmap_sequences,
            'instruction': instructions,
            'trail_name': trail_names,
            'start_step': start_steps
        }

        # 添加可选的张量字段
        tensor_fields = ['img_locations', 'future_poses']
        for field in tensor_fields:
            if field in batch[0]:
                try:
                    batch_dict[field] = torch.stack([item[field] for item in batch])
                except:
                    # 如果无法stack，就保持为list
                    batch_dict[field] = [item[field] for item in batch]

        # 添加metadata
        if 'metadata' in batch[0]:
            batch_dict['metadata'] = [item['metadata'] for item in batch]

        return batch_dict

    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self.train_loader

    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self.val_loader

    def get_sample_batch(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """
        获取一个样本批次用于测试

        Args:
            split: 'train' 或 'val'

        Returns:
            样本批次
        """
        loader = self.train_loader if split == 'train' else self.val_loader
        return next(iter(loader))

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Returns:
            统计信息字典
        """
        # 获取样本以分析数据
        train_sample = self.get_sample_batch('train')
        val_sample = self.get_sample_batch('val')

        stats = {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'batch_size': self.batch_size,
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader),
            'rgb_shape': tuple(train_sample['rgb_image'].shape),
            'heatmap_shape': tuple(train_sample['heatmap_sequence'].shape),
            'sequence_length': train_sample['heatmap_sequence'].shape[1],
            'image_size': train_sample['rgb_image'].shape[-2:],
        }

        # RGB图像统计
        rgb_data = train_sample['rgb_image'].float()
        stats.update({
            'rgb_mean': rgb_data.mean().item(),
            'rgb_std': rgb_data.std().item(),
            'rgb_min': rgb_data.min().item(),
            'rgb_max': rgb_data.max().item()
        })

        # Heatmap统计
        heatmap_data = train_sample['heatmap_sequence'].float()
        stats.update({
            'heatmap_mean': heatmap_data.mean().item(),
            'heatmap_std': heatmap_data.std().item(),
            'heatmap_min': heatmap_data.min().item(),
            'heatmap_max': heatmap_data.max().item()
        })

        return stats

    def visualize_batch(self, split: str = 'train', save_path: Optional[str] = None):
        """
        可视化一个批次的数据

        Args:
            split: 'train' 或 'val'
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        try:
            from ..utils.visualization_utils import plot_comparison_grid
        except ImportError:
            from utils.visualization_utils import plot_comparison_grid

        batch = self.get_sample_batch(split)
        batch_size = len(batch['instruction'])

        # 选择前几个样本进行可视化
        num_vis = min(4, batch_size)

        fig, axes = plt.subplots(num_vis, 3, figsize=(15, 4 * num_vis))
        if num_vis == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_vis):
            # RGB图像
            rgb_img = batch['rgb_image'][i].permute(1, 2, 0).cpu().numpy()
            rgb_img = (rgb_img + 1) / 2  # 从[-1,1]转换到[0,1]
            rgb_img = np.clip(rgb_img, 0, 1)

            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title(f'RGB Image {i+1}')
            axes[i, 0].axis('off')

            # 第一帧heatmap
            heatmap_first = batch['heatmap_sequence'][i, 0].cpu().numpy()
            im1 = axes[i, 1].imshow(heatmap_first, cmap='jet')
            axes[i, 1].set_title(f'First Heatmap {i+1}')
            axes[i, 1].axis('off')

            # 最后一帧heatmap
            heatmap_last = batch['heatmap_sequence'][i, -1].cpu().numpy()
            im2 = axes[i, 2].imshow(heatmap_last, cmap='jet')
            axes[i, 2].set_title(f'Last Heatmap {i+1}')
            axes[i, 2].axis('off')

            # 添加指令信息
            instruction = batch['instruction'][i]
            fig.text(0.02, 0.95 - i * (1.0 / num_vis), f'Instruction {i+1}: {instruction}',
                    fontsize=10, wrap=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch visualization saved to: {save_path}")

        plt.show()

    def check_data_consistency(self) -> Dict[str, bool]:
        """
        检查数据一致性

        Returns:
            检查结果字典
        """
        results = {}

        try:
            # 检查训练加载器
            train_batch = self.get_sample_batch('train')
            results['train_loader_works'] = True

            # 检查数据形状
            rgb_shape = train_batch['rgb_image'].shape
            heatmap_shape = train_batch['heatmap_sequence'].shape

            results['rgb_shape_correct'] = len(rgb_shape) == 4  # (B, C, H, W)
            results['heatmap_shape_correct'] = len(heatmap_shape) == 4  # (B, T, H, W)
            results['batch_size_consistent'] = rgb_shape[0] == heatmap_shape[0]

            # 检查数据范围
            rgb_data = train_batch['rgb_image']
            heatmap_data = train_batch['heatmap_sequence']

            results['rgb_range_reasonable'] = (rgb_data.min() >= -2 and rgb_data.max() <= 2)
            results['heatmap_range_reasonable'] = (heatmap_data.min() >= 0 and heatmap_data.max() <= 1)

        except Exception as e:
            print(f"Train loader check failed: {e}")
            results['train_loader_works'] = False

        try:
            # 检查验证加载器
            val_batch = self.get_sample_batch('val')
            results['val_loader_works'] = True

        except Exception as e:
            print(f"Val loader check failed: {e}")
            results['val_loader_works'] = False

        # 打印结果
        print("Data consistency check results:")
        for key, value in results.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")

        return results


def create_heatmap_dataloader(data_root: str,
                             val_data_root: Optional[str] = None,
                             projection_interface: Optional[ProjectionInterface] = None,
                             **kwargs) -> HeatmapDataLoader:
    """
    创建heatmap数据加载器的便利函数

    Args:
        data_root: 训练数据根目录
        val_data_root: 验证数据根目录
        projection_interface: 投影接口
        **kwargs: 其他参数

    Returns:
        HeatmapDataLoader实例
    """
    if projection_interface is None:
        projection_interface = ProjectionInterface()

    return HeatmapDataLoader(
        train_data_root=data_root,
        val_data_root=val_data_root,
        projection_interface=projection_interface,
        **kwargs
    )


def test_dataloader():
    """
    测试数据加载器功能
    """
    print("Testing dataloader...")

    # 使用示例数据路径
    data_root = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"

    try:
        # 创建投影接口
        projection_interface = ProjectionInterface()

        # 创建数据加载器
        dataloader = HeatmapDataLoader(
            train_data_root=data_root,
            projection_interface=projection_interface,
            batch_size=2,
            sequence_length=5,
            num_workers=0,  # 测试时使用0避免多进程问题
            debug=True
        )

        # 检查数据一致性
        consistency_results = dataloader.check_data_consistency()

        # 获取统计信息
        stats = dataloader.get_dataset_stats()
        print(f"Dataset stats: {stats}")

        # 测试批次加载
        train_batch = dataloader.get_sample_batch('train')
        print(f"Train batch keys: {list(train_batch.keys())}")
        print(f"RGB shape: {train_batch['rgb_image'].shape}")
        print(f"Heatmap shape: {train_batch['heatmap_sequence'].shape}")

        val_batch = dataloader.get_sample_batch('val')
        print(f"Val batch keys: {list(val_batch.keys())}")

        print("Dataloader test completed successfully!")
        return True

    except Exception as e:
        print(f"Dataloader test failed: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    test_dataloader()