"""
训练脚本
实现完整的训练循环，包括模型训练、验证、检查点保存等
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import ExperimentConfig, get_default_config, get_debug_config
from configs.training_config import CompleteTrainingConfig, get_default_training_config, get_debug_training_config
from models.wan_heatmap_model import WanHeatmapModel
from models.sequence_generator import SequenceGenerator
from data.dataloader import HeatmapDataLoader
from data.dataset import ProjectionInterface
from utils.heatmap_utils import calculate_peak_accuracy, calculate_sequence_consistency
from utils.visualization_utils import save_visualization_report


class Trainer:
    """
    训练器类
    负责完整的训练流程管理
    """

    def __init__(self,
                 experiment_config: ExperimentConfig,
                 training_config: CompleteTrainingConfig,
                 output_dir: str = "./outputs",
                 resume_from_checkpoint: Optional[str] = None):
        """
        初始化训练器

        Args:
            experiment_config: 实验配置
            training_config: 训练配置
            output_dir: 输出目录
            resume_from_checkpoint: 从检查点恢复训练
        """
        self.exp_config = experiment_config
        self.train_config = training_config
        self.output_dir = output_dir
        self.device = experiment_config.model.device

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化模型
        self.model = WanHeatmapModel(self.exp_config.model)
        self.model.to(self.device)

        # 初始化序列生成器
        self.generator = SequenceGenerator(self.model, self.exp_config.model)

        # 初始化数据加载器
        self._init_dataloader()

        # 初始化优化器和调度器
        self._init_optimizer()

        # 初始化日志
        self._init_logging()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if not training_config.training.greater_is_better else float('-inf')

        # 从检查点恢复
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # 保存配置
        self._save_configs()

    def _init_dataloader(self):
        """初始化数据加载器"""
        print("Initializing data loaders...")

        # 创建投影接口
        projection_interface = ProjectionInterface(
            img_size=self.exp_config.data.img_size,
            rend_three_views=self.exp_config.data.rend_three_views,
            add_depth=self.exp_config.data.add_depth
        )

        # 创建数据加载器
        self.dataloader = HeatmapDataLoader(
            train_data_root=self.exp_config.data.train_data_root,
            val_data_root=self.exp_config.data.val_data_root,
            projection_interface=projection_interface,
            batch_size=self.exp_config.data.batch_size,
            sequence_length=self.exp_config.data.sequence_length,
            image_size=self.exp_config.data.image_size,
            sigma=self.exp_config.data.sigma,
            num_workers=self.exp_config.data.num_workers,
            shuffle_train=self.exp_config.data.shuffle_train,
            pin_memory=self.exp_config.data.pin_memory,
            drop_last=self.exp_config.data.drop_last,
            val_split=self.exp_config.data.val_split,
            augmentation=self.exp_config.data.augmentation,
            scene_bounds=self.exp_config.data.scene_bounds,
            transform_augmentation_xyz=self.exp_config.data.transform_augmentation_xyz,
            transform_augmentation_rpy=self.exp_config.data.transform_augmentation_rpy
        )

        self.train_loader = self.dataloader.get_train_loader()
        self.val_loader = self.dataloader.get_val_loader()

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("Initializing optimizer and scheduler...")

        # 获取模型参数
        # 注意：Wan2.2 VAE通常是冻结的，我们可能需要添加一些可训练的层
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        print(f"Trainable parameters: {len(trainable_params)}")

        if len(trainable_params) == 0:
            print("Warning: No trainable parameters found!")
            # 为了演示，我们可以添加一个简单的适配层
            self.model.adaptation_layer = nn.Linear(1, 1).to(self.device)
            trainable_params = [p for p in self.model.adaptation_layer.parameters()]

        # 创建优化器
        self.optimizer = self.train_config.optimizer.get_optimizer(trainable_params)

        # 计算总训练步数
        dataset_size = len(self.dataloader.train_dataset)
        self.total_steps = self.train_config.training.get_total_steps(
            dataset_size, self.exp_config.data.batch_size
        )

        # 创建学习率调度器
        self.scheduler = self.train_config.optimizer.get_scheduler(
            self.optimizer, self.total_steps
        )

        print(f"Total training steps: {self.total_steps}")

    def _init_logging(self):
        """初始化日志系统"""
        if self.train_config.logging.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        # TODO: 添加其他日志系统（wandb等）的初始化

    def _save_configs(self):
        """保存配置文件"""
        config_dir = os.path.join(self.output_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)

        # 保存实验配置
        exp_config_path = os.path.join(config_dir, "experiment_config.json")
        self.exp_config.save(exp_config_path)

        # 保存训练配置
        train_config_path = os.path.join(config_dir, "training_config.json")
        self.train_config.save(train_config_path)

    def train(self):
        """
        执行完整的训练循环
        """
        print("Starting training...")
        print(f"Training for {self.train_config.training.num_epochs} epochs")
        print(f"Total steps: {self.total_steps}")

        for epoch in range(self.current_epoch, self.train_config.training.num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self._train_epoch()

            # 验证
            if epoch % self.train_config.validation.validate_every_n_epochs == 0:
                val_metrics = self._validate()
            else:
                val_metrics = {}

            # 记录日志
            self._log_metrics(train_metrics, val_metrics, epoch)

            # 保存检查点
            if epoch % (self.train_config.training.save_steps // len(self.train_loader)) == 0:
                self._save_checkpoint(epoch, val_metrics)

            # 早停检查
            if self._should_early_stop(val_metrics):
                print(f"Early stopping at epoch {epoch}")
                break

        print("Training completed!")

    def _train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            rgb_images = batch['rgb_image'].to(self.device)
            heatmap_sequences = batch['heatmap_sequence'].to(self.device)

            # 前向传播
            outputs = self.model(rgb_images, heatmap_sequences)

            # 计算损失
            loss_dict = self.model.compute_loss(
                outputs['predictions'],
                heatmap_sequences,
                self.train_config.loss
            )
            loss = loss_dict['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.train_config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.training.max_grad_norm
                )

            self.optimizer.step()

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 累计损失
            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # 定期记录日志
            if self.global_step % self.train_config.training.logging_steps == 0:
                if self.writer:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate',
                                         self.optimizer.param_groups[0]['lr'], self.global_step)

        return {'train_loss': total_loss / num_batches}

    def _validate(self) -> Dict[str, float]:
        """
        验证模型

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        peak_accuracies = []
        consistency_scores = []

        # 设置生成器为验证模式
        self.generator.set_generation_strategy("standard")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                rgb_images = batch['rgb_image'].to(self.device)
                heatmap_sequences = batch['heatmap_sequence'].to(self.device)

                # 前向传播
                outputs = self.model(rgb_images, heatmap_sequences)

                # 计算损失
                loss_dict = self.model.compute_loss(
                    outputs['predictions'],
                    heatmap_sequences,
                    self.train_config.loss
                )
                total_loss += loss_dict['total_loss'].item()

                # 计算精度指标
                predictions = outputs['predictions']
                for b in range(predictions.shape[0]):
                    # 峰值精度
                    for threshold in self.train_config.validation.peak_accuracy_thresholds:
                        for t in range(predictions.shape[1]):
                            accuracy = calculate_peak_accuracy(
                                predictions[b, t],
                                heatmap_sequences[b, t],
                                distance_threshold=threshold
                            )
                            peak_accuracies.append(accuracy)

                    # 序列一致性
                    consistency = calculate_sequence_consistency(predictions[b])
                    consistency_scores.append(consistency)

                # 只验证一部分数据以节省时间
                if batch_idx >= 10:  # 限制验证批次数
                    break

        # 计算平均指标
        metrics = {
            'val_loss': total_loss / min(len(self.val_loader), 10),
            'val_peak_accuracy': np.mean(peak_accuracies) if peak_accuracies else 0.0,
            'val_consistency': np.mean(consistency_scores) if consistency_scores else 0.0
        }

        # 生成可视化样本
        if self.train_config.validation.visualize_predictions:
            self._generate_visualization_samples()

        return metrics

    def _generate_visualization_samples(self):
        """生成可视化样本"""
        try:
            # 获取一个验证批次
            val_batch = next(iter(self.val_loader))
            rgb_images = val_batch['rgb_image'][:4].to(self.device)  # 只取前4个样本
            gt_sequences = val_batch['heatmap_sequence'][:4].to(self.device)

            # 生成预测
            with torch.no_grad():
                result = self.generator.generate(rgb_images)
                pred_sequences = result['predictions']

            # 保存可视化
            vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            for i in range(min(4, rgb_images.shape[0])):
                results = {
                    'pred_sequence': pred_sequences[i].cpu().numpy(),
                    'gt_sequence': gt_sequences[i].cpu().numpy(),
                    'rgb_image': (rgb_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2,
                    'instruction': val_batch['instruction'][i]
                }

                save_visualization_report(
                    results,
                    vis_dir,
                    f"epoch_{self.current_epoch}_sample_{i}"
                )

        except Exception as e:
            print(f"Warning: Visualization generation failed: {e}")

    def _log_metrics(self, train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float], epoch: int):
        """记录训练指标"""
        # 打印到控制台
        print(f"\nEpoch {epoch}:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # 记录到TensorBoard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(key, value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(key, value, epoch)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'exp_config': self.exp_config.__dict__,
            'train_config': self.train_config.to_dict()
        }

        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)

        # 保存定期检查点
        epoch_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, epoch_path)

        # 保存最佳模型
        metric_key = self.train_config.training.metric_for_best_model
        if metric_key in metrics:
            current_metric = metrics[metric_key]
            is_better = (current_metric > self.best_metric if self.train_config.training.greater_is_better
                        else current_metric < self.best_metric)

            if is_better:
                self.best_metric = current_metric
                best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"Saved new best model with {metric_key}: {current_metric:.4f}")

        print(f"Checkpoint saved: {latest_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        print(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

        print(f"Resumed from epoch {self.current_epoch}, global step {self.global_step}")

    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        if not self.train_config.training.early_stopping:
            return False

        # TODO: 实现早停逻辑
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train heatmap prediction model")
    parser.add_argument("--config", type=str, default="default",
                       choices=["default", "debug"],
                       help="Configuration preset")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Override data root path")

    args = parser.parse_args()

    # 加载配置
    if args.config == "debug":
        exp_config = get_debug_config()
        train_config = get_debug_training_config()
    else:
        exp_config = get_default_config()
        train_config = get_default_training_config()

    # 覆盖数据路径
    if args.data_root:
        exp_config.data.train_data_root = args.data_root

    print(f"Using configuration: {args.config}")
    print(f"Data root: {exp_config.data.train_data_root}")
    print(f"Output directory: {args.output_dir}")

    # 创建训练器
    trainer = Trainer(
        experiment_config=exp_config,
        training_config=train_config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume
    )

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if train_config.training.save_checkpoint_on_interrupt:
            trainer._save_checkpoint(trainer.current_epoch, {})
            print("Checkpoint saved before exit")


if __name__ == "__main__":
    main()