#!/usr/bin/env python3
"""
简单的训练启动脚本
配置好数据路径后可以直接启动训练
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.model_config import get_default_config, get_debug_config
from configs.training_config import get_default_training_config, get_debug_training_config
from experiments.train import Trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Start heatmap prediction model training")

    # 基础配置
    parser.add_argument("--data-root", type=str, required=True,
                       help="Path to training data root directory")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")

    # 配置选择
    parser.add_argument("--config", type=str, default="default",
                       choices=["default", "debug"],
                       help="Configuration preset to use")

    # 训练参数覆盖
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--sequence-length", type=int, default=None,
                       help="Override sequence length")

    # 硬件设置
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of data loading workers")

    # 其他选项
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (equivalent to --config debug)")

    args = parser.parse_args()

    # 选择配置
    if args.debug or args.config == "debug":
        print("Using debug configuration")
        exp_config = get_debug_config()
        train_config = get_debug_training_config()
    else:
        print("Using default configuration")
        exp_config = get_default_config()
        train_config = get_default_training_config()

    # 应用命令行参数覆盖
    exp_config.data.train_data_root = args.data_root

    if args.batch_size:
        exp_config.data.batch_size = args.batch_size
    if args.epochs:
        train_config.training.num_epochs = args.epochs
    if args.learning_rate:
        train_config.optimizer.learning_rate = args.learning_rate
    if args.sequence_length:
        exp_config.model.sequence_length = args.sequence_length
        exp_config.data.sequence_length = args.sequence_length
    if args.device != "auto":
        exp_config.model.device = args.device
    if args.num_workers:
        exp_config.data.num_workers = args.num_workers

    # 打印配置信息
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Data root: {exp_config.data.train_data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {exp_config.model.device}")
    print(f"Batch size: {exp_config.data.batch_size}")
    print(f"Epochs: {train_config.training.num_epochs}")
    print(f"Learning rate: {train_config.optimizer.learning_rate}")
    print(f"Sequence length: {exp_config.model.sequence_length}")
    print(f"Image size: {exp_config.model.input_image_size}")
    print(f"Num workers: {exp_config.data.num_workers}")
    print("="*50 + "\n")

    # 验证数据路径
    if not os.path.exists(exp_config.data.train_data_root):
        print(f"Error: Data root directory does not exist: {exp_config.data.train_data_root}")
        print("Please check the path and try again.")
        return 1

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 创建训练器
        print("Initializing trainer...")
        trainer = Trainer(
            experiment_config=exp_config,
            training_config=train_config,
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume
        )

        # 开始训练
        print("Starting training...")
        trainer.train()

        print("\nTraining completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)