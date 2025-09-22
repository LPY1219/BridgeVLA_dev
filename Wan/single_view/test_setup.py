#!/usr/bin/env python3
"""
测试设置脚本
验证所有组件是否正常工作，无需真实数据
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.model_config import get_debug_config
from configs.training_config import get_debug_training_config
from models.wan_heatmap_model import WanHeatmapModel
from models.sequence_generator import SequenceGenerator
from data.dataset import ProjectionInterface
from utils.colormap_utils import test_colormap_conversion
from utils.heatmap_utils import test_heatmap_utils
from utils.visualization_utils import test_visualization_utils


def test_configs():
    """测试配置系统"""
    print("Testing configuration system...")

    try:
        # 测试模型配置
        exp_config = get_debug_config()
        print(f"✓ Model config loaded: device={exp_config.model.device}")

        # 测试训练配置
        train_config = get_debug_training_config()
        print(f"✓ Training config loaded: epochs={train_config.training.num_epochs}")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("Testing model creation...")

    try:
        # 创建调试配置
        config = get_debug_config().model
        config.sequence_length = 3
        config.input_image_size = (64, 64)
        config.output_image_size = (64, 64)

        # 创建模型
        model = WanHeatmapModel(config)
        print(f"✓ Model created successfully on device: {model.device}")

        # 测试模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_data_pipeline():
    """测试数据流水线（包括ProjectionInterface）"""
    print("Testing data pipeline with ProjectionInterface...")

    try:
        # 测试ProjectionInterface创建
        from data.dataset import ProjectionInterface
        projection_interface = ProjectionInterface(img_size=64, rend_three_views=True, add_depth=False)
        print("✓ ProjectionInterface created successfully")

        # 创建测试点云数据
        import numpy as np
        num_points = 1000
        pointcloud = torch.randn(num_points, 3) * 0.5  # 小范围的点云
        feat = torch.rand(num_points, 3)  # RGB特征

        # 测试点云投影
        rgb_images = projection_interface.project_pointcloud_to_rgb(pointcloud, feat)
        print(f"✓ Point cloud projected to RGB: {rgb_images.shape}")

        # 测试pose投影（确保在正确的设备上）
        poses = torch.randn(1, 5, 3) * 0.5  # 1个batch，5个pose，3D坐标
        poses = poses.to(projection_interface.renderer_device)
        img_locations = projection_interface.project_pose_to_pixel(poses)
        print(f"✓ Poses projected to pixel locations: {img_locations.shape}")

        # 测试heatmap生成
        heatmaps = projection_interface.generate_heatmap_from_img_locations(
            img_locations, width=64, height=64, sigma=1.5
        )
        print(f"✓ Heatmaps generated from locations: {heatmaps.shape}")

        return True
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(model):
    """测试模型前向传播"""
    print("Testing model forward pass...")

    try:
        model.eval()

        # 创建测试数据
        batch_size = 2
        rgb_images = torch.randn(batch_size, 3, 64, 64)
        heatmap_sequences = torch.rand(batch_size, 3, 64, 64)

        # 训练模式前向传播
        model.train()
        with torch.no_grad():
            train_output = model(rgb_images, heatmap_sequences)

        print(f"✓ Training forward pass: output keys = {list(train_output.keys())}")
        print(f"✓ Predictions shape: {train_output['predictions'].shape}")
        print(f"✓ Reconstruction loss: {train_output['reconstruction_loss'].item():.4f}")

        # 检查潜在空间维度
        latents = train_output['latents']
        print(f"✓ Latent space shape: {latents.shape}")

        # 推理模式前向传播
        model.eval()
        with torch.no_grad():
            inference_output = model.generate_heatmap_sequence(rgb_images)

        print(f"✓ Inference forward pass: output keys = {list(inference_output.keys())}")
        print(f"✓ Generated predictions shape: {inference_output['predictions'].shape}")

        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_creation():
    """测试优化器创建"""
    print("Testing optimizer and scheduler creation...")

    try:
        # 创建模型和配置
        config = get_debug_config().model
        train_config = get_debug_training_config()

        model = WanHeatmapModel(config)

        # 获取可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            # 添加一个简单的适配层用于测试
            model.test_layer = torch.nn.Linear(1, 1)
            trainable_params = [p for p in model.test_layer.parameters()]

        # 创建优化器
        optimizer = train_config.optimizer.get_optimizer(trainable_params)
        print(f"✓ Optimizer created: {type(optimizer).__name__}")

        # 创建调度器
        scheduler = train_config.optimizer.get_scheduler(optimizer, num_training_steps=100)
        print(f"✓ Scheduler created: {type(scheduler).__name__}")

        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """测试工具函数"""
    print("Testing utility functions...")

    try:
        # 测试colormap工具
        print("- Testing colormap utils...")
        test_colormap_conversion()

        # 测试heatmap工具
        print("- Testing heatmap utils...")
        test_heatmap_utils()

        # 测试可视化工具
        print("- Testing visualization utils...")
        test_visualization_utils()

        print("✓ All utility tests passed")
        return True
    except Exception as e:
        print(f"✗ Utility tests failed: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("HEATMAP PREDICTION MODEL SETUP TEST")
    print("="*60)

    all_tests_passed = True

    # 测试各个组件
    tests = [
        ("Configuration System", test_configs),
        ("Model Creation", lambda: test_model_creation()[0]),
        ("Data Pipeline", test_data_pipeline),
        ("Optimizer Creation", test_optimizer_creation),
        ("Utility Functions", test_utils),
    ]

    model = None

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        if test_name == "Model Creation":
            success, model = test_model_creation()
        else:
            success = test_func()

        if not success:
            all_tests_passed = False

    # 如果模型创建成功，测试前向传播
    if model is not None:
        print(f"\nModel Forward Pass:")
        print("-" * len("Model Forward Pass"))
        if not test_forward_pass(model):
            all_tests_passed = False

    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The training setup is ready.")
        print("\nYou can now start training with:")
        print("python run_training.py --data-root /path/to/your/data --debug")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*60)

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)