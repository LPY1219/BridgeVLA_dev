#!/usr/bin/env python3
"""
测试模型加载脚本
专门测试WanPipeline的加载和内存优化
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from configs.model_config import get_debug_config
from models.wan_heatmap_model import WanHeatmapModel


def test_wan_pipeline_loading():
    """测试WanPipeline加载"""
    print("=" * 60)
    print("TESTING WAN PIPELINE LOADING WITH GPU1")
    print("=" * 60)

    try:
        # 创建调试配置
        config = get_debug_config().model
        config.sequence_length = 3
        config.input_image_size = (256, 256)
        config.output_image_size = (256, 256)

        print(f"Using device: {config.device}")

        # 显示GPU信息
        if torch.cuda.is_available():
            print(f"CUDA devices available: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        # 创建模型
        print("\nCreating WanHeatmapModel...")
        model = WanHeatmapModel(config)

        print(f"\nModel device: {model.device}")
        print(f"Model torch_dtype: {model.torch_dtype}")

        # 检查pipeline加载状态
        if model.wan_pipeline is not None:
            if hasattr(model.wan_pipeline, 'transformer'):
                print("✓ Complete WanPipeline loaded successfully!")
                print(f"✓ Pipeline has transformer: {hasattr(model.wan_pipeline, 'transformer')}")
                print(f"✓ Pipeline has vae: {hasattr(model.wan_pipeline, 'vae')}")
                print(f"✓ Pipeline has text_encoder: {hasattr(model.wan_pipeline, 'text_encoder')}")
            else:
                print("⚠ Fallback adaptation layers loaded")
        else:
            print("✗ Pipeline loading failed")

        # 检查参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nParameter Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")

        # 检查GPU内存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nGPU Memory Usage:")
            print(f"Allocated: {memory_allocated:.2f}GB")
            print(f"Reserved: {memory_reserved:.2f}GB")

        print("\n" + "=" * 60)
        print("MODEL LOADING TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_wan_pipeline_loading()
    sys.exit(0 if success else 1)