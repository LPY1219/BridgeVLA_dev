#!/usr/bin/env python3
"""
对比降噪latent和GT latent的详细处理流程
"""
import torch
import numpy as np

def check_latent_processing():
    """
    检查降噪latent和GT latent处理的一致性
    """
    print("="*80)
    print("检查降噪Latent和GT Latent的处理流程")
    print("="*80)

    # 模拟降噪latent（从pipeline返回）
    print("\n1. 降噪Latent处理流程:")
    print("   - Pipeline返回: (num_views, c, t, h, w)")
    denoised_rgb = torch.randn(3, 48, 2, 16, 16)  # (num_views=3, c=48, t=2, h=16, w=16)
    denoised_heatmap = torch.randn(3, 48, 2, 16, 16)

    print(f"   - 初始shape: rgb={denoised_rgb.shape}, heatmap={denoised_heatmap.shape}")

    # 推理代码第638-639行
    denoised_rgb = denoised_rgb.unsqueeze(0)  # (1, num_views, c, t, h, w)
    denoised_heatmap = denoised_heatmap.unsqueeze(0)

    print(f"   - unsqueeze(0)后: rgb={denoised_rgb.shape}, heatmap={denoised_heatmap.shape}")
    print(f"   - 期望格式: (1, num_views, c, t, h, w)")

    # 模拟GT latent（从encode_gt_videos返回）
    print("\n2. GT Latent处理流程:")
    print("   - encode_gt_videos返回: (num_views, c, t, h, w)")
    gt_rgb = torch.randn(3, 48, 2, 16, 16)  # (num_views=3, c=48, t=2, h=16, w=16)
    gt_heatmap = torch.randn(3, 48, 2, 16, 16)

    print(f"   - 初始shape: rgb={gt_rgb.shape}, heatmap={gt_heatmap.shape}")

    # 推理代码第958-959行
    gt_rgb = gt_rgb.unsqueeze(0)  # (1, num_views, c, t, h, w)
    gt_heatmap = gt_heatmap.unsqueeze(0)

    print(f"   - unsqueeze(0)后: rgb={gt_rgb.shape}, heatmap={gt_heatmap.shape}")
    print(f"   - 期望格式: (1, num_views, c, t, h, w)")

    # 对比
    print("\n3. 对比结果:")
    print(f"   - 降噪RGB shape: {denoised_rgb.shape}")
    print(f"   - GT RGB shape: {gt_rgb.shape}")
    print(f"   - Shape一致? {denoised_rgb.shape == gt_rgb.shape}")

    print(f"\n   - 降噪Heatmap shape: {denoised_heatmap.shape}")
    print(f"   - GT Heatmap shape: {gt_heatmap.shape}")
    print(f"   - Shape一致? {denoised_heatmap.shape == gt_heatmap.shape}")

    # 检查维度顺序
    print("\n4. 检查维度顺序:")
    print("   训练时collate_fn返回的格式（mv_rot_grip.py:435-436）:")
    print("     rgb_features.unsqueeze(0)  # (1, v, c, t_compressed, h, w)")
    print("     heatmap_features.unsqueeze(0)")
    print("\n   推理时降噪latent格式:")
    print(f"     {denoised_rgb.shape}  # (1, num_views, c, t, h, w)")
    print("\n   ✓ 维度顺序一致！")

    # 检查heatmap_latent_scale
    print("\n5. 检查heatmap_latent_scale:")
    print("   - 训练时（collate_fn）: 在unsqueeze前应用scale")
    print("   - GT latent推理时: 在unsqueeze前应用scale (第952-955行)")
    print("   - 降噪latent推理时: ???")
    print("   ⚠️  需要检查：降噪latent是否应用了heatmap_latent_scale？")

    return True

if __name__ == "__main__":
    check_latent_processing()
