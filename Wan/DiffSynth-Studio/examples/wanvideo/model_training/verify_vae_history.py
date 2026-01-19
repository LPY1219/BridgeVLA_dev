#!/usr/bin/env python
"""
VAE编码/解码验证脚本

使用真实数据集验证多帧历史条件下VAE的编码和解码是否正确工作。
将重建图像和原始图像放在一起对比可视化。

新的编码规则（保持与VAE预训练一致）：
- 第一帧单独编码 → 1个latent时间步
- 后续每4帧一组编码 → 1个latent时间步

允许的历史帧数量：
- 1帧: 1个条件latent
- 2帧: 第1帧单独编码 + 第2帧单独编码 → 2个条件latent
- 1+4N帧 (5,9,13...): VAE自动处理 → (1+N)个条件latent
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 导入数据集
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam_history import (
    HeatmapDatasetFactoryWithHistory
)


def compute_num_condition_latents(num_history_frames):
    """
    计算条件latent数量（与pipeline中的函数一致）

    编码规则：
    - num_history_frames=1: 1个latent
    - num_history_frames=2: 2个latent（第1帧单独 + 第2帧单独）
    - num_history_frames=1+4N (5,9,13...): 1+N个latent（第一帧单独，后续每4帧一组）
    """
    if num_history_frames <= 1:
        return 1
    elif num_history_frames == 2:
        return 2
    else:
        # 1+4N 形式: 第一帧单独，后续每4帧一组
        return 1 + (num_history_frames - 1 + 3) // 4


def compute_latent_length(num_frames, num_history_frames):
    """
    计算编码后的latent时间维度长度

    编码规则：
    - num_history_frames=1: 正常编码 (num_frames - 1) // 4 + 1
    - num_history_frames=2: 第1帧单独 + (第2帧+剩余帧)正常编码
      = 1 + (num_frames - 2) // 4 + 1
    - num_history_frames=1+4N: 正常编码
    """
    if num_history_frames == 2:
        remaining_frames = num_frames - 1
        return 1 + (remaining_frames - 1) // 4 + 1
    else:
        return (num_frames - 1) // 4 + 1


def tensor_to_pil(tensor):
    """将tensor转换为PIL Image"""
    # tensor: (C, H, W), 范围 [-1, 1]
    img = (tensor + 1) / 2  # 转换到 [0, 1]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(pil_img):
    """将PIL Image转换为tensor"""
    # 返回 (C, H, W), 范围 [-1, 1]
    img = np.array(pil_img, dtype=np.float32) / 255.0
    img = img * 2 - 1  # 转换到 [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return img


def create_comparison_figure(original_frames, reconstructed_frames, title, save_path):
    """
    创建原始帧和重建帧的对比图

    Args:
        original_frames: 原始帧列表 [PIL.Image, ...]
        reconstructed_frames: 重建帧列表 [PIL.Image, ...]
        title: 图片标题
        save_path: 保存路径
    """
    num_frames = len(original_frames)
    num_recon_frames = len(reconstructed_frames)
    max_frames = max(num_frames, num_recon_frames)

    # 创建图形：2行，max_frames列
    fig, axes = plt.subplots(2, max_frames, figsize=(3 * max_frames, 6))

    # 如果只有一列，确保axes是2D数组
    if max_frames == 1:
        axes = axes.reshape(2, 1)

    # 第一行：原始帧
    for i in range(max_frames):
        ax = axes[0, i]
        if i < num_frames:
            ax.imshow(original_frames[i])
            ax.set_title(f'Original {i}', fontsize=10)
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    # 第二行：重建帧
    for i in range(max_frames):
        ax = axes[1, i]
        if i < num_recon_frames:
            ax.imshow(reconstructed_frames[i])
            ax.set_title(f'Recon {i}', fontsize=10)
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    # 添加行标签
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存对比图: {save_path}")


def compute_psnr(original, reconstructed):
    """计算PSNR"""
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 2.0  # 因为范围是 [-1, 1]
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def encode_video_with_history(vae, video, num_history_frames, scale):
    """
    根据历史帧数量编码视频

    Args:
        vae: VAE模型
        video: 视频tensor (B, C, T, H, W)
        num_history_frames: 历史帧数量
        scale: VAE scale参数

    Returns:
        编码后的latent，以及分割信息（用于decode）
    """
    num_frames = video.shape[2]

    if num_history_frames == 2 and num_frames >= 2:
        # 特殊处理：第1帧单独编码，第2帧+剩余帧作为新视频正常编码
        # Frame 0 单独编码
        z_0 = vae.encode(video[:, :, 0:1, :, :], scale)

        # Frame 1 + 剩余帧作为新视频正常编码
        z_remaining = vae.encode(video[:, :, 1:, :, :], scale)

        # Concat: [z_0, z_remaining] 在时间维度
        latent = torch.cat([z_0, z_remaining], dim=2)
        # 返回分割点信息：z_0有1个latent时间步
        split_at = 1
    else:
        # 正常编码（1帧或1+4N帧）
        latent = vae.encode(video, scale)
        split_at = None

    return latent, split_at


def decode_video_with_history(vae, latent, num_history_frames, scale, split_at=None):
    """
    根据历史帧数量解码latent

    Args:
        vae: VAE模型
        latent: latent tensor (B, C, T, H, W)
        num_history_frames: 历史帧数量
        scale: VAE scale参数
        split_at: 分割点（用于2帧历史情况）

    Returns:
        解码后的视频
    """
    if num_history_frames == 2 and split_at is not None:
        # 特殊处理：分开解码
        # z_0 单独解码 → 1帧
        z_0 = latent[:, :, :split_at, :, :]
        recon_0 = vae.decode(z_0, scale)

        # z_remaining 作为新视频解码
        z_remaining = latent[:, :, split_at:, :, :]
        recon_remaining = vae.decode(z_remaining, scale)

        # Concat在时间维度
        reconstructed = torch.cat([recon_0, recon_remaining], dim=2)
    else:
        # 正常解码
        reconstructed = vae.decode(latent, scale)

    return reconstructed


def test_single_video(vae, scale, video_frames_pil, num_history_frames, device, video_type, view_idx, output_dir, sample_idx):
    """
    测试单个视频序列的VAE编码/解码

    Args:
        vae: VAE内部模型
        scale: VAE的scale参数
        video_frames_pil: PIL图像列表
        num_history_frames: 历史帧数量
        device: 设备
        video_type: 视频类型 ("heatmap" 或 "rgb")
        view_idx: 视角索引
        output_dir: 输出目录
        sample_idx: 样本索引

    Returns:
        psnr: PSNR值
    """
    num_frames = len(video_frames_pil)

    # 转换为tensor (B, C, T, H, W)，使用bfloat16
    view_frames_tensor = torch.stack([pil_to_tensor(f) for f in video_frames_pil], dim=1)  # (C, T, H, W)
    view_frames_tensor = view_frames_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # (1, C, T, H, W)

    print(f"      [{video_type}] 输入shape: {view_frames_tensor.shape}")

    # 编码
    with torch.no_grad():
        latent, split_at = encode_video_with_history(vae, view_frames_tensor, num_history_frames, scale)

    print(f"      [{video_type}] Latent shape: {latent.shape}", end="")
    if split_at is not None:
        print(f", split_at={split_at}")
    else:
        print()

    # 解码（使用对应的解码方式）
    with torch.no_grad():
        reconstructed = decode_video_with_history(vae, latent, num_history_frames, scale, split_at)
        reconstructed = reconstructed.clamp(-1, 1)  # 限制范围

    print(f"      [{video_type}] 重建shape: {reconstructed.shape} (原始: {num_frames}帧)")

    # 检查帧数是否匹配
    if reconstructed.shape[2] != num_frames:
        print(f"      ⚠️ [{video_type}] 警告: 帧数不匹配! 原始={num_frames}, 重建={reconstructed.shape[2]}")

    # 计算PSNR（转换为float32计算）
    min_frames = min(num_frames, reconstructed.shape[2])
    psnr = compute_psnr(
        view_frames_tensor[:, :, :min_frames].float(),
        reconstructed[:, :, :min_frames].float()
    )
    print(f"      [{video_type}] PSNR: {psnr:.2f} dB")

    # 转换回PIL图像（只取与原始帧数相同的数量）
    reconstructed_pil = []
    for t in range(min(num_frames, reconstructed.shape[2])):
        recon_frame = reconstructed[0, :, t, :, :].float()  # (C, H, W), 转为float
        reconstructed_pil.append(tensor_to_pil(recon_frame))

    # 创建对比图
    save_dir = os.path.join(output_dir, f"sample_{sample_idx}_history_{num_history_frames}")
    os.makedirs(save_dir, exist_ok=True)

    comparison_path = os.path.join(save_dir, f"view_{view_idx}_{video_type}_comparison.png")
    title = f"Sample {sample_idx}, View {view_idx}, {video_type.upper()}\n" \
            f"History={num_history_frames}, Latent T={latent.shape[2]}, PSNR={psnr:.2f}dB"
    create_comparison_figure(video_frames_pil, reconstructed_pil, title, comparison_path)

    return psnr


def test_vae_with_real_data(vae, scale, sample, num_history_frames, device, output_dir, sample_idx):
    """
    使用真实数据测试VAE的编码和解码（同时测试heatmap和RGB）

    Args:
        vae: VAE内部模型 (VideoVAE_)
        scale: VAE的scale参数
        sample: 数据集样本
        num_history_frames: 历史帧数量
        device: 设备
        output_dir: 输出目录
        sample_idx: 样本索引
    """
    print(f"\n{'='*60}")
    print(f"样本 {sample_idx}, 历史帧数={num_history_frames}")

    # 从样本中提取视频帧
    # sample['video'] 是 heatmap 序列: List[List[PIL.Image]]，格式为 (T, num_views)
    # sample['input_video_rgb'] 是 RGB 序列: List[List[PIL.Image]]，格式为 (T, num_views)
    heatmap_frames = sample['video']
    rgb_frames = sample.get('input_video_rgb', None)

    num_frames = len(heatmap_frames)
    num_views = len(heatmap_frames[0])

    print(f"  视频帧数: {num_frames}")
    print(f"  视角数: {num_views}")
    print(f"  RGB数据: {'有' if rgb_frames is not None else '无'}")

    # 对每个视角进行处理
    for view_idx in range(num_views):
        print(f"\n  视角 {view_idx}:")

        # 测试 Heatmap
        heatmap_pil = [heatmap_frames[t][view_idx] for t in range(num_frames)]
        psnr_heatmap = test_single_video(
            vae, scale, heatmap_pil, num_history_frames, device,
            "heatmap", view_idx, output_dir, sample_idx
        )

        # 测试 RGB（如果有）
        if rgb_frames is not None:
            rgb_pil = [rgb_frames[t][view_idx] for t in range(num_frames)]
            psnr_rgb = test_single_video(
                vae, scale, rgb_pil, num_history_frames, device,
                "rgb", view_idx, output_dir, sample_idx
            )

    print(f"\n  ✓ 视角处理完成")


def load_vae(model_base_path, device):
    """
    加载VAE模型

    Args:
        model_base_path: 模型基础路径 (包含 Wan2.2_VAE.pth 的目录)
        device: 设备

    Returns:
        加载好的VAE模型
    """
    from diffsynth.models.wan_video_vae import WanVideoVAE38

    print("加载VAE模型...")
    vae = WanVideoVAE38()

    # 加载权重
    vae_path = os.path.join(model_base_path, "Wan2.2_VAE.pth")
    print(f"  加载权重: {vae_path}")
    vae_state_dict = torch.load(vae_path, map_location="cpu")

    # 处理state_dict格式
    if 'model_state' in vae_state_dict:
        vae_state_dict = vae_state_dict['model_state']

    # 添加'model.'前缀
    vae_state_dict = {'model.' + k: v for k, v in vae_state_dict.items()}

    vae.load_state_dict(vae_state_dict, strict=True)
    vae = vae.eval().to(device=device, dtype=torch.bfloat16)

    print(f"  VAE模型类型: {type(vae).__name__}")
    print(f"  VAE内部模型类型: {type(vae.model).__name__}")
    print("✓ VAE加载完成")

    return vae


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VAE编码/解码验证（使用真实数据集）")
    parser.add_argument("--model_path", type=str, default="/DATA/disk1/lpy_a100_4/huggingface/Wan2.2-TI2V-5B-fused",
                        help="VAE模型路径（包含Wan2.2_VAE.pth的目录）")
    parser.add_argument("--data_root", type=str, default="/DATA/disk1/lpy_a100_4/data/Franka_data_3zed_2/put_red_bull_in_pink_plate",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./vae_verify_output",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="测试样本数量")
    parser.add_argument("--sequence_length", type=int, default=12,
                        help="序列长度")
    parser.add_argument("--num_history_frames", type=int, default=2,
                        help="历史帧数量 (1, 2, 或 1+4N)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载VAE模型
    vae = load_vae(args.model_path, device)

    # 创建数据集
    print(f"\n加载数据集: {args.data_root}")
    print(f"  sequence_length: {args.sequence_length}")
    print(f"  num_history_frames: {args.num_history_frames}")

    try:
        dataset = HeatmapDatasetFactoryWithHistory.create_robot_trajectory_dataset(
            data_root=args.data_root,
            sequence_length=args.sequence_length,
            step_interval=1,
            min_trail_length=5,
            image_size=(256, 256),
            sigma=1.5,
            augmentation=False,  # 禁用数据增强以便于验证
            mode="train",
            scene_bounds=[0, -0.45, -0.05, 0.8, 0.55, 0.6],
            debug=True,  # 使用少量数据
            wan_type="5B_TI2V_RGB_HEATMAP_MV_HISTORY",
            num_history_frames=args.num_history_frames,
            use_different_projection=True,
        )
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试多个样本
    print(f"\n{'='*60}")
    print(f"开始测试 (num_history_frames={args.num_history_frames})...")
    print(f"编码规则:")
    print(f"  - 1帧: 单独编码 → 1个latent")
    print(f"  - 2帧: 第1帧单独 + 第2帧+剩余帧正常编码")
    print(f"  - 1+4N帧: VAE自动处理 → (1+N)个latent")

    num_samples = min(args.num_samples, len(dataset))
    for i in range(num_samples):
        try:
            sample = dataset[i]
            test_vae_with_real_data(
                vae.model,  # 使用内部模型
                vae.scale,  # 传递scale参数
                sample,
                args.num_history_frames,
                device,
                args.output_dir,
                i
            )
        except Exception as e:
            print(f"\n❌ 样本 {i} 测试失败:")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"测试完成! 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
