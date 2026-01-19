"""
验证VAE编码-加噪-解码对heatmap峰值的影响

实验流程：
1. 从数据集加载heatmap序列
2. 将heatmap转换为colormap格式（与训练时一致）
3. 使用VAE编码heatmap得到latent
4. 对latent添加不同程度的噪声
5. 解码带噪声的latent得到colormap
6. 从colormap中提取heatmap值
7. 比较原始heatmap和解码后heatmap的峰值差异
8. 分析噪声程度与峰值偏差的关系
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from diffsynth.models.wan_video_vae import WanVideoVAE38
from diffsynth.trainers.base_multi_view_dataset_with_rot_grip_3cam_different_projection import (
    RobotTrajectoryDataset, ProjectionInterface
)
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap


def pil_to_tensor(pil_img):
    """将PIL Image转换为tensor"""
    # 返回 (C, H, W), 范围 [-1, 1]
    img = np.array(pil_img, dtype=np.float32) / 255.0
    img = img * 2 - 1  # 转换到 [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return img


def preprocess_video(video_frames_pil, dtype=torch.float32):
    """
    将PIL图像列表转换为VAE输入格式的tensor

    Args:
        video_frames_pil: List[PIL.Image]
        dtype: 目标数据类型

    Returns:
        torch.Tensor: (1, C, T, H, W)
    """
    frames_tensor = torch.stack([pil_to_tensor(f) for f in video_frames_pil], dim=1)  # (C, T, H, W)
    frames_tensor = frames_tensor.unsqueeze(0)  # (1, C, T, H, W)
    return frames_tensor.to(dtype=dtype)


def heatmap_to_colormap_pil(heatmap_tensor):
    """
    将heatmap tensor转换为colormap PIL Image（与训练时的处理一致）

    Args:
        heatmap_tensor: torch.Tensor, shape (T, num_views, H, W)

    Returns:
        List[List[PIL.Image]]: shape (T, num_views)
    """
    T, num_views, H, W = heatmap_tensor.shape

    video_frames = []
    for t in range(T):
        views_at_t = []
        for v in range(num_views):
            heatmap_view = heatmap_tensor[t, v]  # (H, W)
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化到[0, 1]
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap（使用JET colormap，与训练时一致）
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_img = Image.fromarray(view_hm_colored)
            views_at_t.append(pil_img)
        video_frames.append(views_at_t)

    return video_frames


def find_peaks_from_colormap(heatmap_images):
    """
    从colormap PIL Image中提取峰值（与测试代码一致）

    Args:
        heatmap_images: List[List[PIL.Image]] (T, num_views) - 热力图图像

    Returns:
        peak_values: np.ndarray (T, num_views) - 峰值强度
        peak_positions: np.ndarray (T, num_views, 2) - 峰值位置 (x, y)
    """
    num_frames = len(heatmap_images)
    num_views = len(heatmap_images[0])

    peak_values = []
    peak_positions = []

    for frame_idx in range(num_frames):
        frame_peaks = []
        frame_positions = []
        for view_idx in range(num_views):
            heatmap_image = heatmap_images[frame_idx][view_idx]
            # 将PIL Image转换为numpy数组并归一化到[0,1]
            heatmap_image_np = np.array(heatmap_image).astype(np.float32) / 255.0
            # 从colormap中提取heatmap值
            heatmap_array = extract_heatmap_from_colormap(heatmap_image_np, colormap_name='jet')
            # 找到峰值
            max_val = heatmap_array.max()
            max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
            peak = (max_pos[1], max_pos[0])  # (x, y) format

            frame_peaks.append(max_val)
            frame_positions.append(peak)

        peak_values.append(frame_peaks)
        peak_positions.append(frame_positions)

    return np.array(peak_values), np.array(peak_positions)


def decode_latent_to_colormap(vae, latent, device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    """
    将latent解码回colormap PIL Images

    Args:
        vae: VAE模型
        latent: torch.Tensor, VAE latent (num_views, C, T, H, W)
        device: 设备

    Returns:
        List[List[PIL.Image]]: (T, num_views)
    """
    # 使用VAE解码 - decode会遍历第一维(batch/view维)
    decoded = vae.decode(latent, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    # decoded shape: (num_views, C=3, T, H, W)
    # 转换为 (T, num_views, C, H, W)
    num_views, C, T, H, W = decoded.shape
    decoded = decoded.permute(2, 0, 1, 3, 4)  # (T, num_views, C, H, W)

    # 将tensor转换为PIL Images
    video_frames = []
    for t in range(T):
        views_at_t = []
        for v in range(num_views):
            frame = decoded[t, v]  # (C, H, W)
            # 转换为numpy并调整范围
            frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            # VAE输出范围是[-1, 1]，需要转换到[0, 255]
            frame_np = ((frame_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_np)
            views_at_t.append(pil_img)
        video_frames.append(views_at_t)

    return video_frames


def add_noise_to_latent(latent, noise_scale):
    """
    向latent添加高斯噪声

    Args:
        latent: torch.Tensor
        noise_scale: float, 噪声强度

    Returns:
        noisy_latent: torch.Tensor
    """
    noise = torch.randn_like(latent) * noise_scale
    return latent + noise


def main():
    # ========== 配置参数 ==========
    # 数据集参数
    data_root = "/DATA/disk1/lpy_a100_1/Franka_data_3zed_5/cook_5"
    sequence_length = 9  # 必须满足 (sequence_length % 4 == 1) 以匹配VAE的时序压缩机制
    image_size = (256, 256)

    # 模型参数
    model_path = "/DATA/disk1/lpy_a100_1/huggingface/Wan2.2-TI2V-5B-fused"

    # 噪声参数 - 测试不同的噪声强度
    noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    # 实验参数
    num_samples = 3  # 测试的样本数量
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Testing noise scales: {noise_scales}")

    # ========== 初始化数据集 ==========
    print("\n[1/5] Initializing dataset...")

    # 创建投影接口
    projection_interface = ProjectionInterface(
        device=device
    )

    # 创建数据集
    dataset = RobotTrajectoryDataset(
        data_root=data_root,
        projection_interface=projection_interface,
        sequence_length=sequence_length,
        image_size=image_size,
        debug=True,  # 使用debug模式，只加载少量数据
        trail_start=1,
        trail_end=3
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # ========== 初始化VAE ==========
    print("\n[2/5] Initializing VAE model...")

    # 加载VAE模型
    vae = WanVideoVAE38()
    vae_path = f"{model_path}/Wan2.2_VAE.pth"
    print(f"  Loading VAE weights from: {vae_path}")

    vae_state_dict = torch.load(vae_path, map_location="cpu")

    # 处理state_dict格式
    if 'model_state' in vae_state_dict:
        vae_state_dict = vae_state_dict['model_state']

    # 添加'model.'前缀
    vae_state_dict = {'model.' + k: v for k, v in vae_state_dict.items()}

    vae.load_state_dict(vae_state_dict, strict=True)
    vae = vae.eval().to(device=device, dtype=torch.float16)

    print("VAE model loaded successfully")

    # ========== 实验循环 ==========
    print(f"\n[3/5] Running experiments on {num_samples} samples...")

    results = {
        'noise_scales': noise_scales,
        'peak_value_differences': [],  # 存储每个样本在不同噪声下的峰值差异
        'peak_position_differences': [],  # 存储峰值位置差异
    }

    for sample_idx in range(min(num_samples, len(dataset))):
        print(f"\n--- Processing sample {sample_idx + 1}/{num_samples} ---")

        # 获取样本
        sample = dataset[sample_idx]
        heatmap_sequence = sample['heatmap_sequence']  # (T, num_views, H, W)

        print(f"Heatmap sequence shape: {heatmap_sequence.shape}")

        # 1. 将heatmap转换为colormap PIL Images（与训练时处理一致）
        heatmap_colormap_pil = heatmap_to_colormap_pil(heatmap_sequence)

        # 2. 从colormap中提取峰值作为ground truth
        original_peaks, original_positions = find_peaks_from_colormap(heatmap_colormap_pil)
        print(f"Original peak values (mean across views and time): {original_peaks.mean():.4f}")
        print(f"Original peak values range: [{original_peaks.min():.4f}, {original_peaks.max():.4f}]")

        # 3. 使用VAE编码
        num_frames = len(heatmap_colormap_pil)
        num_views = len(heatmap_colormap_pil[0])

        all_view_latents = []
        for view_idx in range(num_views):
            # 提取当前视角的所有帧
            view_frames = [heatmap_colormap_pil[t][view_idx] for t in range(num_frames)]
            # 转换为tensor - 返回 (1, C, T, H, W), 使用float16匹配VAE模型
            view_video = preprocess_video(view_frames, dtype=torch.float16)
            # VAE编码 - encode会遍历第一维(batch维),返回 (1, C, T, H, W)
            with torch.no_grad():
                view_latents = vae.encode(
                    view_video,  # 直接传入,不包装
                    device=device,
                    tiled=True,
                    tile_size=(34, 34),
                    tile_stride=(18, 16)
                ).to(dtype=torch.float16, device=device)  # 保持5D shape (1, C, T, H, W)
            all_view_latents.append(view_latents)

        # 合并所有视角的latents - 沿着batch维拼接得到 (num_views, C, T, H, W)
        latent = torch.cat(all_view_latents, dim=0)
        print(f"Encoded latent shape: {latent.shape}")

        # 4. 测试不同噪声强度
        sample_peak_diffs = []
        sample_position_diffs = []

        for noise_scale in noise_scales:
            # 添加噪声
            if noise_scale > 0:
                noisy_latent = add_noise_to_latent(latent, noise_scale)
            else:
                noisy_latent = latent

            # 解码为colormap
            with torch.no_grad():
                decoded_colormap = decode_latent_to_colormap(vae, noisy_latent, device)

            # 从解码的colormap中提取峰值
            decoded_peaks, decoded_positions = find_peaks_from_colormap(decoded_colormap)

            # 计算峰值差异
            peak_value_diff = np.abs(original_peaks - decoded_peaks).mean()
            peak_position_diff = np.linalg.norm(
                np.array(original_positions) - np.array(decoded_positions),
                axis=-1
            ).mean()

            sample_peak_diffs.append(peak_value_diff)
            sample_position_diffs.append(peak_position_diff)

            print(f"Noise scale {noise_scale:>5.2f}: "
                  f"Peak value diff = {peak_value_diff:.4f}, "
                  f"Position diff = {peak_position_diff:.2f} pixels")

        results['peak_value_differences'].append(sample_peak_diffs)
        results['peak_position_differences'].append(sample_position_diffs)

    # ========== 分析和可视化结果 ==========
    print("\n[4/5] Analyzing results...")

    # 计算平均值和标准差
    peak_diffs_mean = np.array(results['peak_value_differences']).mean(axis=0)
    peak_diffs_std = np.array(results['peak_value_differences']).std(axis=0)
    position_diffs_mean = np.array(results['peak_position_differences']).mean(axis=0)
    position_diffs_std = np.array(results['peak_position_differences']).std(axis=0)

    print("\n=== Summary Statistics ===")
    print(f"{'Noise Scale':<12} {'Peak Value Diff (mean±std)':<30} {'Position Diff (mean±std)'}")
    print("-" * 75)
    for i, noise_scale in enumerate(noise_scales):
        print(f"{noise_scale:<12.2f} {peak_diffs_mean[i]:>8.4f} ± {peak_diffs_std[i]:<8.4f}   "
              f"{position_diffs_mean[i]:>8.2f} ± {position_diffs_std[i]:<8.2f} pixels")

    # ========== 保存可视化图表 ==========
    print("\n[5/5] Generating plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 峰值差异图
    ax1.errorbar(noise_scales, peak_diffs_mean, yerr=peak_diffs_std,
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Scale', fontsize=13)
    ax1.set_ylabel('Peak Value Difference', fontsize=13)
    ax1.set_title('Heatmap Peak Value Difference vs Noise Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    # 确保x轴从noise_scales[1]开始（跳过0）
    if noise_scales[0] == 0:
        ax1.set_xlim(left=noise_scales[1] * 0.5)

    # 峰值位置差异图
    ax2.errorbar(noise_scales[1:], position_diffs_mean[1:], yerr=position_diffs_std[1:],
                 marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Noise Scale', fontsize=13)
    ax2.set_ylabel('Peak Position Difference (pixels)', fontsize=13)
    ax2.set_title('Heatmap Peak Position Difference vs Noise Scale', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()

    # 保存图表
    output_path = Path(__file__).parent / "vae_noise_effect_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # 保存数据
    results_path = Path(__file__).parent / "vae_noise_effect_results.npz"
    np.savez(
        results_path,
        noise_scales=noise_scales,
        peak_diffs_mean=peak_diffs_mean,
        peak_diffs_std=peak_diffs_std,
        position_diffs_mean=position_diffs_mean,
        position_diffs_std=position_diffs_std,
        all_peak_diffs=np.array(results['peak_value_differences']),
        all_position_diffs=np.array(results['peak_position_differences'])
    )
    print(f"Results saved to: {results_path}")

    print("\n=== Experiment completed! ===")

    # 结论
    print("\n=== Conclusions ===")
    print(f"1. Without noise (scale=0.0): Peak value diff = {peak_diffs_mean[0]:.4f}")
    print(f"2. With max noise (scale={noise_scales[-1]}): Peak value diff = {peak_diffs_mean[-1]:.4f}")
    increase_factor = peak_diffs_mean[-1] / (peak_diffs_mean[0] + 1e-8)
    print(f"3. Peak difference increased by {increase_factor:.2f}x from no noise to max noise")
    print(f"4. Peak difference increases {'monotonically' if all(np.diff(peak_diffs_mean) >= 0) else 'non-monotonically'} with noise scale")


if __name__ == "__main__":
    main()
