"""
测试不同resolution的精度和性能
"""
import sys
import os
import numpy as np
import time

# 添加路径
diffsynth_path = "/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio"
sys.path.insert(0, diffsynth_path)

from diffsynth.trainers.heatmap_utils import (
    convert_heatmap_to_colormap,
    extract_heatmap_from_colormap
)

print("="*70)
print("测试不同Resolution的精度和性能")
print("="*70)

# 创建测试数据：13帧 × 3视角
num_frames = 13
num_views = 3
total_images = num_frames * num_views

print(f"\n创建 {total_images} 个测试heatmap (256x256)...")
heatmaps = []
colormaps = []

for i in range(total_images):
    # 随机位置的高斯峰值
    center_y = np.random.randint(64, 192)
    center_x = np.random.randint(64, 192)
    sigma = np.random.uniform(10, 20)

    y, x = np.ogrid[0:256, 0:256]
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    heatmap = (gaussian / gaussian.max()).astype(np.float32)
    heatmaps.append(heatmap)

    # 转换为colormap
    colormap = convert_heatmap_to_colormap(heatmap, 'viridis')
    colormaps.append(colormap)

print(f"测试数据准备完成！\n")

# 测试不同resolution
resolutions = [32, 64, 128]

for res in resolutions:
    print("="*70)
    print(f"Resolution = {res}")
    print("-"*70)

    # 首次运行会构建LUT
    print(f"首次运行（构建LUT）...")
    t_build_start = time.time()
    _ = extract_heatmap_from_colormap(colormaps[0], 'viridis', resolution=res)
    t_build_end = time.time()
    build_time = (t_build_end - t_build_start) * 1000
    print(f"  LUT构建时间: {build_time:.1f}ms")

    # 测试性能（使用缓存的LUT）
    print(f"\n性能测试（{total_images}张图像）...")
    t_start = time.time()
    recovered_heatmaps = []
    for colormap in colormaps:
        recovered = extract_heatmap_from_colormap(colormap, 'viridis', resolution=res)
        recovered_heatmaps.append(recovered)
    t_end = time.time()

    total_time = (t_end - t_start) * 1000
    avg_time = total_time / total_images

    print(f"  总时间: {total_time:.1f}ms")
    print(f"  平均每张: {avg_time:.2f}ms")
    print(f"  吞吐量: {1000/avg_time:.1f} images/s")

    # 测试精度
    print(f"\n精度测试...")
    all_errors = []
    peak_errors = []

    for gt, recovered in zip(heatmaps, recovered_heatmaps):
        abs_err = np.abs(gt - recovered)
        all_errors.append(abs_err)

        # 峰值位置
        gt_peak = np.unravel_index(np.argmax(gt), gt.shape)
        rec_peak = np.unravel_index(np.argmax(recovered), recovered.shape)
        peak_err = np.sqrt((gt_peak[0] - rec_peak[0])**2 + (gt_peak[1] - rec_peak[1])**2)
        peak_errors.append(peak_err)

    all_errors = np.array(all_errors)
    peak_errors = np.array(peak_errors)

    print(f"  值误差:")
    print(f"    最大: {all_errors.max():.6f}")
    print(f"    平均: {all_errors.mean():.6f}")
    print(f"    中位数: {np.median(all_errors):.6f}")
    print(f"  峰值位置误差:")
    print(f"    平均: {peak_errors.mean():.3f} 像素")
    print(f"    最大: {peak_errors.max():.3f} 像素")
    print(f"    中位数: {np.median(peak_errors):.3f} 像素")
    print(f"    <1像素: {(peak_errors < 1.0).sum()}/{total_images} ({(peak_errors < 1.0).sum()/total_images*100:.1f}%)")
    print(f"    完美匹配: {(peak_errors == 0).sum()}/{total_images} ({(peak_errors == 0).sum()/total_images*100:.1f}%)")

    # 检查LUT大小
    from diffsynth.trainers.heatmap_utils import _colormap_lut_cache
    if ('viridis', res) in _colormap_lut_cache:
        lut = _colormap_lut_cache[('viridis', res)]
        lut_size_kb = lut.nbytes / 1024
        print(f"\n  LUT内存: {lut_size_kb:.1f} KB")

print("\n" + "="*70)
print("测试完成！")
print("="*70)

print("\n推荐配置:")
print("  - 对于实时应用且可接受2像素误差: resolution=32")
print("  - 对于需要高精度（<1像素）: resolution=64或128")
print("  - resolution=64是速度和精度的最佳平衡")
