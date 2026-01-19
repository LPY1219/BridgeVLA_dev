"""
Heatmap to Colormap conversion utilities for Wan2.2 training
基于reconstruct_heatmap/test_heatmap_peak_accuracy.py的热力图转换工具
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Union, Tuple
from matplotlib import cm
from copy import deepcopy
import cv2


def convert_heatmap_to_colormap(heatmap, colormap_name='jet'):
    """
    Convert heatmap to RGB image using matplotlib colormap (optimized)
    基于test_heatmap_peak_accuracy.py中的实现
    """
    # Normalize heatmap to [0, 1] if needed
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Apply colormap (vectorized operation)
    colormap = cm.get_cmap(colormap_name)
    rgb_image = colormap(heatmap_norm)[:, :, :3]  # Remove alpha channel

    return rgb_image.astype(np.float32)


def extract_heatmap_from_colormap(rgb_image, colormap_name='jet', resolution=64):
    """
    Extract heatmap from RGB colormap image by finding closest colormap values (adaptive)
    基于test_heatmap_peak_accuracy.py中的实现

    修改：统一使用 cv2 的 JET colormap，与数据准备时保持一致

    优化版本：使用预计算的3D查找表（LUT）大幅加速

    Args:
        rgb_image: RGB图像，范围[0,1]
        colormap_name: colormap名称（目前统一使用 'jet'）
        resolution: 查找表分辨率 (默认64)
                   32: 快速，峰值误差~2像素，128KB内存，构建时间~0.3s
                   64: 平衡，峰值误差~0.5像素，1MB内存，构建时间~2s (推荐)
                   128: 高精度，峰值误差<0.3像素，8MB内存，构建时间~15s

    Returns:
        extracted_heatmap: 提取的heatmap，范围[0,1]
    """
    h, w = rgb_image.shape[:2]

    # 使用预计算的查找表（基于cv2 JET colormap）
    lut = _get_colormap_lut_cv2(colormap_name, resolution=resolution)

    # 将RGB图像量化到查找表的分辨率
    # rgb_image范围是[0,1]，需要映射到[0,resolution-1]
    max_idx = resolution - 1
    rgb_quantized = (rgb_image * (resolution - 0.001)).astype(np.int32)
    rgb_quantized = np.clip(rgb_quantized, 0, max_idx)

    # 使用查找表直接获取heatmap值（极快的向量化操作）
    extracted_heatmap = lut[rgb_quantized[:, :, 0], rgb_quantized[:, :, 1], rgb_quantized[:, :, 2]]

    return extracted_heatmap.astype(np.float32)


# 全局缓存查找表
_colormap_lut_cache = {}
_colormap_lut_cv2_cache = {}

def _get_colormap_lut_cv2(colormap_name='jet', resolution=32):
    """
    获取或创建基于 cv2 的 colormap 3D查找表
    与数据准备时使用的 cv2.COLORMAP_JET 保持一致

    Args:
        colormap_name: colormap名称（目前统一使用 'jet'）
        resolution: 每个RGB通道的分辨率（默认32，即32^3=32768个条目）

    Returns:
        lut: (resolution, resolution, resolution) 的查找表
    """
    cache_key = (colormap_name, resolution)

    if cache_key not in _colormap_lut_cv2_cache:
        print(f"Building cv2-based colormap LUT for {colormap_name} with resolution {resolution}...")

        # 使用 cv2 的 JET colormap 生成参考颜色（与数据准备时一致）
        num_reference = 256  # 更精细的参考值
        reference_values = np.linspace(0, 1, num_reference)

        # 为每个参考值生成对应的 cv2 JET 颜色
        reference_colors = []
        for val in reference_values:
            # 创建单像素的归一化值
            hm_uint8 = np.uint8(val * 255)
            # 应用 cv2 JET colormap
            color_bgr = cv2.applyColorMap(np.array([[hm_uint8]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            # 转换 BGR 到 RGB 并归一化到 [0, 1]
            color_rgb = color_bgr[::-1] / 255.0
            reference_colors.append(color_rgb)

        reference_colors = np.array(reference_colors)  # (256, 3)

        # 创建3D查找表
        lut = np.zeros((resolution, resolution, resolution), dtype=np.float32)

        # 对于每个可能的量化RGB值，找到最近的colormap颜色
        # 使用向量化方法加速构建
        r_vals = np.linspace(0, 1, resolution)
        g_vals = np.linspace(0, 1, resolution)
        b_vals = np.linspace(0, 1, resolution)

        # 创建所有可能的RGB组合
        r_grid, g_grid, b_grid = np.meshgrid(r_vals, g_vals, b_vals, indexing='ij')
        rgb_grid = np.stack([r_grid, g_grid, b_grid], axis=-1).reshape(-1, 3)  # (resolution^3, 3)

        # 批量计算最近邻（分批处理以节省内存）
        batch_size = 4096
        heatmap_values = np.zeros(rgb_grid.shape[0], dtype=np.float32)

        for i in range(0, rgb_grid.shape[0], batch_size):
            batch = rgb_grid[i:i+batch_size]  # (batch_size, 3)
            # 计算距离：(batch_size, 256)
            distances = np.sum((batch[:, None, :] - reference_colors[None, :, :]) ** 2, axis=2)
            closest_indices = np.argmin(distances, axis=1)
            heatmap_values[i:i+batch_size] = reference_values[closest_indices]

        lut = heatmap_values.reshape(resolution, resolution, resolution)
        _colormap_lut_cv2_cache[cache_key] = lut
        print(f"cv2-based colormap LUT built successfully!")

    return _colormap_lut_cv2_cache[cache_key]

def _get_colormap_lut(colormap_name='jet', resolution=32):
    """
    获取或创建colormap的3D查找表

    Args:
        colormap_name: colormap名称
        resolution: 每个RGB通道的分辨率（默认32，即32^3=32768个条目）

    Returns:
        lut: (resolution, resolution, resolution) 的查找表
    """
    cache_key = (colormap_name, resolution)

    if cache_key not in _colormap_lut_cache:
        print(f"Building colormap LUT for {colormap_name} with resolution {resolution}...")
        colormap = cm.get_cmap(colormap_name)

        # 创建参考值和颜色
        num_reference = 256  # 更精细的参考值
        reference_values = np.linspace(0, 1, num_reference)
        reference_colors = colormap(reference_values)[:, :3]  # (256, 3)

        # 创建3D查找表
        lut = np.zeros((resolution, resolution, resolution), dtype=np.float32)

        # 对于每个可能的量化RGB值，找到最近的colormap颜色
        # 使用向量化方法加速构建
        r_vals = np.linspace(0, 1, resolution)
        g_vals = np.linspace(0, 1, resolution)
        b_vals = np.linspace(0, 1, resolution)

        # 创建所有可能的RGB组合
        r_grid, g_grid, b_grid = np.meshgrid(r_vals, g_vals, b_vals, indexing='ij')
        rgb_grid = np.stack([r_grid, g_grid, b_grid], axis=-1).reshape(-1, 3)  # (resolution^3, 3)

        # 批量计算最近邻（分批处理以节省内存）
        batch_size = 4096
        heatmap_values = np.zeros(rgb_grid.shape[0], dtype=np.float32)

        for i in range(0, rgb_grid.shape[0], batch_size):
            batch = rgb_grid[i:i+batch_size]  # (batch_size, 3)
            # 计算距离：(batch_size, 256)
            distances = np.sum((batch[:, None, :] - reference_colors[None, :, :]) ** 2, axis=2)
            closest_indices = np.argmin(distances, axis=1)
            heatmap_values[i:i+batch_size] = reference_values[closest_indices]

        lut = heatmap_values.reshape(resolution, resolution, resolution)
        _colormap_lut_cache[cache_key] = lut
        print(f"Colormap LUT built successfully!")

    return _colormap_lut_cache[cache_key]


def convert_color_to_wan_format(image):
    """
    Convert color image to Wan-VAE format
    基于test_heatmap_peak_accuracy.py中的实现
    """
    # Convert to torch tensor and permute dimensions
    # Shape: (H, W, 3) -> (3, H, W) -> (1, 3, 1, H, W)
    image_tensor = torch.from_numpy(image).float()
    image_chw = image_tensor.permute(2, 0, 1)  # H,W,C -> C,H,W
    image_5d = image_chw.unsqueeze(0).unsqueeze(2)  # Add batch and time dimensions

    return image_5d


def convert_from_wan_format(decoded_5d):
    """
    Convert decoded output back to image format
    基于test_heatmap_peak_accuracy.py中的实现
    """
    # Shape: (1, 3, 1, H, W) -> (3, H, W) -> (H, W, 3)
    decoded_chw = decoded_5d.squeeze(0).squeeze(1)  # Remove batch and time dims
    decoded_hwc = decoded_chw.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C

    # Clamp to [0, 1] range
    decoded_hwc = np.clip(decoded_hwc, 0, 1)

    return decoded_hwc


def heatmap_sequence_to_pil_images(heatmap_sequence: Union[np.ndarray, torch.Tensor],
                                 colormap_name: str = 'jet') -> List[Image.Image]:
    """
    将热力图序列转换为PIL图像列表，用于Wan2.2训练

    Args:
        heatmap_sequence: 热力图序列 (T, H, W)
        colormap_name: colormap名称

    Returns:
        PIL图像列表
    """
    # 转换为numpy
    if isinstance(heatmap_sequence, torch.Tensor):
        heatmap_sequence = heatmap_sequence.cpu().numpy()

    pil_images = []
    for t in range(heatmap_sequence.shape[0]):
        heatmap = heatmap_sequence[t]  # (H, W)

        # 转换为colormap
        colormap_image = convert_heatmap_to_colormap(heatmap, colormap_name)

        # 转换为PIL图像 (需要将值范围调整到[0, 255])
        colormap_uint8 = (colormap_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(colormap_uint8)
        pil_images.append(pil_image)

    return pil_images


def rgb_tensor_to_pil_image(rgb_tensor: torch.Tensor) -> Image.Image:
    """
    将RGB tensor转换为PIL图像

    Args:
        rgb_tensor: RGB tensor (3, H, W) 或 (H, W, 3)
                   支持范围：[-1, 1] 或 [0, 1] 或 [0, 255]

    Returns:
        PIL图像
    """
    # 转换为numpy
    rgb_array = rgb_tensor.cpu().numpy()

    # 处理通道顺序
    if rgb_array.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        rgb_array = rgb_array.transpose(1, 2, 0)

    # 智能检测数值范围并归一化到 [0, 255]
    min_val = rgb_array.min()
    max_val = rgb_array.max()

    if min_val < 0:  # 范围是 [-1, 1]
        # 从 [-1, 1] 转到 [0, 1]
        rgb_array = (rgb_array + 1.0) / 2.0
        rgb_array = np.clip(rgb_array, 0, 1)
        rgb_array = (rgb_array * 255).astype(np.uint8)
    elif max_val <= 1.0:  # 范围是 [0, 1]
        rgb_array = (rgb_array * 255).astype(np.uint8)
    else:  # 范围是 [0, 255]
        rgb_array = rgb_array.astype(np.uint8)

    return Image.fromarray(rgb_array)


def prepare_heatmap_data_for_wan_5B_TI2V(rgb_image: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet') -> dict:
    """
    为Wan2.2训练准备热力图数据，转换为所需格式

    Args:
        rgb_image: RGB图像 tensor (3, H, W)
        heatmap_sequence: 热力图序列 tensor (T, H, W)
        instruction: 文本指令
        colormap_name: colormap名称

    Returns:
        格式化的数据字典，符合UnifiedDataset要求
    """
    # 1. 转换RGB图像为PIL
    first_frame = rgb_tensor_to_pil_image(rgb_image)

    # 2. 转换热力图序列为PIL图像列表
    video_frames = heatmap_sequence_to_pil_images(heatmap_sequence, colormap_name)

    # 3. 构建数据字典
    data = {
        'prompt': instruction,
        'video': video_frames,
        'input_image': first_frame  # 首帧RGB作为条件输入
    }

    return data


def prepare_heatmap_data_for_wan_14B_I2V(rgb_image: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                heatmap_start: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet') -> dict:
    """
    为Wan2.2训练准备热力图数据，转换为所需格式

    Args:
        rgb_image: RGB图像 tensor (3, H, W)
        heatmap_sequence: 热力图序列 tensor (T, H, W)
        heatmap_start: 起始热力图 tensor (1,H, W)
        instruction: 文本指令
        colormap_name: colormap名称

    Returns:
        格式化的数据字典，符合UnifiedDataset要求
    """
    # 1. 转换RGB图像为PIL
    first_frame = rgb_tensor_to_pil_image(rgb_image)

    # 2. 转换热力图序列为PIL图像列表
    heatmap_sequence = torch.cat([heatmap_start, heatmap_sequence], dim=0)  # (T+1,H,W)
    video_frames = heatmap_sequence_to_pil_images(heatmap_sequence, colormap_name)

    # 3. 构建数据字典
    data = {
        'prompt': instruction,
        'video': video_frames,
        'input_image': deepcopy(video_frames[0]),  # 首帧热力图作为条件输入
        "condition_rgb":first_frame # 首帧rgb作为额外条件输入
    }
    return data



def prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP(rgb_image: torch.Tensor,
                                rgb_sequence: torch.Tensor,
                                heatmap_start: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet') -> dict:
    """
    为Wan2.2训练准备热力图数据，转换为所需格式

    Args:
        rgb_image: RGB图像 tensor (3, H, W)
        rgb_sequence: RGB图像序列 tensor (T,3 H, W)
        heatmap_start: 起始热力图 tensor (1,H, W)
        heatmap_sequence: 热力图序列 tensor (T,H, W)
        instruction: 文本指令
        colormap_name: colormap名称

    Returns:
        格式化的数据字典，符合UnifiedDataset要求
    """
    # 1. 转换RGB图像为PIL
    rgb_image = rgb_image.unsqueeze(0)  # (3,H,W) -> (1,3,H,W)
    rgb_sequence = torch.cat([rgb_image, rgb_sequence], dim=0)  # (T+1,3,H,W)
    video_frames_rgb = [rgb_tensor_to_pil_image(rgb_sequence[t]) for t in range(rgb_sequence.shape[0])]

    # 2. 转换热力图序列为PIL图像列表
    heatmap_sequence = torch.cat([heatmap_start, heatmap_sequence], dim=0)  # (T+1,H,W)
    video_frames = heatmap_sequence_to_pil_images(heatmap_sequence, colormap_name)

    # 3. 构建数据字典
    data = {
        'prompt': instruction,
        'video': video_frames,
        'input_image': deepcopy(video_frames[0]),  # 首帧热力图作为条件输入
        "input_image_rgb":deepcopy(video_frames_rgb[0]) , 
        "input_video_rgb": video_frames_rgb,
    }
    return data


def prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW(
                                rgb_image: torch.Tensor,
                                rgb_sequence: torch.Tensor,
                                heatmap_start: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet') -> dict:
    """
    为Wan2.2训练准备热力图数据，转换为所需格式 (支持多视角)

    Args:
        rgb_image: RGB图像 tensor (num_views, 3, H, W) - 多视角起始帧
        rgb_sequence: RGB图像序列 tensor (T, num_views, 3, H, W) - 多视角未来帧序列
        heatmap_start: 起始热力图 tensor (1, num_views, H, W) - 多视角起始热力图
        heatmap_sequence: 热力图序列 tensor (T, num_views, H, W) - 多视角未来热力图序列
        instruction: 文本指令
        colormap_name: colormap名称

    Returns:
        格式化的数据字典，符合UnifiedDataset要求
        返回的数据包含多视角信息：
        - video: List[List[PIL.Image]] - 每个时间步包含多个视角的热力图
        - input_video_rgb: List[List[PIL.Image]] - 每个时间步包含多个视角的RGB图像
    """
    # 获取维度信息
    num_views = rgb_image.shape[0]  # 通常为3
    num_future_frames = rgb_sequence.shape[0]  # T

    # 1. 处理RGB图像序列 (多视角)
    # rgb_image: (num_views, 3, H, W) -> (1, num_views, 3, H, W)
    rgb_image_expanded = rgb_image.unsqueeze(0)
    # rgb_sequence: (T, num_views, 3, H, W) + rgb_image -> (T+1, num_views, 3, H, W)
    rgb_sequence_full = torch.cat([rgb_image_expanded, rgb_sequence], dim=0)  # (T+1, num_views, 3, H, W)

    # 转换为PIL图像: 每个时间步包含num_views个视角
    video_frames_rgb = []
    for t in range(rgb_sequence_full.shape[0]):
        views_at_t = []
        for v in range(num_views):
            rgb_view = rgb_sequence_full[t, v]  # (3, H, W)
            pil_img = rgb_tensor_to_pil_image(rgb_view)
            views_at_t.append(pil_img)
        video_frames_rgb.append(views_at_t)  # List of List[PIL.Image]

    # 2. 处理热力图序列 (多视角)
    # heatmap_start: (1, num_views, H, W)
    # heatmap_sequence: (T, num_views, H, W)
    heatmap_sequence_full = torch.cat([heatmap_start, heatmap_sequence], dim=0)  # (T+1, num_views, H, W)

    # 转换为PIL图像: 每个时间步包含num_views个视角
    # 使用与 RoboWan_client_3zed.py 一致的处理方式
    video_frames = []
    for t in range(heatmap_sequence_full.shape[0]):
        views_at_t = []
        for v in range(num_views):
            heatmap_view = heatmap_sequence_full[t, v]  # (H, W)
            # 转换单个热力图为PIL（与client端一致的处理）
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化到[0, 1]
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap（使用JET colormap，与client端一致）
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_img = Image.fromarray(view_hm_colored)
            views_at_t.append(pil_img)
        video_frames.append(views_at_t)  # List of List[PIL.Image]

    # 3. 构建数据字典
    data = {
        'prompt': instruction,
        'video': video_frames,  # List[List[PIL.Image]] - (T+1) x num_views
        'input_image': deepcopy(video_frames[0]),  # 首帧的多视角热力图 List[PIL.Image]
        "input_image_rgb": deepcopy(video_frames_rgb[0]),  # 首帧的多视角RGB List[PIL.Image]
        "input_video_rgb": video_frames_rgb,  # List[List[PIL.Image]] - (T+1) x num_views
    }
    return data


def prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_ROT_GRIP(
                                rgb_image: torch.Tensor,
                                rgb_sequence: torch.Tensor,
                                heatmap_start: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet',
                                start_pose: torch.Tensor = None,
                                future_poses: torch.Tensor = None,
                                start_gripper_state: bool = None,
                                future_gripper_states: torch.Tensor = None,
                                rotation_resolution: float = 5.0) -> dict:
    """
    为Wan2.2训练准备热力图数据，转换为所需格式 (支持多视角 + rotation + gripper)

    Args:
        rgb_image: RGB图像 tensor (num_views, 3, H, W) - 多视角起始帧
        rgb_sequence: RGB图像序列 tensor (T, num_views, 3, H, W) - 多视角未来帧序列
        heatmap_start: 起始热力图 tensor (1, num_views, H, W) - 多视角起始热力图
        heatmap_sequence: 热力图序列 tensor (T, num_views, H, W) - 多视角未来热力图序列
        instruction: 文本指令
        colormap_name: colormap名称
        start_pose: 起始pose tensor (7,) - [x, y, z, qx, qy, qz, qw]
        future_poses: 未来poses tensor (T, 7)
        start_gripper_state: 起始gripper state (bool)
        future_gripper_states: 未来gripper states tensor (T,)
        rotation_resolution: 旋转角度离散化分辨率（度），默认5度

    Returns:
        格式化的数据字典，符合UnifiedDataset要求
        返回的数据包含多视角信息 + rotation + gripper:
        - video: List[List[PIL.Image]] - 每个时间步包含多个视角的热力图
        - input_video_rgb: List[List[PIL.Image]] - 每个时间步包含多个视角的RGB图像
        - start_pose: torch.Tensor (7,) - 起始pose
        - future_poses: torch.Tensor (T, 7) - 未来poses
        - start_gripper_state: bool - 起始gripper state
        - future_gripper_states: torch.Tensor (T,) - 未来gripper states
        - start_rotation: torch.Tensor (3,) - 起始帧的离散化rotation索引 [roll_bin, pitch_bin, yaw_bin]
        - start_gripper: torch.Tensor (scalar) - 起始帧的离散化gripper索引 (0或1)
        - rotation_targets: torch.Tensor (T, 3) - 未来帧的离散化rotation索引 [roll_bin, pitch_bin, yaw_bin]
        - gripper_targets: torch.Tensor (T,) - 未来帧的离散化gripper索引 (0或1)
    """
    # 获取维度信息
    num_views = rgb_image.shape[0]  # 通常为3
    num_future_frames = rgb_sequence.shape[0]  # T

    # 1. 处理RGB图像序列 (多视角)
    # rgb_image: (num_views, 3, H, W) -> (1, num_views, 3, H, W)
    rgb_image_expanded = rgb_image.unsqueeze(0)
    # rgb_sequence: (T, num_views, 3, H, W) + rgb_image -> (T+1, num_views, 3, H, W)
    rgb_sequence_full = torch.cat([rgb_image_expanded, rgb_sequence], dim=0)  # (T+1, num_views, 3, H, W)

    # 转换为PIL图像: 每个时间步包含num_views个视角
    video_frames_rgb = []
    for t in range(rgb_sequence_full.shape[0]):
        views_at_t = []
        for v in range(num_views):
            rgb_view = rgb_sequence_full[t, v]  # (3, H, W)
            pil_img = rgb_tensor_to_pil_image(rgb_view)
            views_at_t.append(pil_img)
        video_frames_rgb.append(views_at_t)  # List of List[PIL.Image]

    # 2. 处理热力图序列 (多视角)
    # heatmap_start: (1, num_views, H, W)
    # heatmap_sequence: (T, num_views, H, W)
    heatmap_sequence_full = torch.cat([heatmap_start, heatmap_sequence], dim=0)  # (T+1, num_views, H, W)

    # 转换为PIL图像: 每个时间步包含num_views个视角
    # 使用与 RoboWan_client_3zed.py 一致的处理方式
    video_frames = []
    for t in range(heatmap_sequence_full.shape[0]):
        views_at_t = []
        for v in range(num_views):
            heatmap_view = heatmap_sequence_full[t, v]  # (H, W)
            # 转换单个热力图为PIL（与client端一致的处理）
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化到[0, 1]
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap（使用JET colormap，与client端一致）
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_img = Image.fromarray(view_hm_colored)
            views_at_t.append(pil_img)
        video_frames.append(views_at_t)  # List of List[PIL.Image]

    # 3. 构建数据字典
    data = {
        'prompt': instruction,
        'video': video_frames,  # List[List[PIL.Image]] - (T+1) x num_views
        'input_image': deepcopy(video_frames[0]),  # 首帧的多视角热力图 List[PIL.Image]
        "input_image_rgb": deepcopy(video_frames_rgb[0]),  # 首帧的多视角RGB List[PIL.Image]
        "input_video_rgb": video_frames_rgb,  # List[List[PIL.Image]] - (T+1) x num_views
        'heatmap_raw': heatmap_sequence_full,  # torch.Tensor (T+1, num_views, H, W) - 原始热力图数据
    }

    # 4. 添加原始的rotation和gripper数据
    if start_pose is not None:
        data['start_pose'] = start_pose  # (7,)
    if future_poses is not None:
        data['future_poses'] = future_poses  # (T, 7)
    if start_gripper_state is not None:
        data['start_gripper_state'] = start_gripper_state  # bool
    if future_gripper_states is not None:
        data['future_gripper_states'] = future_gripper_states  # (T,)

    # 5. 处理rotation targets：将四元数转换为离散化的欧拉角索引
    # 辅助函数：将四元数转换为离散化的欧拉角索引
    def quaternion_to_discrete_euler(quat, rotation_resolution):
        """将单个四元数转换为离散化的欧拉角索引"""
        from scipy.spatial.transform import Rotation

        # 归一化四元数
        quat_normalized = quat / np.linalg.norm(quat)

        # 确保w为正数
        if quat_normalized[3] < 0:
            quat_normalized = -quat_normalized

        # 使用scipy的Rotation转换（scipy使用[x, y, z, w]顺序）
        r = Rotation.from_quat(quat_normalized)
        euler = r.as_euler("xyz", degrees=True)  # (3,) - [roll, pitch, yaw]

        # 应用gimble fix
        if 89 < euler[1] < 91:
            euler[1] = 90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)
        elif -91 < euler[1] < -89:
            euler[1] = -90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)

        # 将范围从[-180, 180]转换为[0, 360]
        euler += 180

        # 离散化
        disc = np.around(euler / rotation_resolution).astype(np.int64)
        # 处理边界情况：360度 = 0度
        num_bins = int(360 / rotation_resolution)
        disc[disc == num_bins] = 0

        return disc  # (3,) - [roll_bin, pitch_bin, yaw_bin]

    # 处理start_rotation（从start_pose中提取）
    if start_pose is not None:
        # start_pose: (7,) - [x, y, z, qx, qy, qz, qw]
        start_quat = start_pose[3:7].cpu().numpy()  # (4,) - [qx, qy, qz, qw]
        start_rotation = quaternion_to_discrete_euler(start_quat, rotation_resolution)
        data['start_rotation'] = torch.from_numpy(start_rotation).long()  # (3,)

    # 处理rotation targets（从future_poses中提取）
    if future_poses is not None:
        # future_poses: (T, 7) - [x, y, z, qx, qy, qz, qw]
        quaternions = future_poses[:, 3:7].cpu().numpy()  # (T, 4) - [qx, qy, qz, qw]

        T = quaternions.shape[0]
        rotation_targets = np.zeros((T, 3), dtype=np.int64)

        for t in range(T):
            rotation_targets[t] = quaternion_to_discrete_euler(quaternions[t], rotation_resolution)

        # 转换为torch tensor
        data['rotation_targets'] = torch.from_numpy(rotation_targets).long()  # (T, 3)

    # 6. 处理gripper targets：将bool转换为类别索引
    # 处理start_gripper（从start_gripper_state中提取）
    if start_gripper_state is not None:
        # start_gripper_state: bool - 0表示关闭，1表示打开
        if isinstance(start_gripper_state, bool):
            start_gripper = torch.tensor(int(start_gripper_state), dtype=torch.long)
        else:
            start_gripper = torch.tensor(start_gripper_state, dtype=torch.long)
        data['start_gripper'] = start_gripper  # scalar tensor

    # 处理gripper targets（从future_gripper_states中提取）
    if future_gripper_states is not None:
        # future_gripper_states: (T,) - bool或者0/1
        # 转换为long tensor，0表示关闭，1表示打开
        if torch.is_tensor(future_gripper_states):
            gripper_targets = future_gripper_states.long()
        else:
            gripper_targets = torch.tensor(future_gripper_states, dtype=torch.long)
        data['gripper_targets'] = gripper_targets  # (T,)

    # 7. 计算相对于第一帧的变化量（新增字段）
    # 辅助函数：将四元数转换为欧拉角（连续值）
    def quaternion_to_euler_continuous(quat):
        """将单个四元数转换为欧拉角（度）"""
        from scipy.spatial.transform import Rotation

        # 归一化四元数
        quat_normalized = quat / np.linalg.norm(quat)

        # 确保w为正数
        if quat_normalized[3] < 0:
            quat_normalized = -quat_normalized

        # 使用scipy的Rotation转换（scipy使用[x, y, z, w]顺序）
        r = Rotation.from_quat(quat_normalized)
        euler = r.as_euler("xyz", degrees=True)  # (3,) - [roll, pitch, yaw]

        # 应用gimble fix
        if 89 < euler[1] < 91:
            euler[1] = 90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)
        elif -91 < euler[1] < -89:
            euler[1] = -90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)

        return euler  # (3,) - [roll, pitch, yaw] in degrees, range [-180, 180]

    def normalize_angle(angle):
        """将角度归一化到[-180, 180]范围"""
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360
        return angle

    def discretize_delta_rotation(delta_euler, rotation_resolution):
        """将旋转变化量离散化为bins"""
        # 将范围从[-180, 180]转换为[0, 360]
        delta_shifted = delta_euler + 180
        # 离散化
        disc = np.around(delta_shifted / rotation_resolution).astype(np.int64)
        # 处理边界情况：360度 = 0度
        num_bins = int(360 / rotation_resolution)
        disc[disc == num_bins] = 0
        return disc

    # 计算rotation_delta_targets（相对于第一帧的旋转变化量）
    if start_pose is not None and future_poses is not None:
        start_quat = start_pose[3:7].cpu().numpy()
        start_euler = quaternion_to_euler_continuous(start_quat)

        quaternions = future_poses[:, 3:7].cpu().numpy()
        T = quaternions.shape[0]
        rotation_delta_targets = np.zeros((T, 3), dtype=np.int64)

        for t in range(T):
            future_euler = quaternion_to_euler_continuous(quaternions[t])
            # 计算相对于第一帧的变化量
            delta_euler = np.array([
                normalize_angle(future_euler[0] - start_euler[0]),
                normalize_angle(future_euler[1] - start_euler[1]),
                normalize_angle(future_euler[2] - start_euler[2])
            ])
            # 离散化变化量
            rotation_delta_targets[t] = discretize_delta_rotation(delta_euler, rotation_resolution)

        data['rotation_delta_targets'] = torch.from_numpy(rotation_delta_targets).long()  # (T, 3)

    # 计算gripper_change_targets（相对于第一帧的夹爪状态变化）
    if future_gripper_states is not None and start_gripper_state is not None:
        if torch.is_tensor(future_gripper_states):
            future_grippers = future_gripper_states.long()
        else:
            future_grippers = torch.tensor(future_gripper_states, dtype=torch.long)

        start_gripper_val = int(start_gripper_state) if isinstance(start_gripper_state, bool) else int(start_gripper_state)
        # 0=与第一帧相同，1=与第一帧不同
        gripper_change_targets = (future_grippers != start_gripper_val).long()
        data['gripper_change_targets'] = gripper_change_targets  # (T,)

    return data


def prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_HISTORY(
                                history_rgb_images: torch.Tensor,
                                history_heatmaps: torch.Tensor,
                                rgb_sequence: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet',
                                start_pose: torch.Tensor = None,
                                future_poses: torch.Tensor = None,
                                history_poses: torch.Tensor = None,
                                start_gripper_state: bool = None,
                                future_gripper_states: torch.Tensor = None,
                                rotation_resolution: float = 5.0) -> dict:
    """
    为Wan2.2训练准备热力图数据，支持多帧历史作为条件输入

    Args:
        history_rgb_images: 历史RGB图像序列 (num_history, num_views, 3, H, W)
        history_heatmaps: 历史热力图序列 (num_history, num_views, H, W)
        rgb_sequence: 未来RGB图像序列 (T, num_views, 3, H, W)
        heatmap_sequence: 未来热力图序列 (T, num_views, H, W)
        instruction: 文本指令
        colormap_name: colormap名称
        start_pose: 最后一个历史帧的pose (7,)
        future_poses: 未来poses (T, 7)
        history_poses: 所有历史帧的poses (num_history, 7)
        start_gripper_state: 最后一个历史帧的gripper state
        future_gripper_states: 未来gripper states (T,)
        rotation_resolution: 旋转角度离散化分辨率

    Returns:
        格式化的数据字典，包含多帧历史输入:
        - video: List[List[PIL.Image]] - 视频帧（不包含历史帧）
        - input_images: List[List[PIL.Image]] - 多帧历史热力图，每个元素是一帧的多视角列表
        - input_images_rgb: List[List[PIL.Image]] - 多帧历史RGB图像
        - input_video_rgb: List[List[PIL.Image]] - 未来RGB帧
    """
    num_history = history_rgb_images.shape[0]
    num_views = history_rgb_images.shape[1]
    num_future_frames = rgb_sequence.shape[0]

    # 1. 处理多帧历史RGB图像
    history_rgb_pil_list = []  # List[List[PIL.Image]] - num_history x num_views
    for h in range(num_history):
        views_at_h = []
        for v in range(num_views):
            rgb_view = history_rgb_images[h, v]  # (3, H, W)
            pil_img = rgb_tensor_to_pil_image(rgb_view)
            views_at_h.append(pil_img)
        history_rgb_pil_list.append(views_at_h)

    # 2. 处理多帧历史热力图
    history_heatmap_pil_list = []  # List[List[PIL.Image]] - num_history x num_views
    for h in range(num_history):
        views_at_h = []
        for v in range(num_views):
            heatmap_view = history_heatmaps[h, v]  # (H, W)
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(view_hm_colored)
            views_at_h.append(pil_img)
        history_heatmap_pil_list.append(views_at_h)

    # 3. 处理未来帧RGB图像序列
    video_frames_rgb = []
    for t in range(num_future_frames):
        views_at_t = []
        for v in range(num_views):
            rgb_view = rgb_sequence[t, v]  # (3, H, W)
            pil_img = rgb_tensor_to_pil_image(rgb_view)
            views_at_t.append(pil_img)
        video_frames_rgb.append(views_at_t)

    # 4. 处理未来帧热力图序列
    video_frames = []
    for t in range(heatmap_sequence.shape[0]):
        views_at_t = []
        for v in range(num_views):
            heatmap_view = heatmap_sequence[t, v]  # (H, W)
            heatmap_np = heatmap_view.cpu().numpy()

            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(view_hm_colored)
            views_at_t.append(pil_img)
        video_frames.append(views_at_t)

    # 5. 构建完整的视频序列（历史帧 + 未来帧）用于训练
    # video 包含所有帧：历史帧 + 未来帧
    full_video_frames = history_heatmap_pil_list + video_frames
    full_video_frames_rgb = history_rgb_pil_list + video_frames_rgb

    # 6. 构建数据字典
    data = {
        'prompt': instruction,
        # 完整视频序列（历史帧 + 未来帧）
        'video': full_video_frames,  # List[List[PIL.Image]] - (num_history + T) x num_views
        'input_video_rgb': full_video_frames_rgb,  # List[List[PIL.Image]]

        # 多帧历史条件（新增）
        'input_images': history_heatmap_pil_list,  # List[List[PIL.Image]] - num_history x num_views
        'input_images_rgb': history_rgb_pil_list,  # List[List[PIL.Image]] - num_history x num_views

        # 兼容单帧接口（使用最后一个历史帧）
        'input_image': deepcopy(history_heatmap_pil_list[-1]),  # List[PIL.Image] - num_views
        'input_image_rgb': deepcopy(history_rgb_pil_list[-1]),  # List[PIL.Image] - num_views

        # 历史帧数量标记
        'num_history_frames': num_history,
    }

    # 7. 添加pose和gripper数据
    if start_pose is not None:
        data['start_pose'] = start_pose
    if future_poses is not None:
        data['future_poses'] = future_poses
    if history_poses is not None:
        data['history_poses'] = history_poses
    if start_gripper_state is not None:
        data['start_gripper_state'] = start_gripper_state
    if future_gripper_states is not None:
        data['future_gripper_states'] = future_gripper_states

    return data


def prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY(
                                history_rgb_images: torch.Tensor,
                                history_heatmaps: torch.Tensor,
                                rgb_sequence: torch.Tensor,
                                heatmap_sequence: torch.Tensor,
                                instruction: str,
                                colormap_name: str = 'jet',
                                start_pose: torch.Tensor = None,
                                future_poses: torch.Tensor = None,
                                history_poses: torch.Tensor = None,
                                start_gripper_state: bool = None,
                                future_gripper_states: torch.Tensor = None,
                                rotation_resolution: float = 5.0) -> dict:
    """
    为Wan2.2训练准备热力图数据，支持多帧历史作为条件输入 + rotation + gripper预测

    结合了 prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_HISTORY 的多帧历史支持
    和 prepare_heatmap_data_for_wan_5B_TI2V_RGB_HEATMAP_MULTIVIEW_ROT_GRIP 的rot_grip数据处理

    Args:
        history_rgb_images: 历史RGB图像序列 (num_history, num_views, 3, H, W)
        history_heatmaps: 历史热力图序列 (num_history, num_views, H, W)
        rgb_sequence: 未来RGB图像序列 (T, num_views, 3, H, W)
        heatmap_sequence: 未来热力图序列 (T, num_views, H, W)
        instruction: 文本指令
        colormap_name: colormap名称
        start_pose: 最后一个历史帧的pose (7,)
        future_poses: 未来poses (T, 7)
        history_poses: 所有历史帧的poses (num_history, 7)
        start_gripper_state: 最后一个历史帧的gripper state
        future_gripper_states: 未来gripper states (T,)
        rotation_resolution: 旋转角度离散化分辨率

    Returns:
        格式化的数据字典，包含多帧历史输入 + rotation/gripper targets:
        - video: List[List[PIL.Image]] - 视频帧（历史帧 + 未来帧）
        - input_images: List[List[PIL.Image]] - 多帧历史热力图
        - input_images_rgb: List[List[PIL.Image]] - 多帧历史RGB图像
        - input_video_rgb: List[List[PIL.Image]] - 完整RGB帧（历史 + 未来）
        - rotation_targets: torch.Tensor (T, 3) - 未来帧的离散化rotation索引
        - gripper_targets: torch.Tensor (T,) - 未来帧的离散化gripper索引
    """
    num_history = history_rgb_images.shape[0]
    num_views = history_rgb_images.shape[1]
    num_future_frames = rgb_sequence.shape[0]

    # 1. 处理多帧历史RGB图像
    history_rgb_pil_list = []  # List[List[PIL.Image]] - num_history x num_views
    for h in range(num_history):
        views_at_h = []
        for v in range(num_views):
            rgb_view = history_rgb_images[h, v]  # (3, H, W)
            pil_img = rgb_tensor_to_pil_image(rgb_view)
            views_at_h.append(pil_img)
        history_rgb_pil_list.append(views_at_h)

    # 2. 处理多帧历史热力图
    history_heatmap_pil_list = []  # List[List[PIL.Image]] - num_history x num_views
    for h in range(num_history):
        views_at_h = []
        for v in range(num_views):
            heatmap_view = history_heatmaps[h, v]  # (H, W)
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(view_hm_colored)
            views_at_h.append(pil_img)
        history_heatmap_pil_list.append(views_at_h)

    # 3. 处理未来帧RGB图像序列
    video_frames_rgb = []
    for t in range(num_future_frames):
        views_at_t = []
        for v in range(num_views):
            rgb_view = rgb_sequence[t, v]  # (3, H, W)
            pil_img = rgb_tensor_to_pil_image(rgb_view)
            views_at_t.append(pil_img)
        video_frames_rgb.append(views_at_t)

    # 4. 处理未来帧热力图序列
    video_frames = []
    for t in range(heatmap_sequence.shape[0]):
        views_at_t = []
        for v in range(num_views):
            heatmap_view = heatmap_sequence[t, v]  # (H, W)
            heatmap_np = heatmap_view.cpu().numpy()

            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(view_hm_colored)
            views_at_t.append(pil_img)
        video_frames.append(views_at_t)

    # 5. 构建完整的视频序列（历史帧 + 未来帧）
    full_video_frames = history_heatmap_pil_list + video_frames
    full_video_frames_rgb = history_rgb_pil_list + video_frames_rgb

    # 6. 构建数据字典（与MULTIVIEW_HISTORY一致）
    data = {
        'prompt': instruction,
        # 完整视频序列（历史帧 + 未来帧）
        'video': full_video_frames,  # List[List[PIL.Image]] - (num_history + T) x num_views
        'input_video_rgb': full_video_frames_rgb,  # List[List[PIL.Image]]

        # 多帧历史条件
        'input_images': history_heatmap_pil_list,  # List[List[PIL.Image]] - num_history x num_views
        'input_images_rgb': history_rgb_pil_list,  # List[List[PIL.Image]] - num_history x num_views

        # 兼容单帧接口（使用最后一个历史帧）
        'input_image': deepcopy(history_heatmap_pil_list[-1]),  # List[PIL.Image] - num_views
        'input_image_rgb': deepcopy(history_rgb_pil_list[-1]),  # List[PIL.Image] - num_views

        # 历史帧数量标记
        'num_history_frames': num_history,
    }

    # 7. 添加原始的pose和gripper数据
    if start_pose is not None:
        data['start_pose'] = start_pose  # (7,)
    if future_poses is not None:
        data['future_poses'] = future_poses  # (T, 7)
    if history_poses is not None:
        data['history_poses'] = history_poses  # (num_history, 7)
    if start_gripper_state is not None:
        data['start_gripper_state'] = start_gripper_state  # bool
    if future_gripper_states is not None:
        data['future_gripper_states'] = future_gripper_states  # (T,)

    # 8. 处理rotation targets：将四元数转换为离散化的欧拉角索引
    def quaternion_to_discrete_euler(quat, rotation_resolution):
        """将单个四元数转换为离散化的欧拉角索引"""
        from scipy.spatial.transform import Rotation

        # 归一化四元数
        quat_normalized = quat / np.linalg.norm(quat)

        # 确保w为正数
        if quat_normalized[3] < 0:
            quat_normalized = -quat_normalized

        # 使用scipy的Rotation转换（scipy使用[x, y, z, w]顺序）
        r = Rotation.from_quat(quat_normalized)
        euler = r.as_euler("xyz", degrees=True)  # (3,) - [roll, pitch, yaw]

        # 应用gimble fix
        if 89 < euler[1] < 91:
            euler[1] = 90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)
        elif -91 < euler[1] < -89:
            euler[1] = -90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)

        # 将范围从[-180, 180]转换为[0, 360]
        euler += 180

        # 离散化
        disc = np.around(euler / rotation_resolution).astype(np.int64)
        # 处理边界情况：360度 = 0度
        num_bins = int(360 / rotation_resolution)
        disc[disc == num_bins] = 0

        return disc  # (3,) - [roll_bin, pitch_bin, yaw_bin]

    # 处理start_rotation（从start_pose中提取）
    if start_pose is not None:
        start_quat = start_pose[3:7].cpu().numpy()  # (4,) - [qx, qy, qz, qw]
        start_rotation = quaternion_to_discrete_euler(start_quat, rotation_resolution)
        data['start_rotation'] = torch.from_numpy(start_rotation).long()  # (3,)

    # 处理rotation targets（从future_poses中提取）
    if future_poses is not None:
        quaternions = future_poses[:, 3:7].cpu().numpy()  # (T, 4) - [qx, qy, qz, qw]

        T = quaternions.shape[0]
        rotation_targets = np.zeros((T, 3), dtype=np.int64)

        for t in range(T):
            rotation_targets[t] = quaternion_to_discrete_euler(quaternions[t], rotation_resolution)

        data['rotation_targets'] = torch.from_numpy(rotation_targets).long()  # (T, 3)

    # 9. 处理gripper targets：将bool转换为类别索引
    if start_gripper_state is not None:
        if isinstance(start_gripper_state, bool):
            start_gripper = torch.tensor(int(start_gripper_state), dtype=torch.long)
        else:
            start_gripper = torch.tensor(start_gripper_state, dtype=torch.long)
        data['start_gripper'] = start_gripper  # scalar tensor

    if future_gripper_states is not None:
        if torch.is_tensor(future_gripper_states):
            gripper_targets = future_gripper_states.long()
        else:
            gripper_targets = torch.tensor(future_gripper_states, dtype=torch.long)
        data['gripper_targets'] = gripper_targets  # (T,)

    # 10. 计算相对于第一帧的变化量
    def quaternion_to_euler_continuous(quat):
        """将单个四元数转换为欧拉角（度）"""
        from scipy.spatial.transform import Rotation

        # 归一化四元数
        quat_normalized = quat / np.linalg.norm(quat)

        # 确保w为正数
        if quat_normalized[3] < 0:
            quat_normalized = -quat_normalized

        # 使用scipy的Rotation转换
        r = Rotation.from_quat(quat_normalized)
        euler = r.as_euler("xyz", degrees=True)

        # 应用gimble fix
        if 89 < euler[1] < 91:
            euler[1] = 90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)
        elif -91 < euler[1] < -89:
            euler[1] = -90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)

        return euler  # (3,) - [roll, pitch, yaw] in degrees, range [-180, 180]

    def normalize_angle(angle):
        """将角度归一化到[-180, 180]范围"""
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360
        return angle

    def discretize_delta_rotation(delta_euler, rotation_resolution):
        """将旋转变化量离散化为bins"""
        delta_shifted = delta_euler + 180
        disc = np.around(delta_shifted / rotation_resolution).astype(np.int64)
        num_bins = int(360 / rotation_resolution)
        disc[disc == num_bins] = 0
        return disc

    # 计算rotation_delta_targets（相对于最后一个历史帧的旋转变化量）
    if start_pose is not None and future_poses is not None:
        start_quat = start_pose[3:7].cpu().numpy()
        start_euler = quaternion_to_euler_continuous(start_quat)

        quaternions = future_poses[:, 3:7].cpu().numpy()
        T = quaternions.shape[0]
        rotation_delta_targets = np.zeros((T, 3), dtype=np.int64)

        for t in range(T):
            future_euler = quaternion_to_euler_continuous(quaternions[t])
            delta_euler = np.array([
                normalize_angle(future_euler[0] - start_euler[0]),
                normalize_angle(future_euler[1] - start_euler[1]),
                normalize_angle(future_euler[2] - start_euler[2])
            ])
            rotation_delta_targets[t] = discretize_delta_rotation(delta_euler, rotation_resolution)

        data['rotation_delta_targets'] = torch.from_numpy(rotation_delta_targets).long()  # (T, 3)

    # 计算gripper_change_targets（相对于最后一个历史帧的夹爪状态变化）
    if future_gripper_states is not None and start_gripper_state is not None:
        if torch.is_tensor(future_gripper_states):
            future_grippers = future_gripper_states.long()
        else:
            future_grippers = torch.tensor(future_gripper_states, dtype=torch.long)

        start_gripper_val = int(start_gripper_state) if isinstance(start_gripper_state, bool) else int(start_gripper_state)
        gripper_change_targets = (future_grippers != start_gripper_val).long()
        data['gripper_change_targets'] = gripper_change_targets  # (T,)

    return data


# 测试函数
if __name__ == "__main__":
    print("Testing heatmap conversion utilities...")

    # 创建测试数据
    test_heatmap = np.random.rand(64, 64)
    test_sequence = np.random.rand(5, 64, 64)

    # 测试单帧转换
    print("Testing single heatmap conversion...")
    colormap_img = convert_heatmap_to_colormap(test_heatmap)
    restored_heatmap = extract_heatmap_from_colormap(colormap_img)
    print(f"Original shape: {test_heatmap.shape}, Restored shape: {restored_heatmap.shape}")

    # 测试序列转换
    print("Testing sequence conversion...")
    pil_images = heatmap_sequence_to_pil_images(test_sequence)
    print(f"Generated {len(pil_images)} PIL images")

    # 测试完整数据准备
    print("Testing complete data preparation...")
    rgb_tensor = torch.rand(3, 64, 64)
    heatmap_tensor = torch.rand(5, 64, 64)
    data = prepare_heatmap_data_for_wan(rgb_tensor, heatmap_tensor, "test instruction")
    print(f"Data keys: {list(data.keys())}")
    print(f"Video frames: {len(data['video'])}")

    print("All tests completed successfully!")