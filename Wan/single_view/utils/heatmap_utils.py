"""
Heatmap处理工具函数
包含heatmap标准化、peak检测、精度计算等功能
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.distance import euclidean
from typing import Tuple, List, Union


def normalize_heatmap(heatmap: Union[np.ndarray, torch.Tensor],
                     method: str = 'minmax') -> Union[np.ndarray, torch.Tensor]:
    """
    标准化heatmap到[0, 1]范围

    Args:
        heatmap: 输入heatmap (H, W) 或 (T, H, W)
        method: 标准化方法 ('minmax', 'softmax', 'zscore')

    Returns:
        标准化后的heatmap
    """
    is_torch = torch.is_tensor(heatmap)

    if method == 'minmax':
        if is_torch:
            flat_hm = heatmap.view(-1)
            min_val = torch.min(flat_hm)
            max_val = torch.max(flat_hm)
            normalized = (heatmap - min_val) / (max_val - min_val + 1e-8)
        else:
            min_val = np.min(heatmap)
            max_val = np.max(heatmap)
            normalized = (heatmap - min_val) / (max_val - min_val + 1e-8)

    elif method == 'softmax':
        if is_torch:
            if len(heatmap.shape) == 2:
                flat_hm = heatmap.view(-1)
                softmax_hm = F.softmax(flat_hm, dim=0)
                normalized = softmax_hm.view(heatmap.shape)
            else:  # 3D case
                flat_hm = heatmap.view(heatmap.shape[0], -1)
                softmax_hm = F.softmax(flat_hm, dim=1)
                normalized = softmax_hm.view(heatmap.shape)
        else:
            heatmap = torch.from_numpy(heatmap)
            normalized = normalize_heatmap(heatmap, method).numpy()

    elif method == 'zscore':
        if is_torch:
            mean_val = torch.mean(heatmap)
            std_val = torch.std(heatmap)
            normalized = (heatmap - mean_val) / (std_val + 1e-8)
            # Clamp to reasonable range and shift to [0, 1]
            normalized = torch.clamp(normalized, -3, 3)
            normalized = (normalized + 3) / 6
        else:
            mean_val = np.mean(heatmap)
            std_val = np.std(heatmap)
            normalized = (heatmap - mean_val) / (std_val + 1e-8)
            normalized = np.clip(normalized, -3, 3)
            normalized = (normalized + 3) / 6

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def denormalize_heatmap(normalized_heatmap: Union[np.ndarray, torch.Tensor],
                       original_min: float,
                       original_max: float) -> Union[np.ndarray, torch.Tensor]:
    """
    反标准化heatmap到原始值范围

    Args:
        normalized_heatmap: 标准化的heatmap [0, 1]
        original_min: 原始最小值
        original_max: 原始最大值

    Returns:
        反标准化的heatmap
    """
    return normalized_heatmap * (original_max - original_min) + original_min


def find_peak_coordinates(heatmap: Union[np.ndarray, torch.Tensor],
                         threshold: float = 0.1,
                         min_distance: int = 5) -> List[Tuple[int, int]]:
    """
    在heatmap中找到峰值点的坐标

    Args:
        heatmap: 输入heatmap (H, W)
        threshold: 峰值阈值
        min_distance: 峰值间最小距离

    Returns:
        峰值点坐标列表 [(y, x), ...]
    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()

    # 使用局部最大值检测
    try:
        from skimage.feature import peak_local_maxima
        # 找到局部最大值
        local_maxima = peak_local_maxima(
            heatmap,
            min_distance=min_distance,
            threshold_abs=threshold
        )
    except ImportError:
        print("Warning: scikit-image not available, using simple peak detection")
        # 简单的峰值检测替代方案
        peak_y, peak_x = find_global_peak(heatmap)
        local_maxima = (np.array([peak_y]), np.array([peak_x]))

    # 按强度排序
    peak_values = heatmap[local_maxima]
    sorted_indices = np.argsort(peak_values)[::-1]

    peaks = [(local_maxima[0][i], local_maxima[1][i])
             for i in sorted_indices]

    return peaks


def find_global_peak(heatmap: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
    """
    找到heatmap中的全局最大值位置

    Args:
        heatmap: 输入heatmap (H, W)

    Returns:
        全局峰值坐标 (y, x)
    """
    if torch.is_tensor(heatmap):
        flat_idx = torch.argmax(heatmap.view(-1))
        y, x = divmod(flat_idx.item(), heatmap.shape[1])
    else:
        flat_idx = np.argmax(heatmap.flatten())
        y, x = divmod(flat_idx, heatmap.shape[1])

    return (y, x)


def calculate_peak_accuracy(pred_heatmap: Union[np.ndarray, torch.Tensor],
                          gt_heatmap: Union[np.ndarray, torch.Tensor],
                          distance_threshold: float = 5.0) -> float:
    """
    计算预测heatmap与ground truth的峰值精度

    Args:
        pred_heatmap: 预测的heatmap (H, W)
        gt_heatmap: ground truth heatmap (H, W)
        distance_threshold: 距离阈值，小于此距离认为预测正确

    Returns:
        精度值 (0-1)
    """
    # 找到预测和GT的峰值位置
    pred_peak = find_global_peak(pred_heatmap)
    gt_peak = find_global_peak(gt_heatmap)

    # 计算欧氏距离
    distance = euclidean(pred_peak, gt_peak)

    # 如果距离小于阈值，认为预测正确
    accuracy = 1.0 if distance <= distance_threshold else 0.0

    return accuracy


def calculate_sequence_accuracy(pred_sequence: Union[np.ndarray, torch.Tensor],
                              gt_sequence: Union[np.ndarray, torch.Tensor],
                              distance_threshold: float = 5.0) -> float:
    """
    计算heatmap序列的平均峰值精度

    Args:
        pred_sequence: 预测的heatmap序列 (T, H, W)
        gt_sequence: ground truth序列 (T, H, W)
        distance_threshold: 距离阈值

    Returns:
        平均精度值
    """
    if torch.is_tensor(pred_sequence):
        pred_sequence = pred_sequence.cpu().numpy()
    if torch.is_tensor(gt_sequence):
        gt_sequence = gt_sequence.cpu().numpy()

    sequence_length = pred_sequence.shape[0]
    accuracies = []

    for t in range(sequence_length):
        accuracy = calculate_peak_accuracy(
            pred_sequence[t],
            gt_sequence[t],
            distance_threshold
        )
        accuracies.append(accuracy)

    return np.mean(accuracies)


def calculate_mse_loss(pred_heatmap: Union[np.ndarray, torch.Tensor],
                      gt_heatmap: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算预测heatmap与ground truth的MSE损失

    Args:
        pred_heatmap: 预测的heatmap
        gt_heatmap: ground truth heatmap

    Returns:
        MSE损失值
    """
    if torch.is_tensor(pred_heatmap) and torch.is_tensor(gt_heatmap):
        mse = F.mse_loss(pred_heatmap, gt_heatmap)
        return mse.item()
    else:
        if torch.is_tensor(pred_heatmap):
            pred_heatmap = pred_heatmap.cpu().numpy()
        if torch.is_tensor(gt_heatmap):
            gt_heatmap = gt_heatmap.cpu().numpy()

        mse = np.mean((pred_heatmap - gt_heatmap) ** 2)
        return mse


def calculate_sequence_consistency(heatmap_sequence: Union[np.ndarray, torch.Tensor],
                                 method: str = 'peak_distance') -> float:
    """
    计算heatmap序列的连续性/一致性

    Args:
        heatmap_sequence: heatmap序列 (T, H, W)
        method: 计算方法 ('peak_distance', 'correlation', 'mse')

    Returns:
        一致性分数，越高表示越连续
    """
    if torch.is_tensor(heatmap_sequence):
        heatmap_sequence = heatmap_sequence.cpu().numpy()

    sequence_length = heatmap_sequence.shape[0]

    if sequence_length < 2:
        return 1.0

    if method == 'peak_distance':
        # 计算相邻帧峰值点之间的距离
        distances = []
        for t in range(1, sequence_length):
            peak_prev = find_global_peak(heatmap_sequence[t-1])
            peak_curr = find_global_peak(heatmap_sequence[t])
            distance = euclidean(peak_prev, peak_curr)
            distances.append(distance)

        # 距离越小，一致性越高
        avg_distance = np.mean(distances)
        # 归一化到[0, 1]，假设最大可能距离为图像对角线
        max_distance = np.sqrt(heatmap_sequence.shape[1]**2 + heatmap_sequence.shape[2]**2)
        consistency = 1.0 - min(avg_distance / max_distance, 1.0)

    elif method == 'correlation':
        # 计算相邻帧之间的相关系数
        correlations = []
        for t in range(1, sequence_length):
            corr = np.corrcoef(
                heatmap_sequence[t-1].flatten(),
                heatmap_sequence[t].flatten()
            )[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        consistency = np.mean(correlations) if correlations else 0.0

    elif method == 'mse':
        # 计算相邻帧之间的MSE，MSE越小一致性越高
        mses = []
        for t in range(1, sequence_length):
            mse = np.mean((heatmap_sequence[t-1] - heatmap_sequence[t]) ** 2)
            mses.append(mse)

        avg_mse = np.mean(mses)
        # 转换为一致性分数（假设最大MSE为1）
        consistency = max(0.0, 1.0 - avg_mse)

    else:
        raise ValueError(f"Unknown consistency method: {method}")

    return consistency


def generate_gaussian_heatmap(center: Tuple[int, int],
                            image_size: Tuple[int, int],
                            sigma: float = 2.0) -> np.ndarray:
    """
    在指定位置生成高斯heatmap

    Args:
        center: 中心点坐标 (y, x)
        image_size: 图像尺寸 (H, W)
        sigma: 高斯分布标准差

    Returns:
        生成的heatmap (H, W)
    """
    y_center, x_center = center
    h, w = image_size

    # 创建坐标网格
    y, x = np.ogrid[:h, :w]

    # 计算高斯分布
    heatmap = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))

    return heatmap


def blur_heatmap(heatmap: Union[np.ndarray, torch.Tensor],
                sigma: float = 1.0) -> Union[np.ndarray, torch.Tensor]:
    """
    对heatmap应用高斯模糊

    Args:
        heatmap: 输入heatmap
        sigma: 高斯核标准差

    Returns:
        模糊后的heatmap
    """
    if torch.is_tensor(heatmap):
        # 对torch tensor使用conv2d实现高斯模糊
        if len(heatmap.shape) == 2:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif len(heatmap.shape) == 3:
            heatmap = heatmap.unsqueeze(1)  # Add channel dim

        # 创建高斯核
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        gaussian_kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                gaussian_kernel[0, 0, i, j] = np.exp(
                    -((i - center)**2 + (j - center)**2) / (2 * sigma**2)
                )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.to(heatmap.device)

        # 应用卷积
        blurred = F.conv2d(heatmap, gaussian_kernel, padding=kernel_size//2)

        # 恢复原始形状
        if len(heatmap.shape) == 4:
            blurred = blurred.squeeze(1)
        if len(heatmap.shape) == 3:
            blurred = blurred.squeeze(0).squeeze(0)

        return blurred
    else:
        return ndimage.gaussian_filter(heatmap, sigma=sigma)


def evaluate_heatmap_quality(heatmap: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    评估heatmap的质量指标

    Args:
        heatmap: 输入heatmap (H, W)

    Returns:
        质量指标字典
    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()

    # 基本统计
    stats = {
        'mean': np.mean(heatmap),
        'std': np.std(heatmap),
        'min': np.min(heatmap),
        'max': np.max(heatmap),
        'range': np.max(heatmap) - np.min(heatmap)
    }

    # 峰值相关指标
    peak_coord = find_global_peak(heatmap)
    peak_value = heatmap[peak_coord]

    stats.update({
        'peak_value': peak_value,
        'peak_coord': peak_coord,
        'peak_ratio': peak_value / (np.mean(heatmap) + 1e-8)  # 峰值与均值的比率
    })

    # 分布指标
    flat_hm = heatmap.flatten()

    # 计算熵（作为分布集中度的指标）
    normalized_hm = flat_hm / (np.sum(flat_hm) + 1e-8)
    entropy = -np.sum(normalized_hm * np.log(normalized_hm + 1e-8))

    stats.update({
        'entropy': entropy,
        'sparsity': np.sum(flat_hm > 0.1 * np.max(flat_hm)) / len(flat_hm),  # 活跃区域比例
    })

    return stats


def test_heatmap_utils():
    """
    测试heatmap工具函数
    """
    print("Testing heatmap utilities...")

    # 创建测试heatmap
    h, w = 64, 64
    center = (32, 32)
    test_heatmap = generate_gaussian_heatmap(center, (h, w), sigma=5.0)

    # 测试标准化
    normalized = normalize_heatmap(test_heatmap, 'minmax')
    print(f"Normalized range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")

    # 测试峰值检测
    peak = find_global_peak(test_heatmap)
    print(f"Found peak at: {peak}, expected: {center}")

    # 测试质量评估
    quality = evaluate_heatmap_quality(test_heatmap)
    print(f"Heatmap quality: {quality}")

    # 测试精度计算
    accuracy = calculate_peak_accuracy(test_heatmap, test_heatmap, distance_threshold=2.0)
    print(f"Self-accuracy: {accuracy}")

    print("Heatmap utils test completed!")
    return True


if __name__ == "__main__":
    # 运行测试
    test_heatmap_utils()