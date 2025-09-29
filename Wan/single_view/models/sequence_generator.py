"""
序列生成器
为固定长度heatmap序列提供不同的生成策略和质量控制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import time

try:
    from .wan_heatmap_model import WanHeatmapModel
    from ..utils.heatmap_utils import (
        normalize_heatmap,
        calculate_peak_accuracy,
        calculate_sequence_consistency,
        find_global_peak
    )
    from ..utils.visualization_utils import create_heatmap_overlay
    from ..configs.model_config import ModelConfig
except ImportError:
    # 绝对导入用于直接运行时
    from models.wan_heatmap_model import WanHeatmapModel
    from utils.heatmap_utils import (
        normalize_heatmap,
        calculate_peak_accuracy,
        calculate_sequence_consistency,
        find_global_peak
    )
    from utils.visualization_utils import create_heatmap_overlay
    from configs.model_config import ModelConfig


class SequenceGenerator:
    """
    固定长度Heatmap序列生成器

    负责：
    1. 提供不同的生成策略（快速/高质量）
    2. 优化推理速度
    3. 后处理和质量控制
    4. 性能监控
    """

    def __init__(self,
                 model: WanHeatmapModel,
                 config: Optional[ModelConfig] = None,
                 generation_strategy: str = "standard"):
        """
        初始化序列生成器

        Args:
            model: Wan heatmap模型
            config: 模型配置
            generation_strategy: 生成策略 ("standard", "fast", "high_quality")
        """
        self.model = model
        self.config = config or model.config
        self.generation_strategy = generation_strategy
        self.device = model.device

        # 固定序列长度（从配置中获取）
        self.sequence_length = self.config.sequence_length

        # 质量控制参数
        self.quality_threshold = 0.8
        self.consistency_threshold = 0.7

        # 性能统计
        self.generation_stats = {
            'total_generations': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'quality_scores': []
        }

    def generate(self,
                rgb_images: torch.Tensor,
                instruction_embeddings: Optional[torch.Tensor] = None,
                guidance_scale: float = 7.5,
                num_inference_steps: Optional[int] = None,
                temperature: float = 1.0,
                return_intermediate: bool = False) -> Dict[str, Any]:
        """
        生成固定长度的heatmap序列

        Args:
            rgb_images: 输入RGB图像 (B, 3, H, W)
            instruction_embeddings: 指令嵌入 (B, D)
            guidance_scale: 引导尺度
            num_inference_steps: 推理步数
            temperature: 采样温度
            return_intermediate: 是否返回中间结果

        Returns:
            生成结果字典
        """
        start_time = time.time()

        # 设置推理步数
        if num_inference_steps is None:
            if self.generation_strategy == "fast":
                num_inference_steps = 20
            elif self.generation_strategy == "high_quality":
                num_inference_steps = 100
            else:
                num_inference_steps = self.config.num_inference_steps

        # 执行对应的生成策略
        if self.generation_strategy == "fast":
            result = self._generate_fast(
                rgb_images, instruction_embeddings, guidance_scale, num_inference_steps
            )
        elif self.generation_strategy == "high_quality":
            result = self._generate_high_quality(
                rgb_images, instruction_embeddings, guidance_scale, num_inference_steps
            )
        else:  # standard
            result = self._generate_standard(
                rgb_images, instruction_embeddings, guidance_scale, num_inference_steps
            )

        # 后处理
        result = self._post_process(result, rgb_images)

        # 计算质量指标
        quality_metrics = self._compute_quality_metrics(result['predictions'])
        result['quality_metrics'] = quality_metrics

        # 更新统计信息
        generation_time = time.time() - start_time
        self._update_stats(generation_time, quality_metrics)

        result['generation_time'] = generation_time
        result['generation_strategy'] = self.generation_strategy
        result['sequence_length'] = self.sequence_length

        return result

    def _generate_standard(self,
                          rgb_images: torch.Tensor,
                          instruction_embeddings: Optional[torch.Tensor],
                          guidance_scale: float,
                          num_inference_steps: int) -> Dict[str, Any]:
        """标准生成策略 - 使用配置的默认参数"""
        self.model.eval()

        with torch.no_grad():
            result = self.model.generate_heatmap_sequence(
                rgb_images,
                instruction_embeddings,
                num_inference_steps
            )

        return result

    def _generate_fast(self,
                      rgb_images: torch.Tensor,
                      instruction_embeddings: Optional[torch.Tensor],
                      guidance_scale: float,
                      num_inference_steps: int) -> Dict[str, Any]:
        """快速生成策略 - 使用较低分辨率和较少推理步数"""
        original_size = self.config.output_image_size
        fast_size = (original_size[0] // 2, original_size[1] // 2)

        # 临时调整配置到较低分辨率
        self.config.output_image_size = fast_size

        # 下采样输入图像
        h, w = fast_size
        rgb_resized = torch.nn.functional.interpolate(
            rgb_images, size=(h, w), mode='bilinear', align_corners=False
        )

        # 在低分辨率下生成
        result = self._generate_standard(
            rgb_resized, instruction_embeddings, guidance_scale, num_inference_steps
        )

        # 上采样输出到原始分辨率
        predictions = result['predictions']  # (B, T, h, w)
        B, T = predictions.shape[:2]

        predictions_upsampled = torch.nn.functional.interpolate(
            predictions.view(B * T, 1, h, w),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).view(B, T, *original_size)

        result['predictions'] = predictions_upsampled

        # 恢复原始配置
        self.config.output_image_size = original_size

        return result

    def _generate_high_quality(self,
                              rgb_images: torch.Tensor,
                              instruction_embeddings: Optional[torch.Tensor],
                              guidance_scale: float,
                              num_inference_steps: int) -> Dict[str, Any]:
        """高质量生成策略 - 多次采样并选择最佳结果"""
        num_samples = 3  # 生成多个样本并选择最佳

        best_result = None
        best_quality = -1

        for i in range(num_samples):
            result = self._generate_standard(
                rgb_images, instruction_embeddings, guidance_scale, num_inference_steps
            )

            # 计算质量分数
            quality_score = self._compute_overall_quality(result['predictions'])

            if quality_score > best_quality:
                best_quality = quality_score
                best_result = result

        return best_result

    def _post_process(self, result: Dict[str, Any], rgb_images: torch.Tensor) -> Dict[str, Any]:
        """
        后处理生成结果

        Args:
            result: 原始生成结果
            rgb_images: 输入RGB图像

        Returns:
            后处理后的结果
        """
        predictions = result['predictions']

        # 标准化到[0, 1]范围
        predictions = normalize_heatmap(predictions, method='minmax')

        # 应用平滑滤波（仅在高质量模式下）
        if self.generation_strategy == "high_quality":
            predictions = self._apply_temporal_smoothing(predictions)

        # 应用峰值增强（可选）
        if hasattr(self.config, 'enhance_peaks') and getattr(self.config, 'enhance_peaks', False):
            predictions = self._enhance_peaks(predictions)

        result['predictions'] = predictions

        # 生成可视化叠加图像
        if rgb_images is not None:
            overlays = self._create_overlay_sequence(predictions, rgb_images)
            result['overlay_sequence'] = overlays

        return result

    def _apply_temporal_smoothing(self, sequence: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        应用时间维度的平滑滤波

        Args:
            sequence: 输入序列 (B, T, H, W)
            kernel_size: 滤波器大小

        Returns:
            平滑后的序列
        """
        B, T, H, W = sequence.shape

        if T < kernel_size:
            return sequence

        # 创建1D卷积核
        weight = torch.ones(1, 1, kernel_size, device=sequence.device) / kernel_size

        # 对每个像素位置应用时间平滑
        sequence_reshaped = sequence.view(B, T, -1).permute(0, 2, 1)  # (B, H*W, T)

        # 填充并应用卷积
        padded = torch.nn.functional.pad(sequence_reshaped, (kernel_size//2, kernel_size//2), mode='reflect')
        smoothed = torch.nn.functional.conv1d(padded, weight)

        # 恢复形状
        smoothed = smoothed.permute(0, 2, 1).view(B, T, H, W)

        return smoothed

    def _enhance_peaks(self, sequence: torch.Tensor, enhancement_factor: float = 1.2) -> torch.Tensor:
        """
        增强峰值点

        Args:
            sequence: 输入序列 (B, T, H, W)
            enhancement_factor: 增强因子

        Returns:
            增强后的序列
        """
        enhanced = sequence.clone()

        for b in range(sequence.shape[0]):
            for t in range(sequence.shape[1]):
                heatmap = sequence[b, t]

                # 找到峰值位置
                peak_y, peak_x = find_global_peak(heatmap)

                # 创建高斯增强掩码
                y, x = torch.meshgrid(
                    torch.arange(heatmap.shape[0], device=sequence.device),
                    torch.arange(heatmap.shape[1], device=sequence.device),
                    indexing='ij'
                )

                sigma = 5.0
                gaussian_mask = torch.exp(-((x - peak_x)**2 + (y - peak_y)**2) / (2 * sigma**2))

                # 应用增强
                enhanced[b, t] = heatmap + (enhancement_factor - 1) * heatmap * gaussian_mask

        return torch.clamp(enhanced, 0, 1)

    def _create_overlay_sequence(self, predictions: torch.Tensor,
                               rgb_images: torch.Tensor) -> torch.Tensor:
        """
        创建RGB-heatmap叠加序列

        Args:
            predictions: 预测的heatmap序列 (B, T, H, W)
            rgb_images: RGB图像 (B, 3, H, W)

        Returns:
            叠加图像序列 (B, T, 3, H, W)
        """
        B, T, H, W = predictions.shape
        overlays = torch.zeros(B, T, 3, H, W, device=predictions.device)

        for b in range(B):
            rgb_np = rgb_images[b].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            rgb_np = (rgb_np + 1) / 2  # 假设输入在[-1, 1]范围

            for t in range(T):
                heatmap_np = predictions[b, t].cpu().numpy()
                overlay_np = create_heatmap_overlay(rgb_np, heatmap_np, alpha=0.6)
                overlays[b, t] = torch.from_numpy(overlay_np).permute(2, 0, 1)

        return overlays

    def _compute_quality_metrics(self, predictions: torch.Tensor) -> Dict[str, float]:
        """
        计算生成质量指标

        Args:
            predictions: 预测序列 (B, T, H, W)

        Returns:
            质量指标字典
        """
        metrics = {}

        # 序列一致性
        consistency_scores = []
        for b in range(predictions.shape[0]):
            consistency = calculate_sequence_consistency(predictions[b])
            consistency_scores.append(consistency)

        metrics['sequence_consistency'] = np.mean(consistency_scores)

        # 峰值强度
        peak_intensities = []
        for b in range(predictions.shape[0]):
            for t in range(predictions.shape[1]):
                peak_intensity = torch.max(predictions[b, t]).item()
                peak_intensities.append(peak_intensity)

        metrics['average_peak_intensity'] = np.mean(peak_intensities)

        # 空间分布熵（用于衡量heatmap的集中度）
        entropies = []
        for b in range(predictions.shape[0]):
            for t in range(predictions.shape[1]):
                heatmap = predictions[b, t]
                normalized = heatmap / (torch.sum(heatmap) + 1e-8)
                entropy = -torch.sum(normalized * torch.log(normalized + 1e-8)).item()
                entropies.append(entropy)

        metrics['average_entropy'] = np.mean(entropies)

        return metrics

    def _compute_overall_quality(self, predictions: torch.Tensor) -> float:
        """
        计算整体质量分数

        Args:
            predictions: 预测序列

        Returns:
            质量分数 [0, 1]
        """
        metrics = self._compute_quality_metrics(predictions)

        # 综合各项指标
        consistency_score = metrics['sequence_consistency']
        peak_score = min(metrics['average_peak_intensity'], 1.0)
        entropy_score = 1.0 - min(metrics['average_entropy'] / 10.0, 1.0)  # 归一化熵

        overall_score = 0.4 * consistency_score + 0.4 * peak_score + 0.2 * entropy_score
        return overall_score

    def _update_stats(self, generation_time: float, quality_metrics: Dict[str, float]):
        """更新生成统计信息"""
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_time'] += generation_time
        self.generation_stats['average_time'] = (
            self.generation_stats['total_time'] / self.generation_stats['total_generations']
        )

        overall_quality = self._compute_overall_quality_from_metrics(quality_metrics)
        self.generation_stats['quality_scores'].append(overall_quality)

    def _compute_overall_quality_from_metrics(self, metrics: Dict[str, float]) -> float:
        """从质量指标计算整体分数"""
        consistency_score = metrics['sequence_consistency']
        peak_score = min(metrics['average_peak_intensity'], 1.0)
        entropy_score = 1.0 - min(metrics['average_entropy'] / 10.0, 1.0)

        return 0.4 * consistency_score + 0.4 * peak_score + 0.2 * entropy_score

    def get_stats(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        stats = self.generation_stats.copy()
        if stats['quality_scores']:
            stats['average_quality'] = np.mean(stats['quality_scores'])
            stats['quality_std'] = np.std(stats['quality_scores'])
        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.generation_stats = {
            'total_generations': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'quality_scores': []
        }

    def set_generation_strategy(self, strategy: str):
        """设置生成策略"""
        valid_strategies = ["standard", "fast", "high_quality"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
        self.generation_strategy = strategy


def test_sequence_generator():
    """
    测试序列生成器
    """
    print("Testing sequence generator...")

    try:
        # 创建配置和模型
        from ..configs.model_config import get_debug_config
        config = get_debug_config().model
        config.sequence_length = 5
        config.input_image_size = (64, 64)
        config.output_image_size = (64, 64)

        from .wan_heatmap_model import WanHeatmapModel
        model = WanHeatmapModel(config)

        # 创建序列生成器
        generator = SequenceGenerator(model, config, generation_strategy="standard")

        # 创建测试数据
        batch_size = 2
        rgb_images = torch.randn(batch_size, 3, 64, 64)

        print(f"Target sequence length: {generator.sequence_length}")

        # 测试不同生成策略
        strategies = ["standard", "fast", "high_quality"]

        for strategy in strategies:
            print(f"\nTesting {strategy} strategy...")
            generator.set_generation_strategy(strategy)

            result = generator.generate(
                rgb_images,
                num_inference_steps=10
            )

            print(f"Generated sequence shape: {result['predictions'].shape}")
            print(f"Expected shape: ({batch_size}, {generator.sequence_length}, 64, 64)")
            print(f"Generation time: {result['generation_time']:.3f}s")
            print(f"Quality metrics: {result['quality_metrics']}")

        # 测试统计信息
        stats = generator.get_stats()
        print(f"\nGeneration statistics: {stats}")

        print("Sequence generator test completed successfully!")
        return True

    except Exception as e:
        print(f"Sequence generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sequence_generator()