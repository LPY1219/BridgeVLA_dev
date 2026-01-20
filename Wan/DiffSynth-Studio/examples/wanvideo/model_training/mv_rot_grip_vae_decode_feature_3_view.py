"""
Multi-View Rotation and Gripper Prediction Training Script (View Concatenation Version)

**KEY DIFFERENCE from mv_rot_grip_vae_decode_feature_3.py:**
- Channel concatenation (original): RGB and Heatmap are concatenated along channel dimension
  → (b, 3_views, 96_channels, t, h, w) where 96 = 48_rgb + 48_heatmap
- View concatenation (THIS FILE): RGB and Heatmap are concatenated along view dimension
  → (b, 6_views, 48_channels, t, h, w) where 6 = 3_rgb + 3_heatmap

This script trains a model to predict rotation (roll, pitch, yaw) and gripper state
from VAE-encoded RGB and heatmap features using VIEW concatenation.

Key features:
- Uses VAE-compressed video features (RGB + Heatmap as separate views)
- 6 views total: 3 RGB cameras + 3 Heatmap overlays
- Temporal compression: T frames -> 1 + (T-1)//4 frames
- Predicts rotation and gripper for future frames
- Uses SwanLab for logging
- Tracks accuracy metrics during training

View organization:
- View 0-2: RGB cameras (camera1_rgb, camera2_rgb, camera3_rgb)
- View 3-5: Heatmap overlays (camera1_heatmap, camera2_heatmap, camera3_heatmap)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import os
import sys
from pathlib import Path
import argparse
import numpy as np
from typing import Dict, Optional

# Add DiffSynth-Studio to path
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, diffsynth_path)

# from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip import HeatmapDatasetFactory
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam import HeatmapDatasetFactory
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam_history import HeatmapDatasetFactoryWithHistory

class MultiViewRotationGripperPredictorView(nn.Module):
    """
    预测旋转和夹爪变化量的模型 - 直接使用VAE Latents（View拼接版本）

    **View拼接模式**: RGB和Heatmap作为独立的view，而不是同一view的不同channel
    **直接使用Latents**: 不使用VAE decoder的中间特征，直接使用VAE编码后的latents

    基于帧间特征差异预测相对于第一帧的变化量

    输入：
        - rgb_latents: VAE编码后的RGB latents (b, 3, 48, t_compressed, h_latent, w_latent)
        - heatmap_latents: VAE编码后的Heatmap latents (b, 3, 48, t_compressed, h_latent, w_latent)
        - num_future_frames: 需要预测的未来帧数 (T-1，不包括初始帧)

    处理流程：
        1. 沿view维度拼接RGB和Heatmap: (b, 6, 48, t_compressed, h_latent, w_latent)
        2. View 0-2: RGB cameras (camera1_rgb, camera2_rgb, camera3_rgb)
        3. View 3-5: Heatmap overlays (camera1_heatmap, camera2_heatmap, camera3_heatmap)
        4. 内部进行时间上采样：t_compressed → t_upsampled (~num_future_frames+1)

    输出：
        - rotation_logits: (b, num_future_frames, num_rotation_bins*3) - 未来帧相对于第一帧的rotation变化量预测
        - gripper_logits: (b, num_future_frames, 2) - 未来帧的gripper状态变化预测 (0=不变, 1=改变)

    核心设计：
        - t_upsampled 包含所有帧（初始帧 + 未来帧），约等于 T
        - 计算每个未来帧与第一帧的特征差异
        - 基于特征差异预测旋转变化量和夹爪状态变化
        - 不需要初始旋转和夹爪状态作为输入

    注意：
        - latent_channels = 48 (VAE编码后的latent通道数)
        - t_compressed ≈ (T-1)//4 + 1 (VAE时间压缩比为4)
        - 模型内部需要进行时间上采样：t_compressed → t_upsampled
        - rotation预测的是变化量（delta），范围仍使用bins表示
    """

    def __init__(
        self,
        rgb_channels: int = 48,  # VAE latent channels (not decoder intermediate!)
        heatmap_channels: int = 48,  # VAE latent channels (not decoder intermediate!)
        hidden_dim: int = 512,
        num_views: int = 6,  # 6 views in view-concat mode: 3 RGB + 3 Heatmap
        num_rotation_bins: int = 72,
        dropout: float = 0.1,
        vae = None,  # VAE对象，用于解码heatmap找峰值
        local_feature_size: int = 3,  # 局部特征提取的邻域大小
        use_initial_gripper_state: bool = False,  # 是否使用初始夹爪状态作为输入
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.heatmap_channels = heatmap_channels
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_rotation_bins = num_rotation_bins
        self.vae = vae
        self.local_feature_size = local_feature_size
        self.use_initial_gripper_state = use_initial_gripper_state

        # 全局特征提取器 - 为每个视角提取全局特征
        # View拼接模式：每个view只包含一种模态（RGB或Heatmap），不在channel维度拼接
        # 输入是VAE latents，尚未经过decoder上采样
        input_channels = rgb_channels  # 48 (latent channels, 单一模态)
        self.global_feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 空间池化: (b*v, hidden_dim, t, 1, 1)
        )

        # 局部特征提取器 - 用于处理峰值附近区域
        # View拼接模式：每个view只包含一种模态
        # 与全局特征同等重要，使用相同的hidden_dim
        local_input_channels = rgb_channels  # 48 (latent channels, 单一模态)
        self.local_feature_extractor = nn.Sequential(
            nn.Conv3d(local_input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 空间池化
        )

        # 特征融合层 - 将全局和局部特征融合
        # 全局: hidden_dim, 局部: hidden_dim (方案A: 对等重要)
        self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # 多视角融合
        self.view_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 时间特征处理（不再需要上采样，因为VAE decoder已经完成了）
        # 注释掉temporal_processor：后续的Transformer和Cross-Attention已经能够充分处理时序依赖
        # self.temporal_processor = nn.Sequential(
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        # Cross-Attention: 未来帧 attend to 第一帧
        # 用于建模未来帧与第一帧之间的关系
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Transformer编码器 - 时间建模（处理cross-attention后的特征）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 预测头
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rotation_bins * 3)  # roll, pitch, yaw
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # open, close
        )

        # 初始夹爪状态嵌入层（当use_initial_gripper_state=True时使用）
        # 夹爪状态是离散值（0=关闭，1=打开），使用嵌入层将其映射到hidden_dim维度
        # 设计思路：将初始夹爪状态嵌入到历史帧特征中，然后未来帧通过cross attention attend历史帧
        if self.use_initial_gripper_state:
            self.gripper_state_embedding = nn.Embedding(2, hidden_dim)  # 2个状态: 关闭(0), 打开(1)
            # 融合初始夹爪状态到历史帧特征的投影层
            self.gripper_state_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

    def find_peak_positions_from_heatmap_images(
        self,
        heatmap_images: torch.Tensor,
        intermediate_h: int,
        intermediate_w: int,
        colormap_name: str = 'jet'
    ) -> torch.Tensor:
        """
        从完全解码的heatmap colormap图像中找到峰值位置，并映射到中间特征空间

        Args:
            heatmap_images: (b*v, 3, t, H, W) - 完全解码的heatmap colormap图像
            intermediate_h: 中间特征的高度
            intermediate_w: 中间特征的宽度
            colormap_name: colormap名称（用于从colormap提取真实的heatmap值）

        Returns:
            peak_positions: (b*v, t, 2) - 在中间特征空间的峰值位置 [h_idx, w_idx]
        """
        from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap

        bv, _, t, H, W = heatmap_images.shape

        # 计算从解码图像到中间特征的缩放比例
        scale_h = H / intermediate_h
        scale_w = W / intermediate_w

        peak_positions = []

        for t_idx in range(t):
            # 获取当前时间步的所有batch的heatmap图像: (b*v, 3, H, W)
            heatmap_t = heatmap_images[:, :, t_idx, :, :]

            batch_positions = []
            for i in range(bv):
                # 获取单个图像: (3, H, W)
                img = heatmap_t[i]

                # 转换为numpy格式: (H, W, 3)，值域[0, 1]
                img_np = img.permute(1, 2, 0).cpu().float().numpy()
                # 从[-1, 1]转换到[0, 1]（VAE解码输出通常是[-1, 1]）
                img_np = (img_np + 1.0) / 2.0
                img_np = np.clip(img_np, 0, 1)

                # 使用extract_heatmap_from_colormap从colormap提取真实的heatmap值
                heatmap_array = extract_heatmap_from_colormap(img_np, colormap_name)

                # 找峰值位置
                max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
                h_peak = max_pos[0]
                w_peak = max_pos[1]

                # 映射到中间特征空间的坐标
                h_peak_intermediate = int(h_peak / scale_h)
                w_peak_intermediate = int(w_peak / scale_w)

                # 边界检查
                h_peak_intermediate = min(h_peak_intermediate, intermediate_h - 1)
                w_peak_intermediate = min(w_peak_intermediate, intermediate_w - 1)

                batch_positions.append(torch.tensor([h_peak_intermediate, w_peak_intermediate],
                                                   device=heatmap_images.device))

            peak_positions.append(torch.stack(batch_positions, dim=0))  # (b*v, 2)

        # Stack所有时间步: (b*v, t, 2)
        peak_positions = torch.stack(peak_positions, dim=1)
        return peak_positions

    def find_peak_positions_from_intermediate(self, heatmap_features: torch.Tensor) -> torch.Tensor:
        """
        在中间特征上直接找峰值（近似方法，当没有完全解码的图像时使用）

        Args:
            heatmap_features: (b*v, c, t, h, w) - heatmap的中间特征

        Returns:
            peak_positions: (b*v, t, 2) - 峰值位置 [h_idx, w_idx]
        """
        bv, c, t, h, w = heatmap_features.shape
        # 对通道维度求平均
        heatmap_2d = heatmap_features.mean(dim=1)  # (b*v, t, h, w)

        peak_positions = []
        for t_idx in range(t):
            heatmap_t = heatmap_2d[:, t_idx, :, :]  # (b*v, h, w)
            heatmap_flat = heatmap_t.reshape(bv, -1)  # (b*v, h*w)
            max_indices = torch.argmax(heatmap_flat, dim=1)  # (b*v,)
            h_indices = max_indices // w
            w_indices = max_indices % w
            positions = torch.stack([h_indices, w_indices], dim=1)  # (b*v, 2)
            peak_positions.append(positions)

        return torch.stack(peak_positions, dim=1)  # (b*v, t, 2)

    def extract_local_features_at_peaks(
        self,
        combined_features: torch.Tensor,
        peak_positions: torch.Tensor,
        local_size: int = 3
    ) -> torch.Tensor:
        """
        在峰值位置附近提取局部特征

        Args:
            combined_features: (b*v, c, t, h, w) - 合并的RGB和Heatmap特征
            peak_positions: (b*v, t, 2) - 峰值位置 [h_idx, w_idx]
            local_size: 局部区域大小（奇数）

        Returns:
            local_features: (b*v, c, t, local_size, local_size) - 局部特征
        """
        bv, c, t, h, w = combined_features.shape
        half_size = local_size // 2

        # 初始化局部特征张量
        local_features = torch.zeros(bv, c, t, local_size, local_size,
                                    device=combined_features.device,
                                    dtype=combined_features.dtype)

        # 对每个batch和时间步提取局部特征
        for i in range(bv):
            for t_idx in range(t):
                # 获取峰值位置
                h_peak = peak_positions[i, t_idx, 0].item()
                w_peak = peak_positions[i, t_idx, 1].item()

                # 计算局部区域的边界（带边界检查）
                h_start = max(0, h_peak - half_size)
                h_end = min(h, h_peak + half_size + 1)
                w_start = max(0, w_peak - half_size)
                w_end = min(w, w_peak + half_size + 1)

                # 提取局部区域
                local_region = combined_features[i, :, t_idx, h_start:h_end, w_start:w_end]

                # 如果区域不够大（边界情况），进行padding
                local_h = h_end - h_start
                local_w = w_end - w_start

                if local_h < local_size or local_w < local_size:
                    # Padding到目标大小
                    pad_h_before = (local_size - local_h) // 2
                    pad_h_after = local_size - local_h - pad_h_before
                    pad_w_before = (local_size - local_w) // 2
                    pad_w_after = local_size - local_w - pad_w_before

                    local_region = F.pad(local_region,
                                       (pad_w_before, pad_w_after, pad_h_before, pad_h_after),
                                       mode='replicate')

                local_features[i, :, t_idx, :, :] = local_region

        return local_features

    def forward(
        self,
        rgb_latents: torch.Tensor = None,  # (b, 3, 48, t_compressed, h_latent, w_latent) - VAE latents, can be None
        heatmap_latents: torch.Tensor = None,  # (b, 3, 48, t_compressed, h_latent, w_latent) - VAE latents, required
        num_future_frames: int = None,  # T-1，预测的未来帧数
        heatmap_images: torch.Tensor = None,  # (b, 3, 3, t_upsampled, H, W) - 可选，完全解码的heatmap用于找峰值
        peak_positions: torch.Tensor = None,  # (b, t, 3, 2) - 可选，ground truth峰值位置 [h, w]（训练时使用）
        colormap_name: str = 'jet',  # colormap名称
        num_history_frames: int = 1,  # 历史帧数量（新增参数）
        history_gripper_states: torch.Tensor = None,  # (b, num_history_frames) - 每个历史帧的夹爪状态（0=关闭，1=打开），当use_initial_gripper_state=True时使用
    ):
        # Handle both modes: RGB+Heatmap (6 views) or Heatmap-only (3 views)
        if rgb_latents is not None:
            # Standard mode: RGB + Heatmap (6 views)
            b, v_rgb, c, t_compressed, h, w = rgb_latents.shape
            _, v_heatmap, _, _, _, _ = heatmap_latents.shape

            assert v_rgb == 3 and v_heatmap == 3, f"Expected 3 RGB views and 3 Heatmap views, got {v_rgb} and {v_heatmap}"
            assert c == 48, f"Expected 48 latent channels, got {c}"

            # View拼接模式：沿view维度拼接RGB和Heatmap latents
            # RGB views: 0-2, Heatmap views: 3-5
            combined_features = torch.cat([rgb_latents, heatmap_latents], dim=1)  # (b, 6, 48, t_compressed, h, w)
            v_total = 6  # 3 RGB + 3 Heatmap
        else:
            # Heatmap-only mode: only use heatmap latents (3 views)
            assert heatmap_latents is not None, "Either rgb_latents or heatmap_latents must be provided"
            b, v_heatmap, c, t_compressed, h, w = heatmap_latents.shape

            assert v_heatmap == 3, f"Expected 3 Heatmap views, got {v_heatmap}"
            assert c == 48, f"Expected 48 latent channels, got {c}"

            # Only use heatmap latents (3 views)
            combined_features = heatmap_latents  # (b, 3, 48, t_compressed, h, w)
            v_total = 3  # Only 3 Heatmap views

        # 2. 为每个视角提取全局特征
        # Reshape: (b, 6, 48, t, h, w) -> (b*6, 48, t, h, w)
        c_total = self.rgb_channels  # 48 (单一模态)
        combined_features_reshaped = combined_features.view(b * v_total, c_total, t_compressed, h, w)

        # 2.0 确定目标时间维度（用于后续上采样）
        # 如果有ground truth peak_positions，使用其时间维度作为目标
        # 否则使用target_frames (num_history_frames + num_future_frames)
        if peak_positions is not None:
            b_tmp, t_gt, v_gt, _ = peak_positions.shape
            target_t = t_gt
        else:
            target_t = num_history_frames + num_future_frames

        # 2.0.1 时间上采样features：从t_compressed上采样到target_t
        # VAE编码特性：第1帧单独编码，后续每4帧一组编码
        # 因此需要特殊处理第一帧，分别上采样
        if t_compressed != target_t:
            # 分离第一帧和后续帧
            first_frame_features = combined_features_reshaped[:, :, 0:1, :, :]  # (b*6, 48, 1, h, w)
            remaining_features = combined_features_reshaped[:, :, 1:, :, :]  # (b*6, 48, t_compressed-1, h, w)

            # 第一帧保持不变
            # 后续帧上采样到 (target_t - 1) 帧
            target_t_remaining = target_t - 1
            if remaining_features.shape[2] > 0 and target_t_remaining > 0:
                # 上采样后续帧：(b*6, 48, t_compressed-1, h, w) -> (b*6, 48, target_t-1, h, w)
                # trilinear需要5D输入 (N, C, D, H, W) 和 3D size (D_out, H_out, W_out)
                remaining_features_upsampled = F.interpolate(
                    remaining_features,  # (b*6, 48, t_compressed-1, h, w)
                    size=(target_t_remaining, h, w),
                    mode='trilinear',
                    align_corners=True if target_t_remaining > 1 else False
                )  # (b*6, 48, target_t-1, h, w)

                # 拼接第一帧和上采样后的后续帧
                combined_features_upsampled = torch.cat([first_frame_features, remaining_features_upsampled], dim=2)
            else:
                combined_features_upsampled = first_frame_features
        else:
            combined_features_upsampled = combined_features_reshaped

        # 2.1 提取全局特征（在上采样后的features上）
        global_features = self.global_feature_extractor(combined_features_upsampled)  # (b*v, hidden_dim, target_t, 1, 1)
        global_features = global_features.squeeze(-1).squeeze(-1)  # (b*v, hidden_dim, target_t)

        # 2.2 找到heatmap的峰值位置（只在heatmap views 3-5上进行）
        if peak_positions is not None:
            # 训练模式：使用ground truth峰值位置
            # peak_positions: (b, t_gt, 3, 2) -> (b, 3, t_gt, 2) -> (b*3, t_gt, 2)
            # 注意：不再对时间维度进行插值，保持完整的t_gt分辨率
            # 先调整维度顺序，将视角维度移到时间维度之前，避免内存混乱
            peak_positions_reshaped = peak_positions.permute(0, 2, 1, 3).reshape(b_tmp * v_gt, t_gt, 2)  # (b*3, t_gt, 2)

            # 缩放峰值位置到latent空间（只做空间缩放，不做时间插值）
            # Ground truth是在原始图像空间(H, W)，需要缩放到latent空间(h, w)
            # VAE的空间压缩比是16（256x256 -> 16x16）
            spatial_scale = 16
            peak_positions = (peak_positions_reshaped / spatial_scale).long()  # (b*3, t_gt, 2)
        elif heatmap_images is not None:
            # 推理模式：从解码的heatmap图像中找峰值
            # View拼接模式：heatmap_images只包含3个heatmap views
            # heatmap_images: (b, 3, 3, t_img, H, W) -> (b*3, 3, t_img, H, W)
            heatmap_images_reshaped = heatmap_images.view(b * v_heatmap, *heatmap_images.shape[2:])
            t_img = heatmap_images_reshaped.shape[2]

            # 注意：t_img应该与target_t一致（都是完整的时间分辨率）
            # 我们使用heatmap_images的时间维度进行峰值检测

            peak_positions = self.find_peak_positions_from_heatmap_images(
                heatmap_images_reshaped, h, w, colormap_name
            )  # (b*3, t_img, 2)

            # 如果时间维度不匹配target_t，需要插值到target_t
            if t_img != target_t:
                # 将peak_positions从t_img维度插值到target_t维度
                # peak_positions: (b*3, t_img, 2) -> (b*3, target_t, 2)
                peak_positions = peak_positions.permute(0, 2, 1)  # (b*3, 2, t_img)
                peak_positions = F.interpolate(peak_positions.float(), size=target_t, mode='linear',
                                              align_corners=True if target_t > 1 else False)
                peak_positions = peak_positions.permute(0, 2, 1)  # (b*3, target_t, 2)
                peak_positions = peak_positions.long()
        else:
            # 在上采样后的latent features上直接找峰值（近似方法）
            # 只在heatmap views上找峰值
            # 注意：使用上采样后的features (combined_features_upsampled)
            assert False
            heatmap_features_upsampled = combined_features_upsampled[b*3:, :, :, :, :]  # 提取heatmap views (b*3, 48, target_t, h, w)
            peak_positions = self.find_peak_positions_from_intermediate(heatmap_features_upsampled)  # (b*3, target_t, 2)

        # 2.3 扩展峰值位置到所有views
        # peak_positions目前是 (b*3, target_t, 2) - 只针对heatmap views
        if v_total == 6:
            # RGB+Heatmap mode: 扩展到 (b*6, target_t, 2) - 所有views都使用这些峰值
            # RGB views (0-2) 也使用heatmap views的峰值位置作为参考
            peak_positions_expanded = torch.cat([peak_positions, peak_positions], dim=0)  # (b*6, target_t, 2)
        else:
            # Heatmap-only mode: 已经是 (b*3, target_t, 2)，不需要扩展
            peak_positions_expanded = peak_positions  # (b*3, target_t, 2)

        # 2.4 提取局部特征（在上采样后的features上）
        local_features_raw = self.extract_local_features_at_peaks(
            combined_features_upsampled,
            peak_positions_expanded,
            local_size=self.local_feature_size
        )  # (b*6, c_total, target_t, local_size, local_size)

        local_features = self.local_feature_extractor(local_features_raw)  # (b*6, hidden_dim, target_t, 1, 1)
        local_features = local_features.squeeze(-1).squeeze(-1)  # (b*6, hidden_dim, target_t)

        # 2.5 融合全局和局部特征
        # global_features: (b*6, hidden_dim, target_t)
        # local_features: (b*6, hidden_dim, target_t)
        global_features = global_features.permute(0, 2, 1)  # (b*6, target_t, hidden_dim)
        local_features = local_features.permute(0, 2, 1)  # (b*6, target_t, hidden_dim)

        # 拼接全局和局部特征
        combined_global_local = torch.cat([global_features, local_features], dim=-1)  # (b*v_total, target_t, hidden_dim*2)

        # 融合
        features = self.feature_fusion(combined_global_local)  # (b*v_total, target_t, hidden_dim)

        # Reshape back: (b*v_total, target_t, hidden_dim) -> (b, v_total, target_t, hidden_dim)
        features = features.view(b, v_total, target_t, self.hidden_dim)

        # 3. 跨视角融合（在每个时间步）
        # v_total = 6 (RGB+Heatmap mode: view 0-2 RGB, view 3-5 Heatmap)
        # v_total = 3 (Heatmap-only mode: view 0-2 Heatmap)
        fused_features = []
        for t_idx in range(target_t):
            # 取出所有视角在时间步t的特征
            view_features = features[:, :, t_idx, :]  # (b, v_total, hidden_dim)
            # Multi-head attention跨视角融合
            fused, _ = self.view_attention(
                view_features, view_features, view_features
            )  # (b, v_total, hidden_dim)
            # 平均池化所有视角
            fused = fused.mean(dim=1)  # (b, hidden_dim)
            fused_features.append(fused)

        fused_features = torch.stack(fused_features, dim=1)  # (b, target_t, hidden_dim)

        # 4. 时间特征处理
        # 注意：已经在步骤2.0.1进行了时间上采样，这里的fused_features已经是target_t维度
        # target_t = num_history_frames + num_future_frames（包含所有帧：历史帧 + 未来帧）
        # 移除了temporal_processor，直接使用fused_features
        # 后续的Cross-Attention和Transformer会充分处理时序依赖
        processed_features = fused_features  # (b, target_t, hidden_dim)

        # 5. 分离历史帧和未来帧特征
        # 支持多历史帧：取前 num_history_frames 帧作为参考
        history_frame_features = processed_features[:, 0:num_history_frames, :]  # (b, num_history_frames, hidden_dim)

        # 根据帧数情况提取未来帧（跳过历史帧）
        if processed_features.shape[1] > num_history_frames + num_future_frames:
            # 有足够的帧，跳过历史帧
            future_frame_features = processed_features[:, num_history_frames:num_history_frames+num_future_frames, :]  # (b, num_future_frames, hidden_dim)
        elif processed_features.shape[1] >= num_history_frames + num_future_frames:
            # 刚好匹配或有足够的帧
            future_frame_features = processed_features[:, num_history_frames:num_history_frames+num_future_frames, :]  # (b, num_future_frames, hidden_dim)
        elif processed_features.shape[1] > num_history_frames:
            # 有部分未来帧，进行插值
            temp = processed_features[:, num_history_frames:, :]
            temp = temp.permute(0, 2, 1)  # (b, hidden_dim, t)
            temp = F.interpolate(temp, size=num_future_frames, mode='linear',
                               align_corners=True if num_future_frames > 1 else False)
            future_frame_features = temp.permute(0, 2, 1)  # (b, num_future_frames, hidden_dim)
        else:
            # 帧数不够，使用所有可用的帧进行插值
            temp = processed_features
            temp = temp.permute(0, 2, 1)  # (b, hidden_dim, t)
            temp = F.interpolate(temp, size=num_future_frames, mode='linear',
                               align_corners=True if num_future_frames > 1 else False)
            future_frame_features = temp.permute(0, 2, 1)  # (b, num_future_frames, hidden_dim)

        # 6. Cross-Attention: 未来帧 attend to 历史帧
        # Query: 未来帧特征, Key/Value: 历史帧特征（支持多历史帧）
        # 这样每个未来帧都可以从所有历史帧中提取相关信息来计算变化

        # 6.1 如果使用初始夹爪状态，将每个历史帧的夹爪状态嵌入到对应的历史帧特征中
        if self.use_initial_gripper_state and history_gripper_states is not None:
            # 获取每个历史帧的夹爪状态嵌入 (b, num_history_frames) -> (b, num_history_frames, hidden_dim)
            gripper_embedding = self.gripper_state_embedding(history_gripper_states.long())  # (b, num_history_frames, hidden_dim)
            # 拼接并融合：将每个历史帧的夹爪状态嵌入与对应的历史帧特征融合
            combined_history = torch.cat([history_frame_features, gripper_embedding], dim=-1)  # (b, num_history_frames, hidden_dim*2)
            history_frame_features = self.gripper_state_fusion(combined_history)  # (b, num_history_frames, hidden_dim)

        cross_attn_output, _ = self.cross_attention(
            query=future_frame_features,    # (b, num_future_frames, hidden_dim)
            key=history_frame_features,      # (b, num_history_frames, hidden_dim)
            value=history_frame_features     # (b, num_history_frames, hidden_dim)
        )  # (b, num_future_frames, hidden_dim)

        # 残差连接 + LayerNorm
        future_frame_features = self.cross_attn_norm(future_frame_features + cross_attn_output)

        # 7. Transformer时间建模（处理未来帧之间的时序依赖）
        temporal_features = self.transformer(future_frame_features)  # (b, num_future_frames, hidden_dim)

        # 8. 预测未来帧相对于第一帧的rotation变化量和gripper状态变化
        rotation_logits = self.rotation_head(temporal_features)  # (b, num_future_frames, num_bins*3)
        gripper_logits = self.gripper_head(temporal_features)  # (b, num_future_frames, 2)

        return rotation_logits, gripper_logits


class VAEFeatureExtractor:
    """
    VAE特征提取器 - 用于提取RGB和Heatmap的VAE latent特征

    主要方法：
    1. encode_videos: 返回VAE encoder的latent特征
    2. encode_videos_with_augmentation: 编码并支持噪声增强和缩放
    """

    def __init__(self, vae, device, torch_dtype=torch.bfloat16):
        self.vae = vae
        self.device = device
        self.torch_dtype = torch_dtype

    def preprocess_image(self, image, min_value=-1, max_value=1):
        """将 PIL.Image 转换为 torch.Tensor"""
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=self.torch_dtype, device=self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        # pattern: "B C H W"
        image = image.permute(2, 0, 1).unsqueeze(0)  # H W C -> 1 C H W
        return image

    def preprocess_video(self, video, min_value=-1, max_value=1):
        """
        将 list of PIL.Image 转换为 torch.Tensor
        参考 ModelManager.preprocess_video 的实现

        Args:
            video: List[PIL.Image] - 视频帧列表

        Returns:
            torch.Tensor - shape (1, C, T, H, W)
        """
        # Transform a list of PIL.Image to torch.Tensor
        video_tensors = [self.preprocess_image(image, min_value=min_value, max_value=max_value) for image in video]
        # Stack along time dimension: [(1, C, H, W)] -> (1, C, H, W) * T
        # We need to concatenate along a new time dimension
        # First stack to get (T, 1, C, H, W), then squeeze and permute
        video = torch.stack(video_tensors, dim=0)  # (T, 1, C, H, W)
        video = video.squeeze(1)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        video = video.unsqueeze(0)  # (1, C, T, H, W)
        return video

    @torch.no_grad()
    def encode_videos(self, rgb_videos, heatmap_videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """
        编码RGB和Heatmap视频，返回VAE encoder的latent特征

        Multi-view input: List[List[PIL.Image]] with shape [time][view]
        需要转换为 [view][time] 然后分别处理每个视角

        Args:
            rgb_videos: List[List[PIL.Image]] - [time][view] RGB视频
            heatmap_videos: List[List[PIL.Image]] - [time][view] Heatmap视频

        Returns:
            rgb_features: (num_views, c, t_compressed, h, w)
            heatmap_features: (num_views, c, t_compressed, h, w)
        """
        # Multi-view input: List[List[PIL.Image]] with shape (T, num_views)
        num_frames = len(rgb_videos)
        num_views = len(rgb_videos[0])

        # 按视角分组处理 - RGB
        all_rgb_view_latents = []
        for view_idx in range(num_views):
            # 提取当前视角的所有RGB帧: [time] -> List[PIL.Image]
            view_rgb_frames = [rgb_videos[t][view_idx] for t in range(num_frames)]
            # 预处理为tensor: (1, C, T, H, W)
            view_rgb_video = self.preprocess_video(view_rgb_frames)
            # Remove batch dimension: (1, C, T, H, W) -> (C, T, H, W)
            # VAE.encode expects (C, T, H, W) and will add batch dim internally
            view_rgb_video = view_rgb_video.squeeze(0)
            # VAE编码: (C, T, H, W) -> (c_latent, t_compressed, h_latent, w_latent)
            view_rgb_latents = self.vae.encode(
                [view_rgb_video],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            # 取第一个元素并转换类型
            view_rgb_latents = view_rgb_latents[0].to(dtype=self.torch_dtype, device=self.device)
            all_rgb_view_latents.append(view_rgb_latents)

        # 合并所有视角的RGB latents: List[(c, t, h, w)] -> (num_views, c, t, h, w)
        rgb_features = torch.stack(all_rgb_view_latents, dim=0)

        # 按视角分组处理 - Heatmap
        all_heatmap_view_latents = []
        for view_idx in range(num_views):
            # 提取当前视角的所有Heatmap帧
            view_heatmap_frames = [heatmap_videos[t][view_idx] for t in range(num_frames)]
            # 预处理为tensor: (1, C, T, H, W)
            view_heatmap_video = self.preprocess_video(view_heatmap_frames)
            # Remove batch dimension: (1, C, T, H, W) -> (C, T, H, W)
            view_heatmap_video = view_heatmap_video.squeeze(0)
            # VAE编码
            view_heatmap_latents = self.vae.encode(
                [view_heatmap_video],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            view_heatmap_latents = view_heatmap_latents[0].to(dtype=self.torch_dtype, device=self.device)
            all_heatmap_view_latents.append(view_heatmap_latents)

        # 合并所有视角的Heatmap latents: List[(c, t, h, w)] -> (num_views, c, t, h, w)
        heatmap_features = torch.stack(all_heatmap_view_latents, dim=0)

        return rgb_features, heatmap_features

    def _encode_single_view_with_history(self, video_frames, num_history_frames, tiled, tile_size, tile_stride, apply_heatmap_scale=False, heatmap_latent_scale=1.0, latent_noise_std=0.0):
        """
        根据历史帧数量编码单个视角的视频，返回latents

        Args:
            video_frames: List[PIL.Image] - 单个视角的视频帧列表
            num_history_frames: 历史帧数量
            tiled, tile_size, tile_stride: VAE编码参数
            apply_heatmap_scale: 是否应用heatmap缩放因子
            heatmap_latent_scale: heatmap latent的缩放因子
            latent_noise_std: latent噪声标准差

        Returns:
            latents: (c_latent, t_compressed, h_latent, w_latent)
        """
        num_frames = len(video_frames)

        if num_history_frames == 2 and num_frames >= 2:
            # 特殊处理：第1帧单独编码，第2帧+剩余帧正常编码
            # Frame 0 单独编码
            frame_0_video = self.preprocess_video([video_frames[0]])
            frame_0_video = frame_0_video.squeeze(0)
            z_0 = self.vae.encode([frame_0_video], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            z_0 = z_0[0].to(dtype=self.torch_dtype, device=self.device)

            # 应用缩放和噪声
            if apply_heatmap_scale and heatmap_latent_scale != 1.0:
                z_0 = z_0 * heatmap_latent_scale
            if latent_noise_std > 0:
                z_0 = z_0 + torch.randn_like(z_0) * latent_noise_std

            # Frame 1 + 剩余帧作为新视频正常编码
            remaining_frames = video_frames[1:]
            remaining_video = self.preprocess_video(remaining_frames)
            remaining_video = remaining_video.squeeze(0)
            z_remaining = self.vae.encode([remaining_video], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            z_remaining = z_remaining[0].to(dtype=self.torch_dtype, device=self.device)

            # 应用缩放和噪声
            if apply_heatmap_scale and heatmap_latent_scale != 1.0:
                z_remaining = z_remaining * heatmap_latent_scale
            if latent_noise_std > 0:
                z_remaining = z_remaining + torch.randn_like(z_remaining) * latent_noise_std

            # 在时间维度上concat latents
            latents = torch.cat([z_0, z_remaining], dim=1)  # (c, t_compressed, h, w)

        else:
            # 正常编码（1帧或1+4N帧）
            video = self.preprocess_video(video_frames)
            video = video.squeeze(0)
            latents = self.vae.encode([video], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            latents = latents[0].to(dtype=self.torch_dtype, device=self.device)

            # 应用缩放和噪声
            if apply_heatmap_scale and heatmap_latent_scale != 1.0:
                latents = latents * heatmap_latent_scale
            if latent_noise_std > 0:
                latents = latents + torch.randn_like(latents) * latent_noise_std

        return latents

    @torch.no_grad()
    def encode_videos_with_augmentation(
        self,
        rgb_videos,
        heatmap_videos,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        latent_noise_std=0.0,
        heatmap_latent_scale=1.0,
        num_history_frames=1
    ):
        """
        编码RGB和Heatmap视频到VAE latents，支持噪声增强和缩放

        Args:
            rgb_videos: List[List[PIL.Image]] - [time][view] RGB视频
            heatmap_videos: List[List[PIL.Image]] - [time][view] Heatmap视频
            tiled: 是否使用tiled编码
            tile_size: tile大小
            tile_stride: tile步长
            latent_noise_std: 添加到latent的高斯噪声标准差
            heatmap_latent_scale: heatmap latent的缩放因子
            num_history_frames: 历史帧数量

        Returns:
            rgb_latents: (num_views, c_latent, t_compressed, h_latent, w_latent)
            heatmap_latents: (num_views, c_latent, t_compressed, h_latent, w_latent)
        """
        num_frames = len(rgb_videos)
        num_views = len(rgb_videos[0])

        # 按视角分组处理 - RGB
        all_rgb_view_latents = []
        for view_idx in range(num_views):
            view_rgb_frames = [rgb_videos[t][view_idx] for t in range(num_frames)]
            view_rgb_latents = self._encode_single_view_with_history(
                view_rgb_frames, num_history_frames, tiled, tile_size, tile_stride,
                apply_heatmap_scale=False, latent_noise_std=latent_noise_std
            )
            all_rgb_view_latents.append(view_rgb_latents)

        rgb_latents = torch.stack(all_rgb_view_latents, dim=0)

        # 按视角分组处理 - Heatmap
        all_heatmap_view_latents = []
        for view_idx in range(num_views):
            view_heatmap_frames = [heatmap_videos[t][view_idx] for t in range(num_frames)]
            view_heatmap_latents = self._encode_single_view_with_history(
                view_heatmap_frames, num_history_frames, tiled, tile_size, tile_stride,
                apply_heatmap_scale=True, heatmap_latent_scale=heatmap_latent_scale,
                latent_noise_std=latent_noise_std
            )
            all_heatmap_view_latents.append(view_heatmap_latents)

        heatmap_latents = torch.stack(all_heatmap_view_latents, dim=0)

        return rgb_latents, heatmap_latents


def get_latent_cache_dir(data_root, trail_start, trail_end, image_size, num_augmentations=1):
    """
    生成latent缓存目录路径

    Args:
        data_root: 数据根目录
        trail_start: 起始trail编号
        trail_end: 结束trail编号
        image_size: 图像尺寸
        num_augmentations: 增强版本数量

    Returns:
        缓存目录路径: {data_root父目录}/{任务名}_{start}_{end}_{img_size}_aug{num}_latent
    """
    # 获取任务名（data_root的最后一级目录名）
    task_name = os.path.basename(os.path.normpath(data_root))
    # 获取父目录
    parent_dir = os.path.dirname(os.path.normpath(data_root))

    # 构建缓存目录名
    start_str = str(trail_start) if trail_start is not None else "all"
    end_str = str(trail_end) if trail_end is not None else "all"
    cache_dir_name = f"{task_name}_{start_str}_{end_str}_{image_size}_aug{num_augmentations}_latent"

    return os.path.join(parent_dir, cache_dir_name)

class CachedLatentDataset(torch.utils.data.Dataset):
    """
    从预计算的latent缓存加载数据的数据集

    支持多增强版本：
    - 每个样本可以有多个增强版本（sample_{idx}_aug_{aug_idx}.pt）
    - 训练时随机选择一个增强版本，保持数据多样性
    """
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

        # 获取所有缓存文件并组织成字典
        # key: sample_idx, value: list of augmentation file paths
        self.samples = {}

        for filename in os.listdir(cache_dir):
            if filename.startswith("sample_") and filename.endswith(".pt"):
                # 解析文件名：sample_{idx}_aug_{aug_idx}.pt 或 sample_{idx}.pt
                # 两种格式的parts[1]都是sample_idx
                parts = filename.replace(".pt", "").split("_")
                sample_idx = int(parts[1])

                if sample_idx not in self.samples:
                    self.samples[sample_idx] = []
                self.samples[sample_idx].append(filename)

        # 按sample_idx排序
        self.sample_indices = sorted(self.samples.keys())

        # 统计增强版本数量
        total_files = sum(len(files) for files in self.samples.values())
        avg_augmentations = total_files / len(self.sample_indices) if self.sample_indices else 0

        print(f"Loaded {len(self.sample_indices)} unique samples from {cache_dir}")
        print(f"Total cached files: {total_files} (avg {avg_augmentations:.1f} augmentations per sample)")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        # 随机选择一个增强版本
        available_files = self.samples[sample_idx]
        selected_file = np.random.choice(available_files)
        cache_path = os.path.join(self.cache_dir, selected_file)
        return torch.load(cache_path, map_location="cpu")


def precompute_and_cache_latents(
    dataset,
    vae_extractor,
    cache_dir,
    heatmap_latent_scale=1.0,
    num_augmentations=3,
):
    """
    预计算所有样本的VAE latent并缓存到磁盘

    为了保持数据增强的多样性，每个样本会预编码多个增强版本
    训练时随机选择一个版本，这样既保持速度优势又保留增强效果

    注意：这里只保存encoder的latent，不添加噪声
    训练时再添加噪声（直接在latent上操作，不需要decode）

    Args:
        dataset: 原始数据集
        vae_extractor: VAE特征提取器
        cache_dir: 缓存目录
        heatmap_latent_scale: heatmap latent缩放因子
        num_augmentations: 每个样本预编码的增强版本数量
    """
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Precomputing latents for {len(dataset)} samples with {num_augmentations} augmentations each...")
    print(f"Cache directory: {cache_dir}")
    print(f"Total files to generate: {len(dataset) * num_augmentations}")

    for idx in tqdm(range(len(dataset)), desc="Precomputing latents"):
        # 为每个样本生成多个增强版本
        for aug_idx in range(num_augmentations):
            cache_path = os.path.join(cache_dir, f"sample_{idx}_aug_{aug_idx}.pt")

            # 如果已经缓存过，跳过
            if os.path.exists(cache_path):
                continue

            # 每次调用dataset[idx]都会应用不同的随机增强
            sample = dataset[idx]

            # 提取数据
            input_video_rgb = sample['input_video_rgb']  # [time][view] - PIL Images
            input_video_heatmap = sample['video']  # [time][view] - PIL Images
            rotation_targets = sample['rotation_targets']
            gripper_targets = sample['gripper_targets']
            start_rotation = sample['start_rotation']
            start_gripper = sample['start_gripper']
            rotation_delta_targets = sample['rotation_delta_targets']
            gripper_change_targets = sample['gripper_change_targets']

            # 使用VAE encoder编码得到latent（不添加噪声，不decode）
            with torch.no_grad():
                rgb_latents, heatmap_latents = vae_extractor.encode_videos(
                    input_video_rgb, input_video_heatmap
                )

                # 应用heatmap latent缩放
                if heatmap_latent_scale != 1.0:
                    heatmap_latents = heatmap_latents * heatmap_latent_scale

            # 保存到磁盘（转到CPU以节省空间）
            cache_data = {
                'rgb_latents': rgb_latents.cpu(),  # (num_views, c, t_compressed, h, w)
                'heatmap_latents': heatmap_latents.cpu(),  # (num_views, c, t_compressed, h, w)
                'rotation_targets': rotation_targets,
                'gripper_targets': gripper_targets,
                'start_rotation': start_rotation,
                'start_gripper': start_gripper,
                # 新增：delta targets
                'rotation_delta_targets': rotation_delta_targets,
                'gripper_change_targets': gripper_change_targets,
            }
            torch.save(cache_data, cache_path)

    print(f"✓ Latent cache saved to {cache_dir}")


def collate_fn_with_cached_latents_no_decode(batch, latent_noise_std=0.0):
    """
    从缓存加载的latent进行处理的collate函数（不做VAE decode）

    优化版本：
    - 不在collate_fn中做VAE decode，而是返回latent
    - VAE decode在训练循环中进行，这样每个GPU可以并行decode
    - 不涉及CUDA操作，可以使用num_workers > 0
    - 支持batch_size > 1

    Args:
        batch: 从CachedLatentDataset加载的batch（list of samples）
        latent_noise_std: 添加到latent的高斯噪声标准差

    Returns:
        dict: 包含latent和目标的字典，所有tensor都在CPU上
    """
    all_rgb_latents = []
    all_heatmap_latents = []
    all_rotation_delta_targets = []
    all_gripper_change_targets = []
    all_initial_rotations = []
    all_initial_grippers = []

    for sample in batch:
        rgb_latents = sample['rgb_latents']  # (num_views, c, t_compressed, h, w)
        heatmap_latents = sample['heatmap_latents']
        rotation_delta_targets = sample['rotation_delta_targets']
        gripper_change_targets = sample['gripper_change_targets']
        start_rotation = sample['start_rotation']
        start_gripper = sample['start_gripper']

        # 在latent上添加噪声（CPU上操作）
        if latent_noise_std > 0:
            rgb_latents = rgb_latents + torch.randn_like(rgb_latents) * latent_noise_std
            heatmap_latents = heatmap_latents + torch.randn_like(heatmap_latents) * latent_noise_std

        # 处理维度
        if rotation_delta_targets.ndim == 2:
            rotation_delta_targets = rotation_delta_targets.unsqueeze(0)
        if gripper_change_targets.ndim == 1:
            gripper_change_targets = gripper_change_targets.unsqueeze(0)

        all_rgb_latents.append(rgb_latents)
        all_heatmap_latents.append(heatmap_latents)
        all_rotation_delta_targets.append(rotation_delta_targets)
        all_gripper_change_targets.append(gripper_change_targets)
        all_initial_rotations.append(start_rotation)
        all_initial_grippers.append(start_gripper)

    # Stack所有样本形成batch
    rgb_latents = torch.stack(all_rgb_latents, dim=0)  # (batch, num_views, c, t, h, w)
    heatmap_latents = torch.stack(all_heatmap_latents, dim=0)
    rotation_targets = torch.cat(all_rotation_delta_targets, dim=0)  # (batch, t-1, 3) - delta targets
    gripper_targets = torch.cat(all_gripper_change_targets, dim=0)  # (batch, t-1) - change targets
    initial_rotation = torch.stack(all_initial_rotations, dim=0)  # (batch, 3)
    initial_gripper = torch.stack(all_initial_grippers, dim=0)  # (batch,)

    # 从第一个样本获取 num_history_frames（缓存模式可能没有此字段，默认为1）
    num_history_frames = batch[0].get('num_history_frames', 1) if batch else 1

    return {
        'rgb_latents': rgb_latents,
        'heatmap_latents': heatmap_latents,
        'initial_rotation': initial_rotation,
        'initial_gripper': initial_gripper,
        'rotation_targets': rotation_targets,  # delta targets
        'gripper_targets': gripper_targets,  # change targets
        'num_future_frames': rotation_targets.shape[1],
        'num_history_frames': num_history_frames,  # 历史帧数量
    }


def collate_fn_with_vae(batch, vae_extractor, heatmap_latent_scale=1.0, latent_noise_std=0.0, return_heatmap_images=False, num_history_frames=1):
    """
    自定义collate函数（View拼接版本）- 返回VAE latents而不是decoder features

    流程：
    1. VAE encoder编码得到latent
    2. 在latent上添加噪声（数据增强）
    3. 直接返回latents（不经过decoder）

    Args:
        latent_noise_std: 添加到latent的高斯噪声标准差
        heatmap_latent_scale: heatmap latent的缩放因子
        return_heatmap_images: 是否返回heatmap图像（用于精确峰值检测）
        num_history_frames: 历史帧数量（用于VAE编码时的特殊处理）
    """
    # batch是一个列表，每个元素是dataset[i]
    # 由于Wan模型限制，我们一次只处理一个样本
    sample = batch[0]

    # 提取数据
    input_video_rgb = sample['input_video_rgb']  # [time][view] - PIL Images
    input_video_heatmap = sample['video']  # [time][view] - PIL Images
    rotation_targets = sample['rotation_delta_targets']  # (t-1, 3) - delta targets
    gripper_targets = sample['gripper_change_targets']  # (t-1,) - change targets

    # 提取初始状态（第0帧）
    start_rotation = sample['start_rotation']  # (3,) - 离散化的rotation索引
    start_gripper = sample['start_gripper']  # scalar - 离散化的gripper索引

    # 处理维度
    if rotation_targets.ndim == 2:
        rotation_targets = rotation_targets.unsqueeze(0)  # (1, t-1, 3)
    if gripper_targets.ndim == 1:
        gripper_targets = gripper_targets.unsqueeze(0)  # (1, t-1)

    # 使用VAE encoder编码 + 在latent上添加噪声，返回latents
    rgb_latents, heatmap_latents = vae_extractor.encode_videos_with_augmentation(
        input_video_rgb,
        input_video_heatmap,
        latent_noise_std=latent_noise_std,
        heatmap_latent_scale=heatmap_latent_scale,
        num_history_frames=num_history_frames
    )

    # 添加batch维度
    # rgb_latents: (v, c_latent, t_compressed, h, w) -> (1, v, c_latent, t_compressed, h, w)
    rgb_latents = rgb_latents.unsqueeze(0)
    heatmap_latents = heatmap_latents.unsqueeze(0)

    # 处理初始旋转和夹爪状态（第0帧的状态）
    initial_rotation = start_rotation.unsqueeze(0)  # (1, 3)
    initial_gripper = start_gripper.unsqueeze(0)  # (1,)

    # 获取历史帧数量（如果有）
    num_history_frames = sample.get('num_history_frames', 1)

    result = {
        'rgb_latents': rgb_latents,
        'heatmap_latents': heatmap_latents,
        'initial_rotation': initial_rotation,
        'initial_gripper': initial_gripper,
        'rotation_targets': rotation_targets,  # (1, t-1, 3)
        'gripper_targets': gripper_targets,  # (1, t-1)
        'num_future_frames': rotation_targets.shape[1],  # t-1
        'num_history_frames': num_history_frames,  # 历史帧数量
    }

    # 如果dataset提供了ground truth峰值位置（训练时使用）
    if 'img_locations' in sample:
        img_locations = sample['img_locations']  # (t, 3, 2)
        if img_locations.ndim == 3:
            img_locations = img_locations.unsqueeze(0)  # (1, t, 3, 2)
        result['img_locations'] = img_locations

    # 如果需要返回heatmap图像用于精确峰值检测
    if return_heatmap_images:
        # 将PIL图像转换为tensor格式: [time][view] -> (1, v, 3, t, H, W)
        from torchvision.transforms import ToTensor
        to_tensor = ToTensor()

        num_frames = len(input_video_heatmap)
        num_views = len(input_video_heatmap[0])

        # 按视角组织图像
        heatmap_images_list = []
        for view_idx in range(num_views):
            view_frames = [input_video_heatmap[t][view_idx] for t in range(num_frames)]
            # 转换为tensor并归一化到[-1, 1]
            view_tensor = torch.stack([to_tensor(img) * 2 - 1 for img in view_frames], dim=1)  # (3, t, H, W)
            heatmap_images_list.append(view_tensor)

        heatmap_images = torch.stack(heatmap_images_list, dim=0)  # (v, 3, t, H, W)
        heatmap_images = heatmap_images.unsqueeze(0)  # (1, v, 3, t, H, W)
        result['heatmap_images'] = heatmap_images # 这里是colormap版本的heatmap，而不是直接的heatmap

    return result


def train_epoch(
    model: MultiViewRotationGripperPredictorView,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    epoch_id: int,
    args,
    vae=None,
    use_accurate_peak_detection: bool = True,
    use_initial_gripper_state: bool = False,
    ):
    """训练一个epoch

    Args:
        vae: VAE模型，用于在训练循环中decode latent到features（多卡并行优化）
        use_accurate_peak_detection: 是否使用精确的峰值检测（需要完全解码heatmap图像）
        use_initial_gripper_state: 是否使用初始夹爪状态作为模型输入
    """
    model.train()

    epoch_loss = 0
    epoch_loss_rotation = 0
    epoch_loss_gripper = 0
    step_count = 0

    # 准确率累积
    rotation_acc_roll_sum = 0.0
    rotation_acc_pitch_sum = 0.0
    rotation_acc_yaw_sum = 0.0
    gripper_acc_sum = 0.0
    acc_step_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{args.num_epochs}")

    for step, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)

            # View拼接版本：直接使用latents，不经过VAE decoder上采样
            # 检查是使用cached latents还是在线编码
            if 'rgb_latents' in batch:
                # 缓存模式：直接使用预编码的latents
                rgb_latents = batch['rgb_latents'].to(accelerator.device)  # (b, 3, 48, t_compressed, h, w)
                heatmap_latents = batch['heatmap_latents'].to(accelerator.device)  # (b, 3, 48, t_compressed, h, w)
            else:
                # 在线编码模式：从rgb_sequence和heatmap_sequence编码
                rgb_sequence = batch['rgb_sequence'].to(accelerator.device)  # (b, t, v, 3, H, W)
                heatmap_sequence = batch['heatmap_sequence'].to(accelerator.device)  # (b, t, v, H, W)

                with torch.no_grad():
                    # 重排为 (b, v, 3, t, H, W) for RGB, (b, v, 1, t, H, W) for heatmap
                    b, t, v, _, H, W = rgb_sequence.shape
                    rgb_sequence = rgb_sequence.permute(0, 2, 3, 1, 4, 5)  # (b, v, 3, t, H, W)
                    heatmap_sequence = heatmap_sequence.permute(0, 2, 1, 3, 4).unsqueeze(2)  # (b, v, 1, t, H, W)

                    # 将heatmap从单通道扩展到3通道（匹配VAE输入）
                    heatmap_sequence = heatmap_sequence.repeat(1, 1, 3, 1, 1, 1)  # (b, v, 3, t, H, W)

                    # 逐view编码
                    rgb_latents_list = []
                    heatmap_latents_list = []
                    for v_idx in range(v):
                        rgb_v = rgb_sequence[:, v_idx]  # (b, 3, t, H, W)
                        heatmap_v = heatmap_sequence[:, v_idx]  # (b, 3, t, H, W)

                        # VAE编码
                        rgb_latent_v = vae.encode(rgb_v)  # (b, 48, t_compressed, h, w)
                        heatmap_latent_v = vae.encode(heatmap_v)  # (b, 48, t_compressed, h, w)

                        rgb_latents_list.append(rgb_latent_v)
                        heatmap_latents_list.append(heatmap_latent_v)

                    # 堆叠所有views
                    rgb_latents = torch.stack(rgb_latents_list, dim=1)  # (b, 3, 48, t_compressed, h, w)
                    heatmap_latents = torch.stack(heatmap_latents_list, dim=1)  # (b, 3, 48, t_compressed, h, w)

            # 峰值位置的获取：
            # 训练时优先使用ground truth（img_locations），推理时解码heatmap找峰值
            peak_positions_gt = None
            heatmap_images = None

            if 'img_locations' in batch:
                # 使用数据集提供的ground truth峰值位置（训练时）
                # img_locations: (b, num_poses, num_views, 2) - 假设num_poses = t+1 (包括初始帧)
                img_locations = batch['img_locations'].to(accelerator.device)
                peak_positions_gt = img_locations  # (b, t, 3, 2) in image space [h, w]
                # 不需要解码heatmap
            elif use_accurate_peak_detection:
                # 推理时：没有ground truth，需要解码heatmap获取峰值（测试时）
                with torch.no_grad():
                    # heatmap_latents: (b, 3, 48, t_compressed, h_latent, w_latent)
                    b, v, c, t, h, w = heatmap_latents.shape
                    heatmap_latents_flat = heatmap_latents.view(b * v, c, t, h, w)
                    # 完全解码heatmap
                    heatmap_images_flat = vae.decode(heatmap_latents_flat)  # (b*3, 3, t_full, H, W)
                    _, _, t_full, H, W = heatmap_images_flat.shape
                    heatmap_images = heatmap_images_flat.view(b, v, 3, t_full, H, W)  # (b, 3, 3, t_full, H, W)

            # 前向传播（不再需要initial_rotation和initial_gripper）
            # num_history_frames 支持多历史帧作为参考
            num_history_frames = batch.get('num_history_frames', 1)

            # 获取history_gripper_states（如果使用初始夹爪状态）
            # 兼容两种模式：
            # - 多帧历史模式：batch中有'history_gripper_states' (b, num_history_frames)
            # - 单帧历史模式：batch中有'initial_gripper' (b,)，需要转换为 (b, 1)
            history_gripper_states = None
            if use_initial_gripper_state:
                history_gripper_states = batch.get('history_gripper_states', None)
                if history_gripper_states is None:
                    # 单帧历史模式：使用initial_gripper并扩展维度
                    initial_gripper = batch.get('initial_gripper', None)
                    if initial_gripper is not None:
                        history_gripper_states = initial_gripper.unsqueeze(-1)  # (b,) -> (b, 1)
                if history_gripper_states is not None:
                    history_gripper_states = history_gripper_states.to(accelerator.device)

            # 根据use_heatmap_views_only决定是否传递rgb_latents
            if args.use_heatmap_views_only:
                # Heatmap-only模式：不传递rgb_latents
                model_rgb_latents = None
            else:
                # 默认模式：传递rgb_latents
                model_rgb_latents = rgb_latents

            rotation_logits, gripper_logits = model(
                rgb_latents=model_rgb_latents,  # None (heatmap only) or latents (RGB + heatmap)
                heatmap_latents=heatmap_latents,  # View-concat: direct latent input
                num_future_frames=batch['num_future_frames'],
                heatmap_images=heatmap_images,
                peak_positions=peak_positions_gt,  # Ground truth峰值位置（训练时使用）
                num_history_frames=num_history_frames,
                history_gripper_states=history_gripper_states,
            )

            # 计算loss
            rotation_targets = batch['rotation_targets'].to(accelerator.device)  # (b, t-1, 3)
            gripper_targets = batch['gripper_targets'].to(accelerator.device)  # (b, t-1)

            # Rotation loss (cross entropy for each of roll, pitch, yaw)
            num_bins = rotation_logits.shape[-1] // 3
            rotation_logits_reshaped = rotation_logits.view(
                rotation_logits.shape[0], rotation_logits.shape[1], 3, num_bins
            )  # (b, t-1, 3, num_bins)

            loss_roll = F.cross_entropy(
                rotation_logits_reshaped[:, :, 0, :].reshape(-1, num_bins),
                rotation_targets[:, :, 0].reshape(-1).long(),
                reduction='mean'
            )
            loss_pitch = F.cross_entropy(
                rotation_logits_reshaped[:, :, 1, :].reshape(-1, num_bins),
                rotation_targets[:, :, 1].reshape(-1).long(),
                reduction='mean'
            )
            loss_yaw = F.cross_entropy(
                rotation_logits_reshaped[:, :, 2, :].reshape(-1, num_bins),
                rotation_targets[:, :, 2].reshape(-1).long(),
                reduction='mean'
            )
            loss_rotation = (loss_roll + loss_pitch + loss_yaw) / 3.0

            # Gripper loss (cross entropy)
            loss_gripper = F.cross_entropy(
                gripper_logits.reshape(-1, 2),
                gripper_targets.reshape(-1).long(),
                reduction='mean'
            )

            # Total loss
            loss = loss_rotation + loss_gripper

            # 计算准确率
            with torch.no_grad():
                pred_roll = torch.argmax(rotation_logits_reshaped[:, :, 0, :], dim=-1)
                pred_pitch = torch.argmax(rotation_logits_reshaped[:, :, 1, :], dim=-1)
                pred_yaw = torch.argmax(rotation_logits_reshaped[:, :, 2, :], dim=-1)

                acc_roll = (pred_roll == rotation_targets[:, :, 0]).float().mean().item()
                acc_pitch = (pred_pitch == rotation_targets[:, :, 1]).float().mean().item()
                acc_yaw = (pred_yaw == rotation_targets[:, :, 2]).float().mean().item()

                rotation_acc_roll_sum += acc_roll
                rotation_acc_pitch_sum += acc_pitch
                rotation_acc_yaw_sum += acc_yaw

                pred_gripper = torch.argmax(gripper_logits, dim=-1)
                acc_gripper = (pred_gripper == gripper_targets).float().mean().item()
                gripper_acc_sum += acc_gripper

                acc_step_count += 1

            # 反向传播
            accelerator.backward(loss)

            # 梯度裁剪
            if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # 记录
            epoch_loss += loss.item()
            epoch_loss_rotation += loss_rotation.item()
            epoch_loss_gripper += loss_gripper.item()
            step_count += 1

            # SwanLab日志
            global_step = step + epoch_id * len(dataloader)
            should_log = (accelerator.is_main_process and
                         hasattr(args, 'swanlab_run') and args.swanlab_run is not None and
                         hasattr(args, 'logging_steps') and global_step % args.logging_steps == 0)

            if should_log:
                try:
                    import swanlab
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

                    log_data = {
                        "train_loss": loss.item(),
                        "train_loss_rotation": loss_rotation.item(),
                        "train_loss_roll": loss_roll.item(),
                        "train_loss_pitch": loss_pitch.item(),
                        "train_loss_yaw": loss_yaw.item(),
                        "train_loss_gripper": loss_gripper.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch_id,
                        "step": global_step
                    }

                    # 每200步记录准确率
                    if global_step % 200 == 0 and acc_step_count > 0:
                        avg_acc_roll = rotation_acc_roll_sum / acc_step_count
                        avg_acc_pitch = rotation_acc_pitch_sum / acc_step_count
                        avg_acc_yaw = rotation_acc_yaw_sum / acc_step_count
                        avg_acc_rotation = (avg_acc_roll + avg_acc_pitch + avg_acc_yaw) / 3.0
                        avg_acc_gripper = gripper_acc_sum / acc_step_count

                        log_data.update({
                            "train_acc_roll": avg_acc_roll,
                            "train_acc_pitch": avg_acc_pitch,
                            "train_acc_yaw": avg_acc_yaw,
                            "train_acc_rotation": avg_acc_rotation,
                            "train_acc_gripper": avg_acc_gripper,
                        })

                        # 重置
                        rotation_acc_roll_sum = 0.0
                        rotation_acc_pitch_sum = 0.0
                        rotation_acc_yaw_sum = 0.0
                        gripper_acc_sum = 0.0
                        acc_step_count = 0

                    swanlab.log(log_data, step=global_step)

                    # 打印
                    log_msg = f"Step {global_step}: loss={loss.item():.4f}, "
                    log_msg += f"loss_rot={loss_rotation.item():.4f} (r={loss_roll.item():.4f},p={loss_pitch.item():.4f},y={loss_yaw.item():.4f}), "
                    log_msg += f"loss_grip={loss_gripper.item():.4f}"
                    if "train_acc_rotation" in log_data:
                        log_msg += f", acc_rot={log_data['train_acc_rotation']:.4f}"
                        log_msg += f" (r={log_data['train_acc_roll']:.4f},p={log_data['train_acc_pitch']:.4f},y={log_data['train_acc_yaw']:.4f})"
                        log_msg += f", acc_grip={log_data['train_acc_gripper']:.4f}"
                    log_msg += f", lr={current_lr:.2e}"
                    print(log_msg)
                except Exception as e:
                    print(f"Warning: Failed to log to SwanLab: {e}")

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'loss_rot': f"{loss_rotation.item():.4f}",
                'loss_grip': f"{loss_gripper.item():.4f}",
                'avg_loss': f"{epoch_loss/step_count:.4f}"
            })

    avg_loss = epoch_loss / step_count if step_count > 0 else 0
    avg_loss_rotation = epoch_loss_rotation / step_count if step_count > 0 else 0
    avg_loss_gripper = epoch_loss_gripper / step_count if step_count > 0 else 0

    return avg_loss, avg_loss_rotation, avg_loss_gripper


def main():
    parser = argparse.ArgumentParser(description='Multi-View Rotation and Gripper Prediction Training')

    # 数据参数
    parser.add_argument('--data_root', type=str, nargs='+', required=True,
                       help='Data root directory (single path or list of task paths for multi-task training)')
    parser.add_argument('--trail_start', type=int, default=None,
                       help='Starting trail number (e.g., 1 for trail_1). If None, use all trails.')
    parser.add_argument('--trail_end', type=int, default=None,
                       help='Ending trail number (e.g., 50 for trail_50). If None, use all trails.')
    parser.add_argument('--sequence_length', type=int, default=12, help='Sequence length (including initial frame)')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    # 数据集参数
    parser.add_argument('--scene_bounds', type=str,
                       default="0,-0.45,-0.05,0.8,0.55,0.6",
                       help='Scene bounds as comma-separated values: x_min,y_min,z_min,x_max,y_max,z_max')
    parser.add_argument('--transform_augmentation_xyz', type=float, nargs=3,
                       default=[0.0, 0.0, 0.0],
                       help='Transform augmentation for xyz')
    parser.add_argument('--transform_augmentation_rpy', type=float, nargs=3,
                       default=[0.0, 0.0, 0.0],
                       help='Transform augmentation for roll/pitch/yaw')
    parser.add_argument('--wan_type', type=str,
                       default='5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP',
                       help='Wan model type')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--use_merged_pointcloud', action='store_true',
                       help='Use merged pointcloud from 3 cameras (default: False, only use camera 1)')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_rotation_bins', type=int, default=72, help='Number of rotation bins')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU (increase if GPU memory allows)')

    # 保存和日志
    parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    parser.add_argument('--save_epoch_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every N steps')
    parser.add_argument('--swanlab_project', type=str, default='mv_rot_grip', help='SwanLab project name')
    parser.add_argument('--swanlab_experiment', type=str, default=None, help='SwanLab experiment name')

    # VAE参数
    parser.add_argument('--model_base_path', type=str,
                       default='/data/lpy/huggingface/Wan2.2-TI2V-5B-fused',
                       help='Base model path for VAE')
    parser.add_argument('--heatmap_latent_scale', type=float, default=1.0, help='Heatmap latent scale factor')
    parser.add_argument('--latent_noise_std', type=float, default=0.1,
                       help='Standard deviation of Gaussian noise added to latents (for robustness training)')

    # 局部特征提取参数
    parser.add_argument('--use_accurate_peak_detection', action='store_true', default=True,
                       help='Use accurate peak detection by fully decoding heatmap images (default: True)')
    parser.add_argument('--local_feature_size', type=int, default=3,
                       help='Local feature extraction window size (default: 3)')

    # 缓存参数
    parser.add_argument('--num_augmentations', type=int, default=3,
                       help='Number of augmentation versions to precompute for each sample (default: 3)')
    parser.add_argument('--use_online_encoding', action='store_true', default=False,
                       help='Use online VAE encoding instead of cached latents (slower but no cache needed)')
    parser.add_argument('--use_different_projection', action='store_true', default=False,
                       help='Use different projection dataset (base_multi_view_dataset_with_rot_grip_3cam_different_projection.py)')

    # 多历史帧参数
    parser.add_argument('--num_history_frames', type=int, default=1,
                       help='Number of history frames (1, 2, or 1+4N). For multi-history, use wan_type=5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY')

    # 初始夹爪状态输入参数
    parser.add_argument('--use_initial_gripper_state', action='store_true', default=False,
                       help='Use initial gripper state as model input (only gripper state, not rotation)')

    # 视图选择参数
    parser.add_argument('--use_heatmap_views_only', action='store_true', default=False,
                       help='Use only heatmap views (3 views) as input, do not use RGB views (default: False, use all 6 views)')

    args = parser.parse_args()

    # ============================================
    # VSCode Debug支持
    # ============================================
    # 需要调试时，将下面的 ENABLE_DEBUG 改为 True
    ENABLE_DEBUG = False # 改为 True 启用VSCode调试
    DEBUG_PORT = 5678     # VSCode调试端口

    if ENABLE_DEBUG:
        try:
            import debugpy

            # 只在主进程启用debugpy（避免多卡训练时端口冲突）
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))

            if local_rank == 0:
                print("="*60)
                print(f"🐛 VSCode Debug Mode Enabled")
                print(f"  Listening on port: {DEBUG_PORT}")
                print(f"  Waiting for debugger to attach...")
                print("="*60)

                debugpy.listen(("0.0.0.0", DEBUG_PORT))
                debugpy.wait_for_client()

                print("="*60)
                print("✅ Debugger attached! Continuing...")
                print("="*60)
            else:
                print(f"[Rank {local_rank}] Skipping debugpy (only rank 0 uses debugger)")

        except ImportError:
            print("⚠️  debugpy not installed. Install: pip install debugpy")

    # 验证 num_history_frames 的合法性
    def is_valid_history_frames(n):
        if n == 1 or n == 2:
            return True
        if n > 2 and (n - 1) % 4 == 0:
            return True
        return False

    if not is_valid_history_frames(args.num_history_frames):
        raise ValueError(f"num_history_frames must be 1, 2, or 1+4N (5, 9, 13, ...), got {args.num_history_frames}")

    # 验证 wan_type 和 num_history_frames 的一致性（双向检测）
    if args.num_history_frames > 1 and args.wan_type != "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY":
        raise ValueError(
            f"num_history_frames={args.num_history_frames} > 1, but wan_type={args.wan_type}. "
            f"When using multi-frame history, you MUST set wan_type=5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY"
        )
    if args.num_history_frames == 1 and args.wan_type == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY":
        raise ValueError(
            f"wan_type=5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, but num_history_frames={args.num_history_frames}. "
            f"When using 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY, you MUST set num_history_frames > 1. "
            f"If you want single-frame mode, use wan_type=5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"
        )

    # 解析 scene_bounds 字符串为浮点数列表
    if isinstance(args.scene_bounds, str):
        args.scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
        if len(args.scene_bounds) != 6:
            raise ValueError(f"scene_bounds must have 6 values, got {len(args.scene_bounds)}")

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # SwanLab将在Accelerator初始化后进行
    # 这样可以确保使用正确的全局主进程判断
    args.swanlab_run = None

    # 加载VAE（使用标准的wan_video_vae，只需encode/decode功能）
    print("Loading VAE...")
    from diffsynth.models.wan_video_vae import WanVideoVAE38
    vae = WanVideoVAE38()
    # 加载权重
    vae_state_dict = torch.load(f"{args.model_base_path}/Wan2.2_VAE.pth", map_location="cpu")
    # 处理state_dict格式
    if 'model_state' in vae_state_dict:
        vae_state_dict = vae_state_dict['model_state']
    # 添加'model.'前缀
    vae_state_dict = {'model.' + k: v for k, v in vae_state_dict.items()}
    vae.load_state_dict(vae_state_dict,strict=True)
    vae = vae.eval().to(device="cuda", dtype=torch.bfloat16)
    vae_extractor = VAEFeatureExtractor(vae, device="cuda", torch_dtype=torch.bfloat16)
    print("✓ VAE loaded")

    # 创建数据集
    print("Creating dataset...")

    # 处理data_root - 支持单任务或多任务
    data_roots = args.data_root if isinstance(args.data_root, list) else [args.data_root]

    # 检查是否使用缓存（目前只支持单任务）
    use_cached_latents = False
    cache_dir = None

    if args.use_online_encoding:
        # 用户强制使用在线编码
        print("Using online VAE encoding (as specified by --use_online_encoding)")
        use_cached_latents = False
    elif len(data_roots) == 1:
        # 单任务模式：检查缓存
        cache_dir = get_latent_cache_dir(
            data_roots[0],
            args.trail_start,
            args.trail_end,
            args.image_size,
            args.num_augmentations
        )

        # 检查缓存是否存在
        if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
            # 缓存存在，使用缓存
            print(f"Found latent cache at: {cache_dir}")
            use_cached_latents = True
        else:
            print(f"No latent cache found. Will precompute and save to: {cache_dir}")

    # 根据 num_history_frames 选择正确的数据集工厂
    use_history_dataset = args.num_history_frames > 1
    if use_history_dataset:
        DatasetFactory = HeatmapDatasetFactoryWithHistory
        print(f"Using HeatmapDatasetFactoryWithHistory (num_history_frames={args.num_history_frames})")
    else:
        DatasetFactory = HeatmapDatasetFactory
        print(f"Using HeatmapDatasetFactory (num_history_frames={args.num_history_frames})")

    # 多任务训练或需要创建原始数据集
    if len(data_roots) > 1 or not use_cached_latents:
        if len(data_roots) > 1:
            print(f"Multi-task training mode: {len(data_roots)} tasks")
            print("Note: Latent caching is not supported for multi-task training yet.")
            datasets = []
            for task_idx, task_root in enumerate(data_roots):
                print(f"  Loading task {task_idx+1}/{len(data_roots)}: {task_root}")
                dataset_kwargs = {
                    'data_root': task_root,
                    'sequence_length': args.sequence_length,
                    'step_interval': 1,
                    'min_trail_length': 5,
                    'image_size': (args.image_size, args.image_size),
                    'sigma': 1.5,
                    'augmentation': True,
                    'mode': "train",
                    'scene_bounds': args.scene_bounds,
                    'transform_augmentation_xyz': args.transform_augmentation_xyz,
                    'transform_augmentation_rpy': args.transform_augmentation_rpy,
                    'debug': args.debug,
                    'colormap_name': "jet",
                    'repeat': 1,
                    'wan_type': args.wan_type,
                    'rotation_resolution': 360//args.num_rotation_bins,
                    'trail_start': args.trail_start,
                    'trail_end': args.trail_end,
                    'use_merged_pointcloud': args.use_merged_pointcloud,
                    'use_different_projection': args.use_different_projection,
                }
                # 添加多历史帧参数（仅当使用history dataset时）
                if use_history_dataset:
                    dataset_kwargs['num_history_frames'] = args.num_history_frames

                task_dataset = DatasetFactory.create_robot_trajectory_dataset(**dataset_kwargs)
                print(f"    ✓ Task {task_idx+1} loaded: {len(task_dataset)} samples")
                datasets.append(task_dataset)

            # 合并所有数据集
            raw_dataset = ConcatDataset(datasets)
            print(f"✓ Multi-task dataset created: {len(raw_dataset)} samples (from {len(data_roots)} tasks)")
        else:
            # 单任务训练
            print(f"Single-task training mode: {data_roots[0]}")
            dataset_kwargs = {
                'data_root': data_roots[0],
                'sequence_length': args.sequence_length,
                'step_interval': 1,
                'min_trail_length': 5,
                'image_size': (args.image_size, args.image_size),
                'sigma': 1.5,
                'augmentation': True,
                'mode': "train",
                'scene_bounds': args.scene_bounds,
                'transform_augmentation_xyz': args.transform_augmentation_xyz,
                'transform_augmentation_rpy': args.transform_augmentation_rpy,
                'debug': args.debug,
                'colormap_name': "jet",
                'repeat': 1,
                'wan_type': args.wan_type,
                'rotation_resolution': 360//args.num_rotation_bins,
                'trail_start': args.trail_start,
                'trail_end': args.trail_end,
                'use_merged_pointcloud': args.use_merged_pointcloud,
                'use_different_projection': args.use_different_projection,
            }
            # 添加多历史帧参数（仅当使用history dataset时）
            if use_history_dataset:
                dataset_kwargs['num_history_frames'] = args.num_history_frames

            raw_dataset = DatasetFactory.create_robot_trajectory_dataset(**dataset_kwargs)
            print(f"✓ Raw dataset created: {len(raw_dataset)} samples")

        # 如果是单任务且需要预计算缓存（但用户未指定在线编码）
        if len(data_roots) == 1 and not use_cached_latents and not args.use_online_encoding:
            print("\n" + "="*60)
            print("Precomputing latents (this may take a while)...")
            print(f"Generating {args.num_augmentations} augmentation(s) per sample")
            print("="*60)
            precompute_and_cache_latents(
                raw_dataset,
                vae_extractor,
                cache_dir,
                heatmap_latent_scale=args.heatmap_latent_scale,
                num_augmentations=args.num_augmentations,
            )
            use_cached_latents = True

    # 创建最终的数据集和DataLoader
    from functools import partial

    if use_cached_latents:
        # 使用缓存的latent（VAE decode在训练循环中进行，支持多卡并行）
        print(f"Using cached latents from: {cache_dir}")
        dataset = CachedLatentDataset(cache_dir)
        collate_fn = partial(
            collate_fn_with_cached_latents_no_decode,
            latent_noise_std=args.latent_noise_std
        )
        # 不涉及CUDA操作，可以使用多个worker和pin_memory
        num_workers = min(args.num_workers, 4) if args.num_workers > 0 else 4
        pin_memory = True
    else:
        # 在线编码模式：使用原始数据集（无缓存）
        dataset = raw_dataset
        collate_fn = partial(
            collate_fn_with_vae,
            vae_extractor=vae_extractor,
            heatmap_latent_scale=args.heatmap_latent_scale,
            latent_noise_std=args.latent_noise_std,
            return_heatmap_images=args.use_accurate_peak_detection,  # 在线编码时也可以使用精确峰值检测
            num_history_frames=args.num_history_frames
        )
        num_workers = args.num_workers
        pin_memory = False  # Must be False since collate_fn returns CUDA tensors

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"✓ DataLoader created (batch_size={args.batch_size}, cached={use_cached_latents}, num_workers={num_workers}, pin_memory={pin_memory})")

    # 创建模型（View拼接版本）
    print("Creating MultiViewRotationGripperPredictorView model...")

    # 根据use_heatmap_views_only决定num_views
    num_views = 3 if args.use_heatmap_views_only else 6

    if args.use_heatmap_views_only:
        print("  - Using HEATMAP-ONLY mode (3 views: only Heatmap)")
    else:
        print("  - Using VIEW concatenation (6 views: 3 RGB + 3 Heatmap)")
    print("  - Direct latent input (48 channels, not 256)")

    model = MultiViewRotationGripperPredictorView(
        rgb_channels=48,  # VAE latent channels (not decoder intermediate!)
        heatmap_channels=48,  # VAE latent channels (not decoder intermediate!)
        hidden_dim=args.hidden_dim,
        num_views=num_views,  # Dynamic: 3 (heatmap only) or 6 (RGB + Heatmap)
        num_rotation_bins=args.num_rotation_bins,
        dropout=args.dropout,
        local_feature_size=args.local_feature_size,
        use_initial_gripper_state=args.use_initial_gripper_state,
    )
    # Convert model to bfloat16 to match VAE latent dtype
    model = model.to(dtype=torch.bfloat16)
    print(f"✓ Model created (dtype: bfloat16, num_views={num_views}, heatmap_only={args.use_heatmap_views_only})")

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(dataloader),
        eta_min=1e-6
    )

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # 将VAE移到本地设备（每个进程有自己的VAE副本，支持多卡并行decode）
    vae = vae.to(accelerator.device)
    vae_extractor.device = accelerator.device
    print(f"✓ VAE moved to {accelerator.device}")

    # 初始化SwanLab（只在全局主进程，Accelerator初始化后）
    if accelerator.is_main_process:
        try:
            import swanlab
            args.swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment,
                config=vars(args)
            )
            print("✓ SwanLab initialized (main process only)")
        except Exception as e:
            print(f"Warning: Failed to initialize SwanLab: {e}")
            args.swanlab_run = None
    else:
        args.swanlab_run = None

    print("✓ Training setup complete")
    print(f"  - Device: {accelerator.device}")
    print(f"  - Num GPUs: {accelerator.num_processes}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")

    # 训练循环
    print("\n🚀 Starting training...")
    for epoch_id in range(args.num_epochs):
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch_id+1}/{args.num_epochs}")
            print(f"{'='*60}")

        avg_loss, avg_loss_rotation, avg_loss_gripper = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            epoch_id=epoch_id,
            args=args,
            vae=vae,
            use_accurate_peak_detection=args.use_accurate_peak_detection,
            use_initial_gripper_state=args.use_initial_gripper_state,
        )

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch_id+1} Summary:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg Rotation Loss: {avg_loss_rotation:.4f}")
            print(f"  Avg Gripper Loss: {avg_loss_gripper:.4f}")

            # 保存checkpoint
            if (epoch_id + 1) % args.save_epoch_interval == 0 or (epoch_id + 1) == args.num_epochs:
                checkpoint_path = os.path.join(args.output_path, f"epoch-{epoch_id+1}.pth")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch_id + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'avg_loss': avg_loss,
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    if accelerator.is_main_process:
        print("\n🎉 Training completed!")
        if args.swanlab_run is not None:
            args.swanlab_run.finish()


if __name__ == "__main__":
    main()
