"""
Multi-View Rotation and Gripper Prediction Training Script

This script trains a model to predict rotation (roll, pitch, yaw) and gripper state
from VAE-encoded RGB and heatmap features.

Key features:
- Uses VAE-compressed video features (RGB + Heatmap)
- Temporal compression: T frames -> 1 + (T-1)//4 frames
- Predicts rotation and gripper for future frames
- Uses SwanLab for logging
- Tracks accuracy metrics during training
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

from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip import HeatmapDatasetFactory


class MultiViewRotationGripperPredictor(nn.Module):
    """
    预测旋转和夹爪状态的模型 - 使用VAE Decoder的中间上采样特征

    输入：
        - rgb_features: VAE decoder上采样后的RGB中间特征 (b, v, c_intermediate, t_upsampled, h, w)
        - heatmap_features: VAE decoder上采样后的Heatmap中间特征 (b, v, c_intermediate, t_upsampled, h, w)
        - initial_rotation: 初始帧的旋转 (b, 3) - [roll, pitch, yaw] in bins
        - initial_gripper: 初始帧的夹爪状态 (b,) - binary
        - num_future_frames: 需要预测的未来帧数 (T-1，不包括初始帧)

    输出：
        - rotation_logits: (b, num_future_frames, num_rotation_bins*3) - 未来帧的rotation预测
        - gripper_logits: (b, num_future_frames, 2) - 未来帧的gripper预测

    核心设计：
        - t_upsampled 包含所有帧（初始帧 + 未来帧），约等于 T
        - 模型使用所有帧的视觉特征（包括初始帧）进行时间建模
        - 初始帧的rotation和gripper作为condition注入到所有时间步
        - Transformer处理所有帧后，只取未来帧的输出进行预测
        - 这样充分利用了初始帧的视觉和状态信息

    注意：
        - c_intermediate = 256 (VAE decoder最后一个上采样块的输出通道数)
        - VAE decoder已完成4x时间上采样，t_upsampled ≈ 原始帧数
    """

    def __init__(
        self,
        rgb_channels: int = 256,  # VAE decoder intermediate channels
        heatmap_channels: int = 256,  # VAE decoder intermediate channels
        hidden_dim: int = 512,
        num_views: int = 3,
        num_rotation_bins: int = 72,
        dropout: float = 0.1,
        vae = None,  # VAE对象，用于解码heatmap找峰值
        local_feature_size: int = 3,  # 局部特征提取的邻域大小
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.heatmap_channels = heatmap_channels
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_rotation_bins = num_rotation_bins
        self.vae = vae
        self.local_feature_size = local_feature_size

        # 全局特征提取器 - 为每个视角和每种模态提取全局特征
        # 输入是VAE decoder的中间特征，已经完成了时间上采样
        input_channels = rgb_channels + heatmap_channels  # 256 + 256 = 512
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
        # 输入是峰值附近的局部区域特征
        # 与全局特征同等重要，使用相同的hidden_dim
        local_input_channels = rgb_channels + heatmap_channels  # 256 + 256 = 512
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
        self.temporal_processor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 条件编码器 - 编码初始状态
        rotation_embed_dim = hidden_dim // 8
        gripper_embed_dim = hidden_dim // 8

        self.initial_rotation_encoder = nn.Embedding(num_rotation_bins, rotation_embed_dim)
        self.initial_gripper_encoder = nn.Embedding(2, gripper_embed_dim)

        # 特征融合: hidden_dim // 2 -> hidden_dim
        self.condition_proj = nn.Linear(hidden_dim // 2, hidden_dim)

        # Transformer编码器 - 时间建模
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
        rgb_features: torch.Tensor,  # (b, v, c_intermediate, t_upsampled, h, w)
        heatmap_features: torch.Tensor,  # (b, v, c_intermediate, t_upsampled, h, w)
        initial_rotation: torch.Tensor,  # (b, 3) - bin indices
        initial_gripper: torch.Tensor,  # (b,) - binary
        num_future_frames: int,  # T-1
        heatmap_images: torch.Tensor = None,  # (b, v, 3, t_upsampled, H, W) - 可选，完全解码的heatmap用于找峰值
        colormap_name: str = 'jet',  # colormap名称
    ):
        b, v, _, t_upsampled, h, w = rgb_features.shape

        # 1. 合并RGB和Heatmap特征
        combined_features = torch.cat([rgb_features, heatmap_features], dim=2)  # (b, v, c_rgb+c_hm, t, h, w)

        # 2. 为每个视角提取全局特征
        # Reshape: (b, v, c, t, h, w) -> (b*v, c, t, h, w)
        c_total = self.rgb_channels + self.heatmap_channels
        combined_features_reshaped = combined_features.view(b * v, c_total, t_upsampled, h, w)

        # 2.1 提取全局特征
        global_features = self.global_feature_extractor(combined_features_reshaped)  # (b*v, hidden_dim, t, 1, 1)
        global_features = global_features.squeeze(-1).squeeze(-1)  # (b*v, hidden_dim, t)

        # 2.2 找到heatmap的峰值位置
        if heatmap_images is not None:
            # 使用完全解码的heatmap图像找峰值（准确方法）
            # heatmap_images: (b, v, 3, t_img, H, W) -> (b*v, 3, t_img, H, W)
            heatmap_images_reshaped = heatmap_images.view(b * v, *heatmap_images.shape[2:])
            t_img = heatmap_images_reshaped.shape[2]

            # 检查时间维度是否匹配
            if t_img != t_upsampled:
                raise ValueError(
                    f"heatmap_images time dimension ({t_img}) does not match features time dimension ({t_upsampled}). "
                    f"This should not happen in normal operation."
                )

            peak_positions = self.find_peak_positions_from_heatmap_images(
                heatmap_images_reshaped, h, w, colormap_name
            )  # (b*v, t_upsampled, 2)
        else:
            # 在中间特征上直接找峰值（近似方法）
            heatmap_features_reshaped = heatmap_features.view(b * v, self.heatmap_channels, t_upsampled, h, w)
            peak_positions = self.find_peak_positions_from_intermediate(heatmap_features_reshaped)  # (b*v, t, 2)

        # 2.3 提取局部特征
        local_features_raw = self.extract_local_features_at_peaks(
            combined_features_reshaped,
            peak_positions,
            local_size=self.local_feature_size
        )  # (b*v, c_total, t, local_size, local_size)

        local_features = self.local_feature_extractor(local_features_raw)  # (b*v, hidden_dim, t, 1, 1)
        local_features = local_features.squeeze(-1).squeeze(-1)  # (b*v, hidden_dim, t)

        # 2.4 融合全局和局部特征
        # global_features: (b*v, hidden_dim, t)
        # local_features: (b*v, hidden_dim, t)
        global_features = global_features.permute(0, 2, 1)  # (b*v, t, hidden_dim)
        local_features = local_features.permute(0, 2, 1)  # (b*v, t, hidden_dim)

        # 拼接全局和局部特征
        combined_global_local = torch.cat([global_features, local_features], dim=-1)  # (b*v, t, hidden_dim*2)

        # 融合
        features = self.feature_fusion(combined_global_local)  # (b*v, t, hidden_dim)

        # Reshape back: (b*v, t, hidden_dim) -> (b, v, t, hidden_dim)
        features = features.view(b, v, t_upsampled, self.hidden_dim)

        # 3. 跨视角融合（在每个时间步）
        fused_features = []
        for t_idx in range(t_upsampled):
            # 取出所有视角在时间步t的特征
            view_features = features[:, :, t_idx, :]  # (b, v, hidden_dim)
            # Multi-head attention跨视角融合
            fused, _ = self.view_attention(
                view_features, view_features, view_features
            )  # (b, v, hidden_dim)
            # 平均池化所有视角
            fused = fused.mean(dim=1)  # (b, hidden_dim)
            fused_features.append(fused)

        fused_features = torch.stack(fused_features, dim=1)  # (b, t_upsampled, hidden_dim)

        # 4. 时间特征处理（VAE decoder已经完成了时间上采样）
        # 注意：t_upsampled 包含所有帧（初始帧 + 未来帧）
        # (b, t_upsampled, hidden_dim) -> (b, hidden_dim, t_upsampled)
        fused_features = fused_features.permute(0, 2, 1)
        processed_features = self.temporal_processor(fused_features)  # (b, hidden_dim, t_upsampled)
        # (b, hidden_dim, t_upsampled) -> (b, t_upsampled, hidden_dim)
        processed_features = processed_features.permute(0, 2, 1)

        # 5. 编码初始条件（第0帧的rotation和gripper）
        rot_embeds = []
        for i in range(3):
            rot_embeds.append(self.initial_rotation_encoder(initial_rotation[:, i]))
        rot_embed = torch.cat(rot_embeds, dim=-1)  # (b, 3 * hidden_dim//8)

        grip_embed = self.initial_gripper_encoder(initial_gripper)

        condition_embed = torch.cat([rot_embed, grip_embed], dim=-1)  # (b, hidden_dim//2)
        condition_embed = self.condition_proj(condition_embed)  # (b, hidden_dim)

        # 将初始条件添加到所有时间步（包括初始帧）
        # 这样模型能知道初始状态
        condition_embed = condition_embed.unsqueeze(1).expand(-1, processed_features.shape[1], -1)
        conditioned_features = processed_features + condition_embed

        # 6. Transformer时间建模（处理所有帧）
        # 使用所有帧的信息（包括初始帧）来建模时间依赖
        temporal_features = self.transformer(conditioned_features)  # (b, t_upsampled, hidden_dim)

        # 7. 只取未来帧的特征进行预测
        # 如果t_upsampled == num_future_frames + 1，说明包含初始帧
        # 我们跳过第0帧，只对后面的帧进行预测
        if temporal_features.shape[1] == num_future_frames + 1:
            # 跳过初始帧，只取未来帧
            future_features = temporal_features[:, 1:, :]  # (b, num_future_frames, hidden_dim)
        elif temporal_features.shape[1] == num_future_frames:
            # 长度已经匹配，直接使用
            future_features = temporal_features
        else:
            # 需要调整长度：使用插值或裁剪
            if temporal_features.shape[1] > num_future_frames:
                # 如果帧数太多，跳过初始帧并裁剪/插值到目标长度
                # 先跳过第0帧
                temp = temporal_features[:, 1:, :]  # 跳过第0帧
                # 如果还是太长，进行插值
                if temp.shape[1] > num_future_frames:
                    temp = temp.permute(0, 2, 1)  # (b, hidden_dim, t)
                    temp = F.interpolate(temp, size=num_future_frames, mode='linear',
                                       align_corners=True if num_future_frames > 1 else False)
                    future_features = temp.permute(0, 2, 1)  # (b, num_future_frames, hidden_dim)
                else:
                    future_features = temp
            else:
                # 如果帧数不够，进行插值
                temp = temporal_features.permute(0, 2, 1)  # (b, hidden_dim, t)
                temp = F.interpolate(temp, size=num_future_frames, mode='linear',
                                   align_corners=True if num_future_frames > 1 else False)
                future_features = temp.permute(0, 2, 1)  # (b, num_future_frames, hidden_dim)

        # 8. 预测未来帧的rotation和gripper
        rotation_logits = self.rotation_head(future_features)  # (b, num_future_frames, num_bins*3)
        gripper_logits = self.gripper_head(future_features)  # (b, num_future_frames, 2)

        return rotation_logits, gripper_logits


class VAEFeatureExtractor:
    """
    VAE特征提取器 - 用于提取RGB和Heatmap的VAE特征
    支持两种模式：
    1. encode_videos: 返回VAE encoder的latent特征
    2. encode_and_decode_intermediate: 先encode再decode，返回decoder的中间上采样特征
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

    @torch.no_grad()
    def encode_and_decode_intermediate(
        self,
        rgb_videos,
        heatmap_videos,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        latent_noise_std=0.0,
        heatmap_latent_scale=1.0
    ):
        """
        先用VAE encoder编码，可选地在latent上添加噪声，再用VAE decoder解码，返回decoder的中间上采样特征

        Multi-view input: List[List[PIL.Image]] with shape [time][view]

        Args:
            rgb_videos: List[List[PIL.Image]] - [time][view] RGB视频
            heatmap_videos: List[List[PIL.Image]] - [time][view] Heatmap视频
            tiled: 是否使用tiled编码
            tile_size: tile大小
            tile_stride: tile步长
            latent_noise_std: 添加到latent的高斯噪声标准差（在decode之前添加）
            heatmap_latent_scale: heatmap latent的缩放因子

        Returns:
            rgb_intermediate_features: (num_views, c_intermediate, t_upsampled, h_intermediate, w_intermediate)
            heatmap_intermediate_features: (num_views, c_intermediate, t_upsampled, h_intermediate, w_intermediate)

            其中:
            - c_intermediate = 256 (VAE decoder最后一个上采样块的输出通道数)
            - t_upsampled 是时间维度上采样后的长度（约为原始帧数）
            - h_intermediate, w_intermediate 是空间分辨率（原图的1/8）
        """
        # Multi-view input: List[List[PIL.Image]] with shape (T, num_views)
        num_frames = len(rgb_videos)
        num_views = len(rgb_videos[0])

        # 按视角分组处理 - RGB
        all_rgb_view_features = []
        for view_idx in range(num_views):
            # 提取当前视角的所有RGB帧
            view_rgb_frames = [rgb_videos[t][view_idx] for t in range(num_frames)]
            # 预处理为tensor
            view_rgb_video = self.preprocess_video(view_rgb_frames)
            view_rgb_video = view_rgb_video.squeeze(0)

            # VAE编码
            view_rgb_latents = self.vae.encode(
                [view_rgb_video],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )

            # 在latent上添加噪声（在decode之前）
            if latent_noise_std > 0:
                latent_noise = torch.randn_like(view_rgb_latents) * latent_noise_std
                view_rgb_latents = view_rgb_latents + latent_noise

            # VAE decode_intermediate - 获取decoder上采样后的中间特征
            view_rgb_intermediate = self.vae.decode_intermediate(
                view_rgb_latents,
                device=self.device,
                tiled=False  # intermediate模式暂不支持tiled
            )
            # 取第一个元素并转换类型
            view_rgb_intermediate = view_rgb_intermediate[0].to(dtype=self.torch_dtype, device=self.device)
            all_rgb_view_features.append(view_rgb_intermediate)

        # 合并所有视角的RGB中间特征
        rgb_intermediate_features = torch.stack(all_rgb_view_features, dim=0)

        # 按视角分组处理 - Heatmap
        all_heatmap_view_features = []
        for view_idx in range(num_views):
            view_heatmap_frames = [heatmap_videos[t][view_idx] for t in range(num_frames)]
            view_heatmap_video = self.preprocess_video(view_heatmap_frames)
            view_heatmap_video = view_heatmap_video.squeeze(0)

            # VAE编码
            view_heatmap_latents = self.vae.encode(
                [view_heatmap_video],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )

            # 应用heatmap latent缩放
            if heatmap_latent_scale != 1.0:
                view_heatmap_latents = view_heatmap_latents * heatmap_latent_scale

            # 在latent上添加噪声（在decode之前）
            if latent_noise_std > 0:
                latent_noise = torch.randn_like(view_heatmap_latents) * latent_noise_std
                view_heatmap_latents = view_heatmap_latents + latent_noise

            # VAE decode_intermediate
            view_heatmap_intermediate = self.vae.decode_intermediate(
                view_heatmap_latents,
                device=self.device,
                tiled=False
            )
            view_heatmap_intermediate = view_heatmap_intermediate[0].to(dtype=self.torch_dtype, device=self.device)
            all_heatmap_view_features.append(view_heatmap_intermediate)

        # 合并所有视角的Heatmap中间特征
        heatmap_intermediate_features = torch.stack(all_heatmap_view_features, dim=0)

        return rgb_intermediate_features, heatmap_intermediate_features


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
    训练时再添加噪声并decode_intermediate

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
    all_rotation_targets = []
    all_gripper_targets = []
    all_initial_rotations = []
    all_initial_grippers = []

    for sample in batch:
        rgb_latents = sample['rgb_latents']  # (num_views, c, t_compressed, h, w)
        heatmap_latents = sample['heatmap_latents']
        rotation_targets = sample['rotation_targets']
        gripper_targets = sample['gripper_targets']
        start_rotation = sample['start_rotation']
        start_gripper = sample['start_gripper']

        # 在latent上添加噪声（CPU上操作）
        if latent_noise_std > 0:
            rgb_latents = rgb_latents + torch.randn_like(rgb_latents) * latent_noise_std
            heatmap_latents = heatmap_latents + torch.randn_like(heatmap_latents) * latent_noise_std

        # 处理维度
        if rotation_targets.ndim == 2:
            rotation_targets = rotation_targets.unsqueeze(0)
        if gripper_targets.ndim == 1:
            gripper_targets = gripper_targets.unsqueeze(0)

        all_rgb_latents.append(rgb_latents)
        all_heatmap_latents.append(heatmap_latents)
        all_rotation_targets.append(rotation_targets)
        all_gripper_targets.append(gripper_targets)
        all_initial_rotations.append(start_rotation)
        all_initial_grippers.append(start_gripper)

    # Stack所有样本形成batch
    rgb_latents = torch.stack(all_rgb_latents, dim=0)  # (batch, num_views, c, t, h, w)
    heatmap_latents = torch.stack(all_heatmap_latents, dim=0)
    rotation_targets = torch.cat(all_rotation_targets, dim=0)  # (batch, t-1, 3)
    gripper_targets = torch.cat(all_gripper_targets, dim=0)  # (batch, t-1)
    initial_rotation = torch.stack(all_initial_rotations, dim=0)  # (batch, 3)
    initial_gripper = torch.stack(all_initial_grippers, dim=0)  # (batch,)

    return {
        'rgb_latents': rgb_latents,
        'heatmap_latents': heatmap_latents,
        'initial_rotation': initial_rotation,
        'initial_gripper': initial_gripper,
        'rotation_targets': rotation_targets,
        'gripper_targets': gripper_targets,
        'num_future_frames': rotation_targets.shape[1],
    }


def decode_latents_to_features(rgb_latents, heatmap_latents, vae, device, return_heatmap_images=False):
    """
    将latent解码为intermediate features（在训练循环中调用）

    Args:
        rgb_latents: (batch, num_views, c, t, h, w)
        heatmap_latents: (batch, num_views, c, t, h, w)
        vae: VAE模型（已在正确的device上）
        device: 计算设备
        return_heatmap_images: 是否同时返回完全解码的heatmap图像（用于找峰值）

    Returns:
        如果return_heatmap_images=False:
            rgb_features: (batch, num_views, c_intermediate, t_upsampled, h, w)
            heatmap_features: (batch, num_views, c_intermediate, t_upsampled, h, w)
        如果return_heatmap_images=True:
            rgb_features, heatmap_features, heatmap_images
            - heatmap_images: (batch, num_views, 3, t_upsampled, H, W) - 完全解码的heatmap colormap
    """
    batch_size = rgb_latents.shape[0]
    num_views = rgb_latents.shape[1]

    all_rgb_features = []
    all_heatmap_features = []
    all_heatmap_images = [] if return_heatmap_images else None

    with torch.no_grad():
        for b in range(batch_size):
            batch_rgb_features = []
            batch_heatmap_features = []
            batch_heatmap_images = [] if return_heatmap_images else None

            for view_idx in range(num_views):
                view_rgb_latent = rgb_latents[b, view_idx:view_idx+1].to(device=device, dtype=torch.bfloat16)
                view_heatmap_latent = heatmap_latents[b, view_idx:view_idx+1].to(device=device, dtype=torch.bfloat16)

                # RGB只需要中间特征
                rgb_intermediate = vae.decode_intermediate([view_rgb_latent.squeeze(0)], device=device)
                batch_rgb_features.append(rgb_intermediate[0])

                # Heatmap：根据需要返回中间特征或同时返回完全解码的图像
                if return_heatmap_images:
                    heatmap_intermediate, heatmap_full = vae.decode_intermediate_with_full(
                        [view_heatmap_latent.squeeze(0)], device=device
                    )
                    batch_heatmap_features.append(heatmap_intermediate[0])
                    batch_heatmap_images.append(heatmap_full[0])
                else:
                    heatmap_intermediate = vae.decode_intermediate([view_heatmap_latent.squeeze(0)], device=device)
                    batch_heatmap_features.append(heatmap_intermediate[0])

            all_rgb_features.append(torch.stack(batch_rgb_features, dim=0))
            all_heatmap_features.append(torch.stack(batch_heatmap_features, dim=0))
            if return_heatmap_images:
                all_heatmap_images.append(torch.stack(batch_heatmap_images, dim=0))

    rgb_features = torch.stack(all_rgb_features, dim=0)  # (batch, num_views, c, t, h, w)
    heatmap_features = torch.stack(all_heatmap_features, dim=0)

    if return_heatmap_images:
        heatmap_images = torch.stack(all_heatmap_images, dim=0)  # (batch, num_views, 3, t, H, W)
        return rgb_features, heatmap_features, heatmap_images
    return rgb_features, heatmap_features


def collate_fn_with_cached_latents(batch, vae, device, latent_noise_std=0.0):
    """
    从缓存加载的latent进行处理的collate函数（旧版本，保留兼容性）

    流程：
    1. 加载预计算的latent
    2. 在latent上添加噪声（数据增强）
    3. VAE decoder decode_intermediate获取上采样后的中间特征

    Args:
        batch: 从CachedLatentDataset加载的batch
        vae: VAE模型（用于decode_intermediate）
        device: 计算设备
        latent_noise_std: 添加到latent的高斯噪声标准差

    Returns:
        dict: 包含特征和目标的字典，所有tensor都在CUDA上（bfloat16）
    """
    sample = batch[0]

    rgb_latents = sample['rgb_latents']  # (num_views, c, t_compressed, h, w)
    heatmap_latents = sample['heatmap_latents']
    rotation_targets = sample['rotation_targets']
    gripper_targets = sample['gripper_targets']
    start_rotation = sample['start_rotation']
    start_gripper = sample['start_gripper']

    # 处理维度
    if rotation_targets.ndim == 2:
        rotation_targets = rotation_targets.unsqueeze(0)
    if gripper_targets.ndim == 1:
        gripper_targets = gripper_targets.unsqueeze(0)

    num_views = rgb_latents.shape[0]

    # 对每个视角分别处理
    all_rgb_features = []
    all_heatmap_features = []

    for view_idx in range(num_views):
        # 获取当前视角的latent
        view_rgb_latent = rgb_latents[view_idx:view_idx+1]  # (1, c, t, h, w)
        view_heatmap_latent = heatmap_latents[view_idx:view_idx+1]

        # 在latent上添加噪声
        if latent_noise_std > 0:
            view_rgb_latent = view_rgb_latent + torch.randn_like(view_rgb_latent) * latent_noise_std
            view_heatmap_latent = view_heatmap_latent + torch.randn_like(view_heatmap_latent) * latent_noise_std

        # 使用VAE decode_intermediate获取中间特征
        with torch.no_grad():
            # 移到设备上
            view_rgb_latent = view_rgb_latent.to(device=device, dtype=torch.bfloat16)
            view_heatmap_latent = view_heatmap_latent.to(device=device, dtype=torch.bfloat16)

            # decode_intermediate期望的输入是list
            rgb_intermediate = vae.decode_intermediate([view_rgb_latent.squeeze(0)], device=device)
            heatmap_intermediate = vae.decode_intermediate([view_heatmap_latent.squeeze(0)], device=device)

            all_rgb_features.append(rgb_intermediate[0])  # (c_intermediate, t_upsampled, h, w)
            all_heatmap_features.append(heatmap_intermediate[0])

    # 合并所有视角
    rgb_features = torch.stack(all_rgb_features, dim=0)  # (num_views, c, t, h, w)
    heatmap_features = torch.stack(all_heatmap_features, dim=0)

    # 添加batch维度
    rgb_features = rgb_features.unsqueeze(0)  # (1, num_views, c, t, h, w)
    heatmap_features = heatmap_features.unsqueeze(0)

    # 处理初始状态（移到device上）
    initial_rotation = start_rotation.unsqueeze(0).to(device)
    initial_gripper = start_gripper.unsqueeze(0).to(device)

    # 移动targets到device
    rotation_targets = rotation_targets.to(device)
    gripper_targets = gripper_targets.to(device)

    return {
        'rgb_features': rgb_features,
        'heatmap_features': heatmap_features,
        'initial_rotation': initial_rotation,
        'initial_gripper': initial_gripper,
        'rotation_targets': rotation_targets,
        'gripper_targets': gripper_targets,
        'num_future_frames': rotation_targets.shape[1],
    }


def collate_fn_with_vae(batch, vae_extractor, heatmap_latent_scale=1.0, latent_noise_std=0.0, return_heatmap_images=False):
    """
    自定义collate函数，将数据转换为VAE decoder的中间上采样特征

    流程：
    1. VAE encoder编码得到latent
    2. 在latent上添加噪声（数据增强）
    3. VAE decoder decode_intermediate获取上采样后的中间特征

    Args:
        latent_noise_std: 添加到latent的高斯噪声标准差（在decode之前添加）
        heatmap_latent_scale: heatmap latent的缩放因子
        return_heatmap_images: 是否返回heatmap图像（用于精确峰值检测）
    """
    # batch是一个列表，每个元素是dataset[i]
    # 由于Wan模型限制，我们一次只处理一个样本
    sample = batch[0]

    # 提取数据
    input_video_rgb = sample['input_video_rgb']  # [time][view] - PIL Images
    input_video_heatmap = sample['video']  # [time][view] - PIL Images
    rotation_targets = sample['rotation_targets']  # (t-1, 3) or (1, t-1, 3)
    gripper_targets = sample['gripper_targets']  # (t-1,) or (1, t-1)

    # 提取初始状态（第0帧）
    start_rotation = sample['start_rotation']  # (3,) - 离散化的rotation索引
    start_gripper = sample['start_gripper']  # scalar - 离散化的gripper索引

    # 处理维度
    if rotation_targets.ndim == 2:
        rotation_targets = rotation_targets.unsqueeze(0)  # (1, t-1, 3)
    if gripper_targets.ndim == 1:
        gripper_targets = gripper_targets.unsqueeze(0)  # (1, t-1)

    # 使用VAE encoder编码 + 在latent上添加噪声 + decoder解码获取中间上采样特征
    rgb_features, heatmap_features = vae_extractor.encode_and_decode_intermediate(
        input_video_rgb,
        input_video_heatmap,
        latent_noise_std=latent_noise_std,
        heatmap_latent_scale=heatmap_latent_scale
    )

    # 添加batch维度
    # rgb_features: (v, c_intermediate, t_upsampled, h, w) -> (1, v, c_intermediate, t_upsampled, h, w)
    rgb_features = rgb_features.unsqueeze(0)
    heatmap_features = heatmap_features.unsqueeze(0)

    # 处理初始旋转和夹爪状态（第0帧的状态）
    initial_rotation = start_rotation.unsqueeze(0)  # (1, 3)
    initial_gripper = start_gripper.unsqueeze(0)  # (1,)

    result = {
        'rgb_features': rgb_features,
        'heatmap_features': heatmap_features,
        'initial_rotation': initial_rotation,
        'initial_gripper': initial_gripper,
        'rotation_targets': rotation_targets,  # (1, t-1, 3)
        'gripper_targets': gripper_targets,  # (1, t-1)
        'num_future_frames': rotation_targets.shape[1],  # t-1
    }

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
        result['heatmap_images'] = heatmap_images

    return result


def train_epoch(
    model: MultiViewRotationGripperPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    epoch_id: int,
    args,
    vae=None,
    use_accurate_peak_detection: bool = True,
):
    """训练一个epoch

    Args:
        vae: VAE模型，用于在训练循环中decode latent到features（多卡并行优化）
        use_accurate_peak_detection: 是否使用精确的峰值检测（需要完全解码heatmap图像）
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

            # 检查是使用cached latents还是在线编码
            if 'rgb_latents' in batch:
                # 缓存模式：在训练循环中decode latent到features（每个GPU并行decode）
                rgb_latents = batch['rgb_latents'].to(accelerator.device)
                heatmap_latents = batch['heatmap_latents'].to(accelerator.device)

                # 根据是否需要精确峰值检测，决定是否返回完全解码的heatmap图像
                if use_accurate_peak_detection:
                    rgb_features, heatmap_features, heatmap_images = decode_latents_to_features(
                        rgb_latents, heatmap_latents, vae, accelerator.device, return_heatmap_images=True
                    )
                else:
                    rgb_features, heatmap_features = decode_latents_to_features(
                        rgb_latents, heatmap_latents, vae, accelerator.device, return_heatmap_images=False
                    )
                    heatmap_images = None
            else:
                # 在线编码模式：batch已经包含features
                rgb_features = batch['rgb_features'].to(accelerator.device)
                heatmap_features = batch['heatmap_features'].to(accelerator.device)
                # 在线编码模式下，如果开启精确峰值检测，heatmap_images会从原始输入图像中获取
                heatmap_images = batch.get('heatmap_images', None)
                if heatmap_images is not None:
                    heatmap_images = heatmap_images.to(accelerator.device)

            # 前向传播
            rotation_logits, gripper_logits = model(
                rgb_features=rgb_features,
                heatmap_features=heatmap_features,
                initial_rotation=batch['initial_rotation'].to(accelerator.device),
                initial_gripper=batch['initial_gripper'].to(accelerator.device),
                num_future_frames=batch['num_future_frames'],
                heatmap_images=heatmap_images,
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
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length (including initial frame)')
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

    args = parser.parse_args()

    # 解析 scene_bounds 字符串为浮点数列表
    if isinstance(args.scene_bounds, str):
        args.scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
        if len(args.scene_bounds) != 6:
            raise ValueError(f"scene_bounds must have 6 values, got {len(args.scene_bounds)}")

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 初始化SwanLab
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        try:
            import swanlab
            args.swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment,
                config=vars(args)
            )
            print("✓ SwanLab initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize SwanLab: {e}")
            args.swanlab_run = None
    else:
        args.swanlab_run = None

    # 加载VAE（使用新的wan_video_vae_2中的类，支持decode_intermediate）
    print("Loading VAE...")
    from diffsynth.models.wan_video_vae_2 import WanVideoVAE38
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
    print("✓ VAE loaded (with decode_intermediate support)")

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

    # 多任务训练或需要创建原始数据集
    if len(data_roots) > 1 or not use_cached_latents:
        if len(data_roots) > 1:
            print(f"Multi-task training mode: {len(data_roots)} tasks")
            print("Note: Latent caching is not supported for multi-task training yet.")
            datasets = []
            for task_idx, task_root in enumerate(data_roots):
                print(f"  Loading task {task_idx+1}/{len(data_roots)}: {task_root}")
                task_dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
                    data_root=task_root,
                    sequence_length=args.sequence_length,
                    step_interval=1,
                    min_trail_length=10,
                    image_size=(args.image_size, args.image_size),
                    sigma=1.5,
                    augmentation=True,
                    mode="train",
                    scene_bounds=args.scene_bounds,
                    transform_augmentation_xyz=args.transform_augmentation_xyz,
                    transform_augmentation_rpy=args.transform_augmentation_rpy,
                    debug=args.debug,
                    colormap_name="jet",
                    repeat=1,
                    wan_type=args.wan_type,
                    rotation_resolution=360//args.num_rotation_bins,
                    trail_start=args.trail_start,
                    trail_end=args.trail_end,
                )
                print(f"    ✓ Task {task_idx+1} loaded: {len(task_dataset)} samples")
                datasets.append(task_dataset)

            # 合并所有数据集
            raw_dataset = ConcatDataset(datasets)
            print(f"✓ Multi-task dataset created: {len(raw_dataset)} samples (from {len(data_roots)} tasks)")
        else:
            # 单任务训练
            print(f"Single-task training mode: {data_roots[0]}")
            raw_dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
                data_root=data_roots[0],
                sequence_length=args.sequence_length,
                step_interval=1,
                min_trail_length=10,
                image_size=(args.image_size, args.image_size),
                sigma=1.5,
                augmentation=True,
                mode="train",
                scene_bounds=args.scene_bounds,
                transform_augmentation_xyz=args.transform_augmentation_xyz,
                transform_augmentation_rpy=args.transform_augmentation_rpy,
                debug=args.debug,
                colormap_name="jet",
                repeat=1,
                wan_type=args.wan_type,
                rotation_resolution=360//args.num_rotation_bins,
                trail_start=args.trail_start,
                trail_end=args.trail_end,
            )
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
            return_heatmap_images=args.use_accurate_peak_detection  # 在线编码时也可以使用精确峰值检测
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

    # 创建模型
    print("Creating model...")
    model = MultiViewRotationGripperPredictor(
        rgb_channels=256,  # VAE decoder intermediate channels
        heatmap_channels=256,  # VAE decoder intermediate channels
        hidden_dim=args.hidden_dim,
        num_views=3,
        num_rotation_bins=args.num_rotation_bins,
        dropout=args.dropout,
        local_feature_size=args.local_feature_size,
    )
    # Convert model to bfloat16 to match VAE feature dtype
    model = model.to(dtype=torch.bfloat16)
    print("✓ Model created (dtype: bfloat16)")

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
    print(f"✓ VAE moved to {accelerator.device}")

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
