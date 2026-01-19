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
from diffsynth import ModelManager, WanVideoPipeline


class MultiViewRotationGripperPredictor(nn.Module):
    """
    预测旋转和夹爪状态的模型

    输入：
        - rgb_features: VAE编码的RGB特征 (b, v, c_rgb, t_compressed, h, w)
        - heatmap_features: VAE编码的Heatmap特征 (b, v, c_hm, t_compressed, h, w)
        - initial_rotation: 初始帧的旋转 (b, 3) - [roll, pitch, yaw] in bins
        - initial_gripper: 初始帧的夹爪状态 (b,) - binary
    输出：
        - rotation_logits: (b, t_future, num_rotation_bins*3) - 每个future frame的rotation预测
        - gripper_logits: (b, t_future, 2) - 每个future frame的gripper预测

    其中 t_compressed = 1 + (T-1)//4, t_future = T-1 (因为不预测初始帧)
    """

    def __init__(
        self,
        rgb_channels: int = 48,  # VAE latent channels
        heatmap_channels: int = 48,  # VAE latent channels
        hidden_dim: int = 512,
        num_views: int = 3,
        num_rotation_bins: int = 72,
        temporal_upsample_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.heatmap_channels = heatmap_channels
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_rotation_bins = num_rotation_bins
        self.temporal_upsample_factor = temporal_upsample_factor

        # 特征提取器 - 为每个视角和每种模态提取特征
        input_channels = rgb_channels + heatmap_channels
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 空间池化: (b*v, hidden_dim, t, 1, 1)
        ) # 一定要拼接heatmap和 color嘛？

        # 多视角融合
        self.view_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 时间上采样模块 - 基于VAE压缩特性的设计
        # VAE压缩策略: 保留第1帧 + 后续帧4x压缩
        # compressed: 1 + (T-1)//4, target: T-1 (future frames only)

        # 第0帧特征处理 - 单独处理完整保留的第一帧
        self.first_frame_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 后续帧上采样 - 对压缩的帧进行4x上采样
        self.compressed_frames_upsampler = nn.Sequential(
            # 特征增强
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 4x转置卷积上采样
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # 特征提炼
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 时间融合 - 融合第0帧信息到上采样的帧
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 条件编码器 - 编码初始状态
        # rotation有3个维度(roll, pitch, yaw), gripper有1个维度
        # 设计使得拼接后的维度恰好匹配，避免截断
        # 总维度 = 3 * rotation_dim + 1 * gripper_dim
        # 让 3 * rotation_dim + gripper_dim = hidden_dim // 2
        # 设 rotation_dim = hidden_dim // 8, gripper_dim = hidden_dim // 8
        # 则 3 * hidden_dim//8 + hidden_dim//8 = 4 * hidden_dim//8 = hidden_dim // 2 ✓
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

    def forward(
        self,
        rgb_features: torch.Tensor,  # (b, v, c_rgb, t_compressed, h, w)
        heatmap_features: torch.Tensor,  # (b, v, c_hm, t_compressed, h, w)
        initial_rotation: torch.Tensor,  # (b, 3) - bin indices
        initial_gripper: torch.Tensor,  # (b,) - binary
        num_future_frames: int,  # T-1
    ):
        b, v, _, t_compressed, h, w = rgb_features.shape

        # 1. 合并RGB和Heatmap特征
        combined_features = torch.cat([rgb_features, heatmap_features], dim=2)  # (b, v, c_rgb+c_hm, t, h, w)

        # 2. 为每个视角提取特征
        # Reshape: (b, v, c, t, h, w) -> (b*v, c, t, h, w)
        c_total = self.rgb_channels + self.heatmap_channels
        combined_features = combined_features.view(b * v, c_total, t_compressed, h, w)
        features = self.feature_extractor(combined_features)  # (b*v, hidden_dim, t, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (b*v, hidden_dim, t)
        features = features.permute(0, 2, 1)  # (b*v, t, hidden_dim)

        # Reshape back: (b*v, t, hidden_dim) -> (b, v, t, hidden_dim)
        features = features.view(b, v, t_compressed, self.hidden_dim)

        # 3. 跨视角融合（在每个时间步）
        fused_features = []
        for t_idx in range(t_compressed):
            # 取出所有视角在时间步t的特征
            view_features = features[:, :, t_idx, :]  # (b, v, hidden_dim)
            # Multi-head attention跨视角融合
            fused, _ = self.view_attention(
                view_features, view_features, view_features
            )  # (b, v, hidden_dim)
            # 平均池化所有视角
            fused = fused.mean(dim=1)  # (b, hidden_dim)
            fused_features.append(fused)

        fused_features = torch.stack(fused_features, dim=1)  # (b, t_compressed, hidden_dim)

        # 4. 基于VAE压缩特性的时间上采样
        # VAE压缩: [frame_0(完整), compressed_frames(1+(T-1)//4)]
        # 目标: 预测future frames (T-1帧)

        # 分离第0帧和压缩帧
        first_frame_features = fused_features[:, 0, :]  # (b, hidden_dim) - 完整保留的第一帧
        compressed_features = fused_features[:, 1:, :]  # (b, t_compressed-1, hidden_dim) - 压缩的后续帧

        # 处理第0帧特征
        first_frame_features = self.first_frame_proj(first_frame_features)  # (b, hidden_dim)

        # 对压缩帧进行4x上采样
        if compressed_features.shape[1] > 0:  # 如果有压缩帧
            # (b, t_compressed-1, hidden_dim) -> (b, hidden_dim, t_compressed-1)
            compressed_features = compressed_features.permute(0, 2, 1)
            # 4x上采样: (b, hidden_dim, t_compressed-1) -> (b, hidden_dim, (t_compressed-1)*4)
            upsampled_compressed = self.compressed_frames_upsampler(compressed_features)

            # 将第0帧特征broadcast到时间维度，作为参考信息
            # (b, hidden_dim) -> (b, hidden_dim, 1)
            first_frame_broadcast = first_frame_features.unsqueeze(-1)
            # 扩展到上采样后的长度: (b, hidden_dim, 1) -> (b, hidden_dim, upsampled_len)
            first_frame_broadcast = first_frame_broadcast.expand(-1, -1, upsampled_compressed.shape[2])

            # 融合第0帧信息 (residual connection)
            upsampled_features = upsampled_compressed + self.temporal_fusion(first_frame_broadcast)
        else:
            # 如果只有第0帧，直接使用
            upsampled_features = first_frame_features.unsqueeze(-1)  # (b, hidden_dim, 1)

        # 如果上采样后的长度与目标长度不匹配，进行微调
        if upsampled_features.size(2) != num_future_frames:
            upsampled_features = F.interpolate(
                upsampled_features,
                size=num_future_frames,
                mode='linear',
                align_corners=True if num_future_frames > 1 else False
            )  # (b, hidden_dim, num_future_frames)

        # (b, hidden_dim, num_future_frames) -> (b, num_future_frames, hidden_dim)
        upsampled_features = upsampled_features.permute(0, 2, 1)

        # 5. 编码初始条件
        # initial_rotation: (b, 3) -> 3个embedding -> concat
        # 每个rotation维度的embedding: (b, hidden_dim//8)
        rot_embeds = []
        for i in range(3):
            rot_embeds.append(self.initial_rotation_encoder(initial_rotation[:, i]))
        rot_embed = torch.cat(rot_embeds, dim=-1)  # (b, 3 * hidden_dim//8)

        # gripper embedding: (b, hidden_dim//8)
        grip_embed = self.initial_gripper_encoder(initial_gripper)

        # 拼接: (b, 3*hidden_dim//8 + hidden_dim//8) = (b, hidden_dim//2)
        condition_embed = torch.cat([rot_embed, grip_embed], dim=-1)  # (b, hidden_dim//2)

        # 投影到完整的hidden_dim: (b, hidden_dim//2) -> (b, hidden_dim)
        condition_embed = self.condition_proj(condition_embed)  # (b, hidden_dim)

        # 将条件添加到每个时间步
        condition_embed = condition_embed.unsqueeze(1).expand(-1, num_future_frames, -1)  # (b, t_future, hidden_dim)
        conditioned_features = upsampled_features + condition_embed

        # 6. Transformer时间建模
        temporal_features = self.transformer(conditioned_features)  # (b, num_future_frames, hidden_dim)

        # 7. 预测
        rotation_logits = self.rotation_head(temporal_features)  # (b, num_future_frames, num_bins*3)
        gripper_logits = self.gripper_head(temporal_features)  # (b, num_future_frames, 2)

        return rotation_logits, gripper_logits


class VAEFeatureExtractor:
    """
    VAE特征提取器 - 用于提取RGB和Heatmap的VAE特征
    参考 WanVideoUnit_InputVideoEmbedder 的实现方式
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
        编码RGB和Heatmap视频
        参考 WanVideoUnit_InputVideoEmbedder.process 的实现

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


def collate_fn_with_vae(batch, vae_extractor, heatmap_latent_scale=1.0, latent_noise_std=0.0):
    """
    自定义collate函数，将数据转换为VAE特征

    Args:
        latent_noise_std: 添加到latent的高斯噪声标准差（用于训练时数据增强）
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

    # 编码为VAE特征
    rgb_features, heatmap_features = vae_extractor.encode_videos(
        input_video_rgb, input_video_heatmap
    )

    # 应用heatmap缩放
    if heatmap_latent_scale != 1.0:
        heatmap_features = heatmap_features * heatmap_latent_scale

    # 添加噪声增强 - 模拟扩散模型降噪后的latent
    # 这可以提升模型对推理时noisy latents的鲁棒性
    if latent_noise_std > 0:
        # 为RGB latent添加高斯噪声
        rgb_noise = torch.randn_like(rgb_features) * latent_noise_std
        rgb_features = rgb_features + rgb_noise

        # 为Heatmap latent添加高斯噪声
        heatmap_noise = torch.randn_like(heatmap_features) * latent_noise_std
        heatmap_features = heatmap_features + heatmap_noise

    # 添加batch维度
    rgb_features = rgb_features.unsqueeze(0)  # (1, v, c, t_compressed, h, w)
    heatmap_features = heatmap_features.unsqueeze(0)

    # 处理初始旋转和夹爪状态（第0帧的状态）
    # start_rotation和start_gripper是从数据集中直接获取的第0帧的真实状态
    initial_rotation = start_rotation.unsqueeze(0)  # (1, 3)
    initial_gripper = start_gripper.unsqueeze(0)  # (1,)

    return {
        'rgb_features': rgb_features,
        'heatmap_features': heatmap_features,
        'initial_rotation': initial_rotation,
        'initial_gripper': initial_gripper,
        'rotation_targets': rotation_targets,  # (1, t-1, 3)
        'gripper_targets': gripper_targets,  # (1, t-1)
        'num_future_frames': rotation_targets.shape[1],  # t-1
    }


def train_epoch(
    model: MultiViewRotationGripperPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    epoch_id: int,
    args,
):
    """训练一个epoch"""
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

            # 前向传播
            rotation_logits, gripper_logits = model(
                rgb_features=batch['rgb_features'],
                heatmap_features=batch['heatmap_features'],
                initial_rotation=batch['initial_rotation'],
                initial_gripper=batch['initial_gripper'],
                num_future_frames=batch['num_future_frames'],
            )

            # 计算loss
            rotation_targets = batch['rotation_targets']  # (b, t-1, 3)
            gripper_targets = batch['gripper_targets']  # (b, t-1)

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

    # 加载VAE
    print("Loading VAE...")
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
    model_manager.load_model(f"{args.model_base_path}/Wan2.2_VAE.pth")
    vae = model_manager.fetch_model("wan_video_vae")
    vae_extractor = VAEFeatureExtractor(vae, device="cuda", torch_dtype=torch.bfloat16)
    print("✓ VAE loaded")

    # 创建数据集
    print("Creating dataset...")

    # 处理data_root - 支持单任务或多任务
    data_roots = args.data_root if isinstance(args.data_root, list) else [args.data_root]

    # 多任务训练: 为每个任务创建数据集
    if len(data_roots) > 1:
        print(f"Multi-task training mode: {len(data_roots)} tasks")
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
        dataset = ConcatDataset(datasets)
        print(f"✓ Multi-task dataset created: {len(dataset)} samples (from {len(data_roots)} tasks)")
    else:
        # 单任务训练
        print(f"Single-task training mode: {data_roots[0]}")
        dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
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
        print(f"✓ Dataset created: {len(dataset)} samples")

    # 创建DataLoader
    from functools import partial
    collate_fn = partial(
        collate_fn_with_vae,
        vae_extractor=vae_extractor,
        heatmap_latent_scale=args.heatmap_latent_scale,
        latent_noise_std=args.latent_noise_std  # 添加latent噪声增强
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 固定为1
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # Must be False since collate_fn returns CUDA tensors
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"✓ DataLoader created")

    # 创建模型
    print("Creating model...")
    model = MultiViewRotationGripperPredictor(
        rgb_channels=48,  # VAE latent channels for RGB
        heatmap_channels=48,  # VAE latent channels for Heatmap
        hidden_dim=args.hidden_dim,
        num_views=3, 
        num_rotation_bins=args.num_rotation_bins,
        temporal_upsample_factor=4,
        dropout=args.dropout,
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
