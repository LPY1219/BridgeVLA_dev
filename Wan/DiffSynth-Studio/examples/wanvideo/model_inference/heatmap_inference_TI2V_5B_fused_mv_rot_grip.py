"""
Heatmap Inference Script for Wan2.2 with Rotation and Gripper Prediction (Multi-View Version)
用于多视角热力图序列预测 + 旋转和夹爪预测的推断脚本
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import cv2
from pathlib import Path

# 自动检测根路径
def get_root_path():
    """自动检测BridgeVLA根目录"""
    possible_paths = [
        "/share/project/lpy/BridgeVLA",
        "/DATA/disk1/lpy/BridgeVLA_dev",
        "/home/lpy/BridgeVLA_dev"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")

ROOT_PATH = get_root_path()
print(f"Using ROOT_PATH: {ROOT_PATH}")

# 添加项目路径
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, diffsynth_path)

# 导入多视角pipeline (支持旋转和夹爪预测)
from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_rot_grip import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap
from examples.wanvideo.model_training.mv_rot_grip import MultiViewRotationGripperPredictor
# 导入旋转和夹爪预测模型
import torch.nn as nn
import torch.nn.functional as F


class MultiViewRotationGripperPredictor(nn.Module):
    """
    多视角旋转和夹爪状态预测器
    从VAE编码的RGB和Heatmap特征预测机器人的rotation和gripper状态

    与训练文件完全一致: examples/wanvideo/model_training/mv_rot_grip.py
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
        )

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
        debug: bool = False,
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
            first_frame_broadcast = first_frame_features.unsqueeze(-1)
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
        if debug:
            print(f"    [MODEL DEBUG] initial_rotation received in forward: {initial_rotation}")
            print(f"    [MODEL DEBUG] initial_rotation dtype: {initial_rotation.dtype}")
            print(f"    [MODEL DEBUG] initial_rotation min/max: {initial_rotation.min().item()}/{initial_rotation.max().item()}")

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


class HeatmapInferenceMVRotGrip:
    """多视角热力图 + 旋转和夹爪预测推断类"""

    def __init__(self,
                 lora_checkpoint_path: str,
                 rot_grip_checkpoint_path: str,
                 wan_type: str,
                 model_base_path: str = None,
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16,
                 use_dual_head: bool = False,
                 rotation_resolution: float = 5.0,
                 hidden_dim: int = 512,
                 num_rotation_bins: int = 72):
        """
        初始化多视角推断器 + 旋转和夹爪预测器

        Args:
            lora_checkpoint_path: LoRA模型检查点路径 (用于diffusion model)
            rot_grip_checkpoint_path: 旋转和夹爪预测器检查点路径
            wan_type: 模型类型（必须是多视角+旋转夹爪版本）
            model_base_path: 基础模型路径
            device: 设备
            torch_dtype: 张量类型
            use_dual_head: 是否使用双head模式
            rotation_resolution: 旋转角度分辨率（度）
            hidden_dim: 隐藏层维度
            num_rotation_bins: 旋转bins数量
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_checkpoint_path = lora_checkpoint_path
        self.rot_grip_checkpoint_path = rot_grip_checkpoint_path
        self.use_dual_head = use_dual_head
        self.rotation_resolution = rotation_resolution
        self.num_rotation_bins = num_rotation_bins

        print(f"Loading {wan_type} multi-view pipeline with rotation/gripper prediction...")

        # 加载diffusion pipeline
        if wan_type == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP":
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch_dtype,
                device=device,
                wan_type=wan_type,
                use_dual_head=use_dual_head,
                model_configs=[
                    ModelConfig(path=[
                        f"{model_base_path}/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00003-of-00003-bf16.safetensors"
                    ]),
                    ModelConfig(path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth"),
                    ModelConfig(path=f"{model_base_path}/Wan2.2_VAE.pth"),
                ],
            )
        else:
            raise ValueError(f"Unsupported wan_type: {wan_type}. Use '5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP'")

        # 初始化多视角模块
        print("Initializing multi-view modules...")
        self._initialize_mv_modules()

        # 加载diffusion模型的LoRA权重
        print(f"Loading diffusion LoRA checkpoint: {lora_checkpoint_path}")
        self.load_lora_with_base_weights(lora_checkpoint_path, alpha=1.0)

        # 初始化旋转和夹爪预测器
        print("Initializing rotation and gripper predictor...")
        self.rot_grip_predictor = MultiViewRotationGripperPredictor(
            rgb_channels=48,  # VAE latent channels
            heatmap_channels=48,
            hidden_dim=hidden_dim,
            num_views=3,
            num_rotation_bins=num_rotation_bins,
            temporal_upsample_factor=4,
            dropout=0.1,
        ).to(device=device, dtype=torch_dtype)

        # 加载旋转和夹爪预测器权重
        print(f"Loading rotation/gripper predictor checkpoint: {rot_grip_checkpoint_path}")
        self.load_rot_grip_checkpoint(rot_grip_checkpoint_path)

        print("Pipeline initialized successfully!")

    def _initialize_mv_modules(self):
        """初始化多视角模块"""
        from diffsynth.models.wan_video_dit_mv import SelfAttention

        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]

        for block in self.pipe.dit.blocks:
            block.projector = nn.Linear(dim, dim).to(device=self.device, dtype=self.torch_dtype)
            block.projector.weight = nn.Parameter(torch.zeros(dim, dim, device=self.device, dtype=self.torch_dtype))
            block.projector.bias = nn.Parameter(torch.zeros(dim, device=self.device, dtype=self.torch_dtype))
            block.norm_mvs = nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False).to(device=self.device, dtype=self.torch_dtype)
            block.modulation_mvs = nn.Parameter(torch.randn(1, 3, dim, device=self.device, dtype=self.torch_dtype) / dim**0.5)
            block.mvs_attn = SelfAttention(dim, block.self_attn.num_heads, block.self_attn.norm_q.eps).to(device=self.device, dtype=self.torch_dtype)
            block.modulation_mvs.data = block.modulation.data[:, :3, :].clone()
            block.mvs_attn.load_state_dict(block.self_attn.state_dict(), strict=True)

        print("✓ Multi-view modules initialized")

    def load_checkpoint_weights(self, checkpoint_path: str):
        """
        加载checkpoint中的所有训练权重（包括patch_embedding、head、MV模块的base layer等）

        注意：此函数现在不仅加载patch_embedding和head，还包括多视角注意力相关的base layer权重
        这些权重需要在应用LoRA之前加载，以确保LoRA变化量被正确地加到训练时的base layer上

        关键点：checkpoint中的多视角注意力权重以 `base_layer` 命名（如 blocks.0.mvs_attn.k.base_layer.weight），
        但模型中的参数名没有 `base_layer`（如 blocks.0.mvs_attn.k.weight），需要转换。

        Args:
            checkpoint_path: checkpoint文件路径
        """
        try:
            # 加载checkpoint
            print(f"  Loading state dict from: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)

            # 筛选需要的权重
            patch_embedding_weights = {}
            head_weights = {}
            mv_base_layer_weights = {}  # mvs_attn 的 base_layer 权重
            mv_other_weights = {}  # 其他 MV 相关权重（projector, norm_mvs, modulation_mvs）

            for key, value in state_dict.items():
                # 跳过LoRA相关的权重（但不跳过 base_layer）
                if 'lora' in key.lower():
                    continue

                # 筛选patch_embedding相关的权重
                if 'patch_embedding' in key or 'patch_embed' in key:
                    patch_embedding_weights[key] = value

                # 筛选head相关的权重（包括dual head）
                elif any(pattern in key for pattern in ['head']):
                    if 'attention' not in key.lower() and 'attn' not in key.lower():
                        head_weights[key] = value

                # 筛选MV模块的base_layer权重（需要转换键名）
                elif 'base_layer' in key:
                    # 这些是 mvs_attn 中经过 LoRA 训练的层的 base layer
                    mv_base_layer_weights[key] = value

                # 筛选MV模块的其他权重（projector, norm_mvs, modulation_mvs, mvs_attn中的norm等，不需要转换键名）
                elif any(pattern in key for pattern in ['projector', 'norm_mvs', 'modulation_mvs', 'mvs_attn']):
                    mv_other_weights[key] = value

            print(f"  Found {len(patch_embedding_weights)} patch_embedding weights")
            print(f"  Found {len(head_weights)} head weights")
            print(f"  Found {len(mv_base_layer_weights)} MV module base_layer weights (need key conversion)")
            print(f"  Found {len(mv_other_weights)} MV module other weights")

            # 显示找到的权重key样例
            if patch_embedding_weights:
                print("  Patch embedding keys (sample):")
                for key in list(patch_embedding_weights.keys())[:3]:
                    print(f"    - {key}")

            if head_weights:
                print("  Head keys (sample):")
                for key in list(head_weights.keys())[:3]:
                    print(f"    - {key}")

            if mv_base_layer_weights:
                print("  MV base_layer keys (sample, before conversion):")
                for key in list(mv_base_layer_weights.keys())[:5]:
                    print(f"    - {key}")

            if mv_other_weights:
                print("  MV other weights keys (sample):")
                for key in list(mv_other_weights.keys())[:5]:
                    print(f"    - {key}")

            # 合并要加载的权重
            weights_to_load = {}
            weights_to_load.update(patch_embedding_weights)
            weights_to_load.update(head_weights)
            weights_to_load.update(mv_other_weights)

            # 转换 base_layer 键名：blocks.X.mvs_attn.Y.base_layer.weight -> blocks.X.mvs_attn.Y.weight
            for key, value in mv_base_layer_weights.items():
                # 移除 .base_layer
                converted_key = key.replace('.base_layer.', '.')
                weights_to_load[converted_key] = value

            if not weights_to_load:
                print("  Warning: No weights found in checkpoint")
                return

            # 显示转换后的键名样例
            if mv_base_layer_weights:
                print("  MV base_layer keys (sample, after conversion):")
                converted_samples = [k.replace('.base_layer.', '.') for k in list(mv_base_layer_weights.keys())[:5]]
                for key in converted_samples:
                    print(f"    - {key}")

            # 清理权重key（移除前缀）
            weights_clean = {}
            for key, value in weights_to_load.items():
                # 移除可能的前缀: 'dit.', 'model.'
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value

            print(f"  Loading {len(weights_clean)} weights into DIT model...")

            # 加载到DIT模型中
            missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(
                weights_clean, strict=False
            )

            # 统计成功加载的权重
            loaded_keys = set(weights_clean.keys()) - set(unexpected_keys)

            print(f"    ✓ Successfully loaded {len(loaded_keys)}/{len(weights_clean)} weights")

            if missing_keys:
                relevant_missing = [k for k in missing_keys if any(p in k for p in ['patch_embedding', 'head', 'projector', 'mvs_'])]
                if relevant_missing:
                    print(f"    Warning: {len(relevant_missing)} relevant keys not found in checkpoint:")
                    for key in relevant_missing[:5]:
                        print(f"      - {key}")

            if unexpected_keys:
                print(f"    Info: {len(unexpected_keys)} unexpected keys (sample):")
                for key in unexpected_keys[:5]:
                    print(f"      - {key}")

            print("  ✓ All base layer weights loaded successfully!")

        except Exception as e:
            print(f"  Warning: Failed to load weights: {e}")
            print("  Continuing with LoRA weights only...")
            import traceback
            traceback.print_exc()

    def load_lora_with_base_weights(self, checkpoint_path: str, alpha: float = 1.0):
        """
        为多视角视频扩散模型定制的LoRA加载函数

        关键区别：多视角注意力部分的LoRA base layer已经存储在checkpoint中，
        因此需要先加载这些base layer，然后再应用LoRA变化量。

        加载流程：
        1. 先加载checkpoint中所有非LoRA的权重（base layer）
        2. 然后加载LoRA权重，计算变化量并应用到对应的base layer上

        这确保了LoRA变化量被正确地加到训练时的base layer上，
        而不是加到推理时随机初始化的权重上。

        Args:
            checkpoint_path: checkpoint文件路径
            alpha: LoRA的缩放因子
        """
        print("Loading checkpoint with custom LoRA logic for multiview model...")

        # 步骤1：先加载所有base layer权重（包括多视角注意力模块）
        print("\nStep 1: Loading base layer weights from checkpoint...")
        self.load_checkpoint_weights(checkpoint_path)

        # 步骤2：加载LoRA并应用到base layer上
        print("\nStep 2: Loading and applying LoRA weights...")
        self.pipe.load_lora(self.pipe.dit, checkpoint_path, alpha=alpha)

        print("\n✓ Checkpoint loaded successfully with multiview-aware LoRA logic!")

    def load_rot_grip_checkpoint(self, checkpoint_path: str):
        """加载旋转和夹爪预测器的checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Debug: 检查checkpoint信息
        print(f"  [DEBUG] Checkpoint keys: {checkpoint.keys()}")
        if 'num_rotation_bins' in checkpoint:
            print(f"  [DEBUG] Checkpoint num_rotation_bins: {checkpoint['num_rotation_bins']}")
            if checkpoint['num_rotation_bins'] != self.num_rotation_bins:
                print(f"  [WARNING] Checkpoint num_rotation_bins ({checkpoint['num_rotation_bins']}) != current num_rotation_bins ({self.num_rotation_bins})")
        if 'rotation_resolution' in checkpoint:
            print(f"  [DEBUG] Checkpoint rotation_resolution: {checkpoint['rotation_resolution']}")
            if checkpoint['rotation_resolution'] != self.rotation_resolution:
                print(f"  [WARNING] Checkpoint rotation_resolution ({checkpoint['rotation_resolution']}) != current rotation_resolution ({self.rotation_resolution})")
        if 'use_dual_head' in checkpoint:
            print(f"  [DEBUG] Checkpoint use_dual_head: {checkpoint['use_dual_head']}")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 加载权重
        self.rot_grip_predictor.load_state_dict(state_dict, strict=True)
        self.rot_grip_predictor.eval()

        print(f"✓ Loaded rotation/gripper predictor from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")

    def _visualize_input_images(self, input_image: List[Image.Image], input_image_rgb: List[Image.Image], prompt: str):
        """
        可视化输入的多视角图像

        Args:
            input_image: List[PIL.Image] - 多视角热力图 (num_views,)
            input_image_rgb: List[PIL.Image] - 多视角RGB图像 (num_views,)
            prompt: 文本提示
        """
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime

        num_views = len(input_image)

        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), "../../debug_input_visualization")
        os.makedirs(save_dir, exist_ok=True)

        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建图像网格：2行（heatmap和rgb）× num_views列
        fig, axes = plt.subplots(2, num_views, figsize=(num_views * 4, 8))

        # 确保axes是2D数组
        if num_views == 1:
            axes = axes.reshape(2, 1)

        # 绘制热力图
        for view_idx in range(num_views):
            ax = axes[0, view_idx]
            ax.imshow(input_image[view_idx])
            ax.set_title(f"Heatmap View {view_idx}", fontsize=12, fontweight='bold')
            ax.axis('off')

        # 绘制RGB图像
        for view_idx in range(num_views):
            ax = axes[1, view_idx]
            ax.imshow(input_image_rgb[view_idx])
            ax.set_title(f"RGB View {view_idx}", fontsize=12, fontweight='bold')
            ax.axis('off')

        # 添加总标题
        fig.suptitle(f"Input Images (Multi-View)\nPrompt: {prompt}",
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, f"input_multiview_{timestamp}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n{'='*80}")
        print(f"INPUT VISUALIZATION SAVED")
        print(f"{'='*80}")
        print(f"  Location: {save_path}")
        print(f"  Prompt: {prompt}")
        print(f"  Num Views: {num_views}")
        print(f"  Heatmap size: {input_image[0].size}")
        print(f"  RGB size: {input_image_rgb[0].size}")
        print(f"{'='*80}\n")

    def _visualize_generated_frames(self,
                                     video_heatmap_frames: List[List[Image.Image]],
                                     video_rgb_frames: List[List[Image.Image]],
                                     save_path: str = None):
        """
        可视化生成的视频帧

        Args:
            video_heatmap_frames: List[List[PIL.Image]] (num_views, T) - 生成的热力图视频帧
            video_rgb_frames: List[List[PIL.Image]] (num_views, T) - 生成的RGB视频帧
            save_path: 保存路径（可选）
        """
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime

        num_views = len(video_heatmap_frames)
        num_frames = len(video_heatmap_frames[0])

        # 创建保存目录
        if save_path is None:
            save_dir = os.path.join(os.path.dirname(__file__), "../../debug_generated_visualization")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"generated_frames_{timestamp}.png")
        else:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

        # 创建图像网格：(num_views * 2)行（每个视角heatmap和rgb）× num_frames列
        fig, axes = plt.subplots(num_views * 2, num_frames, figsize=(num_frames * 3, num_views * 2 * 3))

        # 确保axes是2D数组
        if num_views * 2 == 1 and num_frames == 1:
            axes = np.array([[axes]])
        elif num_views * 2 == 1:
            axes = axes.reshape(1, -1)
        elif num_frames == 1:
            axes = axes.reshape(-1, 1)

        # 绘制每个视角的帧
        for view_idx in range(num_views):
            # 热力图行
            heatmap_row = view_idx * 2
            for frame_idx in range(num_frames):
                ax = axes[heatmap_row, frame_idx]
                ax.imshow(video_heatmap_frames[view_idx][frame_idx])
                if frame_idx == 0:
                    ax.set_ylabel(f"View {view_idx}\nHeatmap", fontsize=10, fontweight='bold')
                ax.set_title(f"T={frame_idx}", fontsize=10)
                ax.axis('off')

            # RGB行
            rgb_row = view_idx * 2 + 1
            for frame_idx in range(num_frames):
                ax = axes[rgb_row, frame_idx]
                ax.imshow(video_rgb_frames[view_idx][frame_idx])
                if frame_idx == 0:
                    ax.set_ylabel(f"View {view_idx}\nRGB", fontsize=10, fontweight='bold')
                ax.axis('off')

        # 添加总标题
        fig.suptitle(f"Generated Video Frames\n({num_views} views × {num_frames} frames)",
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图像
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n{'='*80}")
        print(f"GENERATED FRAMES VISUALIZATION SAVED")
        print(f"{'='*80}")
        print(f"  Location: {save_path}")
        print(f"  Num Views: {num_views}")
        print(f"  Num Frames: {num_frames}")
        print(f"  Heatmap size: {video_heatmap_frames[0][0].size}")
        print(f"  RGB size: {video_rgb_frames[0][0].size}")
        print(f"{'='*80}\n")

    @torch.no_grad()
    def predict(self,
                prompt: str,
                input_image: List[Image.Image],  # 多视角起始图像 List[PIL.Image] (num_views,)
                input_image_rgb: List[Image.Image],  # 多视角起始RGB图像
                initial_rotation: np.ndarray,  # (3,) - [roll, pitch, yaw] in degrees
                initial_gripper: int,  # 0 or 1
                num_frames: int = 5,
                height: int = 256,
                width: int = 256,
                num_inference_steps: int = 50,
                cfg_scale: float = 1.0,
                seed: int = 0,
                visualize: bool = False,
                visualize_save_path: str ="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img/debug.png" ,
                **kwargs) -> Dict[str, Any]:
        """
        执行推理，生成视频序列并预测旋转和夹爪状态

        Args:
            prompt: 文本提示
            input_image: 多视角热力图起始图像 [view0, view1, view2]
            input_image_rgb: 多视角RGB起始图像 [view0, view1, view2]
            initial_rotation: 初始旋转角度 [roll, pitch, yaw] (度)
            initial_gripper: 初始夹爪状态 (0=close, 1=open)
            num_frames: 生成帧数 (包括初始帧)
            height, width: 图像尺寸
            num_inference_steps: 推理步数
            cfg_scale: CFG引导强度
            seed: 随机种子
            visualize: 是否可视化生成的视频帧
            visualize_save_path: 可视化图像保存路径（可选）

        Returns:
            字典包含:
                - video_frames: 生成的视频帧 List[List[PIL.Image]] (T, num_views)
                - video_rgb_frames: 生成的RGB视频帧 List[List[PIL.Image]] (T, num_views)
                - rotation_predictions: 旋转预测 (T-1, 3) - [roll, pitch, yaw] in degrees
                - gripper_predictions: 夹爪预测 (T-1,) - 0 or 1
                - rotation_logits: 旋转logits (T-1, num_bins, 3)
                - gripper_logits: 夹爪logits (T-1, 2)
        """
        # 0. 可视化输入图像（多视角）
        # self._visualize_input_images(input_image, input_image_rgb, prompt)

        # 1. 生成视频序列 (使用diffusion model)
        print("Generating video sequence with diffusion model...")

        output = self.pipe(
            prompt=prompt,
            input_image=input_image,
            input_image_rgb=input_image_rgb,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            use_dual_head=self.use_dual_head,
            **kwargs
        )

        # Pipeline现在返回字典，包含video和latents
        video_heatmap_frames = output['video_heatmap']  # List[List[PIL.Image]] (num_views,T)
        video_rgb_frames = output['video_rgb']  # List[List[PIL.Image]] (num_views,T)
        rgb_latents = output['rgb_latents']  # (num_views, c_rgb, t, h, w)
        heatmap_latents = output['heatmap_latents']  # (num_views, c_hm, t, h, w)

        # FIX: Pipeline内部已经除以了heatmap_latent_scale（wan_video_5B_TI2V_heatmap_and_rgb_mv_rot_grip.py:917）
        # 但训练时是乘以scale的，所以推理时需要乘回来以匹配训练分布
        HEATMAP_LATENT_SCALE = 1.0  # 应该与训练时的参数一致
        if HEATMAP_LATENT_SCALE != 1.0:
            heatmap_latents = heatmap_latents * HEATMAP_LATENT_SCALE
            print(f"  [INFO] Applied heatmap_latent_scale={HEATMAP_LATENT_SCALE} to denoised latents")

        # 2. 准备latents用于预测
        # 添加batch维度: (num_views, c, t, h, w) -> (1, num_views, c, t, h, w)
        rgb_latents = rgb_latents.unsqueeze(0)
        heatmap_latents = heatmap_latents.unsqueeze(0)


        # 3. 准备初始条件
        # 将初始rotation和gripper转换为离散化的索引
        initial_rotation_bins = self._discretize_rotation(initial_rotation)  # (3,)
        initial_rotation_tensor = torch.from_numpy(initial_rotation_bins).long().unsqueeze(0).to(self.device)  # (1, 3)
        initial_gripper_tensor = torch.tensor([initial_gripper], dtype=torch.long, device=self.device)  # (1,)

        # 4. 预测旋转和夹爪状态
        print("Predicting rotation and gripper states...")
        self.rot_grip_predictor.eval()

        num_future_frames = num_frames - 1
        rotation_logits, gripper_logits = self.rot_grip_predictor(
            rgb_features=rgb_latents,
            heatmap_features=heatmap_latents,
            initial_rotation=initial_rotation_tensor,
            initial_gripper=initial_gripper_tensor,
            num_future_frames=num_future_frames
        )

        # 5. 解码预测结果
        # FIX: rotation_logits需要reshape为(T-1, 3, num_bins)而不是(T-1, num_bins, 3)
        # 以匹配GT latent路径的处理（第1021行）
        # rotation_logits: (1, T-1, num_bins*3) -> (T-1, 3, num_bins)
        rotation_logits = rotation_logits[0].view(num_future_frames, 3, self.num_rotation_bins)

        # gripper_logits: (1, T-1, 2) -> (T-1, 2)
        gripper_logits = gripper_logits[0]

        # 获取预测的离散索引
        rotation_bins = rotation_logits.argmax(dim=-1)  # (T-1, 3) - 在最后一维（num_bins）上argmax
        gripper_states = gripper_logits.argmax(dim=1)  # (T-1,)

        # 转换回角度
        rotation_predictions = self._bins_to_degrees(rotation_bins.cpu().numpy())  # (T-1, 3)
        gripper_predictions = gripper_states.cpu().numpy()  # (T-1,)

        # 6. 可视化生成的视频帧
        if visualize:
            self._visualize_generated_frames(
                video_heatmap_frames,
                video_rgb_frames,
                save_path=visualize_save_path
            )

        return {
            'video_heatmap': video_heatmap_frames,
            'video_rgb': video_rgb_frames,
            'rotation_predictions': rotation_predictions,
            'gripper_predictions': gripper_predictions,
            'rotation_logits': rotation_logits.float().cpu().numpy(),
            'gripper_logits': gripper_logits.float().cpu().numpy(),
        }

    def _discretize_rotation(self, rotation_degrees: np.ndarray) -> np.ndarray:
        """
        将连续的旋转角度离散化为bins

        Args:
            rotation_degrees: (3,) - [roll, pitch, yaw] in degrees [-180, 180]

        Returns:
            rotation_bins: (3,) - bin indices
        """
        # 将范围从[-180, 180]转换为[0, 360]
        rotation_shifted = rotation_degrees + 180

        # 离散化
        rotation_bins = np.around(rotation_shifted / self.rotation_resolution).astype(np.int64)

        # 处理边界情况：360度 = 0度
        rotation_bins[rotation_bins == self.num_rotation_bins] = 0

        return rotation_bins

    def _degrees_to_bins(self, rotation_degrees: np.ndarray) -> np.ndarray:
        """
        将角度转换为离散的bins（与训练代码一致）

        Args:
            rotation_degrees: (..., 3) - [roll, pitch, yaw] in degrees [-180, 180]

        Returns:
            rotation_bins: (..., 3) - bin indices [0, num_rotation_bins)
        """
        # 转换到 [0, 360] 范围
        rotation_degrees_shifted = rotation_degrees + 180

        # 使用四舍五入转换为bins（与训练代码保持一致）
        rotation_bins = np.around(rotation_degrees_shifted / self.rotation_resolution).astype(np.int64)

        # 处理边界情况：360度 = 0度（与训练代码一致）
        rotation_bins[rotation_bins == self.num_rotation_bins] = 0

        # 确保在有效范围内
        rotation_bins = np.clip(rotation_bins, 0, self.num_rotation_bins - 1)

        return rotation_bins

    def _bins_to_degrees(self, rotation_bins: np.ndarray) -> np.ndarray:
        """
        将离散的bins转换回角度（返回bin的中心值）

        Args:
            rotation_bins: (T, 3) - bin indices

        Returns:
            rotation_degrees: (T, 3) - [roll, pitch, yaw] in degrees [-180, 180]
        """
        # bins代表的是中心值
        # bin 0 -> -180度, bin 1 -> -175度, ..., bin 71 -> 175度
        # 转换为角度 [0, 360]（bin的中心值）
        rotation_degrees = rotation_bins * self.rotation_resolution

        # 转换回[-180, 180]
        rotation_degrees = rotation_degrees - 180

        return rotation_degrees

    def find_peak_position(self, heatmap_image: Image.Image, colormap_name: str = 'jet') -> Tuple[int, int]:
        """
        在热力图中找到峰值位置

        Args:
            heatmap_image: 热力图图像 (PIL.Image)
            colormap_name: 使用的colormap名称

        Returns:
            peak_position: (x, y) 峰值位置
        """
        from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap
        # 将PIL Image转换为numpy数组并归一化到[0,1]
        heatmap_image_np = np.array(heatmap_image).astype(np.float32) / 255.0
        heatmap_array = extract_heatmap_from_colormap(heatmap_image_np, colormap_name)
        max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
        return (max_pos[1], max_pos[0])  # (x, y) format

    def calculate_peak_distance(self, pred_peak: Tuple[int, int], gt_peak: Tuple[int, int]) -> float:
        """
        计算两个峰值之间的欧氏距离

        Args:
            pred_peak: 预测的峰值位置 (x, y)
            gt_peak: ground truth峰值位置 (x, y)

        Returns:
            distance: 欧氏距离 (像素)
        """
        return np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    def find_peaks_batch(self, heatmap_images: List[List[Image.Image]], colormap_name: str = 'jet') -> List[List[Tuple[int, int]]]:
        """
        批量计算多个热力图的峰值位置（优化速度）

        Args:
            heatmap_images: List[List[PIL.Image]] (T, num_views) - 热力图图像
            colormap_name: 使用的colormap名称

        Returns:
            peaks: List[List[Tuple[int, int]]] (T, num_views) - 峰值位置列表
        """
        from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap

        num_frames = len(heatmap_images)
        num_views = len(heatmap_images[0])

        peaks = []
        for frame_idx in range(num_frames):
            frame_peaks = []
            for view_idx in range(num_views):
                heatmap_image = heatmap_images[frame_idx][view_idx]
                # 将PIL Image转换为numpy数组并归一化到[0,1]
                heatmap_image_np = np.array(heatmap_image).astype(np.float32) / 255.0
                heatmap_array = extract_heatmap_from_colormap(heatmap_image_np, colormap_name)
                max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
                peak = (max_pos[1], max_pos[0])  # (x, y) format
                frame_peaks.append(peak)
            peaks.append(frame_peaks)

        return peaks

    def preprocess_image(self, image, min_value=-1, max_value=1):
        """将 PIL.Image 转换为 torch.Tensor"""
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=self.torch_dtype, device=self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        # pattern: "B C H W"
        image = image.permute(2, 0, 1).unsqueeze(0)  # H W C -> 1 C H W
        return image

    def preprocess_video(self, video, min_value=-1, max_value=1):
        """
        将 list of PIL.Image 转换为 torch.Tensor
        参考训练代码中的实现

        Args:
            video: List[PIL.Image] - 视频帧列表

        Returns:
            torch.Tensor - shape (1, C, T, H, W)
        """
        video_tensors = [self.preprocess_image(image, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video_tensors, dim=0)  # (T, 1, C, H, W)
        video = video.squeeze(1)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        video = video.unsqueeze(0)  # (1, C, T, H, W)
        return video

    @torch.no_grad()
    def encode_gt_videos(self, rgb_videos, heatmap_videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """
        编码GT RGB和Heatmap视频为VAE latents
        参考训练代码中的VAEFeatureExtractor.encode_videos

        Args:
            rgb_videos: List[List[PIL.Image]] - [time][view] RGB视频
            heatmap_videos: List[List[PIL.Image]] - [time][view] Heatmap视频
            tiled: 是否使用tiled编码
            tile_size: tile大小
            tile_stride: tile步长

        Returns:
            rgb_features: (num_views, c, t_compressed, h, w) - RGB VAE latents
            heatmap_features: (num_views, c, t_compressed, h, w) - Heatmap VAE latents
        """
        num_frames = len(rgb_videos)
        num_views = len(rgb_videos[0])

        # 获取VAE encoder
        vae = self.pipe.vae

        # 按视角分组处理 - RGB
        all_rgb_view_latents = []
        for view_idx in range(num_views):
            # 提取当前视角的所有RGB帧
            view_rgb_frames = [rgb_videos[t][view_idx] for t in range(num_frames)]
            # 预处理为tensor: (1, C, T, H, W)
            view_rgb_video = self.preprocess_video(view_rgb_frames)
            # Remove batch dimension: (1, C, T, H, W) -> (C, T, H, W)
            view_rgb_video = view_rgb_video.squeeze(0)
            # VAE编码: (C, T, H, W) -> (c_latent, t_compressed, h_latent, w_latent)
            view_rgb_latents = vae.encode(
                [view_rgb_video],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            view_rgb_latents = view_rgb_latents[0].to(dtype=self.torch_dtype, device=self.device)
            all_rgb_view_latents.append(view_rgb_latents)

        # 合并所有视角的RGB latents
        rgb_features = torch.stack(all_rgb_view_latents, dim=0)  # (num_views, c, t, h, w)

        # 按视角分组处理 - Heatmap
        all_heatmap_view_latents = []
        for view_idx in range(num_views):
            # 提取当前视角的所有Heatmap帧
            view_heatmap_frames = [heatmap_videos[t][view_idx] for t in range(num_frames)]
            # 预处理为tensor: (1, C, T, H, W)
            view_heatmap_video = self.preprocess_video(view_heatmap_frames)
            # Remove batch dimension
            view_heatmap_video = view_heatmap_video.squeeze(0)
            # VAE编码
            view_heatmap_latents = vae.encode(
                [view_heatmap_video],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            view_heatmap_latents = view_heatmap_latents[0].to(dtype=self.torch_dtype, device=self.device)
            all_heatmap_view_latents.append(view_heatmap_latents)

        # 合并所有视角的Heatmap latents
        heatmap_features = torch.stack(all_heatmap_view_latents, dim=0)  # (num_views, c, t, h, w)

        return rgb_features, heatmap_features

    @torch.no_grad()
    def predict_from_gt_latents(self,
                                 gt_rgb_video: List[List[Image.Image]],
                                 gt_heatmap_video: List[List[Image.Image]],
                                 initial_rotation: np.ndarray,
                                 initial_gripper: int,
                                 heatmap_latent_scale: float = 1.0) -> Dict[str, np.ndarray]:
        """
        从GT视频编码的latents预测旋转和夹爪（用于验证预测器本身是否正常）

        Args:
            gt_rgb_video: List[List[PIL.Image]] - [time][view] GT RGB视频
            gt_heatmap_video: List[List[PIL.Image]] - [time][view] GT Heatmap视频
            initial_rotation: (3,) - 初始旋转角度 [roll, pitch, yaw] in degrees
            initial_gripper: int - 初始夹爪状态
            heatmap_latent_scale: float - heatmap latent缩放因子

        Returns:
            Dict包含:
                - rotation_predictions: (T-1, 3) 旋转预测（度）
                - gripper_predictions: (T-1,) 夹爪预测
        """
        # 1. 编码GT视频为VAE latents
        print("  [GT Latent] Encoding GT videos to VAE latents...")
        rgb_features, heatmap_features = self.encode_gt_videos(gt_rgb_video, gt_heatmap_video)

        # Debug: 打印VAE特征信息
        print(f"  [DEBUG] RGB features shape (before batch): {rgb_features.shape}")
        print(f"  [DEBUG] RGB features stats: mean={rgb_features.mean().item():.4f}, std={rgb_features.std().item():.4f}")
        print(f"  [DEBUG] Heatmap features shape (before batch): {heatmap_features.shape}")
        print(f"  [DEBUG] Heatmap features stats: mean={heatmap_features.mean().item():.4f}, std={heatmap_features.std().item():.4f}")
        print(f"  [DEBUG] heatmap_latent_scale: {heatmap_latent_scale}")

        # 应用heatmap缩放
        if heatmap_latent_scale != 1.0:
            heatmap_features = heatmap_features * heatmap_latent_scale
            print(f"  [DEBUG] After scaling - Heatmap features stats: mean={heatmap_features.mean().item():.4f}, std={heatmap_features.std().item():.4f}")

        # 添加batch维度
        rgb_features = rgb_features.unsqueeze(0)  # (1, v, c, t_compressed, h, w)
        heatmap_features = heatmap_features.unsqueeze(0)

        print(f"  [DEBUG] RGB features shape (after batch): {rgb_features.shape}")
        print(f"  [DEBUG] Heatmap features shape (after batch): {heatmap_features.shape}")

        # 2. 将初始状态转换为bins
        initial_rotation_bins = self._degrees_to_bins(initial_rotation)
        print(f"  [DEBUG] Initial rotation degrees: {initial_rotation}")
        print(f"  [DEBUG] Initial rotation bins: {initial_rotation_bins}")

        # Debug: 检查bins是否在有效范围内
        if np.any(initial_rotation_bins >= self.num_rotation_bins) or np.any(initial_rotation_bins < 0):
            print(f"  [ERROR] Initial rotation bins out of range! bins: {initial_rotation_bins}, valid range: [0, {self.num_rotation_bins-1}]")

        initial_rotation_tensor = torch.from_numpy(initial_rotation_bins).to(
            dtype=torch.long, device=self.device
        ).unsqueeze(0)  # (1, 3)

        initial_gripper_tensor = torch.tensor(
            [initial_gripper], dtype=torch.long, device=self.device
        )  # (1,)

        # Debug: 检查embedding layer的维度
        print(f"  [DEBUG] Rotation embedding layer num_embeddings: {self.rot_grip_predictor.initial_rotation_encoder.num_embeddings}")
        print(f"  [DEBUG] Rotation embedding layer embedding_dim: {self.rot_grip_predictor.initial_rotation_encoder.embedding_dim}")

        # 3. 计算future frames数量
        num_frames = len(gt_rgb_video)
        num_future_frames = num_frames - 1

        # 4. 使用旋转预测器预测
        print("  [GT Latent] Predicting rotation/gripper from GT latents...")
        print(f"  [DEBUG] num_future_frames: {num_future_frames}")
        print("  [DEBUG] Input to predictor:")
        print(f"    - rgb_features: {rgb_features.shape}")
        print(f"    - heatmap_features: {heatmap_features.shape}")
        print(f"    - initial_rotation: {initial_rotation_tensor}")
        print(f"    - initial_gripper: {initial_gripper_tensor}")

        rotation_logits, gripper_logits = self.rot_grip_predictor(
            rgb_features=rgb_features,
            heatmap_features=heatmap_features,
            initial_rotation=initial_rotation_tensor,
            initial_gripper=initial_gripper_tensor,
            num_future_frames=num_future_frames,
            debug=True
        )

        print("  [DEBUG] Output from predictor:")
        print(f"    - rotation_logits shape: {rotation_logits.shape}")
        print(f"    - gripper_logits shape: {gripper_logits.shape}")

        # 5. 转换logits为预测结果
        # Rotation: (1, T-1, num_bins*3) -> (T-1, 3) bins
        rotation_logits = rotation_logits.squeeze(0)  # (T-1, num_bins*3)
        rotation_logits = rotation_logits.view(num_future_frames, 3, self.num_rotation_bins)  # (T-1, 3, num_bins)

        # Debug: 检查yaw（index=2）的logits分布
        yaw_logits_frame0 = rotation_logits[0, 2, :]  # (num_bins,)
        yaw_probs = torch.softmax(yaw_logits_frame0, dim=0)
        top5_probs, top5_indices = torch.topk(yaw_probs, k=5)
        print("  [DEBUG] Frame 0 Yaw logits - Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            deg = idx.item() * self.rotation_resolution - 180
            print(f"    {i+1}. Bin {idx.item()}: {deg:.1f}° (prob={prob.item():.4f})")

        rotation_bins = rotation_logits.argmax(dim=-1)  # (T-1, 3)

        # Gripper: (1, T-1, 2) -> (T-1,)
        gripper_logits = gripper_logits.squeeze(0)  # (T-1, 2)
        gripper_preds = gripper_logits.argmax(dim=-1)  # (T-1,)

        # 6. 转换为numpy并转换bins到degrees
        rotation_bins_np = rotation_bins.float().cpu().numpy()
        print(f"  [DEBUG] Predicted rotation bins from GT latent:\n{rotation_bins_np}")
        rotation_degrees = self._bins_to_degrees(rotation_bins_np)
        gripper_preds_np = gripper_preds.cpu().numpy()

        return {
            'rotation_predictions': rotation_degrees,
            'gripper_predictions': gripper_preds_np,
        }


def convert_colormap_to_heatmap(colormap_images: List[List[Image.Image]], colormap_name: str = 'jet', resolution: int = 64) -> List[List[np.ndarray]]:
    """
    将colormap格式的图像转换为heatmap数组

    Args:
        colormap_images: List[List[PIL.Image]] (T, num_views) - colormap格式的图像
        colormap_name: 使用的colormap名称，默认'jet'
        resolution: LUT分辨率，默认64（推荐）
                   32: 快速但精度较低 (~2像素误差)
                   64: 平衡速度和精度 (~0.5像素误差，推荐)
                   128: 高精度但构建LUT较慢 (<0.3像素误差)

    Returns:
        heatmap_arrays: List[List[np.ndarray]] (T, num_views) - heatmap数组，每个元素shape为(H, W)
    """
    num_frames = len(colormap_images)
    num_views = len(colormap_images[0]) if num_frames > 0 else 0

    heatmap_arrays = []
    for frame_idx in range(num_frames):
        frame_heatmaps = []
        for view_idx in range(num_views):
            # 将PIL Image转换为numpy数组并归一化到[0,1]
            colormap_image_np = np.array(colormap_images[frame_idx][view_idx]).astype(np.float32) / 255.0
            # 从colormap提取heatmap
            heatmap_array = extract_heatmap_from_colormap(colormap_image_np, colormap_name, resolution=resolution)
            frame_heatmaps.append(heatmap_array)
        heatmap_arrays.append(frame_heatmaps)

    return heatmap_arrays


def visualize_heatmaps_with_peaks(colormap_images: List[List[Image.Image]],
                                    heatmap_arrays: List[List[np.ndarray]],
                                    save_dir: str = '/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img'):
    """
    可视化colormap图像和heatmap数组，并标注峰值位置

    Args:
        colormap_images: List[List[PIL.Image]] (T, num_views) - colormap格式的图像
        heatmap_arrays: List[List[np.ndarray]] (T, num_views) - heatmap数组
        save_dir: 保存可视化结果的目录
    """
    import os
    from pathlib import Path

    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_frames = len(colormap_images)
    num_views = len(colormap_images[0]) if num_frames > 0 else 0

    print(f"\n[Visualization] 开始可视化热力图，共 {num_frames} 帧，{num_views} 个视角")
    print(f"[Visualization] 保存到: {save_dir}")

    for frame_idx in range(num_frames):
        # 创建子图：每个视角3行（原始colormap + 标注峰值的colormap + heatmap array）
        fig, axes = plt.subplots(3, num_views, figsize=(5*num_views, 15))

        # 如果只有一个视角，确保axes是2D数组
        if num_views == 1:
            axes = axes.reshape(3, 1)

        for view_idx in range(num_views):
            # 获取当前视角的colormap和heatmap
            colormap_img = colormap_images[frame_idx][view_idx]
            heatmap_arr = heatmap_arrays[frame_idx][view_idx]

            # 找到峰值位置
            peak_value = np.max(heatmap_arr)
            peak_pos = np.unravel_index(np.argmax(heatmap_arr), heatmap_arr.shape)
            peak_y, peak_x = peak_pos

            # 第一行：显示原始colormap图像（不标注峰值）
            axes[0, view_idx].imshow(colormap_img)
            axes[0, view_idx].set_title(f'Frame {frame_idx}, View {view_idx}\nColormap (Original)')
            axes[0, view_idx].axis('off')

            # 第二行：显示colormap图像并标注峰值
            axes[1, view_idx].imshow(colormap_img)
            axes[1, view_idx].set_title(f'Colormap (with Peak)')
            axes[1, view_idx].axis('off')

            # 在colormap上标注峰值
            axes[1, view_idx].plot(peak_x, peak_y, 'r*', markersize=20,
                                   markeredgecolor='white', markeredgewidth=2)
            axes[1, view_idx].text(peak_x, peak_y - 10, f'Peak: ({peak_x}, {peak_y})\nValue: {peak_value:.3f}',
                                   color='white', fontsize=10, ha='center',
                                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

            # 第三行：显示heatmap数组
            im = axes[2, view_idx].imshow(heatmap_arr, cmap='jet', interpolation='nearest')
            axes[2, view_idx].set_title(f'Heatmap Array\nShape: {heatmap_arr.shape}')
            axes[2, view_idx].axis('off')

            # 在heatmap上标注峰值
            axes[2, view_idx].plot(peak_x, peak_y, 'r*', markersize=20,
                                   markeredgecolor='white', markeredgewidth=2)
            axes[2, view_idx].text(peak_x, peak_y - 10, f'Peak: ({peak_x}, {peak_y})\nValue: {peak_value:.3f}',
                                   color='white', fontsize=10, ha='center',
                                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

            # 添加colorbar
            plt.colorbar(im, ax=axes[2, view_idx], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, f'heatmap_visualization_frame_{frame_idx:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] 已保存: {save_path}")

        plt.close(fig)

    print(f"[Visualization] 可视化完成！共保存 {num_frames} 张图片\n")


def get_3d_position_from_pred_heatmap(pred_heatmap_colormap: List[List[Image.Image]],
                                       rev_trans: Any,
                                       projection_interface: Any,
                                       colormap_name: str = 'jet') -> np.ndarray:
    """
    从预测的heatmap colormap中获取3D位置预测

    Args:
        pred_heatmap_colormap: List[List[PIL.Image]] - 预测的colormap格式热力图
                               可以是 [T][num_views] 或 [num_views][T] 格式，函数会自动检测并转换
        rev_trans: 逆变换矩阵，用于从像素坐标转换到3D坐标
        projection_interface: 投影接口对象，包含get_position_from_heatmap方法
        colormap_name: colormap名称，默认'jet'

    Returns:
        pred_position: np.ndarray (T, 3) - 预测的3D位置 [x, y, z]
    """
    import time
    t0 = time.time()

    # 步骤0: 检测并转换输入格式 [num_views][T] -> [T][num_views]
    # 通过检查第一个维度的长度来判断格式（假设视角数 <= 3，时间步数 > 3）
    if len(pred_heatmap_colormap[0]) > 3:
        # 格式是 [num_views][T]，需要转换为 [T][num_views]
        num_views = len(pred_heatmap_colormap)
        num_frames = len(pred_heatmap_colormap[0])
        pred_heatmap_colormap = [[pred_heatmap_colormap[v][t] for v in range(num_views)] for t in range(num_frames)]

    t1 = time.time()
    print(f"[Position extraction] Step 0 (format conversion): {(t1-t0)*1000:.1f}ms")

    # 步骤1: 将colormap转换为heatmap数组 List[List[np.ndarray]] (T, num_views, H, W)  # visualize input and output
    pred_heatmap_arrays = convert_colormap_to_heatmap(pred_heatmap_colormap, colormap_name)# 速度很慢
    t2 = time.time()
    print(f"[Position extraction] Step 1 (colormap→heatmap): {(t2-t1)*1000:.1f}ms")

    # 可视化 pred_heatmap_colormap 和 pred_heatmap_arrays 并标注峰值
    # visualize_heatmaps_with_peaks(pred_heatmap_colormap, pred_heatmap_arrays)

    # 步骤2: 将 List[List[np.ndarray]] 转换为张量 (T, num_views, H*W)
    # get_position_from_heatmap 期望输入shape为 (bs, nc, h*w)
    num_frames = len(pred_heatmap_arrays)
    num_views = len(pred_heatmap_arrays[0])
    h, w = pred_heatmap_arrays[0][0].shape

    # 构建张量
    heatmap_tensor_list = []
    for frame_idx in range(num_frames):
        frame_views = []
        for view_idx in range(num_views):
            heatmap = pred_heatmap_arrays[frame_idx][view_idx]  # (H, W)
            heatmap_flat = heatmap.flatten()  # (H*W,)
            frame_views.append(heatmap_flat)
        heatmap_tensor_list.append(frame_views)

    # 转换为 numpy array 然后转为 torch tensor: (T, num_views, H*W)
    heatmap_np = np.array(heatmap_tensor_list)  # (T, num_views, H*W)
    heatmap_tensor = torch.from_numpy(heatmap_np).float()

    t3 = time.time()
    print(f"[Position extraction] Step 2 (tensor conversion): {(t3-t2)*1000:.1f}ms")

    # 步骤3: 从heatmap中提取3D位置
    get_position_from_heatmap = projection_interface.get_position_from_heatmap
    pred_position = get_position_from_heatmap(heatmap_tensor, rev_trans)

    t4 = time.time()
    print(f"[Position extraction] Step 3 (extract 3D position): {(t4-t3)*1000:.1f}ms")
    print(f"[Position extraction] Total time: {(t4-t0)*1000:.1f}ms")

    return pred_position




def get_3d_position_from_raw_heatmap(heatmap_raw: torch.Tensor,
                                      rev_trans: Any,
                                      projection_interface: Any,
                                      debug_info: dict = None) -> np.ndarray:
    """
    直接从原始heatmap tensor中获取3D位置预测（绕过colormap转换）

    Args:
        heatmap_raw: torch.Tensor (T, num_views, H, W) - 原始热力图tensor
        rev_trans: 逆变换矩阵，用于从像素坐标转换到3D坐标
        projection_interface: 投影接口对象，包含get_position_from_heatmap方法
        debug_info: 可选的调试信息字典

    Returns:
        pred_position: np.ndarray (T, 3) - 预测的3D位置 [x, y, z]
    """
    # 步骤1: 将 (T, num_views, H, W) 转换为 (T, num_views, H*W)
    T, num_views, H, W = heatmap_raw.shape
    heatmap_tensor = heatmap_raw.reshape(T, num_views, H * W).float()

    # 步骤2: 从heatmap中提取3D位置
    get_position_from_heatmap = projection_interface.get_position_from_heatmap
    pred_position = get_position_from_heatmap(heatmap_tensor, rev_trans)

    return pred_position


def visualize_predictions_with_rot_grip(
    gt_heatmap_video: List[List[Image.Image]],
    pred_heatmap_video: List[List[Image.Image]],
    gt_rgb_video: List[List[Image.Image]],
    pred_rgb_video: List[List[Image.Image]],
    gt_rotation: np.ndarray,
    pred_rotation: np.ndarray,
    gt_gripper: np.ndarray,
    pred_gripper: np.ndarray,
    initial_rotation: np.ndarray,
    initial_gripper: int,
    prompt: str,
    dataset_idx: int,
    save_path: str,
    heatmap_distances: Dict[str, List[List[float]]] = None,  # {'distances': (T, num_views), 'gt_peaks': (T, num_views, 2), 'pred_peaks': (T, num_views, 2)}
    colormap_name: str = 'jet'
):
    """
    可视化多视角预测结果，包含rotation、gripper和heatmap peak信息

    Args:
        gt_heatmap_video: List[List[PIL.Image]] (T, num_views) - Ground truth heatmaps
        pred_heatmap_video: List[List[PIL.Image]] (T, num_views) - Predicted heatmaps
        gt_rgb_video: List[List[PIL.Image]] (T, num_views) - Ground truth RGB
        pred_rgb_video: List[List[PIL.Image]] (T, num_views) - Predicted RGB
        gt_rotation: (T-1, 3) - Ground truth rotation [roll, pitch, yaw] degrees
        pred_rotation: (T-1, 3) - Predicted rotation [roll, pitch, yaw] degrees
        gt_gripper: (T-1,) - Ground truth gripper states
        pred_gripper: (T-1,) - Predicted gripper states
        initial_rotation: (3,) - Initial rotation
        initial_gripper: int - Initial gripper state
        prompt: str - Text prompt
        dataset_idx: int - Dataset index
        save_path: str - Path to save visualization
        heatmap_distances: Dict with 'distances', 'gt_peaks', 'pred_peaks' - Optional heatmap peak info
        colormap_name: str - Colormap name for peak extraction
    """
    num_frames = len(gt_heatmap_video)
    num_views = len(gt_heatmap_video[0])

    # 新布局：时间维度沿横轴，每列是一个时间步
    # 总共3个section（每个section有num_view行）：
    # 1. GT RGB (num_view 行)
    # 2. Pred RGB (num_view 行)
    # 3. Heatmap (num_view 行) - 使用GT heatmap，同时显示GT和pred峰值
    # 横轴：num_frames 列

    fig = plt.figure(figsize=(3*num_frames, 3*num_views*3 + 2))  # +2 for rotation/gripper info
    gs = fig.add_gridspec(num_views*3, num_frames, hspace=0.3, wspace=0.1)

    for view_idx in range(num_views):
        # 每个section的起始行
        gt_rgb_row = view_idx
        pred_rgb_row = num_views + view_idx
        heatmap_row = num_views*2 + view_idx

        for frame_idx in range(num_frames):
            # 获取当前帧当前视角的图像
            gt_heatmap_frame = gt_heatmap_video[frame_idx][view_idx]
            gt_rgb_frame = gt_rgb_video[frame_idx][view_idx]
            pred_rgb_frame = pred_rgb_video[frame_idx][view_idx]

            # 获取peak位置（如果提供）
            gt_peak = None
            pred_peak = None
            peak_dist = None
            if heatmap_distances is not None:
                gt_peak = heatmap_distances['gt_peaks'][frame_idx][view_idx]
                pred_peak = heatmap_distances['pred_peaks'][frame_idx][view_idx]
                peak_dist = heatmap_distances['distances'][frame_idx][view_idx]

            # 第1部分：GT RGB
            ax = fig.add_subplot(gs[gt_rgb_row, frame_idx])
            ax.imshow(gt_rgb_frame)
            if gt_peak is not None:
                ax.plot(gt_peak[0], gt_peak[1], 'r*', markersize=8, markeredgecolor='white', markeredgewidth=0.5)
            if frame_idx == 0:
                ax.set_ylabel(f'GT RGB V{view_idx}', fontsize=9, fontweight='bold')
            if view_idx == 0:
                ax.set_title(f'T{frame_idx}', fontsize=8)
            ax.axis('off')

            # 第2部分：Pred RGB
            ax = fig.add_subplot(gs[pred_rgb_row, frame_idx])
            ax.imshow(pred_rgb_frame)
            if pred_peak is not None:
                ax.plot(pred_peak[0], pred_peak[1], 'b*', markersize=8, markeredgecolor='white', markeredgewidth=0.5)
            if frame_idx == 0:
                ax.set_ylabel(f'Pred RGB V{view_idx}', fontsize=9, fontweight='bold')
            ax.axis('off')

            # 第3部分：Heatmap (使用GT heatmap，同时显示GT和pred峰值)
            ax = fig.add_subplot(gs[heatmap_row, frame_idx])
            ax.imshow(gt_heatmap_frame)
            if gt_peak is not None and pred_peak is not None:
                # 红星 = GT, 蓝星 = pred
                ax.plot(gt_peak[0], gt_peak[1], 'r*', markersize=10, markeredgecolor='white', markeredgewidth=0.8, label='GT')
                ax.plot(pred_peak[0], pred_peak[1], 'b*', markersize=10, markeredgecolor='white', markeredgewidth=0.8, label='Pred')
            # 在heatmap下方显示距离
            if peak_dist is not None and view_idx == 0:  # 只在第一个view显示距离
                ax.text(0.5, -0.05, f'Dist: {peak_dist:.1f}px', ha='center', va='top',
                       transform=ax.transAxes, fontsize=6, color='blue')
            if frame_idx == 0:
                ax.set_ylabel(f'Heatmap V{view_idx}', fontsize=9, fontweight='bold')

            # 在heatmap下方显示rotation和gripper信息
            if view_idx == num_views - 1:  # 只在最后一个view显示
                if frame_idx == 0:
                    # 初始状态
                    info_text = f'Init:\nR:{initial_rotation[0]:.0f},{initial_rotation[1]:.0f},{initial_rotation[2]:.0f}\nG:{"O" if initial_gripper==1 else "C"}'
                else:
                    # 预测和真实值
                    idx = frame_idx - 1
                    gt_r = gt_rotation[idx]
                    pred_r = pred_rotation[idx]
                    gt_g = gt_gripper[idx]
                    pred_g = pred_gripper[idx]

                    # 计算rotation误差
                    rot_err = np.abs(pred_r - gt_r)
                    rot_err = np.minimum(rot_err, 360 - rot_err)

                    info_text = f'GT: {gt_r[0]:.0f},{gt_r[1]:.0f},{gt_r[2]:.0f} {"O" if gt_g==1 else "C"}\n'
                    info_text += f'Pred: {pred_r[0]:.0f},{pred_r[1]:.0f},{pred_r[2]:.0f} {"O" if pred_g==1 else "C"}\n'
                    info_text += f'Err: {rot_err[0]:.1f},{rot_err[1]:.1f},{rot_err[2]:.1f}'

                ax.text(0.5, -0.15, info_text, ha='center', va='top',
                       transform=ax.transAxes, fontsize=6, family='monospace')

            ax.axis('off')

    # 计算总体统计
    rotation_errors = np.abs(pred_rotation - gt_rotation)
    rotation_errors = np.minimum(rotation_errors, 360 - rotation_errors)
    mean_rotation_error = rotation_errors.mean(axis=0)
    gripper_accuracy = (pred_gripper == gt_gripper).sum() / len(pred_gripper) * 100

    # 添加总标题
    title = f'Multi-View Sample (Index {dataset_idx})\n{prompt[:80]}...\n'
    title += f'Rotation Error (deg): R={mean_rotation_error[0]:.1f}, P={mean_rotation_error[1]:.1f}, Y={mean_rotation_error[2]:.1f} | '
    title += f'Gripper Acc: {gripper_accuracy:.1f}%'
    if heatmap_distances is not None:
        mean_heatmap_dist = np.mean(heatmap_distances['distances'])
        title += f' | Heatmap Dist: {mean_heatmap_dist:.1f}px'

    fig.suptitle(title, fontsize=10, fontweight='bold')

    # 保存结果
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {save_path}")


def test_on_dataset_mv_rot_grip(inference_engine: HeatmapInferenceMVRotGrip,
                                  data_root: str,
                                  output_dir: str,
                                  test_indices: List[int],
                                  num_frames: int = 5,
                                  num_inference_steps: int = 50,
                                  cfg_scale: float = 1.0,
                                  sequence_length: int = 4,
                                  image_size: Tuple[int, int] = (256, 256),
                                  scene_bounds: List[float] = None,
                                  transform_augmentation_xyz: List[float] = None,
                                  transform_augmentation_rpy: List[float] = None,
                                  wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP"):
    """
    在数据集上测试推理 + 旋转和夹爪预测

    Args:
        inference_engine: 推理引擎
        data_root: 数据集根目录
        output_dir: 输出目录
        test_indices: 测试样本索引列表
        num_frames: 生成帧数
        num_inference_steps: 推理步数
        cfg_scale: CFG scale
        sequence_length: 序列长度
        image_size: 图像尺寸
    """
    from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip import HeatmapDatasetFactory
    from scipy.spatial.transform import Rotation as R

    os.makedirs(output_dir, exist_ok=True)

    # 设置默认值
    if scene_bounds is None:
        scene_bounds = [0, -0.45, -0.05, 0.8, 0.55, 0.6]
    if transform_augmentation_xyz is None:
        transform_augmentation_xyz = [0.0, 0.0, 0.0]
    if transform_augmentation_rpy is None:
        transform_augmentation_rpy = [0.0, 0.0, 0.0]

    print(f"\n=== Testing on Dataset (Multi-View + Rotation/Gripper) ===")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Test indices: {test_indices}")
    print(f"WAN type: {wan_type}")
    print(f"Scene bounds: {scene_bounds}")

    # 创建数据集
    dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
        data_root=data_root,
        sequence_length=sequence_length,
        step_interval=1,
        min_trail_length=10,
        image_size=image_size,
        sigma=1.5,
        augmentation=False,  # 测试时不需要增强
        mode="test",
        scene_bounds=scene_bounds,
        transform_augmentation_xyz=transform_augmentation_xyz,
        transform_augmentation_rpy=transform_augmentation_rpy,
        debug=False,
        colormap_name="jet",
        repeat=1,
        wan_type=wan_type,
        rotation_resolution=inference_engine.rotation_resolution,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    all_rotation_errors = []  # 存储所有旋转误差
    all_gripper_accuracies = []  # 存储所有夹爪准确率
    all_heatmap_distances = []  # 存储所有heatmap峰值距离
    all_position_errors = []  # 存储所有位置误差 (x, y, z)

    for idx, dataset_idx in enumerate(test_indices):
        if dataset_idx >= len(dataset):
            print(f"Warning: Index {dataset_idx} out of range (dataset size: {len(dataset)}), skipping...")
            continue

        print(f"\n[{idx+1}/{len(test_indices)}] Processing dataset index {dataset_idx}...")

        try:
            # 获取数据
            sample = dataset[dataset_idx]

            prompt = sample['prompt']
            input_image = sample['input_image']  # List[PIL.Image] (num_views,)
            input_image_rgb = sample['input_image_rgb']  # List[PIL.Image] (num_views,)

            # 获取ground truth
            gt_rotation_targets = sample['rotation_targets']  # (T-1, 3) - bins
            gt_gripper_targets = sample['gripper_targets']  # (T-1,) - 0 or 1
            start_rotation = sample['start_rotation']  # (3,) - bins
            start_gripper = sample['start_gripper']  # scalar - 0 or 1

            # 获取ground truth position（注意：存储的是cube坐标，需要转换为world坐标）
            start_pose = sample['start_pose']  # (7,) - [x_cube, y_cube, z_cube, qx, qy, qz, qw]
            future_poses = sample['future_poses']  # (T, 7) - [x_cube, y_cube, z_cube, qx, qy, qz, qw]
            rev_trans = sample['rev_trans']  # 从cube坐标转换到world坐标的函数

            # 提取位置信息（cube坐标）
            start_position_cube = start_pose[:3]  # (3,) - [x_cube, y_cube, z_cube]
            gt_positions_cube = future_poses[:, :3]  # (T, 3) - [x_cube, y_cube, z_cube]
            # 拼接起始位置
            gt_position_cube_trajectory = torch.cat([start_position_cube.unsqueeze(0), gt_positions_cube], dim=0)  # (T+1, 3)

            # 转换为world坐标
            gt_position_trajectory = rev_trans(gt_position_cube_trajectory).numpy()  # (T+1, 3) - world coordinates

            # 转换bins到degrees
            start_rotation_degrees = inference_engine._bins_to_degrees(start_rotation.numpy())
            gt_rotation_degrees = inference_engine._bins_to_degrees(gt_rotation_targets.numpy())

            print(f"  Prompt: {prompt}")
            print(f"  Initial rotation (degrees): Roll={start_rotation_degrees[0]:.1f}, Pitch={start_rotation_degrees[1]:.1f}, Yaw={start_rotation_degrees[2]:.1f}")
            print(f"  Initial gripper: {start_gripper.item()}")

            # 执行推理
            # 使用 sequence_length 作为 num_frames 以匹配数据集
            output = inference_engine.predict(
                prompt=prompt,
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                initial_rotation=start_rotation_degrees,
                initial_gripper=start_gripper.item(),
                num_frames=sequence_length+1, # 需要包括初始帧进来
                height=image_size[0],
                width=image_size[1],
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                seed=dataset_idx
            )

            pred_rotation_degrees = output['rotation_predictions']  # (T-1, 3)
            pred_gripper = output['gripper_predictions']  # (T-1,)

            # 计算误差
            # 旋转误差：计算每个角度的绝对误差
            rotation_errors = np.abs(pred_rotation_degrees - gt_rotation_degrees)  # (T-1, 3)
            # 处理角度wrap-around（例如：-179° vs 179°）
            rotation_errors = np.minimum(rotation_errors, 360 - rotation_errors)
            mean_rotation_error = rotation_errors.mean(axis=0)  # (3,) - [roll, pitch, yaw]
            overall_rotation_error = rotation_errors.mean()

            # 夹爪准确率
            gripper_correct = (pred_gripper == gt_gripper_targets.numpy()).sum()
            gripper_accuracy = gripper_correct / len(pred_gripper) * 100

            all_rotation_errors.append(rotation_errors)
            all_gripper_accuracies.append(gripper_accuracy)

            print(f"  Rotation Error (degrees): Roll={mean_rotation_error[0]:.2f}, Pitch={mean_rotation_error[1]:.2f}, Yaw={mean_rotation_error[2]:.2f}")
            print(f"  Overall Rotation Error: {overall_rotation_error:.2f}°")
            print(f"  Gripper Accuracy: {gripper_accuracy:.1f}%")

            # 获取GT视频
            gt_heatmap_video = sample['video']  # List[List[PIL.Image]] (T, num_views)
            gt_rgb_video = sample['input_video_rgb']  # List[List[PIL.Image]] (T, num_views)

            # 检查并转换pred视频格式
            # Pipeline可能返回 [view][time]，需要转换为 [time][view]
            pred_heatmap = output['video_heatmap']
            pred_rgb = output['video_rgb']

            # 从预测的heatmap提取3D位置
            pred_position = get_3d_position_from_pred_heatmap(
                pred_heatmap_colormap=pred_heatmap,
                rev_trans=sample["rev_trans"],
                projection_interface=dataset.robot_dataset_config['projection_interface'],
                colormap_name='jet'
            )  # (num_frames, 3)

            # 计算位置误差
            position_errors = np.abs(pred_position - gt_position_trajectory)
            mean_position_error = position_errors.mean(axis=0)
            overall_position_error = position_errors.mean()
            position_distance_error = np.linalg.norm(position_errors, axis=1).mean()

            all_position_errors.append(position_errors)

            print(f"\n  Position Error (X/Y/Z): {mean_position_error[0]:.4f}m / {mean_position_error[1]:.4f}m / {mean_position_error[2]:.4f}m")
            print(f"  Position Distance Error: {position_distance_error:.4f}m ({position_distance_error*1000:.1f}mm)")

            # 检测格式：如果第一个元素的长度大于视角数，说明是 [view][time]
            if len(pred_heatmap[0]) > 3:  # 假设视角数<=3
                # 转换 [view][time] -> [time][view]
                num_views = len(pred_heatmap)
                num_frames = len(pred_heatmap[0])
                pred_heatmap_video = [[pred_heatmap[v][t] for v in range(num_views)] for t in range(num_frames)]
                pred_rgb_video = [[pred_rgb[v][t] for v in range(num_views)] for t in range(num_frames)]
            else:
                pred_heatmap_video = pred_heatmap
                pred_rgb_video = pred_rgb
                num_views = len(pred_heatmap_video[0])
                num_frames = len(pred_heatmap_video)

            # 计算heatmap峰值距离（批量处理加速）
            print("  Calculating heatmap peak distances...")
            colormap_name = 'jet'  # 使用与训练相同的colormap

            # 批量计算所有峰值位置
            gt_peaks = inference_engine.find_peaks_batch(gt_heatmap_video, colormap_name)
            pred_peaks = inference_engine.find_peaks_batch(pred_heatmap_video, colormap_name)

            # 计算距离
            heatmap_peak_info = {
                'distances': [],
                'gt_peaks': gt_peaks,
                'pred_peaks': pred_peaks
            }

            for frame_idx in range(num_frames):
                frame_distances = []
                for view_idx in range(num_views):
                    gt_peak = gt_peaks[frame_idx][view_idx]
                    pred_peak = pred_peaks[frame_idx][view_idx]
                    distance = inference_engine.calculate_peak_distance(pred_peak, gt_peak)
                    frame_distances.append(distance)
                heatmap_peak_info['distances'].append(frame_distances)

            # 转换为numpy数组
            heatmap_peak_info['distances'] = np.array(heatmap_peak_info['distances'])  # (T, num_views)
            mean_heatmap_distance = heatmap_peak_info['distances'].mean()

            # 添加到总体统计
            all_heatmap_distances.append(heatmap_peak_info['distances'].flatten())

            print(f"  Mean Heatmap Peak Distance: {mean_heatmap_distance:.2f}px")

            # ====== 实验：使用GT latent预测旋转和夹爪（验证预测器本身） ======
            print("\n  [Experiment] Predicting from GT latents for validation...")
            gt_latent_output = inference_engine.predict_from_gt_latents(
                gt_rgb_video=gt_rgb_video,
                gt_heatmap_video=gt_heatmap_video,
                initial_rotation=start_rotation_degrees,
                initial_gripper=start_gripper.item(),
                heatmap_latent_scale=1.0
            )

            gt_latent_rotation = gt_latent_output['rotation_predictions']
            gt_latent_gripper = gt_latent_output['gripper_predictions']

            # 添加调试信息
            print(f"  [DEBUG] GT rotation shape: {gt_rotation_degrees.shape}, values:\n{gt_rotation_degrees}")
            print(f"  [DEBUG] GT latent pred rotation shape: {gt_latent_rotation.shape}, values:\n{gt_latent_rotation}")

            # 计算GT latent预测的误差
            gt_latent_rotation_errors = np.abs(gt_latent_rotation - gt_rotation_degrees)
            gt_latent_rotation_errors = np.minimum(gt_latent_rotation_errors, 360 - gt_latent_rotation_errors)
            gt_latent_mean_rotation_error = gt_latent_rotation_errors.mean(axis=0)
            gt_latent_overall_rotation_error = gt_latent_rotation_errors.mean()

            gt_latent_gripper_correct = (gt_latent_gripper == gt_gripper_targets.numpy()).sum()
            gt_latent_gripper_accuracy = gt_latent_gripper_correct / len(gt_latent_gripper) * 100

            print(f"  [GT Latent] Rotation Error (degrees): Roll={gt_latent_mean_rotation_error[0]:.2f}, Pitch={gt_latent_mean_rotation_error[1]:.2f}, Yaw={gt_latent_mean_rotation_error[2]:.2f}")
            print(f"  [GT Latent] Overall Rotation Error: {gt_latent_overall_rotation_error:.2f}°")
            print(f"  [GT Latent] Gripper Accuracy: {gt_latent_gripper_accuracy:.1f}%")
            print()

            # 保存可视化（类似参考文件的格式）
            vis_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_multiview_comparison.png')
            visualize_predictions_with_rot_grip(
                gt_heatmap_video=gt_heatmap_video,
                pred_heatmap_video=pred_heatmap_video,
                gt_rgb_video=gt_rgb_video,
                pred_rgb_video=pred_rgb_video,
                gt_rotation=gt_rotation_degrees,
                pred_rotation=pred_rotation_degrees,
                gt_gripper=gt_gripper_targets.numpy(),
                pred_gripper=pred_gripper,
                initial_rotation=start_rotation_degrees,
                initial_gripper=start_gripper.item(),
                prompt=prompt,
                dataset_idx=dataset_idx,
                save_path=vis_path,
                heatmap_distances=heatmap_peak_info,
                colormap_name=colormap_name
            )

            # 保存详细统计信息（类似参考文件的格式）
            stats_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_stats.txt')
            num_views = len(gt_heatmap_video[0])
            num_frames = len(gt_heatmap_video)

            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"Multi-View Sample with Rotation/Gripper Prediction (Dataset Index {dataset_idx})\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Num Views: {num_views}, Num Frames: {num_frames}\n")
                f.write("=" * 80 + "\n\n")

                # 初始条件
                f.write(f"Initial Conditions (Frame 0):\n")
                f.write(f"  Rotation (degrees): Roll={start_rotation_degrees[0]:.1f}, Pitch={start_rotation_degrees[1]:.1f}, Yaw={start_rotation_degrees[2]:.1f}\n")
                f.write(f"  Gripper: {'Open' if start_gripper.item() == 1 else 'Close'}\n")
                f.write("\n" + "-" * 80 + "\n\n")

                # 按帧组织统计
                f.write("Frame-by-Frame Results:\n")
                f.write("-" * 80 + "\n")
                for frame_idx in range(num_frames):  # 包含所有帧（包括初始帧0）
                    f.write(f"\nFrame {frame_idx}:\n")

                    # Position信息（所有帧都有位置）
                    gt_pos = gt_position_trajectory[frame_idx]
                    pred_pos = pred_position[frame_idx]
                    pos_err = position_errors[frame_idx]
                    pos_dist = np.linalg.norm(pos_err)
                    f.write("  Position (meters):\n")
                    f.write(f"    GT:   X={gt_pos[0]:7.4f}, Y={gt_pos[1]:7.4f}, Z={gt_pos[2]:7.4f}\n")
                    f.write(f"    Pred: X={pred_pos[0]:7.4f}, Y={pred_pos[1]:7.4f}, Z={pred_pos[2]:7.4f} | Err: {pos_err[0]:.4f}, {pos_err[1]:.4f}, {pos_err[2]:.4f} | Dist: {pos_dist:.4f}m\n")

                    # 如果是初始帧，跳过rotation和gripper（因为只有未来帧才有这些预测）
                    if frame_idx == 0:
                        continue

                    # Rotation信息 - 包含降噪latent和GT latent预测
                    pred_idx = frame_idx - 1
                    gt_rot = gt_rotation_degrees[pred_idx]
                    pred_rot = pred_rotation_degrees[pred_idx]  # 从降噪latent预测
                    gt_latent_rot = gt_latent_rotation[pred_idx]  # 从GT latent预测
                    rot_err = rotation_errors[pred_idx]
                    gt_latent_rot_err = gt_latent_rotation_errors[pred_idx]

                    f.write("  Rotation:\n")
                    f.write(f"    GT:        Roll={gt_rot[0]:7.1f}°, Pitch={gt_rot[1]:7.1f}°, Yaw={gt_rot[2]:7.1f}°\n")
                    f.write(f"    Pred(Denoised): Roll={pred_rot[0]:7.1f}°, Pitch={pred_rot[1]:7.1f}°, Yaw={pred_rot[2]:7.1f}° | Err: {rot_err[0]:5.1f}°, {rot_err[1]:5.1f}°, {rot_err[2]:5.1f}°\n")
                    f.write(f"    Pred(GT-Latent): Roll={gt_latent_rot[0]:7.1f}°, Pitch={gt_latent_rot[1]:7.1f}°, Yaw={gt_latent_rot[2]:7.1f}° | Err: {gt_latent_rot_err[0]:5.1f}°, {gt_latent_rot_err[1]:5.1f}°, {gt_latent_rot_err[2]:5.1f}°\n")

                    # Gripper信息 - 包含降噪latent和GT latent预测
                    gt_grip = gt_gripper_targets[pred_idx].item()
                    pred_grip = pred_gripper[pred_idx]  # 从降噪latent预测
                    gt_latent_grip = gt_latent_gripper[pred_idx]  # 从GT latent预测
                    grip_match = "✓" if gt_grip == pred_grip else "✗"
                    gt_latent_grip_match = "✓" if gt_grip == gt_latent_grip else "✗"
                    f.write("  Gripper:\n")
                    f.write(f"    GT: {'Open ' if gt_grip == 1 else 'Close'}\n")
                    f.write(f"    Pred(Denoised): {'Open ' if pred_grip == 1 else 'Close'} {grip_match}\n")
                    f.write(f"    Pred(GT-Latent): {'Open ' if gt_latent_grip == 1 else 'Close'} {gt_latent_grip_match}\n")

                    # Heatmap Peak信息（所有视角）
                    f.write("  Heatmap Peaks:\n")
                    for view_idx in range(num_views):
                        gt_peak = heatmap_peak_info['gt_peaks'][frame_idx][view_idx]
                        pred_peak = heatmap_peak_info['pred_peaks'][frame_idx][view_idx]
                        peak_dist = heatmap_peak_info['distances'][frame_idx][view_idx]
                        f.write(f"    View {view_idx}: GT=({gt_peak[0]:3.0f},{gt_peak[1]:3.0f}), Pred=({pred_peak[0]:3.0f},{pred_peak[1]:3.0f}), Dist={peak_dist:5.1f}px\n")

                # 总结统计
                f.write("\n" + "=" * 80 + "\n")
                f.write("Summary Statistics:\n")
                f.write("-" * 80 + "\n")

                # Position统计
                f.write("Average Position Error (meters):\n")
                f.write(f"  X:       {mean_position_error[0]:.4f}m\n")
                f.write(f"  Y:       {mean_position_error[1]:.4f}m\n")
                f.write(f"  Z:       {mean_position_error[2]:.4f}m\n")
                f.write(f"  Overall: {overall_position_error:.4f}m\n")
                f.write(f"  Mean Euclidean Distance: {position_distance_error:.4f}m\n\n")

                # Rotation统计 - 对比两种预测方式
                f.write("Average Rotation Error (degrees):\n")
                f.write("  From Denoised Latent:\n")
                f.write(f"    Roll:    {mean_rotation_error[0]:.2f}°\n")
                f.write(f"    Pitch:   {mean_rotation_error[1]:.2f}°\n")
                f.write(f"    Yaw:     {mean_rotation_error[2]:.2f}°\n")
                f.write(f"    Overall: {overall_rotation_error:.2f}°\n")
                f.write("  From GT Latent (Validation):\n")
                f.write(f"    Roll:    {gt_latent_mean_rotation_error[0]:.2f}°\n")
                f.write(f"    Pitch:   {gt_latent_mean_rotation_error[1]:.2f}°\n")
                f.write(f"    Yaw:     {gt_latent_mean_rotation_error[2]:.2f}°\n")
                f.write(f"    Overall: {gt_latent_overall_rotation_error:.2f}°\n")

                # Gripper统计 - 对比两种预测方式
                f.write("\nGripper Accuracy:\n")
                f.write(f"  From Denoised Latent: {gripper_accuracy:.1f}% ({gripper_correct}/{len(pred_gripper)})\n")
                f.write(f"  From GT Latent (Validation): {gt_latent_gripper_accuracy:.1f}% ({gt_latent_gripper_correct}/{len(gt_latent_gripper)})\n")

                # Heatmap Peak统计
                f.write("\nHeatmap Peak Distance (pixels):\n")
                f.write(f"  Mean (all views): {mean_heatmap_distance:.2f}px\n")
                # 按视角统计
                for view_idx in range(num_views):
                    view_distances = heatmap_peak_info['distances'][:, view_idx]
                    mean_view_dist = view_distances.mean()
                    f.write(f"  View {view_idx}: {mean_view_dist:.2f}px\n")

                # 添加说明
                f.write("\n" + "-" * 80 + "\n")
                f.write("Note: 'Denoised Latent' = Prediction from diffusion model output\n")
                f.write("      'GT Latent' = Prediction from VAE-encoded ground truth (validation)\n")

            print(f"  Results saved to: {vis_path}")

        except Exception as e:
            print(f"  Error processing sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算总体统计
    if all_rotation_errors:
        all_rotation_errors = np.concatenate(all_rotation_errors, axis=0)  # (N_total, 3)
        mean_errors = all_rotation_errors.mean(axis=0)  # (3,)
        std_errors = all_rotation_errors.std(axis=0)  # (3,)
        overall_mean = all_rotation_errors.mean()
        overall_std = all_rotation_errors.std()

        mean_gripper_acc = np.mean(all_gripper_accuracies)
        std_gripper_acc = np.std(all_gripper_accuracies)

        # Heatmap distance statistics
        all_heatmap_distances = np.concatenate(all_heatmap_distances, axis=0)  # (N_total,)
        mean_heatmap_dist = all_heatmap_distances.mean()
        std_heatmap_dist = all_heatmap_distances.std()

        # Position error statistics
        all_position_errors = np.concatenate(all_position_errors, axis=0)  # (N_total, 3)
        mean_position_errors = all_position_errors.mean(axis=0)  # (3,) - [x, y, z]
        std_position_errors = all_position_errors.std(axis=0)  # (3,)
        overall_position_mean = all_position_errors.mean()
        overall_position_std = all_position_errors.std()
        # 计算欧氏距离误差
        position_distances = np.linalg.norm(all_position_errors, axis=1)  # (N_total,)
        mean_position_distance = position_distances.mean()
        std_position_distance = position_distances.std()

        print(f"\n=== OVERALL EVALUATION RESULTS ===")
        print(f"Total frames evaluated: {len(all_rotation_errors)}")
        print(f"Position Error (meters):")
        print(f"  X:       {mean_position_errors[0]:.4f} ± {std_position_errors[0]:.4f}m")
        print(f"  Y:       {mean_position_errors[1]:.4f} ± {std_position_errors[1]:.4f}m")
        print(f"  Z:       {mean_position_errors[2]:.4f} ± {std_position_errors[2]:.4f}m")
        print(f"  Overall: {overall_position_mean:.4f} ± {overall_position_std:.4f}m")
        print(f"  Mean Euclidean Distance: {mean_position_distance:.4f} ± {std_position_distance:.4f}m")
        print(f"Rotation Error (degrees):")
        print(f"  Roll:  {mean_errors[0]:.2f} ± {std_errors[0]:.2f}°")
        print(f"  Pitch: {mean_errors[1]:.2f} ± {std_errors[1]:.2f}°")
        print(f"  Yaw:   {mean_errors[2]:.2f} ± {std_errors[2]:.2f}°")
        print(f"  Overall: {overall_mean:.2f} ± {overall_std:.2f}°")
        print(f"Gripper Accuracy: {mean_gripper_acc:.1f} ± {std_gripper_acc:.1f}%")
        print(f"Heatmap Peak Distance: {mean_heatmap_dist:.2f} ± {std_heatmap_dist:.2f}px")

        # 保存总体统计
        stats_path = os.path.join(output_dir, 'evaluation_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Multi-View Position, Rotation, Gripper and Heatmap Evaluation Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total frames evaluated: {len(all_rotation_errors)}\n\n")
            f.write("Position Error (meters):\n")
            f.write(f"  X:       {mean_position_errors[0]:.4f} ± {std_position_errors[0]:.4f}m\n")
            f.write(f"  Y:       {mean_position_errors[1]:.4f} ± {std_position_errors[1]:.4f}m\n")
            f.write(f"  Z:       {mean_position_errors[2]:.4f} ± {std_position_errors[2]:.4f}m\n")
            f.write(f"  Overall: {overall_position_mean:.4f} ± {overall_position_std:.4f}m\n")
            f.write(f"  Mean Euclidean Distance: {mean_position_distance:.4f} ± {std_position_distance:.4f}m\n\n")
            f.write("Rotation Error (degrees):\n")
            f.write(f"  Roll:  {mean_errors[0]:.2f} ± {std_errors[0]:.2f}°\n")
            f.write(f"  Pitch: {mean_errors[1]:.2f} ± {std_errors[1]:.2f}°\n")
            f.write(f"  Yaw:   {mean_errors[2]:.2f} ± {std_errors[2]:.2f}°\n")
            f.write(f"  Overall: {overall_mean:.2f} ± {overall_std:.2f}°\n\n")
            f.write(f"Gripper Accuracy: {mean_gripper_acc:.1f} ± {std_gripper_acc:.1f}%\n\n")
            f.write("Heatmap Peak Distance (pixels):\n")
            f.write(f"  Mean: {mean_heatmap_dist:.2f} ± {std_heatmap_dist:.2f}px\n")

        print(f"\nResults saved to: {output_dir}")


# 示例使用
def main():
    """主函数"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Multi-View Rotation/Gripper Inference Script')

    # 模型配置
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="LoRA checkpoint path for diffusion model")
    parser.add_argument("--rot_grip_checkpoint", type=str, required=True, help="Rotation/gripper predictor checkpoint path")
    parser.add_argument("--model_base_path", type=str, default="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused")
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP")
    parser.add_argument("--use_dual_head", action='store_true', help='Use dual head mode (must match training configuration)')
    parser.add_argument("--device", type=str, default="cuda")

    # 旋转和夹爪预测器配置
    parser.add_argument("--rotation_resolution", type=float, default=5.0, help="Rotation resolution in degrees")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_rotation_bins", type=int, default=72, help="Number of rotation bins (360 / rotation_resolution)")

    # 数据集配置
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--scene_bounds", type=str, default="0,-0.45,-0.05,0.8,0.55,0.6",
                       help='Scene bounds as comma-separated values: x_min,y_min,z_min,x_max,y_max,z_max')
    parser.add_argument("--transform_augmentation_xyz", type=str, default="0.0,0.0,0.0",
                       help='Transform augmentation for xyz as comma-separated values (usually 0 for inference)')
    parser.add_argument("--transform_augmentation_rpy", type=str, default="0.0,0.0,0.0",
                       help='Transform augmentation for roll/pitch/yaw as comma-separated values (usually 0 for inference)')
    parser.add_argument("--sequence_length", type=int, default=4, help="Sequence length")
    parser.add_argument("--test_indices", type=str, default="100,200,300,400,500",
                       help='Comma-separated test indices')

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()

    # 解析逗号分隔的参数
    args.scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
    if len(args.scene_bounds) != 6:
        raise ValueError(f"scene_bounds must have 6 values, got {len(args.scene_bounds)}")
    args.transform_augmentation_xyz = [float(x.strip()) for x in args.transform_augmentation_xyz.split(',')]
    args.transform_augmentation_rpy = [float(x.strip()) for x in args.transform_augmentation_rpy.split(',')]

    print("=== Multi-View Rotation/Gripper Inference Test ===")
    print(f"LoRA Checkpoint: {args.lora_checkpoint}")
    print(f"Rotation/Gripper Checkpoint: {args.rot_grip_checkpoint}")
    print(f"Model Type: {args.wan_type}")
    print(f"Dual Head Mode: {args.use_dual_head}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {args.output_dir}")
    print()

    # 创建推理器
    inferencer = HeatmapInferenceMVRotGrip(
        lora_checkpoint_path=args.lora_checkpoint,
        rot_grip_checkpoint_path=args.rot_grip_checkpoint,
        wan_type=args.wan_type,
        model_base_path=args.model_base_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        use_dual_head=args.use_dual_head,
        rotation_resolution=args.rotation_resolution,
        hidden_dim=args.hidden_dim,
        num_rotation_bins=args.num_rotation_bins
    )

    # 解析测试索引
    test_indices = [int(x.strip()) for x in args.test_indices.split(',')]
    print(f"Test indices: {test_indices}")

    # 在数据集上测试
    print("\n=== Multi-View Dataset Test with Rotation/Gripper Prediction ===")
    test_on_dataset_mv_rot_grip(
        inference_engine=inferencer,
        data_root=args.data_root,
        output_dir=args.output_dir,
        test_indices=test_indices,
        wan_type=args.wan_type,
        sequence_length=args.sequence_length,
        scene_bounds=args.scene_bounds,
        transform_augmentation_xyz=args.transform_augmentation_xyz,
        transform_augmentation_rpy=args.transform_augmentation_rpy,
    )

    print("\n✓ Inference completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
