"""
Diffusion Action Decoder

This module implements an action decoder that predicts motion changes from DiT intermediate features.

Unlike the original VAE-based action decoder, this version:
- Takes DiT intermediate layer features as input (not VAE decoder features)
- Predicts heatmap peak position changes, rotation changes, and gripper changes
- All predictions are relative to the first frame (to avoid overfitting to initial state)

Architecture:
- Input: DiT intermediate features (b*v, seq_len, dim) from a specified block
- Processing: Multi-view fusion → Temporal modeling → Action prediction
- Output: Heatmap peak delta, rotation delta, gripper delta

Key Design Decisions:
1. No need for initial state as input (predict changes only)
2. Multi-view attention for spatial consistency
3. Transformer for temporal dependencies
4. Separate prediction heads for different action components

Usage:
    decoder = DiffusionActionDecoder(
        dit_feature_dim=3072,  # DiT hidden dimension
        num_views=3,
        num_rotation_bins=72,
        num_future_frames=23  # Predict 23 future frames
    )

    # Extract features from DiT
    dit_features = feature_extractor(rgb_latents, heatmap_latents, text_embeddings)

    # Predict actions
    heatmap_delta, rotation_logits, gripper_logits = decoder(
        dit_features['features'],
        shape_info=dit_features['shape_info'],
        num_views=dit_features['num_views']
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange


class DiffusionActionDecoder(nn.Module):
    """
    Action decoder that predicts motion changes from DiT intermediate features.

    This decoder takes features from a DiT intermediate block and predicts:
    1. Heatmap peak position changes (delta_h, delta_w for each future frame)
    2. Rotation changes (delta_roll, delta_pitch, delta_yaw as classification)
    3. Gripper state changes (whether gripper state changes from initial)

    Args:
        dit_feature_dim: DiT hidden dimension (e.g., 3072 for Wan-5B)
        hidden_dim: Hidden dimension for decoder layers
        num_views: Number of camera views
        num_rotation_bins: Number of bins for rotation classification
        num_future_frames: Number of future frames to predict
        dropout: Dropout rate
    """

    def __init__(
        self,
        dit_feature_dim: int = 3072,  # Wan-5B DiT dimension
        hidden_dim: int = 512,
        num_views: int = 3,
        num_rotation_bins: int = 72,  # 360 / 5 = 72 bins
        num_future_frames: int = 24,  # 
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dit_feature_dim = dit_feature_dim
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_rotation_bins = num_rotation_bins
        self.num_future_frames = num_future_frames

        # 1. Deep feature projection: DiT features -> hidden_dim (3 layers for better capacity)
        # DiT features are (b*v, seq_len, dit_feature_dim), project to hidden_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(dit_feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 2. Multi-view fusion
        # For each time frame, fuse information across views
        self.view_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.view_norm = nn.LayerNorm(hidden_dim)

        # 3. Learnable temporal upsampling (if needed)
        # DiT features may have compressed time dimension due to patchification
        # We need to upsample to match the number of output frames
        self.temporal_upsampler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 4. History-conditioned cross-attention
        # Future frames attend to history frame for better conditioning
        self.history_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.history_norm = nn.LayerNorm(hidden_dim)

        # 5. Temporal modeling with Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 6. Action prediction heads
        # 6.1 Multi-view heatmap peak delta prediction (predict for ALL views)
        self.heatmap_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_views * 2)  # (num_views, 2) for each frame
        )

        # 6.2 Rotation delta prediction (classification for each axis)
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rotation_bins * 3)  # roll, pitch, yaw
        )

        # 6.3 Gripper state change prediction (binary classification)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 0=no change, 1=change
        )

    def forward(
        self,
        dit_features: torch.Tensor,  # (b*v, seq_len, dit_feature_dim)
        shape_info: Tuple[int, int, int],  # (f, h, w) from DiT patchify
        num_views: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-view heatmap prediction and history conditioning.

        Args:
            dit_features: DiT intermediate features (b*v, seq_len, dit_feature_dim)
            shape_info: (f, h, w) tuple indicating temporal and spatial dimensions
            num_views: Number of views (default: 3)

        Returns:
            heatmap_delta: (b, num_future_frames, num_views, 2) - Multi-view heatmap peak position changes
            rotation_logits: (b, num_future_frames, num_rotation_bins*3) - Rotation changes
            gripper_logits: (b, num_future_frames, 2) - Gripper state changes
        """
        bv, seq_len, _ = dit_features.shape
        b = bv // num_views
        f, h, w = shape_info

        # 1. Project features to hidden dimension
        features = self.feature_projection(dit_features)  # (b*v, seq_len, hidden_dim)

        # 2. Reshape to separate batch, view, and spatial-temporal dimensions
        # seq_len = f * h * w, we need to reshape to (b, v, f, h*w, hidden_dim)
        # Then pool spatially to get (b, v, f, hidden_dim)

        features = rearrange(
            features,
            '(b v) (f h w) d -> b v f (h w) d',
            b=b, v=num_views, f=f, h=h, w=w
        )

        # Spatial pooling: average over h*w dimension
        features = features.mean(dim=3)  # (b, v, f, hidden_dim)

        # 3. Multi-view fusion for each time frame
        # Process each time frame separately
        fused_features = []
        for t_idx in range(f):
            # Get features from all views at time t
            view_features = features[:, :, t_idx, :]  # (b, v, hidden_dim)

            # Multi-head attention across views
            fused, _ = self.view_attention(
                view_features, view_features, view_features
            )  # (b, v, hidden_dim)

            # Average across views
            fused = fused.mean(dim=1)  # (b, hidden_dim)
            fused = self.view_norm(fused)

            fused_features.append(fused)

        fused_features = torch.stack(fused_features, dim=1)  # (b, f, hidden_dim)

        # 4. Learned temporal upsampling if needed
        # If f < num_future_frames + 1, we need to upsample
        # If f > num_future_frames + 1, we need to downsample
        # If f == num_future_frames + 1, perfect match

        if f != self.num_future_frames + 1:
            target_f = self.num_future_frames + 1  # +1 for the initial frame

            if f < target_f:
                # Upsample: interpolate first (expand time), then refine with learnable network
                fused_features = fused_features.permute(0, 2, 1)  # (b, hidden_dim, f)
                fused_features = F.interpolate(
                    fused_features,
                    size=target_f,
                    mode='linear',
                    align_corners=True
                )
                fused_features = fused_features.permute(0, 2, 1)  # (b, target_f, hidden_dim)
                # Refine interpolated features with learnable MLP
                fused_features = self.temporal_upsampler(fused_features)
            else:
                # Downsample: transform first (extract important features), then interpolate (compress time)
                fused_features = self.temporal_upsampler(fused_features)  # (b, f, hidden_dim)
                fused_features = fused_features.permute(0, 2, 1)  # (b, hidden_dim, f)
                fused_features = F.interpolate(
                    fused_features,
                    size=target_f,
                    mode='linear',
                    align_corners=True
                )
                fused_features = fused_features.permute(0, 2, 1)  # (b, target_f, hidden_dim)

        # 5. Separate history and future frames
        # First frame is the history/condition, rest are future
        history_features = fused_features[:, :1, :]  # (b, 1, hidden_dim)
        future_features = fused_features[:, 1:, :]  # (b, num_future_frames, hidden_dim)

        # 6. History-conditioned temporal modeling
        # Cross-attention: future frames (query) attend to history frame (key, value)
        future_conditioned, _ = self.history_cross_attention(
            query=future_features,
            key=history_features,
            value=history_features
        )  # (b, num_future_frames, hidden_dim)

        # Residual connection + norm
        future_features = future_features + future_conditioned
        future_features = self.history_norm(future_features)

        # 7. Temporal self-attention (process future frames)
        temporal_features = self.temporal_transformer(future_features)  # (b, num_future_frames, hidden_dim)

        # 8. Predict actions for future frames
        # 8.1 Multi-view heatmap delta
        heatmap_delta_flat = self.heatmap_delta_head(temporal_features)  # (b, num_future_frames, num_views*2)
        heatmap_delta = heatmap_delta_flat.view(b, self.num_future_frames, self.num_views, 2)  # (b, T, num_views, 2)

        # 8.2 Rotation logits
        rotation_logits = self.rotation_head(temporal_features)  # (b, num_future_frames, num_rotation_bins*3)

        # 8.3 Gripper logits
        gripper_logits = self.gripper_head(temporal_features)  # (b, num_future_frames, 2)

        return heatmap_delta, rotation_logits, gripper_logits


def compute_action_decoder_loss(
    pred_heatmap_delta: torch.Tensor,  # (b, num_future, num_views, 2)
    pred_rotation_logits: torch.Tensor,  # (b, num_future, num_bins*3)
    pred_gripper_logits: torch.Tensor,  # (b, num_future, 2)
    target_heatmap_delta: torch.Tensor,  # (b, num_future, num_views, 2)
    target_rotation_bins: torch.Tensor,  # (b, num_future, 3) - bin indices
    target_gripper_change: torch.Tensor,  # (b, num_future) - 0 or 1
    num_rotation_bins: int = 72,
    heatmap_loss_weight: float = 1.0,
    rotation_loss_weight: float = 1.0,
    gripper_loss_weight: float = 0.5,
    img_size: tuple = None,  # (width, height) 用于反归一化
    is_normalized: bool = True,  # 是否需要反归一化
) -> dict:
    """
    Compute the loss for action decoder training with multi-view heatmap prediction.

    Args:
        pred_heatmap_delta: Predicted heatmap peak delta (b, num_future, num_views, 2)
        pred_rotation_logits: Predicted rotation logits (b, num_future, num_bins*3)
        pred_gripper_logits: Predicted gripper change logits (b, num_future, 2)
        target_heatmap_delta: Target heatmap peak delta (b, num_future, num_views, 2)
        target_rotation_bins: Target rotation bins (b, num_future, 3)
        target_gripper_change: Target gripper change (b, num_future)
        num_rotation_bins: Number of rotation bins
        heatmap_loss_weight: Weight for heatmap loss
        rotation_loss_weight: Weight for rotation loss
        gripper_loss_weight: Weight for gripper loss

    Returns:
        Dictionary containing:
            - 'loss': Total weighted loss
            - 'heatmap_loss': Multi-view heatmap delta loss
            - 'rotation_loss': Rotation classification loss
            - 'gripper_loss': Gripper change loss
            - 'rotation_accuracy': Rotation prediction accuracy
            - 'gripper_accuracy': Gripper prediction accuracy
    """
    b, num_future, num_views, _ = pred_heatmap_delta.shape

    # 1. Heatmap delta loss (L2 loss)
    # IMPORTANT: 如果heatmap_delta是归一化的，先反归一化到像素尺度再计算loss
    # 这样loss仍然在像素尺度，不需要调整loss权重
    if is_normalized and img_size is not None:
        # 反归一化到像素尺度
        img_width, img_height = img_size
        scale_tensor = torch.tensor(
            [img_width, img_height],
            dtype=pred_heatmap_delta.dtype,
            device=pred_heatmap_delta.device
        )
        pred_heatmap_delta_pixels = pred_heatmap_delta * scale_tensor
        target_heatmap_delta_pixels = target_heatmap_delta * scale_tensor
        heatmap_loss = F.mse_loss(pred_heatmap_delta_pixels, target_heatmap_delta_pixels)
    else:
        # 直接计算loss（像素尺度或未归一化）
        heatmap_loss = F.mse_loss(pred_heatmap_delta, target_heatmap_delta)

    # 2. Rotation loss (cross-entropy for each axis)
    # Reshape rotation logits: (b, num_future, num_bins*3) -> (b, num_future, 3, num_bins)
    pred_rotation_logits = pred_rotation_logits.view(b, num_future, 3, num_rotation_bins)

    # Compute cross-entropy for each axis
    rotation_losses = []
    rotation_accs = []
    for axis in range(3):  # roll, pitch, yaw
        axis_logits = pred_rotation_logits[:, :, axis, :]  # (b, num_future, num_bins)
        axis_targets = target_rotation_bins[:, :, axis]  # (b, num_future)

        # Flatten for cross_entropy
        axis_logits_flat = axis_logits.reshape(-1, num_rotation_bins)  # (b*num_future, num_bins)
        axis_targets_flat = axis_targets.reshape(-1)  # (b*num_future,)

        axis_loss = F.cross_entropy(axis_logits_flat, axis_targets_flat)
        rotation_losses.append(axis_loss)

        # Compute accuracy
        axis_preds = axis_logits_flat.argmax(dim=1)
        axis_acc = (axis_preds == axis_targets_flat).float().mean()
        rotation_accs.append(axis_acc)

    rotation_loss = sum(rotation_losses) / 3
    rotation_accuracy = sum(rotation_accs) / 3

    # 3. Gripper loss (cross-entropy)
    gripper_logits_flat = pred_gripper_logits.reshape(-1, 2)  # (b*num_future, 2)
    gripper_targets_flat = target_gripper_change.reshape(-1).long()  # (b*num_future,)

    gripper_loss = F.cross_entropy(gripper_logits_flat, gripper_targets_flat)

    # Compute gripper accuracy
    gripper_preds = gripper_logits_flat.argmax(dim=1)
    gripper_accuracy = (gripper_preds == gripper_targets_flat).float().mean()

    # 4. Total weighted loss
    total_loss = (
        heatmap_loss_weight * heatmap_loss +
        rotation_loss_weight * rotation_loss +
        gripper_loss_weight * gripper_loss
    )

    return {
        'loss': total_loss,
        'heatmap_loss': heatmap_loss,
        'rotation_loss': rotation_loss,
        'gripper_loss': gripper_loss,
        'rotation_accuracy': rotation_accuracy,
        'gripper_accuracy': gripper_accuracy,
    }
