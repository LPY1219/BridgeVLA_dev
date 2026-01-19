"""
Diffusion Action Decoder with Dual-Branch Architecture

This module implements an action decoder that predicts motion changes from DiT intermediate features.

Unlike the original VAE-based action decoder, this version:
- Takes DiT intermediate layer features as input (not VAE decoder features)
- Predicts heatmap peak position changes, rotation changes, and gripper changes
- All predictions are relative to the first frame (to avoid overfitting to initial state)

Architecture (Dual-Branch Design):
- Input: DiT intermediate features (b*v, seq_len, dim) from a specified block
- Shared: Feature projection → Multi-view attention (preserving view dimension)
- Branch 1 (Heatmap): View-specific features → Temporal modeling → Per-view heatmap prediction
- Branch 2 (Rotation/Gripper): View-averaged features → Temporal modeling → Global prediction

Key Design Rationale:
1. Heatmap prediction is VIEW-SPECIFIC: Each camera has different 2D projection
   → Keep view dimension, decode separately for each view
2. Rotation/Gripper prediction is VIEW-AGNOSTIC: Global robot state
   → Average across views, decode from global representation

Usage:
    decoder = DiffusionActionDecoder(
        dit_feature_dim=3072,  # DiT hidden dimension
        num_views=3,
        num_rotation_bins=72,
        num_future_frames=24
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


class TemporalUpsampler(nn.Module):
    """
    Advanced temporal upsampling module using learnable queries and cross-attention.

    Instead of simple interpolation + MLP, this module uses:
    1. Learnable temporal queries for each target timestep
    2. Cross-attention to extract information from input timesteps
    3. Self-attention among target timesteps for temporal coherence

    This allows the network to learn the optimal temporal upsampling strategy.
    """

    def __init__(self, hidden_dim: int, target_frames: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_frames = target_frames

        # Learnable temporal queries (one for each target frame)
        self.temporal_queries = nn.Parameter(torch.randn(1, target_frames, hidden_dim))
        nn.init.normal_(self.temporal_queries, std=0.02)

        # Positional encoding for target frames
        self.target_pos_embed = nn.Parameter(torch.randn(1, target_frames, hidden_dim))
        nn.init.normal_(self.target_pos_embed, std=0.02)

        # Cross-attention: queries attend to input frames
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Self-attention: temporal coherence among target frames
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        # FFN for refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, input_frames, hidden_dim)
        Returns:
            Upsampled features (batch, target_frames, hidden_dim)
        """
        batch_size = x.shape[0]

        # Expand queries for batch
        queries = self.temporal_queries.expand(batch_size, -1, -1)  # (b, target_frames, hidden_dim)
        queries = queries + self.target_pos_embed  # Add positional encoding

        # Cross-attention: extract information from input frames
        attended, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x
        )
        queries = queries + attended
        queries = self.cross_norm(queries)

        # Self-attention: temporal coherence
        self_attended, _ = self.self_attention(
            queries, queries, queries
        )
        queries = queries + self_attended
        queries = self.self_norm(queries)

        # FFN refinement
        refined = self.ffn(queries)
        output = queries + refined
        output = self.ffn_norm(output)

        return output


class DiffusionActionDecoder(nn.Module):
    """
    Dual-branch action decoder for view-specific heatmap and view-agnostic rotation/gripper prediction.

    This decoder takes features from a DiT intermediate block and predicts:
    1. Heatmap peak position changes (per-view, since each camera has different 2D projection)
    2. Rotation changes (view-agnostic, global robot state)
    3. Gripper state changes (view-agnostic, global robot state)

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
        num_future_frames: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dit_feature_dim = dit_feature_dim
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_rotation_bins = num_rotation_bins
        self.num_future_frames = num_future_frames

        # ========================================================================
        # SHARED COMPONENTS
        # ========================================================================

        # 1. Deep feature projection: DiT features -> hidden_dim
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

        # 2. Multi-view cross-attention (preserve view dimension for information exchange)
        self.view_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.view_attention_norm = nn.LayerNorm(hidden_dim)

        # ========================================================================
        # BRANCH 1: HEATMAP PREDICTION (View-Specific)
        # ========================================================================

        # Advanced temporal upsampling for heatmap branch
        # Uses learnable queries + cross-attention for better temporal modeling
        self.heatmap_temporal_upsampler = TemporalUpsampler(
            hidden_dim=hidden_dim,
            target_frames=num_future_frames + 1,  # +1 for history frame
            dropout=dropout
        )

        # History conditioning for heatmap (per-view)
        self.heatmap_history_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.heatmap_history_norm = nn.LayerNorm(hidden_dim)

        # Temporal transformer for heatmap (processes all views together)
        heatmap_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.heatmap_temporal_transformer = nn.TransformerEncoder(heatmap_encoder_layer, num_layers=4)

        # Per-view heatmap prediction head
        self.heatmap_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # (delta_h, delta_w) for single view
        )

        # ========================================================================
        # BRANCH 2: ROTATION/GRIPPER PREDICTION (View-Agnostic)
        # ========================================================================

        # Advanced temporal upsampling for rotation/gripper branch
        self.global_temporal_upsampler = TemporalUpsampler(
            hidden_dim=hidden_dim,
            target_frames=num_future_frames + 1,
            dropout=dropout
        )

        # History conditioning for rotation/gripper
        self.global_history_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.global_history_norm = nn.LayerNorm(hidden_dim)

        # Temporal transformer for rotation/gripper
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.global_temporal_transformer = nn.TransformerEncoder(global_encoder_layer, num_layers=4)

        # Rotation prediction head (classification)
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rotation_bins * 3)  # roll, pitch, yaw
        )

        # Gripper prediction head (binary classification)
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
        Dual-branch forward pass.

        Args:
            dit_features: DiT intermediate features (b*v, seq_len, dit_feature_dim)
            shape_info: (f, h, w) tuple indicating temporal and spatial dimensions
            num_views: Number of views (default: 3)

        Returns:
            heatmap_delta: (b, num_future_frames, num_views, 2) - Per-view heatmap deltas
            rotation_logits: (b, num_future_frames, num_rotation_bins*3) - View-agnostic rotation
            gripper_logits: (b, num_future_frames, 2) - View-agnostic gripper
        """
        bv, seq_len, _ = dit_features.shape
        b = bv // num_views
        f, h, w = shape_info

        # ====================================================================
        # SHARED PROCESSING
        # ====================================================================

        # 1. Project features to hidden dimension
        features = self.feature_projection(dit_features)  # (b*v, seq_len, hidden_dim)

        # 2. Reshape to separate batch, view, and spatial-temporal dimensions
        features = rearrange(
            features,
            '(b v) (f h w) d -> b v f (h w) d',
            b=b, v=num_views, f=f, h=h, w=w
        )

        # 3. Spatial pooling: average over h*w dimension
        features = features.mean(dim=3)  # (b, v, f, hidden_dim)

        # 4. Multi-view cross-attention (preserve view dimension!)
        # Process each time frame separately, allowing views to exchange information
        view_attended_features = []
        for t_idx in range(f):
            # Get features from all views at time t
            view_features_t = features[:, :, t_idx, :]  # (b, v, hidden_dim)

            # Cross-attention across views (each view attends to all views)
            attended_t, _ = self.view_attention(
                view_features_t, view_features_t, view_features_t
            )  # (b, v, hidden_dim)

            # Residual + norm (IMPORTANT: Do NOT average across views here!)
            attended_t = attended_t + view_features_t
            attended_t = self.view_attention_norm(attended_t)

            view_attended_features.append(attended_t)

        view_attended_features = torch.stack(view_attended_features, dim=2)  # (b, v, f, hidden_dim)

        # ====================================================================
        # BRANCH 1: HEATMAP PREDICTION (View-Specific)
        # ====================================================================

        heatmap_delta = self._forward_heatmap_branch(view_attended_features, f)

        # ====================================================================
        # BRANCH 2: ROTATION/GRIPPER PREDICTION (View-Agnostic)
        # ====================================================================

        rotation_logits, gripper_logits = self._forward_global_branch(view_attended_features, f)

        return heatmap_delta, rotation_logits, gripper_logits

    def _forward_heatmap_branch(
        self,
        view_features: torch.Tensor,  # (b, v, f, hidden_dim)
        f: int
    ) -> torch.Tensor:
        """
        Heatmap prediction branch (view-specific).

        Keep view dimension throughout to allow per-view heatmap decoding.
        """
        b, v, _, hidden_dim = view_features.shape

        # 1. Temporal upsampling using advanced TemporalUpsampler
        # Reshape to (b*v, f, hidden_dim) for temporal processing
        heatmap_features = rearrange(view_features, 'b v f d -> (b v) f d')

        if f != self.num_future_frames + 1:
            # Use learnable query-based upsampling
            heatmap_features = self.heatmap_temporal_upsampler(heatmap_features)  # (b*v, target_f, hidden_dim)
        else:
            # No upsampling needed, features already match target length
            pass

        # 2. Separate history and future
        history_features = heatmap_features[:, :1, :]  # (b*v, 1, hidden_dim)
        future_features = heatmap_features[:, 1:, :]  # (b*v, num_future_frames, hidden_dim)

        # 3. History-conditioned cross-attention
        future_conditioned, _ = self.heatmap_history_attention(
            query=future_features,
            key=history_features,
            value=history_features
        )
        future_features = future_features + future_conditioned
        future_features = self.heatmap_history_norm(future_features)

        # 4. Temporal self-attention
        temporal_features = self.heatmap_temporal_transformer(future_features)  # (b*v, num_future_frames, hidden_dim)

        # 5. Predict per-view heatmap delta
        heatmap_delta_per_view = self.heatmap_delta_head(temporal_features)  # (b*v, num_future_frames, 2)

        # 6. Reshape to (b, num_future_frames, v, 2)
        heatmap_delta = rearrange(
            heatmap_delta_per_view,
            '(b v) t c -> b t v c',
            b=b, v=v
        )

        return heatmap_delta

    def _forward_global_branch(
        self,
        view_features: torch.Tensor,  # (b, v, f, hidden_dim)
        f: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotation/Gripper prediction branch (view-agnostic).

        Average across views to get global representation.
        """
        # 1. Average across views to get view-agnostic global features
        global_features = view_features.mean(dim=1)  # (b, f, hidden_dim)

        # 2. Temporal upsampling using advanced TemporalUpsampler
        if f != self.num_future_frames + 1:
            # Use learnable query-based upsampling
            global_features = self.global_temporal_upsampler(global_features)  # (b, target_f, hidden_dim)
        else:
            # No upsampling needed
            pass

        # 3. Separate history and future
        history_features = global_features[:, :1, :]  # (b, 1, hidden_dim)
        future_features = global_features[:, 1:, :]  # (b, num_future_frames, hidden_dim)

        # 4. History-conditioned cross-attention
        future_conditioned, _ = self.global_history_attention(
            query=future_features,
            key=history_features,
            value=history_features
        )
        future_features = future_features + future_conditioned
        future_features = self.global_history_norm(future_features)

        # 5. Temporal self-attention
        temporal_features = self.global_temporal_transformer(future_features)  # (b, num_future_frames, hidden_dim)

        # 6. Predict rotation and gripper
        rotation_logits = self.rotation_head(temporal_features)  # (b, num_future_frames, num_rotation_bins*3)
        gripper_logits = self.gripper_head(temporal_features)  # (b, num_future_frames, 2)

        return rotation_logits, gripper_logits


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
