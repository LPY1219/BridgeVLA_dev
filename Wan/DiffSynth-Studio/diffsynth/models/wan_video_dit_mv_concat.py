"""
Multi-View Video Diffusion Model with Token Concatenation

This model implements multi-view video generation by concatenating tokens from
different views along the sequence dimension, instead of using separate multi-view
attention modules (mvs_attn).

Key idea:
- Input: (b*v, c, f, h, w) multi-view latents
- After patchify: (b*v, f*h'*w', d) tokens per view
- Concatenate: (b, v*f*h'*w', d) single long sequence
- Standard DiT forward (self-attention covers all views naturally)
- Split back: (b*v, f*h'*w', d)
- Output: (b*v, c, f, h, w)

This approach allows cross-view interaction through standard self-attention
without additional multi-view attention modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    """
    Standard DiT Block without multi-view attention (mvs_attn).
    Used for token-concatenation based multi-view processing.
    """
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        """
        Standard DiT Block forward pass.

        Args:
            x: Input tokens, shape (b, seq, d)
               For multi-view concat: seq = v * f * h' * w'
            context: Text conditioning, shape (b, ctx_len, d)
            t_mod: Time step modulation
            freqs: RoPE position encoding, shape (seq, 1, head_dim//2)
        """
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)

        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        # Self-attention (covers all views when tokens are concatenated)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        # Cross-attention with text
        x = x + self.cross_attn(self.norm3(x), context)

        # FFN
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))

        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel_mv_concat(torch.nn.Module):
    """
    Multi-View Wan Video Diffusion Model with Token Concatenation.

    This model processes multi-view videos by concatenating tokens from all views
    into a single sequence, allowing cross-view interaction through standard
    self-attention without additional multi-view attention modules.

    Key features:
    - Token-level concatenation instead of image-level spatial concatenation
    - Standard DiT blocks (no mvs_attn)
    - Dual head support for RGB and Heatmap outputs
    - Compatible with existing VAE and training pipelines
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        use_dual_head: bool = False,
        num_views: int = 3,
    ):
        super().__init__()
        print(f"[DEBUG] WanModel_mv_concat initialized with num_views={num_views}, use_dual_head={use_dual_head}")

        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.use_dual_head = use_dual_head
        self.num_views = num_views
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # Standard DiT blocks (without mvs_attn)
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # Dual head mode for RGB and Heatmap
        if use_dual_head:
            self.head_rgb = Head(dim, out_dim, patch_size, eps)
            self.head_heatmap = Head(dim, out_dim, patch_size, eps)
            self.head = None
        else:
            self.head = Head(dim, out_dim, patch_size, eps)
            self.head_rgb = None
            self.head_heatmap = None

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def concat_view_tokens(self, x: torch.Tensor, num_views: int) -> torch.Tensor:
        """
        Concatenate multi-view tokens along the sequence dimension.

        Args:
            x: Multi-view tokens, shape (b*v, seq, d) where seq = f*h'*w'
            num_views: Number of views (v)

        Returns:
            Concatenated tokens, shape (b, v*seq, d)
        """
        b_v, seq, d = x.shape
        b = b_v // num_views
        # (b*v, seq, d) -> (b, v, seq, d) -> (b, v*seq, d)
        x = x.view(b, num_views, seq, d)
        x = x.view(b, num_views * seq, d)
        return x

    def split_view_tokens(self, x: torch.Tensor, num_views: int, seq_per_view: int) -> torch.Tensor:
        """
        Split concatenated tokens back to multi-view format.

        Args:
            x: Concatenated tokens, shape (b, v*seq, d) or (b, v*seq, out_features)
            num_views: Number of views (v)
            seq_per_view: Sequence length per view (seq = f*h'*w')

        Returns:
            Multi-view tokens, shape (b*v, seq, d)
        """
        b, v_seq, d = x.shape
        # (b, v*seq, d) -> (b, v, seq, d) -> (b*v, seq, d)
        x = x.view(b, num_views, seq_per_view, d)
        x = x.view(b * num_views, seq_per_view, d)
        return x

    def prepare_multiview_freqs(self, grid_size: Tuple[int, int, int], num_views: int, device: torch.device) -> torch.Tensor:
        """
        Prepare position encoding (RoPE) for multi-view concatenated tokens.

        Each view uses the same spatial position encoding, repeated num_views times.
        This allows the model to learn cross-view correspondence based on
        spatial positions while maintaining separate representations for each view.

        Uses caching to avoid recomputing freqs on every forward pass.

        Args:
            grid_size: (f, h, w) grid dimensions after patchify
            num_views: Number of views
            device: Target device

        Returns:
            Position encoding for concatenated tokens, shape (v*f*h*w, 1, head_dim//2)
        """
        # Check cache
        cache_key = (grid_size, num_views, device)
        if hasattr(self, '_freqs_cache') and self._freqs_cache_key == cache_key:
            return self._freqs_cache

        f, h, w = grid_size

        # Build single-view position encoding
        single_view_freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)

        # Repeat for all views (each view has same spatial position encoding)
        multiview_freqs = single_view_freqs.repeat(num_views, 1, 1)

        # Cache the result
        self._freqs_cache = multiview_freqs
        self._freqs_cache_key = cache_key

        return multiview_freqs

    def prepare_multiview_t_mod(self, t_mod: torch.Tensor, num_views: int, seq_per_view: int) -> torch.Tensor:
        """
        Prepare time modulation for multi-view concatenated tokens.

        Args:
            t_mod: Time modulation, shape (b, seq, 6, d) or (b, 6, d)
            num_views: Number of views
            seq_per_view: Sequence length per view

        Returns:
            Expanded time modulation for concatenated sequence
        """
        if len(t_mod.shape) == 4:
            # Has sequence dimension: (b, seq, 6, d)
            # Expand to (b, v*seq, 6, d)
            b, seq, six, d = t_mod.shape
            # Repeat along sequence dimension
            t_mod_expanded = t_mod.repeat(1, num_views, 1, 1)
            return t_mod_expanded
        else:
            # No sequence dimension: (b, 6, d)
            # Keep as is (will be broadcast)
            return t_mod

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        """
        Convert latent to patch tokens.

        Args:
            x: Input latent, shape (b*v, c, f, h, w)

        Returns:
            tokens: Patch tokens, shape (b*v, f*h'*w', d)
            grid_size: (f, h', w') grid dimensions
        """
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]  # (f, h', w')
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Convert patch tokens back to latent format.

        Args:
            x: Patch tokens, shape (b*v, f*h'*w', out_features)
            grid_size: (f, h', w') grid dimensions

        Returns:
            Latent, shape (b*v, c, f*ps[0], h'*ps[1], w'*ps[2])
        """
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        """
        Forward pass with multi-view token concatenation.

        Args:
            x: Multi-view latents, shape (b*v, c, f, h, w)
            timestep: Time step
            context: Text conditioning, shape (b, ctx_len, text_dim)

        Returns:
            Denoised latents, shape (b*v, c_out, f, h, w)
            If dual_head: c_out = 2 * out_dim (RGB + Heatmap concatenated)
        """
        num_views = self.num_views
        b_v = x.shape[0]
        b = b_v // num_views

        # Time embedding
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        # Patchify each view independently
        x, grid_size = self.patchify(x)  # (b*v, f*h'*w', d)
        f, h, w = grid_size
        seq_per_view = f * h * w

        # Concatenate view tokens: (b*v, seq, d) -> (b, v*seq, d)
        x = self.concat_view_tokens(x, num_views)

        # Prepare multi-view position encoding
        freqs = self.prepare_multiview_freqs(grid_size, num_views, x.device)

        # Prepare multi-view time modulation
        t_mod_mv = self.prepare_multiview_t_mod(t_mod, num_views, seq_per_view)

        # Expand context for batch dimension (context is per-batch, not per-view)
        # context shape: (b, ctx_len, d) - already correct

        # Debug print (only once)
        if not hasattr(self, '_forward_debug_printed'):
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"[DEBUG] WanModel_mv_concat forward:")
                print(f"  Input x shape (after concat): {x.shape}")
                print(f"  Grid size: {grid_size}, seq_per_view: {seq_per_view}")
                print(f"  Freqs shape: {freqs.shape}")
                print(f"  Context shape: {context.shape}")
                print(f"  t_mod shape: {t_mod.shape}, t_mod_mv shape: {t_mod_mv.shape}")
            self._forward_debug_printed = True

        # Process through DiT blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod_mv, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod_mv, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod_mv, freqs)

        # Apply head(s) and unpatchify
        if self.use_dual_head:
            # Dual head mode: separate RGB and Heatmap heads
            x_rgb = self.head_rgb(x, t)
            x_heatmap = self.head_heatmap(x, t)

            # Split back to multi-view: (b, v*seq, out) -> (b*v, seq, out)
            x_rgb = self.split_view_tokens(x_rgb, num_views, seq_per_view)
            x_heatmap = self.split_view_tokens(x_heatmap, num_views, seq_per_view)

            # Unpatchify
            x_rgb = self.unpatchify(x_rgb, grid_size)
            x_heatmap = self.unpatchify(x_heatmap, grid_size)

            # Concatenate RGB and Heatmap along channel dimension
            x = torch.cat([x_rgb, x_heatmap], dim=1)

            if not hasattr(self, '_dual_head_debug_printed'):
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    print(f"[DEBUG] WanModel_mv_concat Dual Head output:")
                    print(f"  x_rgb shape: {x_rgb.shape}, x_heatmap shape: {x_heatmap.shape}")
                    print(f"  Final output shape: {x.shape}")
                self._dual_head_debug_printed = True
        else:
            # Single head mode
            x = self.head(x, t)
            x = self.split_view_tokens(x, num_views, seq_per_view)
            x = self.unpatchify(x, grid_size)

        return x

    @staticmethod
    def state_dict_converter():
        return WanModelConcatStateDictConverter()

    def adapt_pretrained_weights(self, pretrained_state_dict, strict=False):
        """
        Load pretrained weights and adapt dimension changes.
        Compatible with both single-view WanModel and multi-view WanModel_mv weights.

        Args:
            pretrained_state_dict: Pretrained model state_dict
            strict: Whether to strictly match all keys

        Returns:
            Loading result information
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main_process = (local_rank == 0)

        if is_main_process:
            print("        [WanModel_mv_concat] adapt_pretrained_weights called")
            print(f"        [WanModel_mv_concat] Current in_dim: {self.in_dim}, num_views: {self.num_views}")

        current_state_dict = self.state_dict()

        # Remove single head keys if using dual head
        if self.use_dual_head:
            keys_to_remove = [k for k in current_state_dict.keys() if k.startswith('head.')]
            for key in keys_to_remove:
                del current_state_dict[key]

        adapted_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        size_mismatched_keys = []

        # Handle dual head mode
        if self.use_dual_head:
            if is_main_process:
                print("\n" + "="*80)
                print("[DEBUG] DUAL HEAD MODE: Initializing head_rgb and head_heatmap")
                print("="*80)

            # Copy head weights to both head_rgb and head_heatmap
            head_keys = ['head.head.weight', 'head.head.bias', 'head.norm.weight',
                        'head.norm.bias', 'head.modulation']
            for key in head_keys:
                if key in pretrained_state_dict:
                    value = pretrained_state_dict[key]
                    rgb_key = key.replace('head.', 'head_rgb.')
                    heatmap_key = key.replace('head.', 'head_heatmap.')
                    adapted_state_dict[rgb_key] = value.clone()
                    adapted_state_dict[heatmap_key] = value.clone()
                    if is_main_process:
                        print(f"  Copied {key} to both heads")

        # Handle patch_embedding weight adaptation
        if 'patch_embedding.weight' in pretrained_state_dict:
            pretrained_weight = pretrained_state_dict['patch_embedding.weight']
            current_weight = current_state_dict['patch_embedding.weight']

            if pretrained_weight.shape != current_weight.shape:
                if is_main_process:
                    print(f"Adapting patch_embedding.weight: {pretrained_weight.shape} -> {current_weight.shape}")

                pretrained_in_dim = pretrained_weight.shape[1]
                current_in_dim = current_weight.shape[1]

                if current_in_dim > pretrained_in_dim:
                    adapted_weight = torch.zeros_like(current_weight)
                    adapted_weight[:, :pretrained_in_dim] = pretrained_weight
                    adapted_state_dict['patch_embedding.weight'] = adapted_weight
                    size_mismatched_keys.append('patch_embedding.weight')
                else:
                    adapted_weight = pretrained_weight[:, :current_in_dim]
                    adapted_state_dict['patch_embedding.weight'] = adapted_weight
                    size_mismatched_keys.append('patch_embedding.weight')
            else:
                adapted_state_dict['patch_embedding.weight'] = pretrained_weight

        if 'patch_embedding.bias' in pretrained_state_dict:
            adapted_state_dict['patch_embedding.bias'] = pretrained_state_dict['patch_embedding.bias']

        # Copy remaining weights, skipping mvs_attn related keys
        mvs_related_prefixes = ['mvs_attn', 'norm_mvs', 'modulation_mvs', 'projector']

        for key, param in pretrained_state_dict.items():
            if key not in adapted_state_dict:
                # Skip mvs_attn related weights (not needed for concat model)
                skip = False
                for prefix in mvs_related_prefixes:
                    if prefix in key:
                        skip = True
                        if is_main_process and not hasattr(self, f'_skip_warned_{prefix}'):
                            print(f"  Skipping mvs_attn related key: {key}")
                            setattr(self, f'_skip_warned_{prefix}', True)
                        break

                if skip:
                    continue

                # Skip single head keys in dual head mode
                if self.use_dual_head and key.startswith('head.'):
                    continue

                if key in current_state_dict:
                    if param.shape == current_state_dict[key].shape:
                        adapted_state_dict[key] = param
                    else:
                        size_mismatched_keys.append(key)
                        if is_main_process:
                            print(f"Warning: Size mismatch for {key}: {param.shape} vs {current_state_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

        # Check missing keys
        for key in current_state_dict:
            if key not in adapted_state_dict:
                missing_keys.append(key)

        # Load adapted weights
        load_result = self.load_state_dict(adapted_state_dict, strict=False, assign=True)

        if is_main_process:
            print("\n=== Weight Adaptation Summary ===")
            print(f"Successfully adapted keys: {len(adapted_state_dict)}")
            print(f"Size-mismatched keys (adapted): {len(size_mismatched_keys)}")
            print(f"Missing keys: {len(missing_keys)}")
            if missing_keys:
                for key in missing_keys[:5]:
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys) - 5} more")
            print(f"Unexpected keys: {len(unexpected_keys)}")

        return {
            'adapted_keys': list(adapted_state_dict.keys()),
            'size_mismatched_keys': size_mismatched_keys,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'load_result': load_result
        }


class WanModelConcatStateDictConverter:
    """State dict converter for WanModel_mv_concat."""

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        """Convert from civitai format (same as original WanModel)."""
        # Remove mvs_attn related keys if present
        state_dict = {k: v for k, v in state_dict.items()
                     if not any(x in k for x in ['mvs_attn', 'norm_mvs', 'modulation_mvs', 'projector'])}

        # Use same hash-based config detection as original
        from .wan_video_dit import WanModelStateDictConverter
        converter = WanModelStateDictConverter()
        state_dict, config = converter.from_civitai(state_dict)

        # Add num_views to config (default 3)
        config['num_views'] = 3

        return state_dict, config
