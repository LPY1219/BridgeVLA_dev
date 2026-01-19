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
        if isinstance(x,tuple):
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
        has_seq = len(t_mod.shape) == 4
        print("has_seq:", has_seq)
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            print("has seq!!!!!")
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
       
        x = x + self.cross_attn(self.norm3(x), context)
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


class WanModel(torch.nn.Module):
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
    ):
        super().__init__()
        # print("[DEBUG] WanModel initialized from: wan_video_dit.py")
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
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # 根据dual head模式选择不同的head结构
        if use_dual_head:
            # 双head模式：head_rgb和head_heatmap各自输出相同的通道数
            # out_dim 表示每个head的输出通道数（例如48），两个head合起来输出 out_dim*2 (96)
            # 注意：这里不需要除以2，因为out_dim本身就表示单个head的输出
            self.head_rgb = Head(dim, out_dim, patch_size, eps)
            self.head_heatmap = Head(dim, out_dim, patch_size, eps)
            self.head = None  # 不使用单一head
            # Debug信息将在adapt_pretrained_weights中打印（只在主进程）
        else:
            # 单head模式：保持原有行为
            self.head = Head(dim, out_dim, patch_size, eps)
            self.head_rgb = None
            self.head_heatmap = None

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        # print("[DEBUG] patchify called from: wan_video_dit.py")
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
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
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
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
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        # 根据dual head模式使用不同的head
        if self.use_dual_head:
            # 双head模式：分别通过rgb和heatmap的head，然后拼接
            x_rgb = self.head_rgb(x, t)
            x_heatmap = self.head_heatmap(x, t)
            x_rgb = self.unpatchify(x_rgb, (f, h, w))
            x_heatmap = self.unpatchify(x_heatmap, (f, h, w))
            # 沿着通道维度拼接：[rgb_channels, heatmap_channels]
            x = torch.cat([x_rgb, x_heatmap], dim=1)
            # Debug打印（仅在主进程且训练开始时打印一次）
            if not hasattr(self, '_dual_head_debug_printed'):
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    print(f"[DEBUG] WanModel Forward: Using DUAL HEAD - x_rgb.shape={x_rgb.shape}, x_heatmap.shape={x_heatmap.shape}, output.shape={x.shape}")
                self._dual_head_debug_printed = True
        else:
            # 单head模式
            x = self.head(x, t)
            x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()

    def adapt_pretrained_weights(self, pretrained_state_dict, strict=False):
        """
        加载预训练权重并适配维度变化

        Args:
            pretrained_state_dict: 预训练模型的state_dict
            strict: 是否严格匹配所有键

        Returns:
            加载结果信息
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main_process = (local_rank == 0)

        if is_main_process:
            print("        [WanModel] adapt_pretrained_weights called")
            print(f"        [WanModel] Current in_dim: {self.in_dim}")

        current_state_dict = self.state_dict()

        # 在双head模式下，如果current_state_dict中存在单head的键，需要先删除
        # 这些键可能是之前使用assign=True加载时意外创建的
        if self.use_dual_head:
            keys_to_remove = [k for k in current_state_dict.keys() if k.startswith('head.')]
            if keys_to_remove and is_main_process:
                print(f"        [DEBUG] Removing {len(keys_to_remove)} single-head keys from current model state")
            for key in keys_to_remove:
                del current_state_dict[key]

        adapted_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        size_mismatched_keys = []

        # Debug: 打印键名样例（只在主进程）
        if is_main_process:
            print(f"        [DEBUG] Pretrained state_dict sample keys (first 5):")
            for i, key in enumerate(list(pretrained_state_dict.keys())[:5]):
                print(f"          - {key}")
            print(f"        [DEBUG] Current model state_dict sample keys (first 5):")
            for i, key in enumerate(list(current_state_dict.keys())[:5]):
                print(f"          - {key}")
            print(f"        [DEBUG] Total pretrained keys: {len(pretrained_state_dict)}, Current model keys: {len(current_state_dict)}")
            print(f"        [DEBUG] Dual head mode: {self.use_dual_head}")

        # 0. 处理双head模式的特殊情况
        if self.use_dual_head:
            # 在双head模式下，需要将预训练的head权重复制到head_rgb和head_heatmap
            if is_main_process:
                print("\n" + "="*80)
                print("[DEBUG] DUAL HEAD MODE: Initializing head_rgb and head_heatmap from pretrained")
                print("="*80)

            # 处理 head.head.weight -> head_rgb.head.weight 和 head_heatmap.head.weight
            if 'head.head.weight' in pretrained_state_dict:
                pretrained_head_weight = pretrained_state_dict['head.head.weight']
                if is_main_process:
                    print(f"[DEBUG] Copying head.head.weight to both heads")
                    print(f"  Source shape: {pretrained_head_weight.shape}")
                adapted_state_dict['head_rgb.head.weight'] = pretrained_head_weight.clone()
                adapted_state_dict['head_heatmap.head.weight'] = pretrained_head_weight.clone()
                if is_main_process:
                    print(f"  ✓ head_rgb.head.weight initialized (shape: {pretrained_head_weight.shape})")
                    print(f"  ✓ head_heatmap.head.weight initialized (shape: {pretrained_head_weight.shape})")

            # 处理 head.head.bias
            if 'head.head.bias' in pretrained_state_dict:
                pretrained_head_bias = pretrained_state_dict['head.head.bias']
                adapted_state_dict['head_rgb.head.bias'] = pretrained_head_bias.clone()
                adapted_state_dict['head_heatmap.head.bias'] = pretrained_head_bias.clone()
                if is_main_process:
                    print(f"  ✓ Both head biases initialized (shape: {pretrained_head_bias.shape})")

            # 处理 head.norm.weight 和 head.norm.bias
            if 'head.norm.weight' in pretrained_state_dict:
                adapted_state_dict['head_rgb.norm.weight'] = pretrained_state_dict['head.norm.weight'].clone()
                adapted_state_dict['head_heatmap.norm.weight'] = pretrained_state_dict['head.norm.weight'].clone()
                if is_main_process:
                    print(f"  ✓ Both head norm.weight initialized")

            if 'head.norm.bias' in pretrained_state_dict:
                adapted_state_dict['head_rgb.norm.bias'] = pretrained_state_dict['head.norm.bias'].clone()
                adapted_state_dict['head_heatmap.norm.bias'] = pretrained_state_dict['head.norm.bias'].clone()
                if is_main_process:
                    print(f"  ✓ Both head norm.bias initialized")

            # 处理 head.modulation
            if 'head.modulation' in pretrained_state_dict:
                adapted_state_dict['head_rgb.modulation'] = pretrained_state_dict['head.modulation'].clone()
                adapted_state_dict['head_heatmap.modulation'] = pretrained_state_dict['head.modulation'].clone()
                if is_main_process:
                    print(f"  ✓ Both head modulation initialized")

            if is_main_process:
                print("="*80)
                print("[DEBUG] Dual Head Initialization Complete - Both heads ready for training!")
                print("="*80 + "\n")

        # 1. 处理 patch_embedding 权重 (输入维度可能不同)
        if 'patch_embedding.weight' in pretrained_state_dict:
            pretrained_weight = pretrained_state_dict['patch_embedding.weight']
            current_weight = current_state_dict['patch_embedding.weight']

            if pretrained_weight.shape != current_weight.shape:
                print(f"Adapting patch_embedding.weight: {pretrained_weight.shape} -> {current_weight.shape}")

                # 获取维度信息 [out_channels, in_channels, d, h, w]
                pretrained_in_dim = pretrained_weight.shape[1]
                current_in_dim = current_weight.shape[1]

                if current_in_dim > pretrained_in_dim:
                    # 输入通道扩展：新通道权重零初始化，训练初期不影响 patch_embedding 输出
                    adapted_weight = torch.zeros_like(current_weight)
                    adapted_weight[:, :pretrained_in_dim] = pretrained_weight
                    adapted_state_dict['patch_embedding.weight'] = adapted_weight
                    size_mismatched_keys.append('patch_embedding.weight')
                    print(f"  -> Zero-initialized {current_in_dim - pretrained_in_dim} new input channels (no initial impact)")
                else:
                    # 维度缩小：截断
                    adapted_weight = pretrained_weight[:, :current_in_dim]
                    adapted_state_dict['patch_embedding.weight'] = adapted_weight
                    size_mismatched_keys.append('patch_embedding.weight')
                    print("  -> Truncated input channels")
            else:
                adapted_state_dict['patch_embedding.weight'] = pretrained_weight

        # 处理 patch_embedding bias (bias 的维度是 out_channels，与 in_dim 无关，直接复制即可)
        if 'patch_embedding.bias' in pretrained_state_dict:
            adapted_state_dict['patch_embedding.bias'] = pretrained_state_dict['patch_embedding.bias']
            if 'patch_embedding.weight' in size_mismatched_keys:
                print("  -> Copied bias (unchanged, as it depends on out_channels only)")

        # 2. 处理 head 权重 (输出维度可能不同)
        # 注意：在双head模式下，head的权重已经在步骤0中处理过了，这里跳过
        if not self.use_dual_head:
            if 'head.head.weight' in pretrained_state_dict:
                pretrained_weight = pretrained_state_dict['head.head.weight']
                current_weight = current_state_dict['head.head.weight']

                if pretrained_weight.shape != current_weight.shape:
                    print(f"Adapting head.head.weight: {pretrained_weight.shape} -> {current_weight.shape}")

                    # 获取维度信息 [out_features, in_features]
                    pretrained_out_dim = pretrained_weight.shape[0]
                    current_out_dim = current_weight.shape[0]

                    if current_out_dim > pretrained_out_dim:
                        # 输出维度扩展：新输出通道权重零初始化，训练初期不影响原有输出通道
                        adapted_weight = torch.zeros_like(current_weight)
                        adapted_weight[:pretrained_out_dim] = pretrained_weight
                        adapted_state_dict['head.head.weight'] = adapted_weight
                        size_mismatched_keys.append('head.head.weight')
                        print(f"  -> Zero-initialized {current_out_dim - pretrained_out_dim} new output features (no initial impact)")
                    else:
                        # 维度缩小：截断
                        adapted_weight = pretrained_weight[:current_out_dim]
                        adapted_state_dict['head.head.weight'] = adapted_weight
                        size_mismatched_keys.append('head.head.weight')
                        print("  -> Truncated output features")
                else:
                    adapted_state_dict['head.head.weight'] = pretrained_weight

            # 处理 head bias
            if 'head.head.bias' in pretrained_state_dict:
                pretrained_bias = pretrained_state_dict['head.head.bias']
                current_bias = current_state_dict['head.head.bias']

                if pretrained_bias.shape != current_bias.shape:
                    print(f"Adapting head.head.bias: {pretrained_bias.shape} -> {current_bias.shape}")

                    pretrained_out_dim = pretrained_bias.shape[0]
                    current_out_dim = current_bias.shape[0]

                    if current_out_dim > pretrained_out_dim:
                        # 输出维度扩展：新输出通道bias零初始化，配合权重零初始化，确保新输出为零
                        adapted_bias = torch.zeros_like(current_bias)
                        adapted_bias[:pretrained_out_dim] = pretrained_bias
                        adapted_state_dict['head.head.bias'] = adapted_bias
                        size_mismatched_keys.append('head.head.bias')
                        print(f"  -> Zero-initialized {current_out_dim - pretrained_out_dim} new output biases (ensures zero output)")
                    else:
                        # 缩小：截断
                        adapted_bias = pretrained_bias[:current_out_dim]
                        adapted_state_dict['head.head.bias'] = adapted_bias
                        size_mismatched_keys.append('head.head.bias')
                        print("  -> Truncated output biases")
                else:
                    adapted_state_dict['head.head.bias'] = pretrained_bias

        # 3. 复制其他所有权重
        for key, param in pretrained_state_dict.items():
            if key not in adapted_state_dict:
                # 在双head模式下，跳过单head的权重（已经复制到head_rgb和head_heatmap了）
                if self.use_dual_head and key.startswith('head.'):
                    if is_main_process:
                        print(f"  ⏭️  Skipping single-head key in dual-head mode: {key}")
                    continue

                if key in current_state_dict:
                    if param.shape == current_state_dict[key].shape:
                        adapted_state_dict[key] = param
                    else:
                        size_mismatched_keys.append(key)
                        print(f"Warning: Size mismatch for {key}: {param.shape} vs {current_state_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

        # 4. 检查缺失的键
        for key in current_state_dict:
            if key not in adapted_state_dict:
                missing_keys.append(key)

        # 5. 加载适配后的权重
        # 使用 assign=True 来正确处理 meta tensors (从 init_weights_on_device 创建的模型)
        load_result = self.load_state_dict(adapted_state_dict, strict=False, assign=True)

        # 打印摘要
        print("\n=== Weight Adaptation Summary ===")
        print(f"Successfully adapted keys: {len(adapted_state_dict)}")
        print(f"Size-mismatched keys (adapted): {len(size_mismatched_keys)}")
        if size_mismatched_keys:
            for key in size_mismatched_keys:
                print(f"  - {key}")
        print(f"Missing keys: {len(missing_keys)}")
        if missing_keys:
            for key in missing_keys[:5]:  # 只显示前5个
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        if unexpected_keys:
            for key in unexpected_keys[:5]:
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")

        return {
            'adapted_keys': list(adapted_state_dict.keys()),
            'size_mismatched_keys': size_mismatched_keys,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'load_result': load_result
        }


class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            # 1.3B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            # 14B PAI control
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        elif hash_state_dict_keys(state_dict) == "70ddad9d3a133785da5ea371aae09504":
            # 1.3B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "26bde73488a92e64cc20b0a7485b9e5b":
            # 14B PAI control v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True
            }
        elif hash_state_dict_keys(state_dict) == "ac6a5aa74f4a0aab6f64eb9a72f19901":
            # 1.3B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "b61c605c2adbd23124d152ed28e049ae":
            # 14B PAI control-camera v1.1
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 32,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        elif hash_state_dict_keys(state_dict) == "1f5ab7703c6fc803fdded85ff040c316":
            # Wan-AI/Wan2.2-TI2V-5B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
            }
        elif hash_state_dict_keys(state_dict) == "5b013604280dd715f8457c6ed6d6a626":
            # Wan-AI/Wan2.2-I2V-A14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "2267d489f0ceb9f21836532952852ee5":
            # Wan2.2-Fun-A14B-Control
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 52,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": True,
                "require_clip_embedding": False,
            }
        elif hash_state_dict_keys(state_dict) == "47dbeab5e560db3180adf51dc0232fb1":
            # Wan2.2-Fun-A14B-Control-Camera
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_ref_conv": False,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "require_clip_embedding": False,
            }
        else:
            config = {}
        return state_dict, config
