"""
Wan Video Pipeline Loader

这个模块负责加载和初始化Wan视频扩散模型的pipeline。
所有与pipeline相关的代码都在这里，便于后续更换pipeline时只需修改这个文件。

参考: heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed.py
"""

import torch
import torch.nn as nn
from typing import Optional

from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_rot_grip import (
    WanVideoPipeline,
    ModelConfig
)


def _initialize_mv_modules(pipe, device: str, torch_dtype: torch.dtype):
    """
    初始化多视角模块

    这是Wan多视角pipeline所需的关键初始化步骤。
    必须在加载LoRA之前调用。

    Args:
        pipe: WanVideoPipeline实例
        device: 设备
        torch_dtype: 数据类型
    """
    from diffsynth.models.wan_video_dit_mv import SelfAttention

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]

    for block in pipe.dit.blocks:
        # 初始化projector
        block.projector = nn.Linear(dim, dim).to(device=device, dtype=torch_dtype)
        block.projector.weight = nn.Parameter(torch.zeros(dim, dim, device=device, dtype=torch_dtype))
        block.projector.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=torch_dtype))

        # 初始化norm_mvs
        block.norm_mvs = nn.LayerNorm(
            dim,
            eps=block.norm1.eps,
            elementwise_affine=False
        ).to(device=device, dtype=torch_dtype)

        # 初始化modulation_mvs
        block.modulation_mvs = nn.Parameter(
            torch.randn(1, 3, dim, device=device, dtype=torch_dtype) / dim**0.5
        )

        # 初始化mvs_attn
        block.mvs_attn = SelfAttention(
            dim,
            block.self_attn.num_heads,
            block.self_attn.norm_q.eps
        ).to(device=device, dtype=torch_dtype)

        # 复制初始权重
        block.modulation_mvs.data = block.modulation.data[:, :3, :].clone()
        block.mvs_attn.load_state_dict(block.self_attn.state_dict(), strict=True)

    print("✓ Multi-view modules initialized")


def load_wan_pipeline(
    lora_checkpoint_path: str,
    model_base_path: str,
    wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
    use_dual_head: bool = True,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> WanVideoPipeline:
    """
    加载WanVideoPipeline并应用LoRA权重

    这是加载视频扩散模型的统一接口。
    如果要更换pipeline，只需修改这个文件中的实现。

    关键步骤:
    1. 加载基础pipeline（DiT + VAE + T5）
    2. 初始化多视角模块
    3. 加载LoRA权重

    Args:
        lora_checkpoint_path: LoRA checkpoint路径
        model_base_path: 基础模型路径（包含DiT、VAE、T5）
        wan_type: Wan模型类型
        use_dual_head: 是否使用dual head模式
        device: 设备
        torch_dtype: 数据类型

    Returns:
        加载好的WanVideoPipeline实例
    """
    print("=" * 80)
    print(f"Loading Wan Video Pipeline ({wan_type})...")
    print("=" * 80)
    print(f"  Model base: {model_base_path}")
    print(f"  LoRA checkpoint: {lora_checkpoint_path}")
    print(f"  Use dual head: {use_dual_head}")

    # 步骤1: 加载基础pipeline
    print("\nStep 1: Loading base pipeline (DiT + VAE + T5)...")
    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        wan_type=wan_type,
        use_dual_head=use_dual_head,
        model_configs=[
            ModelConfig(path=[
                f"{model_base_path}/diffusion_pytorch_model-00001-of-00003.safetensors",
                f"{model_base_path}/diffusion_pytorch_model-00002-of-00003.safetensors",
                f"{model_base_path}/diffusion_pytorch_model-00003-of-00003.safetensors"
            ]),
            ModelConfig(path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{model_base_path}/Wan2.2_VAE.pth"),
        ],
    )
    print("✓ Base pipeline loaded")

    # 步骤2: 初始化多视角模块
    print("\nStep 2: Initializing multi-view modules...")
    _initialize_mv_modules(pipeline, device, torch_dtype)

    # 步骤3: 加载LoRA权重
    print("\nStep 3: Loading LoRA weights...")
    pipeline.load_lora(pipeline.dit, lora_checkpoint_path, alpha=1.0)
    print("✓ LoRA weights loaded")

    print("\n" + "=" * 80)
    print("✓ Pipeline loaded successfully!")
    print("=" * 80)

    return pipeline


# 如果未来需要支持其他pipeline（如Hunyuan、CogVideo等），可以在此添加对应的加载函数：
#
# def load_hunyuan_pipeline(...)
# def load_cogvideo_pipeline(...)
#
# 然后在主训练脚本中根据配置选择使用哪个加载函数
