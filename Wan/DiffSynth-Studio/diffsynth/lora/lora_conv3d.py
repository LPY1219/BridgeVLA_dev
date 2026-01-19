"""
LoRA加载器 - 支持Conv3d (5D权重)
Extended LoRA loader with Conv3d support for patch_embedding layers
"""

import torch
from typing import Dict, List, Tuple


class LoRALoaderConv3D:
    """LoRA加载器，支持2D/4D/5D权重"""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype

    def get_name_dict(self, state_dict_lora: Dict) -> Dict[str, Tuple[str, str]]:
        """
        从state_dict中提取LoRA权重对的映射

        Returns:
            Dict[module_name, (lora_A_key, lora_B_key)]
        """
        lora_name_dict = {}
        for key in state_dict_lora.keys():
            if "lora_A" in key:
                # 提取模块名称: "blocks.0.cross_attn.k.lora_A.default.weight" -> "blocks.0.cross_attn.k"
                name = key.replace(".lora_A.default.weight", "")
                name = name.replace(".lora_A.weight", "")

                # 查找对应的lora_B键
                lora_A_key = key
                lora_B_key = key.replace("lora_A", "lora_B")

                if lora_B_key in state_dict_lora:
                    lora_name_dict[name] = (lora_A_key, lora_B_key)

        return lora_name_dict

    def load(self, model: torch.nn.Module, state_dict_lora: Dict, alpha: float = 1.0):
        """
        加载LoRA权重到模型

        Args:
            model: 目标模型
            state_dict_lora: LoRA权重字典
            alpha: LoRA缩放因子
        """
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)

        print(f"Found {len(lora_name_dict)} LoRA modules to load")

        for name, module in model.named_modules():
            if name in lora_name_dict:
                lora_A_key, lora_B_key = lora_name_dict[name]
                weight_up = state_dict_lora[lora_B_key].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_A_key].to(device=self.device, dtype=self.torch_dtype)

                # 根据权重维度处理不同类型的层
                ndim = len(weight_up.shape)

                if ndim == 5:
                    # Conv3d: [out_channels, rank, D, H, W] @ [rank, in_channels, D, H, W]
                    weight_lora = self._compute_lora_conv3d(weight_up, weight_down, alpha)
                    print(f"  Loaded Conv3d LoRA for {name}: {weight_lora.shape}")

                elif ndim == 4:
                    # Conv2d: [out_channels, rank, H, W] @ [rank, in_channels, H, W]
                    weight_lora = self._compute_lora_conv2d(weight_up, weight_down, alpha)

                elif ndim == 2:
                    # Linear: [out_features, rank] @ [rank, in_features]
                    weight_lora = self._compute_lora_linear(weight_up, weight_down, alpha)

                else:
                    print(f"  Warning: Unsupported LoRA weight dimension {ndim}D for {name}")
                    continue

                # 加载权重到模块
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1

        print(f"✓ {updated_num} tensors updated by LoRA")
        return updated_num

    def _compute_lora_conv3d(self, weight_up, weight_down, alpha):
        """
        计算Conv3d的LoRA权重

        weight_up (lora_B): [out_channels, rank, D, H, W]
        weight_down (lora_A): [rank, in_channels, D, H, W]

        Returns: [out_channels, in_channels, D, H, W]
        """
        # 获取维度信息
        out_channels, rank, D, H, W = weight_up.shape
        _, in_channels, _, _, _ = weight_down.shape

        # 使用einsum计算: out[o,i,d,h,w] = sum_r( up[o,r,d,h,w] * down[r,i,d,h,w] )
        # 这里假设kernel size相同，直接逐元素相乘后求和
        weight_lora = alpha * torch.einsum('ordhw,ridhw->oidhw', weight_up, weight_down)

        return weight_lora

    def _compute_lora_conv2d(self, weight_up, weight_down, alpha):
        """
        计算Conv2d的LoRA权重

        weight_up: [out_channels, rank, H, W]
        weight_down: [rank, in_channels, H, W]

        Returns: [out_channels, in_channels, H, W]
        """
        # Squeeze spatial dimensions for matrix multiplication
        weight_up_2d = weight_up.squeeze(3).squeeze(2)  # [out_channels, rank]
        weight_down_2d = weight_down.squeeze(3).squeeze(2)  # [rank, in_channels]

        # Matrix multiply and restore spatial dimensions
        weight_lora = alpha * torch.mm(weight_up_2d, weight_down_2d).unsqueeze(2).unsqueeze(3)

        return weight_lora

    def _compute_lora_linear(self, weight_up, weight_down, alpha):
        """
        计算Linear层的LoRA权重

        weight_up: [out_features, rank]
        weight_down: [rank, in_features]

        Returns: [out_features, in_features]
        """
        weight_lora = alpha * torch.mm(weight_up, weight_down)
        return weight_lora


def load_lora_with_conv3d(model: torch.nn.Module,
                          lora_path: str,
                          alpha: float = 1.0,
                          device: str = "cuda",
                          torch_dtype = torch.float16):
    """
    便捷函数：加载LoRA权重（支持Conv3d）

    Args:
        model: 目标模型
        lora_path: LoRA checkpoint路径
        alpha: LoRA缩放因子
        device: 设备
        torch_dtype: 数据类型

    Returns:
        加载的参数数量
    """
    from diffsynth import load_state_dict

    # 加载checkpoint
    state_dict_lora = load_state_dict(lora_path)

    # 创建加载器并加载
    loader = LoRALoaderConv3D(device=device, torch_dtype=torch_dtype)
    num_updated = loader.load(model, state_dict_lora, alpha=alpha)

    return num_updated
