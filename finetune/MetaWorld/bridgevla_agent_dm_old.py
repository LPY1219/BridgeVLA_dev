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
from scipy.spatial.transform import Rotation
import bridgevla.mvt.utils as mvt_utils


# 自动检测根路径
def get_root_path():
    """自动检测BridgeVLA根目录"""
    possible_paths = [
        "/share/project/lpy/BridgeVLA",
        "/DATA/disk1/lpy/BridgeVLA_dev",
        "/home/lpy/BridgeVLA_dev",
        "/mnt/data/cyx/workspace/BridgeVLA_dev",
        "/DATA/disk1/cyx/BridgeVLA_dev",
        "/DATA/disk0/lpy/cyx/BridgeVLA_dev",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")

ROOT_PATH = get_root_path()
print(f"Using ROOT_PATH: {ROOT_PATH}")

from utils.setup_paths import setup_project_paths
setup_project_paths()


# 导入多视角pipeline (支持旋转和夹爪预测)
from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_rot_grip import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap
# 导入旋转和夹爪预测模型（使用 delta 版本）
# 两个模型的接口完全相同，都不需要初始状态作为输入，只预测 delta 值
from examples.wanvideo.model_training.mv_rot_grip_vae_decode_feature_3_metaworld import MultiViewRotationGripperPredictor

import torch.nn as nn
import torch.nn.functional as F

def rgb_to_pil_image(rgb_array: torch.Tensor) -> Image.Image:
    """
    将RGB tensor转换为PIL图像

    Args:
        rgb_tensor: RGB tensor (3, H, W) 或 (H, W, 3)
                   支持范围：[-1, 1] 或 [0, 1] 或 [0, 255]

    Returns:
        PIL图像
    """
    # 转换为numpy

    # 处理通道顺序
    if rgb_array.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        rgb_array = rgb_array.transpose(1, 2, 0)

    # 智能检测数值范围并归一化到 [0, 255]
    min_val = rgb_array.min()
    max_val = rgb_array.max()

    if min_val < 0:  # 范围是 [-1, 1]
        # 从 [-1, 1] 转到 [0, 1]
        rgb_array = (rgb_array + 1.0) / 2.0
        rgb_array = np.clip(rgb_array, 0, 1)
        rgb_array = (rgb_array * 255).astype(np.uint8)
    elif max_val <= 1.0:  # 范围是 [0, 1]
        rgb_array = (rgb_array * 255).astype(np.uint8)
    else:  # 范围是 [0, 255]
        rgb_array = rgb_array.astype(np.uint8)

    return Image.fromarray(rgb_array)


    # 辅助函数：将四元数转换为欧拉角
def quaternion_to_discrete_euler(quat):
        """将单个四元数转换为离散化的欧拉角索引"""
        # 归一化四元数
        quat_normalized = quat / np.linalg.norm(quat)

        # 确保w为正数
        if quat_normalized[3] < 0:
            quat_normalized = -quat_normalized

        # 使用scipy的Rotation转换（scipy使用[x, y, z, w]顺序）
        r = Rotation.from_quat(quat_normalized)
        euler = r.as_euler("xyz", degrees=True)  # (3,) - [roll, pitch, yaw]

        # 应用gimble fix
        if 89 < euler[1] < 91:
            euler[1] = 90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)
        elif -91 < euler[1] < -89:
            euler[1] = -90
            r = Rotation.from_euler("xyz", euler, degrees=True)
            euler = r.as_euler("xyz", degrees=True)

        # 将范围从[-180, 180]转换为[0, 360]
        euler += 180
        return euler

def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat

def _norm_rgb(x):
    if isinstance(x, np.ndarray):
        # 处理负步长问题
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
    return (x.float() / 255.0) * 2.0 - 1.0

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
                        f"{model_base_path}/diffusion_pytorch_model-00001-of-00003.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00002-of-00003.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00003-of-00003.safetensors"
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

        # 初始化旋转和夹爪预测器（使用新的基于VAE decoder中间特征的模型）
        print("Initializing rotation and gripper predictor (VAE decode feature version)...")
        self.rot_grip_predictor = MultiViewRotationGripperPredictor(
            rgb_channels=256,  # VAE decoder intermediate channels
            heatmap_channels=256,  # VAE decoder intermediate channels
            hidden_dim=hidden_dim,
            num_views=3,
            num_rotation_bins=num_rotation_bins,
            dropout=0.1,
            local_feature_size=5,  # 局部特征提取的邻域大小
        ).to(device=device, dtype=torch_dtype)

        # 加载支持decode_intermediate的VAE
        print("Loading VAE with decode_intermediate support...")
        from diffsynth.models.wan_video_vae_2 import WanVideoVAE38
        self.vae_decode_intermediate = WanVideoVAE38()
        # 加载权重
        vae_state_dict = torch.load(f"{model_base_path}/Wan2.2_VAE.pth", map_location="cpu")
        # 处理state_dict格式
        if 'model_state' in vae_state_dict:
            vae_state_dict = vae_state_dict['model_state']
        # 添加'model.'前缀
        vae_state_dict = {'model.' + k: v for k, v in vae_state_dict.items()}
        self.vae_decode_intermediate.load_state_dict(vae_state_dict, strict=True)
        self.vae_decode_intermediate = self.vae_decode_intermediate.eval().to(device=device, dtype=torch_dtype)
        print("✓ VAE with decode_intermediate loaded")

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

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 加载权重
        self.rot_grip_predictor.load_state_dict(state_dict, strict=True)
        self.rot_grip_predictor.eval()

        epoch_info = f" (epoch {checkpoint['epoch']})" if 'epoch' in checkpoint else ""
        print(f"✓ Loaded rotation/gripper predictor{epoch_info}")

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
        save_path = os.path.join("/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img/debug_input.png")
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
                visualize_save_path: str ="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img/debug_output.png" ,
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

        # 2. 使用VAE decode_intermediate获取中间特征（新模型需要256通道的decoder中间特征）
        num_views = rgb_latents.shape[0]

        # 对每个视角分别decode_intermediate
        all_rgb_intermediate = []
        all_heatmap_intermediate = []
        all_heatmap_images = []

        for view_idx in range(num_views):
            view_rgb_latent = rgb_latents[view_idx:view_idx+1]  # (1, c, t, h, w)
            view_heatmap_latent = heatmap_latents[view_idx:view_idx+1]

            # RGB: 只需要中间特征
            rgb_intermediate = self.vae_decode_intermediate.decode_intermediate(
                [view_rgb_latent.squeeze(0)], device=self.device
            )
            all_rgb_intermediate.append(rgb_intermediate[0])

            # Heatmap: 需要中间特征和完全解码的图像（用于找峰值）
            heatmap_intermediate, heatmap_full = self.vae_decode_intermediate.decode_intermediate_with_full(
                [view_heatmap_latent.squeeze(0)], device=self.device
            )
            all_heatmap_intermediate.append(heatmap_intermediate[0])
            all_heatmap_images.append(heatmap_full[0])

        # 合并所有视角: List[(c, t, h, w)] -> (num_views, c, t, h, w)
        rgb_features = torch.stack(all_rgb_intermediate, dim=0)
        heatmap_features = torch.stack(all_heatmap_intermediate, dim=0)
        heatmap_images = torch.stack(all_heatmap_images, dim=0)  # (num_views, 3, t, H, W)

        # 添加batch维度: (num_views, c, t, h, w) -> (1, num_views, c, t, h, w)
        rgb_features = rgb_features.unsqueeze(0)
        heatmap_features = heatmap_features.unsqueeze(0)
        heatmap_images = heatmap_images.unsqueeze(0)  # (1, num_views, 3, t, H, W)

        # 3. 预测旋转和夹爪状态（新模型不需要initial_rotation和initial_gripper输入）
        self.rot_grip_predictor.eval()

        num_future_frames = num_frames - 1
        rotation_logits, gripper_logits = self.rot_grip_predictor(
            rgb_features=rgb_features,
            heatmap_features=heatmap_features,
            num_future_frames=num_future_frames,
            heatmap_images=heatmap_images,
            colormap_name='jet',
        )

        # 4. 解码预测结果（预测的是delta值）
        # rotation_logits: (1, T-1, num_bins*3) -> (T-1, 3, num_bins)
        rotation_logits = rotation_logits[0].view(num_future_frames, 3, self.num_rotation_bins)

        # gripper_logits: (1, T-1, 2) -> (T-1, 2)
        gripper_logits = gripper_logits[0]

        # 获取预测的离散索引（delta值）
        rotation_delta_bins = rotation_logits.argmax(dim=-1)  # (T-1, 3)
        gripper_predictions = gripper_logits.argmax(dim=1)  # (T-1,) - 21class [-1, 1]

        gripper_predictions = gripper_predictions.float() * 0.1 - 1.0
        gripper_predictions = gripper_predictions.cpu().numpy()
        # print(f"gripper_predictions: {gripper_predictions}")

        # 5. 将delta值转换为绝对值
        # rotation delta: bins表示变化量，需要加到初始值上
        rotation_delta_degrees = self._delta_bins_to_degrees(rotation_delta_bins.cpu().numpy())  # (T-1, 3)

        # 注意：训练时delta是基于连续的initial_rotation计算的，所以推理时也应该直接使用连续值
        # 不要对initial_rotation进行离散化再转回来，这会引入不必要的量化误差
        initial_rotation_degrees = initial_rotation  # 直接使用连续值 (3,)

        # 计算累积rotation: 每帧相对于第一帧的rotation
        rotation_predictions = initial_rotation_degrees + rotation_delta_degrees  # (T-1, 3)
        # 归一化到 [-180, 180]
        rotation_predictions = ((rotation_predictions + 180) % 360) - 180



        # 6. 可视化生成的视频帧
        if visualize:
            self._visualize_generated_frames(
                video_heatmap_frames,
                video_rgb_frames,
                save_path=visualize_save_path
            )
        # a=input("begin to send the predicted action?")
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

    def _delta_bins_to_degrees(self, delta_bins: np.ndarray) -> np.ndarray:
        """
        将delta bins转换回角度变化量

        新模型预测的是相对于第一帧的rotation变化量（delta）
        delta bins的范围是 [0, num_rotation_bins)，其中：
        - bin 0 表示 -180度变化
        - bin num_rotation_bins//2 表示 0度变化（无变化）
        - bin num_rotation_bins-1 表示接近 +180度变化

        Args:
            delta_bins: (T, 3) - delta bin indices [0, num_rotation_bins)

        Returns:
            delta_degrees: (T, 3) - rotation change in degrees [-180, 180)
        """
        # delta bins 的含义与绝对值bins相同，都是映射到 [-180, 180] 范围
        # bin 0 -> -180度, bin num_bins//2 -> 0度, bin num_bins-1 -> 175度
        delta_degrees = delta_bins * self.rotation_resolution - 180

        return delta_degrees

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

        新模型使用VAE decoder的中间特征，预测的是delta值

        Args:
            gt_rgb_video: List[List[PIL.Image]] - [time][view] GT RGB视频
            gt_heatmap_video: List[List[PIL.Image]] - [time][view] GT Heatmap视频
            initial_rotation: (3,) - 初始旋转角度 [roll, pitch, yaw] in degrees
            initial_gripper: int - 初始夹爪状态
            heatmap_latent_scale: float - heatmap latent缩放因子

        Returns:
            Dict包含:
                - rotation_predictions: (T-1, 3) 旋转预测（度）- 绝对值
                - gripper_predictions: (T-1,) 夹爪预测 - 绝对状态
        """
        # 1. 编码GT视频为VAE latents
        rgb_latents, heatmap_latents = self.encode_gt_videos(gt_rgb_video, gt_heatmap_video)

        # 应用heatmap缩放
        if heatmap_latent_scale != 1.0:
            heatmap_latents = heatmap_latents * heatmap_latent_scale

        # 2. 使用VAE decode_intermediate获取中间特征
        num_views = rgb_latents.shape[0]

        all_rgb_intermediate = []
        all_heatmap_intermediate = []
        all_heatmap_images = []

        for view_idx in range(num_views):
            view_rgb_latent = rgb_latents[view_idx:view_idx+1]
            view_heatmap_latent = heatmap_latents[view_idx:view_idx+1]

            # RGB: 只需要中间特征
            rgb_intermediate = self.vae_decode_intermediate.decode_intermediate(
                [view_rgb_latent.squeeze(0)], device=self.device
            )
            all_rgb_intermediate.append(rgb_intermediate[0])

            # Heatmap: 需要中间特征和完全解码的图像（用于找峰值）
            heatmap_intermediate, heatmap_full = self.vae_decode_intermediate.decode_intermediate_with_full(
                [view_heatmap_latent.squeeze(0)], device=self.device
            )
            all_heatmap_intermediate.append(heatmap_intermediate[0])
            all_heatmap_images.append(heatmap_full[0])

        # 合并所有视角
        rgb_features = torch.stack(all_rgb_intermediate, dim=0)
        heatmap_features = torch.stack(all_heatmap_intermediate, dim=0)
        heatmap_images = torch.stack(all_heatmap_images, dim=0)

        # 添加batch维度
        rgb_features = rgb_features.unsqueeze(0)  # (1, v, c, t, h, w)
        heatmap_features = heatmap_features.unsqueeze(0)
        heatmap_images = heatmap_images.unsqueeze(0)

        # 3. 计算future frames数量
        num_frames = len(gt_rgb_video)
        num_future_frames = num_frames - 1

        # 4. 使用旋转预测器预测（新模型不需要initial_rotation和initial_gripper输入）
        rotation_logits, gripper_logits = self.rot_grip_predictor(
            rgb_features=rgb_features,
            heatmap_features=heatmap_features,
            num_future_frames=num_future_frames,
            heatmap_images=heatmap_images,
            colormap_name='jet',
        )

        # 5. 转换logits为预测结果（预测的是delta值）
        rotation_logits = rotation_logits.squeeze(0)  # (T-1, num_bins*3)
        rotation_logits = rotation_logits.view(num_future_frames, 3, self.num_rotation_bins)  # (T-1, 3, num_bins)

        rotation_delta_bins = rotation_logits.argmax(dim=-1)  # (T-1, 3)

        # Gripper: (1, T-1, 2) -> (T-1,) - 预测的是变化
        gripper_logits = gripper_logits.squeeze(0)  # (T-1, 2)
        gripper_change = gripper_logits.argmax(dim=-1)  # (T-1,) - 0=不变, 1=变化

        # 6. 将delta值转换为绝对值
        rotation_delta_bins_np = rotation_delta_bins.float().cpu().numpy()

        # rotation delta: 转换为角度变化量
        rotation_delta_degrees = self._delta_bins_to_degrees(rotation_delta_bins_np)  # (T-1, 3)

        # 注意：训练时delta是基于连续的initial_rotation计算的，所以推理时也应该直接使用连续值
        # 不要对initial_rotation进行离散化再转回来，这会引入不必要的量化误差
        initial_rotation_degrees = initial_rotation  # 直接使用连续值

        # 计算累积rotation
        rotation_predictions = initial_rotation_degrees + rotation_delta_degrees
        # 归一化到 [-180, 180]
        rotation_predictions = ((rotation_predictions + 180) % 360) - 180

        # gripper: 根据变化标志计算实际状态
        # gripper_change[t] 表示第 t 帧是否与第一帧不同（不是累积的翻转信号）
        gripper_predictions = np.zeros(num_future_frames, dtype=np.int64)
        for t in range(num_future_frames):
            if gripper_change[t].item() == 1:  # 与第一帧不同
                gripper_predictions[t] = 1 - initial_gripper  # 使用相反的状态
            else:  # 与第一帧相同
                gripper_predictions[t] = initial_gripper  # 使用相同的状态

        return {
            'rotation_predictions': rotation_predictions,
            'gripper_predictions': gripper_predictions,
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
    # 步骤0: 检测并转换输入格式 [num_views][T] -> [T][num_views]
    # 通过检查第一个维度的长度来判断格式（假设视角数 <= 3，时间步数 > 3）
    if len(pred_heatmap_colormap[0]) > 3:
        # 格式是 [num_views][T]，需要转换为 [T][num_views]
        num_views = len(pred_heatmap_colormap)
        num_frames = len(pred_heatmap_colormap[0])
        pred_heatmap_colormap = [[pred_heatmap_colormap[v][t] for v in range(num_views)] for t in range(num_frames)]

    # 步骤1: 将colormap转换为heatmap数组 List[List[np.ndarray]] (T, num_views, H, W)
    pred_heatmap_arrays = convert_colormap_to_heatmap(pred_heatmap_colormap, colormap_name)

    # 步骤2: 将 List[List[np.ndarray]] 转换为张量 (T, num_views, H*W)
    num_frames = len(pred_heatmap_arrays)
    num_views = len(pred_heatmap_arrays[0])

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

    # 步骤3: 从heatmap中提取3D位置
    get_position_from_heatmap = projection_interface.get_position_from_heatmap
    pred_position = get_position_from_heatmap(heatmap_tensor, rev_trans)

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

class ProjectionInterface:
    """
    点云投影接口 - 提供默认实现，用户可以继承并重写
    """
    def __init__(self,
                img_size=256,
                rend_three_views=True,
                add_depth=False,
                ):

        from point_renderer.rvt_renderer import RVTBoxRenderer
        import os
        # 使用LOCAL_RANK环境变量确定当前进程的GPU（分布式训练支持）
        # 这样每个进程的渲染器会使用自己对应的GPU，避免显存不均匀
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.renderer_device = f"cuda:{local_rank}"
            print(f"[ProjectionInterface] Using device: {self.renderer_device} (LOCAL_RANK={local_rank})")
        else:
            self.renderer_device = "cpu"
            print(f"[ProjectionInterface] Using device: cpu")
        self.renderer = RVTBoxRenderer(
            device=self.renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )
        self.img_size = (img_size, img_size)


    def project_pointcloud_to_rgb(self, pointcloud: np.ndarray, feat: np.ndarray, img_aug_before=0.1, img_aug_after=0.05) -> np.ndarray:
        """
        将点云投影到指定视角生成RGB图像

        Args:
            pointcloud: 点云数据 (N, 3) 
            feat: 颜色数据 (N, 3) 

        Returns:
            RGB图像 (N, H, W, 3) 范围[0, 1] N 表示有多少个视角
        """
        # aug before projection
        if img_aug_before !=0:
            stdv = img_aug_before * torch.rand(1, device=feat.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*feat.shape, device=feat.device)) - 1)
            feat = feat + noise
            # 裁剪到 [0, 1] 范围，确保增强后的特征仍然有效
            feat = torch.clamp(feat, 0, 1)

        # 确保数据在正确的设备上
        renderer_device = self.renderer_device
        if hasattr(pointcloud, 'device') and str(pointcloud.device) != str(renderer_device):
            pointcloud = pointcloud.to(renderer_device)
        if hasattr(feat, 'device') and str(feat.device) != str(renderer_device):
            feat = feat.to(renderer_device)

        max_pc = 1.0 if len(pointcloud) == 0 else torch.max(torch.abs(pointcloud))

        img= self.renderer(
                pointcloud,
                torch.cat((pointcloud / max_pc, feat), dim=-1),
                fix_cam=True,
                dyn_cam_info=None
            ).unsqueeze(0)

        # aug after projection  由于增强后范围可能不在0，1之间，所以去掉
        # if img_aug_after != 0:
        #     stdv = img_aug_after * torch.rand(1, device=img.device)
        #     # values in [-stdv, stdv]
        #     noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
        #     img = torch.clamp(img + noise, -1, 1)
        return img


    def project_pose_to_pixel(self, poses: np.ndarray) -> Tuple[int, int]:
        """
        将三维空间中的路径点坐标转换为图像坐标系下的坐标
        :param poses: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """

        pt_img = self.renderer.get_pt_loc_on_img(
            poses, fix_cam=True, dyn_cam_info=None
        )

        # 裁剪像素坐标到图像边界内，防止超出scene_bounds的pose导致无效坐标
        # pt_img shape: (bs, np, num_img, 2), 最后一维是 (x, y)
        h, w = self.img_size
        pt_img[..., 0] = torch.clamp(pt_img[..., 0], min=0, max=w-1)  # x 坐标裁剪到 [0, w-1]
        pt_img[..., 1] = torch.clamp(pt_img[..., 1], min=0, max=h-1)  # y 坐标裁剪到 [0, h-1]

        return pt_img

    def generate_heatmap_from_img_locations(self,
        img_locations,
        width=256, height=256, sigma=1.5):

        # (bs, np, num_img, 2)
        bs, np, num_img, _= img_locations.shape

        action_trans = mvt_utils.generate_hm_from_pt(
            img_locations.reshape(-1, 2),
            (self.img_size[0], self.img_size[1]),
            sigma=sigma,
            thres_sigma_times=3,
        )
        heatmap_sequence=action_trans.view(bs,np,num_img,height,width)
        return heatmap_sequence

    def visualize_hm(self, heatmaps, h, w, save_path=None):
        """
        可视化多视角heatmap序列并保存到指定路径

        Args:
            heatmaps: torch.Tensor (T, num_views, h*w) - heatmap张量
            h: int - heatmap高度
            w: int - heatmap宽度
            save_path: str - 保存图像的路径，如果为None则不保存

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 将heatmap reshape为 (T, num_views, h, w)
        T, num_views, hw = heatmaps.shape
        assert hw == h * w, f"Expected h*w={h*w}, got {hw}"

        # Reshape heatmaps
        heatmaps_reshaped = heatmaps.view(T, num_views, h, w)

        # 转换为numpy并归一化
        if torch.is_tensor(heatmaps_reshaped):
            heatmaps_np = heatmaps_reshaped.detach().cpu().numpy()
        else:
            heatmaps_np = heatmaps_reshaped

        # 对每个heatmap进行归一化到[0,1]
        heatmaps_normalized = []
        for t in range(T):
            frame_views = []
            for v in range(num_views):
                hm = heatmaps_np[t, v]
                # 归一化到[0, 1]
                hm_min = hm.min()
                hm_max = hm.max()
                if hm_max > hm_min:
                    hm_norm = (hm - hm_min) / (hm_max - hm_min)
                else:
                    hm_norm = hm
                frame_views.append(hm_norm)
            heatmaps_normalized.append(frame_views)

        # 创建可视化图形: 行=时间步，列=视角
        fig, axes = plt.subplots(T, num_views, figsize=(num_views * 3, T * 2.5))

        # 处理单行或单列的情况
        if T == 1 and num_views == 1:
            axes = np.array([[axes]])
        elif T == 1:
            axes = axes.reshape(1, -1)
        elif num_views == 1:
            axes = axes.reshape(-1, 1)

        # 绘制每个heatmap
        for t in range(T):
            for v in range(num_views):
                ax = axes[t, v]
                hm = heatmaps_normalized[t][v]

                # 使用jet colormap显示heatmap
                im = ax.imshow(hm, cmap='jet', interpolation='nearest')

                # 添加标题
                if t == 0:
                    ax.set_title(f'View {v}', fontsize=10, fontweight='bold')
                if v == 0:
                    ax.set_ylabel(f'T{t}', fontsize=9, fontweight='bold')

                # 找到最大值位置并标记
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                ax.plot(max_idx[1], max_idx[0], 'r+', markersize=8, markeredgewidth=2)

                # 移除坐标轴刻度
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.suptitle(f'Multi-View Heatmap Sequence (T={T}, Views={num_views})',
                    fontsize=12, fontweight='bold', y=0.995)

        # 保存图像
        if save_path is not None:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap visualization saved to: {save_path}")

        plt.close(fig)

    def get_position_from_heatmap(self, heatmaps,rev_trans,dyn_cam_info=None, y_q=None,visualize=False, use_softmax=True):
        """
        Estimate the q-values given output from mvt
        :param heatmap: heatmaps output from wan  (bs,view,h*w)
        :param rev_trans  逆变换函数
        :param use_softmax: 是否使用softmax归一化（默认True保持兼容性）
        """
        h ,w = self.img_size
        bs,nc,h_w=heatmaps.shape
        # if visualize:
        #     self.visualize_hm(heatmaps, h, w,save_path="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img/debug.png")

        if use_softmax:
            hm = torch.nn.functional.softmax(heatmaps, 2)
        else:
            # 简单归一化，保持原始分布
            hm = heatmaps / (heatmaps.sum(dim=2, keepdim=True) + 1e-8)
        hm = hm.view(bs, nc, h, w)
        hm=  hm.to(self.renderer_device)
        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        pred_wpt = pred_wpt.squeeze(1)
        pred_wpt = rev_trans(pred_wpt.to("cpu"))

        assert y_q is None

        return pred_wpt

class RVTAgent:
    def __init__(self, args):
        self.args = args
        self.projection_interface = ProjectionInterface(args.img_size[0])
        
        self.inferencer = HeatmapInferenceMVRotGrip(
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
        
        self.scene_bounds = args.scene_bounds
        self.use_merged_pointcloud = args.use_merged_pointcloud
        
    
    def preprocess(self, pcd_list, feat_list, all_poses: np.ndarray):
        """
        预处理点云序列、特征序列和姿态（3相机版本）

        Args:
            pcd_list: 点云列表的列表，每个元素为 [pcd_cam1, pcd_cam2, pcd_cam3]
            feat_list: 特征列表的列表，每个元素为 [feat_cam1, feat_cam2, feat_cam3]
            all_poses: 姿态数组 [num_poses, 7]
            trail_info: 轨迹信息，包含外参文件路径

        Returns:
            pc_list: 处理后的点云列表（拼接后的）
            img_feat_list: 处理后的特征列表（拼接后的）
            wpt_local: 局部坐标系下的姿态 [num_poses, 3]
            rot_grip: rotaion and grip (num_poses,3 )
        """
        # 确保输入是列表
        if not isinstance(pcd_list, list):
            pcd_list = [pcd_list]
        if not isinstance(feat_list, list):
            feat_list = [feat_list]

        num_frames = len(pcd_list)


        # 处理每一帧的3个相机数据
        merged_pcd_list = []
        merged_feat_list = []

        for frame_idx in range(num_frames):

            # 获取这一帧的所有相机的点云和特征
            frame_pcds = pcd_list[frame_idx]  # [pcd_cam1, pcd_cam2, pcd_cam3]
            frame_feats = feat_list[frame_idx]  # [feat_cam1, feat_cam2, feat_cam3]

            # 归一化RGB特征
            frame_feats_norm = [_norm_rgb(feat) for feat in frame_feats]

            all_pcds = []
            all_feats = []

            for pcd, feat in zip(frame_pcds, frame_feats_norm):
                # flatten point cloud
                pcd_flat = torch.as_tensor(
                    np.ascontiguousarray(pcd), dtype=torch.float32
                ).view(-1, 3)

                # flatten RGB features & 归一化到 [0,1]
                feat_flat = ((feat.view(-1, 3) + 1) / 2).float()

                all_pcds.append(pcd_flat)
                all_feats.append(feat_flat)

            # 根据配置决定是否合并
            if self.use_merged_pointcloud:
                merged_pcd = torch.cat(all_pcds, dim=0)
                merged_feat = torch.cat(all_feats, dim=0)
            else:
                # 只使用第一个相机的数据（front）
                merged_pcd = all_pcds[0]
                merged_feat = all_feats[0]

            merged_pcd_list.append(merged_pcd)
            merged_feat_list.append(merged_feat)

        # 现在merged_pcd_list和merged_feat_list包含拼接后的点云和特征
        # 后续处理保持不变
        pc_list = merged_pcd_list  # 已经是展平的torch张量了
        img_feat_list = merged_feat_list  # 已经是归一化的torch张量了

        with torch.no_grad():

            action_trans_con = torch.from_numpy(np.array(all_poses)).float()[:, :3]
            action_rot_xyzw = torch.from_numpy(np.array(all_poses)).float()[:, 3:]  # [x,y,z,w]

            # 对每个点云应用边界约束
            processed_pc_list = []
            processed_feat_list = []
            for pc, img_feat in zip(pc_list, img_feat_list):
                pc, img_feat = move_pc_in_bound(
                    pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                )
                processed_pc_list.append(pc[0])
                processed_feat_list.append(img_feat[0])

            # 将点云和wpt放在一个cube里面 (使用第一个点云作为参考)
            wpt_local, rev_trans = mvt_utils.place_pc_in_cube( # 不会影响到旋转
                processed_pc_list[0],
                action_trans_con,
                with_mean_or_bounds=False,
                scene_bounds=self.scene_bounds,
            )

            # 对每个点云应用place_pc_in_cube
            final_pc_list = []
            for pc in processed_pc_list:
                pc = mvt_utils.place_pc_in_cube(
                    pc,
                    with_mean_or_bounds=False,
                    scene_bounds=self.scene_bounds,
                )[0]
                final_pc_list.append(pc)

        return final_pc_list, processed_feat_list, wpt_local, action_rot_xyzw, rev_trans


    def predict_action(self, observation: dict):
        observation = observation
        pcd = observation["point_cloud"][...].cpu().numpy()
        start_pcd, start_rgb = pcd[..., :3], pcd[..., 3:]

        start_xyz = observation["agent_pos"][:3].cpu().numpy()
        start_pose = np.concatenate([
                        start_xyz, 
                        np.array([0, 0, 0, 1])], 
                        axis=0
                    )
        start_gripper = 1.0
        start_rotation = quaternion_to_discrete_euler(start_pose[3:7])  # (3,)
        
        
        prompt = observation["instruction"]   

        start_rotation_degrees = self.inferencer._bins_to_degrees(start_rotation)
        processed_pcd_list, processed_rgb_list, processed_pos, processed_rot_xyzw, rev_trans = self.preprocess(
            [start_pcd], [start_rgb], [start_pose]
        )
        
        processed_start_pcd = processed_pcd_list[0]
        processed_start_rgb = processed_rgb_list[0]
        processed_poses=torch.cat((processed_pos, processed_rot_xyzw), dim=1) # num,7
        processed_start_pose = processed_poses[0]
        
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_start_pcd, processed_start_rgb
        )  # (1, num_views, H, W, 6)
        rgb_image = rgb_image[0, :, :, :, 3:]  # (num_views, H, W, 3)
        # 确保是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)  # (num_views, H, W, 3)
        num_views = rgb_image.shape[0]
        
        img_locations= self.projection_interface.project_pose_to_pixel(
                processed_pos.unsqueeze(0).to(self.projection_interface.renderer_device) 
        )  # (bs, num_poses, num_views, 2)

        # 使用用户提供的heatmap接口生成heatmap
        heatmap_sequence= self.projection_interface.generate_heatmap_from_img_locations(
            img_locations,
            self.args.img_size[0], self.args.img_size[1],
        ) # (bs, seq_len+1, num_views, H, W)
        heatmap_sequence = heatmap_sequence[0, :, :, :, :]  # (seq_len+1, num_views, H, W)
        heatmap_start = heatmap_sequence[0]  # (num_views, H, W)
        
        input_image = []
        input_image_rgb = []
        
        for v in range(num_views):
            rgb_view = rgb_image[v]  # (H, W, 3)
            pil_img = rgb_to_pil_image(rgb_view)
            input_image_rgb.append(pil_img)
            
        for v in range(num_views):
            heatmap_view = heatmap_start[v]  # (H, W)
            # 转换单个热力图为PIL（与client端一致的处理）
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化到[0, 1]
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap（使用JET colormap，与client端一致）
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_img = Image.fromarray(view_hm_colored)
            input_image.append(pil_img)
        
        # 执行推理
        # 使用 sequence_length 作为 num_frames 以匹配数据集
        output = self.inferencer.predict(
            prompt=prompt,
            input_image=input_image,
            input_image_rgb=input_image_rgb,
            initial_rotation=start_rotation_degrees,
            initial_gripper=start_gripper,
            num_frames=self.args.sequence_length+1, # 需要包括初始帧进来
            height=self.args.img_size[0],
            width=self.args.img_size[1],
            num_inference_steps=50,
            cfg_scale=1.0,
        )

        pred_heatmap = output['video_heatmap']

        # 从预测的heatmap提取3D位置
        pred_position = get_3d_position_from_pred_heatmap(
            pred_heatmap_colormap=pred_heatmap,
            rev_trans=rev_trans,
            projection_interface=self.projection_interface,
            colormap_name='jet'
        )  # (num_frames, 3)
        
        pred_gripper = output['gripper_predictions']  # (T-1,)
        if self.args.constant_gripper_num is not None:
            pred_gripper = np.full_like(pred_gripper, self.args.constant_gripper_num)
            # print(pred_gripper)
        
        pred_sequence_length = pred_position.shape[0]
        
        # delta_action_position = pred_position[1:] - pred_position[:-1]  # (T, 3)
        # action = np.concatenate([50*delta_action_position, pred_gripper[:, None]], axis=1)  # (T, 4)

        action = np.concatenate([pred_position[1:], pred_gripper[:, None]], axis=1)  # (T, 4)
        # action = action[:10] # only keep the first 10 actions
        return action