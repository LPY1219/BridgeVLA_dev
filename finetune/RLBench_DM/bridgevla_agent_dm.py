"""
Heatmap Inference Script for Wan2.2 with Rotation and Gripper Prediction (Multi-View Version)
用于多视角热力图序列预测 + 旋转和夹爪预测的推断脚本
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any, Optional
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import bridgevla.mvt.utils as mvt_utils
from yarr.agents.agent import ActResult


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


from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap
# 导入旋转和夹爪预测模型（View拼接版本）
# View concatenation version: uses latents directly (not decoder intermediate features)
# Direct latent input: rgb/heatmap_channels=48 (not 256), num_views=6 (not 3)
from examples.wanvideo.model_training.mv_rot_grip_vae_decode_feature_3_view_rlbench import MultiViewRotationGripperPredictorView
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


def quaternion_to_discrete_euler(quat, rotation_resolution):
        """将单个四元数转换为离散化的欧拉角索引"""
        from scipy.spatial.transform import Rotation

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

        # 离散化
        disc = np.around(euler / rotation_resolution).astype(np.int64)
        # 处理边界情况：360度 = 0度
        num_bins = int(360 / rotation_resolution)
        disc[disc == num_bins] = 0

        return disc  # (3,) - [roll_bin, pitch_bin, yaw_bin]
    
def quaternion_to_euler_continuous(quat):
        """将单个四元数转换为欧拉角（度）"""
        from scipy.spatial.transform import Rotation

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

        return euler  # (3,) - [roll, pitch, yaw] in degrees, range [-180, 180]

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
        model_base_path: str,
        lora_checkpoint: str,
        rot_grip_checkpoint: Optional[str] = None,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
        use_dual_head: bool = True,
        rotation_resolution: float = 5.0,
        hidden_dim: int = 512,
        num_rotation_bins: int = 72,
        num_history_frames: int = 1,
        local_feat_size: int = 5,
        use_initial_gripper_state: bool = False,
        device: str = "cuda",
        is_full_finetune: bool = False,
        torch_dtype=torch.bfloat16,
    ):
        """
        Initialize inference with rotation and gripper prediction.

        Args:
            model_base_path: Path to base model
            lora_checkpoint: Path to LoRA checkpoint for heatmap diffusion model
            rot_grip_checkpoint: Path to rotation/gripper predictor checkpoint (optional)
            wan_type: Model type (default: 5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP)
            use_dual_head: Whether to use dual head mode
            use_merged_pointcloud: Whether to use merged pointcloud from 3 cameras
            use_different_projection: Whether to use different projection mode
            rotation_resolution: Rotation angle resolution (degrees)
            hidden_dim: Hidden layer dimension for rot/grip predictor
            num_rotation_bins: Number of rotation bins
            num_history_frames: Number of history frames (1, 2, or 1+4N)
            local_feat_size: Local feature extraction neighborhood size
            use_initial_gripper_state: Whether to use initial gripper state as input
            device: Device to use
            is_full_finetune: Whether heatmap checkpoint is full finetune
            torch_dtype: Torch dtype
        """
        self.device = device
        self.wan_type = wan_type
        self.use_dual_head = use_dual_head
        self.is_full_finetune = is_full_finetune
        self.torch_dtype = torch_dtype
        self.lora_checkpoint = lora_checkpoint
        self.rot_grip_checkpoint = rot_grip_checkpoint

        # Rotation and gripper parameters
        self.rotation_resolution = rotation_resolution
        self.num_rotation_bins = num_rotation_bins
        self.num_history_frames = num_history_frames
        self.use_initial_gripper_state = use_initial_gripper_state
        self.local_feat_size = local_feat_size
        self.hidden_dim = hidden_dim

        print("="*60)
        print("Initializing MV View Inference with Rot/Grip Prediction")
        print("="*60)
        print(f"  Model base path: {model_base_path}")
        print(f"  LoRA checkpoint: {lora_checkpoint}")
        print(f"  Rot/Grip checkpoint: {rot_grip_checkpoint if rot_grip_checkpoint else '(Not provided)'}")
        print(f"  WAN type: {wan_type}")
        print(f"  Use dual head: {use_dual_head}")
        print(f"  Is full finetune: {is_full_finetune}")
        print(f"  Num history frames: {num_history_frames}")
        print(f"  Rotation resolution: {rotation_resolution}°")
        print(f"  Num rotation bins: {num_rotation_bins}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Local feat size: {local_feat_size}")
        print(f"  Use initial gripper state: {use_initial_gripper_state}")
        print("="*60)

        # Configuration consistency check
        if num_history_frames > 1 and wan_type != "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY":
            raise ValueError(
                f"Configuration mismatch!\n"
                f"  num_history_frames={num_history_frames} (>1) requires wan_type='5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY'\n"
                f"  but got wan_type='{wan_type}'\n"
            )

        # Import correct pipeline based on wan_type
        if wan_type == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP":
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_view import (
                WanVideoPipeline,
                ModelConfig
            )
            print("  Using single-frame pipeline: wan_video_5B_TI2V_heatmap_and_rgb_mv_view")
        elif wan_type == "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP_HISTORY":
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_history import (
                WanVideoPipeline,
                ModelConfig
            )
            print("  Using multi-frame history pipeline: wan_video_5B_TI2V_heatmap_and_rgb_mv_history")
        else:
            raise ValueError(f"Unsupported wan_type: {wan_type}")

        # Store pipeline class for later use
        self.WanVideoPipeline = WanVideoPipeline
        self.ModelConfig = ModelConfig

        # Load pipeline
        self.pipe = self._load_pipeline(model_base_path, lora_checkpoint)

        # Load rotation and gripper predictor if checkpoint provided
        if rot_grip_checkpoint and os.path.exists(rot_grip_checkpoint):
            self._load_rot_grip_predictor(rot_grip_checkpoint)
        else:
            print("\nRotation/gripper predictor not loaded (checkpoint not provided or not found)")
            print("  Inference will only generate heatmap and RGB videos without rot/grip predictions")
            self.rot_grip_predictor = None

    def _load_pipeline(self, model_base_path: str, lora_checkpoint: str):
        """Load the pipeline with model and LoRA weights."""
        # Model configs
        model_configs = [
            self.ModelConfig(
                path=[
                    f"{model_base_path}/diffusion_pytorch_model-00001-of-00003.safetensors",
                    f"{model_base_path}/diffusion_pytorch_model-00002-of-00003.safetensors",
                    f"{model_base_path}/diffusion_pytorch_model-00003-of-00003.safetensors",
                ],
            ),
            self.ModelConfig(
                path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth",
            ),
            self.ModelConfig(
                path=f"{model_base_path}/Wan2.2_VAE.pth",
            ),
        ]

        print("\nStep 1: Loading base pipeline...")
        pipe = self.WanVideoPipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=model_configs,
            wan_type=self.wan_type,
            use_dual_head=self.use_dual_head,
        )

        print("\nStep 2: Model loaded successfully (view-concat mode)")

        # Add multi-view attention modules
        print("\nStep 2.5: Adding multi-view attention modules...")
        from diffsynth.models.wan_video_dit_mv_view import SelfAttention
        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        model_dtype = pipe.dit.blocks[0].self_attn.q.weight.dtype
        model_device = pipe.dit.blocks[0].self_attn.q.weight.device

        for block in pipe.dit.blocks:
            # Create projector
            block.projector = torch.nn.Linear(dim, dim).to(dtype=model_dtype, device=model_device)
            block.projector.weight.data.zero_()
            block.projector.bias.data.zero_()

            # Create norm_mvs
            block.norm_mvs = torch.nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False).to(dtype=model_dtype, device=model_device)

            # Create modulation_mvs parameter
            block.modulation_mvs = torch.nn.Parameter(torch.randn(1, 3, dim, dtype=model_dtype, device=model_device) / dim**0.5)
            block.modulation_mvs.data = block.modulation.data[:, :3, :].clone()

            # Create mvs_attn
            block.mvs_attn = SelfAttention(dim, block.self_attn.num_heads, block.self_attn.norm_q.eps)
            block.mvs_attn.load_state_dict(block.self_attn.state_dict(), strict=True)
            block.mvs_attn = block.mvs_attn.to(dtype=model_dtype, device=model_device)

        print("  Multi-view modules added successfully")

        # Load checkpoint weights
        print("\nStep 3: Loading checkpoint weights...")
        if lora_checkpoint and os.path.exists(lora_checkpoint):
            if self.is_full_finetune:
                self._load_full_finetune_checkpoint(pipe, lora_checkpoint)
            else:
                self._load_lora_with_base_weights(pipe, lora_checkpoint)
        else:
            print(f"  Warning: Checkpoint not found: {lora_checkpoint}")

        # Set models to eval mode
        print(f"\nStep 4: Setting models to eval mode on {self.device}...")
        if pipe.dit is not None:
            pipe.dit.eval()
        if pipe.vae is not None:
            pipe.vae.eval()
        if pipe.text_encoder is not None:
            pipe.text_encoder.eval()

        print("\nPipeline loaded successfully!")
        return pipe

    def _load_rot_grip_predictor(self, checkpoint_path: str):
        """Load rotation and gripper predictor model (View version using VAE latents)."""
        print("\nLoading rotation and gripper predictor (View version)...")

        # Import predictor model (View version, uses VAE latents directly)

        # Initialize predictor
        # View版本：6个views（3 RGB + 3 Heatmap），每个view 48通道latents
        self.rot_grip_predictor = MultiViewRotationGripperPredictorView(
            rgb_channels=48,  # VAE latent channels (not decoder intermediate!)
            heatmap_channels=48,  # VAE latent channels (not decoder intermediate!)
            hidden_dim=self.hidden_dim,
            num_views=6,  # 6 views: 3 RGB + 3 Heatmap (view concat mode)
            num_rotation_bins=self.num_rotation_bins,
            dropout=0.1,
            vae=None,  # VAE object (optional, for decoding heatmaps to find peaks)
            local_feature_size=self.local_feat_size,
            use_initial_gripper_state=self.use_initial_gripper_state,
        ).to(device=self.device, dtype=self.torch_dtype)

        # Load checkpoint
        print(f"  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.rot_grip_predictor.load_state_dict(state_dict, strict=True)
        self.rot_grip_predictor.eval()

        # Verify checkpoint loaded
        num_params = sum(p.numel() for p in self.rot_grip_predictor.parameters())
        trainable_params = sum(p.numel() for p in self.rot_grip_predictor.parameters() if p.requires_grad)
        print(f"  Rotation/gripper predictor loaded successfully!")
        print(f"    Total parameters: {num_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

        # Check if weights are reasonable (not all zeros or random)
        sample_param = next(iter(self.rot_grip_predictor.parameters()))
        print(f"    Sample weight stats: mean={sample_param.mean().item():.6f}, std={sample_param.std().item():.6f}")

    def _verify_lora_loading(self, pipe, checkpoint_path: str, alpha: float = 1.0) -> bool:
        """
        Rigorously verify that LoRA weights were correctly loaded and applied.

        Returns:
            bool: True if LoRA was correctly loaded, False otherwise
        """
        print("\n  === LoRA Loading Verification ===")

        # Step 1: Check if LoRA weights exist in checkpoint
        print("  [1/5] Checking LoRA weights in checkpoint...")
        try:
            state_dict = load_state_dict(checkpoint_path)
        except Exception as e:
            print(f"    ✗ FAILED: Cannot load checkpoint: {e}")
            return False

        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        lora_up_keys = [k for k in lora_keys if 'lora_up' in k or 'lora_A' in k or 'lora.up' in k]
        lora_down_keys = [k for k in lora_keys if 'lora_down' in k or 'lora_B' in k or 'lora.down' in k]

        if not lora_keys:
            print(f"    ✗ FAILED: No LoRA weights found in checkpoint")
            return False

        print(f"    ✓ Found {len(lora_keys)} LoRA keys in checkpoint")
        print(f"      - LoRA up/A keys: {len(lora_up_keys)}")
        print(f"      - LoRA down/B keys: {len(lora_down_keys)}")

        # Step 2: Check if LoRA modules exist in model
        print("\n  [2/5] Checking LoRA modules in model...")
        lora_modules_in_model = []
        for name, module in pipe.dit.named_modules():
            if hasattr(module, 'lora_up') or hasattr(module, 'lora_down') or \
               hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                lora_modules_in_model.append(name)

        if not lora_modules_in_model:
            print(f"    ✗ FAILED: No LoRA modules found in model")
            print(f"      Model may not have LoRA support injected")
            return False

        print(f"    ✓ Found {len(lora_modules_in_model)} LoRA modules in model")
        print(f"      Sample modules: {lora_modules_in_model[:3]}")

        # Step 3: Verify LoRA parameters are not zero
        print("\n  [3/5] Checking LoRA parameter values...")
        lora_params_checked = 0
        zero_params = 0
        non_zero_params = 0

        for name, module in pipe.dit.named_modules():
            for lora_attr in ['lora_up', 'lora_down', 'lora_A', 'lora_B']:
                if hasattr(module, lora_attr):
                    lora_param = getattr(module, lora_attr)
                    if hasattr(lora_param, 'weight'):
                        param_data = lora_param.weight.data
                        lora_params_checked += 1

                        if param_data.abs().sum().item() < 1e-8:
                            zero_params += 1
                        else:
                            non_zero_params += 1

        if lora_params_checked == 0:
            print(f"    ✗ FAILED: No LoRA parameters found to check")
            return False

        print(f"    ✓ Checked {lora_params_checked} LoRA parameters")
        print(f"      - Non-zero parameters: {non_zero_params}")
        print(f"      - Zero parameters: {zero_params}")

        if non_zero_params == 0:
            print(f"    ✗ WARNING: All LoRA parameters are zero!")
            return False

        # Step 4: Compute LoRA contribution to model
        print("\n  [4/5] Computing LoRA contribution magnitude...")
        total_lora_magnitude = 0.0
        lora_contributions = []

        for name, module in pipe.dit.named_modules():
            if hasattr(module, 'lora_up') and hasattr(module, 'lora_down'):
                # Compute LoRA contribution: alpha * (up @ down)
                up_weight = module.lora_up.weight.data
                down_weight = module.lora_down.weight.data

                # LoRA contribution magnitude
                contribution = (up_weight.abs().mean() * down_weight.abs().mean()).item()
                lora_contributions.append(contribution)
                total_lora_magnitude += contribution
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                up_weight = module.lora_A.weight.data
                down_weight = module.lora_B.weight.data
                contribution = (up_weight.abs().mean() * down_weight.abs().mean()).item()
                lora_contributions.append(contribution)
                total_lora_magnitude += contribution

        if len(lora_contributions) == 0:
            print(f"    ✗ FAILED: Cannot compute LoRA contributions")
            return False

        avg_contribution = total_lora_magnitude / len(lora_contributions)
        max_contribution = max(lora_contributions)
        min_contribution = min(lora_contributions)

        print(f"    ✓ LoRA contribution statistics:")
        print(f"      - Average magnitude: {avg_contribution:.6e}")
        print(f"      - Max magnitude: {max_contribution:.6e}")
        print(f"      - Min magnitude: {min_contribution:.6e}")
        print(f"      - Alpha scaling: {alpha}")

        # Step 5: Verify alpha scaling
        print("\n  [5/5] Verifying alpha scaling...")
        if hasattr(pipe, 'lora_alpha') or hasattr(pipe.dit, 'lora_alpha'):
            print(f"    ✓ LoRA alpha is set in model")

        # Final verdict
        print("\n  === Verification Result ===")
        if non_zero_params > 0 and avg_contribution > 1e-8:
            print(f"  ✓ PASSED: LoRA weights are correctly loaded and applied")
            print(f"    - {len(lora_modules_in_model)} modules with LoRA")
            print(f"    - {non_zero_params}/{lora_params_checked} non-zero parameters")
            print(f"    - Average contribution: {avg_contribution:.6e}")
            return True
        else:
            print(f"  ✗ FAILED: LoRA loading verification failed")
            print(f"    Please check:")
            print(f"      1. Checkpoint contains valid LoRA weights")
            print(f"      2. LoRA keys match model architecture")
            print(f"      3. Alpha value is not zero")
            return False

    def _load_checkpoint_weights(self, pipe, checkpoint_path: str):
        """Load non-LoRA weights from checkpoint."""
        try:
            print(f"  Loading state dict from: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)

            # Categorize weights
            head_weights = {}
            patch_embedding_weights = {}
            modulation_weights = {}
            mvs_weights = {}

            for key, value in state_dict.items():
                if 'lora' in key.lower():
                    continue

                if any(pattern in key for pattern in ['head_rgb', 'head_heatmap', 'head.']):
                    if 'attention' not in key.lower() and 'attn' not in key.lower():
                        head_weights[key] = value
                elif 'patch_embedding' in key or 'patch_embed' in key:
                    patch_embedding_weights[key] = value
                elif 'modulation' in key and 'mvs' not in key:
                    modulation_weights[key] = value
                elif any(pattern in key for pattern in ['mvs_attn', 'norm_mvs', 'projector', 'modulation_mvs']):
                    mvs_weights[key] = value
                else:
                    assert False

            print(f"  Found {len(head_weights)} head weights")
            print(f"  Found {len(patch_embedding_weights)} patch_embedding weights")
            print(f"  Found {len(modulation_weights)} modulation weights")
            print(f"  Found {len(mvs_weights)} multi-view module weights")

            # Merge all weights
            weights_to_load = {}
            weights_to_load.update(head_weights)
            weights_to_load.update(patch_embedding_weights)
            weights_to_load.update(modulation_weights)
            weights_to_load.update(mvs_weights)

            if not weights_to_load:
                print("  Warning: No non-LoRA weights found in checkpoint")
                return

            # Clean weight keys
            weights_clean = {}
            for key, value in weights_to_load.items():
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value

            print(f"  Loading {len(weights_clean)} non-LoRA weights into DIT model...")

            # Load into DIT model
            missing_keys, unexpected_keys = pipe.dit.load_state_dict(
                weights_clean, strict=False
            )

            loaded_keys = set(weights_clean.keys()) - set(unexpected_keys)
            print(f"    Successfully loaded {len(loaded_keys)}/{len(weights_clean)} weights")

            print("  Non-LoRA weights loaded successfully!")

        except Exception as e:
            print(f"  Warning: Failed to load non-LoRA weights: {e}")

    def _load_lora_with_base_weights(self, pipe, checkpoint_path: str, alpha: float = 1.0):
        """Load LoRA checkpoint: base weights first, then apply LoRA."""
        print("  Loading checkpoint with LoRA logic...")

        # Step 1: Load non-LoRA weights
        print("\n  Step 3.1: Loading non-LoRA weights...")
        self._load_checkpoint_weights(pipe, checkpoint_path)

        # Step 2: Load LoRA weights
        print("\n  Step 3.2: Loading and applying LoRA weights...")
        pipe.load_lora(pipe.dit, checkpoint_path, alpha=alpha)
        print("    LoRA weights loaded and applied")

        # # Step 3: Verify LoRA loading
        # print("\n  Step 3.3: Verifying LoRA loading...")
        # verification_passed = self._verify_lora_loading(pipe, checkpoint_path, alpha=alpha)
        # if not verification_passed:
        #     raise RuntimeError(
        #         "LoRA verification failed! Please check:\n"
        #         "  1. Checkpoint file contains valid LoRA weights\n"
        #         "  2. LoRA keys match model architecture\n"
        #         "  3. Alpha value is correctly set\n"
        #         "  4. Model has LoRA support properly injected"
        #     )

        print("\n  Checkpoint loaded successfully!")

    def _load_full_finetune_checkpoint(self, pipe, checkpoint_path: str):
        """Load full finetune checkpoint."""
        try:
            print(f"  Loading full finetune checkpoint: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)
            print(f"    Loaded {len(state_dict)} keys from checkpoint")

            # Filter DIT weights
            dit_weights = {}
            lora_keys_count = 0
            for key, value in state_dict.items():
                if 'lora' in key.lower():
                    lora_keys_count += 1
                    continue
                dit_weights[key] = value

            print(f"    Filtered {len(dit_weights)} DIT weights (skipped {lora_keys_count} LoRA keys)")

            # Clean weight keys
            weights_clean = {}
            for key, value in dit_weights.items():
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value

            # Load into DIT model
            missing_keys, unexpected_keys = pipe.dit.load_state_dict(
                weights_clean, strict=False
            )

            loaded_count = len(weights_clean) - len(unexpected_keys)
            print(f"    Successfully loaded {loaded_count}/{len(weights_clean)} weights")

            print("  Full finetune checkpoint loaded successfully!")

        except Exception as e:
            print(f"  Error: Failed to load full finetune checkpoint: {e}")
            raise
        
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
        Convert delta rotation bins to delta degrees.

        Args:
            delta_bins: (T, 3) or (3,) - delta bin indices

        Returns:
            delta_degrees: (T, 3) or (3,) - delta rotation in degrees
        """
        # Delta bins are centered at num_bins/2
        # bins < num_bins/2 => negative delta
        # bins > num_bins/2 => positive delta
        center_bin = self.num_rotation_bins // 2
        delta_degrees = (delta_bins - center_bin) * self.rotation_resolution
        return delta_degrees

    @torch.no_grad()
    def run_inference(
        self,
        input_image: List[Image.Image],  # [num_views] list of heatmap PIL Images
        input_image_rgb: List[Image.Image],  # [num_views] list of RGB PIL Images
        prompt: str = "robot arm manipulation",
        initial_rotation: Optional[np.ndarray] = None,  # (3,) - [roll, pitch, yaw] in degrees
        initial_gripper: Optional[int] = None,  # 0 or 1
        num_frames: int = 25,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        cfg_scale: float = 1.0,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Run inference with rotation and gripper prediction.

        Args:
            input_image: List of input heatmap images for each view
            input_image_rgb: List of input RGB images for each view
            prompt: Text prompt
            initial_rotation: Initial rotation [roll, pitch, yaw] in degrees (for rot/grip prediction)
            initial_gripper: Initial gripper state 0=closed, 1=open (for rot/grip prediction)
            num_frames: Number of output frames
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed

        Returns:
            Dictionary containing:
                - video_heatmap: Generated heatmap videos
                - video_rgb: Generated RGB videos
                - rotation_predictions: (T-1, 3) rotation predictions (if rot/grip enabled)
                - gripper_predictions: (T-1,) gripper predictions (if rot/grip enabled)
        """
        import time
        print(f"\nRunning inference with {len(input_image)} views...")
        print(f"  Prompt: {prompt}")
        print(f"  Frames: {num_frames}, Size: {height}x{width}")
        print(f"  Steps: {num_inference_steps}, CFG: {cfg_scale}, Seed: {seed}")

        start_time = time.time()

        # Run pipeline to generate videos
        output = self.pipe(
            prompt=prompt,
            input_image=input_image,
            input_image_rgb=input_image_rgb,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            use_dual_head=self.use_dual_head,
            seed=seed,
        )

        elapsed_time = time.time() - start_time
        print(f"\nVideo generation completed in {elapsed_time:.2f}s")

        result = {
            'video_heatmap': output['video_heatmap'],
            'video_rgb': output['video_rgb'],
        }

        # If rotation/gripper predictor is available, run prediction
        if self.rot_grip_predictor is not None and initial_rotation is not None and initial_gripper is not None:
            print("\nRunning rotation and gripper prediction...")

            # Get latents from pipeline output
            rgb_latents = output.get('rgb_latents')  # (num_views, c, t, h, w)
            heatmap_latents = output.get('heatmap_latents')  # (num_views, c, t, h, w)

            if rgb_latents is not None and heatmap_latents is not None:
                print(f"  RGB latents shape: {rgb_latents.shape}")
                print(f"  Heatmap latents shape: {heatmap_latents.shape}")

                # Prepare latents for predictor
                # Predictor expects: (b, 3, 48, t_compressed, h, w)
                b = 1
                rgb_latents_batched = rgb_latents.unsqueeze(0)  # (1, num_views, c, t, h, w)
                heatmap_latents_batched = heatmap_latents.unsqueeze(0)

                # Prepare heatmap images from video output
                # video_heatmap is a list of 3 videos, each video is a list of PIL Images
                heatmap_videos = output['video_heatmap']  # List of 3 videos
                # Stack and convert to tensor: (3, T, C, H, W) -> (1, 3, T, C, H, W)
                heatmap_images_list = []
                for video in heatmap_videos:
                    # video is a list of PIL Images (one per frame)
                    frame_tensors = []
                    for pil_image in video:
                        # Convert PIL Image to tensor: (H, W, C) -> (C, H, W)
                        frame_np = np.array(pil_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
                        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)  # (C, H, W)
                        frame_tensors.append(frame_tensor)
                    video_tensor = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
                    heatmap_images_list.append(video_tensor)
                heatmap_images = torch.stack(heatmap_images_list, dim=0)  # (3, T, C, H, W)
                heatmap_images = heatmap_images.unsqueeze(0).to(self.device)  # (1, 3, T, C, H, W)

                print(f"  Heatmap images shape: {heatmap_images.shape}")

                # Calculate num_future_frames
                num_future_frames = num_frames - self.num_history_frames

                # Prepare history_gripper_states if needed
                history_gripper_states = None
                if self.use_initial_gripper_state and initial_gripper is not None:
                    history_gripper_states = torch.full(
                        (1, self.num_history_frames),
                        initial_gripper,
                        dtype=torch.long,
                        device=self.device
                    )

                # Run predictor
                rotation_logits, gripper_logits = self.rot_grip_predictor(
                    rgb_latents=rgb_latents_batched,
                    heatmap_latents=heatmap_latents_batched,
                    num_future_frames=num_future_frames,
                    heatmap_images=heatmap_images,  # Use actual heatmap images
                    peak_positions=None,
                    colormap_name='jet',
                    num_history_frames=self.num_history_frames,
                    history_gripper_states=history_gripper_states,
                )

                # Decode predictions
                rotation_logits = rotation_logits[0].view(num_future_frames, 3, self.num_rotation_bins)
                gripper_logits = gripper_logits[0]

                # print(f"  Rotation logits shape: {rotation_logits.shape}")
                # print(f"  Gripper logits shape: {gripper_logits.shape}")
                # print(f"  Expected num_future_frames: {num_future_frames}")

                # Analyze gripper logits in detail
                gripper_probs = F.softmax(gripper_logits, dim=1)  # (T, 2)
                # print(f"  Gripper logits (first 3 frames):")
                # for i in range(min(3, gripper_logits.shape[0])):
                #     print(f"    Frame {i}: logits=[{gripper_logits[i,0]:.3f}, {gripper_logits[i,1]:.3f}], probs=[{gripper_probs[i,0]:.3f}, {gripper_probs[i,1]:.3f}]")

                rotation_delta_bins = rotation_logits.argmax(dim=-1)  # (T, 3)
                gripper_change = gripper_logits.argmax(dim=1)  # (T,)

                # print(f"  Gripper change predictions: {gripper_change.cpu().numpy()}")
                # print(f"  Initial gripper state: {initial_gripper}")

                # Convert to absolute values
                print(f"pred rotation delta bins: {rotation_delta_bins.cpu().numpy()}")
                rotation_delta_degrees = self._delta_bins_to_degrees(rotation_delta_bins.cpu().numpy())
                rotation_predictions = initial_rotation + rotation_delta_degrees
                rotation_predictions = ((rotation_predictions + 180) % 360) - 180

                # Gripper predictions
                # gripper_change[t] == 0 means same as initial, == 1 means different from initial
                gripper_predictions = np.zeros(num_future_frames, dtype=np.int64)
                for t in range(num_future_frames):
                    if gripper_change[t].item() == 1:
                        # Different from initial
                        gripper_predictions[t] = 1 - initial_gripper
                    else:
                        # Same as initial
                        gripper_predictions[t] = initial_gripper

                result['rotation_predictions'] = rotation_predictions
                result['gripper_predictions'] = gripper_predictions

                print(f"  Rotation/gripper prediction completed")
                print(f"  First 3 rotation predictions: {rotation_predictions[:3]}")
                print(f"  First 3 gripper predictions: {gripper_predictions[:3]}")
            else:
                print("  Warning: Latents not available in pipeline output, skipping rot/grip prediction")
        elif self.rot_grip_predictor is None:
            print("\nRotation/gripper prediction skipped (predictor not loaded)")
        else:
            print("\nRotation/gripper prediction skipped (initial_rotation or initial_gripper not provided)")

        return result

    # Alias for backward compatibility with Server code
    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for run_inference to maintain backward compatibility with Server."""
        return self.run_inference(*args, **kwargs)

    def find_peak_position(self, heatmap_image: Image.Image, colormap_name: str = 'jet') -> Tuple[int, int]:
        """Find peak position in heatmap."""
        heatmap_image_np = np.array(heatmap_image).astype(np.float32) / 255.0
        heatmap_array = extract_heatmap_from_colormap(heatmap_image_np, colormap_name)
        max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
        return (max_pos[1], max_pos[0])  # (x, y) format

    def find_peaks_batch(self, heatmap_images: List[List[Image.Image]], colormap_name: str = 'jet') -> List[List[Tuple[int, int]]]:
        """Batch find peak positions."""
        num_frames = len(heatmap_images)
        num_views = len(heatmap_images[0])

        peaks = []
        for frame_idx in range(num_frames):
            frame_peaks = []
            for view_idx in range(num_views):
                peak = self.find_peak_position(heatmap_images[frame_idx][view_idx], colormap_name)
                frame_peaks.append(peak)
            peaks.append(frame_peaks)

        return peaks

    def calculate_peak_distance(self, pred_peak: Tuple[int, int], gt_peak: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two peaks."""
        return np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)


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


    def get_position_from_heatmap(self, heatmaps,rev_trans,dyn_cam_info=None, y_q=None,visualize=False, use_softmax=True):
        """
        Estimate the q-values given output from mvt
        :param heatmap: heatmaps output from wan  (bs,view,h*w)
        :param rev_trans  逆变换函数
        :param use_softmax: 是否使用softmax归一化（默认True保持兼容性）
        """
        h ,w = self.img_size
        bs,nc,h_w=heatmaps.shape

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
            lora_checkpoint=args.lora_checkpoint,
            rot_grip_checkpoint=args.rot_grip_checkpoint,
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

        # RLBench cameras (front, left_shoulder, right_shoulder, wrist)
        # 训练时使用前3个相机
        self.cameras = ["front", "left_shoulder", "right_shoulder", "wrist"]

    def _extract_obs(self, obs, channels_last: bool = False):
        """
        从RLBench Observation对象中提取点云和RGB数据
        与训练数据集的_extract_obs方法保持一致

        Args:
            obs: RLBench Observation对象
            channels_last: 是否使用channels_last格式

        Returns:
            pcd: List of point clouds for each camera [cam1_pcd, cam2_pcd, ...]
            rgb: List of RGB images for each camera [cam1_rgb, cam2_rgb, ...]
        """
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        obs_dict = {
            k: v for k, v in obs_dict.items()
            if any(kw in k for kw in ["rgb", "point_cloud"])
        }

        for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
            obs_dict[k] = v.astype(np.float32)

        pcd = [obs_dict[f"{cam}_point_cloud"] for cam in self.cameras]
        rgb = [obs_dict[f"{cam}_rgb"] for cam in self.cameras]

        return pcd, rgb
        
    
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


    def predict_action(self, obs, lang_goal):
        """
        从RLBench observation预测动作序列

        Args:
            obs: RLBench Observation对象
            lang_goal: 任务指令文本

        Returns:
            action: (T, 8) array - [x, y, z, qx, qy, qz, qw, gripper]
        """
        # 1. 从RLBench observation中提取多相机数据
        # 返回List of arrays: [cam1, cam2, cam3, cam4]
        start_pcd_list, start_rgb_list = self._extract_obs(obs)

        # 2. 获取当前gripper pose和状态
        start_pose = obs.gripper_pose  # (7,) [x, y, z, qx, qy, qz, qw]
        start_gripper = obs.gripper_open

        # 3. 转换quaternion为欧拉角（用于rotation prediction）
        # start_rotation = quaternion_to_discrete_euler(start_pose[3:7], self.inferencer.rotation_resolution)  # (3,) - [roll, pitch, yaw] bins
        # start_rotation_degrees = self.inferencer.xrxs(start_rotation)
        start_rotation_degrees = quaternion_to_euler_continuous(start_pose[3:7])  # (3,) - [roll, pitch, yaw] in degrees

        # 4. 预处理点云和RGB数据
        # preprocess需要输入格式：pcd_list[[cam1,cam2,cam3,...]], feat_list[[cam1,cam2,cam3,...]]
        processed_pcd_list, processed_rgb_list, processed_pos, processed_rot_xyzw, rev_trans = self.preprocess(
            [start_pcd_list],  # 外层list表示时间步，这里只有1个时间步
            [start_rgb_list],   # 外层list表示时间步
            [start_pose]        # (1, 7) - 只有起始pose
        )

        processed_start_pcd = processed_pcd_list[0]
        processed_start_rgb = processed_rgb_list[0]
        processed_poses = torch.cat((processed_pos, processed_rot_xyzw), dim=1)  # (1, 7)
        processed_start_pose = processed_poses[0]

        # 5. 投影点云到多视角RGB图像
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_start_pcd, processed_start_rgb, img_aug_before=0.0
        )  # (1, num_views, H, W, 6)
        rgb_image = rgb_image[0, :, :, :, 3:]  # (num_views, H, W, 3)

        # 确保是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)  # (num_views, H, W, 3)
        num_views = rgb_image.shape[0]

        # 6. 生成heatmap
        img_locations = self.projection_interface.project_pose_to_pixel(
            processed_pos.unsqueeze(0).to(self.projection_interface.renderer_device)
        )  # (bs, num_poses, num_views, 2)

        heatmap_sequence = self.projection_interface.generate_heatmap_from_img_locations(
            img_locations,
            self.args.img_size[0], self.args.img_size[1],
        )  # (bs, seq_len+1, num_views, H, W)
        heatmap_sequence = heatmap_sequence[0, :, :, :, :]  # (seq_len+1, num_views, H, W)
        heatmap_start = heatmap_sequence[0]  # (num_views, H, W)

        # 7. 准备输入图像（RGB和heatmap）
        input_image = []
        input_image_rgb = []

        for v in range(num_views):
            rgb_view = rgb_image[v]  # (H, W, 3)
            pil_img = rgb_to_pil_image(rgb_view)
            input_image_rgb.append(pil_img)

        for v in range(num_views):
            heatmap_view = heatmap_start[v]  # (H, W)
            heatmap_np = heatmap_view.cpu().numpy()

            # 归一化到[0, 1]
            view_hm_min = heatmap_np.min()
            view_hm_max = heatmap_np.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (heatmap_np - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = heatmap_np

            # 应用colormap（使用jet，与训练一致）
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(view_hm_colored)
            input_image.append(pil_img)
        
        # 8. 执行推理
        output = self.inferencer.predict(
            prompt=lang_goal,
            input_image=input_image,
            input_image_rgb=input_image_rgb,
            initial_rotation=start_rotation_degrees,
            initial_gripper=start_gripper,
            num_frames=self.args.sequence_length + 1,  # 包括初始帧
            height=self.args.img_size[0],
            width=self.args.img_size[1],
            num_inference_steps=50,
            cfg_scale=1.0,
        )
        
        pred_heatmap = output['video_heatmap']

        # 9. 从预测的heatmap提取3D位置
        pred_position = get_3d_position_from_pred_heatmap(
            pred_heatmap_colormap=pred_heatmap,
            rev_trans=rev_trans,
            projection_interface=self.projection_interface,
            colormap_name='jet'
        )  # (T, 3) where T = sequence_length
    
    
        # 10. 提取rotation和gripper预测
        pred_rotation = output['rotation_predictions']  # (T-1, 3) - [roll, pitch, yaw] in degrees
        pred_gripper = output['gripper_predictions']  # (T-1,) - continuous gripper values

        # 11. 转换rotation为quaternion
        # pred_rotation shape: (T-1, 3) in degrees, 未来帧的rotation
        pred_quaternions = []
        for i in range(len(pred_rotation)):
            euler_deg = pred_rotation[i]  # (3,) - [roll, pitch, yaw] in degrees
            r = Rotation.from_euler('xyz', euler_deg, degrees=True)
            quat_xyzw = r.as_quat()  # (4,) - [qx, qy, qz, qw]
            pred_quaternions.append(quat_xyzw)
        pred_quaternions = np.array(pred_quaternions)  # (T-1, 4)

        # 12. 组装动作序列
        # pred_position: (T, 3) - 包含起始帧和未来帧
        # 我们需要未来帧的位置: pred_position[1:]  (T-1, 3)
        # pred_quaternions: (T-1, 4)
        # pred_gripper: (T-1,)
        
        # 组装动作序列为 ActResult 列表
        action_list = []
        for i in range(len(pred_position) - 1):
            continuous_action = np.concatenate([
                pred_position[i + 1],      # (3,) - position
                pred_quaternions[i],        # (4,) - quaternion
                pred_gripper[i:i+1],       # (1,) - gripper
                np.array([1.0])            # (1,) - collision
            ])
            action_list.append(ActResult(continuous_action))

        return action_list