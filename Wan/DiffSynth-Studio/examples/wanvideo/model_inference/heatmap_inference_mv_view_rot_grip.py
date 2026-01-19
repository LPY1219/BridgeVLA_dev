"""
Inference Script for Multi-View Heatmap + Rotation & Gripper Prediction

This script performs inference using WanModel with view-concatenation mode,
including both heatmap prediction and rotation/gripper prediction.

Key features:
- View-concatenation mode (views in batch dimension)
- Dual-head output (RGB + Heatmap)
- Rotation and gripper prediction based on VAE latents
- Uses MultiViewRotationGripperPredictorView (view concat, 48-channel latents)
- Supports both single-frame and multi-frame history modes

Adapted from heatmap_inference_mv_view.py
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import cv2
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from diffsynth.models import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap


def get_root_path():
    """自动检测BridgeVLA根目录"""
    possible_paths = [
        "/DATA/disk1/lpy/BridgeVLA_dev",
        "/home/lpy/BridgeVLA_dev",
        "/DATA/disk0/lpy/BridgeVLA_dev",
        "/DATA/disk1/lpy_a100_4/BridgeVLA_dev",
        "/DATA/disk1/lpy_a100_1/BridgeVLA_dev",
        "/mnt/robot-rfm/user/lpy/BridgeVLA_dev"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")


ROOT_PATH = get_root_path()


class HeatmapInferenceMVViewRotGrip:
    """
    Inference class for multi-view view-concatenation model with rotation and gripper prediction.

    Uses MultiViewRotationGripperPredictorView which operates on VAE latents directly.
    """

    def __init__(
        self,
        model_base_path: str,
        lora_checkpoint: str,
        rot_grip_checkpoint: Optional[str] = None,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
        use_dual_head: bool = True,
        use_merged_pointcloud: bool = False,
        use_different_projection: bool = False,
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
        self.use_merged_pointcloud = use_merged_pointcloud
        self.use_different_projection = use_different_projection
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

        # Debug image counter and directory for visualization
        self._debug_img_counter = 0
        from datetime import datetime
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_img_dir = os.path.join(
            ROOT_PATH,
            "Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img",
            experiment_timestamp
        )
        os.makedirs(self._debug_img_dir, exist_ok=True)
        print(f"Debug images will be saved to: {self._debug_img_dir}")

        print("="*60)
        print("Initializing MV View Inference with Rot/Grip Prediction")
        print("="*60)
        print(f"  Model base path: {model_base_path}")
        print(f"  LoRA checkpoint: {lora_checkpoint}")
        print(f"  Rot/Grip checkpoint: {rot_grip_checkpoint if rot_grip_checkpoint else '(Not provided)'}")
        print(f"  WAN type: {wan_type}")
        print(f"  Use dual head: {use_dual_head}")
        print(f"  Use merged pointcloud: {use_merged_pointcloud}")
        print(f"  Use different projection: {use_different_projection}")
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
        from examples.wanvideo.model_training.mv_rot_grip_vae_decode_feature_3_view import MultiViewRotationGripperPredictorView

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
        visualize: bool = True,
        visualize_save_path: str = None,
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

        # Set default visualization save path
        if visualize_save_path is None:
            visualize_save_path = os.path.join(self._debug_img_dir, f"debug_output_{self._debug_img_counter:04d}.png")

        # Visualize input images
        self._visualize_input_images(input_image, input_image_rgb, prompt)

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

                print(f"  Rotation logits shape: {rotation_logits.shape}")
                print(f"  Gripper logits shape: {gripper_logits.shape}")
                print(f"  Expected num_future_frames: {num_future_frames}")

                # Analyze gripper logits in detail
                gripper_probs = F.softmax(gripper_logits, dim=1)  # (T, 2)
                print(f"  Gripper logits (first 3 frames):")
                for i in range(min(3, gripper_logits.shape[0])):
                    print(f"    Frame {i}: logits=[{gripper_logits[i,0]:.3f}, {gripper_logits[i,1]:.3f}], probs=[{gripper_probs[i,0]:.3f}, {gripper_probs[i,1]:.3f}]")

                rotation_delta_bins = rotation_logits.argmax(dim=-1)  # (T, 3)
                gripper_change = gripper_logits.argmax(dim=1)  # (T,)

                print(f"  Gripper change predictions: {gripper_change.cpu().numpy()}")
                print(f"  Initial gripper state: {initial_gripper}")

                # Convert to absolute values
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

        # Visualize generated video frames if requested
        if visualize:
            print("\nVisualizing generated video frames...")
            # Get videos from result
            video_heatmap_frames = result['video_heatmap']  # [num_views][T]
            video_rgb_frames = result['video_rgb']  # [num_views][T]

            # Convert format [view][time] -> [time][view] if needed
            if len(video_heatmap_frames[0]) > 3:  # Assume num_views <= 3
                num_views = len(video_heatmap_frames)
                num_frames_output = len(video_heatmap_frames[0])
                video_heatmap_frames_T = [[video_heatmap_frames[v][t] for v in range(num_views)]
                                          for t in range(num_frames_output)]
                video_rgb_frames_T = [[video_rgb_frames[v][t] for v in range(num_views)]
                                      for t in range(num_frames_output)]
            else:
                video_heatmap_frames_T = video_heatmap_frames
                video_rgb_frames_T = video_rgb_frames
                num_views = len(video_heatmap_frames_T[0])
                num_frames_output = len(video_heatmap_frames_T)

            # Find heatmap peaks
            print("Finding heatmap peaks for visualization...")
            peaks = self.find_peaks_batch(video_heatmap_frames_T, colormap_name='jet')

            # Mark peaks on RGB images
            print("Marking peaks on RGB images...")
            video_rgb_frames_marked_T = self.mark_peaks_on_images(
                video_rgb_frames_T, peaks,
                marker_size=8,
                marker_color='red',
                marker_width=3
            )

            # Convert back to [view][time] format for visualization
            video_rgb_frames_marked = [[video_rgb_frames_marked_T[t][v] for t in range(num_frames_output)]
                                       for v in range(num_views)]

            # Visualize marked RGB frames and heatmap frames
            self._visualize_generated_frames(
                video_heatmap_frames,
                video_rgb_frames_marked,
                save_path=visualize_save_path
            )

        # Increment counter for next call
        self._debug_img_counter += 1

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

    def mark_peaks_on_images(self,
                             rgb_images: List[List[Image.Image]],
                             peaks: List[List[Tuple[int, int]]],
                             marker_size: int = 5,
                             marker_color: str = 'red',
                             marker_width: int = 2) -> List[List[Image.Image]]:
        """
        Mark peak positions on RGB images.

        Args:
            rgb_images: List[List[PIL.Image]] (T, num_views) - RGB images
            peaks: List[List[Tuple[int, int]]] (T, num_views) - Peak positions (x, y)
            marker_size: Marker size (radius)
            marker_color: Marker color
            marker_width: Marker border width

        Returns:
            marked_images: List[List[PIL.Image]] (T, num_views) - Marked RGB images
        """
        marked_images = []

        for frame_idx in range(len(rgb_images)):
            frame_marked = []
            for view_idx in range(len(rgb_images[frame_idx])):
                # Copy image to avoid modifying original
                img = rgb_images[frame_idx][view_idx].copy()
                draw = ImageDraw.Draw(img)

                # Get peak position
                peak_x, peak_y = peaks[frame_idx][view_idx]

                # Draw cross marker
                # Horizontal line
                draw.line([(peak_x - marker_size, peak_y), (peak_x + marker_size, peak_y)],
                         fill=marker_color, width=marker_width)
                # Vertical line
                draw.line([(peak_x, peak_y - marker_size), (peak_x, peak_y + marker_size)],
                         fill=marker_color, width=marker_width)

                # Draw circle marker
                draw.ellipse([(peak_x - marker_size, peak_y - marker_size),
                             (peak_x + marker_size, peak_y + marker_size)],
                            outline=marker_color, width=marker_width)

                frame_marked.append(img)
            marked_images.append(frame_marked)

        return marked_images

    def _visualize_input_images(self, input_image: List[Image.Image], input_image_rgb: List[Image.Image], prompt: str):
        """
        Visualize input multi-view images.

        Args:
            input_image: List[PIL.Image] - Multi-view heatmaps (num_views,)
            input_image_rgb: List[PIL.Image] - Multi-view RGB images (num_views,)
            prompt: Text prompt
        """
        num_views = len(input_image)

        # Find peaks in input heatmaps
        print("Finding peaks in input heatmaps...")
        input_peaks = []
        for view_idx in range(num_views):
            peak = self.find_peak_position(input_image[view_idx], colormap_name='jet')
            input_peaks.append(peak)

        # Mark peaks on RGB images
        print("Marking peaks on input RGB images...")
        input_rgb_marked = []
        for view_idx in range(num_views):
            img = input_image_rgb[view_idx].copy()
            draw = ImageDraw.Draw(img)
            peak_x, peak_y = input_peaks[view_idx]

            # Draw cross marker
            marker_size = 8
            marker_width = 3
            marker_color = 'red'
            draw.line([(peak_x - marker_size, peak_y), (peak_x + marker_size, peak_y)],
                     fill=marker_color, width=marker_width)
            draw.line([(peak_x, peak_y - marker_size), (peak_x, peak_y + marker_size)],
                     fill=marker_color, width=marker_width)

            # Draw circle marker
            draw.ellipse([(peak_x - marker_size, peak_y - marker_size),
                         (peak_x + marker_size, peak_y + marker_size)],
                        outline=marker_color, width=marker_width)

            input_rgb_marked.append(img)

        # Create visualization
        fig, axes = plt.subplots(2, num_views, figsize=(num_views * 4, 8))

        # Ensure axes is 2D array
        if num_views == 1:
            axes = axes.reshape(2, 1)

        # Plot heatmaps
        for view_idx in range(num_views):
            ax = axes[0, view_idx]
            ax.imshow(input_image[view_idx])
            ax.set_title(f"Heatmap View {view_idx}", fontsize=12, fontweight='bold')
            ax.axis('off')

        # Plot marked RGB images
        for view_idx in range(num_views):
            ax = axes[1, view_idx]
            ax.imshow(input_rgb_marked[view_idx])
            ax.set_title(f"RGB View {view_idx} (Peak Marked)", fontsize=12, fontweight='bold')
            ax.axis('off')

        # Add title
        fig.suptitle(f"Input Images (Multi-View)\nPrompt: {prompt}",
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save image
        save_path = os.path.join(self._debug_img_dir, f"debug_input_{self._debug_img_counter:04d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n{'='*80}")
        print(f"INPUT VISUALIZATION SAVED")
        print(f"{'='*80}")
        print(f"  Location: {save_path}")
        print(f"  Counter: {self._debug_img_counter}")
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
        Visualize generated video frames.

        Args:
            video_heatmap_frames: List[List[PIL.Image]] (num_views, T) - Generated heatmap video frames
            video_rgb_frames: List[List[PIL.Image]] (num_views, T) - Generated RGB video frames
            save_path: Save path (optional)
        """
        num_views = len(video_heatmap_frames)
        num_frames = len(video_heatmap_frames[0])

        # Create save directory
        if save_path is None:
            save_dir = os.path.join(os.path.dirname(__file__), "../../debug_generated_visualization")
            os.makedirs(save_dir, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"generated_frames_{timestamp}.png")
        else:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

        # Create image grid: (num_views * 2) rows × num_frames columns
        fig, axes = plt.subplots(num_views * 2, num_frames, figsize=(num_frames * 3, num_views * 2 * 3))

        # Ensure axes is 2D array
        if num_views * 2 == 1 and num_frames == 1:
            axes = np.array([[axes]])
        elif num_views * 2 == 1:
            axes = axes.reshape(1, -1)
        elif num_frames == 1:
            axes = axes.reshape(-1, 1)

        # Plot each view's frames
        for view_idx in range(num_views):
            # Heatmap row
            heatmap_row = view_idx * 2
            for frame_idx in range(num_frames):
                ax = axes[heatmap_row, frame_idx]
                ax.imshow(video_heatmap_frames[view_idx][frame_idx])
                if frame_idx == 0:
                    ax.set_ylabel(f"View {view_idx}\nHeatmap", fontsize=10, fontweight='bold')
                ax.set_title(f"T={frame_idx}", fontsize=10)
                ax.axis('off')

            # RGB row
            rgb_row = view_idx * 2 + 1
            for frame_idx in range(num_frames):
                ax = axes[rgb_row, frame_idx]
                ax.imshow(video_rgb_frames[view_idx][frame_idx])
                if frame_idx == 0:
                    ax.set_ylabel(f"View {view_idx}\nRGB", fontsize=10, fontweight='bold')
                ax.axis('off')

        # Add title
        fig.suptitle(f"Generated Video Frames\n({num_views} views × {num_frames} frames)",
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save image
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

    def save_results(
        self,
        output: Dict[str, Any],
        output_dir: str,
        sample_id: int,
    ):
        """Save inference results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save output videos
        for key in ['video_heatmap', 'video_rgb']:
            if key in output:
                for view_idx, view_frames in enumerate(output[key]):
                    view_dir = os.path.join(output_dir, f"sample_{sample_id:04d}_{key}_view_{view_idx}")
                    os.makedirs(view_dir, exist_ok=True)

                    for frame_idx, frame in enumerate(view_frames):
                        if isinstance(frame, Image.Image):
                            frame.save(os.path.join(view_dir, f"frame_{frame_idx:04d}.png"))
                        elif isinstance(frame, np.ndarray):
                            cv2.imwrite(
                                os.path.join(view_dir, f"frame_{frame_idx:04d}.png"),
                                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            )

        # Save rotation and gripper predictions if available
        if 'rotation_predictions' in output:
            np.save(
                os.path.join(output_dir, f"sample_{sample_id:04d}_rotation_predictions.npy"),
                output['rotation_predictions']
            )

        if 'gripper_predictions' in output:
            np.save(
                os.path.join(output_dir, f"sample_{sample_id:04d}_gripper_predictions.npy"),
                output['gripper_predictions']
            )

        print(f"Results saved to: {output_dir}/sample_{sample_id:04d}_*")


def visualize_predictions(
    gt_heatmap_video: List[List[Image.Image]],
    pred_heatmap_video: List[List[Image.Image]],
    gt_rgb_video: List[List[Image.Image]],
    pred_rgb_video: List[List[Image.Image]],
    prompt: str,
    dataset_idx: int,
    save_path: str,
    colormap_name: str = 'jet'
):
    """Visualize prediction results."""
    num_frames = len(gt_heatmap_video)
    num_views = len(gt_heatmap_video[0])

    # Calculate peak distances
    peak_distances = []
    for frame_idx in range(num_frames):
        frame_distances = []
        for view_idx in range(num_views):
            gt_peak = extract_heatmap_from_colormap(
                np.array(gt_heatmap_video[frame_idx][view_idx]).astype(np.float32) / 255.0,
                colormap_name
            )
            pred_peak = extract_heatmap_from_colormap(
                np.array(pred_heatmap_video[frame_idx][view_idx]).astype(np.float32) / 255.0,
                colormap_name
            )
            gt_pos = np.unravel_index(np.argmax(gt_peak), gt_peak.shape)
            pred_pos = np.unravel_index(np.argmax(pred_peak), pred_peak.shape)
            distance = np.sqrt((gt_pos[0] - pred_pos[0])**2 + (gt_pos[1] - pred_pos[1])**2)
            frame_distances.append(distance)
        peak_distances.append(frame_distances)

    peak_distances = np.array(peak_distances)
    mean_distance = peak_distances.mean()

    # Create visualization
    fig = plt.figure(figsize=(3*num_frames, 3*num_views*4))
    gs = fig.add_gridspec(num_views*4, num_frames, hspace=0.3, wspace=0.1)

    for view_idx in range(num_views):
        for frame_idx in range(num_frames):
            # GT Heatmap
            ax = fig.add_subplot(gs[view_idx*4, frame_idx])
            ax.imshow(gt_heatmap_video[frame_idx][view_idx])
            if frame_idx == 0:
                ax.set_ylabel(f'GT HM V{view_idx}', fontsize=9)
            if view_idx == 0:
                ax.set_title(f'T{frame_idx}', fontsize=8)
            ax.axis('off')

            # Pred Heatmap
            ax = fig.add_subplot(gs[view_idx*4+1, frame_idx])
            ax.imshow(pred_heatmap_video[frame_idx][view_idx])
            ax.text(0.5, -0.05, f'{peak_distances[frame_idx][view_idx]:.1f}px',
                   ha='center', va='top', transform=ax.transAxes, fontsize=6)
            if frame_idx == 0:
                ax.set_ylabel(f'Pred HM V{view_idx}', fontsize=9)
            ax.axis('off')

            # GT RGB
            ax = fig.add_subplot(gs[view_idx*4+2, frame_idx])
            ax.imshow(gt_rgb_video[frame_idx][view_idx])
            if frame_idx == 0:
                ax.set_ylabel(f'GT RGB V{view_idx}', fontsize=9)
            ax.axis('off')

            # Pred RGB
            ax = fig.add_subplot(gs[view_idx*4+3, frame_idx])
            ax.imshow(pred_rgb_video[frame_idx][view_idx])
            if frame_idx == 0:
                ax.set_ylabel(f'Pred RGB V{view_idx}', fontsize=9)
            ax.axis('off')

    fig.suptitle(f'Sample {dataset_idx}: {prompt[:60]}...\nMean Peak Distance: {mean_distance:.1f}px',
                 fontsize=10, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {save_path}")


def test_on_dataset_mv_view_rot_grip(
    inference_engine: HeatmapInferenceMVViewRotGrip,
    data_root: str,
    output_dir: str,
    test_indices: List[int],
    num_frames: int = 25,
    num_inference_steps: int = 50,
    cfg_scale: float = 1.0,
    sequence_length: int = 24,
    image_size: Tuple[int, int] = (256, 256),
    scene_bounds: List[float] = None,
    transform_augmentation_xyz: List[float] = None,
    transform_augmentation_rpy: List[float] = None,
    wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
    use_merged_pointcloud: bool = False,
    use_different_projection: bool = False,
    trail_start: int = None,
    trail_end: int = None,
):
    """Test inference on dataset.

    Args:
        trail_start: Starting trail number (e.g., 1 for trail_1), None means no limit
        trail_end: Ending trail number (e.g., 50 for trail_50), None means no limit
    """
    from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam import HeatmapDatasetFactory

    os.makedirs(output_dir, exist_ok=True)

    # Set defaults
    if scene_bounds is None:
        scene_bounds = [-0.1, -0.5, -0.1, 0.9, 0.5, 0.9]
    if transform_augmentation_xyz is None:
        transform_augmentation_xyz = [0.0, 0.0, 0.0]
    if transform_augmentation_rpy is None:
        transform_augmentation_rpy = [0.0, 0.0, 0.0]

    print(f"\n=== Testing on Dataset (MV View + Rot/Grip) ===")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Test indices: {test_indices}")

    # Print trail range if specified
    if trail_start is not None or trail_end is not None:
        print(f"Trail range: trail_{trail_start or 'start'} to trail_{trail_end or 'end'}")

    # Create dataset
    dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
        data_root=data_root,
        sequence_length=sequence_length,
        step_interval=1,
        min_trail_length=5,
        image_size=image_size,
        sigma=1.5,
        augmentation=False,
        mode="test",
        scene_bounds=scene_bounds,
        transform_augmentation_xyz=transform_augmentation_xyz,
        transform_augmentation_rpy=transform_augmentation_rpy,
        debug=False,
        colormap_name="jet",
        repeat=1,
        wan_type=wan_type,
        use_merged_pointcloud=use_merged_pointcloud,
        use_different_projection=use_different_projection,
        trail_start=trail_start,
        trail_end=trail_end,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Debug: Show trail info for each test index
    print(f"\n=== Test Indices Trail Information ===")
    for dataset_idx in test_indices:
        if dataset_idx < len(dataset.robot_dataset.valid_samples):
            sample = dataset.robot_dataset.valid_samples[dataset_idx]
            trail_info = sample['trail_info']
            print(f"  Index {dataset_idx}: {trail_info['trail_name']}, start_step={sample['start_step']}, num_steps={trail_info['num_steps']}")
    print("="*60 + "\n")

    all_heatmap_distances = []
    all_rotation_errors = []  # Store rotation errors (degrees)
    all_gripper_predictions = []  # Store predicted gripper states
    all_gripper_ground_truths = []  # Store ground truth gripper states

    for idx, dataset_idx in enumerate(test_indices):
        if dataset_idx >= len(dataset):
            print(f"Warning: Index {dataset_idx} out of range, skipping...")
            continue

        print(f"\n[{idx+1}/{len(test_indices)}] Processing dataset index {dataset_idx}...")

        try:
            sample = dataset[dataset_idx]

            prompt = sample['prompt']
            input_image = sample['input_image']
            input_image_rgb = sample['input_image_rgb']
            gt_heatmap_video = sample['video']
            gt_rgb_video = sample['input_video_rgb']

            # Get initial rotation and gripper if available
            # Dataset returns these as 'start_rotation' and 'start_gripper'
            start_rotation_bins = sample.get('start_rotation')  # (3,) - discrete bin indices [0, num_bins)
            start_gripper = sample.get('start_gripper')  # scalar - 0 or 1

            # Convert rotation bins to degrees for inference
            # Bins are in range [0, 360/rotation_resolution), need to convert to degrees in [-180, 180)
            initial_rotation = None
            initial_gripper = None
            if start_rotation_bins is not None:
                if isinstance(start_rotation_bins, torch.Tensor):
                    start_rotation_bins = start_rotation_bins.cpu().numpy()
                # Convert: bin * resolution gives [0, 360), then shift to [-180, 180)
                initial_rotation = start_rotation_bins * inference_engine.rotation_resolution
                initial_rotation = ((initial_rotation + 180) % 360) - 180
                initial_rotation = initial_rotation.astype(np.float32)
            if start_gripper is not None:
                if isinstance(start_gripper, torch.Tensor):
                    initial_gripper = start_gripper.item()
                else:
                    initial_gripper = int(start_gripper)

            print(f"  Prompt: {prompt}")
            if initial_rotation is not None:
                print(f"  Initial rotation (degrees): [{initial_rotation[0]:.1f}, {initial_rotation[1]:.1f}, {initial_rotation[2]:.1f}]")
            if initial_gripper is not None:
                print(f"  Initial gripper: {initial_gripper}")

            # Run inference
            output = inference_engine.run_inference(
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                prompt=prompt,
                initial_rotation=initial_rotation,
                initial_gripper=initial_gripper,
                num_frames=num_frames,
                height=image_size[0],
                width=image_size[1],
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                seed=dataset_idx,
            )

            # Get predicted videos
            pred_heatmap = output['video_heatmap']
            pred_rgb = output['video_rgb']

            # Convert format if needed: [view][time] -> [time][view]
            if len(pred_heatmap[0]) > 3:
                num_views = len(pred_heatmap)
                num_frames_out = len(pred_heatmap[0])
                pred_heatmap_video = [[pred_heatmap[v][t] for v in range(num_views)] for t in range(num_frames_out)]
                pred_rgb_video = [[pred_rgb[v][t] for v in range(num_views)] for t in range(num_frames_out)]
            else:
                pred_heatmap_video = pred_heatmap
                pred_rgb_video = pred_rgb
                num_views = len(pred_heatmap_video[0])
                num_frames_out = len(pred_heatmap_video)

            # Calculate heatmap peak distances
            gt_peaks = inference_engine.find_peaks_batch(gt_heatmap_video, 'jet')
            pred_peaks = inference_engine.find_peaks_batch(pred_heatmap_video, 'jet')

            num_frames_to_compare = min(len(gt_peaks), len(pred_peaks))
            frame_distances = []
            for frame_idx in range(num_frames_to_compare):
                for view_idx in range(num_views):
                    dist = inference_engine.calculate_peak_distance(
                        pred_peaks[frame_idx][view_idx],
                        gt_peaks[frame_idx][view_idx]
                    )
                    frame_distances.append(dist)

            mean_distance = np.mean(frame_distances)
            all_heatmap_distances.extend(frame_distances)

            print(f"  Mean peak distance: {mean_distance:.2f}px")

            # Process rotation/gripper predictions if available
            if 'rotation_predictions' in output and 'gripper_predictions' in output:
                rotation_preds = output['rotation_predictions']  # (T, 3) in degrees
                gripper_preds = output['gripper_predictions']  # (T,)

                print(f"  Rotation predictions available: {len(rotation_preds)} frames")
                print(f"    First 3 rotations: {rotation_preds[:3].tolist()}")
                print(f"    First 3 grippers: {gripper_preds[:3].tolist()}")

                # Get ground truth if available
                gt_rotation_bins = sample.get('rotation_targets')  # (T, 3) - discrete bin indices
                gt_grippers = sample.get('gripper_targets')  # (T,) - absolute values
                gt_gripper_changes = sample.get('gripper_change_targets')  # (T,) - changes relative to initial

                if gt_rotation_bins is not None and gt_grippers is not None:
                    # Convert to numpy if needed
                    if isinstance(gt_rotation_bins, torch.Tensor):
                        gt_rotation_bins = gt_rotation_bins.cpu().numpy()
                    if isinstance(gt_grippers, torch.Tensor):
                        gt_grippers = gt_grippers.cpu().numpy()
                    if gt_gripper_changes is not None and isinstance(gt_gripper_changes, torch.Tensor):
                        gt_gripper_changes = gt_gripper_changes.cpu().numpy()

                    # Convert ground truth rotation bins to degrees
                    # Bins are in range [0, 360/rotation_resolution), convert to degrees in [-180, 180)
                    gt_rotations = gt_rotation_bins * inference_engine.rotation_resolution
                    gt_rotations = ((gt_rotations + 180) % 360) - 180

                    # Debug: Check gripper change logic
                    if gt_gripper_changes is not None:
                        print(f"    GT gripper changes (first 3): {gt_gripper_changes[:3]}")
                        # Verify: gripper_change should match (gt_gripper != initial_gripper)
                        expected_changes = (gt_grippers != initial_gripper).astype(np.int64)
                        print(f"    Expected changes from GT (first 3): {expected_changes[:3]}")
                        if not np.array_equal(gt_gripper_changes, expected_changes):
                            print(f"    WARNING: gt_gripper_changes does not match expected!")

                    # Calculate errors (compare available frames)
                    num_frames_rot = min(len(rotation_preds), len(gt_rotations))
                    rotation_errors = np.abs(rotation_preds[:num_frames_rot] - gt_rotations[:num_frames_rot])

                    # Handle circular rotation difference (e.g., -179 vs 180 should be close)
                    rotation_errors = np.minimum(rotation_errors, 360 - rotation_errors)

                    mean_rotation_error = np.mean(rotation_errors, axis=0)  # Per axis
                    all_rotation_errors.extend(rotation_errors.flatten())  # Store all errors

                    # Gripper accuracy
                    num_frames_grip = min(len(gripper_preds), len(gt_grippers))
                    gripper_matches = (gripper_preds[:num_frames_grip] == gt_grippers[:num_frames_grip])
                    all_gripper_predictions.extend(gripper_preds[:num_frames_grip].tolist())
                    all_gripper_ground_truths.extend(gt_grippers[:num_frames_grip].tolist())

                    # Additional debug: show mismatches
                    if gripper_matches.sum() < len(gripper_matches):
                        mismatch_indices = np.where(~gripper_matches)[0]
                        print(f"    Gripper mismatches at frames: {mismatch_indices[:10]}")  # Show first 10
                        for i in mismatch_indices[:3]:  # Show details for first 3
                            print(f"      Frame {i}: pred={gripper_preds[i]}, gt={gt_grippers[i]}, initial={initial_gripper}")

                    print(f"    Ground truth rotations (first 3): {gt_rotations[:3].tolist()}")
                    print(f"    Ground truth grippers (first 3): {gt_grippers[:3].tolist()}")
                    print(f"    Rotation error (roll/pitch/yaw): {mean_rotation_error[0]:.2f}°/{mean_rotation_error[1]:.2f}°/{mean_rotation_error[2]:.2f}°")
                    print(f"    Gripper accuracy: {gripper_matches.sum()}/{len(gripper_matches)} ({100*gripper_matches.mean():.1f}%)")
                else:
                    print(f"    Ground truth not available for rotation/gripper evaluation")
            else:
                print(f"  Rotation/gripper predictions not available (likely skipped)")

            # Save visualization
            vis_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_comparison.png')
            visualize_predictions(
                gt_heatmap_video=gt_heatmap_video[:num_frames_to_compare],
                pred_heatmap_video=pred_heatmap_video[:num_frames_to_compare],
                gt_rgb_video=gt_rgb_video[:num_frames_to_compare],
                pred_rgb_video=pred_rgb_video[:num_frames_to_compare],
                prompt=prompt,
                dataset_idx=dataset_idx,
                save_path=vis_path,
            )

        except Exception as e:
            print(f"  Error processing sample {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Overall statistics
    print(f"\n=== OVERALL EVALUATION RESULTS ===")

    # Heatmap statistics
    if all_heatmap_distances:
        mean_dist = np.mean(all_heatmap_distances)
        std_dist = np.std(all_heatmap_distances)
        print(f"\nHeatmap Evaluation:")
        print(f"  Total frames evaluated: {len(all_heatmap_distances)}")
        print(f"  Mean Peak Distance: {mean_dist:.2f} +/- {std_dist:.2f}px")

    # Rotation statistics
    if all_rotation_errors:
        mean_rot_error = np.mean(all_rotation_errors)
        std_rot_error = np.std(all_rotation_errors)
        print(f"\nRotation Evaluation:")
        print(f"  Total rotation predictions: {len(all_rotation_errors)}")
        print(f"  Mean Rotation Error: {mean_rot_error:.2f} +/- {std_rot_error:.2f}°")

    # Gripper statistics
    if all_gripper_predictions and all_gripper_ground_truths:
        gripper_preds_array = np.array(all_gripper_predictions)
        gripper_gts_array = np.array(all_gripper_ground_truths)
        gripper_correct = (gripper_preds_array == gripper_gts_array).sum()
        gripper_total = len(gripper_preds_array)
        gripper_accuracy = 100 * gripper_correct / gripper_total if gripper_total > 0 else 0

        print(f"\nGripper Evaluation:")
        print(f"  Total gripper predictions: {gripper_total}")
        print(f"  Accuracy: {gripper_correct}/{gripper_total} ({gripper_accuracy:.2f}%)")

    # Save stats to file
    stats_path = os.path.join(output_dir, 'evaluation_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("MV View + Rot/Grip Evaluation Results\n")
        f.write("="*60 + "\n\n")

        # Heatmap stats
        if all_heatmap_distances:
            f.write("Heatmap Evaluation:\n")
            f.write(f"  Total frames evaluated: {len(all_heatmap_distances)}\n")
            f.write(f"  Mean Peak Distance: {mean_dist:.2f} +/- {std_dist:.2f}px\n\n")

        # Rotation stats
        if all_rotation_errors:
            f.write("Rotation Evaluation:\n")
            f.write(f"  Total rotation predictions: {len(all_rotation_errors)}\n")
            f.write(f"  Mean Rotation Error: {mean_rot_error:.2f} +/- {std_rot_error:.2f}°\n\n")

        # Gripper stats
        if all_gripper_predictions and all_gripper_ground_truths:
            f.write("Gripper Evaluation:\n")
            f.write(f"  Total gripper predictions: {gripper_total}\n")
            f.write(f"  Accuracy: {gripper_correct}/{gripper_total} ({gripper_accuracy:.2f}%)\n\n")

    print(f"\nResults saved to: {output_dir}")
    print(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="MV View Inference with Rotation and Gripper Prediction")

    # Model paths
    parser.add_argument("--model_base_path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="Path to LoRA/full finetune checkpoint")
    parser.add_argument("--rot_grip_checkpoint", type=str, default=None,
                       help="Path to rotation/gripper predictor checkpoint (optional)")

    # Model config
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
                       help="Model type")
    parser.add_argument("--use_dual_head", action="store_true",
                       help="Use dual head mode")
    parser.add_argument("--is_full_finetune", action="store_true",
                       help="Whether checkpoint is full finetune")

    # Rotation/gripper parameters
    parser.add_argument("--rotation_resolution", type=float, default=5.0,
                       help="Rotation angle resolution (degrees)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden layer dimension")
    parser.add_argument("--num_rotation_bins", type=int, default=72,
                       help="Number of rotation bins")
    parser.add_argument("--num_history_frames", type=int, default=1,
                       help="Number of history frames")
    parser.add_argument("--local_feat_size", type=int, default=5,
                       help="Local feature extraction size")
    parser.add_argument("--use_initial_gripper_state", action="store_true",
                       help="Use initial gripper state as input")

    # Inference params
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--num_frames", type=int, default=25,
                       help="Number of output frames")
    parser.add_argument("--height", type=int, default=256,
                       help="Output height")
    parser.add_argument("--width", type=int, default=256,
                       help="Output width")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                       help="Classifier-free guidance scale")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")

    # Test data
    parser.add_argument("--data_root", type=str, default=None,
                       help="Data root for test samples")
    parser.add_argument("--test_indices", type=str, default="0",
                       help="Test sample indices (comma-separated)")

    # Dataset config
    parser.add_argument("--sequence_length", type=int, default=24,
                       help="Sequence length")
    parser.add_argument("--scene_bounds", type=str, default="-0.1,-0.5,-0.1,0.9,0.5,0.9",
                       help='Scene bounds as comma-separated values')
    parser.add_argument("--transform_augmentation_xyz", type=str, default="0.0,0.0,0.0",
                       help='Transform augmentation for xyz')
    parser.add_argument("--transform_augmentation_rpy", type=str, default="0.0,0.0,0.0",
                       help='Transform augmentation for rpy')
    parser.add_argument("--use_merged_pointcloud", action='store_true',
                       help='Use merged pointcloud from 3 cameras')
    parser.add_argument("--use_different_projection", action='store_true',
                       help='Use different projection mode')
    parser.add_argument("--trail_start", type=int, default=None,
                       help='Starting trail number (e.g., 1 for trail_1), None means no limit')
    parser.add_argument("--trail_end", type=int, default=None,
                       help='Ending trail number (e.g., 50 for trail_50), None means no limit')

    args = parser.parse_args()

    # Parse parameters
    test_indices = [int(x.strip()) for x in args.test_indices.split(',')]
    scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
    transform_xyz = [float(x.strip()) for x in args.transform_augmentation_xyz.split(',')]
    transform_rpy = [float(x.strip()) for x in args.transform_augmentation_rpy.split(',')]

    print("=== MV View Inference with Rot/Grip ===")
    print(f"LoRA Checkpoint: {args.lora_checkpoint}")
    print(f"Rot/Grip Checkpoint: {args.rot_grip_checkpoint if args.rot_grip_checkpoint else '(Not provided)'}")
    print(f"Model Type: {args.wan_type}")
    print(f"Dual Head Mode: {args.use_dual_head}")
    print(f"Is Full Finetune: {args.is_full_finetune}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {args.output_dir}")
    print()

    # Create inference engine
    inference = HeatmapInferenceMVViewRotGrip(
        model_base_path=args.model_base_path,
        lora_checkpoint=args.lora_checkpoint,
        rot_grip_checkpoint=args.rot_grip_checkpoint,
        wan_type=args.wan_type,
        use_dual_head=args.use_dual_head,
        use_merged_pointcloud=args.use_merged_pointcloud,
        use_different_projection=args.use_different_projection,
        rotation_resolution=args.rotation_resolution,
        hidden_dim=args.hidden_dim,
        num_rotation_bins=args.num_rotation_bins,
        num_history_frames=args.num_history_frames,
        local_feat_size=args.local_feat_size,
        use_initial_gripper_state=args.use_initial_gripper_state,
        device=args.device,
        is_full_finetune=args.is_full_finetune,
    )

    if args.data_root is None:
        # No dataset: dummy test
        print("No data_root provided, creating dummy test inputs...")
        num_views = 3
        for sample_id in test_indices:
            input_images = [
                Image.new("RGB", (args.width, args.height), color="gray")
                for _ in range(num_views)
            ]
            input_images_rgb = [
                Image.new("RGB", (args.width, args.height), color="gray")
                for _ in range(num_views)
            ]

            # Dummy initial states
            initial_rotation = np.array([0.0, 0.0, 0.0])
            initial_gripper = 0

            output = inference.run_inference(
                input_image=input_images,
                input_image_rgb=input_images_rgb,
                prompt="robot arm manipulation",
                initial_rotation=initial_rotation,
                initial_gripper=initial_gripper,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )

            inference.save_results(output, args.output_dir, sample_id)
    else:
        # Test on dataset
        test_on_dataset_mv_view_rot_grip(
            inference_engine=inference,
            data_root=args.data_root,
            output_dir=args.output_dir,
            test_indices=test_indices,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sequence_length=args.sequence_length,
            image_size=(args.height, args.width),
            scene_bounds=scene_bounds,
            transform_augmentation_xyz=transform_xyz,
            transform_augmentation_rpy=transform_rpy,
            wan_type=args.wan_type,
            use_merged_pointcloud=args.use_merged_pointcloud,
            use_different_projection=args.use_different_projection,
            trail_start=args.trail_start,
            trail_end=args.trail_end,
        )

    print("\nInference completed!")
    print(f"Results saved to: {args.output_dir}")


# ============================================================================
# Helper Functions for Position Extraction (used by Server)
# ============================================================================

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


if __name__ == "__main__":
    main()
