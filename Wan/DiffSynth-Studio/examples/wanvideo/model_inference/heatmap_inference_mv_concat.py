"""
Inference Script for Multi-View Token Concatenation Model

This script performs inference using WanModel_mv_concat which uses
token concatenation instead of multi-view attention modules.

参考自: heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed.py
"""

import torch
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

from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_concat import (
    WanVideoPipelineMVConcat,
    convert_wan_model_to_mv_concat
)
from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import ModelConfig
from diffsynth.models.wan_video_dit_mv_concat import WanModel_mv_concat
from diffsynth.models import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap


def get_root_path():
    """自动检测BridgeVLA根目录"""
    possible_paths = [
        "/DATA/disk1/lpy/BridgeVLA_dev",
        "/home/lpy/BridgeVLA_dev",
        "/DATA/disk0/lpy/BridgeVLA_dev",
        "/DATA/disk1/lpy_a100_4/BridgeVLA_dev",
        "/DATA/disk1/lpy_a100_1/BridgeVLA_dev"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")


ROOT_PATH = get_root_path()


class HeatmapInferenceMVConcat:
    """
    Inference class for multi-view token concatenation model.
    """

    def __init__(
        self,
        model_base_path: str,
        lora_checkpoint: str,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_CONCAT",
        use_dual_head: bool = True,
        num_views: int = 3,
        device: str = "cuda",
        is_full_finetune: bool = False,
        torch_dtype=torch.bfloat16,
    ):
        """
        Initialize inference.

        Args:
            model_base_path: Path to base model
            lora_checkpoint: Path to LoRA checkpoint
            wan_type: Model type
            use_dual_head: Whether to use dual head mode
            num_views: Number of views
            device: Device to use
            is_full_finetune: Whether checkpoint is full finetune
            torch_dtype: Torch dtype
        """
        self.device = device
        self.wan_type = wan_type
        self.use_dual_head = use_dual_head
        self.num_views = num_views
        self.is_full_finetune = is_full_finetune
        self.torch_dtype = torch_dtype
        self.lora_checkpoint = lora_checkpoint

        print("="*60)
        print("Initializing MV Concat Inference")
        print("="*60)
        print(f"  Model base path: {model_base_path}")
        print(f"  LoRA checkpoint: {lora_checkpoint}")
        print(f"  WAN type: {wan_type}")
        print(f"  Use dual head: {use_dual_head}")
        print(f"  Num views: {num_views}")
        print(f"  Is full finetune: {is_full_finetune}")
        print("="*60)

        # Load pipeline
        self.pipe = self._load_pipeline(model_base_path, lora_checkpoint)

    def _load_pipeline(self, model_base_path: str, lora_checkpoint: str):
        """
        Load the pipeline with model and LoRA weights.
        """
        # Model configs
        model_configs = [
            ModelConfig(
                path=[
                    f"{model_base_path}/diffusion_pytorch_model-00001-of-00003.safetensors",
                    f"{model_base_path}/diffusion_pytorch_model-00002-of-00003.safetensors",
                    f"{model_base_path}/diffusion_pytorch_model-00003-of-00003.safetensors",
                ],
            ),
            ModelConfig(
                path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth",
            ),
            ModelConfig(
                path=f"{model_base_path}/Wan2.2_VAE.pth",
            ),
        ]

        print("\nStep 1: Loading base pipeline...")
        pipe = WanVideoPipelineMVConcat.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,  # Load directly to GPU like original inference script
            model_configs=model_configs,
            wan_type=self.wan_type,
            use_dual_head=self.use_dual_head,
            num_views=self.num_views
        )

        # Convert to MV Concat model if needed
        print("\nStep 2: Converting to MV Concat model...")
        if not isinstance(pipe.dit, WanModel_mv_concat):
            print("  Converting model to WanModel_mv_concat...")
            pipe.dit = convert_wan_model_to_mv_concat(
                pipe.dit,
                use_dual_head=self.use_dual_head,
                num_views=self.num_views
            )
        else:
            print("  Model is already WanModel_mv_concat")

        # Load checkpoint weights
        print("\nStep 3: Loading checkpoint weights...")
        if lora_checkpoint and os.path.exists(lora_checkpoint):
            if self.is_full_finetune:
                # Full finetune checkpoint - load all weights directly
                self._load_full_finetune_checkpoint(pipe, lora_checkpoint)
            else:
                # LoRA checkpoint - load base weights first, then apply LoRA
                self._load_lora_with_base_weights(pipe, lora_checkpoint)
        else:
            print(f"  Warning: Checkpoint not found: {lora_checkpoint}")

        # Ensure models are on device and in eval mode
        print(f"\nStep 4: Setting models to eval mode on {self.device}...")
        # Models should already be on device from from_pretrained, but ensure eval mode
        if pipe.dit is not None:
            pipe.dit.eval()
        if pipe.vae is not None:
            pipe.vae.eval()
        if pipe.text_encoder is not None:
            pipe.text_encoder.eval()

        # Debug: Print critical model attributes
        print("\n" + "="*60)
        print("DEBUG: Critical Model Attributes")
        print("="*60)
        print(f"  dit.fuse_vae_embedding_in_latents: {pipe.dit.fuse_vae_embedding_in_latents}")
        print(f"  dit.require_vae_embedding: {pipe.dit.require_vae_embedding}")
        print(f"  dit.require_clip_embedding: {pipe.dit.require_clip_embedding}")
        print(f"  dit.seperated_timestep: {pipe.dit.seperated_timestep}")
        print(f"  dit.use_dual_head: {pipe.dit.use_dual_head}")
        print(f"  dit.num_views: {pipe.dit.num_views}")
        print("="*60)

        print("\n" + "="*60)
        print("Pipeline loaded successfully!")
        print("="*60)
        return pipe

    def _load_checkpoint_weights(self, pipe, checkpoint_path: str):
        """
        加载checkpoint中的非LoRA权重（head, patch_embedding等）

        Args:
            pipe: Pipeline对象
            checkpoint_path: checkpoint文件路径
        """
        try:
            print(f"  Loading state dict from: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)

            # 分类checkpoint中的权重
            head_weights = {}
            patch_embedding_weights = {}
            modulation_weights = {}
            other_weights = {}

            for key, value in state_dict.items():
                # 跳过LoRA相关的权重
                if 'lora' in key.lower():
                    continue

                # 筛选head相关的权重（包括dual head）
                if any(pattern in key for pattern in ['head_rgb', 'head_heatmap', 'head.']):
                    if 'attention' not in key.lower() and 'attn' not in key.lower():
                        head_weights[key] = value

                # 筛选patch_embedding相关的权重
                elif 'patch_embedding' in key or 'patch_embed' in key:
                    patch_embedding_weights[key] = value

                # 筛选modulation参数
                elif 'modulation' in key and 'mvs' not in key:
                    modulation_weights[key] = value

            print(f"  Found {len(head_weights)} head weights")
            print(f"  Found {len(patch_embedding_weights)} patch_embedding weights")
            print(f"  Found {len(modulation_weights)} modulation weights")

            if head_weights:
                print("  Head keys (sample):")
                for key in list(head_weights.keys())[:5]:
                    print(f"    - {key}")

            # 合并所有需要加载的权重
            weights_to_load = {}
            weights_to_load.update(head_weights)
            weights_to_load.update(patch_embedding_weights)
            weights_to_load.update(modulation_weights)

            if not weights_to_load:
                print("  Warning: No non-LoRA weights found in checkpoint")
                return

            # 清理权重key（移除可能的前缀）
            weights_clean = {}
            for key, value in weights_to_load.items():
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value

            print(f"  Loading {len(weights_clean)} non-LoRA weights into DIT model...")

            # 加载到DIT模型中
            missing_keys, unexpected_keys = pipe.dit.load_state_dict(
                weights_clean, strict=False
            )

            # 统计成功加载的权重
            loaded_keys = set(weights_clean.keys()) - set(unexpected_keys)

            print(f"    Successfully loaded {len(loaded_keys)}/{len(weights_clean)} weights")

            if missing_keys:
                relevant_missing = [k for k in missing_keys if any(p in k for p in ['head', 'patch_embedding', 'modulation'])]
                if relevant_missing:
                    print(f"    Warning: {len(relevant_missing)} relevant keys not found in model:")
                    for key in relevant_missing[:5]:
                        print(f"      - {key}")

            if unexpected_keys:
                print(f"    Info: {len(unexpected_keys)} unexpected keys (first 5):")
                for key in unexpected_keys[:5]:
                    print(f"      - {key}")

            print("  Non-LoRA weights loaded successfully!")

            # Debug: Print head weights statistics after loading
            if hasattr(pipe.dit, 'head_rgb') and pipe.dit.head_rgb is not None:
                head_rgb_mean = pipe.dit.head_rgb.head.weight.mean().item()
                print(f"  [DEBUG] After loading: head_rgb.head.weight.mean() = {head_rgb_mean:.6f}")
            if hasattr(pipe.dit, 'head_heatmap') and pipe.dit.head_heatmap is not None:
                head_heatmap_mean = pipe.dit.head_heatmap.head.weight.mean().item()
                print(f"  [DEBUG] After loading: head_heatmap.head.weight.mean() = {head_heatmap_mean:.6f}")

        except Exception as e:
            print(f"  Warning: Failed to load non-LoRA weights: {e}")
            import traceback
            traceback.print_exc()

    def _load_lora_with_base_weights(self, pipe, checkpoint_path: str, alpha: float = 1.0):
        """
        加载LoRA checkpoint：先加载base weights，再应用LoRA

        Args:
            pipe: Pipeline对象
            checkpoint_path: checkpoint文件路径
            alpha: LoRA alpha
        """
        print("  Loading checkpoint with LoRA logic...")

        # Step 1: 先加载所有非LoRA权重
        print("\n  Step 3.1: Loading non-LoRA weights (head, patch_embedding, etc.)...")
        self._load_checkpoint_weights(pipe, checkpoint_path)

        # Step 2: 加载LoRA权重
        print("\n  Step 3.2: Loading and applying LoRA weights...")

        # DEBUG: Record weight before LoRA
        weight_before = pipe.dit.blocks[0].self_attn.q.weight.clone().mean().item()
        print(f"    [DEBUG] blocks.0.self_attn.q weight mean BEFORE LoRA: {weight_before:.8f}")

        pipe.load_lora(pipe.dit, checkpoint_path, alpha=alpha)

        # DEBUG: Record weight after LoRA
        weight_after = pipe.dit.blocks[0].self_attn.q.weight.mean().item()
        print(f"    [DEBUG] blocks.0.self_attn.q weight mean AFTER LoRA: {weight_after:.8f}")
        print(f"    [DEBUG] Weight change: {weight_after - weight_before:.8f}")

        # DEBUG: Check patch_embedding
        if hasattr(pipe.dit, 'patch_embedding'):
            pe_weight = pipe.dit.patch_embedding.weight
            print(f"\n    [DEBUG] patch_embedding after loading:")
            print(f"      shape: {pe_weight.shape}")
            print(f"      mean: {pe_weight.mean().item():.8f}")
            print(f"      first 48ch mean: {pe_weight[:, :48].mean().item():.8f}")
            print(f"      second 48ch mean: {pe_weight[:, 48:].mean().item():.8f}")
            print(f"      EXPECTED first 48ch: 0.00005031, second 48ch: 0.00000447")

        print("    LoRA weights loaded and applied")

        print("\n  Checkpoint loaded successfully!")

    def _load_full_finetune_checkpoint(self, pipe, checkpoint_path: str):
        """
        加载全量微调的checkpoint

        Args:
            pipe: Pipeline对象
            checkpoint_path: checkpoint文件路径
        """
        try:
            print(f"  Loading full finetune checkpoint: {checkpoint_path}")

            # 加载checkpoint
            state_dict = load_state_dict(checkpoint_path)
            print(f"    Loaded {len(state_dict)} keys from checkpoint")

            # 筛选dit相关权重（排除lora相关）
            dit_weights = {}
            lora_keys_count = 0
            for key, value in state_dict.items():
                if 'lora' in key.lower():
                    lora_keys_count += 1
                    continue
                dit_weights[key] = value

            print(f"    Filtered {len(dit_weights)} DIT weights (skipped {lora_keys_count} LoRA keys)")

            # 清理权重key
            weights_clean = {}
            for key, value in dit_weights.items():
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value

            # 加载到DIT模型中
            missing_keys, unexpected_keys = pipe.dit.load_state_dict(
                weights_clean, strict=False
            )

            loaded_count = len(weights_clean) - len(unexpected_keys)
            print(f"    Successfully loaded {loaded_count}/{len(weights_clean)} weights")

            if missing_keys:
                important_missing = [k for k in missing_keys if any(p in k for p in
                    ['patch_embedding', 'head', 'self_attn', 'cross_attn', 'ffn'])]
                if important_missing:
                    print(f"    Warning: {len(important_missing)} important keys not found:")
                    for key in important_missing[:5]:
                        print(f"      - {key}")

            print("  Full finetune checkpoint loaded successfully!")

        except Exception as e:
            print(f"  Error: Failed to load full finetune checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise

    @torch.no_grad()
    def run_inference(
        self,
        input_image: List[Image.Image],  # [num_views] list of heatmap PIL Images
        input_image_rgb: List[Image.Image],  # [num_views] list of RGB PIL Images
        prompt: str = "robot arm manipulation",
        num_frames: int = 25,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        cfg_scale: float = 1.0,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Run inference.

        NOTE: MV Concat model is expected to be ~9x slower than single-view models
        because token concatenation creates 3x longer sequences, and self-attention
        is O(n^2). If original takes ~3s/step, expect ~27-30s/step for MV Concat.

        Args:
            input_image: List of input heatmap images for each view
            input_image_rgb: List of input RGB images for each view
            prompt: Text prompt
            num_frames: Number of output frames
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed

        Returns:
            Dictionary containing output RGB and heatmap videos
        """
        import time
        print(f"\nRunning inference with {len(input_image)} views...")
        print(f"  Prompt: {prompt}")
        print(f"  Frames: {num_frames}, Size: {height}x{width}")
        print(f"  Steps: {num_inference_steps}, CFG: {cfg_scale}, Seed: {seed}")
        print(f"  NOTE: MV Concat uses 3x longer sequences (token concat)")
        print(f"        Expected ~9x slower than single-view due to O(n^2) attention")

        start_time = time.time()

        # Run pipeline
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
        print(f"\nInference completed in {elapsed_time:.2f}s ({elapsed_time/num_inference_steps:.2f}s/step)")

        return output

    def find_peak_position(self, heatmap_image: Image.Image, colormap_name: str = 'jet') -> Tuple[int, int]:
        """找到热力图的峰值位置"""
        heatmap_image_np = np.array(heatmap_image).astype(np.float32) / 255.0
        heatmap_array = extract_heatmap_from_colormap(heatmap_image_np, colormap_name)
        max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
        return (max_pos[1], max_pos[0])  # (x, y) format

    def find_peaks_batch(self, heatmap_images: List[List[Image.Image]], colormap_name: str = 'jet') -> List[List[Tuple[int, int]]]:
        """批量计算峰值位置"""
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
        """计算两个峰值之间的欧氏距离"""
        return np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    def save_results(
        self,
        output: Dict[str, Any],
        output_dir: str,
        sample_id: int,
    ):
        """
        Save inference results.

        Args:
            output: Inference output
            output_dir: Output directory
            sample_id: Sample ID for naming
        """
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
    """
    可视化预测结果

    Args:
        gt_heatmap_video: List[List[PIL.Image]] (T, num_views) - Ground truth heatmaps
        pred_heatmap_video: List[List[PIL.Image]] (T, num_views) - Predicted heatmaps
        gt_rgb_video: List[List[PIL.Image]] (T, num_views) - Ground truth RGB
        pred_rgb_video: List[List[PIL.Image]] (T, num_views) - Predicted RGB
        prompt: Text prompt
        dataset_idx: Dataset index
        save_path: Save path
        colormap_name: Colormap name
    """
    num_frames = len(gt_heatmap_video)
    num_views = len(gt_heatmap_video[0])

    # 计算峰值距离
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

    # 创建可视化
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


def test_on_dataset_mv_concat(
    inference_engine: HeatmapInferenceMVConcat,
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
    wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_CONCAT",
    use_different_projection: bool = False,
):
    """
    在数据集上测试推理

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
    from diffsynth.trainers.heatmap_dataset_mv_concat import HeatmapDatasetFactory

    os.makedirs(output_dir, exist_ok=True)

    # 设置默认值
    if scene_bounds is None:
        scene_bounds = [-0.1, -0.5, -0.1, 0.9, 0.5, 0.9]
    if transform_augmentation_xyz is None:
        transform_augmentation_xyz = [0.0, 0.0, 0.0]
    if transform_augmentation_rpy is None:
        transform_augmentation_rpy = [0.0, 0.0, 0.0]

    print(f"\n=== Testing on Dataset (MV Concat) ===")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Test indices: {test_indices}")
    print(f"WAN type: {wan_type}")
    print(f"Scene bounds: {scene_bounds}")
    print(f"Use different projection: {use_different_projection}")

    # 创建数据集
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
        use_different_projection=use_different_projection,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    all_heatmap_distances = []

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

            # GT视频
            gt_heatmap_video = sample['video']  # List[List[PIL.Image]] (T, num_views)
            gt_rgb_video = sample['input_video_rgb']  # List[List[PIL.Image]] (T, num_views)

            print(f"  Prompt: {prompt}")

            # 执行推理
            output = inference_engine.run_inference(
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                prompt=prompt,
                num_frames=num_frames,
                height=image_size[0],
                width=image_size[1],
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                seed=dataset_idx,
            )

            # 获取预测视频
            pred_heatmap = output['video_heatmap']  # [view][time] 或 [time][view]
            pred_rgb = output['video_rgb']

            # 检测并转换格式: [view][time] -> [time][view]
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

            # 计算heatmap峰值距离
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

            # 保存可视化
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

    # 总体统计
    if all_heatmap_distances:
        mean_dist = np.mean(all_heatmap_distances)
        std_dist = np.std(all_heatmap_distances)

        print(f"\n=== OVERALL EVALUATION RESULTS ===")
        print(f"Total frames evaluated: {len(all_heatmap_distances)}")
        print(f"Mean Peak Distance: {mean_dist:.2f} +/- {std_dist:.2f}px")

        # 保存统计
        stats_path = os.path.join(output_dir, 'evaluation_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("MV Concat Evaluation Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total frames evaluated: {len(all_heatmap_distances)}\n")
            f.write(f"Mean Peak Distance: {mean_dist:.2f} +/- {std_dist:.2f}px\n")

        print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MV Concat Inference")

    # Model paths
    parser.add_argument("--model_base_path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="Path to LoRA/full finetune checkpoint")

    # Model config
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_CONCAT",
                       help="Model type")
    parser.add_argument("--use_dual_head", action="store_true",
                       help="Use dual head mode")
    parser.add_argument("--num_views", type=int, default=3,
                       help="Number of views")
    parser.add_argument("--is_full_finetune", action="store_true",
                       help="Whether checkpoint is full finetune")

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
    parser.add_argument("--use_different_projection", action='store_true',
                       help='Use different projection mode')

    args = parser.parse_args()

    # 解析参数
    test_indices = [int(x.strip()) for x in args.test_indices.split(',')]
    scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
    transform_xyz = [float(x.strip()) for x in args.transform_augmentation_xyz.split(',')]
    transform_rpy = [float(x.strip()) for x in args.transform_augmentation_rpy.split(',')]

    print("=== MV Concat Inference ===")
    print(f"LoRA Checkpoint: {args.lora_checkpoint}")
    print(f"Model Type: {args.wan_type}")
    print(f"Dual Head Mode: {args.use_dual_head}")
    print(f"Is Full Finetune: {args.is_full_finetune}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {args.output_dir}")
    print()

    # 创建推理器
    inference = HeatmapInferenceMVConcat(
        model_base_path=args.model_base_path,
        lora_checkpoint=args.lora_checkpoint,
        wan_type=args.wan_type,
        use_dual_head=args.use_dual_head,
        num_views=args.num_views,
        device=args.device,
        is_full_finetune=args.is_full_finetune,
    )

    if args.data_root is None:
        # 无数据集时使用dummy测试
        print("No data_root provided, creating dummy test inputs...")
        for sample_id in test_indices:
            input_images = [
                Image.new("RGB", (args.width, args.height), color="gray")
                for _ in range(args.num_views)
            ]
            input_images_rgb = [
                Image.new("RGB", (args.width, args.height), color="gray")
                for _ in range(args.num_views)
            ]

            output = inference.run_inference(
                input_image=input_images,
                input_image_rgb=input_images_rgb,
                prompt="robot arm manipulation",
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )

            inference.save_results(output, args.output_dir, sample_id)
    else:
        # 在数据集上测试
        test_on_dataset_mv_concat(
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
            use_different_projection=args.use_different_projection,
        )

    print("\nInference completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
