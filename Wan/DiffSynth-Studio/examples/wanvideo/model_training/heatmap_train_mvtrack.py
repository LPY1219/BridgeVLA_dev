"""
MVTrack Pretraining Script for Wan2.2 Multi-View Video Diffusion Model

This script is based on heatmap_train_mv.py but adapted for pretraining on the MVTrack dataset.
The MVTrack dataset is a multi-view tracking dataset without robot poses/gripper states.

Key differences from heatmap_train_mv.py:
1. Uses MVTrackDatasetFactory instead of HeatmapDatasetFactory
2. No rotation/gripper prediction (wan_type = "5B_TI2V_RGB_HEATMAP_MV")
3. Different data parameters (split files instead of trail ranges)
"""

import torch
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn

# Force flush output to ensure error messages are displayed
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# SwanLab import (optional)
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available. Install with: pip install swanlab")

from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import ConcatDataset

# Import MVTrack dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from diffsynth.trainers.mvtrack_dataset import MVTrackDatasetFactory

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MVTrackWanTrainingModule(DiffusionTrainingModule):
    """
    MVTrack pretraining module for Wan multi-view video diffusion model.
    Based on HeatmapWanTrainingModule but simplified for pretraining (no rotation/gripper).
    """

    def __init__(
        self,
        wan_type,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_dual_head=False,
        unfreeze_modulation_and_norms=False,
    ):
        super().__init__()
        self.wan_type = wan_type

        # Import appropriate pipeline based on wan_type
        if self.wan_type == "5B_TI2V_RGB_HEATMAP_MV":
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import WanVideoPipeline, ModelConfig
            from diffsynth.models.wan_video_dit_mv import SelfAttention
        else:
            raise ValueError(f"Unsupported wan_type for MVTrack pretraining: {self.wan_type}")

        # Load models
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if local_rank == 0:
            print("\n" + "="*80)
            print(f"[DEBUG] Loading pipeline with use_dual_head={use_dual_head}")
            print("="*80)

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            wan_type=self.wan_type,
            use_dual_head=use_dual_head
        )
        self.use_dual_head = use_dual_head
        self.unfreeze_modulation_and_norms = unfreeze_modulation_and_norms

        if local_rank == 0:
            print("\n" + "="*80)
            print(f"[DEBUG] Pipeline loaded successfully with use_dual_head={use_dual_head}")
            print("="*80 + "\n")

        # Add multi-view attention modules
        if self.wan_type == "5B_TI2V_RGB_HEATMAP_MV":
            dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            for block in self.pipe.dit.blocks:
                block.projector = nn.Linear(dim, dim)
                block.projector.weight = nn.Parameter(torch.zeros(dim, dim))
                block.projector.bias = nn.Parameter(torch.zeros(dim))
                block.norm_mvs = nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False)
                block.modulation_mvs = nn.Parameter(torch.randn(1, 3, dim) / dim**0.5)
                block.mvs_attn = SelfAttention(dim, block.self_attn.num_heads, block.self_attn.norm_q.eps)
                block.modulation_mvs.data = block.modulation.data[:, :3, :].clone()
                block.mvs_attn.load_state_dict(block.self_attn.state_dict(), strict=True)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Unfreeze patch_embedding and head parameters
        self._unfreeze_patch_embedding_and_head()

        # Unfreeze modulation parameters if requested
        if self.unfreeze_modulation_and_norms:
            self._unfreeze_modulation()
        else:
            if local_rank == 0:
                print("\n" + "="*80)
                print("ℹ️  MODULATION PARAMETERS: KEEPING FROZEN (Backward Compatible)")
                print("="*80)
                print("  Modulation parameters will stay frozen (using pretrained values).")
                print("  To unfreeze, add --unfreeze_modulation_and_norms flag.")
                print("="*80 + "\n")

        # Unfreeze multi-view modules
        if self.wan_type == "5B_TI2V_RGB_HEATMAP_MV":
            self._unfreeze_mv_modules()

        # Print trainable parameters info
        self._print_trainable_parameters_info()

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Debug settings
        self.debug_counter = 0
        self.debug_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../debug_log"))

    def _unfreeze_patch_embedding_and_head(self):
        """Unfreeze patch_embedding and head parameters for full training."""
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []
        all_head_params = []

        for name, param in self.pipe.dit.named_parameters():
            if 'head' in name:
                all_head_params.append((name, param.requires_grad))

            is_patch_embedding = 'patch_embedding' in name
            is_head = ('head' in name) if not self.use_dual_head else ('head_rgb' in name or 'head_heatmap' in name)
            is_not_lora = ('lora_A' not in name and 'lora_B' not in name)

            if (is_patch_embedding or is_head) and is_not_lora:
                param.requires_grad = True
                unfrozen_params.append(name)

        if local_rank == 0:
            print("\n" + "="*80)
            mode_str = "DUAL HEAD" if self.use_dual_head else "SINGLE HEAD"
            print(f"FULL PARAMETER UNFREEZING FOR PATCH_EMBEDDING AND HEAD ({mode_str} MODE)")
            print("="*80)
            if len(unfrozen_params) > 0:
                print(f"\nUnfroze {len(unfrozen_params)} parameter(s) for full training:")
                for name in unfrozen_params:
                    param = dict(self.pipe.dit.named_parameters())[name]
                    print(f"  ✓ {name}")
                    print(f"    Shape: {param.shape}, Trainable: {param.requires_grad}")
            else:
                print("\n⚠️  WARNING: No parameters were unfrozen!")
            print("="*80 + "\n")

    def _unfreeze_modulation(self):
        """Unfreeze modulation parameters (AdaLN) for better task adaptation."""
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []

        for name, param in self.pipe.dit.named_parameters():
            is_modulation = 'modulation' in name and 'blocks.' in name and 'modulation_mvs' not in name
            is_not_lora = ('lora_A' not in name and 'lora_B' not in name)

            if is_modulation and is_not_lora:
                param.requires_grad = True
                unfrozen_params.append(name)

        if local_rank == 0:
            print("\n" + "="*80)
            print("⚡ MODULATION PARAMETER UNFREEZING")
            print("="*80)
            if len(unfrozen_params) > 0:
                total_params = sum(dict(self.pipe.dit.named_parameters())[name].numel() for name in unfrozen_params)
                print(f"\n✅ Unfroze {len(unfrozen_params)} modulation parameters:")
                print(f"  📦 AdaLN modulation: {len(unfrozen_params)} parameters")
                print(f"  📊 Total: {total_params:,} parameters (~{total_params/1e6:.2f}M)")
                print(f"\n💡 训练策略:")
                print(f"  - Modulation: 全量训练（适应新任务的时间步调制模式）")
            else:
                print("\n⚠️  WARNING: No modulation parameters were unfrozen!")
            print("="*80 + "\n")

    def _unfreeze_mv_modules(self):
        """Unfreeze multi-view module parameters."""
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []
        unfrozen_norms = []

        for name, param in self.pipe.dit.named_parameters():
            is_projector = 'projector' in name and 'blocks.' in name
            is_modulation_mvs = 'modulation_mvs' in name
            is_mvs_attn_norm = 'mvs_attn' in name and ('norm_q' in name or 'norm_k' in name)
            is_not_lora = ('lora_A' not in name and 'lora_B' not in name)

            if (is_projector or is_modulation_mvs) and is_not_lora:
                param.requires_grad = True
                unfrozen_params.append(name)
            elif is_mvs_attn_norm and is_not_lora and self.unfreeze_modulation_and_norms:
                param.requires_grad = True
                unfrozen_norms.append(name)

        if local_rank == 0:
            print("\n" + "="*80)
            print("MULTI-VIEW MODULE PARAMETER UNFREEZING")
            print("="*80)
            if len(unfrozen_params) > 0:
                total_params = sum(dict(self.pipe.dit.named_parameters())[name].numel() for name in unfrozen_params)
                print(f"\n✅ Unfroze {len(unfrozen_params)} parameter(s)")
                print(f"  📦 Projector & Modulation_mvs (全量训练): {len(unfrozen_params)} parameters")
                print(f"  📊 Total: {total_params:,} parameters (~{total_params/1e6:.2f}M)")

            if len(unfrozen_norms) > 0:
                total_norm_params = sum(dict(self.pipe.dit.named_parameters())[name].numel() for name in unfrozen_norms)
                print(f"\n✅ Unfroze {len(unfrozen_norms)} MVS_Attn norm parameters:")
                print(f"  📦 MVS_Attn Norms (全量训练): {len(unfrozen_norms)} parameters")
                print(f"  📊 Total: {total_norm_params:,} parameters (~{total_norm_params/1e6:.2f}M)")

            print(f"\n💡 训练策略:")
            print(f"  - Projector & Modulation_mvs: 全量训练（从零开始学习）")
            print(f"  - MVS_Attn: LoRA微调（利用预训练知识，节省显存）")
            if len(unfrozen_norms) > 0:
                print(f"  - MVS_Attn norm_q/norm_k: 全量训练（适应多视角特征分布）")
            print("="*80 + "\n")

    def _print_trainable_parameters_info(self):
        """Print trainable parameters information."""
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return

        print("\n" + "="*80)
        print("TRAINABLE PARAMETERS MONITORING")
        print("="*80)

        total_params = 0
        trainable_params = 0

        for name, param in self.pipe.dit.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"\n📊 Parameter Overview:")
        print(f"  Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M, {trainable_params/total_params*100:.2f}%)")
        print("="*80 + "\n")

    def forward_preprocess(self, data):
        """Preprocess input data for forward pass."""
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0][0].size[1],
            "width": data["video"][0][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "use_dual_head": self.use_dual_head,
            "num_view": len(data["video"][0])
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["input_image"]
            elif extra_input == "input_image_rgb":
                inputs_shared["input_image_rgb"] = data["input_image_rgb"]
            elif extra_input == "input_video_rgb":
                inputs_shared["input_video_rgb"] = data["input_video_rgb"]
            elif extra_input in data:
                inputs_shared[extra_input] = data[extra_input]

        # Process through pipeline units
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        return {**inputs_shared, **inputs_posi}

    def visualize_forward_inputs(self, inputs):
        """
        Visualize forward inputs before model processing (Multi-View support).

        This function visualizes:
        - input_image: First frame heatmaps for each view
        - input_image_rgb: First frame RGB images for each view
        - input_video: Heatmap video sequence (T+1) x num_views
        - input_video_rgb: RGB video sequence (T+1) x num_views

        Only runs on main process.
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return

        try:
            # Ensure debug directory exists
            os.makedirs(self.debug_save_dir, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"FORWARD INPUT VISUALIZATION - MVTRACK MULTI-VIEW (Step: {self.debug_counter})")
            print(f"{'='*80}")
            print(f"Available keys: {list(inputs.keys())}")

            # Collect data to visualize
            input_image = inputs.get('input_image')
            input_image_rgb = inputs.get('input_image_rgb')
            input_video = inputs.get('input_video')
            input_video_rgb = inputs.get('input_video_rgb')

            # Detect data structure
            num_views = 0
            num_frames = 0

            # Detect number of views
            if input_image and isinstance(input_image, list):
                num_views = len(input_image)
                print(f"✓ input_image: {num_views} views")
            elif input_image_rgb and isinstance(input_image_rgb, list):
                num_views = len(input_image_rgb)
                print(f"✓ input_image_rgb: {num_views} views")

            # Detect number of frames
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                num_frames = len(input_video)
                if isinstance(input_video[0], list):
                    num_views = max(num_views, len(input_video[0]))
                    print(f"✓ input_video: {num_frames} frames × {len(input_video[0])} views")
                else:
                    print(f"✓ input_video: {num_frames} frames (single view)")

            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                if isinstance(input_video_rgb[0], list):
                    print(f"✓ input_video_rgb: {len(input_video_rgb)} frames × {len(input_video_rgb[0])} views")
                else:
                    print(f"✓ input_video_rgb: {len(input_video_rgb)} frames (single view)")

            # Calculate layout
            cols = max(num_views, 3)

            # Dynamically calculate rows
            rows_list = []

            # Title row for images
            rows_list.append(('title_img', 0.3))

            # Input images rows
            if input_image or input_image_rgb:
                rows_list.append(('input_image', 1.5))
                rows_list.append(('input_image_rgb', 1.5))

            # Video title
            if input_video or input_video_rgb:
                rows_list.append(('title_video', 0.3))

            # Input video rows (show up to 13 frames)
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                for _ in range(min(num_frames, 13)):
                    rows_list.append(('video_frame', 1.2))

            # RGB video title
            if input_video_rgb:
                rows_list.append(('title_video_rgb', 0.3))

            # Input video RGB rows
            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                for _ in range(min(len(input_video_rgb), 13)):
                    rows_list.append(('video_rgb_frame', 1.2))

            # Create figure
            total_rows = len(rows_list)
            height_ratios = [r[1] for r in rows_list]
            fig_height = sum(height_ratios) * 2.5

            fig = plt.figure(figsize=(2.5*cols, fig_height))
            gs = fig.add_gridspec(total_rows, cols, hspace=0.3, wspace=0.1, height_ratios=height_ratios)

            current_row = 0

            # ============ Title: Input Images ============
            if rows_list[current_row][0] == 'title_img':
                ax_title = fig.add_subplot(gs[current_row, :])
                ax_title.text(0.5, 0.5, 'MVTrack Input Images (Multi-View)', ha='center', va='center',
                             fontsize=14, fontweight='bold', transform=ax_title.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
                ax_title.axis('off')
                current_row += 1

            # ============ input_image (multi-view heatmaps) ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'input_image':
                if input_image and isinstance(input_image, list):
                    for i, img in enumerate(input_image):
                        if i < cols:
                            ax = fig.add_subplot(gs[current_row, i])
                            if hasattr(img, 'save'):
                                img_array = np.array(img)
                                ax.imshow(img_array)
                                ax.set_title(f'Heatmap View {i}', fontsize=10, fontweight='bold')
                                if i == 0:
                                    print(f"  Heatmap View {i}: Size {img.size}, Mode {img.mode}")
                            else:
                                ax.text(0.5, 0.5, f'View {i}\nInvalid', ha='center', va='center',
                                       fontsize=9, transform=ax.transAxes)
                            ax.axis('off')
                    for i in range(len(input_image), cols):
                        ax = fig.add_subplot(gs[current_row, i])
                        ax.axis('off')
                else:
                    ax = fig.add_subplot(gs[current_row, :])
                    ax.text(0.5, 0.5, 'No input_image', ha='center', va='center',
                           fontsize=10, transform=ax.transAxes)
                    ax.axis('off')
                current_row += 1

            # ============ input_image_rgb (multi-view RGB) ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'input_image_rgb':
                if input_image_rgb and isinstance(input_image_rgb, list):
                    for i, img in enumerate(input_image_rgb):
                        if i < cols:
                            ax = fig.add_subplot(gs[current_row, i])
                            if hasattr(img, 'save'):
                                img_array = np.array(img)
                                ax.imshow(img_array)
                                ax.set_title(f'RGB View {i}', fontsize=10, fontweight='bold')
                                if i == 0:
                                    print(f"  RGB View {i}: Size {img.size}, Mode {img.mode}")
                            else:
                                ax.text(0.5, 0.5, f'View {i}\nInvalid', ha='center', va='center',
                                       fontsize=9, transform=ax.transAxes)
                            ax.axis('off')
                    for i in range(len(input_image_rgb), cols):
                        ax = fig.add_subplot(gs[current_row, i])
                        ax.axis('off')
                else:
                    ax = fig.add_subplot(gs[current_row, :])
                    ax.text(0.5, 0.5, 'No input_image_rgb', ha='center', va='center',
                           fontsize=10, transform=ax.transAxes)
                    ax.axis('off')
                current_row += 1

            # ============ Video title ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'title_video':
                ax_title = fig.add_subplot(gs[current_row, :])
                ax_title.text(0.5, 0.5, 'Input Video (Heatmap Sequence, Multi-View)', ha='center', va='center',
                             fontsize=14, fontweight='bold', transform=ax_title.transAxes,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax_title.axis('off')
                current_row += 1

            # ============ input_video (multi-view × multi-frame heatmaps) ============
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                frames_to_show = min(len(input_video), 13)
                for frame_idx in range(frames_to_show):
                    if current_row < len(rows_list) and rows_list[current_row][0] == 'video_frame':
                        frame_data = input_video[frame_idx]

                        if isinstance(frame_data, list):
                            for view_idx, img in enumerate(frame_data):
                                if view_idx < cols:
                                    ax = fig.add_subplot(gs[current_row, view_idx])
                                    if hasattr(img, 'save'):
                                        img_array = np.array(img)
                                        ax.imshow(img_array)
                                        ax.set_title(f'T{frame_idx}V{view_idx}', fontsize=8)
                                    ax.axis('off')
                            for view_idx in range(len(frame_data), cols):
                                ax = fig.add_subplot(gs[current_row, view_idx])
                                ax.axis('off')
                        else:
                            ax = fig.add_subplot(gs[current_row, 0])
                            if hasattr(frame_data, 'save'):
                                img_array = np.array(frame_data)
                                ax.imshow(img_array)
                                ax.set_title(f'Frame {frame_idx}', fontsize=9)
                            ax.axis('off')
                            for i in range(1, cols):
                                ax = fig.add_subplot(gs[current_row, i])
                                ax.axis('off')

                        current_row += 1

            # ============ RGB video title ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'title_video_rgb':
                ax_title = fig.add_subplot(gs[current_row, :])
                ax_title.text(0.5, 0.5, 'Input Video RGB (RGB Sequence, Multi-View)', ha='center', va='center',
                             fontsize=14, fontweight='bold', transform=ax_title.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                ax_title.axis('off')
                current_row += 1

            # ============ input_video_rgb (multi-view × multi-frame RGB) ============
            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                frames_to_show = min(len(input_video_rgb), 13)
                for frame_idx in range(frames_to_show):
                    if current_row < len(rows_list) and rows_list[current_row][0] == 'video_rgb_frame':
                        frame_data = input_video_rgb[frame_idx]

                        if isinstance(frame_data, list):
                            for view_idx, img in enumerate(frame_data):
                                if view_idx < cols:
                                    ax = fig.add_subplot(gs[current_row, view_idx])
                                    if hasattr(img, 'save'):
                                        img_array = np.array(img)
                                        ax.imshow(img_array)
                                        ax.set_title(f'T{frame_idx}V{view_idx}', fontsize=8)
                                    ax.axis('off')
                            for view_idx in range(len(frame_data), cols):
                                ax = fig.add_subplot(gs[current_row, view_idx])
                                ax.axis('off')
                        else:
                            ax = fig.add_subplot(gs[current_row, 0])
                            if hasattr(frame_data, 'save'):
                                img_array = np.array(frame_data)
                                ax.imshow(img_array)
                                ax.set_title(f'Frame {frame_idx}', fontsize=9)
                            ax.axis('off')
                            for i in range(1, cols):
                                ax = fig.add_subplot(gs[current_row, i])
                                ax.axis('off')

                        current_row += 1

            # Save combined figure
            save_path = os.path.join(self.debug_save_dir, f"mvtrack_step_{self.debug_counter:04d}_all_inputs.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print("\n✅ MVTrack multi-view inputs visualization saved to:")
            print(f"   {save_path}")

            # Print other key info
            print("\nOther inputs:")
            for key in ['prompt', 'height', 'width', 'num_frames', 'num_view']:
                if key in inputs:
                    print(f"  {key}: {inputs[key]}")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"❌ Error in forward input visualization: {e}")
            import traceback
            traceback.print_exc()

    def forward(self, data, inputs=None):
        """Forward pass to compute loss."""
        if inputs is None:
            inputs = self.forward_preprocess(data)

        # Device consistency
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        expected_device = f"cuda:{local_rank}"

        if "rand_device" in inputs:
            inputs["rand_device"] = expected_device

        def move_to_device(obj, device):
            if hasattr(obj, 'to') and hasattr(obj, 'device'):
                return obj.to(device)
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(item, device) for item in obj)
            elif isinstance(obj, dict):
                return {key: move_to_device(value, device) for key, value in obj.items()}
            else:
                return obj

        try:
            inputs = move_to_device(inputs, expected_device)
            data = move_to_device(data, expected_device)
        except Exception as e:
            print(f"Warning: Could not move inputs to device {expected_device}: {e}")

        # HARDCODED DEBUG - Set to True to enable visualization
        DEBUG_VISUALIZATION = False  # Manually change to True to enable debug visualization

        if DEBUG_VISUALIZATION:
            print(f"\n🔍 DEBUG MODE ACTIVATED (Step {self.debug_counter})")
            # Visualize forward inputs
            self.visualize_forward_inputs(inputs)
            import pdb;pdb.set_trace()
            self.debug_counter += 1

        # Forward pass
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


def launch_mvtrack_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    args=None,
):
    """
    Launch MVTrack pretraining task.
    """
    if args is None:
        raise ValueError("args is required for training")

    # Extract parameters
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    save_epochs_interval = getattr(args, 'save_epochs_interval', 0)
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters
    train_batch_size = args.train_batch_size

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process_for_print = local_rank == 0

    if is_main_process_for_print:
        print(f"Training configuration:")
        print(f"  - Batch size per GPU: 1 (Wan model limitation)")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Data workers: {num_workers}")
        print(f"  - Save epochs interval: {save_epochs_interval}")

    def collate_single_sample(batch):
        return batch[0]

    dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 1 if num_workers > 0 else None,
        'drop_last': True,
        'collate_fn': collate_single_sample,
    }

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    try:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        if accelerator.is_main_process:
            print("✅ Model preparation completed successfully")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ Error during model preparation: {e}")
        raise e

    # Training loop
    try:
        if accelerator.is_main_process:
            print("🚀 Starting training loop...")
            print(f"📊 Training for {num_epochs} epochs")
            print(f"📊 Dataset size: {len(dataloader)} batches")

        for epoch_id in range(num_epochs):
            if accelerator.is_main_process:
                print(f"\n🔄 Starting epoch {epoch_id+1}/{num_epochs}")

            model.train()
            epoch_loss = 0
            step_count = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")

            for step, data in enumerate(pbar):
                with accelerator.accumulate(model):
                    optimizer.zero_grad(set_to_none=True)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    try:
                        loss_dict = model(data)
                    except RuntimeError as e:
                        if accelerator.is_main_process:
                            print(f"Error at step {step}: {e}")
                        raise e

                    if isinstance(loss_dict, dict):
                        loss = loss_dict["loss"]
                        loss_rgb = loss_dict.get("loss_rgb", None)
                        loss_heatmap = loss_dict.get("loss_heatmap", None)
                    else:
                        loss = loss_dict
                        loss_rgb = None
                        loss_heatmap = None

                    accelerator.backward(loss)

                    if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    epoch_loss += loss.item()
                    step_count += 1

                    # SwanLab logging
                    global_step = step + epoch_id * len(dataloader)
                    should_log = (accelerator.is_main_process and
                                 hasattr(args, 'swanlab_run') and args.swanlab_run is not None and
                                 hasattr(args, 'logging_steps') and global_step % args.logging_steps == 0)

                    if should_log:
                        try:
                            import swanlab
                            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

                            log_data = {
                                "train_loss": loss.item(),
                                "learning_rate": current_lr,
                                "epoch": epoch_id,
                                "step": global_step
                            }

                            if loss_rgb is not None:
                                log_data["train_loss_rgb"] = loss_rgb.item()
                            if loss_heatmap is not None:
                                log_data["train_loss_heatmap"] = loss_heatmap.item()

                            swanlab.log(log_data, step=global_step)
                            print(f"SwanLab logged: step={global_step}, loss={loss.item():.4f}")
                        except Exception as e:
                            print(f"Warning: Failed to log to SwanLab: {e}")

                    # Update progress bar
                    postfix_dict = {
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss/step_count:.4f}"
                    }
                    if loss_rgb is not None:
                        postfix_dict['loss_rgb'] = f"{loss_rgb.item():.4f}"
                    if loss_heatmap is not None:
                        postfix_dict['loss_hm'] = f"{loss_heatmap.item():.4f}"
                    pbar.set_postfix(postfix_dict)

                    if save_steps > 0:
                        model_logger.on_step_end(accelerator, model, save_steps)

            # Epoch end
            should_save_epoch = (save_epochs_interval == 0) or ((epoch_id + 1) % save_epochs_interval == 0) or (epoch_id == num_epochs - 1)
            if should_save_epoch:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
                if accelerator.is_main_process:
                    print(f"✅ Saved checkpoint at epoch {epoch_id + 1}")

            accelerator.print(f"Epoch {epoch_id+1} completed. Average loss: {epoch_loss/step_count:.4f}")

        if save_steps > 0:
            model_logger.on_training_end(accelerator, model, save_steps)

        if accelerator.is_main_process:
            print("🎉 Training completed successfully!")

    except Exception as training_error:
        if accelerator.is_main_process:
            print(f"❌ Training failed with error: {training_error}")
            import traceback
            traceback.print_exc()
        raise training_error


def create_mvtrack_parser():
    """Create argument parser for MVTrack pretraining."""
    import argparse
    parser = argparse.ArgumentParser(description="MVTrack pretraining for Wan2.2 Multi-View Video Diffusion")

    # Basic training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="", help="Remove prefix in checkpoint.")
    parser.add_argument("--trainable_models", type=str, default="", help="Trainable models.")
    parser.add_argument("--lora_base_model", type=str, default="", help="LoRA base model.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="LoRA target modules.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_checkpoint", type=str, default="", help="LoRA checkpoint.")
    parser.add_argument("--extra_inputs", type=str, default="", help="Extra inputs.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--dataloader_pin_memory", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_epochs_interval", type=int, default=0)
    parser.add_argument("--dataset_num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--model_paths", type=str, default="")
    parser.add_argument("--model_id_with_origin_paths", type=str, default="")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV")
    parser.add_argument("--use_dual_head", action="store_true")
    parser.add_argument("--unfreeze_modulation_and_norms", action="store_true",
                       help="Unfreeze modulation and norm parameters for better task adaptation")

    # MVTrack specific arguments
    parser.add_argument("--mvtrack_data_root", type=str, required=True,
                       help="Root directory of MVTrack dataset")
    parser.add_argument("--sequence_length", type=int, default=12,
                       help="Number of future frames to predict (not including first frame)")
    parser.add_argument("--step_interval", type=int, default=1,
                       help="Frame sampling interval")
    parser.add_argument("--min_sequence_length", type=int, default=15,
                       help="Minimum sequence length requirement")
    parser.add_argument("--heatmap_sigma", type=float, default=5.0,
                       help="Standard deviation for Gaussian heatmap generation")
    parser.add_argument("--num_views", type=int, default=3,
                       help="Number of views to sample for each training example")
    parser.add_argument("--colormap_name", type=str, default="jet",
                       help="Colormap name for heatmap visualization")
    parser.add_argument("--disable_augmentation", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode (use fewer data)")

    # SwanLab arguments
    parser.add_argument("--enable_swanlab", action="store_true")
    parser.add_argument("--swanlab_api_key", type=str, default="h1x6LOLp5qGLTfsPuB7Qw")
    parser.add_argument("--swanlab_project", type=str, default="wan2.2-mvtrack-pretrain")
    parser.add_argument("--swanlab_experiment", type=str, default="mvtrack-pretrain")

    return parser


if __name__ == "__main__":
    parser = create_mvtrack_parser()
    args = parser.parse_args()

    print("="*60)
    print("MVTRACK PRETRAINING FOR WAN2.2 MULTI-VIEW VIDEO DIFFUSION")
    print("="*60)
    print(f"Data root: {args.mvtrack_data_root}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Num views: {args.num_views}")
    print(f"Heatmap sigma: {args.heatmap_sigma}")
    print(f"Output path: {args.output_path}")
    print(f"Debug mode: {args.debug_mode}")
    print("="*60)

    # Initialize SwanLab
    swanlab_run = None
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    if is_main_process and args.enable_swanlab and not args.debug_mode and SWANLAB_AVAILABLE:
        try:
            print("Initializing SwanLab...")
            swanlab.login(api_key=args.swanlab_api_key)
            swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment,
                config={
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.num_epochs,
                    "sequence_length": args.sequence_length,
                    "num_views": args.num_views,
                    "heatmap_sigma": args.heatmap_sigma,
                    "lora_rank": args.lora_rank,
                    "height": args.height,
                    "width": args.width,
                }
            )
            print(f"✅ SwanLab initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize SwanLab: {e}")
            swanlab_run = None

    # Create dataset
    if is_main_process:
        print("Creating MVTrack dataset...")

    try:
        dataset = MVTrackDatasetFactory.create_mvtrack_dataset(
            data_root=args.mvtrack_data_root,
            sequence_length=args.sequence_length,
            step_interval=args.step_interval,
            min_sequence_length=args.min_sequence_length,
            image_size=(args.height, args.width),
            sigma=args.heatmap_sigma,
            num_views=args.num_views,
            augmentation=not args.disable_augmentation,
            debug=args.debug_mode,
            colormap_name=args.colormap_name,
            repeat=args.dataset_repeat,
            wan_type=args.wan_type,
        )

        if is_main_process:
            print(f"✓ Dataset created: {len(dataset)} samples")

            # Test data loading
            print("Testing data loading...")
            test_sample = dataset[0]
            print(f"Sample keys: {list(test_sample.keys())}")
            print(f"Video frames: {len(test_sample['video'])}")
            print(f"Views per frame: {len(test_sample['video'][0])}")
            print(f"Prompt: {test_sample['prompt']}")

    except Exception as e:
        if is_main_process:
            print(f"Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
        exit(1)

    # Create training module
    if is_main_process:
        print("Creating training module...")

    try:
        model_id_with_origin_paths = args.model_id_with_origin_paths if args.model_id_with_origin_paths else None
        lora_checkpoint = args.lora_checkpoint if args.lora_checkpoint else None
        trainable_models = args.trainable_models if args.trainable_models else None
        lora_base_model = args.lora_base_model if args.lora_base_model else None

        model = MVTrackWanTrainingModule(
            wan_type=args.wan_type,
            model_paths=args.model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=lora_checkpoint,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
            use_dual_head=args.use_dual_head,
            unfreeze_modulation_and_norms=args.unfreeze_modulation_and_norms,
        )

        if is_main_process:
            print("Training module created successfully")

    except Exception as e:
        if is_main_process:
            print(f"Error creating training module: {e}")
            import traceback
            traceback.print_exc()
        exit(1)

    # Create model logger
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    # Launch training
    if is_main_process:
        print("Starting training...")

    try:
        if swanlab_run is not None:
            args.swanlab_run = swanlab_run
        else:
            args.swanlab_run = None

        launch_mvtrack_training_task(dataset, model, model_logger, args=args)

        if is_main_process:
            print("Training completed successfully!")

        if swanlab_run is not None:
            swanlab_run.finish()

    except Exception as e:
        if is_main_process:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

        if swanlab_run is not None:
            swanlab_run.finish()
        exit(1)
