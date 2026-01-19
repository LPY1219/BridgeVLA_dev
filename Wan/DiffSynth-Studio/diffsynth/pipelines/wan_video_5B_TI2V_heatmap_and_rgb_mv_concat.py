"""
Multi-View Video Pipeline with Token Concatenation

This pipeline uses WanModel_mv_concat which concatenates multi-view tokens
along the sequence dimension instead of using separate multi-view attention modules.

Key differences from wan_video_5B_TI2V_heatmap_and_rgb_mv.py:
- Uses WanModel_mv_concat instead of WanModel_mv
- No freqs_mvs or shape_info parameters needed
- Token concatenation/splitting happens inside the model
"""

import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import load_state_dict
from ..models.model_manager_adaptive import AdaptiveModelManager
from ..models.wan_video_dit_mv_concat import WanModel_mv_concat, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader


# Heatmap Latent Scaling Configuration
HEATMAP_LATENT_SCALE_FACTOR = 1.0


def set_heatmap_scale_factor(scale: float):
    """Set heatmap latent scale factor."""
    global HEATMAP_LATENT_SCALE_FACTOR
    HEATMAP_LATENT_SCALE_FACTOR = scale
    print(f"Heatmap latent scale factor set to: {scale}")


def convert_wan_model_to_mv_concat(wan_model, use_dual_head: bool = False, num_views: int = 3):
    """
    Convert WanModel to WanModel_mv_concat.

    Args:
        wan_model: Original WanModel instance
        use_dual_head: Whether to use dual head mode
        num_views: Number of views

    Returns:
        WanModel_mv_concat instance with transferred weights
    """
    # Get original model config
    config = {
        'dim': wan_model.dim,
        'in_dim': wan_model.in_dim,
        'freq_dim': wan_model.freq_dim,
        'has_image_input': wan_model.has_image_input,
        'patch_size': wan_model.patch_size,
        'use_dual_head': use_dual_head,
        'num_views': num_views,
    }

    # Get config from blocks
    if len(wan_model.blocks) > 0:
        first_block = wan_model.blocks[0]
        config['num_heads'] = first_block.num_heads
        config['ffn_dim'] = first_block.ffn_dim
        config['num_layers'] = len(wan_model.blocks)
        config['eps'] = 1e-6

    # Check optional attributes
    if hasattr(wan_model, 'has_image_pos_emb'):
        config['has_image_pos_emb'] = wan_model.has_image_pos_emb
    if hasattr(wan_model, 'has_ref_conv'):
        config['has_ref_conv'] = wan_model.has_ref_conv
    if hasattr(wan_model, 'control_adapter'):
        config['add_control_adapter'] = wan_model.control_adapter is not None
    if hasattr(wan_model, 'seperated_timestep'):
        config['seperated_timestep'] = wan_model.seperated_timestep
    if hasattr(wan_model, 'require_vae_embedding'):
        config['require_vae_embedding'] = wan_model.require_vae_embedding
    if hasattr(wan_model, 'require_clip_embedding'):
        config['require_clip_embedding'] = wan_model.require_clip_embedding
    if hasattr(wan_model, 'fuse_vae_embedding_in_latents'):
        config['fuse_vae_embedding_in_latents'] = wan_model.fuse_vae_embedding_in_latents

    # Get out_dim from head
    if hasattr(wan_model, 'head') and wan_model.head is not None:
        out_features = wan_model.head.head.out_features
        patch_prod = wan_model.patch_size[0] * wan_model.patch_size[1] * wan_model.patch_size[2]
        config['out_dim'] = out_features // patch_prod
    else:
        config['out_dim'] = 48

    # Get text_dim from text_embedding
    if hasattr(wan_model, 'text_embedding'):
        config['text_dim'] = wan_model.text_embedding[0].in_features
    else:
        config['text_dim'] = 4096

    print(f"    Creating WanModel_mv_concat with config: dim={config['dim']}, in_dim={config['in_dim']}, "
          f"out_dim={config['out_dim']}, num_layers={config.get('num_layers', 'N/A')}, "
          f"use_dual_head={use_dual_head}, num_views={num_views}")

    # Create new model instance
    wan_model_mv_concat = WanModel_mv_concat(**config)

    # Copy weights
    original_state_dict = wan_model.state_dict()

    if use_dual_head and 'head.head.weight' in original_state_dict:
        # Handle dual head mode
        new_state_dict = {}
        for key, value in original_state_dict.items():
            if not key.startswith('head.'):
                new_state_dict[key] = value
            else:
                # Copy head weights to both heads
                if key.startswith('head.head.'):
                    new_key_rgb = key.replace('head.head.', 'head_rgb.head.')
                    new_key_heatmap = key.replace('head.head.', 'head_heatmap.head.')
                    new_state_dict[new_key_rgb] = value.clone()
                    new_state_dict[new_key_heatmap] = value.clone()
                elif key.startswith('head.norm.'):
                    new_key_rgb = key.replace('head.norm.', 'head_rgb.norm.')
                    new_key_heatmap = key.replace('head.norm.', 'head_heatmap.norm.')
                    new_state_dict[new_key_rgb] = value.clone()
                    new_state_dict[new_key_heatmap] = value.clone()
                elif key.startswith('head.modulation'):
                    new_key_rgb = key.replace('head.modulation', 'head_rgb.modulation')
                    new_key_heatmap = key.replace('head.modulation', 'head_heatmap.modulation')
                    new_state_dict[new_key_rgb] = value.clone()
                    new_state_dict[new_key_heatmap] = value.clone()

        wan_model_mv_concat.load_state_dict(new_state_dict, strict=False)
        print("    ✓ Converted to WanModel_mv_concat with dual head")
    else:
        missing_keys, unexpected_keys = wan_model_mv_concat.load_state_dict(original_state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f"    ⚠️  Missing keys: {len(missing_keys)} keys")
        if len(unexpected_keys) > 0:
            print(f"    ⚠️  Unexpected keys: {len(unexpected_keys)} keys")
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("    ✓ Successfully converted to WanModel_mv_concat")

    # Move to same device and dtype
    wan_model_mv_concat = wan_model_mv_concat.to(
        device=wan_model.patch_embedding.weight.device,
        dtype=wan_model.patch_embedding.weight.dtype
    )

    return wan_model_mv_concat


class WanVideoPipelineMVConcat(BasePipeline):
    """
    Pipeline for multi-view video generation using token concatenation.
    """

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel_mv_concat = None
        self.dit2: WanModel_mv_concat = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.audio_encoder = None  # Not used in MV Concat, but needed for compatibility with S2V units
        self.audio_processor = None  # Not used in MV Concat, but needed for compatibility
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.model_fn = model_fn_wan_video_mv_concat
        self.use_dual_head = False

        # Import pipeline units from original multi-view pipeline
        from .wan_video_5B_TI2V_heatmap_and_rgb_mv import (
            WanVideoUnit_ShapeChecker,
            WanVideoUnit_NoiseInitializer,
            WanVideoUnit_PromptEmbedder,
            WanVideoUnit_S2V,
            WanVideoUnit_InputVideoEmbedder,
            WanVideoUnit_ImageEmbedderVAE,
            WanVideoUnit_ImageEmbedderCLIP,
            WanVideoUnit_ImageEmbedderFused,
            WanVideoUnit_FunControl,
            WanVideoUnit_FunReference,
            WanVideoUnit_FunCameraControl,
            WanVideoUnit_SpeedControl,
            WanVideoUnit_VACE,
            WanVideoUnit_UnifiedSequenceParallel,
            WanVideoUnit_TeaCache,
            WanVideoUnit_CfgMerger,
            WanVideoPostUnit_S2V,
        )

        # Setup pipeline units for preprocessing
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]

    @staticmethod
    def from_pretrained(
        wan_type="5B_TI2V_RGB_HEATMAP_MV_CONCAT",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list = [],
        tokenizer_config: ModelConfig = None,
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
        use_dual_head: bool = False,
        num_views: int = 3,
    ):
        """
        Load pretrained model and create pipeline.

        Args:
            wan_type: Model type (e.g., "5B_TI2V_RGB_HEATMAP_MV_CONCAT")
            torch_dtype: Torch data type
            device: Device to use
            model_configs: List of ModelConfig for model loading
            tokenizer_config: Tokenizer config
            audio_processor_config: Audio processor config
            redirect_common_files: Whether to redirect common files
            use_usp: Whether to use unified sequence parallel
            use_dual_head: Whether to use dual head mode
            num_views: Number of views

        Returns:
            WanVideoPipelineMVConcat instance
        """
        if tokenizer_config is None:
            # Auto-detect huggingface path
            for base_path in [
                "/DATA/disk1/lpy/huggingface",
                "/data/lpy/huggingface",
                "/DATA/disk1/lpy_a100_4/huggingface",
                "/DATA/disk0/lpy/huggingface",
                "/DATA/disk1/lpy_a100_1/huggingface"
            ]:
                if os.path.exists(base_path):
                    tokenizer_config = ModelConfig(path=f"{base_path}/Wan2.2-TI2V-5B-fused/google/umt5-xxl")
                    break
            if tokenizer_config is None:
                raise RuntimeError("Cannot find huggingface model directory")

        # Initialize pipeline
        pipe = WanVideoPipelineMVConcat(device=device, torch_dtype=torch_dtype)
        pipe.use_dual_head = use_dual_head

        # Download and load models
        model_manager = AdaptiveModelManager(torch_dtype=torch_dtype, device=device, use_dual_head=use_dual_head)
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)

        # Convert WanModel to WanModel_mv_concat
        print(f"Converting DIT model to WanModel_mv_concat for {wan_type}")
        if isinstance(dit, list):
            converted_dit_list = []
            for single_dit in dit:
                if single_dit is not None and type(single_dit).__name__ == 'WanModel':
                    converted_dit = convert_wan_model_to_mv_concat(single_dit, use_dual_head, num_views)
                    converted_dit_list.append(converted_dit)
                else:
                    converted_dit_list.append(single_dit)
            dit = converted_dit_list
        else:
            if dit is not None and type(dit).__name__ == 'WanModel':
                dit = convert_wan_model_to_mv_concat(dit, use_dual_head, num_views)

        # Handle DIT dimension adaptation
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit

        # Adapt dimensions if needed
        first_model_path = model_configs[0].path[0] if isinstance(model_configs[0].path, list) else model_configs[0].path
        config_path = os.path.join(os.path.dirname(first_model_path), "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                config = json.load(f)

            config_in_dim = config.get("in_dim")
            if pipe.dit and hasattr(pipe.dit, 'in_dim') and config_in_dim and pipe.dit.in_dim != config_in_dim:
                print(f"DIT dimension mismatch: model has {pipe.dit.in_dim}, config has {config_in_dim}")
                # Reload and adapt weights
                pretrained_state_dict = {}
                dit_model_config = model_configs[0]
                if isinstance(dit_model_config.path, list):
                    for path in dit_model_config.path:
                        pretrained_state_dict.update(load_state_dict(path))
                else:
                    pretrained_state_dict.update(load_state_dict(dit_model_config.path))

                state_dict_converter = pipe.dit.state_dict_converter()
                converted_state_dict, _ = state_dict_converter.from_civitai(pretrained_state_dict)

                current_dtype = next(pipe.dit.parameters()).dtype
                pipe.dit.in_dim = config_in_dim
                pipe.dit.patch_embedding = torch.nn.Conv3d(
                    pipe.dit.in_dim, pipe.dit.dim,
                    kernel_size=pipe.dit.patch_size,
                    stride=pipe.dit.patch_size
                ).to(dtype=current_dtype)

                if hasattr(pipe.dit, 'adapt_pretrained_weights'):
                    print("Adapting pretrained weights...")
                    pipe.dit.adapt_pretrained_weights(converted_state_dict, strict=False)
                    pipe.dit = pipe.dit.to(dtype=torch_dtype, device=device)
                    print(f"DIT adapted to in_dim={pipe.dit.in_dim}")

        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)

        if use_usp:
            pipe.enable_usp()

        return pipe

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # RGB condition
        input_image_rgb: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list] = None,
        # ControlNet
        control_video: Optional[list] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[str] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_frames: int = 25,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 1.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple] = (30, 52),
        tile_stride: Optional[tuple] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # dual head
        use_dual_head: bool = False,
    ):
        """
        Run inference with multi-view token concatenation model.

        Returns:
            dict with 'video_heatmap' and 'video_rgb' keys, each containing
            a list of videos for each view.
        """
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Infer num_view from input_image
        if input_image is not None and isinstance(input_image, list):
            num_view = len(input_image)
        else:
            num_view = 3

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "input_image_rgb": input_image_rgb,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "use_dual_head": use_dual_head,
            "num_view": num_view,
        }

        # Ensure models are on the correct device (only move once)
        if self.text_encoder is not None and next(self.text_encoder.parameters()).device != self.device:
            self.text_encoder = self.text_encoder.to(self.device)
        if self.vae is not None and next(self.vae.parameters()).device != self.device:
            self.vae = self.vae.to(self.device)
        if self.dit is not None and next(self.dit.parameters()).device != self.device:
            self.dit = self.dit.to(self.device)

        # Ensure eval mode
        if self.dit is not None:
            self.dit.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.vae is not None:
            self.vae.eval()

        # Run preprocessing units
        print("\n[DEBUG] Running preprocessing units:")
        for unit in self.units:
            unit_name = unit.__class__.__name__
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            # Debug: check what keys are set after each unit
            if "latents" in inputs_shared:
                print(f"  - {unit_name}: latents shape = {inputs_shared['latents'].shape}")
            if "first_frame_latents" in inputs_shared:
                print(f"    -> first_frame_latents set: {inputs_shared['first_frame_latents'].shape}")
            if "first_frame_latents_rgb" in inputs_shared:
                print(f"    -> first_frame_latents_rgb set: {inputs_shared['first_frame_latents_rgb'].shape}")
            if "fuse_vae_embedding_in_latents" in inputs_shared:
                print(f"    -> fuse_vae_embedding_in_latents = {inputs_shared['fuse_vae_embedding_in_latents']}")

        # Denoise with torch.no_grad() for inference
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        with torch.no_grad():
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                # Switch DiT if necessary
                if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                    self.load_models_to_device(self.in_iteration_models_2)
                    models["dit"] = self.dit2

                # Timestep
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

                # Inference
                noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
                if cfg_scale != 1.0:
                    if cfg_merge:
                        noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    else:
                        noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler step
                inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

                # Preserve first frame latents if available
                if "first_frame_latents" in inputs_shared and "first_frame_latents_rgb" in inputs_shared:
                    first_frame_latents_fused = torch.cat((inputs_shared["first_frame_latents_rgb"], inputs_shared["first_frame_latents"]), dim=1)
                    inputs_shared["latents"][:, :, 0:1] = first_frame_latents_fused

        # VACE post-processing
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Post-denoising processing
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Decode latents to video
        self.load_models_to_device(['vae'])
        num_channels = inputs_shared["latents"].shape[1]
        assert num_channels % 2 == 0, f"Expected even number of channels, got {num_channels}"

        video_heatmap_list = []
        video_rgb_list = []

        for view_idx in range(num_view):
            # Decode RGB
            rgb_latents = inputs_shared["latents"][view_idx:view_idx+1, :num_channels//2]
            video_rgb = self.vae.decode(rgb_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_rgb = self.vae_output_to_video(video_rgb)
            video_rgb_list.append(video_rgb)

            # Decode Heatmap (with inverse scaling if needed)
            heatmap_latents = inputs_shared["latents"][view_idx:view_idx+1, num_channels//2:]
            if HEATMAP_LATENT_SCALE_FACTOR != 1.0:
                heatmap_latents = heatmap_latents / HEATMAP_LATENT_SCALE_FACTOR
            video_heatmap = self.vae.decode(heatmap_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_heatmap = self.vae_output_to_video(video_heatmap)
            video_heatmap_list.append(video_heatmap)

        self.load_models_to_device([])

        return {
            "video_heatmap": video_heatmap_list,
            "video_rgb": video_rgb_list
        }

    def training_loss(self, rgb_loss_weight=0.5, **inputs):
        """
        Calculate training loss.

        Args:
            rgb_loss_weight: Weight for RGB loss (0.0~1.0)
                            total_loss = rgb_loss_weight * loss_rgb + (1 - rgb_loss_weight) * loss_heatmap
            **inputs: Other input parameters
        """
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        noise_pred = self.model_fn(**inputs, timestep=timestep)

        # Debug print (only once on main process)
        if not hasattr(self, '_training_loss_debug_printed'):
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"[DEBUG] Training Loss (MV Concat): noise_pred.shape={noise_pred.shape}, training_target.shape={training_target.shape}")
                print(f"[DEBUG] RGB Loss Weight: {rgb_loss_weight}")
            self._training_loss_debug_printed = True

        # Calculate RGB and Heatmap losses
        num_channels = noise_pred.shape[1]
        assert num_channels % 2 == 0, f"Expected even number of channels, got {num_channels}"

        rgb_channels = num_channels // 2
        noise_pred_rgb = noise_pred[:, :rgb_channels]
        noise_pred_heatmap = noise_pred[:, rgb_channels:]
        training_target_rgb = training_target[:, :rgb_channels]
        training_target_heatmap = training_target[:, rgb_channels:]

        loss_rgb_raw = torch.nn.functional.mse_loss(noise_pred_rgb.float(), training_target_rgb.float())
        loss_rgb_raw = loss_rgb_raw * self.scheduler.training_weight(timestep)

        loss_heatmap_raw = torch.nn.functional.mse_loss(noise_pred_heatmap.float(), training_target_heatmap.float())
        loss_heatmap_raw = loss_heatmap_raw * self.scheduler.training_weight(timestep)

        # Calculate weighted loss
        weighted_loss_rgb = rgb_loss_weight * loss_rgb_raw
        weighted_loss_heatmap = (1.0 - rgb_loss_weight) * loss_heatmap_raw

        loss = weighted_loss_rgb + weighted_loss_heatmap

        # Debug print (only once)
        if not hasattr(self, '_loss_split_debug_printed'):
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"[DEBUG] Loss Split: RGB channels={rgb_channels}, Heatmap channels={rgb_channels}")
                print(f"[DEBUG] Raw loss: loss_rgb_raw={loss_rgb_raw.item():.4f}, loss_heatmap_raw={loss_heatmap_raw.item():.4f}")
            self._loss_split_debug_printed = True

        return {
            "loss": loss,
            "loss_rgb": weighted_loss_rgb,
            "loss_heatmap": weighted_loss_heatmap
        }

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        """Enable VRAM management for memory optimization."""
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer

        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )

        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )


def model_fn_wan_video_mv_concat(
    dit: WanModel_mv_concat,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache=None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    use_dual_head: bool = False,
    **kwargs,
):
    """
    Model forward function for multi-view video generation with token concatenation.

    Key differences from original model_fn_wan_video:
    - No freqs_mvs computation needed (handled inside model)
    - No shape_info parameter for DiT blocks
    - Token concatenation/splitting happens inside WanModel_mv_concat.forward()

    Args:
        dit: WanModel_mv_concat instance
        latents: Multi-view latents, shape (b*v, c, f, h, w)
        timestep: Diffusion timestep
        context: Text conditioning
        use_dual_head: Whether to use dual head mode

    Returns:
        Denoised latents, shape (b*v, c_out, f, h, w)
    """
    num_views = dit.num_views

    # Time embedding
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep_seq = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_seq).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    # Text embedding
    context = dit.text_embedding(context)

    x = latents
    # Merged CFG
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image embedding (should be skipped for this model)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Patchify each view independently
    x, grid_size = dit.patchify(x, control_camera_latents_input)
    f, h, w = grid_size
    seq_per_view = f * h * w

    # Reference image handling
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
        seq_per_view = f * h * w

    # Concatenate view tokens: (b*v, seq, d) -> (b, v*seq, d)
    x = dit.concat_view_tokens(x, num_views)

    # Prepare multi-view position encoding (repeats for all views)
    freqs = dit.prepare_multiview_freqs(grid_size, num_views, x.device)

    # Prepare multi-view time modulation
    t_mod_mv = dit.prepare_multiview_t_mod(t_mod, num_views, seq_per_view)

    # Debug print (only once)
    if not hasattr(model_fn_wan_video_mv_concat, '_debug_printed'):
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print(f"[DEBUG] model_fn_wan_video_mv_concat:")
            print(f"  Input x shape (after concat): {x.shape}")
            print(f"  Grid size: {grid_size}, seq_per_view: {seq_per_view}")
            print(f"  Freqs shape: {freqs.shape}")
            print(f"  Context shape: {context.shape}")
        model_fn_wan_video_mv_concat._debug_printed = True

    # Process through DiT blocks (standard blocks without mvs_attn)
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod_mv, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod_mv, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod_mv, freqs)

        # VACE support (if applicable)
        if vace_context is not None and hasattr(vace, 'vace_layers_mapping') and block_id in vace.vace_layers_mapping:
            # Note: VACE with concat model needs special handling
            pass

    # Split back to multi-view BEFORE applying head(s)
    # Head modulation expects per-view tokens, not concatenated tokens
    x = dit.split_view_tokens(x, num_views, seq_per_view)  # (b, v*seq, d) -> (b*v, seq, d)

    # Debug: print once
    if not hasattr(model_fn_wan_video_mv_concat, '_head_debug_printed'):
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print(f"[DEBUG] model_fn head processing:")
            print(f"  use_dual_head = {use_dual_head}")
            print(f"  dit.use_dual_head = {dit.use_dual_head}")
            print(f"  has head_rgb: {hasattr(dit, 'head_rgb')}")
            print(f"  has head_heatmap: {hasattr(dit, 'head_heatmap')}")
            if hasattr(dit, 'head_rgb') and dit.head_rgb is not None:
                print(f"  head_rgb.head.weight.mean() = {dit.head_rgb.head.weight.mean().item():.6f}")
            if hasattr(dit, 'head_heatmap') and dit.head_heatmap is not None:
                print(f"  head_heatmap.head.weight.mean() = {dit.head_heatmap.head.weight.mean().item():.6f}")
        model_fn_wan_video_mv_concat._head_debug_printed = True

    # Apply head(s) to each view independently
    if use_dual_head:
        x_rgb = dit.head_rgb(x, t)
        x_heatmap = dit.head_heatmap(x, t)

        # Remove reference latents if present
        if reference_latents is not None:
            x_rgb = x_rgb[:, reference_latents.shape[1]:]
            x_heatmap = x_heatmap[:, reference_latents.shape[1]:]
            f -= 1

        # Unpatchify
        x_rgb = dit.unpatchify(x_rgb, (f, h, w))
        x_heatmap = dit.unpatchify(x_heatmap, (f, h, w))

        # Concatenate RGB and Heatmap along channel dimension
        x = torch.cat([x_rgb, x_heatmap], dim=1)

        return x
    else:
        x = dit.head(x, t)

        # Remove reference latents if present
        if reference_latents is not None:
            x = x[:, reference_latents.shape[1]:]
            f -= 1

        x = dit.unpatchify(x, (f, h, w))
        return x


# Alias for compatibility
WanVideoPipeline = WanVideoPipelineMVConcat
