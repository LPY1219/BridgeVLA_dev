"""
DiT Intermediate Feature Extractor Module

This module provides functionality to extract intermediate layer features from a pre-trained
DiT model during the denoising process. It's designed to work with the WanVideoPipeline.

Key Features:
- Extract features from specified intermediate blocks of the DiT model
- Support for multi-view video diffusion models
- Random sampling of denoising timesteps (aligned with original training)
- Fixed denoising timestep during inference
- Freeze DiT weights (feature extraction only)
- Uses the correct FlowMatchScheduler for noise addition

Design Philosophy:
- **CRITICAL**: Align with original training process in training_loss()
- Reuse the existing model_fn_wan_video logic for consistency
- Extract features by stopping forward pass at a specific block
- Return intermediate features for action decoder training
- Properly handle flow matching noise schedule

Usage:
    extractor = DiffusionFeatureExtractor(
        pipeline,  # WanVideoPipeline instance
        extract_block_id=20,  # Extract from block 20
        freeze_dit=True
    )

    features = extractor.extract_features(
        rgb_latents,
        heatmap_latents,
        text_embeddings,
        denoising_timestep_id=None  # None for random sampling
    )
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np


def model_fn_wan_video_with_intermediate_output(
    dit,
    motion_controller=None,
    vace=None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    motion_bucket_id: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    use_dual_head: bool = False,
    extract_block_id: Optional[int] = None,  # NEW: Block ID to extract features from
    **kwargs,
):
    """
    Modified version of model_fn_wan_video that can extract intermediate features.

    This function is identical to model_fn_wan_video except:
    1. It can stop at a specific block (extract_block_id)
    2. It returns intermediate features from that block
    3. It also returns shape_info for downstream processing

    Args:
        extract_block_id: If specified, stop at this block and return its output.
                         If None, run the full model (original behavior).
        All other args are the same as model_fn_wan_video.

    Returns:
        If extract_block_id is None:
            Same as model_fn_wan_video (final output)
        If extract_block_id is specified:
            (intermediate_features, shape_info) where:
                intermediate_features: (b*v, seq_len, dim) - output from the specified block
                shape_info: (f, h, w) tuple
    """
    from diffsynth.models.wan_video_dit_mv import sinusoidal_embedding_1d

    # Timestep embedding
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep_expanded = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_expanded).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    # Positional encodings
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    v = 3  # hardcoded number of views
    freqs_mvs = torch.cat([
        dit.freqs[0][:v].view(v, 1, 1, -1).expand(v, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(v, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(v, h, w, -1)
    ], dim=-1).reshape(v * h * w, 1, -1).to(x.device)

    # Process through DiT blocks
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        # If extract_block_id is specified and we reached it, return intermediate features
        if extract_block_id is not None and block_id == extract_block_id:
            return x, (f, h, w)  # Return early with intermediate features

        # Forward through block
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs, freqs_mvs, (f, h, w),
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs, freqs_mvs, (f, h, w),
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs, freqs_mvs, (f, h, w))

    # If we didn't return early, continue with final head
    if use_dual_head:
        x_rgb = dit.head_rgb(x, t)
        x_heatmap = dit.head_heatmap(x, t)
        x_rgb = dit.unpatchify(x_rgb, (f, h, w))
        x_heatmap = dit.unpatchify(x_heatmap, (f, h, w))
        x = torch.cat((x_rgb, x_heatmap), dim=1)
        return x
    else:
        x = dit.head(x, t)
        x = dit.unpatchify(x, (f, h, w))
        return x


class DiffusionFeatureExtractor(nn.Module):
    """
    Extract intermediate features from a pre-trained DiT model.

    This module wraps the WanVideoPipeline and provides methods to extract features
    from a specified intermediate block during the denoising process.

    The key design principle is to align with the original training process in
    WanVideoPipeline.training_loss(), which:
    1. Samples timestep_id randomly from [min_boundary, max_boundary)
    2. Gets timestep from scheduler.timesteps[timestep_id]
    3. Adds noise using scheduler.add_noise()
    4. Runs model_fn_wan_video with the noisy latents

    Args:
        pipeline: WanVideoPipeline instance with loaded models
        extract_block_id: Block ID to extract features from (0 to num_blocks-1)
                         Default: 20 (middle layer for 42-block model)
        freeze_dit: Whether to freeze DiT weights (default: True)
        min_timestep_boundary: Minimum timestep boundary ratio (0.0-1.0), default: 0.0
        max_timestep_boundary: Maximum timestep boundary ratio (0.0-1.0), default: 1.0
        device: Device to run on
        torch_dtype: Data type (default: torch.bfloat16)
    """

    def __init__(
        self,
        pipeline,  # WanVideoPipeline instance
        extract_block_id: int = 20,  # Extract from block 20 by default
        freeze_dit: bool = True,
        min_timestep_boundary: float = 0.0,  # Align with training
        max_timestep_boundary: float = 1.0,  # Align with training
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.pipeline = pipeline
        self.extract_block_id = extract_block_id
        self.min_timestep_boundary = min_timestep_boundary
        self.max_timestep_boundary = max_timestep_boundary
        self.device = device
        self.torch_dtype = torch_dtype

        # Verify block ID is valid
        num_blocks = len(self.pipeline.dit.blocks)
        if extract_block_id >= num_blocks:
            raise ValueError(
                f"extract_block_id={extract_block_id} is out of range. "
                f"Model has {num_blocks} blocks (0 to {num_blocks-1})."
            )

        # Freeze DiT model if specified
        if freeze_dit:
            for param in self.pipeline.dit.parameters():
                param.requires_grad = False
            self.pipeline.dit.eval()
            print(f"DiT model frozen for feature extraction (block {extract_block_id})")

    def extract_features(
        self,
        rgb_latents: torch.Tensor,  # (num_views, c, t, h, w)
        heatmap_latents: torch.Tensor,  # (num_views, c, t, h, w)
        text_embeddings: torch.Tensor,  # (1, seq_len, text_dim)
        denoising_timestep_id: Optional[int] = None,  # Specific timestep_id, None for random
        return_shape_info: bool = True,  # Whether to return shape info
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features from DiT at a specific denoising timestep.

        **IMPORTANT**: This function follows the EXACT same process as WanVideoPipeline.training_loss():
        1. Sample timestep_id from [min_boundary, max_boundary) of scheduler.num_train_timesteps
        2. Get timestep from scheduler.timesteps[timestep_id]
        3. Add noise using scheduler.add_noise(input_latents, noise, timestep)
        4. Run model_fn with fuse_vae_embedding_in_latents=True

        **NOTE**: Removed @torch.no_grad() decorator to preserve gradient flow for action decoder training.
        DiT model is still frozen via requires_grad=False in __init__.

        Args:
            rgb_latents: RGB latents from VAE encoder (num_views, c, t, h, w) - already detached
            heatmap_latents: Heatmap latents from VAE encoder (num_views, c, t, h, w) - already detached
            text_embeddings: Text embeddings from T5 encoder (1, seq_len, text_dim) - already detached
            denoising_timestep_id: Specific timestep_id (index), None for random sampling
            return_shape_info: Whether to return shape information

        Returns:
            Dictionary containing:
                - 'features': Intermediate features from the specified block (b*v, seq_len, dim)
                - 'timestep': The actual timestep value used
                - 'timestep_id': The timestep index used
                - 'shape_info': (f, h, w) tuple (if return_shape_info=True)
                - 'num_views': Number of views
        """
        # 1. Concatenate RGB and heatmap latents
        combined_latents = torch.cat([rgb_latents, heatmap_latents], dim=1)  # (v, 2c, t, h, w)
        num_views, _, t, h, w = combined_latents.shape

        # 2. Sample timestep_id (aligned with training_loss())
        if denoising_timestep_id is None:
            # Random sampling, same as training_loss()
            max_boundary = int(self.max_timestep_boundary * self.pipeline.scheduler.num_train_timesteps)
            min_boundary = int(self.min_timestep_boundary * self.pipeline.scheduler.num_train_timesteps)
            timestep_id = torch.randint(min_boundary, max_boundary, (1,)).item()
        else:
            timestep_id = denoising_timestep_id

        # 3. Convert timestep_id to actual timestep value by indexing scheduler.timesteps
        # NOTE: scheduler.timesteps has been initialized with 1000 timesteps via set_timesteps(1000, training=True)
        # in the training script, so timestep_id in range [0, 1000) is a valid index
        # This matches the reference training code in wan_video_5B_TI2V_heatmap_and_rgb_mv.py:232
        # Use unsqueeze(0) to ensure timestep has shape (1,) instead of ()
        timestep = self.pipeline.scheduler.timesteps[timestep_id].unsqueeze(0).to(
            dtype=self.torch_dtype, device=self.device
        )
        print("timestep:",timestep)

        # 4. Add noise using scheduler (aligned with training_loss())
        noise = torch.randn_like(combined_latents)
        noisy_latents = self.pipeline.scheduler.add_noise(
            combined_latents,  # input_latents
            noise,
            timestep
        )

        # CRITICAL: Keep first frame clean as condition (no noise)
        # FlowMatch add_noise: noisy = (1-sigma) * original + sigma * noise
        # Even if noise=0, first frame gets scaled by (1-sigma), so we must restore it
        noisy_latents[:, :, 0, :, :] = combined_latents[:, :, 0, :, :]

        # 5. Extract intermediate features using modified model_fn
        # DiT is already frozen (requires_grad=False), but we don't use no_grad here
        # to preserve gradient flow for downstream action decoder training
        intermediate_features, shape_info = model_fn_wan_video_with_intermediate_output(
            dit=self.pipeline.dit,
            latents=noisy_latents,
            timestep=timestep,
            context=text_embeddings,
            fuse_vae_embedding_in_latents=True,  # CRITICAL: Same as training
            use_dual_head=False,  # Not relevant for intermediate extraction
            extract_block_id=self.extract_block_id,  # Extract from this block
        )

        # 6. Prepare output
        output = {
            'features': intermediate_features,  # (b*v, seq_len, dim)
            'timestep': timestep.item() if isinstance(timestep, torch.Tensor) else timestep,
            'timestep_id': timestep_id,
            'num_views': num_views,
        }

        if return_shape_info:
            output['shape_info'] = shape_info  # (f, h, w)

        return output

    def extract_features_with_denoising(
        self,
        rgb_latents: torch.Tensor,  # (num_views, c, t, h, w)
        heatmap_latents: torch.Tensor,  # (num_views, c, t, h, w)
        text_embeddings: torch.Tensor,  # (1, seq_len, text_dim)
        num_denoising_steps: int = 0,  # Number of denoising steps before extraction
        num_inference_steps: int = 50,  # Total inference steps (for scheduler setup)
        return_shape_info: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features with optional partial denoising (for inference).

        This method is designed for inference where:
        1. The first frame is clean (from VAE encoding)
        2. Future frames can be pure noise or partially denoised
        3. Supports flexible control of denoising depth

        Usage scenarios:
        - num_denoising_steps=0: Extract features from [clean_frame_1, noise_frames_2_to_24]
          → Fast inference, relies on first frame condition
        - num_denoising_steps=5-10: Light denoising before extraction
          → Balance between speed and semantic richness
        - num_denoising_steps=20-30: Heavy denoising before extraction
          → High quality but slower

        Args:
            rgb_latents: RGB latents (num_views, c, t, h, w)
                        - Frame 1: clean (from VAE encoding)
                        - Frames 2-24: can be pure noise (randn) or anything
            heatmap_latents: Heatmap latents (num_views, c, t, h, w)
                            - Frame 1: clean (from VAE encoding)
                            - Frames 2-24: can be pure noise (randn) or anything
            text_embeddings: Text embeddings (1, seq_len, text_dim)
            num_denoising_steps: Number of denoising steps to perform before extraction
                                0 = no denoising, directly extract features
                                N > 0 = denoise N steps, then extract features
            num_inference_steps: Total inference steps for scheduler setup
                                Default: 50 (standard video generation setting)
            return_shape_info: Whether to return shape information

        Returns:
            Dictionary containing:
                - 'features': Intermediate features (b*v, seq_len, dim)
                - 'timestep': The timestep at which features were extracted
                - 'timestep_id': The timestep index (step number in denoising sequence)
                - 'shape_info': (f, h, w) tuple (if return_shape_info=True)
                - 'num_views': Number of views
        """
        # 1. Setup scheduler
        # CRITICAL: Only reset scheduler if we need to denoise
        # If num_denoising_steps=0, use current scheduler state (should be training mode)
        if num_denoising_steps > 0:
            self.pipeline.scheduler.set_timesteps(num_inference_steps)

        timesteps = self.pipeline.scheduler.timesteps

        # 2. Concatenate RGB and heatmap latents
        combined_latents = torch.cat([rgb_latents, heatmap_latents], dim=1)  # (v, 2c, t, h, w)
        num_views = combined_latents.shape[0]
        latents = combined_latents.clone()

        # 3. Partial denoising (if requested)
        if num_denoising_steps > 0:
            if num_denoising_steps > len(timesteps):
                raise ValueError(
                    f"num_denoising_steps ({num_denoising_steps}) exceeds "
                    f"num_inference_steps ({num_inference_steps})"
                )

            for i in range(num_denoising_steps):
                timestep = timesteps[i].unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

                # DiT forward to predict noise
                with torch.no_grad():  # Denoising doesn't need gradients
                    from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import model_fn_wan_video
                    noise_pred = model_fn_wan_video(
                        dit=self.pipeline.dit,
                        latents=latents,
                        timestep=timestep,
                        context=text_embeddings,
                        fuse_vae_embedding_in_latents=True,
                        use_dual_head=True,  # CRITICAL: Must match model architecture
                    )

                # Denoise one step
                latents = self.pipeline.scheduler.step(noise_pred, timestep, latents)

                # CRITICAL: Keep first frame clean (preserve condition)
                latents[:, :, 0, :, :] = combined_latents[:, :, 0, :, :]

        # 4. Determine the timestep at which to extract features
        # This is the "current noise level" after num_denoising_steps
        if num_denoising_steps < len(timesteps):
            current_timestep = timesteps[num_denoising_steps].unsqueeze(0)
        else:
            current_timestep = timesteps[-1].unsqueeze(0)

        # CRITICAL: Convert timestep to correct device and dtype
        current_timestep = current_timestep.to(dtype=self.torch_dtype, device=self.device)

        # 5. Extract intermediate features at the current timestep
        # Note: We allow gradients here for potential fine-tuning scenarios
        intermediate_features, shape_info = model_fn_wan_video_with_intermediate_output(
            dit=self.pipeline.dit,
            latents=latents,
            timestep=current_timestep,
            context=text_embeddings,
            fuse_vae_embedding_in_latents=True,
            use_dual_head=True,  # CRITICAL: Must match model architecture
            extract_block_id=self.extract_block_id,
        )

        # 6. Prepare output
        output = {
            'features': intermediate_features,  # (b*v, seq_len, dim)
            'timestep': current_timestep.item() if isinstance(current_timestep, torch.Tensor) else current_timestep,
            'timestep_id': num_denoising_steps,  # Represents "how many steps denoised"
            'num_views': num_views,
        }

        if return_shape_info:
            output['shape_info'] = shape_info  # (f, h, w)

        return output

    def forward(
        self,
        rgb_latents: torch.Tensor,
        heatmap_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        denoising_timestep_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for feature extraction.
        Same as extract_features, provided for nn.Module compatibility.
        """
        return self.extract_features(
            rgb_latents,
            heatmap_latents,
            text_embeddings,
            denoising_timestep_id,
            return_shape_info=True
        )
