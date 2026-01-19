"""
Debug script to compare extract_features vs extract_features_with_denoising
"""
import torch
import sys
import os

# Add DiffSynth to path
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, diffsynth_path)
sys.path.insert(0, os.path.join(diffsynth_path, "examples/wanvideo/model_training"))

from modules.wan_pipeline_loader import load_wan_pipeline
from modules.diffusion_feature_extractor import DiffusionFeatureExtractor

# Configuration
model_base_path = "/DATA/disk0/lpy/huggingface/Wan2.2-TI2V-5B-fused"
lora_checkpoint = "/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/6_trajectory_cook_4_3camera_cook_4_pretrain_true_history_1_new_projection_rgb_loss_0.08/20251227_230119/epoch-99.safetensors"
device = "cuda"
torch_dtype = torch.bfloat16

print("=" * 80)
print("Loading pipeline...")
print("=" * 80)

# Load pipeline
pipeline = load_wan_pipeline(
    lora_checkpoint_path=lora_checkpoint,
    model_base_path=model_base_path,
    wan_type="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
    use_dual_head=True,
    device=device,
    torch_dtype=torch_dtype,
)

# Initialize scheduler for training mode (as done in original script)
pipeline.scheduler.set_timesteps(1000, training=True)
print(f"✓ Scheduler initialized with {len(pipeline.scheduler.timesteps)} timesteps (training mode)")

# Create feature extractor
extractor = DiffusionFeatureExtractor(
    pipeline=pipeline,
    extract_block_id=15,
    freeze_dit=True,
    device=device,
    torch_dtype=torch_dtype,
)

# Create dummy data
num_views = 3
c = 48  # latent channels (MUST match model's in_dim / 2)
t = 25  # total frames
h, w = 32, 32  # latent spatial dims

# Simulate: first frame clean, rest are noise
rgb_latents = torch.randn(num_views, c, t, h, w, dtype=torch_dtype, device=device)
heatmap_latents = torch.randn(num_views, c, t, h, w, dtype=torch_dtype, device=device)

# Text embeddings
text_embeddings = pipeline.prompter.encode_prompt("", device=device)

print("\n" + "=" * 80)
print("TEST 1: extract_features (original method, timestep_id=0)")
print("=" * 80)
output1 = extractor.extract_features(
    rgb_latents=rgb_latents,
    heatmap_latents=heatmap_latents,
    text_embeddings=text_embeddings,
    denoising_timestep_id=0,
)
print(f"Output shape: {output1['features'].shape}")
print(f"Timestep used: {output1['timestep']}")

print("\n" + "=" * 80)
print("TEST 2: extract_features_with_denoising (new method, steps=0)")
print("=" * 80)
output2 = extractor.extract_features_with_denoising(
    rgb_latents=rgb_latents,
    heatmap_latents=heatmap_latents,
    text_embeddings=text_embeddings,
    num_denoising_steps=0,
    num_inference_steps=50,
)
print(f"Output shape: {output2['features'].shape}")
print(f"Timestep used: {output2['timestep']}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Shape match: {output1['features'].shape == output2['features'].shape}")
print(f"Timestep 1: {output1['timestep']}")
print(f"Timestep 2: {output2['timestep']}")
print(f"Feature difference (L2): {torch.norm(output1['features'] - output2['features']).item()}")
print(f"Feature 1 mean/std: {output1['features'].mean().item():.4f} / {output1['features'].std().item():.4f}")
print(f"Feature 2 mean/std: {output2['features'].mean().item():.4f} / {output2['features'].std().item():.4f}")
