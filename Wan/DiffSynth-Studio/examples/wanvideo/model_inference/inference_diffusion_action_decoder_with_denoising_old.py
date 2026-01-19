"""
Inference script for Diffusion Action Decoder with Denoising-based Feature Extraction

This script evaluates a trained action decoder on test data with a key difference:
Instead of using ground truth latents with added noise, it uses the diffusion model's
denoising process to generate intermediate features.

Key differences from inference_diffusion_action_decoder.py:
1. Uses extract_features_with_denoising() instead of extract_features()
2. Only the first frame is encoded from ground truth
3. Future frames are initialized as pure noise
4. The diffusion model denoises for N steps before feature extraction
5. Features are extracted from partially denoised latents

This approach better reflects real-world inference where future frames are unknown.

Usage:
    python inference_diffusion_action_decoder_with_denoising.py \
        --model_base_path /path/to/Wan2.2-TI2V-5B-fused \
        --lora_checkpoint /path/to/lora/checkpoint.safetensors \
        --decoder_checkpoint /path/to/decoder/best_model.pth \
        --data_root /path/to/test/data \
        --num_denoising_steps 10 \
        --num_inference_steps 50 \
        --output_dir /path/to/output
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation

# Add DiffSynth to path
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, diffsynth_path)

# Import datasets
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam import HeatmapDatasetFactory

# Import our modules
sys.path.insert(0, os.path.join(diffsynth_path, "examples/wanvideo/model_training"))
from modules.diffusion_feature_extractor import DiffusionFeatureExtractor
from modules.diffusion_action_decoder_old import DiffusionActionDecoder
from modules.wan_pipeline_loader import load_wan_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Action Decoder Inference with Denoising")

    # Model paths
    parser.add_argument("--model_base_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--decoder_checkpoint", type=str, required=True)

    # Data paths
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--trail_start", type=int, default=None)
    parser.add_argument("--trail_end", type=int, default=None)

    # Dataset parameters
    parser.add_argument("--sequence_length", type=int, default=24)
    parser.add_argument("--step_interval", type=int, default=1)
    parser.add_argument("--min_trail_length", type=int, default=10)
    parser.add_argument("--heatmap_sigma", type=float, default=1.5)
    parser.add_argument("--colormap_name", type=str, default="jet")
    parser.add_argument("--scene_bounds", type=str, default="-0.1,-0.5,-0.1,0.9,0.5,0.9")
    parser.add_argument("--transform_augmentation_xyz", type=str, default="0.0,0.0,0.0")
    parser.add_argument("--transform_augmentation_rpy", type=str, default="0.0,0.0,0.0")
    parser.add_argument("--use_different_projection", action="store_true")

    # Model parameters
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP")
    parser.add_argument("--use_dual_head", action="store_true", default=True)
    parser.add_argument("--extract_block_id", type=int, default=20)
    parser.add_argument("--dit_feature_dim", type=int, default=3072)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--num_rotation_bins", type=int, default=72)
    parser.add_argument("--num_future_frames", type=int, default=23)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Denoising parameters (NEW)
    parser.add_argument("--num_denoising_steps", type=int, default=10,
                        help="Number of denoising steps before feature extraction (0 = no denoising)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Total inference steps for scheduler setup")
    parser.add_argument("--num_samples", type=int, default=None)

    # Image parameters
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_predictions", action="store_true")

    # Device
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def load_decoder(args, torch_dtype):
    """Load trained action decoder"""
    print("=" * 80)
    print("Loading Action Decoder...")
    print("=" * 80)

    decoder = DiffusionActionDecoder(
        dit_feature_dim=args.dit_feature_dim,
        hidden_dim=args.hidden_dim,
        num_views=args.num_views,
        num_rotation_bins=args.num_rotation_bins,
        num_future_frames=args.num_future_frames,
        dropout=args.dropout,
    ).to(args.device).to(torch_dtype)

    # Load checkpoint
    print(f"Loading checkpoint: {args.decoder_checkpoint}")
    checkpoint = torch.load(args.decoder_checkpoint, map_location=args.device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch_info = f" (epoch {checkpoint['epoch']})" if 'epoch' in checkpoint else ""
    elif 'decoder_state_dict' in checkpoint:
        state_dict = checkpoint['decoder_state_dict']
        epoch_info = f" (epoch {checkpoint['epoch']})" if 'epoch' in checkpoint else ""
    else:
        state_dict = checkpoint
        epoch_info = ""

    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval()
    print(f"✓ Decoder loaded{epoch_info}")

    return decoder


def quaternion_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数(xyzw)转欧拉角(roll, pitch, yaw)，单位：度 - from training script

    Args:
        quat: (..., 4) quaternion in xyzw format
    Returns:
        euler: (..., 3) Euler angles in degrees (roll, pitch, yaw)
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # 转为度
    roll_deg = roll * 180.0 / np.pi
    pitch_deg = pitch * 180.0 / np.pi
    yaw_deg = yaw * 180.0 / np.pi

    euler = torch.stack([roll_deg, pitch_deg, yaw_deg], dim=-1)
    return euler


def angle_to_bin(angle_deg: torch.Tensor, num_bins: int = 72) -> torch.Tensor:
    """将角度（度）转换为bin索引，范围[-180, 180]映射到[0, num_bins) - from training script"""
    # 先将角度包装到[-180, 180]
    angle_wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    # 映射到[0, 360)
    angle_normalized = (angle_wrapped + 180.0) % 360.0
    # 转换为bin
    bin_size = 360.0 / num_bins
    bin_index = (angle_normalized / bin_size).long()
    bin_index = torch.clamp(bin_index, 0, num_bins - 1)
    return bin_index


def compute_rotation_delta(initial_euler: torch.Tensor, future_euler: torch.Tensor) -> torch.Tensor:
    """
    计算旋转delta，正确处理角度包装 - from training script

    Args:
        initial_euler: (..., 3) 初始欧拉角
        future_euler: (..., 3) 未来欧拉角
    Returns:
        delta: (..., 3) 旋转delta，包装到[-180, 180]
    """
    delta = future_euler - initial_euler
    # 包装到[-180, 180]
    delta = (delta + 180.0) % 360.0 - 180.0
    return delta


def preprocess_image(image, torch_dtype, device):
    """Convert PIL image to tensor (1, C, H, W) in range [-1, 1]"""
    img_array = torch.Tensor(np.array(image, dtype=np.float32))
    img_array = img_array.to(dtype=torch_dtype, device=device)
    img_array = img_array * 2.0 / 255.0 - 1.0  # [0, 255] -> [-1, 1]
    img_array = img_array.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return img_array


def preprocess_video(video, torch_dtype, device):
    """Convert list of PIL images to video tensor (1, C, T, H, W) in range [-1, 1]"""
    tensors = [preprocess_image(img, torch_dtype, device) for img in video]
    video_tensor = torch.stack(tensors, dim=0).squeeze(1)  # (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    return video_tensor


@torch.no_grad()
def evaluate_sample(sample, feature_extractor, decoder, args, torch_dtype):
    """
    Evaluate a single sample using denoising-based feature extraction.

    Key differences from the original evaluate_sample:
    1. Only encode the first frame from ground truth
    2. Initialize future frames as pure noise
    3. Use extract_features_with_denoising() instead of extract_features()
    """

    # Extract data
    img_locations = sample['img_locations']  # (1, T+1, num_views, 2)
    start_pose = sample['start_pose']  # (7,)
    future_poses = sample['future_poses']  # (T, 7)
    start_gripper_state = sample['start_gripper_state']
    future_gripper_states = sample['future_gripper_states']  # (T,)

    input_video_rgb = sample['input_video_rgb']  # List[List[PIL.Image]]
    video_frames = sample['video']  # List[List[PIL.Image]] - heatmaps

    num_views = len(input_video_rgb[0])
    num_total_frames = len(input_video_rgb)  # T+1

    # ========================================================================
    # CRITICAL DIFFERENCE: Only encode FIRST FRAME, not all frames
    # ========================================================================

    # Step 1: Encode ONLY the first frame RGB per view
    all_rgb_latents = []
    for v in range(num_views):
        # Only first frame
        first_rgb_img = preprocess_image(input_video_rgb[0][v], torch_dtype, args.device)  # (1, C, H, W)
        first_rgb_img = first_rgb_img.transpose(0, 1)  # (C, 1, H, W)
        first_rgb_latent = feature_extractor.pipeline.vae.encode([first_rgb_img], device=args.device, tiled=False)
        first_rgb_latent = first_rgb_latent[0].to(dtype=torch_dtype, device=args.device)  # (c, 1, h, w)
        all_rgb_latents.append(first_rgb_latent)

    # Stack: (num_views, c, 1, h, w)
    first_frame_rgb_latents = torch.stack(all_rgb_latents, dim=0)

    # Step 2: Encode ONLY the first frame Heatmap per view
    all_heatmap_latents = []
    for v in range(num_views):
        # Only first frame
        first_heatmap_img = preprocess_image(video_frames[0][v], torch_dtype, args.device)  # (1, C, H, W)
        first_heatmap_img = first_heatmap_img.transpose(0, 1)  # (C, 1, H, W)
        first_heatmap_latent = feature_extractor.pipeline.vae.encode([first_heatmap_img], device=args.device, tiled=False)
        first_heatmap_latent = first_heatmap_latent[0].to(dtype=torch_dtype, device=args.device)  # (c, 1, h, w)
        all_heatmap_latents.append(first_heatmap_latent)

    # Stack: (num_views, c, 1, h, w)
    first_frame_heatmap_latents = torch.stack(all_heatmap_latents, dim=0)

    # Step 3: Initialize future frames as NOISE
    # Get latent dimensions from first frame
    num_views, c, _, h, w = first_frame_rgb_latents.shape

    # CRITICAL: Calculate number of LATENT frames, not raw frames!
    # VAE temporal compression formula: output_frames = (input_frames - 1) // 4 + 1
    # For 25 input frames: (25-1)//4 + 1 = 7 latent frames
    # Since we encoded 1 frame (->1 latent frame), we need 6 more latent frames of noise
    num_latent_frames_total = (num_total_frames - 1) // 4 + 1  # 7 latent frames
    num_latent_frames_noise = num_latent_frames_total - 1  # 6 latent frames of noise

    # Create noise for future frames IN LATENT SPACE
    noise_rgb = torch.randn(
        num_views, c, num_latent_frames_noise, h, w,
        dtype=torch_dtype, device=args.device
    )
    noise_heatmap = torch.randn(
        num_views, c, num_latent_frames_noise, h, w,
        dtype=torch_dtype, device=args.device
    )

    # Concatenate first frame with noise: (num_views, c, T, h, w)
    rgb_latents = torch.cat([first_frame_rgb_latents, noise_rgb], dim=2)
    heatmap_latents = torch.cat([first_frame_heatmap_latents, noise_heatmap], dim=2)

    # Text encoding (empty prompt for unconditional generation)
    text_embeddings = feature_extractor.pipeline.prompter.encode_prompt("", device=args.device)

    # ========================================================================
    # CRITICAL: Use extract_features_with_denoising instead of extract_features
    # ========================================================================
    dit_output = feature_extractor.extract_features_with_denoising(
        rgb_latents=rgb_latents,
        heatmap_latents=heatmap_latents,
        text_embeddings=text_embeddings,
        num_denoising_steps=args.num_denoising_steps,
        num_inference_steps=args.num_inference_steps,
    )

    # Predict actions
    heatmap_delta_pred_norm, rotation_logits_pred, gripper_logits_pred = decoder(
        dit_features=dit_output['features'],
        shape_info=dit_output['shape_info'],
        num_views=num_views,
    )

    # CRITICAL: Denormalize heatmap_delta from [-1, 1] to pixel coordinates
    heatmap_delta_pred = heatmap_delta_pred_norm * torch.tensor(
        [args.width, args.height],
        dtype=heatmap_delta_pred_norm.dtype,
        device=heatmap_delta_pred_norm.device
    )  # (1, T, num_views, 2) in pixels

    # Compute ground truth (same as original)
    if img_locations.dim() == 4:
        img_locations = img_locations.squeeze(0)  # (num_poses, num_views, 2)

    num_future_frames = args.num_future_frames
    expected_num_poses = 1 + num_future_frames
    actual_num_poses = img_locations.shape[0]

    if actual_num_poses != expected_num_poses:
        raise ValueError(
            f"img_locations dimension mismatch!\n"
            f"Expected num_poses = 1 + num_future_frames = 1 + {num_future_frames} = {expected_num_poses}\n"
            f"Got actual_num_poses = {actual_num_poses}\n"
            f"This means the dataset's SEQUENCE_LENGTH doesn't match the model's NUM_FUTURE_FRAMES configuration.\n"
            f"Check that SEQUENCE_LENGTH = {expected_num_poses} in the bash script."
        )

    initial_peaks = img_locations[0, :, :]  # (num_views, 2)
    future_peaks = img_locations[1:, :, :]  # (num_future_frames, num_views, 2)
    heatmap_delta_gt = future_peaks - initial_peaks.unsqueeze(0)  # (num_future_frames, num_views, 2)

    # Rotation bins
    initial_quat = start_pose[3:]  # (4,) xyzw
    future_quats = future_poses[:, 3:]  # (T, 4)
    T = len(future_poses)

    initial_euler = quaternion_to_euler(initial_quat.unsqueeze(0)).squeeze(0)  # (3,)
    future_euler = quaternion_to_euler(future_quats)  # (T, 3)

    rotation_delta = compute_rotation_delta(
        initial_euler.unsqueeze(0).expand(T, -1), future_euler
    )  # (T, 3)

    rotation_bins_gt = angle_to_bin(rotation_delta, num_bins=args.num_rotation_bins)  # (T, 3)

    # Gripper change
    init_grip = int(start_gripper_state) if isinstance(start_gripper_state, bool) else start_gripper_state.item()
    gripper_change_gt = (future_gripper_states.cpu() != init_grip).long()

    # Move to device
    heatmap_delta_gt = heatmap_delta_gt.to(args.device).to(torch_dtype)
    rotation_bins_gt = rotation_bins_gt.to(args.device)
    gripper_change_gt = gripper_change_gt.to(args.device)

    # Compute metrics
    hm_delta_diff = heatmap_delta_pred.squeeze(0) - heatmap_delta_gt  # (T, num_views, 2)
    hm_err = torch.norm(hm_delta_diff, dim=2)  # (T, num_views)
    mean_hm_err = hm_err.mean().item()

    # Rotation accuracy
    rot_logits = rotation_logits_pred.squeeze(0).view(args.num_future_frames, 3, args.num_rotation_bins)
    rot_pred = rot_logits.argmax(dim=2)  # (T, 3)
    rot_correct = (rot_pred == rotation_bins_gt).float()
    rot_acc_per_axis = rot_correct.mean(dim=0)
    rot_acc = rot_correct.mean().item()

    # Gripper accuracy
    grip_pred = gripper_logits_pred.squeeze(0).argmax(dim=1)
    grip_correct = (grip_pred == gripper_change_gt).float()
    grip_acc = grip_correct.mean().item()

    metrics = {
        'heatmap_error_pixels': mean_hm_err,
        'rotation_accuracy': rot_acc,
        'rotation_accuracy_roll': rot_acc_per_axis[0].item(),
        'rotation_accuracy_pitch': rot_acc_per_axis[1].item(),
        'rotation_accuracy_yaw': rot_acc_per_axis[2].item(),
        'gripper_accuracy': grip_acc,
    }

    predictions = None
    if args.save_predictions:
        predictions = {
            'heatmap_delta_pred': heatmap_delta_pred.squeeze(0).float().cpu().numpy().tolist(),
            'heatmap_delta_gt': heatmap_delta_gt.float().cpu().numpy().tolist(),
            'rotation_pred': rot_pred.cpu().numpy().tolist(),
            'rotation_gt': rotation_bins_gt.cpu().numpy().tolist(),
            'gripper_pred': grip_pred.cpu().numpy().tolist(),
            'gripper_gt': gripper_change_gt.cpu().numpy().tolist(),
        }

    return metrics, predictions


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse parameters
    scene_bounds = [float(x) for x in args.scene_bounds.split(',')]
    transform_aug_xyz = [float(x) for x in args.transform_augmentation_xyz.split(',')]
    transform_aug_rpy = [float(x) for x in args.transform_augmentation_rpy.split(',')]

    # Torch dtype
    if args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    # Load pipeline (using decoupled loader)
    pipeline = load_wan_pipeline(
        lora_checkpoint_path=args.lora_checkpoint,
        model_base_path=args.model_base_path,
        wan_type=args.wan_type,
        use_dual_head=args.use_dual_head,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    # CRITICAL: Initialize scheduler with training timesteps (MUST match original version)
    pipeline.scheduler.set_timesteps(1000, training=True)
    print(f"✓ Scheduler initialized with {len(pipeline.scheduler.timesteps)} timesteps (training mode)")

    # Load decoder
    decoder = load_decoder(args, torch_dtype)

    # Create feature extractor
    print("=" * 80)
    print("Creating Feature Extractor...")
    print("=" * 80)
    feature_extractor = DiffusionFeatureExtractor(
        pipeline=pipeline,
        extract_block_id=args.extract_block_id,
        freeze_dit=True,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    print(f"✓ Feature extractor created (block={args.extract_block_id})")
    print(f"  Denoising mode: {args.num_denoising_steps} steps out of {args.num_inference_steps}")

    # Load dataset
    print("=" * 80)
    print("Loading Test Dataset...")
    print("=" * 80)

    test_dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        step_interval=args.step_interval,
        min_trail_length=args.min_trail_length,
        image_size=(args.height, args.width),
        sigma=args.heatmap_sigma,
        augmentation=False,
        mode="test",
        scene_bounds=scene_bounds,
        transform_augmentation_xyz=transform_aug_xyz,
        transform_augmentation_rpy=transform_aug_rpy,
        debug=False,
        colormap_name=args.colormap_name,
        repeat=1,
        wan_type="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
        rotation_resolution=360.0 / args.num_rotation_bins,
        trail_start=args.trail_start,
        trail_end=args.trail_end,
        use_different_projection=args.use_different_projection,
    )

    print(f"✓ Dataset loaded: {len(test_dataset)} samples")

    num_samples = args.num_samples if args.num_samples else len(test_dataset)
    num_samples = min(num_samples, len(test_dataset))

    print(f"\nEvaluating {num_samples} samples with denoising...")
    print(f"  Denoising steps: {args.num_denoising_steps}/{args.num_inference_steps}")
    print("=" * 80)

    # Run inference
    all_metrics = []
    all_predictions = []

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        sample = test_dataset[idx]
        metrics, predictions = evaluate_sample(sample, feature_extractor, decoder, args, torch_dtype)
        all_metrics.append(metrics)
        if predictions:
            all_predictions.append({'sample_idx': idx, **predictions})

    # Report results
    print("=" * 80)
    print("EVALUATION RESULTS (with Denoising)")
    print("=" * 80)

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        std_metrics = np.std([m[key] for m in all_metrics])
        print(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics:.4f}")

    # Save results
    results = {
        'args': vars(args),
        'num_samples_evaluated': num_samples,
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'inference_mode': 'denoising',
        'num_denoising_steps': args.num_denoising_steps,
        'num_inference_steps': args.num_inference_steps,
    }

    if all_predictions:
        results['predictions'] = all_predictions

    results_path = os.path.join(args.output_dir, 'inference_results_with_denoising.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
