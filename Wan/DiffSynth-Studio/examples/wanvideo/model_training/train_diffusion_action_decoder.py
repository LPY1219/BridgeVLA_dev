"""
Diffusion Action Decoder Training Script

训练一个基于DiT中间特征的action decoder，预测动作变化量（heatmap峰值、旋转、夹爪）

核心设计：
===========
1. 使用预训练的DiT模型（冻结权重）提取中间层特征
2. Action decoder全量训练（不使用LoRA）
3. 预测相对第一帧的变化量，避免过拟合初始状态
4. 支持多GPU训练（accelerate）

训练流程：
===========
1. 加载预训练DiT + VAE + T5（冻结DiT）
2. 初始化DiffusionFeatureExtractor（从DiT指定block提取特征）
3. 初始化DiffusionActionDecoder（全量训练）
4. 对于每个batch：
   a. VAE encode RGB + Heatmap -> latents
   b. T5 encode prompt -> text embeddings
   c. 提取DiT中间特征（随机降噪时间步）
   d. 预测动作变化量
   e. 计算loss（heatmap delta + rotation + gripper）
   f. 反向传播（仅action decoder参数）
5. 保存checkpoint

参考文件：
===========
- Pipeline初始化: heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed.py
- 训练流程对齐: wan_video_5B_TI2V_heatmap_and_rgb_mv.py (training_loss方法)
- 训练参数配置: Wan2.2-TI2V-5B_heatmap_rgb_mv_4.sh
- 数据集: base_multi_view_dataset_with_rot_grip_3cam_different_projection.py

关键注意事项：
==============
1. 数据集必须返回start_pose，否则无法计算旋转delta
2. 旋转delta计算使用角度包装（wrap to [-180, 180]）
3. VAE编码和文本编码完全对齐原始训练流程
4. 时间步采样对齐scheduler的random sampling逻辑
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from PIL import Image

# Add DiffSynth-Studio to path
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, diffsynth_path)

# Import pipeline and models
from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict

# Import datasets
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam import HeatmapDatasetFactory
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam_history import HeatmapDatasetFactoryWithHistory

# Import our modules
from modules.diffusion_feature_extractor import DiffusionFeatureExtractor
from modules.diffusion_action_decoder import DiffusionActionDecoder, compute_action_decoder_loss
from modules.wan_pipeline_loader import load_wan_pipeline

# SwanLab for logging
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


# ============================================================
# Helper Functions
# ============================================================

def angle_to_bin(angle_deg: torch.Tensor, num_bins: int = 72) -> torch.Tensor:
    """将角度（度）转换为bin索引，范围[-180, 180]映射到[0, num_bins)"""
    # 先将角度包装到[-180, 180]
    angle_wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    # 映射到[0, 360)
    angle_normalized = (angle_wrapped + 180.0) % 360.0
    # 转换为bin
    bin_size = 360.0 / num_bins
    bin_index = (angle_normalized / bin_size).long()
    bin_index = torch.clamp(bin_index, 0, num_bins - 1)
    return bin_index


def quaternion_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数(xyzw)转欧拉角(roll, pitch, yaw)，单位：度

    Args:
        quat: (..., 4) 四元数，xyzw格式
    Returns:
        euler: (..., 3) 欧拉角(roll, pitch, yaw)，单位：度
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


def compute_rotation_delta(initial_euler: torch.Tensor, future_euler: torch.Tensor) -> torch.Tensor:
    """
    计算旋转delta，正确处理角度包装

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


# Pipeline loading is now handled by wan_pipeline_loader.py module
# This allows easy swapping of different video diffusion pipelines

def prepare_batch_for_training(
    batch: Dict,
    pipeline: WanVideoPipeline,
    device: str,
    torch_dtype: torch.dtype,
    num_rotation_bins: int = 72,
    normalize_heatmap_delta: bool = True,  # 归一化heatmap_delta到图像尺寸
):
    """
    准备训练batch：VAE编码、T5编码、计算ground truth

    对齐原始训练流程（参考wan_video_5B_TI2V_heatmap_and_rgb_mv.py的prepare逻辑）

    Args:
        batch: 数据集返回的batch (single dict, not list of dicts)
        pipeline: WanVideoPipeline实例
        device: 设备
        torch_dtype: 数据类型
        num_rotation_bins: 旋转bins数量

    Returns:
        rgb_latents: (num_views, c, t, h, w)
        heatmap_latents: (num_views, c, t, h, w)
        text_embeddings: (1, seq_len, dim)
        ground_truth: dict with heatmap_delta, rotation_bins, gripper_change, img_size
    """
    # 检查必需字段
    if 'start_pose' not in batch:
        raise ValueError(
            "Dataset must return 'start_pose' field! "
            "Please modify the dataset to include start_pose in the returned dict."
        )

    # 1. 从数据集获取PIL图像列表
    # batch['video']: List[List[PIL.Image]] - (T+1) x num_views heatmap
    # batch['input_video_rgb']: List[List[PIL.Image]] - (T+1) x num_views RGB
    heatmap_videos = batch['video']  # List[List[PIL.Image]]
    rgb_videos = batch['input_video_rgb']  # List[List[PIL.Image]]

    T_plus_1 = len(heatmap_videos)  # T+1
    num_views = len(heatmap_videos[0])
    T = T_plus_1 - 1

    # 2. VAE编码（对齐原始训练流程）
    def preprocess_image(image):
        img_array = torch.Tensor(np.array(image, dtype=np.float32))
        img_array = img_array.to(dtype=torch_dtype, device=device)
        img_array = img_array * 2.0 / 255.0 - 1.0  # [0, 255] -> [-1, 1]
        img_array = img_array.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return img_array

    def preprocess_video(video):
        tensors = [preprocess_image(img) for img in video]
        video_tensor = torch.stack(tensors, dim=0).squeeze(1)  # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        return video_tensor

    # 按视角编码RGB
    all_rgb_latents = []
    for v in range(num_views):
        view_frames = [rgb_videos[t][v] for t in range(T_plus_1)]
        view_video = preprocess_video(view_frames).squeeze(0)  # (C, T+1, H, W)
        with torch.no_grad():
            latents = pipeline.vae.encode([view_video], device=device, tiled=False)
            latents = latents[0].to(dtype=torch_dtype, device=device)
        all_rgb_latents.append(latents)

    rgb_latents = torch.stack(all_rgb_latents, dim=0)  # (num_views, c, t+1, h, w)

    # 按视角编码Heatmap
    all_heatmap_latents = []
    for v in range(num_views):
        view_frames = [heatmap_videos[t][v] for t in range(T_plus_1)]
        view_video = preprocess_video(view_frames).squeeze(0)
        with torch.no_grad():
            latents = pipeline.vae.encode([view_video], device=device, tiled=False)
            latents = latents[0].to(dtype=torch_dtype, device=device)
        all_heatmap_latents.append(latents)

    heatmap_latents = torch.stack(all_heatmap_latents, dim=0)  # (num_views, c, t+1, h, w)

    # 2.5. 单独编码第一帧并替换（对齐ImageEmbedderFused的行为）
    # 训练时，WanVideoUnit_ImageEmbedderFused会单独encode第一帧，然后替换latents[:, :, 0:1]
    # 这确保第一帧的latent与训练完全一致（避免VAE temporal层的影响）
    # 参考: wan_video_5B_TI2V_heatmap_and_rgb_mv.py, Line 1204-1233
    for v in range(num_views):
        # RGB第一帧
        # pipe.preprocess_image(img).transpose(0, 1) => (1, C, H, W) -> (C, 1, H, W)
        first_rgb_img = preprocess_image(rgb_videos[0][v])  # (1, C, H, W)
        first_rgb_img = first_rgb_img.transpose(0, 1)  # (C, 1, H, W) - align with ImageEmbedderFused
        with torch.no_grad():
            first_rgb_latent = pipeline.vae.encode([first_rgb_img], device=device, tiled=False)
            first_rgb_latent = first_rgb_latent[0].to(dtype=torch_dtype, device=device)  # (c, 1, h, w)

        # Heatmap第一帧
        first_heatmap_img = preprocess_image(heatmap_videos[0][v])  # (1, C, H, W)
        first_heatmap_img = first_heatmap_img.transpose(0, 1)  # (C, 1, H, W)
        with torch.no_grad():
            first_heatmap_latent = pipeline.vae.encode([first_heatmap_img], device=device, tiled=False)
            first_heatmap_latent = first_heatmap_latent[0].to(dtype=torch_dtype, device=device)  # (c, 1, h, w)

        # 替换第一帧（对齐训练时ImageEmbedderFused的行为: latents[:, :, 0:1] = z_fused）
        rgb_latents[v, :, 0:1, :, :] = first_rgb_latent
        heatmap_latents[v, :, 0:1, :, :] = first_heatmap_latent

    # 3. T5编码（使用数据集中的任务指令）
    # CRITICAL: 使用数据集中的实际任务指令，而不是空字符串
    # 这样模型才能学到基于任务指令的条件生成
    prompt = batch.get('prompt', '')  # 从数据集获取任务指令
    if not prompt:
        prompt = ""  # 如果没有指令，fallback 到空字符串

    with torch.no_grad():
        text_embeddings = pipeline.prompter.encode_prompt(prompt, device=device)

    # 4. 计算ground truth
    # Heatmap峰值delta（所有视角）
    # img_locations shape: (num_poses, num_views, 2) or possibly (1, num_poses, num_views, 2)
    # num_poses = 1 (initial) + sequence_length (future frames)
    img_locations = batch['img_locations']

    # Remove extra batch dimension if present
    if img_locations.dim() == 4:
        img_locations = img_locations.squeeze(0)  # (num_poses, num_views, 2)

    # img_locations: (num_poses, num_views, 2) where num_poses = 1 + num_future_frames
    # 计算所有视角的heatmap delta（而不只是view 0）
    initial_peaks = img_locations[0, :, :]  # (num_views, 2) - 所有视角的初始帧
    future_peaks = img_locations[1:, :, :]  # (num_future_frames, num_views, 2) - 所有视角的未来帧
    heatmap_delta = future_peaks - initial_peaks.unsqueeze(0)  # (num_future_frames, num_views, 2)

    # 获取图像尺寸（用于归一化和inference时反归一化）
    img_width = rgb_videos[0][0].size[0]  # PIL.Image.size = (width, height)
    img_height = rgb_videos[0][0].size[1]

    # CRITICAL: 归一化heatmap_delta到[-1, 1]范围（可选，但强烈推荐）
    # 原因：img_locations是像素坐标，delta范围约[-30, 30]像素
    # 网络初始化后输出范围约[-1, 1]，归一化后训练更稳定
    # 归一化后：delta范围变为约[-0.12, 0.12]，与网络输出尺度匹配
    if normalize_heatmap_delta:
        heatmap_delta = heatmap_delta / torch.tensor(
            [img_width, img_height], dtype=heatmap_delta.dtype, device=heatmap_delta.device
        )

    # 旋转delta
    start_pose = batch['start_pose']  # (7,)
    future_poses = batch['future_poses']  # (num_future_frames, 7) - already excludes initial frame

    # Strict assertion: future_poses must match heatmap_delta in temporal dimension
    num_future_frames = heatmap_delta.shape[0]
    assert future_poses.shape[0] == num_future_frames, (
        f"future_poses temporal dimension mismatch! "
        f"Expected {num_future_frames} (from heatmap_delta), got {future_poses.shape[0]}. "
        f"Check SEQUENCE_LENGTH and NUM_FUTURE_FRAMES configuration."
    )

    initial_quat = start_pose[3:]  # (4,) xyzw
    future_quats = future_poses[:, 3:]  # (num_future_frames, 4)

    T = num_future_frames

    initial_euler = quaternion_to_euler(initial_quat.unsqueeze(0)).squeeze(0)  # (3,)
    future_euler = quaternion_to_euler(future_quats)  # (T, 3)

    rotation_delta = compute_rotation_delta(
        initial_euler.unsqueeze(0).expand(T, -1), future_euler
    )  # (T, 3)

    rotation_bins = angle_to_bin(rotation_delta, num_bins=num_rotation_bins)  # (T, 3)

    # 夹爪变化
    start_gripper = batch['start_gripper_state']  # bool or scalar
    future_gripper = batch['future_gripper_states']  # (num_future_frames,) - already excludes initial frame

    # Strict assertion: future_gripper_states must match heatmap_delta in temporal dimension
    assert future_gripper.shape[0] == num_future_frames, (
        f"future_gripper_states temporal dimension mismatch! "
        f"Expected {num_future_frames} (from heatmap_delta), got {future_gripper.shape[0]}. "
        f"Check SEQUENCE_LENGTH and NUM_FUTURE_FRAMES configuration."
    )

    # 转换为tensor（处理np.bool_类型）
    if isinstance(start_gripper, bool):
        start_gripper = torch.tensor(int(start_gripper))
    elif not isinstance(start_gripper, torch.Tensor):
        # 先转为Python int避免np.bool_警告
        start_gripper = torch.tensor(int(start_gripper))

    gripper_change = (future_gripper != start_gripper).long()  # (num_future_frames,)

    ground_truth = {
        'heatmap_delta': heatmap_delta.unsqueeze(0).to(device),  # (1, T, num_views, 2)
        'rotation_bins': rotation_bins.unsqueeze(0).to(device),  # (1, T, 3)
        'gripper_change': gripper_change.unsqueeze(0).to(device),  # (1, T)
        'img_size': (img_width, img_height),  # 用于inference时反归一化
        'is_normalized': normalize_heatmap_delta,  # 标记是否归一化
    }

    return rgb_latents, heatmap_latents, text_embeddings, ground_truth


# ============================================================
# Main Training
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Action Decoder")

    # Paths
    parser.add_argument("--heatmap_data_root", type=str, required=True)
    parser.add_argument("--model_base_path", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # Data
    parser.add_argument("--trail_start", type=int, default=None)
    parser.add_argument("--trail_end", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=25)
    parser.add_argument("--step_interval", type=int, default=1)
    parser.add_argument("--min_trail_length", type=int, default=10)
    parser.add_argument("--heatmap_sigma", type=float, default=1.5)
    parser.add_argument("--colormap_name", type=str, default="jet")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--scene_bounds", type=str, default="-0.1,-0.5,-0.1,0.9,0.5,0.9")
    parser.add_argument("--transform_augmentation_xyz", type=str, default="0.1,0.1,0.1")
    parser.add_argument("--transform_augmentation_rpy", type=str, default="5.0,5.0,5.0")
    parser.add_argument("--num_history_frames", type=int, default=1)
    parser.add_argument("--use_merged_pointcloud", action="store_true")
    parser.add_argument("--use_different_projection", action="store_true", default=True)

    # Model
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV")
    parser.add_argument("--use_dual_head", action="store_true")
    parser.add_argument("--extract_block_id", type=int, default=20)
    parser.add_argument("--dit_feature_dim", type=int, default=3072)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--num_rotation_bins", type=int, default=72)
    parser.add_argument("--num_future_frames", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_dit", action="store_true")
    parser.add_argument("--denoising_timestep_id", type=int, default=None,
                        help="Specific denoising timestep ID for feature extraction. None for random sampling.")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataset_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Enable pin_memory for DataLoader (only for CPU tensors)")
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Loss weights
    parser.add_argument("--heatmap_loss_weight", type=float, default=1.0)
    parser.add_argument("--rotation_loss_weight", type=float, default=1.0)
    parser.add_argument("--gripper_loss_weight", type=float, default=0.5)

    # Logging
    parser.add_argument("--save_epochs_interval", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--enable_swanlab", action="store_true")
    parser.add_argument("--swanlab_api_key", type=str, default="")
    parser.add_argument("--swanlab_project", type=str, default="diffusion_action_decoder")
    parser.add_argument("--swanlab_experiment", type=str, default="experiment")

    args = parser.parse_args()

    # Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs]
    )

    device = accelerator.device
    torch_dtype = torch.bfloat16

    os.makedirs(args.output_path, exist_ok=True)

    if accelerator.is_main_process:
        print("=" * 80)
        print("Diffusion Action Decoder Training")
        print("=" * 80)
        print(f"Output: {args.output_path}")
        print(f"Extract block: {args.extract_block_id}")
        print(f"Hidden dim: {args.hidden_dim}")
        print(f"Rotation bins: {args.num_rotation_bins}")
        print("=" * 80)

    # Load pipeline
    if accelerator.is_main_process:
        print("\n[1/5] Loading pipeline...")

    pipeline = load_wan_pipeline(
        lora_checkpoint_path=args.lora_checkpoint,
        model_base_path=args.model_base_path,
        wan_type=args.wan_type,
        use_dual_head=args.use_dual_head,
        device=device,
        torch_dtype=torch_dtype
    )

    # Initialize scheduler for training mode (CRITICAL: must be done before feature extraction)
    if accelerator.is_main_process:
        print("\n[1.5/5] Setting up scheduler for training...")

    pipeline.scheduler.set_timesteps(1000, training=True)

    if accelerator.is_main_process:
        print(f"  ✓ Scheduler initialized with {len(pipeline.scheduler.timesteps)} timesteps")
        print(f"  ✓ Training mode enabled with Gaussian weighting")

    # Feature extractor
    if accelerator.is_main_process:
        print("\n[2/5] Initializing feature extractor...")

    feature_extractor = DiffusionFeatureExtractor(
        pipeline, args.extract_block_id, freeze_dit=args.freeze_dit, device=device, torch_dtype=torch_dtype
    )

    # Action decoder
    if accelerator.is_main_process:
        print("\n[3/5] Initializing action decoder...")

    action_decoder = DiffusionActionDecoder(
        args.dit_feature_dim, args.hidden_dim, args.num_views,
        args.num_rotation_bins, args.num_future_frames, args.dropout
    ).to(device=device, dtype=torch_dtype)

    if accelerator.is_main_process:
        print(f"  Trainable params: {sum(p.numel() for p in action_decoder.parameters()):,}")

    # Dataset
    if accelerator.is_main_process:
        print("\n[4/5] Loading dataset...")

    scene_bounds = [float(x) for x in args.scene_bounds.split(',')]
    transform_aug_xyz = [float(x) for x in args.transform_augmentation_xyz.split(',')]
    transform_aug_rpy = [float(x) for x in args.transform_augmentation_rpy.split(',')]

    use_history_dataset = args.num_history_frames > 1
    DatasetFactory = HeatmapDatasetFactoryWithHistory if use_history_dataset else HeatmapDatasetFactory

    # 构建数据集参数
    dataset_kwargs = dict(
        data_root=args.heatmap_data_root,
        sequence_length=args.sequence_length,
        step_interval=args.step_interval,
        min_trail_length=args.min_trail_length,
        image_size=(args.height, args.width),
        sigma=args.heatmap_sigma,
        augmentation=True,  # 训练时启用数据增强
        mode="train",
        scene_bounds=scene_bounds,
        transform_augmentation_xyz=transform_aug_xyz,
        transform_augmentation_rpy=transform_aug_rpy,
        debug=args.debug_mode if hasattr(args, 'debug_mode') else False,
        colormap_name=args.colormap_name,
        repeat=args.dataset_repeat,
        wan_type=args.wan_type,
        trail_start=args.trail_start,
        trail_end=args.trail_end,
        use_merged_pointcloud=args.use_merged_pointcloud,
        use_different_projection=args.use_different_projection,
    )

    # 多帧历史需要额外参数
    if use_history_dataset:
        dataset_kwargs['num_history_frames'] = args.num_history_frames

    dataset = DatasetFactory.create_robot_trajectory_dataset(**dataset_kwargs)

    if accelerator.is_main_process:
        print(f"  Dataset size: {len(dataset)}")

    # 自定义collate_fn: 因为batch_size=1且数据集返回PIL.Image对象
    # 默认collate_fn无法处理PIL.Image，所以直接返回第一个元素
    # 同时将所有CUDA tensor移到CPU，避免pin_memory错误
    def simple_collate_fn(batch):
        """
        简单collate函数，batch_size=1时直接返回第一个元素
        递归地将所有CUDA tensor移到CPU，避免pin_memory错误
        """
        sample = batch[0]

        def move_to_cpu(obj):
            """递归地将CUDA tensor移到CPU"""
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: move_to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_cpu(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_cpu(v) for v in obj)
            else:
                return obj

        return move_to_cpu(sample)

    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataset_num_workers,
        pin_memory=False,  # 强制关闭，数据集可能返回CUDA tensor，无法pin_memory
        drop_last=True,
        collate_fn=simple_collate_fn  # 使用自定义collate_fn避免PIL.Image错误
    )

    # Optimizer
    if accelerator.is_main_process:
        print("\n[5/5] Initializing optimizer...")

    optimizer = torch.optim.AdamW(
        action_decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    action_decoder, optimizer, dataloader = accelerator.prepare(action_decoder, optimizer, dataloader)

    # SwanLab
    if accelerator.is_main_process and args.enable_swanlab and SWANLAB_AVAILABLE:
        if args.swanlab_api_key:
            import os as swanlab_os
            swanlab_os.environ['SWANLAB_API_KEY'] = args.swanlab_api_key
        swanlab.init(project=args.swanlab_project, experiment_name=args.swanlab_experiment, config=vars(args))

    # Training loop
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

    global_step = 0

    for epoch in range(args.num_epochs):
        action_decoder.train()
        epoch_metrics = {'loss': 0, 'heatmap': 0, 'rotation': 0, 'gripper': 0, 'rot_acc': 0, 'grip_acc': 0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(action_decoder):
                # Prepare data
                rgb_latents, heatmap_latents, text_embeddings, gt = prepare_batch_for_training(
                    batch, pipeline, device, torch_dtype, args.num_rotation_bins
                )

                # Extract features
                dit_features = feature_extractor(rgb_latents, heatmap_latents, text_embeddings, args.denoising_timestep_id)

                # Predict
                hm_delta, rot_logits, grip_logits = action_decoder(
                    dit_features['features'], dit_features['shape_info'], args.num_views
                )

                # Debug: Print statistics for first batch
                if global_step == 0 and accelerator.is_main_process:
                    norm_status = "NORMALIZED" if gt['is_normalized'] else "PIXEL"
                    img_w, img_h = gt['img_size']
                    print("\n" + "="*80)
                    print(f"FIRST BATCH STATISTICS (Debugging heatmap_loss) - {norm_status}")
                    print("="*80)
                    print(f"Image size: {img_w}x{img_h}")
                    print(f"\nGround Truth heatmap_delta shape: {gt['heatmap_delta'].shape}")
                    print(f"  Min: {gt['heatmap_delta'].min().item():.4f}")
                    print(f"  Max: {gt['heatmap_delta'].max().item():.4f}")
                    print(f"  Mean: {gt['heatmap_delta'].mean().item():.4f}")
                    print(f"  Std: {gt['heatmap_delta'].std().item():.4f}")
                    if gt['is_normalized']:
                        print(f"  Range in pixels: [{gt['heatmap_delta'].min()*img_w:.2f}, {gt['heatmap_delta'].max()*img_h:.2f}]")
                    print(f"\nPredicted heatmap_delta shape: {hm_delta.shape}")
                    print(f"  Min: {hm_delta.min().item():.4f}")
                    print(f"  Max: {hm_delta.max().item():.4f}")
                    print(f"  Mean: {hm_delta.mean().item():.4f}")
                    print(f"  Std: {hm_delta.std().item():.4f}")
                    print(f"\nHeatmap delta L2 distance (normalized): {torch.norm(hm_delta - gt['heatmap_delta']).item():.4f}")
                    print("="*80 + "\n")

                # Loss
                loss_dict = compute_action_decoder_loss(
                    hm_delta, rot_logits, grip_logits,
                    gt['heatmap_delta'], gt['rotation_bins'], gt['gripper_change'],
                    args.num_rotation_bins,
                    args.heatmap_loss_weight, args.rotation_loss_weight, args.gripper_loss_weight,
                    img_size=gt['img_size'],
                    is_normalized=gt['is_normalized']
                )

                # Backward
                accelerator.backward(loss_dict['loss'])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(action_decoder.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Metrics
                epoch_metrics['loss'] += loss_dict['loss'].item()
                epoch_metrics['heatmap'] += loss_dict['heatmap_loss'].item()
                epoch_metrics['rotation'] += loss_dict['rotation_loss'].item()
                epoch_metrics['gripper'] += loss_dict['gripper_loss'].item()
                epoch_metrics['rot_acc'] += loss_dict['rotation_accuracy'].item()
                epoch_metrics['grip_acc'] += loss_dict['gripper_accuracy'].item()

                if accelerator.is_main_process and global_step % args.logging_steps == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['loss'].item():.4f}",
                        'rot_acc': f"{loss_dict['rotation_accuracy'].item():.3f}",
                        'grip_acc': f"{loss_dict['gripper_accuracy'].item():.3f}"
                    })

                    if args.enable_swanlab and SWANLAB_AVAILABLE:
                        swanlab.log({
                            'train/loss': loss_dict['loss'].item(),
                            'train/heatmap_loss': loss_dict['heatmap_loss'].item(),
                            'train/rotation_loss': loss_dict['rotation_loss'].item(),
                            'train/gripper_loss': loss_dict['gripper_loss'].item(),
                            'train/rotation_acc': loss_dict['rotation_accuracy'].item(),
                            'train/gripper_acc': loss_dict['gripper_accuracy'].item(),
                            'step': global_step
                        })

                global_step += 1

        # Epoch summary
        n = len(dataloader)
        for k in epoch_metrics:
            epoch_metrics[k] /= n

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}: Loss={epoch_metrics['loss']:.4f}, "
                  f"RotAcc={epoch_metrics['rot_acc']:.3f}, GripAcc={epoch_metrics['grip_acc']:.3f}")

            if args.enable_swanlab and SWANLAB_AVAILABLE:
                swanlab.log({f'epoch/{k}': v for k, v in epoch_metrics.items()})
                swanlab.log({'epoch': epoch + 1})

        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_epochs_interval == 0:
            ckpt_path = os.path.join(args.output_path, f"epoch-{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': accelerator.unwrap_model(action_decoder).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, ckpt_path)
            print(f"✓ Saved: {ckpt_path}")

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        if args.enable_swanlab and SWANLAB_AVAILABLE:
            swanlab.finish()


if __name__ == "__main__":
    main()
