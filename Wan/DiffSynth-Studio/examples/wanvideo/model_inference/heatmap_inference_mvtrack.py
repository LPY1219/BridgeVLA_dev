"""
MVTrack Inference Script for Wan2.2 Multi-View Video Diffusion Model
用于在MVTrack测试集上评估预训练模型的推断脚本
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
from tqdm import tqdm

# 自动检测根路径
def get_root_path():
    """自动检测BridgeVLA根目录"""
    possible_paths = [
        "/share/project/lpy/BridgeVLA",
        "/DATA/disk1/lpy/BridgeVLA_dev",
        "/DATA/disk0/lpy/BridgeVLA_dev",
        "/DATA/disk2/lpy/BridgeVLA_dev",
        "/home/lpy/BridgeVLA_dev"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")

ROOT_PATH = get_root_path()
print(f"Using ROOT_PATH: {ROOT_PATH}")

# 添加项目路径
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, diffsynth_path)

# 导入多视角pipeline
from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap
from diffsynth.models.wan_video_dit_mv import SelfAttention
import torch.nn as nn


class MVTrackInference:
    """MVTrack多视角热力图推断类"""

    def __init__(self,
                 lora_checkpoint_path: str,
                 wan_type: str,
                 model_base_path: str = None,
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16,
                 use_dual_head: bool = False):
        """
        初始化MVTrack推断器

        Args:
            lora_checkpoint_path: LoRA模型检查点路径
            wan_type: 模型类型
            model_base_path: 基础模型路径
            device: 设备
            torch_dtype: 张量类型
            use_dual_head: 是否使用双head模式
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_checkpoint_path = lora_checkpoint_path
        self.use_dual_head = use_dual_head

        print(f"Loading {wan_type} multi-view pipeline...")

        # 加载diffusion pipeline
        if wan_type == "5B_TI2V_RGB_HEATMAP_MV":
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
            raise ValueError(f"Unsupported wan_type: {wan_type}. Use '5B_TI2V_RGB_HEATMAP_MV'")

        # 初始化多视角模块
        print("Initializing multi-view modules...")
        self._initialize_mv_modules()

        # 加载diffusion模型的LoRA权重
        print(f"Loading LoRA checkpoint: {lora_checkpoint_path}")
        self.load_lora_with_base_weights(lora_checkpoint_path, alpha=1.0)

        print("Pipeline initialized successfully!")

    def _initialize_mv_modules(self):
        """初始化多视角模块"""
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
        加载checkpoint中的所有训练权重
        """
        try:
            print(f"  Loading state dict from: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)

            print(f"  Total keys in checkpoint: {len(state_dict)}")

            # 筛选需要的权重
            patch_embedding_weights = {}
            head_weights = {}
            mv_base_layer_weights = {}
            mv_other_weights = {}
            modulation_weights = {}  # 新增：原始DiT blocks的modulation权重
            lora_weights = {}

            for key, value in state_dict.items():
                if 'lora' in key.lower():
                    lora_weights[key] = value
                    continue

                if 'patch_embedding' in key or 'patch_embed' in key:
                    patch_embedding_weights[key] = value
                elif any(pattern in key for pattern in ['head']):
                    if 'attention' not in key.lower() and 'attn' not in key.lower():
                        head_weights[key] = value
                elif 'base_layer' in key:
                    mv_base_layer_weights[key] = value
                elif any(pattern in key for pattern in ['projector', 'norm_mvs', 'modulation_mvs', 'mvs_attn']):
                    mv_other_weights[key] = value
                # 新增：处理原始DiT blocks的modulation权重（不包括modulation_mvs）
                elif 'modulation' in key and 'modulation_mvs' not in key and 'blocks.' in key:
                    modulation_weights[key] = value

            print(f"  Found {len(patch_embedding_weights)} patch_embedding weights")
            print(f"  Found {len(head_weights)} head weights")
            print(f"  Found {len(mv_base_layer_weights)} MV module base_layer weights")
            print(f"  Found {len(mv_other_weights)} MV module other weights")
            print(f"  Found {len(modulation_weights)} modulation weights (DiT blocks)")  # 新增
            print(f"  Found {len(lora_weights)} LoRA weights")

            # 显示部分关键权重名
            if patch_embedding_weights:
                print(f"    Patch embedding keys: {list(patch_embedding_weights.keys())[:3]}")
            if head_weights:
                print(f"    Head keys: {list(head_weights.keys())[:5]}")

            # 合并要加载的权重
            weights_to_load = {}
            weights_to_load.update(patch_embedding_weights)
            weights_to_load.update(head_weights)
            weights_to_load.update(mv_other_weights)
            weights_to_load.update(modulation_weights)  # 新增：加载modulation权重

            # 转换 base_layer 键名
            for key, value in mv_base_layer_weights.items():
                converted_key = key.replace('.base_layer.', '.')
                weights_to_load[converted_key] = value

            if not weights_to_load:
                print("  Warning: No weights found in checkpoint")
                return

            # 清理权重key
            weights_clean = {}
            for key, value in weights_to_load.items():
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value

            print(f"  Loading {len(weights_clean)} weights into DIT model...")

            missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(
                weights_clean, strict=False
            )

            loaded_keys = set(weights_clean.keys()) - set(unexpected_keys)
            print(f"    ✓ Successfully loaded {len(loaded_keys)}/{len(weights_clean)} weights")

            if unexpected_keys:
                print(f"    ⚠ Unexpected keys (not in model): {len(unexpected_keys)}")
                for k in unexpected_keys[:5]:
                    print(f"      - {k}")

        except Exception as e:
            print(f"  Warning: Failed to load weights: {e}")
            import traceback
            traceback.print_exc()

    def load_lora_with_base_weights(self, checkpoint_path: str, alpha: float = 1.0):
        """加载LoRA权重"""
        print("Loading checkpoint with custom LoRA logic for multiview model...")

        # 先加载所有base layer权重
        print("\nStep 1: Loading base layer weights from checkpoint...")
        self.load_checkpoint_weights(checkpoint_path)

        # 加载LoRA并应用
        print("\nStep 2: Loading and applying LoRA weights...")
        self.pipe.load_lora(self.pipe.dit, checkpoint_path, alpha=alpha)

        print("\n✓ Checkpoint loaded successfully!")

    def find_peak_position(self, heatmap_pil: Image.Image, colormap_name: str = 'jet') -> Tuple[int, int]:
        """从热力图中找到峰值位置"""
        heatmap_np = np.array(heatmap_pil)

        # 重要：extract_heatmap_from_colormap期望输入范围是[0, 1]
        # PIL Image转换后是[0, 255]，需要归一化
        if heatmap_np.max() > 1.0:
            heatmap_np = heatmap_np.astype(np.float32) / 255.0

        heatmap_gray = extract_heatmap_from_colormap(heatmap_np, colormap_name)

        max_idx = np.argmax(heatmap_gray)
        peak_y, peak_x = np.unravel_index(max_idx, heatmap_gray.shape)

        return (int(peak_x), int(peak_y))

    def find_peaks_batch(self, video_frames: List[List[Image.Image]], colormap_name: str = 'jet') -> List[List[Tuple[int, int]]]:
        """批量找到热力图峰值"""
        peaks = []
        for frame_views in video_frames:
            frame_peaks = []
            for view_img in frame_views:
                peak = self.find_peak_position(view_img, colormap_name)
                frame_peaks.append(peak)
            peaks.append(frame_peaks)
        return peaks

    def mark_peaks_on_images(self,
                            video_frames: List[List[Image.Image]],
                            peaks: List[List[Tuple[int, int]]],
                            marker_size: int = 8,
                            marker_color: str = 'red',
                            marker_width: int = 3) -> List[List[Image.Image]]:
        """在图像上标记峰值点"""
        marked_frames = []
        for frame_idx, (frame_views, frame_peaks) in enumerate(zip(video_frames, peaks)):
            marked_views = []
            for view_img, peak in zip(frame_views, frame_peaks):
                img = view_img.copy()
                draw = ImageDraw.Draw(img)
                peak_x, peak_y = peak

                # 绘制十字标记
                draw.line([(peak_x - marker_size, peak_y), (peak_x + marker_size, peak_y)],
                         fill=marker_color, width=marker_width)
                draw.line([(peak_x, peak_y - marker_size), (peak_x, peak_y + marker_size)],
                         fill=marker_color, width=marker_width)

                # 绘制圆圈标记
                draw.ellipse([(peak_x - marker_size, peak_y - marker_size),
                             (peak_x + marker_size, peak_y + marker_size)],
                            outline=marker_color, width=marker_width)

                marked_views.append(img)
            marked_frames.append(marked_views)
        return marked_frames

    @torch.no_grad()
    def predict(self,
                prompt: str,
                input_image: List[Image.Image],
                input_image_rgb: List[Image.Image],
                num_frames: int = 13,
                height: int = 256,
                width: int = 256,
                num_inference_steps: int = 50,
                cfg_scale: float = 1.0,
                seed: int = 0,
                tiled: bool = False,  # 对于小尺寸图像(256x256)应设为False
                **kwargs) -> Dict[str, Any]:
        """
        执行推理，生成视频序列

        Args:
            prompt: 文本提示
            input_image: 多视角热力图起始图像 [view0, view1, view2]
            input_image_rgb: 多视角RGB起始图像 [view0, view1, view2]
            num_frames: 生成帧数 (包括初始帧)
            height, width: 图像尺寸
            num_inference_steps: 推理步数
            cfg_scale: CFG引导强度
            seed: 随机种子
            tiled: 是否使用VAE tiling (小尺寸图像应设为False)

        Returns:
            字典包含:
                - video_heatmap: 生成的热力图帧 List[List[PIL.Image]] (num_views, T)
                - video_rgb: 生成的RGB帧 List[List[PIL.Image]] (num_views, T)
        """
        print("Generating video sequence with diffusion model...")
        print(f"  Prompt: {prompt}")
        print(f"  Input image views: {len(input_image)}, sizes: {[img.size for img in input_image]}")
        print(f"  Input RGB views: {len(input_image_rgb)}, sizes: {[img.size for img in input_image_rgb]}")
        print(f"  Output: {num_frames} frames at {height}x{width}")
        print(f"  Inference steps: {num_inference_steps}, CFG scale: {cfg_scale}, Seed: {seed}")
        print(f"  Tiled: {tiled}, Dual head: {self.use_dual_head}")

        video_heatmap_frames, video_rgb_frames = self.pipe(
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
            tiled=tiled,  # 关键：对于256x256的图像，应该设为False
            **kwargs
        )

        print(f"  Generated heatmap views: {len(video_heatmap_frames)}, frames per view: {len(video_heatmap_frames[0]) if video_heatmap_frames else 0}")
        print(f"  Generated RGB views: {len(video_rgb_frames)}, frames per view: {len(video_rgb_frames[0]) if video_rgb_frames else 0}")

        return {
            'video_heatmap': video_heatmap_frames,  # List[List[PIL.Image]] (num_views, T)
            'video_rgb': video_rgb_frames,  # List[List[PIL.Image]] (num_views, T)
        }


def visualize_comparison(gt_sample: Dict,
                        pred_result: Dict,
                        save_path: str,
                        max_frames: int = 13):
    """
    可视化GT和预测结果的对比

    Args:
        gt_sample: 包含GT数据的字典
        pred_result: 预测结果字典
        save_path: 保存路径
        max_frames: 最大显示帧数
    """
    # GT数据
    gt_video_heatmap = gt_sample['video']  # (T, num_views)
    gt_video_rgb = gt_sample['input_video_rgb']  # (T, num_views)

    # 预测数据 (num_views, T) -> 转置为 (T, num_views)
    pred_heatmaps = pred_result['video_heatmap']
    pred_rgbs = pred_result['video_rgb']

    num_views = len(pred_heatmaps)
    num_frames = min(len(pred_heatmaps[0]), max_frames)

    # 转置预测数据
    pred_video_heatmap = [[pred_heatmaps[v][t] for v in range(num_views)] for t in range(num_frames)]
    pred_video_rgb = [[pred_rgbs[v][t] for v in range(num_views)] for t in range(num_frames)]

    # 限制GT帧数
    gt_video_heatmap = gt_video_heatmap[:num_frames]
    gt_video_rgb = gt_video_rgb[:num_frames]

    # 创建图像: 4行 × (num_frames * num_views // 3) 列
    # Row 1: GT Heatmap
    # Row 2: Pred Heatmap
    # Row 3: GT RGB
    # Row 4: Pred RGB
    fig_height = 16
    fig_width = num_frames * 3

    fig, axes = plt.subplots(4 * num_views, num_frames, figsize=(fig_width, fig_height * num_views // 3))

    if num_frames == 1:
        axes = axes.reshape(-1, 1)

    for v in range(num_views):
        for t in range(num_frames):
            # GT Heatmap
            ax = axes[v * 4 + 0, t]
            if t < len(gt_video_heatmap) and v < len(gt_video_heatmap[t]):
                ax.imshow(np.array(gt_video_heatmap[t][v]))
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(f'V{v} GT HM', fontsize=8)
            if v == 0:
                ax.set_title(f'T={t}', fontsize=8)

            # Pred Heatmap
            ax = axes[v * 4 + 1, t]
            if t < len(pred_video_heatmap) and v < len(pred_video_heatmap[t]):
                ax.imshow(np.array(pred_video_heatmap[t][v]))
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(f'V{v} Pred HM', fontsize=8)

            # GT RGB
            ax = axes[v * 4 + 2, t]
            if t < len(gt_video_rgb) and v < len(gt_video_rgb[t]):
                ax.imshow(np.array(gt_video_rgb[t][v]))
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(f'V{v} GT RGB', fontsize=8)

            # Pred RGB
            ax = axes[v * 4 + 3, t]
            if t < len(pred_video_rgb) and v < len(pred_video_rgb[t]):
                ax.imshow(np.array(pred_video_rgb[t][v]))
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(f'V{v} Pred RGB', fontsize=8)

    plt.suptitle(f"Prompt: {gt_sample['prompt']}", fontsize=12, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")


def visualize_simple_comparison(gt_sample: Dict,
                                pred_result: Dict,
                                save_path: str,
                                inferencer: MVTrackInference = None):
    """
    简化的对比可视化：仅显示第一帧和最后一帧，标记预测位置

    Args:
        gt_sample: 包含GT数据的字典
        pred_result: 预测结果字典
        save_path: 保存路径
        inferencer: 推断器实例（用于找峰值）
    """
    # GT数据
    gt_video_heatmap = gt_sample['video']  # (T, num_views)
    gt_video_rgb = gt_sample['input_video_rgb']  # (T, num_views)

    # 预测数据
    pred_heatmaps = pred_result['video_heatmap']  # (num_views, T)
    pred_rgbs = pred_result['video_rgb']  # (num_views, T)

    num_views = len(pred_heatmaps)
    num_gt_frames = len(gt_video_heatmap)
    num_pred_frames = len(pred_heatmaps[0])

    # 选择要显示的帧：第一帧和最后一帧
    frame_indices = [0, min(num_gt_frames, num_pred_frames) - 1]

    # 创建图像: 2行(GT/Pred) × (2帧 * num_views)列 × 2(Heatmap/RGB)
    fig, axes = plt.subplots(4, num_views * 2, figsize=(num_views * 8, 12))

    for frame_plot_idx, frame_idx in enumerate(frame_indices):
        for v in range(num_views):
            col = frame_plot_idx * num_views + v

            # GT Heatmap
            ax = axes[0, col]
            if frame_idx < len(gt_video_heatmap) and v < len(gt_video_heatmap[frame_idx]):
                ax.imshow(np.array(gt_video_heatmap[frame_idx][v]))
            ax.axis('off')
            ax.set_title(f'GT HM V{v} T{frame_idx}', fontsize=10)

            # Pred Heatmap
            ax = axes[1, col]
            if frame_idx < num_pred_frames:
                ax.imshow(np.array(pred_heatmaps[v][frame_idx]))
            ax.axis('off')
            ax.set_title(f'Pred HM V{v} T{frame_idx}', fontsize=10)

            # GT RGB with GT peak marked
            ax = axes[2, col]
            if frame_idx < len(gt_video_rgb) and v < len(gt_video_rgb[frame_idx]):
                rgb_img = gt_video_rgb[frame_idx][v].copy()
                if inferencer is not None and frame_idx < len(gt_video_heatmap):
                    peak = inferencer.find_peak_position(gt_video_heatmap[frame_idx][v])
                    draw = ImageDraw.Draw(rgb_img)
                    marker_size = 8
                    draw.ellipse([(peak[0] - marker_size, peak[1] - marker_size),
                                 (peak[0] + marker_size, peak[1] + marker_size)],
                                outline='green', width=3)
                ax.imshow(np.array(rgb_img))
            ax.axis('off')
            ax.set_title(f'GT RGB V{v} T{frame_idx}', fontsize=10)

            # Pred RGB with Pred peak marked
            ax = axes[3, col]
            if frame_idx < num_pred_frames:
                rgb_img = pred_rgbs[v][frame_idx].copy()
                if inferencer is not None:
                    peak = inferencer.find_peak_position(pred_heatmaps[v][frame_idx])
                    draw = ImageDraw.Draw(rgb_img)
                    marker_size = 8
                    draw.ellipse([(peak[0] - marker_size, peak[1] - marker_size),
                                 (peak[0] + marker_size, peak[1] + marker_size)],
                                outline='red', width=3)
                ax.imshow(np.array(rgb_img))
            ax.axis('off')
            ax.set_title(f'Pred RGB V{v} T{frame_idx}', fontsize=10)

    plt.suptitle(f"Prompt: {gt_sample['prompt']}\nGreen=GT, Red=Pred", fontsize=12, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Simple visualization saved to: {save_path}")


def compute_heatmap_metrics(gt_sample: Dict, pred_result: Dict, inferencer: MVTrackInference) -> Dict[str, float]:
    """
    计算热力图预测的评估指标

    Args:
        gt_sample: GT数据
        pred_result: 预测结果
        inferencer: 推断器

    Returns:
        metrics字典
    """
    gt_video = gt_sample['video']  # (T, num_views)
    pred_heatmaps = pred_result['video_heatmap']  # (num_views, T)

    num_views = len(pred_heatmaps)
    num_frames = min(len(gt_video), len(pred_heatmaps[0]))

    # 计算每帧每视角的峰值位置误差
    peak_errors = []
    for t in range(num_frames):
        for v in range(num_views):
            if t < len(gt_video) and v < len(gt_video[t]):
                gt_peak = inferencer.find_peak_position(gt_video[t][v])
                pred_peak = inferencer.find_peak_position(pred_heatmaps[v][t])

                error = np.sqrt((gt_peak[0] - pred_peak[0])**2 + (gt_peak[1] - pred_peak[1])**2)
                peak_errors.append(error)

    metrics = {
        'mean_peak_error': np.mean(peak_errors) if peak_errors else 0.0,
        'max_peak_error': np.max(peak_errors) if peak_errors else 0.0,
        'min_peak_error': np.min(peak_errors) if peak_errors else 0.0,
        'std_peak_error': np.std(peak_errors) if peak_errors else 0.0,
    }

    return metrics


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='MVTrack Multi-View Inference Script')

    # 模型配置
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="LoRA checkpoint path")
    parser.add_argument("--model_base_path", type=str, default="/data/lpy/huggingface/Wan2.2-TI2V-5B-fused")
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV")
    parser.add_argument("--use_dual_head", action='store_true', help='Use dual head mode')
    parser.add_argument("--device", type=str, default="cuda")

    # 数据集配置
    parser.add_argument("--data_root", type=str, required=True, help="MVTrack dataset root directory")
    parser.add_argument("--split_file", type=str, default="test_split.txt", help="Split file name")
    parser.add_argument("--sequence_length", type=int, default=12, help="Sequence length (not including first frame)")
    parser.add_argument("--step_interval", type=int, default=2, help="Frame sampling interval")
    parser.add_argument("--num_views", type=int, default=3, help="Number of views")
    parser.add_argument("--heatmap_sigma", type=float, default=2.0, help="Heatmap sigma")
    parser.add_argument("--img_size", type=str, default="256,256", help="Image size: height,width")

    # 推理配置
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 测试配置
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--test_indices", type=str, default="", help="Specific indices to test (comma-separated)")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./mvtrack_inference_results")

    args = parser.parse_args()

    # 解析参数
    args.img_size = [int(x.strip()) for x in args.img_size.split(',')]

    print("=" * 80)
    print("MVTrack Multi-View Inference Test")
    print("=" * 80)
    print(f"  LoRA Checkpoint: {args.lora_checkpoint}")
    print(f"  Model Base Path: {args.model_base_path}")
    print(f"  WAN Type: {args.wan_type}")
    print(f"  Dual Head: {args.use_dual_head}")
    print(f"  Data Root: {args.data_root}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Num Views: {args.num_views}")
    print(f"  Sequence Length: {args.sequence_length}")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建数据集
    print("\nCreating MVTrack test dataset...")
    sys.path.append(os.path.join(diffsynth_path, "examples/wanvideo/model_training"))
    from diffsynth.trainers.mvtrack_dataset import MVTrackDataset

    split_file_path = os.path.join(args.data_root, args.split_file)
    if not os.path.exists(split_file_path):
        print(f"Split file not found: {split_file_path}")
        print("Available split files:")
        for f in os.listdir(args.data_root):
            if f.endswith('.txt'):
                print(f"  - {f}")
        return

    dataset = MVTrackDataset(
        data_root=args.data_root,
        split_file=split_file_path,
        sequence_length=args.sequence_length,
        step_interval=args.step_interval,
        min_sequence_length=args.sequence_length * args.step_interval + 1,
        image_size=tuple(args.img_size),
        sigma=args.heatmap_sigma,
        num_views=args.num_views,
        augmentation=False,  # 测试时不使用数据增强
        debug=False,
    )

    print(f"Dataset created with {len(dataset)} samples")

    # 确定要测试的样本索引
    if args.test_indices:
        test_indices = [int(x.strip()) for x in args.test_indices.split(',')]
    else:
        # 均匀采样
        total_samples = len(dataset)
        if total_samples <= args.num_samples:
            test_indices = list(range(total_samples))
        else:
            step = total_samples // args.num_samples
            test_indices = [i * step for i in range(args.num_samples)]

    print(f"Testing {len(test_indices)} samples: {test_indices}")

    # 创建推断器
    print("\nInitializing inference model...")
    inferencer = MVTrackInference(
        lora_checkpoint_path=args.lora_checkpoint,
        wan_type=args.wan_type,
        model_base_path=args.model_base_path,
        device=args.device,
        use_dual_head=args.use_dual_head,
    )

    # 运行推理
    all_metrics = []
    for idx, sample_idx in enumerate(tqdm(test_indices, desc="Running inference")):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(test_indices)} (index={sample_idx})")

        # 获取GT样本
        gt_sample = dataset[sample_idx]
        prompt = gt_sample['prompt']
        print(f"  Prompt: {prompt}")

        # 准备输入
        input_image = gt_sample['input_image']  # 第一帧热力图
        input_image_rgb = gt_sample['input_image_rgb']  # 第一帧RGB

        # 运行推理
        pred_result = inferencer.predict(
            prompt=prompt,
            input_image=input_image,
            input_image_rgb=input_image_rgb,
            num_frames=args.sequence_length + 1,
            height=args.img_size[0],
            width=args.img_size[1],
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )

        # 计算指标
        metrics = compute_heatmap_metrics(gt_sample, pred_result, inferencer)
        all_metrics.append(metrics)
        print(f"  Mean Peak Error: {metrics['mean_peak_error']:.2f} px")

        # 保存可视化
        sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_idx:04d}")
        os.makedirs(sample_output_dir, exist_ok=True)

        # 完整对比图
        visualize_comparison(
            gt_sample, pred_result,
            os.path.join(sample_output_dir, "full_comparison.png"),
            max_frames=args.sequence_length + 1
        )

        # 简化对比图
        visualize_simple_comparison(
            gt_sample, pred_result,
            os.path.join(sample_output_dir, "simple_comparison.png"),
            inferencer=inferencer
        )

        # 保存单独的帧
        pred_heatmaps = pred_result['video_heatmap']
        pred_rgbs = pred_result['video_rgb']
        num_views = len(pred_heatmaps)
        num_frames = len(pred_heatmaps[0])

        for t in range(min(num_frames, 5)):  # 保存前5帧
            for v in range(num_views):
                pred_heatmaps[v][t].save(os.path.join(sample_output_dir, f"pred_heatmap_t{t}_v{v}.png"))
                pred_rgbs[v][t].save(os.path.join(sample_output_dir, f"pred_rgb_t{t}_v{v}.png"))
                if t < len(gt_sample['video']) and v < len(gt_sample['video'][t]):
                    gt_sample['video'][t][v].save(os.path.join(sample_output_dir, f"gt_heatmap_t{t}_v{v}.png"))
                    gt_sample['input_video_rgb'][t][v].save(os.path.join(sample_output_dir, f"gt_rgb_t{t}_v{v}.png"))

    # 汇总指标
    print("\n" + "=" * 80)
    print("Summary Metrics")
    print("=" * 80)

    mean_peak_errors = [m['mean_peak_error'] for m in all_metrics]
    print(f"  Overall Mean Peak Error: {np.mean(mean_peak_errors):.2f} ± {np.std(mean_peak_errors):.2f} px")
    print(f"  Best Sample Error: {np.min(mean_peak_errors):.2f} px")
    print(f"  Worst Sample Error: {np.max(mean_peak_errors):.2f} px")

    # 保存指标到文件
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("MVTrack Inference Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.lora_checkpoint}\n")
        f.write(f"Samples tested: {len(test_indices)}\n")
        f.write(f"Overall Mean Peak Error: {np.mean(mean_peak_errors):.2f} ± {np.std(mean_peak_errors):.2f} px\n")
        f.write("\nPer-sample metrics:\n")
        for idx, (sample_idx, metrics) in enumerate(zip(test_indices, all_metrics)):
            f.write(f"  Sample {sample_idx}: Mean={metrics['mean_peak_error']:.2f}px\n")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
