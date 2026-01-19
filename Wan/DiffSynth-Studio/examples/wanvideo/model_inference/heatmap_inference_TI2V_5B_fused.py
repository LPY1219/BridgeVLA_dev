"""
Heatmap Inference Script for Wan2.2
用于热力图序列预测的推断脚本
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Any
import cv2
from pathlib import Path

# 自动检测根路径
def get_root_path():
    """自动检测BridgeVLA根目录"""
    possible_paths = [
        "/share/project/lpy/BridgeVLA",
        "/home/lpy/BridgeVLA_dev"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")

ROOT_PATH = get_root_path()
print(f"Using ROOT_PATH: {ROOT_PATH}")

# 添加项目路径
sys.path.append(f"{ROOT_PATH}/Wan/DiffSynth-Studio")
sys.path.append(f"{ROOT_PATH}/Wan/single_view")

from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap

# 导入训练时的数据集工厂

from diffsynth.trainers.heatmap_dataset import HeatmapDatasetFactory
from data.dataset import RobotTrajectoryDataset, ProjectionInterface
DATASET_AVAILABLE = True



class HeatmapInference:
    """热力图推断类"""

    def __init__(self,
                 lora_checkpoint_path: str,
                 wan_type: str,
                 model_base_path: str = None,
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16,
                 use_dual_head: bool = False):
        """
        初始化推断器

        Args:
            lora_checkpoint_path: LoRA模型检查点路径
            model_base_path: 基础模型路径
            device: 设备
            torch_dtype: 张量类型
            use_dual_head: 是否使用双head模式（需要与训练时一致）
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_checkpoint_path = lora_checkpoint_path
        self.use_dual_head = use_dual_head

        print(f"Loading {wan_type} pipeline (use_dual_head={use_dual_head})...")
        # 加载pipeline
        if wan_type=="WAN_2_1_14B_I2V":
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch_dtype,
                device=device,
                wan_type=wan_type,
                use_dual_head=use_dual_head,
                model_configs=[
                    ModelConfig(path=[
                        f"{model_base_path}/diffusion_pytorch_model-00001-of-00007.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00002-of-00007.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00003-of-00007.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00004-of-00007.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00005-of-00007.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00006-of-00007.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00007-of-00007.safetensors"
                    ]),
                    ModelConfig(path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth"),
                    ModelConfig(path=f"{model_base_path}/Wan2.1_VAE.pth"),
                    ModelConfig(path=f"{model_base_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                ],
            )
        elif wan_type=="5B_TI2V_RGB_HEATMAP":
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch_dtype,
                device=device,
                wan_type=wan_type,
                use_dual_head=use_dual_head,
                model_configs=[
                    ModelConfig(path=[

                        f"{model_base_path}/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
                        f"{model_base_path}/diffusion_pytorch_model-00003-of-00003-bf16.safetensors"
                    ]),
                    ModelConfig(path=f"{model_base_path}/models_t5_umt5-xxl-enc-bf16.pth"),
                    ModelConfig(path=f"{model_base_path}/Wan2.2_VAE.pth"),
                ],
            )
        else:
            raise ValueError(f"Unsupported wan_type: {wan_type}")
            

        # 使用pipeline的load_lora方法加载checkpoint（包含LoRA和全量训练参数）
        print(f"Loading checkpoint: {lora_checkpoint_path}")
        self.pipe.load_lora(self.pipe.dit, lora_checkpoint_path, alpha=1.0)
        
        # 加载patch_embedding和head相关的全量训练权重
        print("Loading patch_embedding and head weights...")
        self.load_patch_embedding_and_head_weights(lora_checkpoint_path)
        
        print("✓ Checkpoint loaded successfully!")

        print("Pipeline initialized successfully!")

    def load_patch_embedding_and_head_weights(self, checkpoint_path: str):
        """
        只加载patch_embedding和head相关的全量训练权重
        
        Args:
            checkpoint_path: checkpoint文件路径
        """
        try:
            # 加载checkpoint
            print(f"  Loading state dict from: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)
            
            # 筛选patch_embedding和head相关的权重
            patch_embedding_weights = {}
            head_weights = {}
            
            for key, value in state_dict.items():
                # 跳过LoRA相关的权重
                if 'lora' in key.lower():
                    continue
                
                # 筛选patch_embedding相关的权重
                # 可能的key格式: 'dit.patch_embedding.xxx', 'model.patch_embedding.xxx', 'patch_embedding.xxx'
                if 'patch_embedding' in key or 'patch_embed' in key:
                    patch_embedding_weights[key] = value
                
                # 筛选head相关的权重
                # 可能的key格式: 'dit.head.xxx', 'model.head.xxx', 'head.xxx', 
                # 'dit.final_layer.xxx', 'output_head.xxx'等
                if any(pattern in key for pattern in ['head']):
                    # 排除attention head等不相关的
                    if 'attention' not in key.lower() and 'attn' not in key.lower():
                        head_weights[key] = value
            
            print(f"  Found {len(patch_embedding_weights)} patch_embedding weights")
            print(f"  Found {len(head_weights)} head weights")
            
            # 显示找到的权重key样例
            if patch_embedding_weights:
                print("  Patch embedding keys (sample):")
                for i, key in enumerate(list(patch_embedding_weights.keys())):
                    print(f"    - {key}")
            
            if head_weights:
                print("  Head keys (sample):")
                for i, key in enumerate(list(head_weights.keys())):
                    print(f"    - {key}")
            
            # 合并要加载的权重
            weights_to_load = {}
            weights_to_load.update(patch_embedding_weights)
            weights_to_load.update(head_weights)
            
            if not weights_to_load:
                print("  Warning: No patch_embedding or head weights found in checkpoint")
                return
            
            # 清理权重key（移除前缀）
            weights_clean = {}
            for key, value in weights_to_load.items():
                # 移除可能的前缀: 'dit.', 'model.'
                clean_key = key
                for prefix in ['dit.', 'model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                weights_clean[clean_key] = value
            
            print(f"  Loading {len(weights_clean)} weights into DIT model...")
            
            # 加载到DIT模型中
            missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(
                weights_clean, strict=False
            )
            
            # 统计成功加载的权重
            loaded_keys = set(weights_clean.keys()) - set(unexpected_keys)
            
            print(f"    ✓ Successfully loaded {len(loaded_keys)}/{len(weights_clean)} weights")
            
            if missing_keys:
                # 只显示patch_embedding和head相关的missing keys
                relevant_missing = [k for k in missing_keys if 'patch_embedding' in k or 'patch_embed' in k or 'head' in k or 'final_layer' in k]
                if relevant_missing:
                    print(f"    Warning: {len(relevant_missing)} relevant keys not found in checkpoint:")
                    for key in relevant_missing[:5]:
                        print(f"      - {key}")
            
            if unexpected_keys:
                print(f"    Note: {len(unexpected_keys)} keys not found in model (may be normal):")
                for key in unexpected_keys[:5]:
                    print(f"      - {key}")
            
            print("  ✓ Patch embedding and head weights loaded successfully!")
            
        except Exception as e:
            print(f"  Warning: Failed to load patch_embedding and head weights: {e}")
            print(f"  Continuing with LoRA weights only...")
            import traceback
            traceback.print_exc()



    def predict_heatmap_sequence(self,
                                input_image: Image.Image,
                                prompt: str,
                                input_image_rgb: Image.Image,
                                num_frames: int = 5,
                                height: int = 256,
                                width: int = 256,
                                seed: int = None) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        预测热力图序列和RGB视频

        Args:
            input_image: 第一帧热力图
            prompt: 语言指令
            input_image_rgb: 第一帧RGB图像
            num_frames: 预测帧数
            height: 输出高度
            width: 输出宽度
            seed: 随机种子

        Returns:
            (预测的热力图序列, 预测的RGB视频序列)
        """
        print(f"  Predicting heatmap and RGB sequence...")
        print(f"  Input heatmap size: {input_image.size}")
        print(f"  Input RGB size: {input_image_rgb.size}")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {num_frames} frames of {width}x{height}")

        # 调整输入图像尺寸
        input_image_resized = input_image.resize((width, height))
        input_image_rgb_resized = input_image_rgb.resize((width, height))

        # 生成视频序列 - pipeline应该返回(heatmap_frames, rgb_frames)
        result = self.pipe(
            prompt=prompt,
            input_image=input_image_resized,
            input_image_rgb=input_image_rgb_resized,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            use_dual_head=self.use_dual_head,
        )

        # 检查返回值格式
        if isinstance(result, tuple) and len(result) == 2:
            heatmap_frames, rgb_frames = result
            print(f"Generated {len(heatmap_frames)} heatmap frames and {len(rgb_frames)} RGB frames")
        else:
            # 如果只返回热力图，RGB设为None
            heatmap_frames = result
            rgb_frames = None
            print(f"Generated {len(heatmap_frames)} heatmap frames (no RGB output)")

        return heatmap_frames, rgb_frames

    def find_peak_position(self, heatmap_image: Image.Image, colormap_name: str = 'jet') -> Tuple[int, int]:
        """
        找到热力图中的峰值位置

        Args:
            heatmap_image: 热力图PIL图像（可能是colormap格式）
            colormap_name: 使用的colormap名称

        Returns:
            (x, y) 峰值位置坐标
        """
        # 将PIL图像转换为numpy数组
        rgb_array = np.array(heatmap_image)

        # 检查是否是RGB格式（colormap格式）
        if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
            # 如果是RGB格式，从colormap提取热力图数值
            rgb_normalized = rgb_array.astype(np.float32) / 255.0
            heatmap_array = extract_heatmap_from_colormap(rgb_normalized, colormap_name)
        else:
            # 如果是灰度图，直接使用
            heatmap_array = np.array(heatmap_image.convert('L')).astype(np.float32) / 255.0

        # 找到最大值位置
        max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)

        # 返回(x, y)格式，注意numpy是(row, col)格式，需要转换
        return (max_pos[1], max_pos[0])

    def calculate_peak_distance(self, pred_peak: Tuple[int, int], gt_peak: Tuple[int, int]) -> float:
        """
        计算两个峰值位置之间的欧几里得距离

        Args:
            pred_peak: 预测峰值位置
            gt_peak: 真实峰值位置

        Returns:
            欧几里得距离
        """
        return np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    def convert_colormap_to_heatmap(self, colormap_image: Image.Image, colormap_name: str = 'jet') -> np.ndarray:
        """
        将colormap格式的图像转换为热力图数值数组

        Args:
            colormap_image: colormap格式的PIL图像
            colormap_name: 使用的colormap名称

        Returns:
            热力图数值数组 (H, W)
        """
        # 转换为numpy数组并归一化
        rgb_array = np.array(colormap_image).astype(np.float32) / 255.0

        # 从colormap提取热力图数值
        heatmap_array = extract_heatmap_from_colormap(rgb_array, colormap_name)

        return heatmap_array

    def save_heatmap_visualization(self, output_path: str, original_image: Image.Image,
                                 colormap_sequence: List[Image.Image], colormap_name: str = 'jet'):
        """
        保存热力图可视化结果，包括原始colormap和提取的热力图

        Args:
            output_path: 输出路径前缀
            original_image: 原始输入的条件图像
            colormap_sequence: colormap格式的预测序列
            colormap_name: 使用的colormap名称
        """
        import matplotlib.pyplot as plt

        num_frames = len(colormap_sequence)
        fig, axes = plt.subplots(3, num_frames + 1, figsize=(3*(num_frames + 1), 9))

        # 如果只有一帧，调整axes格式
        if num_frames == 0:
            return
        if num_frames == 1:
            axes = axes.reshape(3, -1)

        # 第一列：输入图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Condition Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        # 其余列：预测序列
        for i, colormap_img in enumerate(colormap_sequence):
            col_idx = i + 1

            # 第一行：colormap格式（模型直接输出）
            axes[0, col_idx].imshow(colormap_img)
            axes[0, col_idx].set_title(f'Colormap Frame {i}')
            axes[0, col_idx].axis('off')

            # 第二行：提取的热力图
            heatmap_array = self.convert_colormap_to_heatmap(colormap_img, colormap_name)
            im = axes[1, col_idx].imshow(heatmap_array, cmap='jet')
            axes[1, col_idx].set_title(f'Extracted Heatmap {i}')
            axes[1, col_idx].axis('off')

            # 找到峰值位置并标记
            peak_pos = self.find_peak_position(colormap_img, colormap_name)
            axes[1, col_idx].plot(peak_pos[0], peak_pos[1], 'r*', markersize=10)

            # 第三行：热力图数值统计
            max_val = np.max(heatmap_array)
            min_val = np.min(heatmap_array)
            mean_val = np.mean(heatmap_array)
            axes[2, col_idx].text(0.1, 0.8, f'Max: {max_val:.3f}', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].text(0.1, 0.6, f'Min: {min_val:.3f}', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].text(0.1, 0.4, f'Mean: {mean_val:.3f}', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].text(0.1, 0.2, f'Peak: ({peak_pos[0]}, {peak_pos[1]})', transform=axes[2, col_idx].transAxes)
            axes[2, col_idx].set_title(f'Stats Frame {i}')
            axes[2, col_idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Heatmap visualization saved to: {output_path}")


def test_on_dataset(inference_engine: HeatmapInference,
                   data_root: str,
                   wan_type: str,
                   output_dir: str = "./inference_results",
                   test_indices: List[int] = None,
                   sequence_length: int = 4):
    """
    在数据集上进行测试

    Args:
        inference_engine: 推断引擎
        data_root: 数据根目录
        wan_type: 模型类型，用于数据集准备 (e.g., "5B_TI2V_RGB_HEATMAP", "WAN_2_1_14B_I2V")
        output_dir: 输出目录
        test_indices: 要测试的样本索引列表。如果为None，则测试前10个样本
        sequence_length: 序列长度
    """
    if not DATASET_AVAILABLE:
        print("Dataset not available, skipping dataset test")
        return

    print(f"Testing on dataset: {data_root}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建测试数据集
    try:
        dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
            data_root=data_root,
            sequence_length=sequence_length,
            step_interval=1,
            min_trail_length=10,
            image_size=(256, 256),
            sigma=1.5,
            augmentation=False,  # 测试时不使用数据增强
            mode="train",  # 暂时使用训练集
            scene_bounds=[0, -0.45, -0.05, 0.8, 0.55, 0.6],
            transform_augmentation_xyz=[0.0, 0.0, 0.0],  # 测试时不增强
            transform_augmentation_rpy=[0.0, 0.0, 0.0],
            debug=False,
            colormap_name="jet",
            repeat=1,
            wan_type=wan_type  # 使用传入的wan_type参数
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 测试样本
    # 如果没有指定索引，默认测试前10个样本
    if test_indices is None:
        test_indices = list(range(min(10, len(dataset))))

    # 验证索引有效性
    valid_indices = [idx for idx in test_indices if 0 <= idx < len(dataset)]
    if len(valid_indices) < len(test_indices):
        invalid_indices = [idx for idx in test_indices if idx not in valid_indices]
        print(f"Warning: Invalid indices will be skipped: {invalid_indices}")

    print(f"Testing on {len(valid_indices)} samples with indices: {valid_indices}")

    all_distances = []

    for sample_idx, dataset_idx in enumerate(valid_indices):
        print(f"\nProcessing sample {sample_idx+1}/{len(valid_indices)} (dataset index: {dataset_idx})")

        try:
            # 获取数据样本
            sample = dataset[dataset_idx]

            # Debug: 打印sample的键
            print(f"  Sample keys: {list(sample.keys())}")

            input_image = sample['input_image']  # 热力图序列的第一帧
            input_image_rgb = sample.get('input_image_rgb', None)  # RGB图像的第一帧
            prompt = sample['prompt']
            gt_heatmap_video = sample['video']  # 真实的热力图序列
            gt_rgb_video = sample.get('input_video_rgb', None)  # 真实的RGB视频序列

            # 如果没有RGB数据，跳过
            if input_image_rgb is None or gt_rgb_video is None:
                print(f"  Warning: Sample missing RGB data, checking available keys...")
                print(f"  Available keys: {list(sample.keys())}")
                raise KeyError("Missing RGB data in sample")

            print(f"  Prompt: {prompt[:50]}...")

            # 生成预测
            pred_heatmap_video, pred_rgb_video = inference_engine.predict_heatmap_sequence(
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                prompt=prompt,
                num_frames=len(gt_heatmap_video),
                seed=42
            )

            # 计算每帧的峰值距离
            frame_distances = []
            gt_peaks = []  # 记录ground truth峰值位置
            pred_peaks = []  # 记录预测峰值位置

            # 创建可视化 - 4行 x num_frames列
            # 第1行: GT RGB视频
            # 第2行: 预测RGB视频
            # 第3行: GT热力图序列
            # 第4行: 预测热力图序列
            num_frames = len(gt_heatmap_video)
            fig, axes = plt.subplots(4, num_frames, figsize=(3*num_frames, 12))
            if num_frames == 1:
                axes = axes.reshape(-1, 1)

            for frame_idx in range(num_frames):
                gt_heatmap_frame = gt_heatmap_video[frame_idx]
                pred_heatmap_frame = pred_heatmap_video[frame_idx]
                gt_rgb_frame = gt_rgb_video[frame_idx]

                # 找到热力图峰值位置
                gt_peak = inference_engine.find_peak_position(gt_heatmap_frame)
                pred_peak = inference_engine.find_peak_position(pred_heatmap_frame)

                # 记录峰值位置
                gt_peaks.append(gt_peak)
                pred_peaks.append(pred_peak)

                # 计算距离
                distance = inference_engine.calculate_peak_distance(pred_peak, gt_peak)
                frame_distances.append(distance)

                # 第1行：GT RGB视频
                axes[0, frame_idx].imshow(gt_rgb_frame)
                axes[0, frame_idx].plot(gt_peak[0], gt_peak[1], 'r*', markersize=10)
                axes[0, frame_idx].set_title(f'GT RGB {frame_idx}')
                axes[0, frame_idx].axis('off')

                # 第2行：预测RGB视频
                if pred_rgb_video is not None and frame_idx < len(pred_rgb_video):
                    pred_rgb_frame = pred_rgb_video[frame_idx]
                    axes[1, frame_idx].imshow(pred_rgb_frame)
                    axes[1, frame_idx].plot(pred_peak[0], pred_peak[1], 'b*', markersize=10)
                    axes[1, frame_idx].set_title(f'Pred RGB {frame_idx}')
                else:
                    axes[1, frame_idx].text(0.5, 0.5, 'No RGB Output',
                                           ha='center', va='center', transform=axes[1, frame_idx].transAxes)
                    axes[1, frame_idx].set_title(f'Pred RGB {frame_idx}')
                axes[1, frame_idx].axis('off')

                # 第3行：GT热力图
                axes[2, frame_idx].imshow(gt_heatmap_frame)
                axes[2, frame_idx].plot(gt_peak[0], gt_peak[1], 'r*', markersize=10, label='GT Peak')
                axes[2, frame_idx].set_title(f'GT Heatmap {frame_idx}')
                axes[2, frame_idx].legend(fontsize=8)
                axes[2, frame_idx].axis('off')

                # 第4行：预测热力图
                axes[3, frame_idx].imshow(pred_heatmap_frame)
                axes[3, frame_idx].plot(pred_peak[0], pred_peak[1], 'b*', markersize=10, label='Pred Peak')
                axes[3, frame_idx].set_title(f'Pred Heatmap {frame_idx}\nDist: {distance:.1f}px')
                axes[3, frame_idx].legend(fontsize=8)
                axes[3, frame_idx].axis('off')

            # 添加输入图像
            fig.suptitle(f'Sample (Dataset Index {dataset_idx}): {prompt[:50]}...\nAvg Distance: {np.mean(frame_distances):.2f} pixels', fontsize=10)

            # 保存结果
            result_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_comparison.png')
            plt.tight_layout()
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 保存输入图像（热力图和RGB）
            input_heatmap_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_input_heatmap.png')
            input_image.save(input_heatmap_path)
            input_rgb_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_input_rgb.png')
            input_image_rgb.save(input_rgb_path)

            # 保存峰值位置序列到txt文件
            peaks_txt_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_peaks.txt')
            with open(peaks_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Sample (Dataset Index {dataset_idx})\n")
                f.write(f"Prompt: {prompt}\n")
                f.write("=" * 60 + "\n\n")

                f.write("Ground Truth Peak Positions:\n")
                for frame_idx, gt_peak in enumerate(gt_peaks):
                    f.write(f"  Frame {frame_idx}: ({gt_peak[0]}, {gt_peak[1]})\n")

                f.write("\nPredicted Peak Positions:\n")
                for frame_idx, pred_peak in enumerate(pred_peaks):
                    f.write(f"  Frame {frame_idx}: ({pred_peak[0]}, {pred_peak[1]})\n")

                f.write("\nPeak Distances (pixels):\n")
                for frame_idx, distance in enumerate(frame_distances):
                    f.write(f"  Frame {frame_idx}: {distance:.2f}\n")

                f.write(f"\nAverage Distance: {np.mean(frame_distances):.2f} pixels\n")

            all_distances.extend(frame_distances)
            print(f"  Avg distance for this sample: {np.mean(frame_distances):.2f} pixels")
            print(f"  Peak positions saved to: {peaks_txt_path}")

        except Exception as e:
            print(f"  Error processing sample (dataset index {dataset_idx}): {e}")
            continue

    # 计算总体统计
    if all_distances:
        avg_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Total frames evaluated: {len(all_distances)}")
        print(f"Average peak distance: {avg_distance:.2f} ± {std_distance:.2f} pixels")
        print(f"Min distance: {np.min(all_distances):.2f} pixels")
        print(f"Max distance: {np.max(all_distances):.2f} pixels")
        print(f"Results saved to: {output_dir}")

        # 保存统计结果
        stats_path = os.path.join(output_dir, 'evaluation_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"Total frames evaluated: {len(all_distances)}\n")
            f.write(f"Average peak distance: {avg_distance:.2f} ± {std_distance:.2f} pixels\n")
            f.write(f"Min distance: {np.min(all_distances):.2f} pixels\n")
            f.write(f"Max distance: {np.max(all_distances):.2f} pixels\n")
            f.write(f"\nIndividual distances:\n")
            for i, dist in enumerate(all_distances):
                f.write(f"Frame {i+1}: {dist:.2f}\n")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Heatmap Inference Script')
    parser.add_argument('--use_dual_head', action='store_true',
                       help='Use dual head mode (must match training configuration)')
    args = parser.parse_args()

    # 配置
    # LORA_CHECKPOINT = f"{ROOT_PATH}/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251003_063522/epoch-15.safetensors"
    # LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251007_181828/epoch-50.safetensors"
    LORA_CHECKPOINT="/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_freeze_all/20251011_032216/epoch-99.safetensors"
    MODEL_BASE_PATH = "/data/lpy/huggingface/Wan2.2-TI2V-5B-fused" # TODO
    # 自动检测数据集路径
    possible_data_roots = [
        "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf",
        "/data/wxn/V2W_Real/put_the_lion_on_the_top_shelf_eval"
    ]
    wan_type = "5B_TI2V_RGB_HEATMAP" # "WAN_2_1_14B_I2V"  # "WAN_2_1_14B_I2V" or "5B_TI2V"
    DATA_ROOT = None
    for path in possible_data_roots:
        if os.path.exists(path):
            DATA_ROOT = path
            break
    if DATA_ROOT is None:
        raise RuntimeError(f"Cannot find dataset in any of: {possible_data_roots}")

    print(f"Using DATA_ROOT: {DATA_ROOT}")

    OUTPUT_DIR = f"{ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/5B_TI2V_RGB_HEATMAP/debug"
    print("=== Heatmap Inference Test ===")
    print(f"Dual Head Mode: {args.use_dual_head}")

    # 创建推断引擎
    inference_engine = HeatmapInference(
        lora_checkpoint_path=LORA_CHECKPOINT,
        model_base_path=MODEL_BASE_PATH,
        device="cuda",
        torch_dtype=torch.bfloat16,
        wan_type=wan_type,
        use_dual_head=args.use_dual_head
    )


    # 数据集测试
    print("\n=== Dataset Test ===")
    # 指定要测试的样本索引列表
    # TEST_INDICES = [100, 500, 1000, 1500, 2000]  # 可以修改为任意想要测试的索引
    TEST_INDICES = [100, 200,300,400,500]  # 可以修改为任意想要测试的索引
    test_on_dataset(
        inference_engine=inference_engine,
        data_root=DATA_ROOT,
        wan_type=wan_type,  # 传递wan_type参数
        output_dir=OUTPUT_DIR,
        test_indices=TEST_INDICES,  # 使用索引列表而不是数量
        sequence_length=4
    )


if __name__ == "__main__":
    main()