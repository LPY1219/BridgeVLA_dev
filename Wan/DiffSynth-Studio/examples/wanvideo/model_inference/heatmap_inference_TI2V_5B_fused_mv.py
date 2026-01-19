"""
Heatmap Inference Script for Wan2.2 (Multi-View Version)
用于多视角热力图序列预测的推断脚本
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

# 添加项目路径 - 使用相对路径与训练文件保持一致
diffsynth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, diffsynth_path)
# sys.path.append(f"{ROOT_PATH}/Wan/single_view")

# 导入多视角pipeline
from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import WanVideoPipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.trainers.heatmap_utils import extract_heatmap_from_colormap

# 导入训练时的数据集工厂
from diffsynth.trainers.heatmap_dataset import HeatmapDatasetFactory
# from data.dataset import RobotTrajectoryDataset, ProjectionInterface
DATASET_AVAILABLE = True


class HeatmapInferenceMV:
    """多视角热力图推断类"""

    def __init__(self,
                 lora_checkpoint_path: str,
                 wan_type: str,
                 model_base_path: str = None,
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16,
                 use_dual_head: bool = False):
        """
        初始化多视角推断器

        Args:
            lora_checkpoint_path: LoRA模型检查点路径
            wan_type: 模型类型（必须是多视角版本）
            model_base_path: 基础模型路径
            device: 设备
            torch_dtype: 张量类型
            use_dual_head: 是否使用双head模式（需要与训练时一致）
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_checkpoint_path = lora_checkpoint_path
        self.use_dual_head = use_dual_head

        print(f"Loading {wan_type} multi-view pipeline (use_dual_head={use_dual_head})...")

        # 只支持多视角版本
        if wan_type == "5B_TI2V_RGB_HEATMAP_MV":
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
            raise ValueError(f"Unsupported wan_type for multi-view: {wan_type}. Use '5B_TI2V_RGB_HEATMAP_MV'")

        # 加载MV模块（需要在加载checkpoint之前初始化）
        print("Initializing multi-view modules...")
        self._initialize_mv_modules()

        # 使用新的统一加载函数：先加载base layer，再应用LoRA
        # 这确保了多视角注意力模块的LoRA变化量被正确地加到训练时的base layer上
        print(f"Loading checkpoint: {lora_checkpoint_path}")
        self.load_lora_with_base_weights(lora_checkpoint_path, alpha=1.0)

        print("Pipeline initialized successfully!")

    def _initialize_mv_modules(self):
        """
        初始化多视角模块
        与训练文件中的逻辑一致（heatmap_train_mv.py:86-97）
        """
        import torch.nn as nn
        from diffsynth.models.wan_video_dit_mv import SelfAttention

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

        print("✓ Multi-view modules initialized and moved to device")

    def load_checkpoint_weights(self, checkpoint_path: str):
        """
        加载checkpoint中的所有训练权重（包括patch_embedding、head、MV模块的base layer等）

        注意：此函数现在不仅加载patch_embedding和head，还包括多视角注意力相关的base layer权重
        这些权重需要在应用LoRA之前加载，以确保LoRA变化量被正确地加到训练时的base layer上

        关键点：checkpoint中的多视角注意力权重以 `base_layer` 命名（如 blocks.0.mvs_attn.k.base_layer.weight），
        但模型中的参数名没有 `base_layer`（如 blocks.0.mvs_attn.k.weight），需要转换。

        Args:
            checkpoint_path: checkpoint文件路径
        """
        try:
            # 加载checkpoint
            print(f"  Loading state dict from: {checkpoint_path}")
            state_dict = load_state_dict(checkpoint_path)

            # 筛选需要的权重
            patch_embedding_weights = {}
            head_weights = {}
            mv_base_layer_weights = {}  # mvs_attn 的 base_layer 权重
            mv_other_weights = {}  # 其他 MV 相关权重（projector, norm_mvs, modulation_mvs）

            for key, value in state_dict.items():
                # 跳过LoRA相关的权重（但不跳过 base_layer）
                if 'lora' in key.lower():
                    continue

                # 筛选patch_embedding相关的权重
                if 'patch_embedding' in key or 'patch_embed' in key:
                    patch_embedding_weights[key] = value

                # 筛选head相关的权重（包括dual head）
                elif any(pattern in key for pattern in ['head']):
                    if 'attention' not in key.lower() and 'attn' not in key.lower():
                        head_weights[key] = value

                # 筛选MV模块的base_layer权重（需要转换键名）
                elif 'base_layer' in key:
                    # 这些是 mvs_attn 中经过 LoRA 训练的层的 base layer
                    mv_base_layer_weights[key] = value

                # 筛选MV模块的其他权重（projector, norm_mvs, modulation_mvs, mvs_attn中的norm等，不需要转换键名）
                elif any(pattern in key for pattern in ['projector', 'norm_mvs', 'modulation_mvs', 'mvs_attn']):
                    mv_other_weights[key] = value

            print(f"  Found {len(patch_embedding_weights)} patch_embedding weights")
            print(f"  Found {len(head_weights)} head weights")
            print(f"  Found {len(mv_base_layer_weights)} MV module base_layer weights (need key conversion)")
            print(f"  Found {len(mv_other_weights)} MV module other weights")

            # 显示找到的权重key样例
            if patch_embedding_weights:
                print("  Patch embedding keys (sample):")
                for key in list(patch_embedding_weights.keys())[:3]:
                    print(f"    - {key}")

            if head_weights:
                print("  Head keys (sample):")
                for key in list(head_weights.keys())[:3]:
                    print(f"    - {key}")

            if mv_base_layer_weights:
                print("  MV base_layer keys (sample, before conversion):")
                for key in list(mv_base_layer_weights.keys())[:5]:
                    print(f"    - {key}")

            if mv_other_weights:
                print("  MV other weights keys (sample):")
                for key in list(mv_other_weights.keys())[:5]:
                    print(f"    - {key}")

            # 合并要加载的权重
            weights_to_load = {}
            weights_to_load.update(patch_embedding_weights)
            weights_to_load.update(head_weights)
            weights_to_load.update(mv_other_weights)

            # 转换 base_layer 键名：blocks.X.mvs_attn.Y.base_layer.weight -> blocks.X.mvs_attn.Y.weight
            for key, value in mv_base_layer_weights.items():
                # 移除 .base_layer
                converted_key = key.replace('.base_layer.', '.')
                weights_to_load[converted_key] = value

            if not weights_to_load:
                print("  Warning: No weights found in checkpoint")
                return

            # 显示转换后的键名样例
            if mv_base_layer_weights:
                print("  MV base_layer keys (sample, after conversion):")
                converted_samples = [k.replace('.base_layer.', '.') for k in list(mv_base_layer_weights.keys())[:5]]
                for key in converted_samples:
                    print(f"    - {key}")

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
                relevant_missing = [k for k in missing_keys if any(p in k for p in ['patch_embedding', 'head', 'projector', 'mvs_'])]
                if relevant_missing:
                    print(f"    Warning: {len(relevant_missing)} relevant keys not found in checkpoint:")
                    for key in relevant_missing[:5]:
                        print(f"      - {key}")

            if unexpected_keys:
                print(f"    Info: {len(unexpected_keys)} unexpected keys (sample):")
                for key in unexpected_keys[:5]:
                    print(f"      - {key}")

            print("  ✓ All base layer weights loaded successfully!")

        except Exception as e:
            print(f"  Warning: Failed to load weights: {e}")
            print("  Continuing with LoRA weights only...")
            import traceback
            traceback.print_exc()

    def load_lora_with_base_weights(self, checkpoint_path: str, alpha: float = 1.0):
        """
        为多视角视频扩散模型定制的LoRA加载函数

        关键区别：多视角注意力部分的LoRA base layer已经存储在checkpoint中，
        因此需要先加载这些base layer，然后再应用LoRA变化量。

        加载流程：
        1. 先加载checkpoint中所有非LoRA的权重（base layer）
        2. 然后加载LoRA权重，计算变化量并应用到对应的base layer上

        这确保了LoRA变化量被正确地加到训练时的base layer上，
        而不是加到推理时随机初始化的权重上。

        Args:
            checkpoint_path: checkpoint文件路径
            alpha: LoRA的缩放因子
        """
        print("Loading checkpoint with custom LoRA logic for multiview model...")

        # 步骤1：先加载所有base layer权重（包括多视角注意力模块）
        print("\nStep 1: Loading base layer weights from checkpoint...")
        self.load_checkpoint_weights(checkpoint_path)

        # 步骤2：加载LoRA并应用到base layer上
        print("\nStep 2: Loading and applying LoRA weights...")
        self.pipe.load_lora(self.pipe.dit, checkpoint_path, alpha=alpha)

        print("\n✓ Checkpoint loaded successfully with multiview-aware LoRA logic!")

    # 保留旧函数名作为别名，以兼容旧代码
    def load_patch_embedding_and_head_weights(self, checkpoint_path: str):
        """
        [已弃用] 请使用 load_checkpoint_weights 替代

        此函数保留以兼容旧代码，但建议使用新的 load_checkpoint_weights 函数名，
        因为它更准确地反映了函数的功能（加载的不仅仅是patch_embedding和head）。
        """
        import warnings
        warnings.warn(
            "load_patch_embedding_and_head_weights is deprecated. "
            "Use load_checkpoint_weights instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.load_checkpoint_weights(checkpoint_path)

    def predict_heatmap_sequence(self,
                                input_image: List[Image.Image],
                                prompt: str,
                                input_image_rgb: List[Image.Image],
                                num_frames: int = 5,
                                height: int = 256,
                                width: int = 256,
                                num_view: int = 3,
                                seed: int = None) -> Tuple[List[List[Image.Image]], List[List[Image.Image]]]:
        """
        预测多视角热力图序列和RGB视频

        Args:
            input_image: 多个视角的第一帧热力图列表 [view1, view2, ...]
            prompt: 语言指令
            input_image_rgb: 多个视角的第一帧RGB图像列表 [view1, view2, ...]
            num_frames: 预测帧数
            height: 输出高度
            width: 输出宽度
            num_view: 视角数量
            seed: 随机种子

        Returns:
            (预测的热力图序列 [time][view], 预测的RGB视频序列 [time][view])
        """
        print(f"  Predicting multi-view heatmap and RGB sequence...")
        print(f"  Number of views: {len(input_image)}")
        print(f"  Input heatmap sizes: {[img.size for img in input_image]}")
        print(f"  Input RGB sizes: {[img.size for img in input_image_rgb]}")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {num_frames} frames × {num_view} views of {width}x{height}")

        # 调整输入图像尺寸（每个视角）
        input_image_resized = [img.resize((width, height)) for img in input_image]
        input_image_rgb_resized = [img.resize((width, height)) for img in input_image_rgb]

        # 生成视频序列 - pipeline应该返回([time][view], [time][view])格式
        # num_view 会从 input_image 的长度自动推断
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

        # 检查返回值格式并转换 [view][time] -> [time][view]
        if isinstance(result, tuple) and len(result) == 2:
            heatmap_views, rgb_views = result
            # 转换格式: [view][time] -> [time][view]
            # heatmap_views = [view1_frames, view2_frames, view3_frames]
            # 转换为 [[v1_f0, v2_f0, v3_f0], [v1_f1, v2_f1, v3_f1], ...]
            num_frames_actual = len(heatmap_views[0]) if heatmap_views and len(heatmap_views) > 0 else 0
            heatmap_frames = [[heatmap_views[v][t] for v in range(len(heatmap_views))]
                             for t in range(num_frames_actual)]
            rgb_frames = [[rgb_views[v][t] for v in range(len(rgb_views))]
                         for t in range(num_frames_actual)]
            print(f"Generated {len(heatmap_frames)} frames × {len(heatmap_frames[0]) if heatmap_frames else 0} views (heatmap)")
            print(f"Generated {len(rgb_frames)} frames × {len(rgb_frames[0]) if rgb_frames else 0} views (RGB)")
        else:
            # 如果只返回热力图，RGB设为None
            heatmap_views = result
            num_frames_actual = len(heatmap_views[0]) if heatmap_views and len(heatmap_views) > 0 else 0
            heatmap_frames = [[heatmap_views[v][t] for v in range(len(heatmap_views))]
                             for t in range(num_frames_actual)]
            rgb_frames = None
            print(f"Generated {len(heatmap_frames)} frames × {len(heatmap_frames[0]) if heatmap_frames else 0} views (heatmap)")

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


def test_on_dataset_mv(inference_engine: HeatmapInferenceMV,
                      data_root: str,
                      wan_type: str,
                      output_dir: str = "./inference_results_mv",
                      test_indices: List[int] = None,
                      sequence_length: int = 4):
    """
    在多视角数据集上进行测试

    Args:
        inference_engine: 多视角推断引擎
        data_root: 数据根目录
        wan_type: 模型类型
        output_dir: 输出目录
        test_indices: 要测试的样本索引列表
        sequence_length: 序列长度
    """
    if not DATASET_AVAILABLE:
        print("Dataset not available, skipping dataset test")
        return

    print(f"Testing on multi-view dataset: {data_root}")

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
            mode="train",
            scene_bounds=[0, -0.45, -0.05, 0.8, 0.55, 0.6],
            transform_augmentation_xyz=[0.0, 0.0, 0.0],
            transform_augmentation_rpy=[0.0, 0.0, 0.0],
            debug=False,
            colormap_name="jet",
            repeat=1,
            wan_type=wan_type
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 测试样本
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

            # 多视角数据格式：
            # input_image: [view1, view2, view3]
            # input_video: [time][view]
            input_image = sample['input_image']  # 列表：多个视角的第一帧热力图
            input_image_rgb = sample.get('input_image_rgb', None)  # 列表：多个视角的第一帧RGB
            prompt = sample['prompt']
            gt_heatmap_video = sample['video']  # [time][view]
            gt_rgb_video = sample.get('input_video_rgb', None)  # [time][view]

            if input_image_rgb is None or gt_rgb_video is None:
                print(f"  Warning: Sample missing RGB data")
                raise KeyError("Missing RGB data in sample")

            num_view = len(input_image)
            num_frames = len(gt_heatmap_video)

            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Num views: {num_view}, Num frames: {num_frames}")

            # 生成预测
            pred_heatmap_video, pred_rgb_video = inference_engine.predict_heatmap_sequence(
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                prompt=prompt,
                num_frames=num_frames,
                num_view=num_view,
                seed=42
            )

            # 创建多视角可视化
            # 新布局：时间维度沿横轴，每列是一个时间步
            # 总共4个section（每个section有num_view行）：
            # 1. GT RGB (num_view 行)
            # 2. Pred RGB (num_view 行)
            # 3. GT Heatmap (num_view 行)
            # 4. Pred Heatmap (num_view 行)
            # 横轴：num_frames 列

            fig = plt.figure(figsize=(3*num_frames, 3*num_view*4))
            gs = fig.add_gridspec(num_view*4, num_frames, hspace=0.3, wspace=0.1)

            frame_distances = []  # 所有视角所有帧的距离
            # 存储峰值位置信息，用于后续保存到txt文件
            peak_positions = {
                'gt': {},  # {(view_idx, frame_idx): (x, y)}
                'pred': {}  # {(view_idx, frame_idx): (x, y)}
            }

            for view_idx in range(num_view):
                # 每个section的起始行（每个视角一行）
                gt_rgb_row = view_idx
                pred_rgb_row = num_view + view_idx
                gt_heatmap_row = num_view*2 + view_idx
                pred_heatmap_row = num_view*3 + view_idx

                for frame_idx in range(num_frames):
                    # 获取当前帧当前视角的图像
                    gt_heatmap_frame = gt_heatmap_video[frame_idx][view_idx]
                    pred_heatmap_frame = pred_heatmap_video[frame_idx][view_idx]
                    gt_rgb_frame = gt_rgb_video[frame_idx][view_idx]

                    # 找到热力图峰值位置
                    gt_peak = inference_engine.find_peak_position(gt_heatmap_frame)
                    pred_peak = inference_engine.find_peak_position(pred_heatmap_frame)

                    # 保存峰值位置
                    peak_positions['gt'][(view_idx, frame_idx)] = gt_peak
                    peak_positions['pred'][(view_idx, frame_idx)] = pred_peak

                    # 计算距离
                    distance = inference_engine.calculate_peak_distance(pred_peak, gt_peak)
                    frame_distances.append(distance)

                    # 第1部分：GT RGB
                    ax = fig.add_subplot(gs[gt_rgb_row, frame_idx])
                    ax.imshow(gt_rgb_frame)
                    ax.plot(gt_peak[0], gt_peak[1], 'r*', markersize=8)
                    if frame_idx == 0:
                        ax.set_ylabel(f'GT RGB V{view_idx}', fontsize=9, fontweight='bold')
                    if view_idx == 0:
                        ax.set_title(f'T{frame_idx}', fontsize=8)
                    ax.axis('off')

                    # 第2部分：Pred RGB
                    ax = fig.add_subplot(gs[pred_rgb_row, frame_idx])
                    if pred_rgb_video is not None and frame_idx < len(pred_rgb_video):
                        pred_rgb_frame = pred_rgb_video[frame_idx][view_idx]
                        ax.imshow(pred_rgb_frame)
                        ax.plot(pred_peak[0], pred_peak[1], 'b*', markersize=8)
                    else:
                        ax.text(0.5, 0.5, 'No RGB', ha='center', va='center', transform=ax.transAxes)
                    if frame_idx == 0:
                        ax.set_ylabel(f'Pred RGB V{view_idx}', fontsize=9, fontweight='bold')
                    ax.axis('off')

                    # 第3部分：GT Heatmap
                    ax = fig.add_subplot(gs[gt_heatmap_row, frame_idx])
                    ax.imshow(gt_heatmap_frame)
                    ax.plot(gt_peak[0], gt_peak[1], 'r*', markersize=8)
                    if frame_idx == 0:
                        ax.set_ylabel(f'GT Heatmap V{view_idx}', fontsize=9, fontweight='bold')
                    ax.axis('off')

                    # 第4部分：Pred Heatmap
                    ax = fig.add_subplot(gs[pred_heatmap_row, frame_idx])
                    ax.imshow(pred_heatmap_frame)
                    ax.plot(pred_peak[0], pred_peak[1], 'b*', markersize=8)
                    if frame_idx == 0:
                        ax.set_ylabel(f'Pred Heatmap V{view_idx}', fontsize=9, fontweight='bold')
                    # 在每个子图上显示距离
                    ax.text(0.5, -0.1, f'D:{distance:.1f}', ha='center', va='top',
                           transform=ax.transAxes, fontsize=7)
                    ax.axis('off')

            # 添加总标题
            avg_distance = np.mean(frame_distances)
            fig.suptitle(f'Multi-View Sample (Index {dataset_idx})\n{prompt[:60]}...\nAvg Distance: {avg_distance:.2f} pixels',
                        fontsize=11, fontweight='bold')

            # 保存结果
            result_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_multiview_comparison.png')
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 保存统计信息（包括像素位置）
            stats_path = os.path.join(output_dir, f'sample_idx{dataset_idx:03d}_stats.txt')
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"Multi-View Sample (Dataset Index {dataset_idx})\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Num Views: {num_view}, Num Frames: {num_frames}\n")
                f.write("=" * 80 + "\n\n")

                # 按视角和帧组织统计
                for view_idx in range(num_view):
                    f.write(f"View {view_idx}:\n")
                    f.write("-" * 80 + "\n")
                    for frame_idx in range(num_frames):
                        # 计算当前的索引（由于遍历顺序是view->frame）
                        idx = view_idx * num_frames + frame_idx

                        if idx < len(frame_distances):
                            gt_pos = peak_positions['gt'][(view_idx, frame_idx)]
                            pred_pos = peak_positions['pred'][(view_idx, frame_idx)]
                            dist = frame_distances[idx]

                            f.write(f"  Frame {frame_idx}:\n")
                            f.write(f"    GT Position:   (x={gt_pos[0]:3d}, y={gt_pos[1]:3d})\n")
                            f.write(f"    Pred Position: (x={pred_pos[0]:3d}, y={pred_pos[1]:3d})\n")
                            f.write(f"    Distance: {dist:.2f} pixels\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write(f"Average Distance (all views, all frames): {avg_distance:.2f} pixels\n")

            all_distances.extend(frame_distances)
            print(f"  Avg distance for this sample: {avg_distance:.2f} pixels")
            print(f"  Results saved to: {result_path}")

        except Exception as e:
            print(f"  Error processing sample (dataset index {dataset_idx}): {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算总体统计
    if all_distances:
        avg_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        print(f"\n=== MULTI-VIEW EVALUATION RESULTS ===")
        print(f"Total predictions evaluated: {len(all_distances)} (views × frames)")
        print(f"Average peak distance: {avg_distance:.2f} ± {std_distance:.2f} pixels")
        print(f"Min distance: {np.min(all_distances):.2f} pixels")
        print(f"Max distance: {np.max(all_distances):.2f} pixels")
        print(f"Results saved to: {output_dir}")

        # 保存总体统计
        stats_path = os.path.join(output_dir, 'evaluation_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Multi-View Evaluation Results\n")
            f.write(f"==============================\n")
            f.write(f"Total predictions evaluated: {len(all_distances)}\n")
            f.write(f"Average peak distance: {avg_distance:.2f} ± {std_distance:.2f} pixels\n")
            f.write(f"Min distance: {np.min(all_distances):.2f} pixels\n")
            f.write(f"Max distance: {np.max(all_distances):.2f} pixels\n")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Multi-View Heatmap Inference Script')
    parser.add_argument('--use_dual_head', action='store_true',
                       help='Use dual head mode (must match training configuration)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--test_indices', type=str, default="100,200,300,400,500",
                       help='Comma-separated test indices')
    args = parser.parse_args()

    # 配置
    if args.checkpoint:
        LORA_CHECKPOINT = args.checkpoint
    else:
        # 默认checkpoint路径
        LORA_CHECKPOINT = "/home/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/20251017_170901/epoch-56.safetensors"

    MODEL_BASE_PATH = "/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"

    # 自动检测数据集路径
    possible_data_roots = [
        "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf",
        "/data/wxn/V2W_Real/put_the_lion_on_the_top_shelf_eval"
    ]

    wan_type = "5B_TI2V_RGB_HEATMAP_MV"  # 多视角版本

    DATA_ROOT = None
    for path in possible_data_roots:
        if os.path.exists(path):
            DATA_ROOT = path
            break
    if DATA_ROOT is None:
        raise RuntimeError(f"Cannot find dataset in any of: {possible_data_roots}")

    print(f"Using DATA_ROOT: {DATA_ROOT}")

    OUTPUT_DIR = f"{ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/5B_TI2V_MV"

    print("=== Multi-View Heatmap Inference Test ===")
    print(f"Checkpoint: {LORA_CHECKPOINT}")
    print(f"Dual Head Mode: {args.use_dual_head}")
    print(f"Model Type: {wan_type}")

    # 创建推断引擎
    inference_engine = HeatmapInferenceMV(
        lora_checkpoint_path=LORA_CHECKPOINT,
        model_base_path=MODEL_BASE_PATH,
        device="cuda",
        torch_dtype=torch.bfloat16,
        wan_type=wan_type,
        use_dual_head=args.use_dual_head
    )

    # 数据集测试
    print("\n=== Multi-View Dataset Test ===")

    # 解析测试索引
    TEST_INDICES = [int(x.strip()) for x in args.test_indices.split(',')]
    print(f"Test indices: {TEST_INDICES}")

    test_on_dataset_mv(
        inference_engine=inference_engine,
        data_root=DATA_ROOT,
        wan_type=wan_type,
        output_dir=OUTPUT_DIR,
        test_indices=TEST_INDICES,
        sequence_length=4
    )


if __name__ == "__main__":
    main()
