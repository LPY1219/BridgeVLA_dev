"""
基于Wan2.2的heatmap序列生成模型
利用Wan2.2的video generation能力，通过colormap编码方式处理heatmap数据
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import os

# 环境变量设置，避免flash attention兼容性问题
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["DISABLE_FLASH_ATTN"] = "1"

try:
    from diffusers import AutoencoderKLWan, DDIMScheduler
except ImportError as e:
    print(f"ImportError: {e}")
    print("Trying to import without flash attention...")
    import sys

    # 模拟flash_attn模块以绕过导入错误
    class MockFlashAttn:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    sys.modules['flash_attn'] = MockFlashAttn()
    sys.modules['flash_attn.flash_attn_interface'] = MockFlashAttn()

    from diffusers import AutoencoderKLWan, DDIMScheduler

try:
    from ..utils.colormap_utils import (
        convert_heatmap_sequence_to_colormap,
        extract_heatmap_sequence_from_colormap,
        convert_colormap_video_to_wan_format,
        convert_from_wan_format,
        load_vae_model
    )
    from ..utils.heatmap_utils import normalize_heatmap
    from ..configs.model_config import ModelConfig
except ImportError:
    # 绝对导入用于直接运行时
    from utils.colormap_utils import (
        convert_heatmap_sequence_to_colormap,
        extract_heatmap_sequence_from_colormap,
        convert_colormap_video_to_wan_format,
        convert_from_wan_format,
        load_vae_model
    )
    from utils.heatmap_utils import normalize_heatmap
    from configs.model_config import ModelConfig


class WanHeatmapModel(nn.Module):
    """
    基于Wan2.2的heatmap序列生成模型

    该模型使用以下策略：
    1. 将RGB图像作为条件输入
    2. 将heatmap序列转换为colormap视频格式
    3. 使用Wan2.2的VAE进行编码和重建
    4. 将重建的colormap视频转回heatmap序列
    """

    def __init__(self, config: ModelConfig):
        """
        初始化Wan heatmap模型

        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.device = config.device
        self.torch_dtype = config.get_torch_dtype()

        # 加载Wan2.2 VAE模型
        self.vae, self.vae_device = self._load_wan_vae()

        # 创建噪声调度器（用于推理）
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # 冻结VAE参数
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE parameters frozen for training")

        # 加载Wan2.2 pipeline进行微调
        self.wan_pipeline = self._load_wan_pipeline()

        # 设置可训练组件
        self._setup_finetuning_components()

        # 模型状态
        self.is_training = True

    def _load_wan_vae(self) -> Tuple[AutoencoderKLWan, str]:
        """
        加载Wan2.2 VAE模型

        Returns:
            (vae_model, device)
        """
        print(f"Loading Wan2.2 VAE from: {self.config.wan_model_path}")

        vae = AutoencoderKLWan.from_pretrained(
            self.config.wan_model_path,
            subfolder=self.config.vae_subfolder,
            torch_dtype=self.torch_dtype
        )

        # 移动到指定设备
        device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device)

        # 设置为评估模式
        vae.eval()

        # 启用内存优化
        if self.config.enable_attention_slicing:
            try:
                vae.enable_attention_slicing()
            except:
                print("Warning: Attention slicing not supported")

        if self.config.enable_cpu_offload:
            try:
                vae.enable_sequential_cpu_offload()
            except:
                print("Warning: CPU offload not supported")

        print(f"VAE loaded successfully on device: {device}")
        return vae, device

    def _load_wan_pipeline(self):
        """
        加载Wan2.2 pipeline用于微调，包含内存优化策略
        """
        try:
            from diffusers import WanPipeline
            import gc

            print(f"Loading Wan2.2 pipeline from: {self.config.wan_model_path}")

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            # 使用内存优化选项加载pipeline
            pipeline = WanPipeline.from_pretrained(
                self.config.wan_model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,  # 启用低CPU内存使用
                variant="fp16" if self.torch_dtype == torch.float16 else None,
            )

            # 启用内存优化
            if hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                print("Enabled model CPU offload")

            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
                print("Enabled attention slicing")

            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
                print("Enabled VAE slicing")

            # 移动到设备
            pipeline = pipeline.to(self.device)

            if torch.cuda.is_available():
                print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            print("Wan2.2 pipeline loaded successfully with memory optimization")
            return pipeline

        except Exception as e:
            print(f"Error loading Wan2.2 pipeline: {e}")
            print("Attempting to free memory and retry...")

            # 清理内存后重试
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            try:
                # 第二次尝试，使用更激进的内存优化
                pipeline = WanPipeline.from_pretrained(
                    self.config.wan_model_path,
                    torch_dtype=torch.float16,  # 强制使用fp16
                    low_cpu_mem_usage=True,
                    device_map="auto",  # 自动设备映射
                )

                print("Wan2.2 pipeline loaded successfully with aggressive optimization")
                return pipeline

            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                print("Fallback: creating simple trainable layers")

            # 简单的替代方案：创建可训练的适应层
            adaptation_layers = nn.ModuleDict({
                'rgb_encoder': nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 512)
                ),
                'latent_predictor': nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 48 * self.config.sequence_length * 4 * 4)
                )
            }).to(self.device)

            return adaptation_layers

    def _setup_finetuning_components(self):
        """
        设置微调组件：冻结VAE和文本编码器，只训练transformer backbone
        """
        if self.wan_pipeline is None:
            print("No pipeline to configure")
            return

        trainable_params = 0
        total_params = 0

        # 统计总参数
        for param in self.wan_pipeline.parameters():
            total_params += param.numel()

        # 如果是完整pipeline，冻结某些组件
        if hasattr(self.wan_pipeline, 'transformer'):
            # 冻结VAE
            if hasattr(self.wan_pipeline, 'vae'):
                for param in self.wan_pipeline.vae.parameters():
                    param.requires_grad = False
                print("Pipeline VAE frozen")

            # 冻结文本编码器（如果存在）
            if hasattr(self.wan_pipeline, 'text_encoder'):
                for param in self.wan_pipeline.text_encoder.parameters():
                    param.requires_grad = False
                print("Text encoder frozen")

            # 冻结scheduler相关组件
            if hasattr(self.wan_pipeline, 'scheduler'):
                if hasattr(self.wan_pipeline.scheduler, 'parameters'):
                    for param in self.wan_pipeline.scheduler.parameters():
                        param.requires_grad = False

            # 保持transformer可训练（这是diffusion backbone）
            for param in self.wan_pipeline.transformer.parameters():
                param.requires_grad = True
                trainable_params += param.numel()

            print(f"Transformer (diffusion backbone) parameters: {trainable_params:,}")

        else:
            # 如果是简单适应层，全部可训练
            for param in self.wan_pipeline.parameters():
                param.requires_grad = True
                trainable_params += param.numel()

        frozen_params = total_params - trainable_params
        trainable_ratio = trainable_params / total_params * 100

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.1f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({100-trainable_ratio:.1f}%)")

    def encode_heatmaps_to_latent(self, heatmap_sequence: torch.Tensor) -> torch.Tensor:
        """
        将heatmap序列编码到潜在空间

        Args:
            heatmap_sequence: heatmap序列 (B, T, H, W)

        Returns:
            潜在表示 (B, C, T, H_latent, W_latent)
        """
        batch_size, seq_len, height, width = heatmap_sequence.shape

        # 转换为numpy以使用colormap工具
        if heatmap_sequence.is_cuda:
            heatmap_np = heatmap_sequence.cpu().numpy()
        else:
            heatmap_np = heatmap_sequence.numpy()

        # 逐批次处理
        latents = []
        for b in range(batch_size):
            # 转换heatmap序列为colormap视频
            colormap_video = convert_heatmap_sequence_to_colormap(
                heatmap_np[b], self.config.colormap_name
            )  # (T, H, W, 3)

            # 转换为Wan格式
            colormap_tensor = convert_colormap_video_to_wan_format(colormap_video)  # (1, 3, T, H, W)
            colormap_tensor = colormap_tensor.to(self.vae_device, dtype=self.torch_dtype)

            # VAE编码
            with torch.no_grad():
                latent_dist = self.vae.encode(colormap_tensor)
                latent = latent_dist.latent_dist.sample()  # (1, C, T, H_latent, W_latent)

            latents.append(latent)

        # 合并批次
        latents = torch.cat(latents, dim=0)  # (B, C, T, H_latent, W_latent)
        return latents

    def decode_latent_to_heatmaps(self, latents: torch.Tensor) -> torch.Tensor:
        """
        将潜在表示解码为heatmap序列

        Args:
            latents: 潜在表示 (B, C, T, H_latent, W_latent)

        Returns:
            heatmap序列 (B, T, H, W)
        """
        batch_size = latents.shape[0]

        # 逐批次解码
        heatmap_sequences = []
        for b in range(batch_size):
            latent_single = latents[b:b+1]  # (1, C, T, H_latent, W_latent)

            # VAE解码
            with torch.no_grad():
                decoded = self.vae.decode(latent_single).sample  # (1, 3, T, H, W)

            # 转换回numpy格式
            decoded_np = convert_from_wan_format(decoded)  # (T, H, W, 3)

            # 从colormap提取heatmap
            heatmap_sequence = extract_heatmap_sequence_from_colormap(
                decoded_np, self.config.colormap_name
            )  # (T, H, W)

            heatmap_sequences.append(torch.from_numpy(heatmap_sequence))

        # 合并批次
        heatmap_sequences = torch.stack(heatmap_sequences, dim=0)  # (B, T, H, W)

        # 确保返回的张量在正确的设备上，且维度正确
        heatmap_sequences = heatmap_sequences.to(self.device)

        # 如果时间维度不正确，则重复到正确的长度
        expected_seq_len = self.config.sequence_length
        current_seq_len = heatmap_sequences.shape[1]

        if current_seq_len != expected_seq_len:
            if current_seq_len == 1:
                # 如果只有1帧，重复到所需长度
                heatmap_sequences = heatmap_sequences.repeat(1, expected_seq_len, 1, 1)
            else:
                # 如果长度不匹配，使用简单的重复或截断
                if current_seq_len < expected_seq_len:
                    # 如果序列太短，重复最后一帧
                    last_frame = heatmap_sequences[:, -1:, :, :]  # (B, 1, H, W)
                    repeat_times = expected_seq_len - current_seq_len
                    repeated_frames = last_frame.repeat(1, repeat_times, 1, 1)
                    heatmap_sequences = torch.cat([heatmap_sequences, repeated_frames], dim=1)
                else:
                    # 如果序列太长，截断
                    heatmap_sequences = heatmap_sequences[:, :expected_seq_len, :, :]

        return heatmap_sequences

    def _train_heatmap_prediction(self, rgb_images: torch.Tensor,
                                heatmap_sequences: torch.Tensor,
                                instruction_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        训练模式：基于RGB图像预测heatmap序列
        """
        # 标准化heatmap
        normalized_heatmaps = normalize_heatmap(heatmap_sequences, method='minmax')
        normalized_heatmaps = normalized_heatmaps.to(self.device)

        # 编码目标heatmap到潜在空间
        target_latents = self.encode_heatmaps_to_latent(normalized_heatmaps)

        if hasattr(self.wan_pipeline, 'transformer'):
            # 使用Wan2.2 transformer进行预测
            # 这里需要实现具体的transformer调用逻辑
            # 目前先用简单的重建任务作为占位符
            predicted_latents = target_latents  # 占位符
            reconstruction_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # 使用简单适应层进行预测
            rgb_features = self.wan_pipeline['rgb_encoder'](rgb_images)
            predicted_latents_flat = self.wan_pipeline['latent_predictor'](rgb_features)

            # 重塑为潜在空间格式
            batch_size = rgb_images.shape[0]
            predicted_latents = predicted_latents_flat.view(
                batch_size, 48, self.config.sequence_length, 4, 4
            )

            # 计算潜在空间损失
            reconstruction_loss = nn.MSELoss()(predicted_latents, target_latents)

        # 解码回heatmap用于可视化
        reconstructed_heatmaps = self.decode_latent_to_heatmaps(predicted_latents)

        return {
            'predictions': reconstructed_heatmaps,
            'latents': predicted_latents,
            'target_latents': target_latents,
            'reconstruction_loss': reconstruction_loss,
            'total_loss': reconstruction_loss
        }

    def forward(self, rgb_images: torch.Tensor,
                heatmap_sequences: Optional[torch.Tensor] = None,
                instruction_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            rgb_images: 输入RGB图像 (B, 3, H, W)
            heatmap_sequences: 目标heatmap序列 (B, T, H, W)，训练时提供
            instruction_embeddings: 指令嵌入 (B, D)，可选

        Returns:
            包含预测结果和损失的字典
        """
        if self.training and heatmap_sequences is None:
            raise ValueError("heatmap_sequences is required during training")

        # 确保输入张量在正确的设备上
        rgb_images = rgb_images.to(self.device)
        if heatmap_sequences is not None:
            heatmap_sequences = heatmap_sequences.to(self.device)

        batch_size = rgb_images.shape[0]

        if self.training:
            # 训练模式：基于RGB图像预测heatmap序列
            return self._train_heatmap_prediction(rgb_images, heatmap_sequences, instruction_embeddings)

        else:
            # 推理模式：基于RGB图像生成heatmap序列
            return self.generate_heatmap_sequence(rgb_images, instruction_embeddings)

    def generate_heatmap_sequence(self,
                                rgb_images: torch.Tensor,
                                instruction_embeddings: Optional[torch.Tensor] = None,
                                num_inference_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        基于RGB图像生成heatmap序列

        Args:
            rgb_images: 输入RGB图像 (B, 3, H, W)
            instruction_embeddings: 指令嵌入 (B, D)
            num_inference_steps: 推理步数

        Returns:
            生成结果字典
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        batch_size = rgb_images.shape[0]

        # 设置调度器
        self.scheduler.set_timesteps(num_inference_steps)

        # 获取潜在空间尺寸
        # 注意：这里我们需要根据实际的VAE配置来确定潜在空间尺寸
        latent_height = self.config.output_image_size[0] // 8  # 假设VAE下采样倍数为8
        latent_width = self.config.output_image_size[1] // 8
        latent_channels = 48  # Wan2.2 VAE的潜在空间是48通道

        # 初始化随机噪声
        shape = (batch_size, latent_channels, self.config.sequence_length, latent_height, latent_width)
        latents = torch.randn(shape, device=self.device, dtype=self.torch_dtype)

        # TODO: 这里需要实现完整的扩散过程
        # 由于Wan2.2的完整diffusion pipeline比较复杂，这里先用VAE重建作为占位符
        # 在实际实现中，需要：
        # 1. 将RGB图像编码为条件
        # 2. 使用diffusion模型进行去噪
        # 3. 结合指令嵌入（如果有）

        # 临时实现：使用简单的噪声生成
        with torch.no_grad():
            # 直接解码随机潜在表示
            generated_heatmaps = self.decode_latent_to_heatmaps(latents)

        return {
            'predictions': generated_heatmaps,
            'latents': latents
        }

    def compute_loss(self, predictions: torch.Tensor,
                    targets: torch.Tensor,
                    loss_config: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            predictions: 预测结果 (B, T, H, W)
            targets: 目标值 (B, T, H, W)
            loss_config: 损失配置

        Returns:
            损失字典
        """
        try:
            from ..utils.heatmap_utils import calculate_peak_accuracy, calculate_sequence_consistency
        except ImportError:
            from utils.heatmap_utils import calculate_peak_accuracy, calculate_sequence_consistency

        losses = {}

        # 主要重建损失
        mse_loss = nn.MSELoss()(predictions, targets)
        losses['mse_loss'] = mse_loss

        # 峰值损失（可选）
        if loss_config and hasattr(loss_config, 'peak_loss_weight') and loss_config.peak_loss_weight > 0:
            # TODO: 实现峰值损失
            peak_loss = torch.tensor(0.0, device=predictions.device)
            losses['peak_loss'] = peak_loss

        # 一致性损失（可选）
        if loss_config and hasattr(loss_config, 'consistency_loss_weight') and loss_config.consistency_loss_weight > 0:
            # TODO: 实现一致性损失
            consistency_loss = torch.tensor(0.0, device=predictions.device)
            losses['consistency_loss'] = consistency_loss

        # 总损失
        total_loss = mse_loss
        if 'peak_loss' in losses:
            total_loss += loss_config.peak_loss_weight * losses['peak_loss']
        if 'consistency_loss' in losses:
            total_loss += loss_config.consistency_loss_weight * losses['consistency_loss']

        losses['total_loss'] = total_loss

        return losses

    def save_pretrained(self, save_directory: str):
        """
        保存模型

        Args:
            save_directory: 保存目录
        """
        os.makedirs(save_directory, exist_ok=True)

        # 保存模型配置
        config_path = os.path.join(save_directory, "model_config.json")
        self.config.save(config_path)

        # 保存模型状态（如果有可训练参数）
        state_dict_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), state_dict_path)

        print(f"Model saved to: {save_directory}")

    @classmethod
    def from_pretrained(cls, model_directory: str, config: Optional[ModelConfig] = None):
        """
        从保存的目录加载模型

        Args:
            model_directory: 模型目录
            config: 可选的配置，如果不提供则从目录加载

        Returns:
            WanHeatmapModel实例
        """
        if config is None:
            config_path = os.path.join(model_directory, "model_config.json")
            config = ModelConfig.load(config_path)

        model = cls(config)

        # 加载模型状态（如果存在）
        state_dict_path = os.path.join(model_directory, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=config.device)
            model.load_state_dict(state_dict)

        print(f"Model loaded from: {model_directory}")
        return model

    def enable_attention_slicing(self):
        """启用注意力切片以节省内存"""
        if hasattr(self.vae, 'enable_attention_slicing'):
            self.vae.enable_attention_slicing()

    def disable_attention_slicing(self):
        """禁用注意力切片"""
        if hasattr(self.vae, 'disable_attention_slicing'):
            self.vae.disable_attention_slicing()

    def enable_cpu_offload(self):
        """启用CPU卸载"""
        if hasattr(self.vae, 'enable_sequential_cpu_offload'):
            self.vae.enable_sequential_cpu_offload()

    def to(self, device):
        """移动模型到指定设备"""
        super().to(device)
        self.device = device
        if hasattr(self, 'vae'):
            self.vae = self.vae.to(device)
            self.vae_device = device
        return self


def test_wan_heatmap_model():
    """
    测试Wan heatmap模型
    """
    print("Testing Wan heatmap model...")

    try:
        # 创建配置
        from ..configs.model_config import get_debug_config
        config = get_debug_config().model
        config.sequence_length = 5
        config.input_image_size = (64, 64)
        config.output_image_size = (64, 64)

        # 创建模型
        model = WanHeatmapModel(config)
        model.eval()

        # 创建测试数据
        batch_size = 2
        rgb_images = torch.randn(batch_size, 3, 64, 64)
        heatmap_sequences = torch.rand(batch_size, 5, 64, 64)  # [0, 1]范围

        print(f"Input RGB shape: {rgb_images.shape}")
        print(f"Input heatmap shape: {heatmap_sequences.shape}")

        # 测试前向传播（训练模式）
        model.train()
        with torch.no_grad():
            train_output = model(rgb_images, heatmap_sequences)

        print(f"Training output keys: {list(train_output.keys())}")
        print(f"Predictions shape: {train_output['predictions'].shape}")
        print(f"Reconstruction loss: {train_output['reconstruction_loss'].item():.4f}")

        # 测试推理模式
        model.eval()
        with torch.no_grad():
            inference_output = model.generate_heatmap_sequence(rgb_images, num_inference_steps=10)

        print(f"Inference output keys: {list(inference_output.keys())}")
        print(f"Generated heatmaps shape: {inference_output['predictions'].shape}")

        print("Wan heatmap model test completed successfully!")
        return True

    except Exception as e:
        print(f"Wan heatmap model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_wan_heatmap_model()