"""
Test VAE Channel Statistics
测试VAE编码video和input_video_rgb后的通道间差异
"""

import torch
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime

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
from diffsynth.trainers.heatmap_dataset import HeatmapDatasetFactory


class Logger:
    """双重输出：同时输出到终端和文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def print_highlight(message):
    """高亮打印关键信息"""
    # ANSI颜色代码
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    # 在终端显示高亮，在文件中显示普通文本
    print(f"{BOLD}{GREEN}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}{message}{RESET}")
    print(f"{BOLD}{GREEN}{'='*80}{RESET}")


class VAEChannelTester:
    """VAE通道统计测试器"""

    def __init__(self,
                 model_base_path: str,
                 wan_type: str = "5B_TI2V_RGB_HEATMAP",
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16):
        """
        初始化测试器

        Args:
            model_base_path: 基础模型路径
            wan_type: 模型类型
            device: 设备
            torch_dtype: 张量类型
        """
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading {wan_type} pipeline...")
        # 加载pipeline
        if wan_type == "5B_TI2V_RGB_HEATMAP":
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch_dtype,
                device=device,
                wan_type=wan_type,
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

        print("Pipeline initialized successfully!")

    def compute_scaling_constants(self,
                                  data_root: str,
                                  sample_sizes: list = [100, 500, 1000],
                                  sequence_length: int = 4):
        """
        计算RGB和Heatmap latent之间的缩放常数

        目标：让 heatmap_latent * scale 的均值/方差与 rgb_latent 对齐

        Args:
            data_root: 数据根目录
            sample_sizes: 要测试的样本数量列表
            sequence_length: 序列长度
        """
        print(f"\n{'='*80}")
        print("COMPUTING SCALING CONSTANTS FOR LATENT ALIGNMENT")
        print(f"{'='*80}")
        print(f"Data root: {data_root}")
        print(f"Sample sizes to test: {sample_sizes}")
        print(f"Sequence length: {sequence_length}")
        print(f"{'='*80}\n")

        # 创建数据集
        dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
            data_root=data_root,
            sequence_length=sequence_length,
            step_interval=1,
            min_trail_length=10,
            image_size=(256, 256),
            sigma=1.5,
            augmentation=False,
            mode="train",
            scene_bounds=[0, -0.45, -0.05, 0.8, 0.55, 0.6],
            transform_augmentation_xyz=[0.0, 0.0, 0.0],
            transform_augmentation_rpy=[0.0, 0.0, 0.0],
            debug=False,
            colormap_name="jet",
            repeat=1,
            wan_type="5B_TI2V_RGB_HEATMAP"
        )

        print(f"Dataset loaded: {len(dataset)} samples")

        # 确定最大样本数
        max_samples = max(sample_sizes)
        max_samples = min(max_samples, len(dataset))

        # 收集所有latent（一次性收集最大数量）
        all_z_rgb = []
        all_z_heatmap = []

        print(f"\nProcessing {max_samples} samples for all tests...")

        with torch.no_grad():
            for i in tqdm(range(max_samples), desc="Encoding samples"):
                try:
                    # 获取数据样本
                    sample = dataset[i]

                    # 获取video (heatmap) 和 input_video_rgb
                    video_frames = sample['video']  # List[PIL.Image]
                    rgb_frames = sample['input_video_rgb']  # List[PIL.Image]

                    # 使用pipeline的preprocess_video方法（参考WanVideoUnit_InputVideoEmbedder）
                    input_video = self.pipe.preprocess_video(video_frames)  # [1, C, T, H, W]
                    input_video_rgb = self.pipe.preprocess_video(rgb_frames)  # [1, C, T, H, W]

                    # VAE编码（参考WanVideoUnit_InputVideoEmbedder的编码方式）
                    z_heatmap = self.pipe.vae.encode(
                        input_video,
                        device=self.device
                    ).to(dtype=self.torch_dtype, device=self.device)  # [1, C_latent, T, H_latent, W_latent]

                    z_rgb = self.pipe.vae.encode(
                        input_video_rgb,
                        device=self.device
                    ).to(dtype=self.torch_dtype, device=self.device)  # [1, C_latent, T, H_latent, W_latent]

                    # 收集latent
                    all_z_rgb.append(z_rgb.cpu())
                    all_z_heatmap.append(z_heatmap.cpu())

                except Exception as e:
                    print(f"\nError processing sample {i}: {e}")
                    continue

        if len(all_z_rgb) == 0:
            print("No samples successfully processed!")
            return

        print(f"\nSuccessfully processed {len(all_z_rgb)} samples")

        # 全部latent的形状
        z_rgb_all = torch.cat(all_z_rgb, dim=0)  # [N, C, T, H, W]
        z_heatmap_all = torch.cat(all_z_heatmap, dim=0)  # [N, C, T, H, W]

        print(f"\nTotal latent shapes:")
        print(f"  z_rgb: {z_rgb_all.shape}")
        print(f"  z_heatmap: {z_heatmap_all.shape}")

        # 存储不同样本数量下的缩放常数
        scaling_results = {}

        # 对每个样本大小进行测试
        for num_samples in sample_sizes:
            if num_samples > len(all_z_rgb):
                print(f"\nWarning: Requested {num_samples} samples but only {len(all_z_rgb)} available. Skipping.")
                continue

            print(f"\n{'='*80}")
            print(f"TESTING WITH {num_samples} SAMPLES")
            print(f"{'='*80}\n")

            # 截取前N个样本
            z_rgb = torch.cat(all_z_rgb[:num_samples], dim=0)  # [N, C, T, H, W]
            z_heatmap = torch.cat(all_z_heatmap[:num_samples], dim=0)  # [N, C, T, H, W]

            # 计算RGB统计
            mu_rgb = z_rgb.mean(dim=[0, 2, 3, 4])  # [C]
            sigma_rgb = z_rgb.std(dim=[0, 2, 3, 4])  # [C]

            # 计算Heatmap统计
            mu_heatmap = z_heatmap.mean(dim=[0, 2, 3, 4])  # [C]
            sigma_heatmap = z_heatmap.std(dim=[0, 2, 3, 4])  # [C]

            print("RGB Latent Statistics:")
            print(f"  Per-channel mean: {mu_rgb.tolist()}")
            print(f"  Per-channel std:  {sigma_rgb.tolist()}")
            print(f"  Overall mean: {mu_rgb.mean().item():.6f}")
            print(f"  Overall std:  {sigma_rgb.mean().item():.6f}")

            print("\nHeatmap Latent Statistics:")
            print(f"  Per-channel mean: {mu_heatmap.tolist()}")
            print(f"  Per-channel std:  {sigma_heatmap.tolist()}")
            print(f"  Overall mean: {mu_heatmap.mean().item():.6f}")
            print(f"  Overall std:  {sigma_heatmap.mean().item():.6f}")

            # 计算缩放常数
            # scale = rgb / heatmap
            # 这样 heatmap * scale ≈ rgb
            mean_scale = mu_rgb / (mu_heatmap + 1e-8)  # 避免除零
            std_scale = sigma_rgb / (sigma_heatmap + 1e-8)

            print(f"\n{'='*80}")
            print("SCALING CONSTANTS")
            print(f"{'='*80}\n")

            print("Mean Scaling (RGB / Heatmap):")
            print(f"  Per-channel: {mean_scale.tolist()}")
            print(f"  Average: {mean_scale.mean().item():.6f}")
            print(f"  Range: [{mean_scale.min().item():.6f}, {mean_scale.max().item():.6f}]")

            print("\nStd Scaling (RGB / Heatmap):")
            print(f"  Per-channel: {std_scale.tolist()}")
            print(f"  Average: {std_scale.mean().item():.6f}")
            print(f"  Range: [{std_scale.min().item():.6f}, {std_scale.max().item():.6f}]")

            # 验证缩放效果
            z_heatmap_scaled = z_heatmap * std_scale.view(1, -1, 1, 1, 1)
            mu_heatmap_scaled = z_heatmap_scaled.mean(dim=[0, 2, 3, 4])
            sigma_heatmap_scaled = z_heatmap_scaled.std(dim=[0, 2, 3, 4])

            print("\nScaled Heatmap Latent Statistics (after std scaling):")
            print(f"  Per-channel mean: {mu_heatmap_scaled.tolist()}")
            print(f"  Per-channel std:  {sigma_heatmap_scaled.tolist()}")
            print(f"  Overall mean: {mu_heatmap_scaled.mean().item():.6f}")
            print(f"  Overall std:  {sigma_heatmap_scaled.mean().item():.6f}")

            # 测试合并后的分布
            print("\nCombined Distribution (RGB + Scaled Heatmap):")
            x_combined = torch.cat([z_rgb, z_heatmap_scaled], dim=1)
            mu_combined = x_combined.mean(dim=[0, 2, 3, 4])
            sigma_combined = x_combined.std(dim=[0, 2, 3, 4])
            print(f"  Overall mean: {mu_combined.mean().item():.6f}")
            print(f"  Overall std:  {sigma_combined.mean().item():.6f}")
            print(f"  Std range: [{sigma_combined.min().item():.6f}, {sigma_combined.max().item():.6f}]")

            # 存储结果
            scaling_results[num_samples] = {
                'mean_scale': mean_scale,
                'std_scale': std_scale,
                'mean_scale_avg': mean_scale.mean().item(),
                'std_scale_avg': std_scale.mean().item(),
            }

        # 最终推荐
        print(f"\n{'='*80}")
        print("FINAL RECOMMENDATIONS")
        print(f"{'='*80}\n")

        print("Scaling constants for different sample sizes:")
        print(f"{'Samples':<10} {'Mean Scale (avg)':<20} {'Std Scale (avg)':<20}")
        print("-" * 50)
        for num_samples in sample_sizes:
            if num_samples in scaling_results:
                result = scaling_results[num_samples]
                print(f"{num_samples:<10} {result['mean_scale_avg']:<20.6f} {result['std_scale_avg']:<20.6f}")

        # 推荐使用最大样本数的结果
        max_tested_samples = max([s for s in sample_sizes if s in scaling_results])
        recommended_scale = scaling_results[max_tested_samples]['std_scale']
        recommended_scale_avg = recommended_scale.mean().item()

        # ========== 高亮显示关键结果 ==========
        print_highlight(f"RECOMMENDED SCALING CONSTANT (based on {max_tested_samples} samples)")
        print(f"\n  Average Std Scale: {recommended_scale_avg:.6f}")
        print(f"  Per-channel Std Scale: {recommended_scale.tolist()}\n")

        print_highlight("USAGE INSTRUCTIONS")
        print("\nTraining:")
        print(f"  z_heatmap_scaled = z_heatmap * {recommended_scale_avg:.6f}")
        print("\n  # Or per-channel:")
        print(f"  scale = torch.tensor({recommended_scale.tolist()}).view(1, -1, 1, 1, 1)")
        print(f"  z_heatmap_scaled = z_heatmap * scale")
        print("\nInference:")
        print(f"  z_heatmap_original = z_heatmap_predicted / {recommended_scale_avg:.6f}\n")

        print(f"\n{'='*80}")
        print("TEST COMPLETED")
        print(f"{'='*80}\n")

        return scaling_results, recommended_scale_avg


def main():
    """主函数"""
    # 配置
    MODEL_BASE_PATH = "/data/lpy/huggingface/Wan2.2-TI2V-5B-fused"

    # 自动检测数据集路径
    possible_data_roots = [
        "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf",
        "/data/wxn/V2W_Real/put_the_lion_on_the_top_shelf"
    ]
    DATA_ROOT = None
    for path in possible_data_roots:
        if os.path.exists(path):
            DATA_ROOT = path
            break
    if DATA_ROOT is None:
        raise RuntimeError(f"Cannot find dataset in any of: {possible_data_roots}")

    # 创建输出目录和日志文件
    output_dir = f"{ROOT_PATH}/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/scaling_results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"vae_scaling_test_{timestamp}.log")

    # 重定向stdout到日志文件
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"Log file: {log_file}")
    print(f"Using DATA_ROOT: {DATA_ROOT}")
    print(f"Using MODEL_BASE_PATH: {MODEL_BASE_PATH}\n")

    try:
        # 创建测试器
        tester = VAEChannelTester(
            model_base_path=MODEL_BASE_PATH,
            wan_type="5B_TI2V_RGB_HEATMAP",
            device="cuda",
            torch_dtype=torch.bfloat16
        )

        # 运行测试 - 计算缩放常数
        scaling_results, recommended_scale = tester.compute_scaling_constants(
            data_root=DATA_ROOT,
            sample_sizes=[100, 500, 2000,4770],  # 测试100、500、1000个样本
            sequence_length=4
        )

        # 保存结果到JSON文件
        if scaling_results:
            json_file = os.path.join(output_dir, f"scaling_constants_{timestamp}.json")

            # 转换tensor为list以便保存
            results_to_save = {
                'timestamp': timestamp,
                'data_root': DATA_ROOT,
                'model_base_path': MODEL_BASE_PATH,
                'recommended_scale': recommended_scale,
                'results': {}
            }

            for num_samples, result in scaling_results.items():
                results_to_save['results'][str(num_samples)] = {
                    'mean_scale': result['mean_scale'].tolist(),
                    'std_scale': result['std_scale'].tolist(),
                    'mean_scale_avg': result['mean_scale_avg'],
                    'std_scale_avg': result['std_scale_avg'],
                }

            with open(json_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)

            print_highlight(f"RESULTS SAVED")
            print(f"\nLog file: {log_file}")
            print(f"JSON file: {json_file}\n")

    finally:
        # 恢复stdout并关闭日志文件
        sys.stdout = logger.terminal
        logger.close()

        print(f"\n✓ Test completed! Results saved to:")
        print(f"  - Log: {log_file}")
        print(f"  - JSON: {os.path.join(output_dir, f'scaling_constants_{timestamp}.json')}")


if __name__ == "__main__":
    main()
