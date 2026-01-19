"""
Heatmap Training Script for Wan2.2
基于原始train.py，专门用于热力图序列生成训练
"""

import torch
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 强制刷新输出，确保错误信息能被显示
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# SwanLab导入（可选）
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available. Install with: pip install swanlab")
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import torch.utils.data

# 添加trainers路径以导入我们的自定义数据集
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from diffsynth.trainers.heatmap_dataset import HeatmapDatasetFactory

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HeatmapWanTrainingModule(DiffusionTrainingModule):
    """
    热力图专用的Wan训练模块
    继承原始WanTrainingModule，针对热力图序列生成进行优化
    """

    def __init__(
        self,
        wan_type,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_dual_head=False,
    ):
        super().__init__()
        self.wan_type = wan_type
        if self.wan_type == "WAN_2_1_14B_I2V":
            from diffsynth.pipelines.wan_video_14BI2V_condition_rgb_heatmap_first import WanVideoPipeline, ModelConfig
        elif self.wan_type == "5B_TI2V":
            from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP":
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb import WanVideoPipeline, ModelConfig
        else:
            assert False, f"Unsupported wan_type: {self.wan_type}"
        # Load models (使用与train.py相同的方式)
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if local_rank == 0:
            print("\n" + "="*80)
            print(f"[DEBUG] Loading pipeline with use_dual_head={use_dual_head}")
            print("="*80)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, wan_type=self.wan_type, use_dual_head=use_dual_head)
        self.use_dual_head = use_dual_head
        if local_rank == 0:
            print("\n" + "="*80)
            print(f"[DEBUG] Pipeline loaded successfully with use_dual_head={use_dual_head}")
            print("="*80 + "\n")

        # Training mode (使用与train.py相同的方式)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # ============================================================
        # 解冻 patch_embedding 和 head.head 的所有参数
        # 这些层使用全量训练而非LoRA
        # ============================================================
        self._unfreeze_patch_embedding_and_head()

        # ============================================================
        # 参数监控：打印可训练参数信息
        # ============================================================
        self._print_trainable_parameters_info()

        # ============================================================
        # 参数更新检测：保存 patch_embedding 和 head.head 的初始参数
        # ============================================================
        # self._save_initial_parameters()  # 保存初始参数以检测bias更新

        # Store other configs (使用与train.py相同的方式)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = False  # 添加此参数以保持与train.py的兼容性
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Debug settings
        self.debug_counter = 0
        self.debug_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../debug_log"))

    def _unfreeze_patch_embedding_and_head(self):
        """
        解冻 patch_embedding 和 head 的所有参数（权重和bias）

        这些层使用全量训练而非LoRA微调。
        由于它们被重新初始化以适应新的输入/输出维度，
        所有参数都需要从头学习。

        在双head模式下，会解冻 head_rgb 和 head_heatmap 而不是 head
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []
        all_head_params = []  # 用于调试：所有包含head的参数

        for name, param in self.pipe.dit.named_parameters():
            # 调试：收集所有包含head的参数名
            if 'head' in name:
                all_head_params.append((name, param.requires_grad))

            # 检查是否是 patch_embedding 或 head 的参数
            # 在双head模式下，匹配 head_rgb 和 head_heatmap
            # 在单head模式下，匹配 head
            # 注意：这里不包括LoRA参数（lora_A/lora_B），只包括原始层的weight和bias
            is_patch_embedding = 'patch_embedding' in name
            is_head = ('head' in name) if not self.use_dual_head else ('head_rgb' in name or 'head_heatmap' in name)
            is_not_lora = ('lora_A' not in name and 'lora_B' not in name)

            if (is_patch_embedding or is_head) and is_not_lora:
                # 解冻这个参数
                param.requires_grad = True
                unfrozen_params.append(name)

        # 只在主进程打印
        if local_rank == 0:
            print("\n" + "="*80)
            mode_str = "DUAL HEAD" if self.use_dual_head else "SINGLE HEAD"
            print(f"FULL PARAMETER UNFREEZING FOR PATCH_EMBEDDING AND HEAD ({mode_str} MODE)")
            print("="*80)

            # 调试信息：打印所有包含head的参数
            print(f"\n[DEBUG] All parameters with 'head' in name ({len(all_head_params)} total):")
            for name, requires_grad in all_head_params:
                print(f"  - {name}: requires_grad={requires_grad}")

            if len(unfrozen_params) > 0:
                print(f"\nUnfroze {len(unfrozen_params)} parameter(s) for full training:")
                for name in unfrozen_params:
                    param = dict(self.pipe.dit.named_parameters())[name]
                    print(f"  ✓ {name}")
                    print(f"    Shape: {param.shape}, Trainable: {param.requires_grad}")
            else:
                print("\n⚠️  WARNING: No parameters were unfrozen!")
            print("="*80 + "\n")

    def _print_trainable_parameters_info(self):
        """
        打印可训练参数的详细信息
        特别关注 patch_embedding 和 head 的参数状态
        只在主进程打印
        """
        # 只在主进程打印
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return

        print("\n" + "="*80)
        print("TRAINABLE PARAMETERS MONITORING")
        print("="*80)

        # 统计信息
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        # 特别关注的模块
        patch_embedding_params = {}
        head_params = {}
        lora_params = {}
        other_trainable_params = {}

        # 遍历所有参数
        for name, param in self.pipe.dit.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()

                # 分类参数
                if 'patch_embedding' in name:
                    patch_embedding_params[name] = {
                        'shape': tuple(param.shape),
                        'numel': param.numel(),
                        'dtype': param.dtype
                    }
                elif 'head' in name:
                    head_params[name] = {
                        'shape': tuple(param.shape),
                        'numel': param.numel(),
                        'dtype': param.dtype
                    }
                elif 'lora' in name.lower():
                    lora_params[name] = {
                        'shape': tuple(param.shape),
                        'numel': param.numel(),
                        'dtype': param.dtype
                    }
                else:
                    other_trainable_params[name] = {
                        'shape': tuple(param.shape),
                        'numel': param.numel(),
                        'dtype': param.dtype
                    }
            else:
                frozen_params += param.numel()

        # 打印总览
        print(f"\n📊 Parameter Overview:")
        print(f"  Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M, {trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/1e6:.2f}M, {frozen_params/total_params*100:.2f}%)")

        # 打印 patch_embedding 参数
        print(f"\n🔍 PATCH_EMBEDDING Parameters:")
        if patch_embedding_params:
            print(f"  ✅ {len(patch_embedding_params)} trainable parameter(s) found:")
            for name, info in patch_embedding_params.items():
                print(f"    - {name}")
                print(f"      Shape: {info['shape']}, Count: {info['numel']:,}, Dtype: {info['dtype']}")
        else:
            print(f"  ❌ NO trainable parameters found in patch_embedding!")
            print(f"     → patch_embedding parameters are FROZEN")

        # 打印 head 参数
        print(f"\n🔍 HEAD Parameters:")
        if head_params:
            print(f"  ✅ {len(head_params)} trainable parameter(s) found:")
            for name, info in head_params.items():
                print(f"    - {name}")
                print(f"      Shape: {info['shape']}, Count: {info['numel']:,}, Dtype: {info['dtype']}")
        else:
            print(f"  ❌ NO trainable parameters found in head!")
            print(f"     → head parameters are FROZEN")

        # 打印 LoRA 参数
        print(f"\n🔍 LoRA Parameters:")
        if lora_params:
            lora_param_count = sum(info['numel'] for info in lora_params.values())
            print(f"  ✅ {len(lora_params)} LoRA parameter(s) found ({lora_param_count:,} total, {lora_param_count/1e6:.2f}M):")
            # 只显示前5个，避免输出过长
            for i, (name, info) in enumerate(list(lora_params.items())[:5]):
                print(f"    - {name}")
                print(f"      Shape: {info['shape']}, Count: {info['numel']:,}")
            if len(lora_params) > 5:
                print(f"    ... and {len(lora_params) - 5} more LoRA parameters")
        else:
            print(f"  ⚠️  NO LoRA parameters found!")

        # 打印其他可训练参数
        if other_trainable_params:
            other_param_count = sum(info['numel'] for info in other_trainable_params.values())
            print(f"\n📋 Other Trainable Parameters:")
            print(f"  {len(other_trainable_params)} parameter(s) ({other_param_count:,} total, {other_param_count/1e6:.2f}M)")
            # 只显示前3个
            for i, (name, info) in enumerate(list(other_trainable_params.items())[:3]):
                print(f"    - {name}: {info['shape']}")
            if len(other_trainable_params) > 3:
                print(f"    ... and {len(other_trainable_params) - 3} more")

        # 重要提示
        print(f"\n💡 Important Notes:")
        if not patch_embedding_params:
            print(f"  ⚠️  WARNING: patch_embedding is NOT being trained!")
            print(f"     If you modified in_dim, you should add 'patch_embedding' to lora_target_modules")
        if not head_params:
            print(f"  ⚠️  WARNING: head is NOT being trained!")
            print(f"     If you modified out_dim, you should add 'head.head' to lora_target_modules")

        print("="*80 + "\n")

    def _save_initial_parameters(self):
        """
        保存 patch_embedding 和 head.head 的初始参数值
        用于后续检测这些参数是否真的在训练中更新
        """
        self.initial_params = {}

        for name, param in self.pipe.dit.named_parameters():
            if ('patch_embedding' in name or 'head' in name) and param.requires_grad:
                # 保存参数的克隆副本（detach以避免影响梯度计算）
                self.initial_params[name] = param.data.clone().detach()

        # 只在主进程打印
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0 and len(self.initial_params) > 0:
            weight_count = sum(1 for name in self.initial_params.keys() if 'weight' in name or 'lora' in name)
            bias_count = sum(1 for name in self.initial_params.keys() if 'bias' in name)
            print(f"📸 Saved initial parameters for {len(self.initial_params)} parameters")
            print(f"   - Weight/LoRA parameters: {weight_count}")
            print(f"   - Bias parameters: {bias_count}")
            print(f"   (patch_embedding and head parameters will be monitored for updates)")

    def check_parameter_updates(self, step):
        """
        检查 patch_embedding 和 head.head 的参数是否已更新
        只在主进程打印

        Args:
            step: 当前训练步数
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return

        if not hasattr(self, 'initial_params') or len(self.initial_params) == 0:
            return

        print(f"\n🔍 Checking parameter updates at step {step}:")
        print("="*80)

        updated_count = 0
        unchanged_count = 0
        bias_updated_count = 0
        bias_unchanged_count = 0

        # 获取当前参数字典（只创建一次，提高效率）
        current_params_dict = dict(self.pipe.dit.named_parameters())

        # 分别处理 weight/LoRA 参数和 bias 参数
        weight_params = {name: val for name, val in self.initial_params.items() if 'bias' not in name}
        bias_params = {name: val for name, val in self.initial_params.items() if 'bias' in name}

        # 检查 weight/LoRA 参数
        if weight_params:
            print("\n📊 Weight/LoRA Parameters:")
            for name, initial_value in weight_params.items():
                # 获取当前参数值
                if name not in current_params_dict:
                    print(f"  ⚠️  WARNING: {name} not found in current model parameters!")
                    continue

                current_value = current_params_dict[name].data

                # 确保初始值和当前值在同一设备上进行比较
                if initial_value.device != current_value.device:
                    initial_value = initial_value.to(current_value.device)

                # 计算参数变化的程度
                diff = (current_value - initial_value).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                # 判断参数是否更新（使用较小的阈值，因为LoRA更新可能很小）
                is_updated = max_diff > 1e-8

                if is_updated:
                    updated_count += 1
                    status = "✅ UPDATED"
                    print(f"  {status}: {name}")
                    print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
                else:
                    unchanged_count += 1
                    status = "❌ UNCHANGED"
                    print(f"  {status}: {name}")
                    print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

        # 检查 bias 参数
        if bias_params:
            print("\n📊 Bias Parameters:")
            for name, initial_value in bias_params.items():
                # 获取当前参数值
                if name not in current_params_dict:
                    print(f"  ⚠️  WARNING: {name} not found in current model parameters!")
                    continue

                current_value = current_params_dict[name].data

                # 确保初始值和当前值在同一设备上进行比较
                if initial_value.device != current_value.device:
                    initial_value = initial_value.to(current_value.device)

                # 计算参数变化的程度
                diff = (current_value - initial_value).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                # 判断参数是否更新（使用较小的阈值）
                is_updated = max_diff > 1e-8

                if is_updated:
                    bias_updated_count += 1
                    status = "✅ UPDATED"
                    print(f"  {status}: {name}")
                    print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
                else:
                    bias_unchanged_count += 1
                    status = "❌ UNCHANGED"
                    print(f"  {status}: {name}")
                    print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

        print("="*80)
        print("Summary:")
        print(f"  Weight/LoRA: {updated_count} updated, {unchanged_count} unchanged")
        print(f"  Bias: {bias_updated_count} updated, {bias_unchanged_count} unchanged")
        print(f"  Total: {updated_count + bias_updated_count} updated, {unchanged_count + bias_unchanged_count} unchanged")

        if unchanged_count > 0:
            print(f"⚠️  WARNING: {unchanged_count} parameter(s) have NOT been updated!")
            print(f"   This may indicate that these parameters are not in the optimizer")

        print("="*80 + "\n")

    def forward_preprocess(self, data):
        """
        预处理输入数据，专门针对热力图数据格式
        注意：Wan模型只支持batch_size=1
        """
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # 热力图序列作为视频输入
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # 训练相关参数
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "use_dual_head":self.use_dual_head,
        }

        # Extra inputs - 对于热力图任务，主要是input_image
        '''
        # check only the follow conditions for self.extra_inputs:
        (input_image)
        (input_image,condition_rgb)
        (input_image,input_image_rgb,input_video_rgb)
        '''
        # check self.extra_inputs satisfy the above three conditions
        assert self.extra_inputs == ["input_image"] or \
               self.extra_inputs == ["input_image", "condition_rgb"] or \
               self.extra_inputs == ["input_image", "input_image_rgb", "input_video_rgb"]
        
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # 使用首帧RGB图像/热力图作为条件输入
                inputs_shared["input_image"] = data["input_image"]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "condition_rgb":
                # 将RGB作为额外的条件输入
                inputs_shared["condition_rgb"] = data["condition_rgb"]
            elif extra_input == "input_image_rgb":
                inputs_shared["input_image_rgb"] = data["input_image_rgb"]
            elif extra_input == "input_video_rgb":
                inputs_shared["input_video_rgb"] = data["input_video_rgb"]
            else:
                if extra_input in data:
                    inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def visualize_forward_inputs(self, inputs):
        """
        可视化 forward 前的输入数据
        包括: input_image, input_image_rgb, input_video, input_video_rgb
        所有输入组合到一个大图中便于查看
        只在主进程执行
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return

        try:
            # 确保debug目录存在
            os.makedirs(self.debug_save_dir, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"FORWARD INPUT VISUALIZATION (Step: {self.debug_counter})")
            print(f"{'='*80}")
            print(f"Available keys: {list(inputs.keys())}")

            # 收集所有要可视化的数据
            input_image = inputs.get('input_image')
            input_image_rgb = inputs.get('input_image_rgb')
            input_video = inputs.get('input_video')
            input_video_rgb = inputs.get('input_video_rgb')

            # 计算布局
            video_frames = len(input_video) if (input_video and isinstance(input_video, list)) else 0
            video_rgb_frames = len(input_video_rgb) if (input_video_rgb and isinstance(input_video_rgb, list)) else 0
            max_video_frames = max(video_frames, video_rgb_frames)
            video_cols = min(8, max_video_frames) if max_video_frames > 0 else 2

            # 创建大图：5行
            total_rows = 5
            fig = plt.figure(figsize=(2.5*video_cols, 14))
            gs = fig.add_gridspec(total_rows, video_cols, hspace=0.25, wspace=0.05,
                                 height_ratios=[2, 0.3, 1.5, 0.3, 1.5])

            # ============ 第1行：input_image 和 input_image_rgb ============
            ax_input_image = fig.add_subplot(gs[0, :video_cols//2])
            if input_image and hasattr(input_image, 'save'):
                img_array = np.array(input_image)
                ax_input_image.imshow(img_array)
                ax_input_image.set_title('Input Image\n(Heatmap First Frame)', fontsize=11, fontweight='bold', pad=10)
                print(f"✓ input_image (heatmap): Size {input_image.size}, Mode {input_image.mode}")
            else:
                ax_input_image.text(0.5, 0.5, 'No input_image', ha='center', va='center', fontsize=10, transform=ax_input_image.transAxes)
                ax_input_image.set_title('Input Image\n(Heatmap First Frame)', fontsize=11, fontweight='bold', pad=10)
            ax_input_image.axis('off')

            ax_input_image_rgb = fig.add_subplot(gs[0, video_cols//2:])
            if input_image_rgb and hasattr(input_image_rgb, 'save'):
                img_rgb_array = np.array(input_image_rgb)
                ax_input_image_rgb.imshow(img_rgb_array)
                ax_input_image_rgb.set_title('Input Image RGB\n(RGB First Frame)', fontsize=11, fontweight='bold', pad=10)
                print(f"✓ input_image_rgb: Size {input_image_rgb.size}, Mode {input_image_rgb.mode}")
            else:
                ax_input_image_rgb.text(0.5, 0.5, 'No input_image_rgb', ha='center', va='center', fontsize=10, transform=ax_input_image_rgb.transAxes)
                ax_input_image_rgb.set_title('Input Image RGB\n(RGB First Frame)', fontsize=11, fontweight='bold', pad=10)
            ax_input_image_rgb.axis('off')

            # ============ 第2行：视频标题 ============
            ax_video_title = fig.add_subplot(gs[1, :])
            ax_video_title.text(0.5, 0.5, 'Input Video (Heatmap Sequence)', ha='center', va='center',
                               fontsize=13, fontweight='bold', transform=ax_video_title.transAxes,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax_video_title.axis('off')

            # ============ 第3行：input_video (heatmap序列) ============
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                print(f"✓ input_video (heatmap sequence): {len(input_video)} frames")
                for i, frame in enumerate(input_video[:video_cols]):
                    ax = fig.add_subplot(gs[2, i])
                    if hasattr(frame, 'save'):
                        frame_array = np.array(frame)
                        ax.imshow(frame_array)
                        ax.set_title(f"F{i}", fontsize=9)
                    ax.axis('off')
                # 填充空白
                for i in range(len(input_video), video_cols):
                    ax = fig.add_subplot(gs[2, i])
                    ax.axis('off')
            else:
                ax = fig.add_subplot(gs[2, :])
                ax.text(0.5, 0.5, 'No input_video', ha='center', va='center', fontsize=10, transform=ax.transAxes)
                ax.axis('off')

            # ============ 第4行：RGB视频标题 ============
            ax_video_rgb_title = fig.add_subplot(gs[3, :])
            ax_video_rgb_title.text(0.5, 0.5, 'Input Video RGB (RGB Sequence)', ha='center', va='center',
                                   fontsize=13, fontweight='bold', transform=ax_video_rgb_title.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax_video_rgb_title.axis('off')

            # ============ 第5行：input_video_rgb (RGB序列) ============
            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                print(f"✓ input_video_rgb (RGB sequence): {len(input_video_rgb)} frames")
                for i, frame in enumerate(input_video_rgb[:video_cols]):
                    ax = fig.add_subplot(gs[4, i])
                    if hasattr(frame, 'save'):
                        frame_array = np.array(frame)
                        ax.imshow(frame_array)
                        ax.set_title(f"F{i}", fontsize=9)
                    ax.axis('off')
                # 填充空白
                for i in range(len(input_video_rgb), video_cols):
                    ax = fig.add_subplot(gs[4, i])
                    ax.axis('off')
            else:
                ax = fig.add_subplot(gs[4, :])
                ax.text(0.5, 0.5, 'No input_video_rgb', ha='center', va='center', fontsize=10, transform=ax.transAxes)
                ax.axis('off')

            # 保存组合图
            save_path = os.path.join(self.debug_save_dir, f"step_{self.debug_counter:04d}_all_inputs.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print("\n✅ All inputs combined visualization saved to:")
            print(f"   {save_path}")

            # 打印其他关键信息
            print("\nOther inputs:")
            for key in ['prompt', 'height', 'width', 'num_frames']:
                if key in inputs:
                    print(f"  {key}: {inputs[key]}")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"❌ Error in forward input visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualize_processed_inputs(self, inputs, data):
        """
        可视化经过forward_preprocess处理后的input_video和input_image
        """
        try:
            # 确保debug目录存在
            os.makedirs(self.debug_save_dir, exist_ok=True)

            print(f"\n=== DEBUG VISUALIZATION (Counter: {self.debug_counter}) ===")
            print(f"Original prompt: {data.get('prompt', 'N/A')}")
            print(f"Processed inputs keys: {list(inputs.keys())}")

            # 可视化processed input_image
            if 'input_image' in inputs and inputs['input_image'] is not None:
                input_img = inputs['input_image']
                if hasattr(input_img, 'save'):  # PIL Image
                    input_img_path = os.path.join(self.debug_save_dir, f"debug_{self.debug_counter:04d}_processed_input_image.png")
                    input_img.save(input_img_path)
                    print(f"Processed input image saved: {input_img_path}")
                    print(f"Processed input image size: {input_img.size}")
                else:
                    print(f"Input image type: {type(input_img)}, shape: {getattr(input_img, 'shape', 'N/A')}")

            # 可视化processed input_video
            if 'input_video' in inputs and inputs['input_video'] is not None:
                input_video = inputs['input_video']

                if isinstance(input_video, list) and len(input_video) > 0:
                    # 如果是PIL图像列表
                    video_frames = input_video
                    num_frames = len(video_frames)

                    print(f"Processed video frames count: {num_frames}")

                    # 创建网格显示所有帧
                    cols = min(5, num_frames)  # 最多5列
                    rows = (num_frames + cols - 1) // cols

                    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
                    if rows == 1 and cols == 1:
                        axes = [axes]
                    elif rows == 1 or cols == 1:
                        axes = axes.flatten()
                    else:
                        axes = axes.flatten()

                    for i, frame in enumerate(video_frames):
                        if i < len(axes):
                            # 将PIL图像转换为numpy数组
                            if hasattr(frame, 'save'):  # PIL Image
                                frame_array = np.array(frame)
                                axes[i].imshow(frame_array)
                                axes[i].set_title(f"Processed Frame {i}")
                                axes[i].axis('off')

                                # 同时保存单独的帧
                                frame_path = os.path.join(self.debug_save_dir, f"debug_{self.debug_counter:04d}_processed_frame_{i:02d}.png")
                                frame.save(frame_path)
                            else:
                                axes[i].text(0.5, 0.5, f"Frame {i}\n{type(frame)}",
                                           ha='center', va='center', transform=axes[i].transAxes)
                                axes[i].axis('off')

                    # 隐藏多余的子图
                    for i in range(num_frames, len(axes)):
                        axes[i].axis('off')

                    # 保存组合图
                    combined_path = os.path.join(self.debug_save_dir, f"debug_{self.debug_counter:04d}_processed_video_sequence.png")
                    plt.tight_layout()
                    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"Processed video sequence saved: {combined_path}")
                    print(f"Individual processed frames saved with pattern: debug_{self.debug_counter:04d}_processed_frame_XX.png")

                    if len(video_frames) > 0 and hasattr(video_frames[0], 'size'):
                        print(f"Processed video info: {num_frames} frames, size: {video_frames[0].size}")
                else:
                    print(f"Input video type: {type(input_video)}, shape: {getattr(input_video, 'shape', 'N/A')}")

            # 打印其他重要的inputs信息
            for key, value in inputs.items():
                if key not in ['input_image', 'input_video']:
                    if isinstance(value, (int, float, str, bool)):
                        print(f"  {key}: {value}")
                    elif hasattr(value, 'shape'):
                        print(f"  {key}: shape {value.shape}, dtype {getattr(value, 'dtype', 'N/A')}")
                    elif isinstance(value, (list, tuple)):
                        print(f"  {key}: {type(value)} of length {len(value)}")
                    else:
                        print(f"  {key}: {type(value)}")

            print(f"=== END DEBUG VISUALIZATION ===")

        except Exception as e:
            print(f"Error in debug visualization: {e}")
            import traceback
            traceback.print_exc()

    def forward(self, data, inputs=None):
        """
        前向传播，计算损失
        """
        # 预处理
        if inputs is None:
            inputs = self.forward_preprocess(data)

        # 强制设备一致性检查和修复
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        expected_device = f"cuda:{local_rank}"

        # 修复 rand_device 设置
        if "rand_device" in inputs:
            inputs["rand_device"] = expected_device

        # 检查并修复所有输入张量的设备
        def move_to_device(obj, device):
            if hasattr(obj, 'to') and hasattr(obj, 'device'):
                return obj.to(device)
            elif isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(item, device) for item in obj)
            elif isinstance(obj, dict):
                return {key: move_to_device(value, device) for key, value in obj.items()}
            else:
                return obj

        try:
            # 将所有输入移动到正确的设备
            inputs = move_to_device(inputs, expected_device)

            # 同时确保数据也在正确的设备上
            data = move_to_device(data, expected_device)
        except Exception as device_move_error:
            print(f"Warning: Could not move inputs to device {expected_device}: {device_move_error}")

        # HARDCODED DEBUG分支 - 手动设置为True时启用可视化
        DEBUG_VISUALIZATION = False  # 手动修改此处为True来启用debug可视化

        if DEBUG_VISUALIZATION:
            print(f"\n🔍 DEBUG MODE ACTIVATED (Step {self.debug_counter})")
            # 可视化 forward 前的输入
            self.visualize_forward_inputs(inputs)
            # 可视化经过处理后的输入（保留原有功能）
            # self.visualize_processed_inputs(inputs, data)
            self.debug_counter += 1

        # 前向传播计算

        # 正常前向传播
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    def log_to_swanlab(self, loss_value, step, swanlab_run=None):
        """
        记录损失到SwanLab
        """
        if swanlab_run is not None and SWANLAB_AVAILABLE:
            try:
                import swanlab
                # 只记录有效的数值，避免None值导致的错误
                log_data = {
                    "train_loss": loss_value,
                    "step": step
                }

                # 如果有学习率方法且返回有效值，才记录学习率
                if hasattr(self, 'get_current_lr'):
                    lr = self.get_current_lr()
                    if lr is not None:
                        log_data["learning_rate"] = lr

                swanlab.log(log_data, step=step)
                print(f"SwanLab logged: step={step}, loss={loss_value:.4f}")
            except Exception as e:
                print(f"Warning: Failed to log to SwanLab: {e}")
                import traceback
                traceback.print_exc()


def launch_optimized_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    args=None,
):
    """
    优化版本的训练任务启动器，专门针对40GB A100进行内存和性能优化
    """
    if args is None:
        raise ValueError("args is required for optimized training")

    # 参数提取
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters
    train_batch_size = args.train_batch_size

    # 只在主进程打印配置信息（避免多进程重复打印）
    # 使用环境变量检查主进程，避免创建临时Accelerator
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process_for_print = local_rank == 0

    # 在主进程中打印配置信息
    if is_main_process_for_print:
        print(f"Training configuration:")
        print(f"  - Batch size per GPU: 1 (Wan model limitation)")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Effective batch size per GPU: {gradient_accumulation_steps}")
        print(f"  - Total effective batch size (all GPUs): {gradient_accumulation_steps} × {os.environ.get('WORLD_SIZE', 1)} = {gradient_accumulation_steps * int(os.environ.get('WORLD_SIZE', 1))}")
        print(f"  - Data workers: {num_workers}")
        print(f"  - Gradient checkpointing: {args.use_gradient_checkpointing}")

    # 优化的数据加载器配置
    # 注意：Wan模型只支持batch_size=1，因为：
    # 1. 每个样本包含完整的视频序列（多帧）
    # 2. pipeline的设计不支持batch处理
    # 3. 使用gradient_accumulation来模拟更大的batch size

    def collate_single_sample(batch):
        """
        Collate function: 只返回第一个样本
        Wan模型不支持batch_size>1
        """
        return batch[0]

    dataloader_kwargs = {
        'batch_size': 1,  # 固定为1，Wan模型限制
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 1 if num_workers > 0 else None,
        'drop_last': True,
        'collate_fn': collate_single_sample,
    }

    if is_main_process_for_print and train_batch_size > 1:
        print(f"ℹ️  Note: train_batch_size={train_batch_size} is ignored.")
        print(f"   Wan model only supports batch_size=1")
        print(f"   Use gradient_accumulation_steps={gradient_accumulation_steps} to simulate larger batches")

    if is_main_process_for_print:
        print(f"DataLoader configuration: {dataloader_kwargs}")

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    # Accelerator配置 - 使用与train.py相同的简单方式
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    if is_main_process_for_print:
        print(f"Accelerator initialized with distributed_type: {accelerator.state.distributed_type}")

    # 优化器配置 - 使用与train.py相同的方式
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # 学习率调度器 - 使用与train.py相同的方式
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # 准备训练组件 - 使用与train.py相同的简单方式
    try:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

        if accelerator.is_main_process:
            print("✅ Model preparation completed successfully")

    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ Error during model preparation: {e}")
        raise e

    # 开始训练循环
    try:
        if accelerator.is_main_process:
            print("🚀 Starting training loop...")
            print(f"📊 Training for {num_epochs} epochs")
            print(f"📊 Dataset size: {len(dataloader)} batches")

        # 训练循环
        for epoch_id in range(num_epochs):
            if accelerator.is_main_process:
                print(f"\n🔄 Starting epoch {epoch_id+1}/{num_epochs}")

            model.train()
            epoch_loss = 0
            step_count = 0

            # 添加进度条
            pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")

            for step, data in enumerate(pbar):
                with accelerator.accumulate(model):
                    # 梯度清零 - 使用与train.py相同的方式
                    optimizer.zero_grad(set_to_none=True)

                    # 强制CUDA同步，确保设备状态一致
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # 前向传播
                    try:
                        if dataset.load_from_cache:
                            loss_dict = model({}, inputs=data)
                        else:
                            loss_dict = model(data)
                    except RuntimeError as e:
                        if "device" in str(e).lower():
                            if accelerator.is_main_process:
                                print(f"Device error at step {step}: {e}")
                                print(f"Current device: {accelerator.device}")
                                print(f"Model device: {next(model.parameters()).device}")
                            raise e
                        else:
                            raise e

                    # 提取loss（支持旧版本返回单个值或新版本返回dict）
                    if isinstance(loss_dict, dict):
                        loss = loss_dict["loss"]
                        loss_rgb = loss_dict.get("loss_rgb", None)
                        loss_heatmap = loss_dict.get("loss_heatmap", None)
                    else:
                        # 向后兼容：如果返回的是单个loss值
                        loss = loss_dict
                        loss_rgb = None
                        loss_heatmap = None

                    # 反向传播
                    accelerator.backward(loss)

                    # 梯度裁剪
                    if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # 优化器步骤 - 使用与train.py相同的方式
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    # 记录损失
                    epoch_loss += loss.item()
                    step_count += 1

                    # SwanLab日志记录 - 只在主进程且满足logging_steps频率时记录
                    global_step = step + epoch_id * len(dataloader)
                    should_log = (accelerator.is_main_process and
                                 hasattr(args, 'swanlab_run') and args.swanlab_run is not None and
                                 hasattr(args, 'logging_steps') and global_step % args.logging_steps == 0)

                    if should_log:
                        print(f"Logging to SwanLab at step {global_step}")
                        # 记录训练指标到SwanLab
                        try:
                            import swanlab
                            # 获取当前学习率 - 使用与train.py相同的方式
                            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

                            log_data = {
                                "train_loss": loss.item(),
                                "learning_rate": current_lr,
                                "epoch": epoch_id,
                                "step": global_step
                            }

                            # 添加RGB和Heatmap loss的记录
                            if loss_rgb is not None:
                                log_data["train_loss_rgb"] = loss_rgb.item()
                            if loss_heatmap is not None:
                                log_data["train_loss_heatmap"] = loss_heatmap.item()

                            swanlab.log(log_data, step=global_step)

                            # 打印日志信息
                            log_msg = f"SwanLab logged: step={global_step}, loss={loss.item():.4f}"
                            if loss_rgb is not None and loss_heatmap is not None:
                                log_msg += f", loss_rgb={loss_rgb.item():.4f}, loss_heatmap={loss_heatmap.item():.4f}"
                            log_msg += f", lr={current_lr:.2e}"
                            print(log_msg)
                        except Exception as e:
                            print(f"Warning: Failed to log to SwanLab: {e}")
                    elif global_step % args.logging_steps == 0:
                        print(f"Step {global_step}: main_process={accelerator.is_main_process}, swanlab_run={args.swanlab_run is not None}")

                    # 更新进度条
                    postfix_dict = {
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss/step_count:.4f}"
                    }
                    # 添加RGB和Heatmap loss到进度条
                    if loss_rgb is not None:
                        postfix_dict['loss_rgb'] = f"{loss_rgb.item():.4f}"
                    if loss_heatmap is not None:
                        postfix_dict['loss_hm'] = f"{loss_heatmap.item():.4f}"
                    pbar.set_postfix(postfix_dict)

                    # 定期保存检查点（只在save_steps > 0时保存）
                    if save_steps > 0:
                        model_logger.on_step_end(accelerator, model, save_steps)

                    # 参数更新检测 - 在训练早期和关键步骤检测
                    # 只在主进程执行，检测步数：1, 10, 50, 100
                    # if global_step in [1, 10, 50, 100]:
                    #     # 获取unwrapped模型以访问check_parameter_updates方法
                    #     unwrapped_model = accelerator.unwrap_model(model)
                    #     if hasattr(unwrapped_model, 'check_parameter_updates'):
                    #         unwrapped_model.check_parameter_updates(global_step)

            # 每个epoch结束时的处理 - 总是保存epoch检查点
            model_logger.on_epoch_end(accelerator, model, epoch_id)

            # 每个epoch结束时检测参数更新
            # 已验证参数会更新，暂时注释掉
            # if accelerator.is_main_process:
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     if hasattr(unwrapped_model, 'check_parameter_updates'):
            #         unwrapped_model.check_parameter_updates(f"Epoch {epoch_id+1} end")

            accelerator.print(f"Epoch {epoch_id+1} completed. Average loss: {epoch_loss/step_count:.4f}")

        # 训练结束处理（只在save_steps > 0时保存最后的step检查点）
        if save_steps > 0:
            model_logger.on_training_end(accelerator, model, save_steps)

        if accelerator.is_main_process:
            print("🎉 Training completed successfully!")

    except Exception as training_error:
        if accelerator.is_main_process:
            print(f"❌ Training failed with error: {training_error}")
            import traceback
            print(f"📊 Full traceback:")
            traceback.print_exc()
        raise training_error

    # 训练完成


def create_heatmap_parser():
    """
    创建热力图训练专用的参数解析器
    """
    import argparse
    parser = argparse.ArgumentParser(description="Heatmap sequence training for Wan2.2")

    # 从wan_parser复制必要的参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="", help="Remove prefix in checkpoint.")
    parser.add_argument("--trainable_models", type=str, default="", help="Trainable models.")
    parser.add_argument("--lora_base_model", type=str, default="", help="LoRA base model.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="LoRA target modules.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_checkpoint", type=str, default="", help="LoRA checkpoint.")
    parser.add_argument("--extra_inputs", type=str, default="", help="Extra inputs.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--dataloader_pin_memory", type=lambda x: x.lower() == 'true', default=True, help="Enable pin memory for dataloader.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--find_unused_parameters", action="store_true", help="Find unused parameters.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Dataset num workers.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--model_paths", type=str, default="", help="Model paths.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default="", help="Model ID with origin paths.")
    parser.add_argument("--height", type=int, default=256, help="Image height.")
    parser.add_argument("--width", type=int, default=256, help="Image width.")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Dataset repeat.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps.")
    parser.add_argument("--wan_type", type=str, default="14B_I2V", help="Type of Wan model (e.g., '14B_I2V', '5B_TI2V', or 'WAN_2_1_14B_I2V')")
    parser.add_argument("--use_dual_head", action="store_true", help="Use dual head mode (separate heads for RGB and Heatmap)")

    # 添加热力图专用参数
    parser.add_argument("--heatmap_data_root", type=str, required=True,
                       help="Root directory containing robot trajectory data")
    parser.add_argument("--sequence_length", type=int, default=10,
                       help="Length of heatmap sequence to predict")
    parser.add_argument("--step_interval", type=int, default=1,
                       help="Interval between trajectory steps")
    parser.add_argument("--min_trail_length", type=int, default=15,
                       help="Minimum trajectory length requirement")
    parser.add_argument("--heatmap_sigma", type=float, default=1.5,
                       help="Standard deviation for Gaussian heatmap generation")
    parser.add_argument("--colormap_name", type=str, default="jet",
                       help="Colormap name for heatmap conversion (统一使用cv2 JET)")
    parser.add_argument("--scene_bounds", type=str,
                       default="0,-0.45,-0.05,0.8,0.55,0.6",
                       help="Scene bounds as comma-separated values")
    parser.add_argument("--transform_augmentation_xyz", type=str,
                       default="0.1,0.1,0.1",
                       help="XYZ augmentation range as comma-separated values")
    parser.add_argument("--transform_augmentation_rpy", type=str,
                       default="0.0,0.0,20.0",
                       help="RPY augmentation range as comma-separated values")
    parser.add_argument("--disable_augmentation", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode (use fewer data)")

    # SwanLab相关参数
    parser.add_argument("--enable_swanlab", action="store_true",
                       help="Enable SwanLab logging")
    parser.add_argument("--swanlab_api_key", type=str, default="h1x6LOLp5qGLTfsPuB7Qw",
                       help="SwanLab API key")
    parser.add_argument("--swanlab_project", type=str, default="wan2.2-heatmap-training",
                       help="SwanLab project name")
    parser.add_argument("--swanlab_experiment", type=str, default="heatmap-lora",
                       help="SwanLab experiment name")

    return parser


def parse_float_list(s: str, name: str):
    """
    解析逗号分隔的浮点数列表
    """
    try:
        return [float(x.strip()) for x in s.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid format for {name}: {s}. Expected comma-separated numbers.") from e


if __name__ == "__main__":
    parser = create_heatmap_parser()
    args = parser.parse_args()

    print("="*60)
    print("HEATMAP SEQUENCE TRAINING FOR WAN2.2")
    print("="*60)
    print(f"Data root: {args.heatmap_data_root}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Colormap: {args.colormap_name}")
    print(f"Output path: {args.output_path}")
    print(f"Debug mode: {args.debug_mode}")
    print(f"SwanLab enabled: {args.enable_swanlab and not args.debug_mode}")
    print("="*60)

    # 初始化SwanLab（只在主进程进行）
    swanlab_run = None

    # 使用分布式环境变量检查是否为主进程，避免创建临时Accelerator
    # 在DeepSpeed环境下，任何Accelerator创建都会触发DeepSpeed初始化
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    print(f"🚀 Process {local_rank}: Starting heatmap training initialization...")
    print(f"   - Local rank: {local_rank}")
    print(f"   - World size: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"   - Process rank: {os.environ.get('RANK', 'Not set')}")

    # 注意：不能在Accelerator初始化前使用dist.barrier()，因为进程组还未初始化

    if is_main_process and args.enable_swanlab and not args.debug_mode and SWANLAB_AVAILABLE:
        try:
            print("Initializing SwanLab...")
            print(f"API Key: {args.swanlab_api_key[:8]}***")
            print(f"Project: {args.swanlab_project}")
            print(f"Experiment: {args.swanlab_experiment}")

            swanlab.login(api_key=args.swanlab_api_key)
            swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment,
                config={
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.num_epochs,
                    "sequence_length": args.sequence_length,
                    "train_batch_size": args.train_batch_size,
                    "lora_rank": args.lora_rank,
                    "heatmap_sigma": args.heatmap_sigma,
                    "colormap_name": args.colormap_name,
                    "height": args.height,
                    "width": args.width,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "logging_steps": args.logging_steps,
                }
            )
            print(f"✅ SwanLab initialized successfully!")
            print(f"   Project: {args.swanlab_project}")
            print(f"   Experiment: {args.swanlab_experiment}")
            print(f"   Logging frequency: every {args.logging_steps} steps")

            # 测试一次日志记录
            swanlab.log({"test": 1.0}, step=0)
            print("✅ Test log sent to SwanLab")

        except Exception as e:
            print(f"❌ Failed to initialize SwanLab: {e}")
            import traceback
            traceback.print_exc()
            swanlab_run = None
    elif is_main_process and args.enable_swanlab and args.debug_mode:
        print("SwanLab disabled in debug mode")
    elif is_main_process and args.enable_swanlab and not SWANLAB_AVAILABLE:
        print("Warning: SwanLab requested but not available. Install with: pip install swanlab")

    # 解析参数
    scene_bounds = parse_float_list(args.scene_bounds, "scene_bounds")
    transform_augmentation_xyz = parse_float_list(args.transform_augmentation_xyz, "transform_augmentation_xyz")
    transform_augmentation_rpy = parse_float_list(args.transform_augmentation_rpy, "transform_augmentation_rpy")

    # 创建热力图数据集
    if is_main_process:
        print("Creating heatmap dataset...")
    try:
        dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
            data_root=args.heatmap_data_root,
            sequence_length=args.sequence_length,
            step_interval=args.step_interval,
            min_trail_length=args.min_trail_length,
            image_size=(args.height, args.width),
            sigma=args.heatmap_sigma,
            augmentation=not args.disable_augmentation,
            mode="train",
            scene_bounds=scene_bounds,
            transform_augmentation_xyz=transform_augmentation_xyz,
            transform_augmentation_rpy=transform_augmentation_rpy,
            debug=args.debug_mode,
            colormap_name=args.colormap_name,
            repeat=args.dataset_repeat,
            wan_type=args.wan_type,
        )

        # 等待所有进程完成数据集创建（在分布式环境下由barrier自动处理）
        pass

        if is_main_process:
            print(f"Dataset created successfully with {len(dataset)} samples")

            # 测试数据加载
            print("Testing data loading...")
            test_sample = dataset[0]
            print(f"Sample keys: {list(test_sample.keys())}")
            print(f"Video frames: {len(test_sample['video'])}")
            print(f"First frame size: {test_sample['video'][0].size}")
            print(f"Prompt: {test_sample['prompt']}")

    except Exception as e:
        if is_main_process:
            print(f"Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
        # 确保所有进程都退出
        exit(1)

    # 创建训练模块
    if is_main_process:
        print("Creating training module...")
    try:
        # 处理空字符串参数，避免导致错误
        model_id_with_origin_paths = args.model_id_with_origin_paths if args.model_id_with_origin_paths else None
        lora_checkpoint = args.lora_checkpoint if args.lora_checkpoint else None
        trainable_models = args.trainable_models if args.trainable_models else None
        lora_base_model = args.lora_base_model if args.lora_base_model else None

        print(f"Initializing training module with wan_type: {args.wan_type}")
        print(f"Model paths: {args.model_paths}")
        print(f"LoRA base model: {args.lora_base_model}")

        # 初始化训练模型
        print(f"Process {local_rank}: Starting model creation...")
        print(f"Use dual head mode: {args.use_dual_head}")
        model = HeatmapWanTrainingModule(
            wan_type=args.wan_type,
            model_paths=args.model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=lora_checkpoint,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
            use_dual_head=args.use_dual_head,
        )

        print(f"Process {local_rank}: Model creation completed successfully!")

        if is_main_process:
            print("Training module created successfully")

    except Exception as e:
        if is_main_process:
            print(f"Error creating training module: {e}")
            import traceback
            traceback.print_exc()
        # 确保所有进程都退出
        exit(1)

    # 创建模型日志器
    if is_main_process:
        print("Setting up model logger...")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    # 启动训练
    if is_main_process:
        print("Starting training...")
        print(f"Optimizing data loading with {args.dataset_num_workers} workers...")
    try:
        # 如果SwanLab可用，添加到args中
        if swanlab_run is not None:
            args.swanlab_run = swanlab_run
        else:
            args.swanlab_run = None

        # 使用优化版本的训练函数，它有更好的错误处理
        launch_optimized_training_task(dataset, model, model_logger, args=args)

        if is_main_process:
            print("Training completed successfully!")

        # 结束SwanLab实验
        if swanlab_run is not None:
            swanlab_run.finish()
            if is_main_process:
                print("SwanLab experiment finished")

    except Exception as e:
        if is_main_process:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

        # 确保SwanLab实验正确结束
        if swanlab_run is not None:
            swanlab_run.finish()

        exit(1)