"""
Heatmap Training Script for Wan2.2
基于原始train.py，专门用于热力图序列生成训练 且支持多视角
"""

import torch
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn

# 强制刷新输出，确保错误信息能被显示
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# SwanLab导入（可选）
try:
    import swanlab
    SWANLAB_AVAILABLE = True
    print(f"✓ SwanLab imported successfully (version: {swanlab.__version__})")
except ImportError as e:
    SWANLAB_AVAILABLE = False
    print(f"Warning: SwanLab not available. Install with: pip install swanlab")
    print(f"ImportError details: {e}")
    assert False
except Exception as e:
    SWANLAB_AVAILABLE = False
    print(f"Warning: SwanLab import failed with error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import torch.utils.data
from torch.utils.data import ConcatDataset

# 添加trainers路径以导入我们的自定义数据集
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip import HeatmapDatasetFactory
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam import HeatmapDatasetFactory
# 多帧历史支持的数据集和工厂
from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip_3cam_history import HeatmapDatasetFactoryWithHistory

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
        unfreeze_modulation_and_norms=False,  # 新增：控制是否解冻 modulation 和 norm 参数
        num_history_frames=1,  # 历史帧数量，用于多帧条件模式
        rgb_loss_weight=0.5,  # RGB loss权重，总loss = rgb_loss_weight * loss_rgb + (1 - rgb_loss_weight) * loss_heatmap
    ):
        super().__init__()

        # 保存配置参数
        self.unfreeze_modulation_and_norms = unfreeze_modulation_and_norms
        self.num_history_frames = num_history_frames
        self.rgb_loss_weight = rgb_loss_weight

        # Debug: 显示预训练checkpoint配置
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print("\n" + "="*80)
            print("🚀 INITIALIZING TRAINING MODULE")
            print("="*80)
            print(f"  WAN Type: {wan_type}")
            print(f"  LoRA Base Model: {lora_base_model}")
            print(f"  LoRA Target Modules: {lora_target_modules}")
            print(f"  LoRA Rank: {lora_rank}")
            print(f"  Use Dual Head: {use_dual_head}")
            print(f"  RGB Loss Weight: {rgb_loss_weight}")
            print(f"  Unfreeze Modulation & Norms: {unfreeze_modulation_and_norms}")
            if lora_checkpoint is not None:
                print(f"\n  📦 PRETRAINED CHECKPOINT SPECIFIED:")
                print(f"     Path: {lora_checkpoint}")
                if os.path.exists(lora_checkpoint):
                    file_size_mb = os.path.getsize(lora_checkpoint) / (1024 * 1024)
                    print(f"     ✓ File exists, size: {file_size_mb:.2f} MB")
                else:
                    print(f"     ✗ WARNING: File does not exist!")
            else:
                print(f"\n  ℹ️  No pretrained checkpoint specified (training from scratch)")
            print("="*80 + "\n")

        self.wan_type = wan_type
        if self.wan_type == "WAN_2_1_14B_I2V":
            from diffsynth.pipelines.wan_video_14BI2V_condition_rgb_heatmap_first import WanVideoPipeline, ModelConfig
        elif self.wan_type == "5B_TI2V":
            from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP":
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb import WanVideoPipeline, ModelConfig
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP_MV":
            # View concatenation mode: uses view-concat pipeline
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_view import WanVideoPipeline, ModelConfig
            from diffsynth.models.wan_video_dit_mv import SelfAttention
        elif self.wan_type == "5B_TI2V_RGB_HEATMAP_MV_HISTORY":
            # 多帧历史条件的pipeline，使用修改后的VAE
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_history import WanVideoPipeline, ModelConfig
            from diffsynth.models.wan_video_dit_mv import SelfAttention
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

        if self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV", "5B_TI2V_RGB_HEATMAP_MV_HISTORY"]:
            # add mv attention module - 确保dtype一致
            dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            # 获取模型的dtype和device，确保新添加的模块与现有模型一致
            model_dtype = self.pipe.dit.blocks[0].self_attn.q.weight.dtype
            model_device = self.pipe.dit.blocks[0].self_attn.q.weight.device

            if local_rank == 0:
                print(f"[DEBUG] Adding multi-view attention modules with dtype={model_dtype}, device={model_device}")

            for block in self.pipe.dit.blocks:
                # 创建projector并转换到正确的dtype和device
                block.projector = nn.Linear(dim, dim).to(dtype=model_dtype, device=model_device)
                block.projector.weight.data.zero_()
                block.projector.bias.data.zero_()

                # 创建norm_mvs并转换dtype
                block.norm_mvs = nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False).to(dtype=model_dtype, device=model_device)

                # 创建modulation_mvs参数，使用正确的dtype
                block.modulation_mvs = nn.Parameter(torch.randn(1, 3, dim, dtype=model_dtype, device=model_device) / dim**0.5)
                block.modulation_mvs.data = block.modulation.data[:, :3, :].clone()

                # 创建mvs_attn，先加载权重，再转换dtype
                block.mvs_attn = SelfAttention(dim, block.self_attn.num_heads, block.self_attn.norm_q.eps)
                block.mvs_attn.load_state_dict(block.self_attn.state_dict(), strict=True)
                block.mvs_attn = block.mvs_attn.to(dtype=model_dtype, device=model_device)
        

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
        # 解冻多视角模块的所有参数（仅在MV模式下）
        # 这些模块包括：projector, norm_mvs, modulation_mvs, mvs_attn
        # ============================================================
        if self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV", "5B_TI2V_RGB_HEATMAP_MV_HISTORY"]:
            self._unfreeze_mv_modules()

        # ============================================================
        # 解冻 modulation 参数（可选，由 unfreeze_modulation_and_norms 控制）
        # AdaLN 调制参数，用于根据时间步 t 调整归一化
        # ============================================================
        self._unfreeze_modulation()

        # ============================================================
        # 参数监控：打印可训练参数信息
        # ============================================================
        self._print_trainable_parameters_info()


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

        在双head模式下：
        - 解冻 head_rgb 和 head_heatmap (而不是 head)
        - 解冻 patch_embedding_rgb 和 patch_embedding_heatmap (而不是 patch_embedding)
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []
        all_head_params = []  # 用于调试：所有包含head的参数
        all_patch_embedding_params = []  # 用于调试：所有patch_embedding参数

        for name, param in self.pipe.dit.named_parameters():
            # 调试：收集所有包含head的参数名
            if 'head' in name:
                all_head_params.append((name, param.requires_grad))
            # 调试：收集所有patch_embedding的参数名
            if 'patch_embedding' in name:
                all_patch_embedding_params.append((name, param.requires_grad))

            # 检查是否是 patch_embedding 或 head 的参数
            # 在双head模式下：
            #   - 匹配 head_rgb 和 head_heatmap
            #   - 匹配 patch_embedding_rgb 和 patch_embedding_heatmap
            # 在单head模式下：
            #   - 匹配 head
            #   - 匹配 patch_embedding
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
            mode_str = "DUAL HEAD + DUAL PATCH_EMBEDDING" if self.use_dual_head else "SINGLE HEAD + SINGLE PATCH_EMBEDDING"
            print(f"FULL PARAMETER UNFREEZING FOR PATCH_EMBEDDING AND HEAD ({mode_str} MODE)")
            print("="*80)

            # 调试信息：打印所有patch_embedding的参数
            print(f"\n[DEBUG] All parameters with 'patch_embedding' in name ({len(all_patch_embedding_params)} total):")
            for name, requires_grad in all_patch_embedding_params:
                print(f"  - {name}: requires_grad={requires_grad}")

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

    def _unfreeze_modulation(self):
        """
        解冻 modulation 参数（AdaLN 调制参数）

        Modulation 参数在 DiT 中用于 Adaptive Layer Normalization (AdaLN)：
        - 根据时间步 t 动态调整每层的 scale 和 shift
        - 对于不同任务（如机械臂轨迹预测 vs 通用视频生成），时间步的语义可能不同
        - 参数量很小（每个 block 约 18K 参数），但影响很大

        只有当 self.unfreeze_modulation_and_norms=True 时才解冻
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if not self.unfreeze_modulation_and_norms:
            if local_rank == 0:
                print("\n" + "="*80)
                print("ℹ️  MODULATION PARAMETERS: KEEPING FROZEN (Backward Compatible)")
                print("="*80)
                print("  Modulation parameters will stay frozen (using pretrained values).")
                print("  To unfreeze, add --unfreeze_modulation_and_norms flag.")
                print("="*80 + "\n")
            return

        unfrozen_params = []

        for name, param in self.pipe.dit.named_parameters():
            # 只解冻 modulation 参数（不包括 modulation_mvs，它已在 _unfreeze_mv_modules 中处理）
            if 'modulation' in name and 'modulation_mvs' not in name and 'blocks.' in name:
                param.requires_grad = True
                unfrozen_params.append(name)

        # 只在主进程打印
        if local_rank == 0:
            print("\n" + "="*80)
            print("⚡ MODULATION PARAMETER UNFREEZING")
            print("="*80)

            if len(unfrozen_params) > 0:
                print(f"\n✅ Unfroze {len(unfrozen_params)} modulation parameters:")
                print(f"  📦 AdaLN modulation: {len(unfrozen_params)} parameters")

                # 计算总参数量
                total_modulation_params = sum(
                    self.pipe.dit.state_dict()[name].numel()
                    for name in unfrozen_params
                )
                print(f"  📊 Total: {total_modulation_params:,} parameters (~{total_modulation_params/1e6:.2f}M)")

                print("\n💡 训练策略:")
                print("  - Modulation: 全量训练（适应新任务的时间步调制模式）")
            else:
                print("\n⚠️  WARNING: No modulation parameters were unfrozen!")
            print("="*80 + "\n")

    def _unfreeze_mv_modules(self):
        """
        解冻多视角（Multi-View）模块的参数

        这些模块在第86-97行被添加到每个DiT block中：
        - projector: Linear层，用于多视角特征投影 (全量训练)
        - norm_mvs: LayerNorm层，用于多视角特征归一化 (无可训练参数)
        - modulation_mvs: 调制参数 (全量训练)
        - mvs_attn: SelfAttention层，用于多视角注意力计算 (LoRA训练)

        训练策略：
        - projector, modulation_mvs: 全量训练（解冻所有权重和bias）
        - mvs_attn.norm_q/norm_k: 可选全量训练（如果 unfreeze_modulation_and_norms=True）
        - mvs_attn: 只保持 LoRA 参数可训练，原始权重保持冻结
          这样可以减少显存占用，同时利用预训练知识
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []
        all_mv_params = []  # 用于调试：所有 MV 相关的参数

        for name, param in self.pipe.dit.named_parameters():
            # 检查是否是多视角相关的参数
            is_projector = 'projector' in name and 'blocks.' in name
            is_modulation_mvs = 'modulation_mvs' in name
            is_mvs_attn = 'mvs_attn' in name

            # 收集所有 MV 相关参数用于调试
            if is_projector or is_modulation_mvs or is_mvs_attn:
                all_mv_params.append((name, param.requires_grad))

            # 排除 LoRA 参数（lora_A/lora_B 已经在添加时设为可训练）
            is_not_lora = ('lora_A' not in name and 'lora_B' not in name)

            # 解冻策略：
            # 1. projector 和 modulation_mvs: 全量训练（所有参数）
            # 2. mvs_attn: LoRA 参数 + 可选的 norm_q/norm_k
            if (is_projector or is_modulation_mvs) and is_not_lora:
                # projector 和 modulation_mvs 全量训练
                param.requires_grad = True
                unfrozen_params.append(name)
            elif is_mvs_attn:
                # mvs_attn 的 norm_q/norm_k（如果启用）
                if self.unfreeze_modulation_and_norms and is_not_lora:
                    if 'norm_q' in name or 'norm_k' in name:
                        param.requires_grad = True
                        unfrozen_params.append(name)
                # mvs_attn 的 LoRA 参数保持可训练（已经在添加 LoRA 时设置）
                elif not is_not_lora:
                    unfrozen_params.append(name)

        # 只在主进程打印
        if local_rank == 0:
            print("\n" + "="*80)
            print("MULTI-VIEW MODULE PARAMETER UNFREEZING")
            print("="*80)

            # 调试信息：打印所有 MV 相关参数
            print(f"\n[DEBUG] All MV-related parameters ({len(all_mv_params)} total):")
            for name, requires_grad in all_mv_params[:10]:  # 只显示前10个
                print(f"  - {name}: requires_grad={requires_grad}")
            if len(all_mv_params) > 10:
                print(f"  ... and {len(all_mv_params) - 10} more")

            if len(unfrozen_params) > 0:
                print(f"\n✅ Unfroze {len(unfrozen_params)} parameter(s):")

                # 按模块类型分组统计
                projector_params = [n for n in unfrozen_params if 'projector' in n]
                modulation_mvs_params = [n for n in unfrozen_params if 'modulation_mvs' in n]
                mvs_attn_lora_params = [n for n in unfrozen_params if 'mvs_attn' in n and 'lora' in n]
                mvs_attn_norm_params = [n for n in unfrozen_params if 'mvs_attn' in n and ('norm_q' in n or 'norm_k' in n)]

                print(f"\n  📦 Projector (全量训练): {len(projector_params)} parameters")
                print(f"  📦 Modulation_mvs (全量训练): {len(modulation_mvs_params)} parameters")
                print(f"  📦 MVS_Attn LoRA (LoRA训练): {len(mvs_attn_lora_params)} parameters")
                if mvs_attn_norm_params:
                    print(f"  📦 MVS_Attn Norms (全量训练): {len(mvs_attn_norm_params)} parameters")

                print("\n💡 训练策略:")
                print("  - Projector & Modulation_mvs: 全量训练（从零开始学习）")
                print("  - MVS_Attn: LoRA微调（利用预训练知识，节省显存）")
                if mvs_attn_norm_params:
                    print("  - MVS_Attn norm_q/norm_k: 全量训练（适应多视角特征分布）")
            else:
                print("\n⚠️  WARNING: No MV module parameters were unfrozen!")
                print("    This might indicate that the MV modules were not properly initialized.")
            print("="*80 + "\n")

    def _print_trainable_parameters_info(self):
        """
        打印可训练参数的详细信息  
        特别关注 patch_embedding 和 head 的参数状态
        只在主进程打印
        """
        # debug 打印多视角模块到底是lora在调还是全量训练。如果是lora调的话，那么可训练参数应该占总量的比例非常少？
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
        mv_module_params = {}  # 新增：多视角模块参数
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
                # 新增：检查是否是MV模块参数
                elif any(['projector' in name and 'blocks.' in name, 'norm_mvs' in name,
                         'modulation_mvs' in name, 'mvs_attn' in name]):
                    mv_module_params[name] = {
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

        # 打印 MV 模块参数（仅在MV模式下）
        if self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV", "5B_TI2V_RGB_HEATMAP_MV_HISTORY"]:
            print(f"\n🔍 MULTI-VIEW MODULE Parameters:")
            if mv_module_params:
                mv_param_count = sum(info['numel'] for info in mv_module_params.values())
                print(f"  ✅ {len(mv_module_params)} trainable parameter(s) found ({mv_param_count:,} total, {mv_param_count/1e6:.2f}M):")

                # 按模块类型统计
                projector_count = sum(info['numel'] for name, info in mv_module_params.items() if 'projector' in name)
                norm_mvs_count = sum(info['numel'] for name, info in mv_module_params.items() if 'norm_mvs' in name)
                modulation_mvs_count = sum(info['numel'] for name, info in mv_module_params.items() if 'modulation_mvs' in name)
                mvs_attn_count = sum(info['numel'] for name, info in mv_module_params.items() if 'mvs_attn' in name)

                print(f"    📦 Projector: {projector_count:,} params ({projector_count/1e6:.2f}M)")
                print(f"    📦 Norm_mvs: {norm_mvs_count:,} params ({norm_mvs_count/1e6:.2f}M)")
                print(f"    📦 Modulation_mvs: {modulation_mvs_count:,} params ({modulation_mvs_count/1e6:.2f}M)")
                print(f"    📦 MVS_Attn: {mvs_attn_count:,} params ({mvs_attn_count/1e6:.2f}M)")

                # 显示前3个参数示例
                print(f"\n  Sample parameters:")
                for i, (name, info) in enumerate(list(mv_module_params.items())[:3]):
                    print(f"    - {name}")
                    print(f"      Shape: {info['shape']}, Count: {info['numel']:,}")
                if len(mv_module_params) > 3:
                    print(f"    ... and {len(mv_module_params) - 3} more MV parameters")
            else:
                print(f"  ❌ NO trainable parameters found in MV modules!")
                print(f"     → MV module parameters are FROZEN or not initialized")

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
            "input_video": data["video"], # ( T,num_view) List of PIL list
            "height": data["video"][0][0].size[1], # 注意区分高和宽
            "width": data["video"][0][0].size[0],
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
            # View concatenation mode: num_view = num_heatmap_views + num_rgb_views
            # When we have both RGB and Heatmap: num_view = 3 + 3 = 6
            # When we only have Heatmap: num_view = 3
            "num_view": len(data["video"][0]) + (len(data.get("input_video_rgb", [[]])[0]) if "input_video_rgb" in self.extra_inputs else 0),
            "num_history_frames": self.num_history_frames,  # 历史帧数量
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

        # 多帧历史支持：添加 input_images 和 input_images_rgb
        # 这些字段在多历史帧模式下由数据准备函数生成
        if "input_images" in data:
            inputs_shared["input_images"] = data["input_images"]
        if "input_images_rgb" in data:
            inputs_shared["input_images_rgb"] = data["input_images_rgb"]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def visualize_forward_inputs(self, inputs):
        """
        可视化 forward 前的输入数据（支持多视角）
        包括: input_image, input_image_rgb, input_video, input_video_rgb

        数据结构：
        - input_image: 列表，包含N个视角的PIL图像 [view1, view2, ...]
        - input_image_rgb: 列表，包含N个视角的PIL图像 [view1, view2, ...]
        - input_video: 双层列表 [time][view]，每个时间步包含多个视角
        - input_video_rgb: 双层列表 [time][view]，每个时间步包含多个视角

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
            print(f"FORWARD INPUT VISUALIZATION - MULTI-VIEW (Step: {self.debug_counter})")
            print(f"{'='*80}")
            print(f"Available keys: {list(inputs.keys())}")

            # 收集所有要可视化的数据
            input_image = inputs.get('input_image')
            input_image_rgb = inputs.get('input_image_rgb')
            input_video = inputs.get('input_video')
            input_video_rgb = inputs.get('input_video_rgb')

            # 检测数据结构
            num_views = 0
            num_frames = 0

            # 检测视角数量
            if input_image and isinstance(input_image, list):
                num_views = len(input_image)
                print(f"✓ input_image: {num_views} views")
            elif input_image_rgb and isinstance(input_image_rgb, list):
                num_views = len(input_image_rgb)
                print(f"✓ input_image_rgb: {num_views} views")

            # 检测帧数
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                num_frames = len(input_video)
                if isinstance(input_video[0], list):
                    num_views = max(num_views, len(input_video[0]))
                    print(f"✓ input_video: {num_frames} frames × {len(input_video[0])} views")
                else:
                    print(f"✓ input_video: {num_frames} frames (single view)")

            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                if isinstance(input_video_rgb[0], list):
                    print(f"✓ input_video_rgb: {len(input_video_rgb)} frames × {len(input_video_rgb[0])} views")
                else:
                    print(f"✓ input_video_rgb: {len(input_video_rgb)} frames (single view)")

            # 计算布局
            # 每个section的列数 = max(num_views, 3)
            cols = max(num_views, 3)

            # 动态计算行数
            rows_list = []

            # 第1行：标题
            rows_list.append(('title_img', 0.3))

            # 第2-3行：input_image 和 input_image_rgb
            if input_image or input_image_rgb:
                rows_list.append(('input_image', 1.5))
                rows_list.append(('input_image_rgb', 1.5))

            # 第4行：视频标题
            if input_video or input_video_rgb:
                rows_list.append(('title_video', 0.3))

            # 第5-N行：input_video (每帧一行)
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                for _ in range(min(num_frames, 15)):  # 最多显示4帧
                    rows_list.append(('video_frame', 1.2))

            # 第M行：RGB视频标题
            if input_video_rgb:
                rows_list.append(('title_video_rgb', 0.3))

            # 第M+1-K行：input_video_rgb (每帧一行)
            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                for _ in range(min(len(input_video_rgb), 15)):  # 最多显示4帧
                    rows_list.append(('video_rgb_frame', 1.2))

            # 创建figure
            total_rows = len(rows_list)
            height_ratios = [r[1] for r in rows_list]
            fig_height = sum(height_ratios) * 2.5

            fig = plt.figure(figsize=(2.5*cols, fig_height))
            gs = fig.add_gridspec(total_rows, cols, hspace=0.3, wspace=0.1, height_ratios=height_ratios)

            current_row = 0

            # ============ 标题：Input Images ============
            if rows_list[current_row][0] == 'title_img':
                ax_title = fig.add_subplot(gs[current_row, :])
                ax_title.text(0.5, 0.5, 'Input Images (Multi-View)', ha='center', va='center',
                             fontsize=14, fontweight='bold', transform=ax_title.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
                ax_title.axis('off')
                current_row += 1

            # ============ input_image (多视角) ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'input_image':
                if input_image and isinstance(input_image, list):
                    for i, img in enumerate(input_image):
                        if i < cols:
                            ax = fig.add_subplot(gs[current_row, i])
                            if hasattr(img, 'save'):
                                img_array = np.array(img)
                                ax.imshow(img_array)
                                ax.set_title(f'Heatmap View {i}', fontsize=10, fontweight='bold')
                                if i == 0:
                                    print(f"  View {i}: Size {img.size}, Mode {img.mode}")
                            else:
                                ax.text(0.5, 0.5, f'View {i}\nInvalid', ha='center', va='center',
                                       fontsize=9, transform=ax.transAxes)
                            ax.axis('off')
                    # 填充空白
                    for i in range(len(input_image), cols):
                        ax = fig.add_subplot(gs[current_row, i])
                        ax.axis('off')
                else:
                    ax = fig.add_subplot(gs[current_row, :])
                    ax.text(0.5, 0.5, 'No input_image', ha='center', va='center',
                           fontsize=10, transform=ax.transAxes)
                    ax.axis('off')
                current_row += 1

            # ============ input_image_rgb (多视角) ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'input_image_rgb':
                if input_image_rgb and isinstance(input_image_rgb, list):
                    for i, img in enumerate(input_image_rgb):
                        if i < cols:
                            ax = fig.add_subplot(gs[current_row, i])
                            if hasattr(img, 'save'):
                                img_array = np.array(img)
                                ax.imshow(img_array)
                                ax.set_title(f'RGB View {i}', fontsize=10, fontweight='bold')
                                if i == 0:
                                    print(f"  View {i}: Size {img.size}, Mode {img.mode}")
                            else:
                                ax.text(0.5, 0.5, f'View {i}\nInvalid', ha='center', va='center',
                                       fontsize=9, transform=ax.transAxes)
                            ax.axis('off')
                    # 填充空白
                    for i in range(len(input_image_rgb), cols):
                        ax = fig.add_subplot(gs[current_row, i])
                        ax.axis('off')
                else:
                    ax = fig.add_subplot(gs[current_row, :])
                    ax.text(0.5, 0.5, 'No input_image_rgb', ha='center', va='center',
                           fontsize=10, transform=ax.transAxes)
                    ax.axis('off')
                current_row += 1

            # ============ 视频标题 ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'title_video':
                ax_title = fig.add_subplot(gs[current_row, :])
                ax_title.text(0.5, 0.5, 'Input Video (Heatmap Sequence, Multi-View)', ha='center', va='center',
                             fontsize=14, fontweight='bold', transform=ax_title.transAxes,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax_title.axis('off')
                current_row += 1

            # ============ input_video (多视角×多帧) ============
            if input_video and isinstance(input_video, list) and len(input_video) > 0:
                frames_to_show = min(len(input_video), 15)
                for frame_idx in range(frames_to_show):
                    if current_row < len(rows_list) and rows_list[current_row][0] == 'video_frame':
                        frame_data = input_video[frame_idx]

                        # 检查是否为多视角
                        if isinstance(frame_data, list):
                            # 多视角模式
                            for view_idx, img in enumerate(frame_data):
                                if view_idx < cols:
                                    ax = fig.add_subplot(gs[current_row, view_idx])
                                    if hasattr(img, 'save'):
                                        img_array = np.array(img)
                                        ax.imshow(img_array)
                                        ax.set_title(f'T{frame_idx}V{view_idx}', fontsize=8)
                                    ax.axis('off')
                            # 填充空白
                            for view_idx in range(len(frame_data), cols):
                                ax = fig.add_subplot(gs[current_row, view_idx])
                                ax.axis('off')
                        else:
                            # 单视角模式
                            ax = fig.add_subplot(gs[current_row, 0])
                            if hasattr(frame_data, 'save'):
                                img_array = np.array(frame_data)
                                ax.imshow(img_array)
                                ax.set_title(f'Frame {frame_idx}', fontsize=9)
                            ax.axis('off')
                            # 填充剩余列
                            for i in range(1, cols):
                                ax = fig.add_subplot(gs[current_row, i])
                                ax.axis('off')

                        current_row += 1

            # ============ RGB视频标题 ============
            if current_row < len(rows_list) and rows_list[current_row][0] == 'title_video_rgb':
                ax_title = fig.add_subplot(gs[current_row, :])
                ax_title.text(0.5, 0.5, 'Input Video RGB (RGB Sequence, Multi-View)', ha='center', va='center',
                             fontsize=14, fontweight='bold', transform=ax_title.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                ax_title.axis('off')
                current_row += 1

            # ============ input_video_rgb (多视角×多帧) ============
            if input_video_rgb and isinstance(input_video_rgb, list) and len(input_video_rgb) > 0:
                frames_to_show = min(len(input_video_rgb), 15)
                for frame_idx in range(frames_to_show):
                    if current_row < len(rows_list) and rows_list[current_row][0] == 'video_rgb_frame':
                        frame_data = input_video_rgb[frame_idx]

                        # 检查是否为多视角
                        if isinstance(frame_data, list):
                            # 多视角模式
                            for view_idx, img in enumerate(frame_data):
                                if view_idx < cols:
                                    ax = fig.add_subplot(gs[current_row, view_idx])
                                    if hasattr(img, 'save'):
                                        img_array = np.array(img)
                                        ax.imshow(img_array)
                                        ax.set_title(f'T{frame_idx}V{view_idx}', fontsize=8)
                                    ax.axis('off')
                            # 填充空白
                            for view_idx in range(len(frame_data), cols):
                                ax = fig.add_subplot(gs[current_row, view_idx])
                                ax.axis('off')
                        else:
                            # 单视角模式
                            ax = fig.add_subplot(gs[current_row, 0])
                            if hasattr(frame_data, 'save'):
                                img_array = np.array(frame_data)
                                ax.imshow(img_array)
                                ax.set_title(f'Frame {frame_idx}', fontsize=9)
                            ax.axis('off')
                            # 填充剩余列
                            for i in range(1, cols):
                                ax = fig.add_subplot(gs[current_row, i])
                                ax.axis('off')

                        current_row += 1

            # 保存组合图
            save_path = os.path.join(self.debug_save_dir, f"step_{self.debug_counter:04d}_all_inputs_multiview.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print("\n✅ Multi-view inputs visualization saved to:")
            print(f"   {save_path}")

            # 打印其他关键信息
            print("\nOther inputs:")
            for key in ['prompt', 'height', 'width', 'num_frames', 'num_view']:
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
        DEBUG_VISUALIZATION = False # 手动修改此处为True来启用debug可视化
        # import pdb;pdb.set_trace()
        if DEBUG_VISUALIZATION:
            print(f"\n🔍 DEBUG MODE ACTIVATED (Step {self.debug_counter})")
            # 可视化 forward 前的输入
            self.visualize_forward_inputs(inputs)
            import pdb;pdb.set_trace()
            # 可视化经过处理后的输入（保留原有功能）
            # self.visualize_processed_inputs(inputs, data)
            self.debug_counter += 1

        # 前向传播计算

        # 正常前向传播
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs, rgb_loss_weight=self.rgb_loss_weight)
        return loss

    def log_to_swanlab(self, loss_value, step, swanlab_run=None):
        """
        记录损失到SwanLab
        """
        if swanlab_run is not None and SWANLAB_AVAILABLE:
            try:
                # swanlab 已经在文件顶部导入，不需要重复导入
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
    save_epochs_interval = getattr(args, 'save_epochs_interval', 0)  # 默认每个epoch都保存
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters
    train_batch_size = args.train_batch_size

    # 只在主进程打印配置信息（避免多进程重复打印）
    # 使用全局RANK环境变量检查主进程（多机训练时每个节点都有LOCAL_RANK=0）
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    is_main_process_for_print = global_rank == 0  # 只有全局RANK=0才打印

    # 在主进程中打印配置信息
    if is_main_process_for_print:
        print(f"Training configuration:")
        print(f"  - Batch size per GPU: 1 (Wan model limitation)")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Effective batch size per GPU: {gradient_accumulation_steps}")
        print(f"  - Total effective batch size (all GPUs): {gradient_accumulation_steps} × {os.environ.get('WORLD_SIZE', 1)} = {gradient_accumulation_steps * int(os.environ.get('WORLD_SIZE', 1))}")
        print(f"  - Data workers: {num_workers}")
        print(f"  - Gradient checkpointing: {args.use_gradient_checkpointing}")
        print(f"  - Save epochs interval: {save_epochs_interval} (0=every epoch)")

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

    # 显示多机分布式训练的详细信息
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("📊 DISTRIBUTED TRAINING CONFIGURATION")
        print("="*80)
        print(f"  Distributed type: {accelerator.state.distributed_type}")
        print(f"  Total GPUs (num_processes): {accelerator.num_processes}")
        print(f"  Current process index: {accelerator.process_index}")
        print(f"  Local process index: {accelerator.local_process_index}")
        print(f"  Device: {accelerator.device}")

        # 从环境变量获取预期的配置
        expected_num_machines = int(os.environ.get('NUM_MACHINES', os.environ.get('WORLD_SIZE', '1')))
        num_gpus_per_node = int(os.environ.get('NUM_GPUS_PER_NODE', '8'))
        expected_total_gpus = expected_num_machines * num_gpus_per_node

        print(f"\n  Expected configuration:")
        print(f"    - Machines: {expected_num_machines}")
        print(f"    - GPUs per machine: {num_gpus_per_node}")
        print(f"    - Expected total GPUs: {expected_total_gpus}")

        print(f"\n  🎯 Actual total GPUs being used: {accelerator.num_processes}")

        if accelerator.num_processes < expected_total_gpus:
            print(f"  ⚠️  WARNING: Using {accelerator.num_processes} GPUs, but expected {expected_total_gpus}!")
            print(f"     Ensure you've started the script on ALL {expected_num_machines} machines")
            print(f"     with NODE_RANK values from 0 to {expected_num_machines-1}.")
        elif accelerator.num_processes == expected_total_gpus:
            print(f"  ✅ All expected GPUs are active!")
        print("="*80 + "\n")

    # 初始化SwanLab（只在主进程进行，Accelerator初始化后）
    # 使用 accelerator.is_main_process 确保只有全局主进程初始化SwanLab
    if accelerator.is_main_process and args.enable_swanlab and not args.debug_mode and SWANLAB_AVAILABLE:
        try:
            print("\n" + "="*80)
            print("🦢 INITIALIZING SWANLAB")
            print("="*80)
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
                    "num_machines": accelerator.num_processes // accelerator.state.num_processes,
                    "num_gpus": accelerator.num_processes,
                }
            )
            print(f"✅ SwanLab initialized successfully!")
            print(f"   Project: {args.swanlab_project}")
            print(f"   Experiment: {args.swanlab_experiment}")
            print(f"   Logging frequency: every {args.logging_steps} steps")

            # 测试一次日志记录
            swanlab.log({"test": 1.0}, step=0)
            print("✅ Test log sent to SwanLab")
            print("="*80 + "\n")

        except Exception as e:
            print(f"❌ Failed to initialize SwanLab: {e}")
            import traceback
            traceback.print_exc()
            swanlab_run = None
    elif accelerator.is_main_process and args.enable_swanlab and args.debug_mode:
        print("\n⚠️  SwanLab disabled in debug mode\n")
    elif accelerator.is_main_process and args.enable_swanlab and not SWANLAB_AVAILABLE:
        print("\n⚠️  Warning: SwanLab requested but not available. Install with: pip install swanlab\n")

    # 将SwanLab run对象设置到args中，这样训练循环可以访问它
    # 注意：swanlab_run在上面的if-elif块中初始化，如果没有初始化则为None
    if 'swanlab_run' not in locals():
        swanlab_run = None
    args.swanlab_run = swanlab_run

    # 更新 is_main_process 为 Accelerator 提供的准确值
    # 这样后续代码可以继续使用 is_main_process 变量
    is_main_process = accelerator.is_main_process

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
                        # 检查 load_from_cache 属性
                        # 对于 ConcatDataset，检查第一个子数据集的属性
                        if isinstance(dataset, ConcatDataset):
                            load_from_cache = getattr(dataset.datasets[0], 'load_from_cache', False)
                        else:
                            load_from_cache = getattr(dataset, 'load_from_cache', False)

                        if load_from_cache:
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
                            # swanlab 已经在文件顶部导入，不需要重复导入
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

            # 每个epoch结束时的处理 - 根据save_epochs_interval决定是否保存
            # save_epochs_interval=0 表示每个epoch都保存
            # save_epochs_interval=N 表示每N个epoch保存一次
            should_save_epoch = (save_epochs_interval == 0) or ((epoch_id + 1) % save_epochs_interval == 0) or (epoch_id == num_epochs - 1)
            if should_save_epoch:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
                if accelerator.is_main_process:
                    print(f"✅ Saved checkpoint at epoch {epoch_id + 1}")

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
    parser.add_argument("--save_epochs_interval", type=int, default=0, help="Save model every N epochs. 0 means save every epoch.")
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
    parser.add_argument("--rgb_loss_weight", type=float, default=0.5,
                       help="Weight for RGB loss in total loss. Total loss = rgb_loss_weight * loss_rgb + (1 - rgb_loss_weight) * loss_heatmap. Range: 0.0 ~ 1.0 (default: 0.5)")
    parser.add_argument("--use_merged_pointcloud", action="store_true", help="Use merged pointcloud from 3 cameras (default: False, only use camera 1)")
    parser.add_argument("--use_different_projection", action="store_true", help="Use different projection mode (base_multi_view_dataset_with_rot_grip_3cam_different_projection.py)")

    # 高级训练参数：控制是否解冻 modulation 和 norm 参数
    parser.add_argument("--unfreeze_modulation_and_norms", action="store_true",
                       help="Unfreeze modulation and RMSNorm parameters (modulation, mvs_attn.norm_q/norm_k). "
                            "Default: False (keep frozen for backward compatibility with existing checkpoints). "
                            "Set to True when training new models for better adaptation.")

    # 添加热力图专用参数
    parser.add_argument("--heatmap_data_root", type=str, nargs='+', required=True,
                       help="Root directory containing robot trajectory data (single path or list of task paths for multi-task training)")
    parser.add_argument("--trail_start", type=int, default=None,
                       help="Starting trail number (e.g., 1 for trail_1). If None, use all trails.")
    parser.add_argument("--trail_end", type=int, default=None,
                       help="Ending trail number (e.g., 50 for trail_50). If None, use all trails.")
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

    # 历史帧配置
    parser.add_argument("--num_history_frames", type=int, default=1,
                       help="Number of history frames as condition. "
                            "Allowed values: 1, 2, or 1+4N (5, 9, 13, ...). "
                            "1: single-frame condition (backward compatible), 1 condition latent. "
                            "2: two frames separately encoded, 2 condition latents. "
                            "1+4N: first frame alone + groups of 4, (1+N) condition latents.")

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
    # 处理data_root - 支持单任务或多任务
    data_roots = args.heatmap_data_root if isinstance(args.heatmap_data_root, list) else [args.heatmap_data_root]
    if len(data_roots) > 1:
        print(f"Multi-task training mode: {len(data_roots)} tasks")
        for i, root in enumerate(data_roots):
            print(f"  Task {i+1}: {root}")
    else:
        print(f"Data root: {data_roots[0]}")
    if args.trail_start is not None or args.trail_end is not None:
        print(f"Trail range: [{args.trail_start or 'start'}, {args.trail_end or 'end'}]")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Colormap: {args.colormap_name}")
    print(f"Output path: {args.output_path}")
    print(f"Debug mode: {args.debug_mode}")
    print(f"SwanLab enabled: {args.enable_swanlab and not args.debug_mode}")
    print("="*60)

    # SwanLab初始化将在Accelerator初始化后进行
    # 这样可以确保使用正确的进程信息
    swanlab_run = None

    # 打印分布式环境信息（仅用于调试）
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    # 定义 is_main_process 用于 Accelerator 初始化之前的代码
    # 注意：这是基于环境变量的初步判断，可能在某些情况下不准确
    # Accelerator 初始化后应使用 accelerator.is_main_process
    is_main_process = global_rank == 0

    print(f"🚀 Process initialization...")
    print(f"   - LOCAL_RANK from env: {local_rank}")
    print(f"   - RANK from env: {global_rank}")
    print(f"   - WORLD_SIZE from env: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"   - Is main process (preliminary): {is_main_process}")
    print(f"   ⚠️  Note: These values might not be final until Accelerator is initialized")

    # 解析参数
    scene_bounds = parse_float_list(args.scene_bounds, "scene_bounds")
    transform_augmentation_xyz = parse_float_list(args.transform_augmentation_xyz, "transform_augmentation_xyz")
    transform_augmentation_rpy = parse_float_list(args.transform_augmentation_rpy, "transform_augmentation_rpy")

    # 验证 num_history_frames 的合法性：必须是 1, 2, 或 1+4N
    def is_valid_history_frames(n):
        if n == 1 or n == 2:
            return True
        if n > 2 and (n - 1) % 4 == 0:
            return True
        return False

    if not is_valid_history_frames(args.num_history_frames):
        raise ValueError(
            f"Invalid num_history_frames={args.num_history_frames}.\n"
            f"Allowed values: 1, 2, or 1+4N (5, 9, 13, ...).\n"
            f"This ensures proper VAE encoding: first frame alone, then groups of 4."
        )

    # 验证 wan_type 和 num_history_frames 的一致性（双向检测）
    # 规则1: num_history_frames > 1 必须使用 5B_TI2V_RGB_HEATMAP_MV_HISTORY
    if args.num_history_frames > 1 and args.wan_type != "5B_TI2V_RGB_HEATMAP_MV_HISTORY":
        raise ValueError(
            f"Invalid configuration: num_history_frames={args.num_history_frames} > 1, "
            f"but wan_type={args.wan_type}.\n"
            f"When using multi-frame history (num_history_frames > 1), you MUST set "
            f"wan_type='5B_TI2V_RGB_HEATMAP_MV_HISTORY'."
        )
    # 规则2: 使用 5B_TI2V_RGB_HEATMAP_MV_HISTORY 必须设置 num_history_frames > 1
    if args.num_history_frames == 1 and args.wan_type == "5B_TI2V_RGB_HEATMAP_MV_HISTORY":
        raise ValueError(
            f"Invalid configuration: wan_type=5B_TI2V_RGB_HEATMAP_MV_HISTORY, "
            f"but num_history_frames={args.num_history_frames}.\n"
            f"When using 5B_TI2V_RGB_HEATMAP_MV_HISTORY, you MUST set num_history_frames > 1.\n"
            f"If you want single-frame mode, use wan_type='5B_TI2V_RGB_HEATMAP_MV'."
        )

    # 创建热力图数据集
    if is_main_process:
        print("Creating heatmap dataset...")
        print(f"  num_history_frames: {args.num_history_frames}")

    # 选择数据集工厂：多帧历史使用 HeatmapDatasetFactoryWithHistory
    use_history_dataset = args.num_history_frames > 1
    DatasetFactory = HeatmapDatasetFactoryWithHistory if use_history_dataset else HeatmapDatasetFactory

    try:
        # 多任务训练: 为每个任务创建数据集
        if len(data_roots) > 1:
            if is_main_process:
                print(f"Multi-task training mode: {len(data_roots)} tasks")
                if use_history_dataset:
                    print(f"  Using HeatmapDatasetFactoryWithHistory (num_history_frames={args.num_history_frames})")
            datasets = []
            for task_idx, task_root in enumerate(data_roots):
                if is_main_process:
                    print(f"  Loading task {task_idx+1}/{len(data_roots)}: {task_root}")

                # 构建基础参数
                dataset_kwargs = dict(
                    data_root=task_root,
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
                    trail_start=args.trail_start,
                    trail_end=args.trail_end,
                    use_merged_pointcloud=args.use_merged_pointcloud,
                    use_different_projection=args.use_different_projection,
                )
                # 多帧历史需要额外参数
                if use_history_dataset:
                    dataset_kwargs['num_history_frames'] = args.num_history_frames

                task_dataset = DatasetFactory.create_robot_trajectory_dataset(**dataset_kwargs)
                if is_main_process:
                    print(f"    ✓ Task {task_idx+1} loaded: {len(task_dataset)} samples")
                datasets.append(task_dataset)

            # 合并所有数据集
            dataset = ConcatDataset(datasets)
            if is_main_process:
                print(f"✓ Multi-task dataset created: {len(dataset)} samples (from {len(data_roots)} tasks)")
        else:
            # 单任务训练
            if is_main_process:
                print(f"Single-task training mode: {data_roots[0]}")
                if use_history_dataset:
                    print(f"  Using HeatmapDatasetFactoryWithHistory (num_history_frames={args.num_history_frames})")

            # 构建基础参数
            dataset_kwargs = dict(
                data_root=data_roots[0],
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
                trail_start=args.trail_start,
                trail_end=args.trail_end,
                use_merged_pointcloud=args.use_merged_pointcloud,
                use_different_projection=args.use_different_projection,
            )
            # 多帧历史需要额外参数
            if use_history_dataset:
                dataset_kwargs['num_history_frames'] = args.num_history_frames

            dataset = DatasetFactory.create_robot_trajectory_dataset(**dataset_kwargs)
            if is_main_process:
                print(f"✓ Dataset created: {len(dataset)} samples")

        # 等待所有进程完成数据集创建（在分布式环境下由barrier自动处理）
        pass

        if is_main_process:
            print(f"Dataset created successfully with {len(dataset)} samples")

            # 测试数据加载
            print("Testing data loading...")
            test_sample = dataset[10]
            print(f"Sample keys: {list(test_sample.keys())}")
            print(f"Video frames: {len(test_sample['video'])}")
            print(f"First frame size: {test_sample['video'][0][0].size}")
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
            unfreeze_modulation_and_norms=args.unfreeze_modulation_and_norms,
            num_history_frames=args.num_history_frames,
            rgb_loss_weight=args.rgb_loss_weight,
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