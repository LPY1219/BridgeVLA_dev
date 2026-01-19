"""
Heatmap Training Script for Multi-View Token Concatenation Model

This script trains the WanModel_mv_concat which uses token concatenation
instead of multi-view attention modules for multi-view video generation.

Key differences from heatmap_train_mv.py:
- Uses WanModel_mv_concat instead of WanModel_mv
- No MVS attention modules to initialize
- Simpler architecture with standard DiT blocks
"""

import torch
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn

# Force flush output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# SwanLab import (optional)
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
from torch.utils.data import ConcatDataset

# Add trainers path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

# Import dataset for token concatenation model
from diffsynth.trainers.heatmap_dataset_mv_concat import HeatmapDatasetFactoryMVConcat

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HeatmapWanTrainingModuleMVConcat(DiffusionTrainingModule):
    """
    Training module for multi-view token concatenation model.

    Key differences from original HeatmapWanTrainingModule:
    - Uses WanModel_mv_concat (no mvs_attn modules)
    - Simpler parameter unfreezing (no MV attention modules)
    """

    def __init__(
        self,
        wan_type,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_dual_head=False,
        unfreeze_modulation_and_norms=False,
        rgb_loss_weight=0.5,
        num_views=3,
    ):
        super().__init__()

        self.unfreeze_modulation_and_norms = unfreeze_modulation_and_norms
        self.rgb_loss_weight = rgb_loss_weight
        self.num_views = num_views

        # Debug info
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print("\n" + "="*80)
            print("INITIALIZING MV CONCAT TRAINING MODULE")
            print("="*80)
            print(f"  WAN Type: {wan_type}")
            print(f"  Num Views: {num_views}")
            print(f"  LoRA Base Model: {lora_base_model}")
            print(f"  LoRA Target Modules: {lora_target_modules}")
            print(f"  LoRA Rank: {lora_rank}")
            print(f"  Use Dual Head: {use_dual_head}")
            print(f"  RGB Loss Weight: {rgb_loss_weight}")
            print(f"  Unfreeze Modulation & Norms: {unfreeze_modulation_and_norms}")
            if lora_checkpoint is not None:
                print(f"\n  PRETRAINED CHECKPOINT:")
                print(f"     Path: {lora_checkpoint}")
                if os.path.exists(lora_checkpoint):
                    file_size_mb = os.path.getsize(lora_checkpoint) / (1024 * 1024)
                    print(f"     File exists, size: {file_size_mb:.2f} MB")
                else:
                    print(f"     WARNING: File does not exist!")
            else:
                print(f"\n  No pretrained checkpoint (training from scratch)")
            print("="*80 + "\n")

        self.wan_type = wan_type

        # Import pipeline based on wan_type
        if self.wan_type in ["5B_TI2V_RGB_HEATMAP_MV_CONCAT", "5B_TI2V_RGB_HEATMAP_MV_CONCAT_ROT_GRIP"]:
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_concat import (
                WanVideoPipelineMVConcat as WanVideoPipeline,
            )
            # Import ModelConfig from original pipeline (for model loading)
            from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv import ModelConfig
        else:
            raise ValueError(f"Unsupported wan_type for MV Concat: {self.wan_type}")

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)

        if local_rank == 0:
            print("\n" + "="*80)
            print(f"[DEBUG] Loading pipeline with use_dual_head={use_dual_head}, num_views={num_views}")
            print("="*80)

        # Create pipeline
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            wan_type=self.wan_type,
            use_dual_head=use_dual_head,
            num_views=num_views
        )

        # Convert model to MV Concat version if needed
        self._convert_to_mv_concat_model(use_dual_head, num_views)

        self.use_dual_head = use_dual_head

        if local_rank == 0:
            print("\n" + "="*80)
            print(f"[DEBUG] Pipeline loaded successfully")
            print(f"  Model type: {type(self.pipe.dit).__name__}")
            print(f"  Num views: {self.pipe.dit.num_views}")
            print("="*80 + "\n")

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Unfreeze patch_embedding and head
        self._unfreeze_patch_embedding_and_head()

        # Unfreeze modulation (optional)
        if self.unfreeze_modulation_and_norms:
            self._unfreeze_modulation()

        # Print trainable parameters info
        self._print_trainable_parameters_info()

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Debug settings
        self.debug_counter = 0
        self.debug_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../debug_log"))

    def _convert_to_mv_concat_model(self, use_dual_head: bool, num_views: int):
        """
        Convert the loaded model to WanModel_mv_concat if needed.
        """
        from diffsynth.models.wan_video_dit_mv_concat import WanModel_mv_concat
        from diffsynth.pipelines.wan_video_5B_TI2V_heatmap_and_rgb_mv_concat import convert_wan_model_to_mv_concat

        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Check if already converted
        if isinstance(self.pipe.dit, WanModel_mv_concat):
            if local_rank == 0:
                print("[DEBUG] Model is already WanModel_mv_concat")
            return

        if local_rank == 0:
            print("[DEBUG] Converting model to WanModel_mv_concat...")

        # Convert model
        original_dit = self.pipe.dit
        self.pipe.dit = convert_wan_model_to_mv_concat(
            original_dit,
            use_dual_head=use_dual_head,
            num_views=num_views
        )

        if local_rank == 0:
            print(f"[DEBUG] Model converted successfully")
            print(f"  Type: {type(self.pipe.dit).__name__}")
            print(f"  Num views: {self.pipe.dit.num_views}")

    def _unfreeze_patch_embedding_and_head(self):
        """
        Unfreeze patch_embedding and head parameters.
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []

        for name, param in self.pipe.dit.named_parameters():
            is_patch_embedding = 'patch_embedding' in name
            is_head = ('head' in name) if not self.use_dual_head else ('head_rgb' in name or 'head_heatmap' in name)
            is_not_lora = ('lora_A' not in name and 'lora_B' not in name)

            if (is_patch_embedding or is_head) and is_not_lora:
                param.requires_grad = True
                unfrozen_params.append(name)

        if local_rank == 0:
            print("\n" + "="*80)
            mode_str = "DUAL HEAD" if self.use_dual_head else "SINGLE HEAD"
            print(f"PARAMETER UNFREEZING ({mode_str} MODE)")
            print("="*80)

            if len(unfrozen_params) > 0:
                print(f"\nUnfroze {len(unfrozen_params)} parameter(s):")
                for name in unfrozen_params:
                    param = dict(self.pipe.dit.named_parameters())[name]
                    print(f"  - {name}: {param.shape}")
            print("="*80 + "\n")

    def _unfreeze_modulation(self):
        """
        Unfreeze modulation parameters (AdaLN).
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        unfrozen_params = []

        for name, param in self.pipe.dit.named_parameters():
            if 'modulation' in name and 'blocks.' in name:
                param.requires_grad = True
                unfrozen_params.append(name)

        if local_rank == 0:
            print("\n" + "="*80)
            print("MODULATION PARAMETER UNFREEZING")
            print("="*80)
            print(f"Unfroze {len(unfrozen_params)} modulation parameters")
            print("="*80 + "\n")

    def _print_trainable_parameters_info(self):
        """
        Print trainable parameters info.
        """
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if local_rank != 0:
            return

        total_params = 0
        trainable_params = 0
        trainable_by_type = {}

        for name, param in self.pipe.dit.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # Categorize
                if 'lora' in name.lower():
                    cat = 'LoRA'
                elif 'patch_embedding' in name:
                    cat = 'Patch Embedding'
                elif 'head' in name:
                    cat = 'Head'
                elif 'modulation' in name:
                    cat = 'Modulation'
                else:
                    cat = 'Other'
                trainable_by_type[cat] = trainable_by_type.get(cat, 0) + param.numel()

        print("\n" + "="*80)
        print("TRAINABLE PARAMETERS SUMMARY")
        print("="*80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print("\nBy category:")
        for cat, count in sorted(trainable_by_type.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count:,}")
        print("="*80 + "\n")

    def forward_preprocess(self, data):
        """
        Preprocess input data.
        """
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0][0].size[1],
            "width": data["video"][0][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "use_dual_head": self.use_dual_head,
            "num_view": len(data["video"][0]),
        }

        for extra_input in self.extra_inputs:
            if extra_input in data:
                inputs_shared[extra_input] = data[extra_input]

        # Process through pipeline units (converts video to latents, etc.)
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        return {**inputs_posi, **inputs_nega, **inputs_shared}

    def forward(self, data, inputs=None):
        """
        Forward pass, compute loss.
        Returns a dict with 'loss', 'loss_rgb', 'loss_heatmap' keys.
        """
        if inputs is None:
            inputs = self.forward_preprocess(data)

        # Device consistency
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        expected_device = f"cuda:{local_rank}"

        if "rand_device" in inputs:
            inputs["rand_device"] = expected_device

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
            inputs = move_to_device(inputs, expected_device)
            data = move_to_device(data, expected_device)
        except Exception as e:
            print(f"Warning: Could not move inputs to device {expected_device}: {e}")

        # Forward pass
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss_dict = self.pipe.training_loss(**models, **inputs, rgb_loss_weight=self.rgb_loss_weight)

        # Return the full loss dict - training loop handles both dict and tensor
        return loss_dict

    def log_to_swanlab(self, loss_value, step, swanlab_run=None):
        """
        Log loss to SwanLab.
        """
        if swanlab_run is not None and SWANLAB_AVAILABLE:
            log_data = {
                "train/total_loss": loss_value["loss"].item() if hasattr(loss_value["loss"], 'item') else loss_value["loss"]
            }
            if "loss_rgb" in loss_value:
                log_data["train/rgb_loss"] = loss_value["loss_rgb"].item() if hasattr(loss_value["loss_rgb"], 'item') else loss_value["loss_rgb"]
            if "loss_heatmap" in loss_value:
                log_data["train/heatmap_loss"] = loss_value["loss_heatmap"].item() if hasattr(loss_value["loss_heatmap"], 'item') else loss_value["loss_heatmap"]
            swanlab_run.log(log_data, step=step)


def create_mv_concat_dataset(args):
    """
    Create dataset for multi-view token concatenation training.
    """
    # Get sigma value (prefer heatmap_sigma over sigma)
    sigma = getattr(args, 'heatmap_sigma', None) or getattr(args, 'sigma', 1.5)

    # Get colormap name
    colormap_name = getattr(args, 'colormap_name', 'jet')

    # Get trail range
    trail_start = getattr(args, 'trail_start', None)
    trail_end = getattr(args, 'trail_end', None)

    dataset = HeatmapDatasetFactoryMVConcat.create_robot_trajectory_dataset(
        data_root=args.dataset_base_path,
        sequence_length=getattr(args, 'sequence_length', 10),
        step_interval=getattr(args, 'step_interval', 1),
        min_trail_length=getattr(args, 'min_trail_length', 5),
        image_size=(args.height, args.width) if args.height and args.width else (256, 256),
        sigma=sigma,
        augmentation=True,
        mode="train",
        colormap_name=colormap_name,
        repeat=getattr(args, 'data_repeat', 1),
        wan_type=args.wan_type,
        use_different_projection=getattr(args, 'use_different_projection', False),
        num_views=getattr(args, 'num_views', 3),
        trail_start=trail_start,
        trail_end=trail_end,
    )
    return dataset


def launch_optimized_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    args=None,
):
    """
    Optimized training task launcher with SwanLab integration.
    Based on the original heatmap_train_mv.py implementation.
    """
    if args is None:
        raise ValueError("args is required for optimized training")

    # Parameter extraction
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = getattr(args, 'save_steps', 0) or 0  # Handle None
    save_epochs_interval = getattr(args, 'save_epochs_interval', 0) or 0  # Handle None
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters
    train_batch_size = args.train_batch_size

    # Main process check using global RANK
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    is_main_process_for_print = global_rank == 0

    # Handle pin_memory - can be bool or string
    pin_memory_arg = getattr(args, 'dataloader_pin_memory', True)
    if isinstance(pin_memory_arg, str):
        pin_memory = pin_memory_arg.lower() == 'true'
    else:
        pin_memory = bool(pin_memory_arg)

    if is_main_process_for_print:
        print(f"Training configuration:")
        print(f"  - Batch size per GPU: 1 (Wan model limitation)")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Effective batch size per GPU: {gradient_accumulation_steps}")
        print(f"  - Total effective batch size (all GPUs): {gradient_accumulation_steps} × {os.environ.get('WORLD_SIZE', 1)}")
        print(f"  - Data workers: {num_workers}")
        print(f"  - Pin memory: {pin_memory} (raw arg: {pin_memory_arg})")
        print(f"  - Gradient checkpointing: {getattr(args, 'use_gradient_checkpointing', False)}")
        print(f"  - Save epochs interval: {save_epochs_interval} (0=every epoch)")

    def collate_single_sample(batch):
        """Collate function: return first sample only (Wan model limitation)"""
        return batch[0]

    dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 1 if num_workers > 0 else None,
        'drop_last': True,
        'collate_fn': collate_single_sample,
    }

    if is_main_process_for_print and train_batch_size > 1:
        print(f"Note: train_batch_size={train_batch_size} is ignored. Wan model only supports batch_size=1")
        print(f"Use gradient_accumulation_steps={gradient_accumulation_steps} to simulate larger batches")

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    # Accelerator
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    if is_main_process_for_print:
        print(f"Accelerator initialized with distributed_type: {accelerator.state.distributed_type}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Prepare training components
    try:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        if accelerator.is_main_process:
            print("Model preparation completed successfully")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error during model preparation: {e}")
        raise e

    # Training loop
    try:
        if accelerator.is_main_process:
            print("Starting training loop...")
            print(f"Training for {num_epochs} epochs")
            print(f"Dataset size: {len(dataloader)} batches")

        for epoch_id in range(num_epochs):
            if accelerator.is_main_process:
                print(f"\nStarting epoch {epoch_id+1}/{num_epochs}")

            model.train()
            epoch_loss = 0
            step_count = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")

            for step, data in enumerate(pbar):
                with accelerator.accumulate(model):
                    optimizer.zero_grad(set_to_none=True)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Forward pass
                    try:
                        loss_dict = model(data)
                    except RuntimeError as e:
                        if "device" in str(e).lower():
                            if accelerator.is_main_process:
                                print(f"Device error at step {step}: {e}")
                            raise e
                        else:
                            raise e

                    # Extract loss (supports dict or single tensor)
                    if isinstance(loss_dict, dict):
                        loss = loss_dict["loss"]
                        loss_rgb = loss_dict.get("loss_rgb", None)
                        loss_heatmap = loss_dict.get("loss_heatmap", None)
                    else:
                        loss = loss_dict
                        loss_rgb = None
                        loss_heatmap = None

                    # Backward
                    accelerator.backward(loss)

                    # Gradient clipping
                    if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    epoch_loss += loss.item()
                    step_count += 1

                    # SwanLab logging
                    global_step = step + epoch_id * len(dataloader)
                    logging_steps = getattr(args, 'logging_steps', 10)
                    should_log = (accelerator.is_main_process and
                                 hasattr(args, 'swanlab_run') and args.swanlab_run is not None and
                                 global_step % logging_steps == 0)

                    if should_log:
                        try:
                            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

                            log_data = {
                                "train_loss": loss.item(),
                                "learning_rate": current_lr,
                                "epoch": epoch_id,
                                "step": global_step
                            }

                            if loss_rgb is not None:
                                log_data["train_loss_rgb"] = loss_rgb.item()
                            if loss_heatmap is not None:
                                log_data["train_loss_heatmap"] = loss_heatmap.item()

                            swanlab.log(log_data, step=global_step)

                            log_msg = f"SwanLab logged: step={global_step}, loss={loss.item():.4f}"
                            if loss_rgb is not None and loss_heatmap is not None:
                                log_msg += f", loss_rgb={loss_rgb.item():.4f}, loss_heatmap={loss_heatmap.item():.4f}"
                            log_msg += f", lr={current_lr:.2e}"
                            print(log_msg)
                        except Exception as e:
                            print(f"Warning: Failed to log to SwanLab: {e}")

                    # Progress bar
                    postfix_dict = {
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss/step_count:.4f}"
                    }
                    if loss_rgb is not None:
                        postfix_dict['loss_rgb'] = f"{loss_rgb.item():.4f}"
                    if loss_heatmap is not None:
                        postfix_dict['loss_hm'] = f"{loss_heatmap.item():.4f}"
                    pbar.set_postfix(postfix_dict)

                    # Step checkpoint save
                    if save_steps > 0:
                        model_logger.on_step_end(accelerator, model, save_steps)

            # Epoch checkpoint save
            should_save_epoch = (save_epochs_interval == 0) or ((epoch_id + 1) % save_epochs_interval == 0) or (epoch_id == num_epochs - 1)
            if should_save_epoch:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
                if accelerator.is_main_process:
                    print(f"Saved checkpoint at epoch {epoch_id + 1}")

            accelerator.print(f"Epoch {epoch_id+1} completed. Average loss: {epoch_loss/step_count:.4f}")

        # Training end
        if save_steps > 0:
            model_logger.on_training_end(accelerator, model, save_steps)

        if accelerator.is_main_process:
            print("Training completed successfully!")

    except Exception as training_error:
        if accelerator.is_main_process:
            print(f"Training failed with error: {training_error}")
            import traceback
            traceback.print_exc()
        raise training_error


def train_mv_concat(args):
    """
    Main training function for multi-view token concatenation model.
    """
    # Initialize SwanLab (optional)
    swanlab_run = None
    enable_swanlab = getattr(args, 'enable_swanlab', False)
    debug_mode = getattr(args, 'debug_mode', False)

    # Use global RANK to determine main process (for multi-node training)
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    is_main_process = global_rank == 0

    if is_main_process:
        print(f"Starting MV Concat training initialization...")
        print(f"   - SwanLab enabled: {enable_swanlab and not debug_mode and SWANLAB_AVAILABLE}")

    if is_main_process and enable_swanlab and not debug_mode and SWANLAB_AVAILABLE:
        try:
            api_key = getattr(args, 'swanlab_api_key', None)
            project = getattr(args, 'swanlab_project', None)
            experiment_name = getattr(args, 'swanlab_experiment', None) or getattr(args, 'swanlab_name', None)

            print("Initializing SwanLab...")
            if api_key:
                print(f"   API Key: {api_key[:8]}***")
            print(f"   Project: {project}")
            print(f"   Experiment: {experiment_name}")

            if api_key:
                swanlab.login(api_key=api_key)

            swanlab_run = swanlab.init(
                project=project,
                experiment_name=experiment_name,
                config={
                    "learning_rate": getattr(args, 'learning_rate', 1e-4),
                    "num_epochs": getattr(args, 'num_epochs', 100),
                    "sequence_length": getattr(args, 'sequence_length', 10),
                    "train_batch_size": getattr(args, 'train_batch_size', 1),
                    "lora_rank": getattr(args, 'lora_rank', 32),
                    "height": getattr(args, 'height', 256),
                    "width": getattr(args, 'width', 256),
                    "num_views": getattr(args, 'num_views', 3),
                    "wan_type": getattr(args, 'wan_type', '5B_TI2V_RGB_HEATMAP_MV_CONCAT'),
                    "rgb_loss_weight": getattr(args, 'rgb_loss_weight', 0.5),
                }
            )
            print(f"SwanLab initialized successfully!")
            print(f"   Project: {project}")
            print(f"   Experiment: {experiment_name}")

            # Test log to confirm SwanLab is working
            swanlab.log({"test": 1.0}, step=0)
            print("Test log sent to SwanLab")

        except Exception as e:
            print(f"Failed to initialize SwanLab: {e}")
            import traceback
            traceback.print_exc()
            swanlab_run = None
    elif is_main_process and enable_swanlab and debug_mode:
        print("SwanLab disabled in debug mode")
    elif is_main_process and enable_swanlab and not SWANLAB_AVAILABLE:
        print("Warning: SwanLab requested but not available. Install with: pip install swanlab")

    # Create dataset
    dataset = create_mv_concat_dataset(args)
    print(f"Dataset created with {len(dataset)} samples")

    # Create training module
    training_module = HeatmapWanTrainingModuleMVConcat(
        wan_type=args.wan_type,
        model_paths=getattr(args, 'model_paths', None),
        model_id_with_origin_paths=getattr(args, 'model_id_with_origin_paths', None),
        trainable_models=getattr(args, 'trainable_models', "dit"),
        lora_base_model=getattr(args, 'lora_base_model', None),
        lora_target_modules=getattr(args, 'lora_target_modules', "q,k,v,o,ffn.0,ffn.2"),
        lora_rank=getattr(args, 'lora_rank', 32),
        lora_checkpoint=getattr(args, 'lora_checkpoint', None),
        use_gradient_checkpointing=getattr(args, 'use_gradient_checkpointing', False),
        extra_inputs=getattr(args, 'extra_inputs', "input_image,input_image_rgb,input_video_rgb"),
        max_timestep_boundary=getattr(args, 'max_timestep_boundary', 1.0),
        min_timestep_boundary=getattr(args, 'min_timestep_boundary', 0.0),
        use_dual_head=getattr(args, 'use_dual_head', True),
        unfreeze_modulation_and_norms=getattr(args, 'unfreeze_modulation_and_norms', False),
        rgb_loss_weight=getattr(args, 'rgb_loss_weight', 0.5),
        num_views=getattr(args, 'num_views', 3),
    )

    # Create model logger
    output_dir = getattr(args, 'output_path', './output')
    remove_prefix = getattr(args, 'remove_prefix_in_ckpt', 'pipe.dit.')
    model_logger = ModelLogger(output_dir, remove_prefix_in_ckpt=remove_prefix)

    # Pass swanlab_run through args (for logging inside training loop)
    args.swanlab_run = swanlab_run

    # Launch training using optimized training task (with SwanLab integration)
    launch_optimized_training_task(
        dataset=dataset,
        model=training_module,
        model_logger=model_logger,
        args=args,
    )

    # Close SwanLab
    if swanlab_run is not None:
        swanlab_run.finish()
        if is_main_process:
            print("SwanLab experiment finished")


if __name__ == "__main__":
    # Parse arguments
    parser = wan_parser()

    # Add MV Concat specific arguments
    parser.add_argument("--num_views", type=int, default=3, help="Number of views")
    parser.add_argument("--data_repeat", type=int, default=1, help="Data repeat count")
    parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length")
    parser.add_argument("--step_interval", type=int, default=1, help="Step interval")
    parser.add_argument("--min_trail_length", type=int, default=5, help="Min trail length")
    parser.add_argument("--sigma", type=float, default=1.5, help="Heatmap gaussian sigma")
    parser.add_argument("--heatmap_sigma", type=float, default=1.5, help="Heatmap gaussian sigma (alias for --sigma)")
    parser.add_argument("--use_different_projection", action="store_true", help="Use different projection")
    parser.add_argument("--use_dual_head", action="store_true", help="Use dual head mode")
    parser.add_argument("--unfreeze_modulation_and_norms", action="store_true", help="Unfreeze modulation params")
    parser.add_argument("--rgb_loss_weight", type=float, default=0.5, help="RGB loss weight")

    # Trail range
    parser.add_argument("--trail_start", type=int, default=None, help="Starting trail number")
    parser.add_argument("--trail_end", type=int, default=None, help="Ending trail number")

    # Heatmap data parameters
    parser.add_argument("--colormap_name", type=str, default="jet", help="Colormap name")
    parser.add_argument("--scene_bounds", type=str, default="0,-0.45,-0.05,0.8,0.55,0.6", help="Scene bounds")
    parser.add_argument("--transform_augmentation_xyz", type=str, default="0.1,0.1,0.1", help="XYZ augmentation range")
    parser.add_argument("--transform_augmentation_rpy", type=str, default="0.0,0.0,20.0", help="RPY augmentation range")

    # Model type
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_CONCAT", help="Wan model type")

    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--save_epochs_interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--dataloader_pin_memory", type=str, default="true", help="Pin memory for dataloader")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing")

    # Debug mode
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")

    # SwanLab parameters
    parser.add_argument("--enable_swanlab", action="store_true", help="Enable SwanLab logging")
    parser.add_argument("--swanlab_api_key", type=str, default=None, help="SwanLab API key")
    parser.add_argument("--swanlab_project", type=str, default=None, help="SwanLab project name")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--swanlab_experiment", type=str, default=None, help="SwanLab experiment name (alias)")

    # Note: --max_timestep_boundary and --min_timestep_boundary are already defined in wan_parser()

    args = parser.parse_args()

    # Set default wan_type if not specified
    if not hasattr(args, 'wan_type') or args.wan_type is None:
        args.wan_type = "5B_TI2V_RGB_HEATMAP_MV_CONCAT"

    # Run training
    train_mv_concat(args)
