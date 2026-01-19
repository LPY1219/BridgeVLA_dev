#!/usr/bin/env python3
"""
Single GPU version of heatmap training script for debugging
"""

import os
import sys

# Add the path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
diffsynth_dir = os.path.join(script_dir, "../../../")
sys.path.insert(0, diffsynth_dir)

# Import the original script
from heatmap_train import *

if __name__ == "__main__":
    # Force single GPU mode by setting environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

    parser = create_heatmap_parser()
    args = parser.parse_args()

    # Override distributed training settings
    args.find_unused_parameters = False
    args.dataset_num_workers = 0  # Single process

    print("="*60)
    print("SINGLE GPU HEATMAP TRAINING TEST")
    print("="*60)
    print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Data root: {args.heatmap_data_root}")
    print(f"Sequence length: {args.sequence_length}")
    print("="*60)

    # Disable SwanLab for testing
    args.enable_swanlab = False
    swanlab_run = None

    try:
        # Test dataset creation first
        print("1. Testing dataset creation...")
        scene_bounds = parse_float_list(args.scene_bounds, "scene_bounds")
        transform_augmentation_xyz = parse_float_list(args.transform_augmentation_xyz, "transform_augmentation_xyz")
        transform_augmentation_rpy = parse_float_list(args.transform_augmentation_rpy, "transform_augmentation_rpy")

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
            debug=True,  # Force debug mode for faster testing
            colormap_name=args.colormap_name,
            repeat=args.dataset_repeat,
            wan_type=args.wan_type,
        )
        print(f"✅ Dataset created successfully with {len(dataset)} samples")

        # Test data loading
        print("2. Testing data sample loading...")
        test_sample = dataset[0]
        print(f"✅ Sample loaded. Keys: {list(test_sample.keys())}")
        print(f"   Video frames: {len(test_sample['video'])}")
        print(f"   Prompt: {test_sample['prompt'][:50]}...")

        # Test model creation
        print("3. Testing model creation...")
        print(f"GPU memory before model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        model_id_with_origin_paths = args.model_id_with_origin_paths if args.model_id_with_origin_paths else None
        lora_checkpoint = args.lora_checkpoint if args.lora_checkpoint else None
        trainable_models = args.trainable_models if args.trainable_models else None

        model = HeatmapWanTrainingModule(
            wan_type=args.wan_type,
            model_paths=args.model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            trainable_models=trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=lora_checkpoint,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
        print(f"✅ Model created successfully")
        print(f"GPU memory after model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # Test forward pass
        print("4. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            loss = model(test_sample)
            print(f"✅ Forward pass successful. Loss: {loss.item():.4f}")

        print("\n" + "="*60)
        print("ALL TESTS PASSED! Single GPU mode works correctly.")
        print("The issue is likely with distributed training configuration.")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed at step: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis helps identify exactly where the failure occurs.")