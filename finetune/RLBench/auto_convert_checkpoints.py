#!/usr/bin/env python3

import os
import glob
import sys
from convert_checkpoint_simple import convert_checkpoint

def find_deepspeed_checkpoints(log_dir):
    """Find all DeepSpeed checkpoint directories in the log directory"""
    checkpoint_dirs = []
    
    # Look for epoch_* directories
    epoch_pattern = os.path.join(log_dir, "epoch_*")
    epoch_dirs = glob.glob(epoch_pattern)
    
    for epoch_dir in epoch_dirs:
        if os.path.isdir(epoch_dir):
            # Check if it contains DeepSpeed checkpoint files
            model_states_file = os.path.join(epoch_dir, "mp_rank_00_model_states.pt")
            if os.path.exists(model_states_file):
                checkpoint_dirs.append(epoch_dir)
    
    return sorted(checkpoint_dirs)

def find_converted_models(log_dir):
    """Find all converted model files in the log directory"""
    model_pattern = os.path.join(log_dir, "model_*.pth")
    model_files = glob.glob(model_pattern)
    return sorted(model_files)

def get_epoch_from_path(path):
    """Extract epoch number from path"""
    basename = os.path.basename(path)
    if basename.startswith("epoch_"):
        try:
            return int(basename.split("_")[1])
        except:
            return None
    elif basename.startswith("model_") and basename.endswith(".pth"):
        try:
            return int(basename.replace("model_", "").replace(".pth", ""))
        except:
            return None
    return None

def auto_convert_checkpoints(log_dir, target_epoch=None):
    """Automatically convert DeepSpeed checkpoints that haven't been converted yet"""
    
    print(f"Scanning for checkpoints in: {log_dir}")
    
    # Find all DeepSpeed checkpoint directories
    checkpoint_dirs = find_deepspeed_checkpoints(log_dir)
    print(f"Found {len(checkpoint_dirs)} DeepSpeed checkpoint directories:")
    for dir_path in checkpoint_dirs:
        epoch = get_epoch_from_path(dir_path)
        print(f"  - {os.path.basename(dir_path)} (epoch {epoch})")
    
    # Find all converted model files
    converted_models = find_converted_models(log_dir)
    print(f"Found {len(converted_models)} converted model files:")
    for model_path in converted_models:
        epoch = get_epoch_from_path(model_path)
        print(f"  - {os.path.basename(model_path)} (epoch {epoch})")
    
    # Find checkpoints that need conversion
    converted_epochs = set()
    for model_path in converted_models:
        epoch = get_epoch_from_path(model_path)
        if epoch is not None:
            converted_epochs.add(epoch)
    
    checkpoints_to_convert = []
    for checkpoint_dir in checkpoint_dirs:
        epoch = get_epoch_from_path(checkpoint_dir)
        if epoch is not None:
            # If target_epoch is specified, only convert that epoch
            if target_epoch is not None:
                if epoch == target_epoch and epoch not in converted_epochs:
                    checkpoints_to_convert.append((checkpoint_dir, epoch))
            else:
                # Convert all unconverted epochs
                if epoch not in converted_epochs:
                    checkpoints_to_convert.append((checkpoint_dir, epoch))
    
    if not checkpoints_to_convert:
        print("✓ All checkpoints have been converted!")
        return True
    
    print(f"\nNeed to convert {len(checkpoints_to_convert)} checkpoints:")
    for checkpoint_dir, epoch in checkpoints_to_convert:
        print(f"  - epoch_{epoch} -> model_{epoch}.pth")
    
    # Convert checkpoints
    success_count = 0
    failed_epochs = []
    for checkpoint_dir, epoch in checkpoints_to_convert:
        print(f"\nConverting epoch_{epoch}...")
        
        # Generate output path
        output_path = os.path.join(log_dir, f"model_{epoch}.pth")
        
        # Convert checkpoint
        success = convert_checkpoint(checkpoint_dir, output_path)
        if success:
            print(f"✓ Successfully converted epoch_{epoch}")
            success_count += 1
        else:
            print(f"✗ Failed to convert epoch_{epoch}")
            failed_epochs.append(epoch)
    
    print(f"\nConversion summary: {success_count}/{len(checkpoints_to_convert)} successful")
    
    if failed_epochs:
        print(f"Failed epochs: {failed_epochs}")
        print("Note: Some checkpoints may be corrupted or incomplete")
        print("Continuing with successfully converted models...")
    
    # Return True if at least one conversion was successful
    return success_count > 0

def main():
    # Default log directory
    default_log_dir = "/share/project/lpy/BridgeVLA/finetune/RLBench/logs/train/debug/debug/08_16_04_20"
    
    # Parse command line arguments
    log_dir = default_log_dir
    target_epoch = None
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            target_epoch = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid epoch number: {sys.argv[2]}")
            sys.exit(1)
    
    print("="*60)
    print("DeepSpeed Checkpoint Auto-Converter")
    if target_epoch is not None:
        print(f"Target epoch: {target_epoch}")
    print("="*60)
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)
    
    success = auto_convert_checkpoints(log_dir, target_epoch)
    
    if success:
        print("\n" + "="*60)
        print("✓ AUTO-CONVERSION COMPLETED!")
        print("✓ Successfully converted available DeepSpeed checkpoints to PyTorch format")
        print("✓ You can now use load_agent() function to load the models")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ AUTO-CONVERSION FAILED!")
        print("✗ No checkpoints could be converted")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main() 