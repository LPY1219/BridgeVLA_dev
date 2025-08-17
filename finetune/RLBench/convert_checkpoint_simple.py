#!/usr/bin/env python3

import os
import torch
import sys
import argparse

def convert_checkpoint(checkpoint_dir=None, output_path=None):
    """Convert DeepSpeed checkpoint to standard PyTorch format"""
    
    # Default paths if not provided
    if checkpoint_dir is None:
        checkpoint_dir = "/share/project/lpy/BridgeVLA/finetune/RLBench/logs/train/debug/debug/08_16_04_20/epoch_0"
    
    if output_path is None:
        # Extract epoch from directory name
        dir_name = os.path.basename(checkpoint_dir)
        if dir_name.startswith("epoch_"):
            try:
                epoch = int(dir_name.split("_")[1])
                output_path = os.path.join(os.path.dirname(checkpoint_dir), f"model_{epoch}.pth")
            except:
                output_path = os.path.join(os.path.dirname(checkpoint_dir), "model_0.pth")
        else:
            output_path = os.path.join(os.path.dirname(checkpoint_dir), "model_0.pth")
    
    print(f"Converting checkpoint from: {checkpoint_dir}")
    print(f"Output path: {output_path}")
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    try:
        # Method 1: Try using DeepSpeed's zero_to_fp32 utility
        print("Attempting conversion using DeepSpeed's zero_to_fp32...")
        
        # Import DeepSpeed utilities
        try:
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert ZeRO checkpoint to FP32
            state_dict = load_state_dict_from_zero_checkpoint(checkpoint_dir)
            
            # Extract epoch from directory name
            dir_name = os.path.basename(checkpoint_dir)
            epoch = 0
            if dir_name.startswith("epoch_"):
                try:
                    epoch = int(dir_name.split("_")[1])
                except:
                    epoch = 0
            
            # Create standard checkpoint format
            checkpoint = {
                "epoch": epoch,
                "model_state": state_dict
            }
            
            torch.save(checkpoint, output_path)
        except (ImportError, TypeError) as e:
            print(f"DeepSpeed zero_to_fp32 not available or not callable: {e}")
            raise ImportError("DeepSpeed utilities not available")
        
        print(f"Successfully converted using zero_to_fp32!")
        
    except ImportError as e:
        print(f"DeepSpeed utilities not available: {e}")
        print("Trying manual conversion...")
        
        try:
            # Method 2: Manual conversion
            print("Attempting manual conversion...")
            
            # Load the model states file
            model_states_path = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
            if not os.path.exists(model_states_path):
                print(f"Error: Model states file not found: {model_states_path}")
                return False
            
            print(f"Loading model states from: {model_states_path}")
            model_states = torch.load(model_states_path, map_location="cpu")
            
            print(f"Model states keys: {list(model_states.keys())}")
            
            # Extract the module state dict
            if "module" in model_states:
                module_state = model_states["module"]
                print(f"Module state keys: {list(module_state.keys())}")
            else:
                print("Warning: 'module' key not found in model states")
                module_state = model_states
            
            # Extract epoch from directory name
            dir_name = os.path.basename(checkpoint_dir)
            epoch = 0
            if dir_name.startswith("epoch_"):
                try:
                    epoch = int(dir_name.split("_")[1])
                except:
                    epoch = 0
            
            # Create standard checkpoint format
            checkpoint = {
                "epoch": epoch,
                "model_state": module_state
            }
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the converted model
            torch.save(checkpoint, output_path)
            
            print(f"Successfully converted checkpoint to: {output_path}")
            
        except Exception as e:
            print(f"Error in manual conversion: {e}")
            return False
    
    # Verify the converted model
    if os.path.exists(output_path):
        try:
            checkpoint = torch.load(output_path, map_location="cpu")
            print(f"Converted model keys: {list(checkpoint.keys())}")
            
            if "model_state" in checkpoint:
                model_state = checkpoint["model_state"]
                print(f"Model state contains {len(model_state)} parameters")
                
                # Show some parameter names as examples
                param_names = list(model_state.keys())
                print(f"Example parameters: {param_names[:5]}")
            
            return True
        except Exception as e:
            print(f"Error verifying converted model: {e}")
            return False
    else:
        print(f"Error: Converted model file not found at {output_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to PyTorch format")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Directory containing DeepSpeed checkpoint files")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for the converted model")
    
    args = parser.parse_args()
    
    success = convert_checkpoint(args.checkpoint_dir, args.output_path)
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 