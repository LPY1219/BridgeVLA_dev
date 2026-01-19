"""检查checkpoint文件的内容结构"""
import torch
from safetensors import safe_open
import sys

def check_checkpoint(checkpoint_path):
    """检查checkpoint文件的keys和结构"""
    print(f"Checking checkpoint: {checkpoint_path}")
    print("=" * 80)

    try:
        # 尝试用safetensors加载
        if checkpoint_path.endswith('.safetensors'):
            print("\nLoading with safetensors...")
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"\nTotal keys: {len(keys)}")

                # 分析key的类型
                key_types = {
                    'dit': [],
                    'patch_embedding': [],
                    'head': [],
                    'blocks': [],
                    'mvs_attn': [],
                    'base_layer': [],
                    'lora': [],
                    'modulation': [],
                    'other': []
                }

                for key in keys:
                    if 'lora' in key.lower():
                        key_types['lora'].append(key)
                    elif 'dit' in key:
                        key_types['dit'].append(key)
                    elif 'patch_embedding' in key or 'patch_embed' in key:
                        key_types['patch_embedding'].append(key)
                    elif 'head' in key:
                        key_types['head'].append(key)
                    elif 'blocks' in key:
                        key_types['blocks'].append(key)
                    elif 'mvs_attn' in key:
                        key_types['mvs_attn'].append(key)
                    elif 'base_layer' in key:
                        key_types['base_layer'].append(key)
                    elif 'modulation' in key:
                        key_types['modulation'].append(key)
                    else:
                        key_types['other'].append(key)

                # 打印统计信息
                print("\n" + "=" * 80)
                print("KEY STATISTICS:")
                print("=" * 80)
                for key_type, key_list in key_types.items():
                    if key_list:
                        print(f"\n{key_type.upper()}: {len(key_list)} keys")
                        print(f"  Sample keys (first 5):")
                        for k in key_list[:5]:
                            # 获取tensor的shape
                            tensor = f.get_tensor(k)
                            print(f"    - {k} | shape: {tensor.shape}")
                        if len(key_list) > 5:
                            print(f"    ... and {len(key_list) - 5} more")

                # 特别检查blocks相关的keys
                print("\n" + "=" * 80)
                print("DETAILED BLOCKS ANALYSIS:")
                print("=" * 80)
                blocks_keys = [k for k in keys if 'blocks' in k]

                # 统计不同的模块
                modules = set()
                for key in blocks_keys:
                    parts = key.split('.')
                    if len(parts) >= 3:
                        modules.add(parts[2])  # blocks.X.MODULE_NAME

                print(f"\nModules in blocks: {sorted(modules)}")

                # 对每个模块显示示例
                for module in sorted(modules)[:10]:
                    module_keys = [k for k in blocks_keys if f'.{module}.' in k or k.endswith(f'.{module}')]
                    print(f"\n{module}: {len(module_keys)} keys")
                    for k in module_keys[:3]:
                        tensor = f.get_tensor(k)
                        print(f"  - {k} | shape: {tensor.shape}")
                    if len(module_keys) > 3:
                        print(f"  ... and {len(module_keys) - 3} more")

        else:
            # 尝试用torch加载
            print("\nLoading with torch...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                print(f"\nCheckpoint keys: {checkpoint.keys()}")

                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"\nModel state_dict has {len(state_dict)} keys")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"\nState_dict has {len(state_dict)} keys")
                else:
                    state_dict = checkpoint
                    print(f"\nDirect checkpoint has {len(state_dict)} keys")

                # 显示前20个keys
                print("\nFirst 20 keys:")
                for i, key in enumerate(list(state_dict.keys())[:20]):
                    print(f"  {i+1}. {key} | shape: {state_dict[key].shape}")
            else:
                print(f"\nCheckpoint type: {type(checkpoint)}")

    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    checkpoint_path = "/DATA/disk0/lpy/BridgeVLA_dev/logs/Wan/train/Wan2.2-TI2V-5B_heatmap_rgb_lora/full_finetune/push_T_5-epoch-69.safetensors"
    check_checkpoint(checkpoint_path)
