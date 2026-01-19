#!/usr/bin/env python3
"""
快速测试DeepSpeed是否正常工作的脚本
使用方法: accelerate launch --use_deepspeed --deepspeed_config_file=config.json --num_processes=2 test_deepspeed.py
"""

import torch
from accelerate import Accelerator
import os

def main():
    print(f"🔍 Environment Variables:")
    print(f"   - LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"   - WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"   - RANK: {os.environ.get('RANK', 'Not set')}")

    # 初始化Accelerator
    print(f"\n🚀 Initializing Accelerator...")
    accelerator = Accelerator()

    # 检查DeepSpeed状态
    print(f"\n🔍 Accelerator State:")
    print(f"   - distributed_type: {accelerator.state.distributed_type}")
    print(f"   - use_deepspeed: {accelerator.state.use_deepspeed}")
    print(f"   - deepspeed_plugin: {accelerator.state.deepspeed_plugin is not None}")

    if accelerator.state.deepspeed_plugin:
        print(f"   ✅ DeepSpeed is ENABLED!")
        config = accelerator.state.deepspeed_plugin.deepspeed_config
        print(f"   - ZeRO stage: {config.get('zero_optimization', {}).get('stage', 'Unknown')}")
    else:
        print(f"   ❌ DeepSpeed is NOT enabled!")

    # 检查GPU显存
    if torch.cuda.is_available():
        print(f"\n🔍 GPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved, {mem_total:.2f}GB total")

    # 创建一个简单的模型测试
    print(f"\n🧪 Testing with a small model...")
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 使用accelerator.prepare
    model, optimizer = accelerator.prepare(model, optimizer)

    print(f"   - Model type: {type(model)}")
    print(f"   - Is DeepSpeed model: {hasattr(model, 'module') and 'DeepSpeed' in str(type(model))}")

    # 检查显存变化
    if torch.cuda.is_available():
        print(f"\n🔍 GPU Memory After Model Preparation:")
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   - GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

    print(f"\n✅ DeepSpeed test completed!")

if __name__ == "__main__":
    main()