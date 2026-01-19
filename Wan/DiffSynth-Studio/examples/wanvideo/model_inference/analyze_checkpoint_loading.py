"""分析checkpoint加载逻辑和全量微调checkpoint的兼容性"""

print("=" * 80)
print("CHECKPOINT ANALYSIS REPORT")
print("=" * 80)

print("\n1. 全量微调Checkpoint内容 (已检查):")
print("-" * 80)
print("""
总共: 1218 keys

关键模块:
  - patch_embedding: 2 keys
    例: patch_embedding.weight, patch_embedding.bias

  - head (dual head): 6 keys
    例: head_heatmap.head.weight, head_heatmap.head.bias
        head_rgb.head.weight, head_rgb.head.bias

  - blocks.*: 1200 keys
    ├─ self_attn: 300 keys (例: blocks.0.self_attn.k.weight)
    ├─ cross_attn: 300 keys (例: blocks.0.cross_attn.k.weight)
    ├─ mvs_attn: 300 keys (例: blocks.0.mvs_attn.k.weight) ⚠️ 关键！
    ├─ ffn: 120 keys
    ├─ modulation: 30 keys (例: blocks.0.modulation)
    ├─ modulation_mvs: 30 keys (例: blocks.0.modulation_mvs)
    ├─ projector: 60 keys (例: blocks.0.projector.weight)
    └─ norm3: 60 keys

  - text_embedding: 4 keys
  - time_embedding: 6 keys

⚠️ 关键发现:
  - mvs_attn的keys是直接的权重名称 (如 blocks.0.mvs_attn.k.weight)
  - 没有 .base_layer 后缀
  - 没有任何 lora 相关的 keys
""")

print("\n2. LoRA Checkpoint内容 (推测):")
print("-" * 80)
print("""
包含内容:
  - patch_embedding: 相同
  - head: 相同
  - blocks.*.mvs_attn: 包含 .base_layer 后缀
    例: blocks.0.mvs_attn.k.base_layer.weight  ⚠️ 注意差异！
  - blocks.*.mvs_attn: 包含 lora 权重
    例: blocks.0.mvs_attn.k.lora_A.weight
        blocks.0.mvs_attn.k.lora_B.weight
  - modulation, modulation_mvs, projector 等: 相同
""")

print("\n3. 现有 load_checkpoint_weights() 函数逻辑:")
print("-" * 80)
print("""
位置: heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed.py:360-509

加载步骤:
  Step 1: 筛选 patch_embedding 权重
    - 匹配规则: 'patch_embedding' in key or 'patch_embed' in key
    - ✓ 可以处理全量微调checkpoint

  Step 2: 筛选 head 权重
    - 匹配规则: 'head' in key (排除attention相关)
    - ✓ 可以处理全量微调checkpoint

  Step 3: 筛选 MV base_layer 权重
    - 匹配规则: 'base_layer' in key
    - 转换: blocks.X.mvs_attn.Y.base_layer.weight -> blocks.X.mvs_attn.Y.weight
    - ⚠️ 全量微调checkpoint中没有base_layer，这一步会跳过

  Step 4: 筛选 modulation 权重
    - 匹配规则: '.modulation' in key and 'mvs' not in key
    - ✓ 可以处理全量微调checkpoint

  Step 5: 筛选 MV 其他权重
    - 匹配规则: 'projector' in key or 'norm_mvs' in key or
                'modulation_mvs' in key or 'mvs_attn' in key
    - ✓ 可以处理全量微调checkpoint
    - ⚠️ mvs_attn的权重会在这一步被捕获！

  Step 6: 合并并加载
    - 使用 dit.load_state_dict(weights_clean, strict=False)
""")

print("\n4. 现有 load_lora_with_base_weights() 函数逻辑:")
print("-" * 80)
print("""
位置: heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed.py:517-545

加载步骤:
  Step 1: 调用 load_checkpoint_weights() 加载所有base layer权重
  Step 2: 调用 pipe.load_lora() 加载LoRA权重并应用到base layer上

设计目的:
  - 专门为LoRA训练设计
  - 先加载base_layer权重，再应用LoRA变化量
""")

print("\n5. 兼容性分析:")
print("-" * 80)
print("""
✓ load_checkpoint_weights() 可以处理全量微调checkpoint:
  - patch_embedding: 可以正确加载
  - head: 可以正确加载
  - mvs_attn: 会在Step 5被捕获并加载 (无需转换)
  - modulation: 可以正确加载
  - modulation_mvs: 可以正确加载
  - projector: 可以正确加载
  - 其他blocks.*权重: 虽然没有明确筛选，但strict=False会忽略

⚠️ 潜在问题:
  - load_checkpoint_weights() 只加载了特定模块的权重
  - 全量微调的self_attn, cross_attn, ffn等权重可能被忽略？

✓ 解决方案:
  方案A: 直接使用 load_checkpoint_weights()
    - 优点: 代码改动最小
    - 缺点: 可能遗漏一些权重

  方案B: 为全量微调创建专门的加载函数
    - 加载所有dit相关的权重（不筛选）
    - 更加完整和可靠

推荐: 方案B - 创建 load_full_finetune_checkpoint() 函数
""")

print("\n6. 需要验证的问题:")
print("-" * 80)
print("""
问题1: 全量微调训练时，哪些参数被冻结？哪些被训练？
  - 需要检查训练脚本确认
  - 用户说: "训练时只保存了可训练的dit模块"

问题2: checkpoint中是否包含 self_attn, cross_attn, ffn 等权重？
  - 从检查结果看: 是的，都包含了
  - 说明这些模块也被训练了

问题3: load_checkpoint_weights() 会加载这些权重吗？
  - 从代码看: 不会，因为没有明确筛选
  - 需要确认: 这些权重是否需要加载？

建议: 创建一个完整的加载函数，加载checkpoint中的所有dit相关权重
""")

print("\n7. 推荐的修改方案:")
print("-" * 80)
print("""
方案: 创建新函数 load_full_finetune_checkpoint()

def load_full_finetune_checkpoint(self, checkpoint_path: str):
    '''加载全量微调的checkpoint'''
    print(f"Loading full finetune checkpoint: {checkpoint_path}")

    # 加载checkpoint
    state_dict = load_state_dict(checkpoint_path)

    # 筛选dit相关的权重（排除lora相关）
    dit_weights = {}
    for key, value in state_dict.items():
        # 跳过lora相关权重
        if 'lora' in key.lower():
            continue
        dit_weights[key] = value

    # 清理权重key（移除前缀）
    weights_clean = {}
    for key, value in dit_weights.items():
        clean_key = key
        for prefix in ['dit.', 'model.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        weights_clean[clean_key] = value

    print(f"Loading {len(weights_clean)} weights into DIT model...")

    # 加载到DIT模型中
    missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(
        weights_clean, strict=False
    )

    print(f"✓ Loaded {len(weights_clean) - len(unexpected_keys)} weights")

然后在 __init__ 中:
    if self.is_full_finetune:
        self.load_full_finetune_checkpoint(lora_checkpoint_path)
    else:
        self.load_lora_with_base_weights(lora_checkpoint_path, alpha=1.0)
""")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
推荐使用方案B，创建专门的 load_full_finetune_checkpoint() 函数:
  1. 更加简单和直接
  2. 加载所有训练过的权重
  3. 避免遗漏任何重要权重
  4. 代码更加清晰易懂

这比重用 load_checkpoint_weights() 更加安全和可靠。
""")
