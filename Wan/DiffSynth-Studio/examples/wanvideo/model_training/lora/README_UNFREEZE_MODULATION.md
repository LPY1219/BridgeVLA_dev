# Modulation & Norm 参数解冻功能说明

## 🎯 功能概述

新增了 `--unfreeze_modulation_and_norms` 参数，用于控制是否解冻以下参数：
- **modulation**: AdaLN 调制参数 (~0.55M 参数，**影响很大**）
- **mvs_attn.norm_q/norm_k**: 多视角注意力中的 RMSNorm (~0.18M 参数)

**总增加**: ~0.73M 参数（相比 LoRA 的 80M，增加不到 1%）

## 📊 为什么需要这个功能？

### 1. **Modulation (AdaLN) 的重要性**
```
AdaLN(x, t) = scale(t) * LayerNorm(x) + shift(t)
```
- 根据时间步 t 动态调整每层的归一化
- 对于不同任务（通用视频生成 vs 机械臂轨迹），时间步语义不同
- **影响程度**: ⭐⭐⭐⭐⭐（极高）

### 2. **norm_q/norm_k 的必要性**
- 多视角融合特征分布 ≠ 单视角特征分布
- LoRA 改变了 q/k 的输出，norm 也应相应调整
- **影响程度**: ⭐⭐⭐⭐（高）

## 🔄 向后兼容性设计

### 默认行为（`UNFREEZE_MODULATION_AND_NORMS=false`）
- **保持冻结** modulation 和 norm 参数
- **完全兼容**已有的 checkpoint
- 适用于测试旧模型

### 新行为（`UNFREEZE_MODULATION_AND_NORMS=true`）
- **解冻** modulation 和 norm 参数
- **更好的任务适应**性
- 适用于训练新模型

## 📝 使用方法

### 方案 A：测试旧 Checkpoint（默认）✅

**预训练脚本** (`Wan2.2-TI2V-5B_mvtrack_pretrain.sh`):
```bash
# 保持默认值 false
UNFREEZE_MODULATION_AND_NORMS=false

# 或者直接注释掉该行，使用默认值
# UNFREEZE_MODULATION_AND_NORMS=false
```

**微调脚本** (`Wan2.2-TI2V-5B_heatmap_rgb_mv_3.sh`):
```bash
# 保持默认值 false，与预训练保持一致
UNFREEZE_MODULATION_AND_NORMS=false
```

**结果**:
- ✅ 可以正常加载和测试旧的 checkpoint
- ✅ 训练策略与旧模型完全一致
- ⚠️  适应性可能稍弱

### 方案 B：训练新模型（推荐）✨

**预训练脚本**:
```bash
# 启用 modulation 和 norm 解冻
UNFREEZE_MODULATION_AND_NORMS=true
```

**微调脚本**:
```bash
# 同样启用，保持一致性
UNFREEZE_MODULATION_AND_NORMS=true
```

**结果**:
- ✅ 最大化任务适应性
- ✅ 参数量增加极小（<1%）
- ✅ 理论上性能更好
- ⚠️  与旧 checkpoint 不兼容

## 🔍 输出示例

### 当 `UNFREEZE_MODULATION_AND_NORMS=false` 时：
```
================================================================================
ℹ️  MODULATION PARAMETERS: KEEPING FROZEN (Backward Compatible)
================================================================================
  Modulation parameters will stay frozen (using pretrained values).
  To unfreeze, add --unfreeze_modulation_and_norms flag.
================================================================================
```

### 当 `UNFREEZE_MODULATION_AND_NORMS=true` 时：
```
================================================================================
⚡ MODULATION PARAMETER UNFREEZING
================================================================================

✅ Unfroze 30 modulation parameters:
  📦 AdaLN modulation: 30 parameters
  📊 Total: 552,960 parameters (~0.55M)

💡 训练策略:
  - Modulation: 全量训练（适应新任务的时间步调制模式）
================================================================================

================================================================================
MULTI-VIEW MODULE PARAMETER UNFREEZING
================================================================================
...
  📦 MVS_Attn Norms (全量训练): 60 parameters

💡 训练策略:
  - Projector & Modulation_mvs: 全量训练（从零开始学习）
  - MVS_Attn: LoRA微调（利用预训练知识，节省显存）
  - MVS_Attn norm_q/norm_k: 全量训练（适应多视角特征分布）
================================================================================
```

## ⚙️ 技术细节

### 修改的文件
1. **训练脚本**: `heatmap_train_mv.py`
   - 添加 `--unfreeze_modulation_and_norms` 参数
   - 新增 `_unfreeze_modulation()` 函数
   - 修改 `_unfreeze_mv_modules()` 支持 norm 解冻

2. **Shell 脚本**:
   - `Wan2.2-TI2V-5B_mvtrack_pretrain.sh`
   - `Wan2.2-TI2V-5B_heatmap_rgb_mv_3.sh`

### 参数传递流程
```
Shell 脚本变量 UNFREEZE_MODULATION_AND_NORMS
    ↓
accelerate launch --unfreeze_modulation_and_norms (条件)
    ↓
args.unfreeze_modulation_and_norms
    ↓
HeatmapWanTrainingModule.__init__(unfreeze_modulation_and_norms=...)
    ↓
self.unfreeze_modulation_and_norms
    ↓
_unfreeze_modulation() 和 _unfreeze_mv_modules()
```

## 📋 参数量对比

| 模块 | 参数量 | 相对于 LoRA (80M) |
|------|--------|------------------|
| LoRA (当前) | 80.61M | 100% |
| Patch & Head | 2.4M | 3% |
| MV modules (projector, modulation_mvs) | 307M | 381% |
| **新增: modulation** | **0.55M** | **0.68%** |
| **新增: mvs_attn norms** | **0.18M** | **0.22%** |
| **总新增** | **0.73M** | **0.91%** |

## 🎯 建议的使用策略

### 1. **测试已有模型** (当前场景)
```bash
# 两个脚本都设置为 false
UNFREEZE_MODULATION_AND_NORMS=false
```

### 2. **训练全新模型** (推荐)
```bash
# 两个脚本都设置为 true
UNFREEZE_MODULATION_AND_NORMS=true
```

### 3. **消融实验**
```bash
# 实验 A: 冻结（baseline）
UNFREEZE_MODULATION_AND_NORMS=false

# 实验 B: 解冻（改进）
UNFREEZE_MODULATION_AND_NORMS=true

# 对比性能差异
```

## ❓ 常见问题

### Q: 我已经训练了一个 checkpoint，设置的是 false，现在想改成 true 继续训练？
A: 不建议！会导致不一致：
- Checkpoint 中没有训练过的 modulation/norm 参数
- 新的训练会解冻这些参数，但它们的初始值来自 base model
- 建议：要么重新训练，要么保持 false

### Q: 我可以用 false 训练的模型加载到 true 的环境测试吗？
A: 可以！向后兼容是设计目标：
- Checkpoint 加载时会忽略 missing keys（modulation/norm）
- 这些参数会使用 base model 的值（与训练时一致）

### Q: 如果我不确定该用 true 还是 false？
A: 建议：
- **当前场景**（测试旧模型）: 用 false
- **训练新模型**: 用 true
- **不确定**: 先用 false（保守），后续可以做消融实验对比

## 📝 总结

这个功能提供了：
- ✅ **灵活性**: 可以选择是否解冻
- ✅ **向后兼容**: 默认保持旧行为
- ✅ **性能提升潜力**: 解冻后适应性更强
- ✅ **低成本**: 参数增加不到 1%

**推荐**: 新模型训练时启用 `UNFREEZE_MODULATION_AND_NORMS=true`，测试旧模型时保持 `false`。
