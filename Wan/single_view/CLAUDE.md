# Single View Heatmap Sequence Prediction Pipeline

## 项目概述

这是一个基于单帧RGB图像预测未来heatmap序列的深度学习pipeline。该项目利用Wan2.2模型的视频生成能力，通过colormap编码方式处理heatmap数据，实现从首帧RGB到后续heatmap轨迹的预测。

## 项目结构

```
single_view/
├── CLAUDE.md                  # 项目文档（本文件）
├── data/                      # 数据处理模块
│   ├── __init__.py
│   ├── dataloader.py         # 数据加载器
│   └── dataset.py            # 数据集类定义
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── wan_heatmap_model.py  # Wan2.2 heatmap预测模型
│   └── sequence_generator.py # 序列生成器
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── heatmap_utils.py      # heatmap处理工具
│   ├── colormap_utils.py     # colormap编码/解码工具
│   └── visualization_utils.py # 可视化工具
├── configs/                   # 配置文件
│   ├── __init__.py
│   ├── training_config.py    # 训练配置
│   └── model_config.py       # 模型配置
├── visualization/             # 可视化模块
│   ├── __init__.py
│   ├── heatmap_animator.py   # heatmap动画生成
│   └── comparison_plots.py   # 对比图表生成
├── experiments/               # 实验脚本
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── inference.py          # 推理脚本
└── docs/                      # 文档目录
    ├── setup.md              # 环境配置
    ├── usage.md              # 使用说明
    └── api.md                # API文档
```

## 核心功能模块

### 1. 数据处理 (data/)

#### dataloader.py
- **功能**: 实现RGB-to-heatmap序列的数据加载
- **输入**: 训练集和测试集路径
- **输出**: 首帧RGB图像和对应的ground truth heatmap序列
- **关键方法**: `__getitem__()` 返回 (rgb_frame, heatmap_sequence)

#### dataset.py
- **功能**: 数据集类定义，处理数据路径、预处理等
- **支持格式**: 支持多种数据格式的heatmap和RGB图像

### 2. 模型架构 (models/)

#### wan_heatmap_model.py
- **功能**: 基于Wan2.2的heatmap序列生成模型
- **输入**: 首帧RGB图像 (H, W, 3)
- **输出**: heatmap序列 (T, H, W) 通过colormap编码
- **架构**: 利用Wan2.2的video generation能力

#### sequence_generator.py
- **功能**: 序列生成器，控制生成长度和质量
- **特性**: 支持不同长度的序列生成，优化推理速度

### 3. 工具函数 (utils/)

#### heatmap_utils.py
- **功能**: heatmap相关的处理函数
- **包含**:
  - heatmap标准化和去标准化
  - peak检测和精度计算
  - heatmap质量评估

#### colormap_utils.py
- **功能**: heatmap与colormap的相互转换
- **基于**: `/share/project/lpy/BridgeVLA/Wan/reconstruct_heatmap/test_heatmap_peak_accuracy.py`中的方法
- **包含**:
  - `convert_heatmap_to_colormap()`: heatmap → RGB colormap
  - `extract_heatmap_from_colormap()`: RGB colormap → heatmap
  - `convert_color_to_wan_format()`: 格式转换为Wan VAE输入
  - `convert_from_wan_format()`: Wan VAE输出格式转换

#### visualization_utils.py
- **功能**: 可视化相关的工具函数
- **包含**: 图表生成、动画制作等辅助函数

### 4. 可视化模块 (visualization/)

#### heatmap_animator.py
- **功能**: 生成heatmap序列的动图
- **输出**: GIF或MP4格式的动画
- **对比**: 支持预测序列与ground truth的并排对比

#### comparison_plots.py
- **功能**: 生成静态对比图表
- **包含**: 精度分析、误差分布、peak追踪等可视化

### 5. 配置管理 (configs/)

#### training_config.py
- **功能**: 训练相关的配置参数
- **包含**: 学习率、batch size、epoch数、优化器设置等

#### model_config.py
- **功能**: 模型相关的配置参数
- **包含**: Wan2.2模型路径、输入尺寸、序列长度等

## 核心技术路线

### 1. Heatmap编码策略
采用colormap作为中转格式，利用Wan2.2的图像/视频生成能力：

```
Heatmap (单通道) → Colormap (RGB) → Wan2.2处理 → Colormap (RGB) → Heatmap (单通道)
```

### 2. 模型训练流程
1. **数据准备**: RGB首帧 + heatmap序列pairs
2. **编码**: heatmap序列转换为colormap视频
3. **训练**: Wan2.2模型学习RGB到colormap视频的映射
4. **解码**: 推理时将生成的colormap序列转回heatmap

### 3. 评估指标
- **Peak精度**: 预测peak位置与ground truth的距离误差
- **序列一致性**: 相邻帧之间的连续性评估
- **生成质量**: heatmap的形状和强度分布评估

## 依赖环境

### 主要依赖
- PyTorch >= 1.12.0
- diffusers (包含AutoencoderKLWan)
- numpy, matplotlib
- opencv-python
- pillow

### Wan2.2模型
- 模型路径: `/share/project/lpy/huggingface/Wan_2_2_TI2V_5B_Diffusers`
- VAE子模块: 用于heatmap的编码解码

## 使用方法

### 训练模型
```bash
cd /share/project/lpy/BridgeVLA/Wan/single_view
python experiments/train.py --config configs/training_config.py
```

### 推理预测
```bash
python experiments/inference.py --input_rgb path/to/rgb.jpg --output_dir results/
```

### 评估性能
```bash
python experiments/evaluate.py --test_data path/to/test --model_path checkpoints/model.pth
```

## 开发计划

### Phase 1: 基础框架 ✅
- [x] 项目目录结构搭建
- [x] CLAUDE.md文档创建
- [ ] 核心工具函数实现

### Phase 2: 数据处理
- [ ] dataloader实现
- [ ] dataset类定义
- [ ] 数据预处理pipeline

### Phase 3: 模型开发
- [ ] Wan2.2集成
- [ ] heatmap序列生成器
- [ ] 训练循环实现

### Phase 4: 可视化与评估
- [ ] 动画生成器
- [ ] 对比可视化
- [ ] 性能评估工具

### Phase 5: 优化与部署
- [ ] 推理速度优化
- [ ] 内存使用优化
- [ ] 批量处理支持

## 注意事项

1. **内存管理**: Wan2.2模型较大，需要注意GPU内存使用
2. **数据格式**: 确保输入RGB和heatmap的尺寸匹配
3. **colormap选择**: 使用viridis colormap保证最佳的编码解码精度
4. **序列长度**: 根据实际需求调整预测序列的长度

## 参考资料

- Wan2.2模型文档
- Heatmap处理参考: `reconstruct_heatmap/test_heatmap_peak_accuracy.py`
- Calvin数据集处理参考: `RoboVLMs-dev/robovlms/data/calvin_dataset.py`

## Claude Code 默认配置

- **根目录**: `/share/project/lpy/BridgeVLA/Wan/single_view`
- **默认Conda环境**: `test`

## 联系信息

项目开发者: Claude Code Assistant
最后更新: 2025-09-22