# Single View Heatmap Sequence Prediction

基于Wan2.2的单帧RGB图像到heatmap序列预测模型。该项目利用Wan2.2的视频生成能力，通过colormap编码方式处理heatmap数据，实现从首帧RGB到后续heatmap轨迹的预测。

## 快速开始

### 1. 环境检查

首先运行设置测试脚本来验证所有组件是否正常工作：

```bash
cd /share/project/lpy/BridgeVLA/Wan/single_view
python test_setup.py
```

如果所有测试通过，你会看到 "🎉 ALL TESTS PASSED!" 的消息。

### 2. 准备数据

确保你的数据按以下结构组织：

```
data_root/
├── trail_1/
│   ├── poses/
│   │   ├── 000000.pkl
│   │   ├── 000001.pkl
│   │   └── ...
│   ├── pcd/
│   │   ├── 000000.pkl
│   │   ├── 000001.pkl
│   │   └── ...
│   ├── 3rd_bgr/
│   │   ├── 000000.pkl
│   │   ├── 000001.pkl
│   │   └── ...
│   └── instruction.txt
├── trail_2/
│   └── ...
└── ...
```

### 3. 开始训练

#### 调试模式（推荐首次使用）

```bash
python run_training.py --data-root /path/to/your/data --debug
```

调试模式使用较小的配置，适合快速验证训练流程。

#### 正常训练模式

```bash
python run_training.py --data-root /path/to/your/data --output-dir ./outputs
```

#### 自定义参数

```bash
python run_training.py \
    --data-root /path/to/your/data \
    --output-dir ./my_training \
    --batch-size 4 \
    --epochs 50 \
    --learning-rate 1e-4 \
    --sequence-length 10 \
    --num-workers 2
```

### 4. 监控训练

训练日志和检查点会保存在指定的输出目录中：

```
outputs/
├── checkpoints/
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   └── checkpoint_epoch_*.pth
├── logs/
│   └── events.out.tfevents.*  # TensorBoard日志
├── configs/
│   ├── experiment_config.json
│   └── training_config.json
└── visualizations/
    └── epoch_*_sample_*.png
```

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir outputs/logs
```

## 配置说明

### 预设配置

- **default**: 标准训练配置，适合完整训练
- **debug**: 调试配置，使用较小的参数和较少的epoch，适合快速测试

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root` | 必须 | 训练数据根目录 |
| `--output-dir` | ./outputs | 输出目录 |
| `--batch-size` | 8 (default) / 2 (debug) | 批次大小 |
| `--epochs` | 100 (default) / 5 (debug) | 训练轮数 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--sequence-length` | 10 (default) / 5 (debug) | 预测序列长度 |
| `--device` | auto | 设备选择 (auto/cuda/cpu) |
| `--num-workers` | 4 | 数据加载工作进程数 |

## 项目结构

```
single_view/
├── configs/                  # 配置文件
│   ├── model_config.py      # 模型配置
│   └── training_config.py   # 训练配置
├── data/                    # 数据处理模块
│   ├── dataset.py          # 数据集定义
│   └── dataloader.py       # 数据加载器
├── models/                  # 模型定义
│   ├── wan_heatmap_model.py # 主模型
│   └── sequence_generator.py # 序列生成器
├── utils/                   # 工具函数
│   ├── colormap_utils.py    # colormap转换
│   ├── heatmap_utils.py     # heatmap处理
│   └── visualization_utils.py # 可视化工具
├── experiments/             # 实验脚本
│   └── train.py            # 训练主脚本
├── run_training.py         # 训练启动脚本
├── test_setup.py          # 设置测试脚本
└── README.md              # 本文件
```

## 技术原理

### 核心思路

1. **Colormap编码**: 将单通道heatmap转换为RGB colormap格式
2. **Wan2.2处理**: 利用Wan2.2的VAE进行编码和重建
3. **序列生成**: 基于RGB图像条件生成heatmap序列
4. **Colormap解码**: 将生成的colormap序列转回heatmap

### 数据流

```
RGB图像 → [条件] → Wan2.2 → Heatmap序列
     ↑                          ↓
Heatmap → Colormap → VAE编码 → 潜在表示
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'diffusers'**
   ```bash
   pip install diffusers
   ```

2. **CUDA out of memory**
   - 减小batch size: `--batch-size 2`
   - 使用较小的图像尺寸
   - 使用debug配置: `--debug`

3. **Data loading errors**
   - 检查数据路径是否正确
   - 确保数据格式符合要求
   - 检查文件权限

4. **训练速度慢**
   - 增加num_workers: `--num-workers 8`
   - 使用较小的sequence_length
   - 启用混合精度训练

### 性能优化

- **内存优化**: 模型会自动启用attention slicing来节省内存
- **数据加载**: 调整num_workers以平衡CPU和GPU使用
- **批次大小**: 根据GPU内存调整batch_size

## 依赖环境

### 主要依赖

- PyTorch >= 1.12.0
- diffusers (包含AutoencoderKLWan)
- numpy, matplotlib
- opencv-python
- pillow
- scipy
- tqdm

### 可选依赖

- tensorboard (训练监控)
- scikit-image (高级peak检测)
- transformers (高级学习率调度)

## 联系信息

如有问题，请检查：
1. 运行 `python test_setup.py` 验证环境
2. 检查数据格式和路径
3. 查看训练日志中的错误信息