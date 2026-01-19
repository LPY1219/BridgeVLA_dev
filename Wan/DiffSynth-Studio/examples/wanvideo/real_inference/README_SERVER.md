# RoboWan Server-Client 部署指南

## 概述

RoboWan Server 是一个基于 FastAPI 的推理服务，用于多视角热力图预测，同时预测机器人的位置、旋转和夹爪状态。

## 架构

```
┌─────────────┐         HTTP/REST API        ┌──────────────┐
│   Client    │ ────────────────────────────> │    Server    │
│             │                                │              │
│ - 发送图像  │ <────────────────────────────  │ - 加载模型   │
│ - 发送指令  │     Rotation + Gripper        │ - 推理预测   │
│ - 接收预测  │                                │ - 返回结果   │
└─────────────┘                                └──────────────┘
```

## 文件结构

```
real_inference/
├── RoboWan_server_fastapi.py  # Server 主文件
├── run_server.sh              # Server 启动脚本
├── test_client.py             # 测试 Client 示例
└── README_SERVER.md           # 本文档
```

## Server 端

### 1. 启动 Server

#### 方法 1: 使用启动脚本（推荐）

```bash
cd /home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/real_inference
bash run_server.sh
```

#### 方法 2: 手动启动

```bash
python3 RoboWan_server_fastapi.py \
    --lora_checkpoint "/path/to/lora/checkpoint.safetensors" \
    --rot_grip_checkpoint "/path/to/rot_grip/checkpoint.pth" \
    --model_base_path "/path/to/base/model" \
    --wan_type "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP" \
    --rotation_resolution 5.0 \
    --hidden_dim 512 \
    --num_rotation_bins 72 \
    --device "cuda:0" \
    --host "0.0.0.0" \
    --port 5555
```

### 2. 服务器参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lora_checkpoint` | LoRA checkpoint 路径 | 必填 |
| `--rot_grip_checkpoint` | 旋转/夹爪预测器 checkpoint 路径 | 必填 |
| `--model_base_path` | 基础模型路径 | 必填 |
| `--wan_type` | 模型类型 | `5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP` |
| `--use_dual_head` | 是否使用双头模式 | `True` |
| `--rotation_resolution` | 旋转离散化分辨率（度） | `5.0` |
| `--hidden_dim` | 隐藏层维度 | `512` |
| `--num_rotation_bins` | 旋转bins数量 | `72` |
| `--device` | 推理设备 | `cuda:0` |
| `--host` | 绑定地址 | `0.0.0.0` |
| `--port` | 监听端口 | `5555` |

### 3. API 端点

#### `/health` - 健康检查

**方法**: GET

**返回**:
```json
{
    "status": "healthy",
    "timestamp": "2025-01-16T10:30:00"
}
```

#### `/predict` - 动作预测

**方法**: POST

**输入**:
- `heatmap_images`: 多视角热力图图像 (List[File])
- `rgb_images`: 多视角RGB图像 (List[File])
- `prompt`: 任务指令 (str)
- `initial_rotation`: 初始旋转角度 "roll,pitch,yaw" (str)
- `initial_gripper`: 初始夹爪状态 0或1 (int)
- `num_frames`: 预测帧数 (int, 默认12)

**返回**:
```json
{
    "success": true,
    "position": [
        [0.4, 0.1, 0.3],
        [0.41, 0.12, 0.31],
        ...
    ],
    "rotation": [
        [-180.0, 0.0, 0.0],
        [-180.0, 5.0, 0.0],
        ...
    ],
    "gripper": [1, 1, 1, 0, ...],
    "error": null
}
```

**注意**: `position`、`rotation` 和 `gripper` 数组的长度为 `num_frames - 1`（不包括初始帧的预测，因为初始状态已知）。

## Client 端

### 1. 使用 Python Client

```python
from test_client import RoboWanClient
from PIL import Image

# 初始化 client
client = RoboWanClient(server_url="http://localhost:5555")

# 检查服务器健康状态
if not client.check_health():
    print("Server is not healthy!")
    exit(1)

# 准备输入
heatmap_images = [Image.open(f"heatmap_view{i}.png") for i in range(3)]
rgb_images = [Image.open(f"rgb_view{i}.png") for i in range(3)]

# 发送预测请求
result = client.predict(
    heatmap_images=heatmap_images,
    rgb_images=rgb_images,
    prompt="put the lion on the top shelf",
    initial_rotation=[-180.0, 0.0, 0.0],  # [roll, pitch, yaw]
    initial_gripper=1,
    num_frames=12
)

# 处理结果
if result['success']:
    positions = result['position']  # (num_frames-1, 3) in meters
    rotations = result['rotation']  # (num_frames-1, 3) in degrees
    grippers = result['gripper']    # (num_frames-1,)
    print(f"Predicted {len(rotations)} frames")
    print(f"First position: {positions[0]}")
else:
    print(f"Error: {result['error']}")
```

### 2. 运行测试 Client

```bash
# 确保服务器正在运行
python3 test_client.py
```

### 3. 使用 curl 测试

```bash
# 健康检查
curl http://localhost:5555/health

# 预测请求（需要准备图像文件）
curl -X POST http://localhost:5555/predict \
    -F "heatmap_images=@heatmap_0.png" \
    -F "heatmap_images=@heatmap_1.png" \
    -F "heatmap_images=@heatmap_2.png" \
    -F "rgb_images=@rgb_0.png" \
    -F "rgb_images=@rgb_1.png" \
    -F "rgb_images=@rgb_2.png" \
    -F "prompt=put the lion on the top shelf" \
    -F "initial_rotation=-180.0,0.0,0.0" \
    -F "initial_gripper=1" \
    -F "num_frames=12"
```

## 数据格式说明

### 输入格式

1. **heatmap_images**:
   - 类型: List[PIL.Image]
   - 数量: 3张（对应3个视角）
   - 尺寸: 256x256
   - 格式: RGB或灰度图

2. **rgb_images**:
   - 类型: List[PIL.Image]
   - 数量: 3张（对应3个视角）
   - 尺寸: 256x256
   - 格式: RGB

3. **initial_rotation**:
   - 格式: [roll, pitch, yaw]
   - 单位: 度（degrees）
   - 范围: [-180, 180]

4. **initial_gripper**:
   - 格式: int
   - 值: 0 (打开) 或 1 (关闭)

### 输出格式

1. **position**:
   - 类型: List[List[float]]
   - 形状: (num_frames-1, 3)
   - 含义: 每帧的 [x, y, z] 位置（米）
   - 注意: 不包括初始帧，因为初始位置通过heatmap已知

2. **rotation**:
   - 类型: List[List[float]]
   - 形状: (num_frames-1, 3)
   - 含义: 每帧的 [roll, pitch, yaw] 角度（度）

3. **gripper**:
   - 类型: List[int]
   - 形状: (num_frames-1,)
   - 含义: 每帧的夹爪状态 (0或1)

## 性能优化建议

1. **批处理**: 如果需要处理多个样本，考虑在client端批量发送
2. **GPU内存**: 根据GPU内存调整 `num_frames` 参数
3. **并发**: Server支持异步处理，可以同时发送多个请求
4. **网络**: 对于大图像，考虑压缩或降低分辨率

## 故障排查

### Server 无法启动

1. 检查 checkpoint 文件是否存在
2. 检查 CUDA 是否可用: `python -c "import torch; print(torch.cuda.is_available())"`
3. 检查端口是否被占用: `lsof -i :5555`

### Client 连接失败

1. 检查 server 是否运行: `curl http://localhost:5555/health`
2. 检查防火墙设置
3. 检查 server URL 是否正确

### 预测失败

1. 检查输入图像格式和尺寸
2. 检查 initial_rotation 格式是否正确
3. 查看 server 日志获取详细错误信息

## 开发与扩展

### 添加新的预测功能

1. 在 `RoboWanInferenceEngine.predict()` 中添加新的输出
2. 更新 `PredictResponse` Pydantic 模型
3. 在 `/predict` endpoint 中返回新字段

### 添加新的 API 端点

```python
@app.post("/your_endpoint")
async def your_endpoint(...):
    # 实现逻辑
    pass
```

## 许可证

与 BridgeVLA 项目保持一致
