# 旋转预测问题分析

## 问题描述
用户报告："旋转的预测比较奇怪，是不是训练和测试对于旋转的定义也没有对上"

类似于夹爪问题，旋转可能也存在训练数据和实际执行之间的定义不匹配。

## 旋转数据流分析

### 1. 训练数据处理 (heatmap_utils.py:592)
```python
r = Rotation.from_quat(quat_normalized)  # quat: [qx, qy, qz, qw]
euler = r.as_euler("xyz", degrees=True)  # 使用scipy的XYZ内旋欧拉角
# 范围: [-180, 180] 度
```

**训练数据中的旋转表示:**
- 从四元数转换为欧拉角using scipy's `as_euler('xyz')`
- 使用**内旋(intrinsic) XYZ**顺序
- 角度范围: **[-180, 180]** 度
- 计算delta: `future_euler - start_euler`, 归一化到 [-180, 180]

### 2. 推理服务器 (heatmap_inference...py:639)
```python
# 预测rotation_delta (相对于第一帧的变化量)
rotation_predictions = initial_rotation_degrees + rotation_delta_degrees
# 归一化到 [-180, 180]
rotation_predictions = ((rotation_predictions + 180) % 360) - 180
```

**服务器返回:**
- 绝对欧拉角 (度), 范围 [-180, 180]
- 假设initial_rotation也是scipy XYZ格式

### 3. 客户端读取机器人状态 (RoboWan_client.py:881-882)
```python
current_rotation_rad = current_pose_obj.euler_angles  # [roll, pitch, yaw] in radians
current_rotation_deg = np.rad2deg(current_rotation_rad).tolist()
```

**关键问题:** `current_pose_obj.euler_angles` 返回的是什么格式的欧拉角？
- 是否是 XYZ 内旋欧拉角？
- 是否与 scipy 的 `as_euler('xyz')` 一致？

### 4. 客户端执行动作 (robot_interface.py:240)
```python
# is_relative=False (绝对动作)
rot_euler = action[3:6]  # radians
rot_mat = R.from_euler('xyz', rot_euler).as_matrix()
```

**动作执行:**
- 使用 `R.from_euler('xyz', rot_euler)` 创建旋转矩阵
- 这使用scipy的XYZ内旋格式

## 潜在问题

### 可能的不匹配点

1. **autolab_core的RigidTransform.euler_angles可能使用不同的欧拉角约定**
   - 如果它使用ZYX顺序，或外旋(extrinsic)顺序，就会不匹配
   - 如果它使用不同的角度范围，也会有问题

2. **角度归一化问题**
   - 训练数据: [-180, 180]
   - RigidTransform: 可能返回不同范围

3. **Gimbal Lock处理**
   - 训练数据有gimbal fix (heatmap_utils.py:515-523)
   - RigidTransform可能没有这个处理

## 诊断方法

### 已添加的Debug代码 (RoboWan_client.py:884-891)
```python
# 对比两种方法获得的欧拉角
current_rotation_rad = current_pose_obj.euler_angles
current_rotation_deg = np.rad2deg(current_rotation_rad)

quat_scipy = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
euler_from_quat = R.from_quat(quat_scipy).as_euler("xyz", degrees=True)

rospy.loginfo(f"[DEBUG Rotation] euler_angles: {current_rotation_deg}")
rospy.loginfo(f"[DEBUG Rotation] from quaternion (xyz): {euler_from_quat.tolist()}")
```

**如果这两个值不一致，说明存在定义不匹配！**

### 验证步骤

1. 运行客户端，查看debug输出
2. 对比 `euler_angles` 和 `from quaternion (xyz)` 的值
3. 如果不同，需要确定差异的模式:
   - 是顺序不同? (例如 ZYX vs XYZ)
   - 是符号不同? (例如右手系 vs 左手系)
   - 是范围不同? (例如 [0, 360] vs [-180, 180])

## 可能的修复方案

### 修复1: 如果euler_angles使用不同顺序

假设euler_angles返回ZYX顺序，需要转换:

```python
# RoboWan_client.py 修改
current_rotation_rad = current_pose_obj.euler_angles  # 假设是 ZYX
# 转换为 XYZ 用于发送给服务器
from scipy.spatial.transform import Rotation as R
rot_mat = R.from_euler('zyx', current_rotation_rad).as_matrix()
current_rotation_xyz = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
```

### 修复2: 如果只是范围问题

```python
# 归一化到 [-180, 180]
def normalize_angle(angle):
    return ((angle + 180) % 360) - 180

current_rotation_deg = [normalize_angle(a) for a in current_rotation_deg]
```

### 修复3: 统一使用四元数作为中间格式

最安全的方法是完全绕过euler_angles:

```python
# RoboWan_client.py - 读取当前状态
current_quat = current_pose_obj.quaternion  # [w, x, y, z]
quat_scipy = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
current_rotation_deg = R.from_quat(quat_scipy).as_euler("xyz", degrees=True)
```

这样确保使用的是与训练数据完全一致的转换方法。

## 推荐的修复 (需要验证后确定)

**最安全的做法是直接使用四元数转欧拉角，完全避免依赖RigidTransform.euler_angles:**

```python
# 在 RoboWan_client.py:881 处修改

# 旧代码:
# current_rotation_rad = current_pose_obj.euler_angles
# current_rotation_deg = np.rad2deg(current_rotation_rad).tolist()

# 新代码: 直接从四元数转换，确保与训练数据一致
from scipy.spatial.transform import Rotation as R
current_quat = current_pose_obj.quaternion  # [w, x, y, z]
# 转换为 scipy 格式 [x, y, z, w]
quat_scipy = np.array([
    current_quat[1],
    current_quat[2],
    current_quat[3],
    current_quat[0]
])
# 使用与训练数据完全相同的转换方法
current_rotation_deg = R.from_quat(quat_scipy).as_euler("xyz", degrees=True).tolist()
```

这样可以确保:
1. 使用与训练数据完全相同的欧拉角转换 (`as_euler("xyz")`)
2. 避免依赖RigidTransform的euler_angles实现
3. 与heatmap_utils.py:592的处理完全一致

## 下一步

1. **先运行机器人查看debug输出，确认是否存在不匹配**
2. **如果确认有问题，应用上述修复**
3. **测试修复后的旋转预测是否正常**
