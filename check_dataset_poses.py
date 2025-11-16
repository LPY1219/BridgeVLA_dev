#!/usr/bin/env python3
"""
检查数据集中实际的pose值
"""
import numpy as np
import os
from scipy.spatial.transform import Rotation

def quaternion_to_euler_bins(quat, rotation_resolution=5.0):
    """将四元数转换为欧拉角bins（与训练代码完全一致）"""
    # 归一化四元数
    quat_normalized = quat / np.linalg.norm(quat)

    # 确保w为正数
    if quat_normalized[3] < 0:
        quat_normalized = -quat_normalized

    # 使用scipy的Rotation转换（scipy使用[x, y, z, w]顺序）
    r = Rotation.from_quat(quat_normalized)
    euler = r.as_euler("xyz", degrees=True)  # (3,) - [roll, pitch, yaw]

    # 应用gimble fix
    if 89 < euler[1] < 91:
        euler[1] = 90
        r = Rotation.from_euler("xyz", euler, degrees=True)
        euler = r.as_euler("xyz", degrees=True)
    elif -91 < euler[1] < -89:
        euler[1] = -90
        r = Rotation.from_euler("xyz", euler, degrees=True)
        euler = r.as_euler("xyz", degrees=True)

    # 将范围从[-180, 180]转换为[0, 360]
    euler_shifted = euler + 180

    # 离散化
    disc = np.around(euler_shifted / rotation_resolution).astype(np.int64)
    # 处理边界情况：360度 = 0度
    num_bins = int(360 / rotation_resolution)
    disc[disc == num_bins] = 0

    return euler, disc


# 读取trail_0的poses
trail_path = "/data/wxn/V2W_Real/put_the_lion_on_the_top_shelf/trail_0"
poses_path = os.path.join(trail_path, "poses")

# 列出poses目录中的文件
import pickle

pose_files = sorted([f for f in os.listdir(poses_path) if f.endswith('.pkl')])
print(f"Found {len(pose_files)} pose files")

# 读取前5个pose
print("\n" + "="*80)
print("检查前5个pose的quaternion和euler angles:")
print("="*80)

for i, pose_file in enumerate(pose_files[:5]):
    with open(os.path.join(poses_path, pose_file), 'rb') as f:
        pose = pickle.load(f, encoding='latin1')
    # pose应该是 (7,) - [x, y, z, qx, qy, qz, qw]

    position = pose[:3]
    quaternion = pose[3:7]  # [qx, qy, qz, qw]

    euler, bins = quaternion_to_euler_bins(quaternion)

    print(f"\nFrame {i} ({pose_file}):")
    print(f"  Position: {position}")
    print(f"  Quaternion [qx, qy, qz, qw]: {quaternion}")
    print(f"  Euler [roll, pitch, yaw]: {euler}")
    print(f"  Bins [roll, pitch, yaw]: {bins}")

# 特别检查第100帧（对应测试样本）
print("\n" + "="*80)
print("检查frame 100的pose (对应测试样本index 100):")
print("="*80)

if len(pose_files) > 100:
    with open(os.path.join(poses_path, pose_files[100]), 'rb') as f:
        pose = pickle.load(f, encoding='latin1')
    position = pose[:3]
    quaternion = pose[3:7]

    euler, bins = quaternion_to_euler_bins(quaternion)

    print(f"Frame 100:")
    print(f"  Position: {position}")
    print(f"  Quaternion [qx, qy, qz, qw]: {quaternion}")
    print(f"  Euler [roll, pitch, yaw]: {euler}")
    print(f"  Bins [roll, pitch, yaw]: {bins}")
    print(f"\n期望的bins应该是: [0, 37, 30]")
    print(f"实际计算的bins是: {bins}")
    print(f"匹配? {np.array_equal(bins, np.array([0, 37, 30]))}")
else:
    print(f"只有 {len(pose_files)} 个pose文件，无法检查frame 100")
