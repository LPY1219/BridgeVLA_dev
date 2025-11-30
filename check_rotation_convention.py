#!/usr/bin/env python3
"""
检查训练数据中的旋转表示方式，并与RigidTransform.euler_angles进行对比
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# 测试用例：创建几个已知的旋转，验证euler_angles和as_euler('xyz')是否一致

print("="*80)
print("测试旋转表示的一致性")
print("="*80)

# 测试1: 绕Z轴旋转90度
print("\n测试1: 绕Z轴旋转90度")
quat_wxyz = np.array([0.7071068, 0, 0, 0.7071068])  # [w, x, y, z]
quat_xyzw = np.array([0, 0, 0.7071068, 0.7071068])  # [x, y, z, w] for scipy

r = R.from_quat(quat_xyzw)
euler_xyz = r.as_euler('xyz', degrees=True)
print(f"  Quaternion (wxyz): {quat_wxyz}")
print(f"  Euler angles (xyz, degrees): {euler_xyz}")

# 测试2: 绕X轴旋转45度
print("\n测试2: 绕X轴旋转45度")
quat_wxyz = np.array([0.9238795, 0.3826834, 0, 0])
quat_xyzw = np.array([0.3826834, 0, 0, 0.9238795])

r = R.from_quat(quat_xyzw)
euler_xyz = r.as_euler('xyz', degrees=True)
print(f"  Quaternion (wxyz): {quat_wxyz}")
print(f"  Euler angles (xyz, degrees): {euler_xyz}")

# 测试3: 组合旋转
print("\n测试3: 组合旋转 (roll=30°, pitch=20°, yaw=10°)")
r = R.from_euler('xyz', [30, 20, 10], degrees=True)
quat_xyzw = r.as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
print(f"  Input euler (xyz, degrees): [30, 20, 10]")
print(f"  Quaternion (wxyz): {quat_wxyz}")

# 反向转换验证
r_back = R.from_quat(quat_xyzw)
euler_back = r_back.as_euler('xyz', degrees=True)
print(f"  Recovered euler (xyz, degrees): {euler_back}")

print("\n" + "="*80)
print("检查训练数据中的旋转处理")
print("="*80)

# 读取一个训练样本，查看旋转是如何存储的
try:
    sys.path.insert(0, '/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio')
    from diffsynth.trainers.heatmap_dataset_mv_with_rot_grip import HeatmapDatasetFactory

    data_root = "/data/Franka_data/put_the_lion_on_the_top_shelf"
    dataset = HeatmapDatasetFactory.create_robot_trajectory_dataset(
        data_root=data_root,
        sequence_length=12,
        image_size=(384, 384),
        scene_bounds=[0, -0.65, -0.05, 0.8, 0.55, 0.75],
        wan_type="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
        rotation_resolution=5.0,
        trail_start=1,
        trail_end=2,  # 只加载1个trail
        mode='train'
    )

    print(f"\nDataset size: {len(dataset)}")

    # 获取第一个样本
    sample = dataset[0]

    # 查看rotation相关字段
    if 'start_pose' in sample:
        start_pose = sample['start_pose']
        print(f"\nstart_pose shape: {start_pose.shape}")
        print(f"start_pose (x,y,z,qx,qy,qz,qw): {start_pose}")

        # 提取四元数并转换为euler
        quat_xyzw = start_pose[3:7].cpu().numpy()
        r = R.from_quat(quat_xyzw)
        euler = r.as_euler('xyz', degrees=True)
        print(f"Start rotation (euler xyz, degrees): {euler}")

    if 'start_rotation' in sample:
        start_rotation = sample['start_rotation']
        print(f"\nstart_rotation (discretized bins): {start_rotation}")

    if 'rotation_delta_targets' in sample:
        rotation_delta = sample['rotation_delta_targets']
        print(f"\nrotation_delta_targets shape: {rotation_delta.shape}")
        print(f"First 3 frames rotation_delta (bins): {rotation_delta[:3]}")

except Exception as e:
    print(f"\n无法加载训练数据: {e}")

print("\n" + "="*80)
print("总结")
print("="*80)
print("""
训练数据使用的旋转表示:
  1. 从四元数(qx,qy,qz,qw)转换为欧拉角: r.as_euler('xyz', degrees=True)
  2. 这给出的是内旋(intrinsic)XYZ欧拉角，范围[-180, 180]
  3. 通过discretize转换为bins: (euler + 180) / resolution

实际机器人接口需要确保:
  1. RigidTransform.euler_angles 返回的也是相同的XYZ内旋欧拉角
  2. 如果不是，需要进行转换

可以通过运行机器人并查看debug输出来验证:
  - euler_angles property的值
  - 从quaternion通过as_euler('xyz')计算的值
  - 如果两者不一致，说明存在定义不匹配
""")
