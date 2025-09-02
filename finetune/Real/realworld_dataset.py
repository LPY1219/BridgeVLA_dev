# import clip
import gc
import os
import pickle as pkl
import time

import cv2
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def read_action_file(action_dir_path):
    """
    从每一个eposide的actions中按顺序将夹爪进行排序
    返回夹爪位姿序列
    """
    action_files = os.listdir(action_dir_path)
    action_files.sort()
    result = []
    for action_path in action_files:
        path = os.path.join(action_dir_path, action_path)
        with open(path, "rb") as f:
            action = pkl.load(f)
            # extract XYZ
            position = [float(x) for x in action[:3]]

            # extract orientation (Rx Ry Rz)
            # 由于采集的数据是存储wxyz的四元数，将其转化为 rx ry rz的欧拉角（弧度）
            q = [float(x) for x in action[3:7]]
            # w x y z -> x y z w
            # r = R.from_quat([q[1], q[2], q[3], q[0]])

            # rx, ry, rz = r.as_euler("xyz", degrees=False)
            # orientation = [rx, ry, rz]
            # 输出四元数
            orientation = q # w x y z

            # 夹爪状态 1 表示打开 0 表示闭合
            gripper_state = float(action[7])

            entry = {
                "position": position,  # xyz
                "orientation": orientation,  # wxyz
                "gripper_state": gripper_state,
            }
            result.append(entry)

    return result


def build_extrinsic_matrix(translation, quaternion):
    """
    输入：
        translation: 长度3数组或列表 [tx, ty, tz]
        quaternion: 长度4数组或列表 [w, x, y, z]
    输出：
        4x4 numpy 外参矩阵 (相机坐标系 -> base 坐标系)
    """
    t = np.array(translation, dtype=np.float64)
    w, x, y, z = quaternion

    # 四元数转旋转矩阵
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ],
        dtype=np.float64,
    )

    # 拼成4x4外参矩阵
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def convert_pcd_to_base(extrinsic_martix, type="3rd", pcd=[]):
    transform = extrinsic_martix

    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)  # 去掉A
    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd


class Real_Dataset(Dataset):
    def __init__(self, data_path, device, cameras, ep_per_task=10):
        self.device = device
        self.data_path = data_path
        self.cameras = cameras
        self.train_data = []
        print(f"You use {ep_per_task} episodes per task")
        self.extrinsic_matrix = build_extrinsic_matrix(
            translation=np.array(
                [1.361550352259638, -0.16979593858972575, 0.7206894851921658]
            ),
            quaternion=np.array(
                [
                    0.3708036563584307,
                    -0.621104489139664,
                    -0.606290367152192,
                    0.33037229408689267,
                ]
            ),
        )
        time.sleep(5)
        self.construct_dataset(ep_per_task)

    def construct_dataset(self, ep_per_task=10):
        """构建数据集"""
        # 读取任务数量
        # 训练集格式：data/task1,task2,task3
        # 每一个task下面有 0,1,2,3,....个episode
        self.num_tasks = len(
            [
                path_name
                for path_name in os.listdir(self.data_path)
                if os.path.isdir(os.path.join(self.data_path, path_name))
            ]
        )
        self.num_task_paths = 0
        for task in os.listdir(self.data_path):
            task_path = os.path.join(self.data_path, task)
            if os.path.isdir(task_path):
                for episode_num in tqdm(
                    os.listdir(task_path)
                ):  # 一个任务下的多个episode
                    print("episode_num", episode_num)
                    if not episode_num.isdigit():
                        continue
                    if int(episode_num) >= ep_per_task:  # 一个任务最多ep个episode
                        print(f"episode num {episode_num} is larger than {ep_per_task}")
                        continue
                    self.num_task_paths += 1
                    episode_path = os.path.join(
                        task_path, episode_num
                    )  # 一个任务的某一个演示的地址

                    action_path = os.path.join(episode_path, "actions")
                    rgb_3rd = os.path.join(episode_path, "3rd_cam_rgb")
                    pcd_3rd = os.path.join(episode_path, "3rd_cam_pcd")

                    # 加载夹爪位姿
                    gripper_pose = read_action_file(action_path)

                    # 离散的步骤数量
                    num_steps = sum(
                        1
                        for file_name in os.listdir(rgb_3rd)
                        if file_name.endswith(".pkl")
                    )
                    for step in range(num_steps - 1):
                        sample = {}
                        #! next pose action
                        gripper_pose_xyz = gripper_pose[step + 1]["position"]
                        gripper_pose_quat = gripper_pose[step + 1]["orientation"] # wxyz
                        #! w x y z -> x y z w
                        gripper_pose_quat = np.array([gripper_pose_quat[1], gripper_pose_quat[2], gripper_pose_quat[3], gripper_pose_quat[0]]).astype(np.float32)  # wxyz -> xyzw

                        sample["gripper_pose"] = np.concatenate(
                            (
                                gripper_pose_xyz,
                                gripper_pose_quat,
                                [gripper_pose[step + 1]["gripper_state"]]
                            ),
                            axis=0,
                        ).astype(np.float32)
                        
                        current_gripper_pose_xyz = np.array(gripper_pose[step]["position"])
                        current_gripper_pose_quat = gripper_pose[step]["orientation"] # wxyz

                        # * 当前夹爪状态
                        current_gripper_state = gripper_pose[step]["gripper_state"]

                        time = (1.0 - (step / float(num_steps - 1))) * 2.0 - 1.0
                        sample["low_dim_state"] = np.concatenate(
                            [[current_gripper_state], [time]]
                        ).astype(np.float32)

                        sample["ignore_collisions"] = np.array([1.0], dtype=np.float32)
                        
                        sample['3rd'], sample['wrist'] = {}, {}

                        # * 采集相机信息 - 使用扁平化键名格式
                        if "3rd" in self.cameras:
                            # * RGB(实际上是BGR顺序存储的图像)
                            with open(os.path.join(rgb_3rd, f"{step}.pkl"), "rb") as f:
                                sample['3rd']['rgb'] = pkl.load(f)[:, :, :3]  # step时刻的rgb图像
                                sample['3rd']['rgb'] = np.ascontiguousarray(sample['3rd']['rgb'])
                                #! BGR --> RGB
                                # rgb_data = rgb_data[:, :, ::-1].copy()
                                sample['3rd']['rgb'] = np.ascontiguousarray(sample['3rd']['rgb'])[:, :, ::-1].copy()
                                sample['3rd']['rgb'] = np.transpose(sample['3rd']['rgb'], [2, 0, 1])
                                
                            # * 点云
                            with open(os.path.join(pcd_3rd, f"{step}.pkl"), "rb") as f:
                                sample['3rd']['pcd'] = pkl.load(f)[:, :, :3]
                                sample['3rd']['pcd'] = convert_pcd_to_base(
                                    extrinsic_martix=self.extrinsic_matrix,
                                    type="3rd",
                                    pcd=sample['3rd']['pcd'],
                                )
                                sample['3rd']['pcd'] = np.transpose(sample['3rd']['pcd'], [2, 0, 1]).astype(np.float32)
                            # 腕部相机省略
                        with open(
                            os.path.join(episode_path, "instruction.pkl"), "rb"
                        ) as f:
                            instruction = pkl.load(f)
                        sample["lang_goal"] = instruction.strip()
                        sample["tasks"] = task
                        self.train_data.append(sample)
        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]


def save_pcd_with_gripper_ply(
    pcd, rgb, gripper_pose_xyz, save_path, gripper_radius=0.02, gripper_density=1000
):
    """
    保存点云和RGB为一个ply文件，并在gripper_pose_xyz处添加红色球体
    Args:
        pcd: numpy.ndarray, shape (3, H, W)
        rgb: numpy.ndarray, shape (3, H, W), 值范围0~255或0~1
        gripper_pose_xyz: (3,) array-like, 夹爪空间坐标
        save_path: str, ply文件保存路径
        gripper_radius: float, 球体半径
        gripper_density: int, 球体点的数量
    """
    # 1. reshape为(N, 3)
    C, H, W = pcd.shape
    pcd_flat = pcd.reshape(C, -1).T  # (N, 3)
    #! 反色
    # rgb = rgb[::-1, :, :]  # bgr --> rgb
    rgb_flat = rgb.reshape(C, -1).T  # (N, 3)
    # 2. 去除NaN
    valid_mask = ~np.isnan(pcd_flat).any(axis=1)
    pcd_valid = pcd_flat[valid_mask]  # N, 3
    rgb_valid = rgb_flat[valid_mask]  # N, 3
    # 3. 归一化颜色到0~1
    if rgb_valid.max() > 1.1:
        rgb_valid = rgb_valid / 255.0
    # 4. 生成gripper球体点
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gripper_radius)
    sphere = sphere.sample_points_uniformly(number_of_points=gripper_density)
    sphere_points = np.asarray(sphere.points) + np.array(gripper_pose_xyz).reshape(1, 3)
    sphere_colors = np.tile(
        np.array([[1.0, 0.0, 0.0]]), (sphere_points.shape[0], 1)
    )  # 红色

    # 5. 合并
    pcd_all = np.vstack([pcd_valid, sphere_points])
    rgb_all = np.vstack([rgb_valid, sphere_colors])
    # 6. 构建open3d点云
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_all)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_all)

    # 7. 保存
    o3d.io.write_point_cloud(save_path, pcd_o3d)
    print(f"点云+gripper球体已保存到: {save_path}")


def save_image_from_array(arr, save_path):
    """
    将形状为(C,H,W)的numpy数组保存为图像文件。
    Args:
        arr: 按照BGR顺序，numpy.ndarray, shape (C,H,W), 值范围0~255或0~1
        save_path: str, 图像保存路径
    """
    # 确保数组是(C,H,W)格式
    if arr.shape[0] == 3:
        # 转换为(H,W,C)格式
        img = np.transpose(arr, (1, 2, 0))
        # 如果值范围是0~1，则缩放到0~255
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        #! 反色
        img = img[:, :, ::-1]
        cv2.imwrite(save_path, img)
        print(f"图像已保存到: {save_path}")
    else:
        raise ValueError("输入数组必须是(C,H,W)格式，且C=3")


def is_bgr_image(arr):
    """
    判断一个形状为(C,H,W)的numpy数组是否为BGR格式。
    通过比较第一个通道（B）和最后一个通道（R）的平均值来判断：
    如果第一个通道的平均值大于最后一个通道的平均值，则认为是BGR格式。
    Args:
        arr: numpy.ndarray, shape (C,H,W), 值范围0~255或0~1
    Returns:
        bool: True表示BGR格式，False表示RGB格式
    """
    if arr.shape[0] != 3:
        raise ValueError("输入数组必须是(C,H,W)格式，且C=3")
    # 计算第一个通道（B）和最后一个通道（R）的平均值
    avg_b = np.mean(arr[0])
    avg_r = np.mean(arr[2])
    return avg_b > avg_r


if __name__ == "__main__":
    # gripper_pose: x y z qx qy qz qw state
    # 3rd rgb: (C,H,W) RGB
    # 3rd pcd: (C,H,W) XYZ
    # lang_goal: str
    # tasks: str
    dataset = Real_Dataset(
        data_path="/home/lpy/BridgeVLA_dev/finetune/Real/data",
        device="cuda:4",
        cameras="3rd",
        ep_per_task=3,
    )
    print(f"total samples: {len(dataset)}")
    index = 0
    for data in dataset:
        if index == 1:
            for key, value in data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"{key}_{sub_key}: {sub_value.shape}")
                elif isinstance(value, str):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value.shape}")
        pcd = data["3rd"]["pcd"]
        rgb = data["3rd"]["rgb"]
        gripper_pose_xyz = data["gripper_pose"][:3]
        index+=1
        
