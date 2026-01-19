"""
RoboWan Client for Real Robot Control
Provides both test client and real robot control loop
"""

# 禁用代理，确保可以连接本地SSH隧道
import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# 解决ZED SDK与PyTorch/CUDA的TLS冲突
# 设置环境变量以避免OpenMP冲突
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# 预加载libgomp以避免TLS初始化错误
import ctypes
try:
    # 尝试预加载libgomp（OpenMP库）
    ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
except:
    pass

# 先导入ZED相关的库，再导入PyTorch
try:
    import pyzed.sl as sl
except:
    pass

import requests
from PIL import Image
import io
import numpy as np
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import rospy
import open3d as o3d
import sys
import torch
import cv2
import matplotlib.pyplot as plt
import atexit
import time

# 添加训练代码路径
diffsynth_path = "/media/casia/data4/lpy/RoboWan/BridgeVLA_dev/Wan/DiffSynth-Studio"
sys.path.insert(0, diffsynth_path)

# 导入机器人接口模块
try:
    from robot_interface import RobotController, action_to_pose
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Robot interface not available: {e}")
    ROBOT_AVAILABLE = False

# 导入相机接口模块
try:
    from zedcamera_interface import Camera, get_cam_extrinsic
    CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Camera interface not available: {e}")
    CAMERA_AVAILABLE = False

# 导入训练代码中的预处理模块

from diffsynth.trainers.base_multi_view_dataset_with_rot_grip import (
    ProjectionInterface,
    RobotTrajectoryDataset,
    build_extrinsic_matrix,
    convert_pcd_to_base,
    _norm_rgb
)
PREPROCESSING_AVAILABLE = True


# ====================== 配置参数（全局） ======================
# ⚠️ 重要：只需在此处修改一次即可！
# 场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]
# 注意：必须与服务器端的 scene_bounds 保持完全一致！
#
# 默认值：[0, -0.7, -0.05, 0.8, 0.7, 0.65]
# 含义：
#   x: [0, 0.8] 米
#   y: [-0.7, 0.7] 米
#   z: [-0.05, 0.65] 米
# SCENE_BOUNDS = [0, -0.45, -0.05, 0.8, 0.55, 0.6]
# SCENE_BOUNDS=[0, -0.55, -0.05, 0.8, 0.45, 0.6]
SCENE_BOUNDS=[0,-0.65,-0.05,0.8,0.55,0.75]
# 图像尺寸
# 注意：必须与服务器端的 img_size 保持完全一致！
IMG_SIZE = 384

# 动作截断配置
# 是否启用动作截断（防止过大的动作变化）
ENABLE_ACTION_CLIPPING = True
# 位置最大变化量（米）- 默认4cm
MAX_POSITION_CHANGE = 0.04
# 旋转最大变化量（度）- 默认10度
MAX_ROTATION_CHANGE = 10.0

class RoboWanClient:
    """Client for communicating with RoboWan server"""

    def __init__(self,
                 server_url: str = "http://localhost:5555",
                 img_size: int = 256,
                 scene_bounds: List[float] = None,  # 默认使用全局 SCENE_BOUNDS
                 sigma: float = 1.5,
                 augmentation: bool = False):
        """
        Initialize client

        Args:
            server_url: URL of the server (e.g., "http://localhost:5555")
            img_size: 图像尺寸
            scene_bounds: 场景边界
            sigma: heatmap高斯分布标准差
            augmentation: 是否使用数据增强（推理时通常为False）
        """
        self.server_url = server_url
        self.predict_url = f"{server_url}/predict"
        self.health_url = f"{server_url}/health"

        # 初始化预处理参数
        self.img_size = (img_size, img_size)
        self.scene_bounds = scene_bounds if scene_bounds is not None else SCENE_BOUNDS
        self.sigma = sigma
        self.augmentation = augmentation
        self.mode = "test"  # 推理模式

        # 初始化外参矩阵（与训练时相同）
        self.extrinsic_matrix = build_extrinsic_matrix(
            translation=np.array([1.0472367143501216, 0.023761683274528322, 0.8609737768789085]),
            quaternion=np.array([0.311290132566853, -0.6359435618886714, -0.64373193090706, 0.29031610459898505])
        )

        # 初始化投影接口
        if PREPROCESSING_AVAILABLE:
            self.projection_interface = ProjectionInterface(
                img_size=img_size,
                rend_three_views=True,
                add_depth=False
            )
        else:
            self.projection_interface = None
            print("Warning: ProjectionInterface not available!")

    def check_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def predict(
        self,
        heatmap_images: List[Image.Image],
        rgb_images: List[Image.Image],
        prompt: str,
        initial_rotation: List[float],
        initial_gripper: int,
        num_frames: int = 12
    ) -> dict:
        """
        Send prediction request to server

        Args:
            heatmap_images: List of PIL Images for heatmap (multi-view)
            rgb_images: List of PIL Images for RGB (multi-view)
            prompt: Task instruction
            initial_rotation: Initial rotation [roll, pitch, yaw] in degrees
            initial_gripper: Initial gripper state (0 or 1)
            num_frames: Number of frames to predict

        Returns:
            Dictionary containing:
                - success: bool
                - rotation: List[List[float]] - (num_frames, 3)
                - gripper: List[int] - (num_frames,)
                - error: str (if failed)
        """
        # Prepare files
        files = []

        # Add heatmap images
        for i, img in enumerate(heatmap_images):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            files.append(('heatmap_images', (f'heatmap_{i}.png', buffer, 'image/png')))

        # Add RGB images
        for i, img in enumerate(rgb_images):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            files.append(('rgb_images', (f'rgb_{i}.png', buffer, 'image/png')))

        # Prepare form data
        rotation_str = ','.join(map(str, initial_rotation))
        scene_bounds_str = ','.join(map(str, self.scene_bounds))
        data = {
            'prompt': prompt,
            'initial_rotation': rotation_str,
            'initial_gripper': initial_gripper,
            'num_frames': num_frames,
            'scene_bounds': scene_bounds_str
        }

        try:
            # Send request (timeout=600秒=10分钟，因为推理可能需要较长时间)
            response = requests.post(self.predict_url, files=files, data=data, timeout=600)

            # Parse response
            result = response.json()
            return result

        except Exception as e:
            print(f"Request failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def preprocess(self, pcd_list, feat_list, all_poses: np.ndarray):
        """
        预处理点云序列、特征序列和姿态（与训练代码一致）

        Args:
            pcd_list: 点云列表，每个元素为 np.ndarray
            feat_list: 特征列表（RGB图像），每个元素为 np.ndarray
            all_poses: 姿态数组 [num_poses, 7] - (x,y,z,w,x,y,z) wxyz格式

        Returns:
            pc_list: 处理后的点云列表
            img_feat_list: 处理后的特征列表
            wpt_local: 局部坐标系下的姿态 [num_poses, 3]
            rot_xyzw: 旋转四元数 [num_poses, 4] - xyzw格式
            rev_trans: 逆变换函数
        """
        import bridgevla.mvt.utils as mvt_utils
        from bridgevla.mvt.augmentation import apply_se3_aug_con_shared

        # 确保输入是列表
        if not isinstance(pcd_list, list):
            pcd_list = [pcd_list]
        if not isinstance(feat_list, list):
            feat_list = [feat_list]

        num_frames = len(pcd_list)

        # 归一化RGB特征
        feat_list = [_norm_rgb(feat) for feat in feat_list]

        # 使用外参矩阵对pcd进行变换到base坐标系
        pcd_list = [convert_pcd_to_base(
            extrinsic_martix=self.extrinsic_matrix,
            pcd=pcd,
        ) for pcd in pcd_list]

        # 转换为torch张量
        pcd_list = [torch.from_numpy(np.ascontiguousarray(pcd)).float() if isinstance(pcd, np.ndarray) else pcd
                    for pcd in pcd_list]

        with torch.no_grad():
            # 展平点云和特征 [num_points, 3]
            pc_list = [pcd.view(-1, 3).float() for pcd in pcd_list]
            img_feat_list = [((feat.view(-1, 3) + 1) / 2).float() for feat in feat_list]

            # 数据增强（推理时通常关闭）
            if self.augmentation and self.mode == "train":
                assert False
                # 堆叠成batch [num_frames, num_points, 3]
                pc_batch = torch.stack(pc_list, dim=0)

                # 转换poses为tensor [num_frames, 7] - wxyz格式
                all_poses_tensor = torch.from_numpy(np.array(all_poses)).float()

                # 转换wxyz为xyzw格式（匹配augmentation代码的预期）
                position = all_poses_tensor[:, :3]  # [x, y, z]
                quat_wxyz = all_poses_tensor[:, 3:]  # [w, x, y, z]
                quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
                all_poses_tensor_fixed = torch.cat([position, quat_xyzw], dim=1)

                # 应用共享增强
                perturbed_poses, pc_batch = apply_se3_aug_con_shared(
                    pcd=pc_batch,
                    action_gripper_pose=all_poses_tensor_fixed,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=torch.tensor([0.1, 0.1, 0.1]),
                    rot_aug_range=torch.tensor([0.0, 0.0, 20.0]),
                )

                # 分解回列表
                pc_list = [pc_batch[i] for i in range(num_frames)]
                action_trans_con = perturbed_poses[:, :3]
                action_rot_xyzw = perturbed_poses[:, 3:]
            else:
                # 没有数据增强时，直接使用原始poses
                action_trans_con = torch.from_numpy(np.array(all_poses)).float()[:, :3]
                # 将wxyz格式转换为xyzw格式
                quat_wxyz = torch.from_numpy(np.array(all_poses)).float()[:, 3:]
                action_rot_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]

            # 对每个点云应用边界约束
            processed_pc_list = []
            processed_feat_list = []
            for pc, img_feat in zip(pc_list, img_feat_list):
                pc, img_feat = self.move_pc_in_bound(
                    pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                )
                processed_pc_list.append(pc[0])
                processed_feat_list.append(img_feat[0])

            # 将点云和wpt放在一个cube里面（使用第一个点云作为参考）
            wpt_local, rev_trans = mvt_utils.place_pc_in_cube(
                processed_pc_list[0],
                action_trans_con,
                with_mean_or_bounds=False,
                scene_bounds=self.scene_bounds,
            )

            # 对每个点云应用place_pc_in_cube
            final_pc_list = []
            for pc in processed_pc_list:
                pc = mvt_utils.place_pc_in_cube(
                    pc,
                    with_mean_or_bounds=False,
                    scene_bounds=self.scene_bounds,
                )[0]
                final_pc_list.append(pc)

        return final_pc_list, processed_feat_list, wpt_local, action_rot_xyzw, rev_trans

    def get_rgb_input(self, processed_pcd, processed_rgb):
        """
        从处理后的点云和RGB生成投影的RGB图像

        Args:
            processed_pcd: 处理后的点云 (torch.Tensor)
            processed_rgb: 处理后的RGB特征 (torch.Tensor)

        Returns:
            rgb_images: PIL.Image列表，每个视角一张图像
        """
        # 使用投影接口生成RGB图像
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_pcd, processed_rgb,
            img_aug_before=0.0,  # 推理时不做增强
            img_aug_after=0.0
        )  # (1, num_views, H, W, 6)

        rgb_image = rgb_image[0, :, :, :, 3:]  # (num_views, H, W, 3)

        # 确保是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()

        # 转换每个视角为PIL Image
        num_views = rgb_image.shape[0]
        rgb_images = []
        for view_idx in range(num_views):
            view_img = rgb_image[view_idx]  # (H, W, 3)

            # 从[-1, 1]归一化到[0, 1]，然后到[0, 255]
            view_img = np.clip((view_img + 1) / 2, 0, 1)
            view_img = (view_img * 255).astype(np.uint8)

            # 转换为PIL Image
            pil_img = Image.fromarray(view_img)
            rgb_images.append(pil_img)

        return rgb_images

    def get_heatmap_input(self, processed_pos):
        """
        从处理后的位置生成热力图

        Args:
            processed_pos: 处理后的位置 [num_poses, 3] (torch.Tensor)

        Returns:
            heatmap_images: PIL.Image列表，每个视角一张热力图
        """
        # 将位置投影到像素坐标
        img_locations = self.projection_interface.project_pose_to_pixel(
            processed_pos.unsqueeze(0).to(self.projection_interface.renderer_device)
        )  # (1, num_poses, num_views, 2)

        # 生成热力图
        heatmap_sequence = self.projection_interface.generate_heatmap_from_img_locations(
            img_locations,
            self.img_size[0], self.img_size[1],
            self.sigma
        )  # (1, num_poses, num_views, H, W)

        # 取第一个batch和第一个pose的热力图
        heatmap = heatmap_sequence[0, 0, :, :, :]  # (num_views, H, W)

        # 确保是numpy数组
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()

        # 转换每个视角为PIL Image
        num_views = heatmap.shape[0]
        heatmap_images = []
        for view_idx in range(num_views):
            view_hm = heatmap[view_idx]  # (H, W)

            # 归一化到[0, 1]
            view_hm_min = view_hm.min()
            view_hm_max = view_hm.max()
            if view_hm_max > view_hm_min:
                view_hm_norm = (view_hm - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = view_hm

            # 应用colormap（使用JET colormap与深度图类似）
            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_img = Image.fromarray(view_hm_colored)
            heatmap_images.append(pil_img)

        return heatmap_images
        
        
    @staticmethod   
    def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
        """
        :param no_op: no operation
        """
        if no_op:
            return pc, img_feat

        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        inv_pnt = (
            (pc[:, :, 0] < x_min)
            | (pc[:, :, 0] > x_max)
            | (pc[:, :, 1] < y_min)
            | (pc[:, :, 1] > y_max)
            | (pc[:, :, 2] < z_min)
            | (pc[:, :, 2] > z_max)
            | torch.isnan(pc[:, :, 0])
            | torch.isnan(pc[:, :, 1])
            | torch.isnan(pc[:, :, 2])
        )

        # TODO: move from a list to a better batched version
        pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
        img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
        return pc, img_feat


    @staticmethod    
    def convert_pcd_to_base(
            type="3rd",
            pcd=[]
        ):
        transform = get_cam_extrinsic(type)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3)
        
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 


    @staticmethod
    def vis_pcd(pcd, rgb):

        # 将点云和颜色转换为二维的形状 (N, 3)
        pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
        rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

        # 将点云和颜色信息保存为 PLY 文件
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
        pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
        # o3d.io.write_point_cloud(save_path, pcd)
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def visualize_pointcloud(processed_pcd, processed_rgb=None, title="Processed Point Cloud"):
        """
        可视化处理后的点云 (line 635)

        Args:
            processed_pcd: torch.Tensor or np.ndarray, shape (num_points, 3)
            processed_rgb: torch.Tensor or np.ndarray, shape (num_points, 3), 可选的RGB颜色
            title: 可视化窗口标题
        """
        # 转换为numpy数组
        if isinstance(processed_pcd, torch.Tensor):
            pcd_np = processed_pcd.cpu().numpy()
        else:
            pcd_np = processed_pcd

        # 创建Open3D点云对象
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)

        # 如果提供了RGB颜色
        if processed_rgb is not None:
            if isinstance(processed_rgb, torch.Tensor):
                rgb_np = processed_rgb.cpu().numpy()
            else:
                rgb_np = processed_rgb

            # 确保颜色在[0, 1]范围内
            if rgb_np.max() > 1.0:
                rgb_np = rgb_np / 255.0

            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_np)

        # 可视化
        print(f"\n{'='*60}")
        print(f"可视化: {title}")
        print(f"点云数量: {len(pcd_o3d.points)}")
        if processed_rgb is not None:
            print(f"包含RGB颜色信息")
        print(f"{'='*60}\n")

        o3d.visualization.draw_geometries(
            [pcd_o3d],
            window_name=title,
            width=800,
            height=600
        )

    @staticmethod
    def visualize_rgb_images(rgb_images, title="Multi-view RGB Images", save_path=None):
        """
        可视化多视角RGB图像 (line 639)

        Args:
            rgb_images: List[PIL.Image], 多视角RGB图像列表
            title: 显示窗口标题
            save_path: 可选的保存路径
        """
        num_views = len(rgb_images)

        # 创建子图显示所有视角
        import matplotlib.pyplot as plt

        # 计算子图布局 (尽量接近正方形)
        cols = int(np.ceil(np.sqrt(num_views)))
        rows = int(np.ceil(num_views / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle(title, fontsize=16)

        # 展平axes以便索引
        if num_views == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows * cols > 1 else [axes]

        for i, (img, ax) in enumerate(zip(rgb_images, axes)):
            ax.imshow(img)
            ax.set_title(f"View {i+1}")
            ax.axis('off')

        # 隐藏多余的子图
        for i in range(num_views, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"RGB图像已保存到: {save_path}")

        print(f"\n{'='*60}")
        print(f"可视化: {title}")
        print(f"视角数量: {num_views}")
        print(f"图像尺寸: {rgb_images[0].size}")
        print(f"{'='*60}\n")

        plt.show()

    @staticmethod
    def visualize_heatmap_images(heatmap_images, title="Multi-view Heatmaps", save_path=None):
        """
        可视化多视角热力图 (line 642)

        Args:
            heatmap_images: List[PIL.Image], 多视角热力图列表
            title: 显示窗口标题
            save_path: 可选的保存路径
        """
        num_views = len(heatmap_images)

        # 创建子图显示所有视角
        import matplotlib.pyplot as plt

        # 计算子图布局
        cols = int(np.ceil(np.sqrt(num_views)))
        rows = int(np.ceil(num_views / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle(title, fontsize=16)

        # 展平axes以便索引
        if num_views == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows * cols > 1 else [axes]

        for i, (img, ax) in enumerate(zip(heatmap_images, axes)):
            ax.imshow(img)
            ax.set_title(f"Heatmap View {i+1}")
            ax.axis('off')

        # 隐藏多余的子图
        for i in range(num_views, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"热力图已保存到: {save_path}")

        print(f"\n{'='*60}")
        print(f"可视化: {title}")
        print(f"视角数量: {num_views}")
        print(f"图像尺寸: {heatmap_images[0].size}")
        print(f"{'='*60}\n")

        plt.show()


def clip_action(
    target_action: np.ndarray,
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    max_position_change: float = 0.04,
    max_rotation_change: float = 10.0
) -> np.ndarray:
    """
    截断动作，限制位置和旋转的最大变化量

    Args:
        target_action: 目标动作 [x, y, z, roll, pitch, yaw, gripper] (rotation in radians)
        current_position: 当前位置 [x, y, z] (meters)
        current_rotation: 当前旋转 [roll, pitch, yaw] (radians)
        max_position_change: 位置最大变化量（米）
        max_rotation_change: 旋转最大变化量（度）

    Returns:
        clipped_action: 截断后的动作 [x, y, z, roll, pitch, yaw, gripper]
    """
    clipped_action = target_action.copy()

    # 计算位置变化
    target_position = target_action[:3]
    position_delta = target_position - current_position
    position_distance = np.linalg.norm(position_delta)

    # 如果位置变化超过阈值，进行截断
    if position_distance > max_position_change:
        # 沿着原方向缩放到最大允许距离
        scale_factor = max_position_change / position_distance
        clipped_position = current_position + position_delta * scale_factor
        clipped_action[:3] = clipped_position
        rospy.logwarn(f"  ⚠️  Position clipped: {position_distance:.4f}m -> {max_position_change:.4f}m")

    # 计算旋转变化（度）
    target_rotation = target_action[3:6]
    rotation_delta_rad = target_rotation - current_rotation
    rotation_delta_deg = np.rad2deg(rotation_delta_rad)

    # 对每个旋转轴分别检查和截断
    clipped_rotation = target_rotation.copy()
    rotation_clipped = False

    for i, axis_name in enumerate(['roll', 'pitch', 'yaw']):
        if abs(rotation_delta_deg[i]) > max_rotation_change:
            # 截断到最大允许变化
            sign = np.sign(rotation_delta_deg[i])
            max_change_rad = np.deg2rad(max_rotation_change * sign)
            clipped_rotation[i] = current_rotation[i] + max_change_rad
            rotation_clipped = True
            rospy.logwarn(f"  ⚠️  {axis_name.capitalize()} clipped: {rotation_delta_deg[i]:.2f}° -> {max_rotation_change * sign:.2f}°")

    if rotation_clipped:
        clipped_action[3:6] = clipped_rotation

    return clipped_action


def real_robot_control_loop(
    server_url: str = "http://localhost:5555",
    task_prompt: str = "put the lion on the top shelf",
    max_steps: int = 1000000,
    num_frames: int = 13,
    save_dir: Optional[str] = None,
    action_duration: float = 0.5,
    gripper_threshold: float = 0.5,
    scene_bounds: List[float] = None,
    img_size: int = None,
    enable_action_clipping: bool = None,
    max_position_change: float = None,
    max_rotation_change: float = None,
):
    """
    真机控制主循环

    Args:
        server_url: RoboWan服务器地址
        task_prompt: 任务指令
        max_steps: 最大控制步数
        num_frames: 每次预测的帧数
        save_dir: 数据保存目录（None则自动创建）
        action_duration: 每个动作执行时长（秒）
        gripper_threshold: 夹爪动作阈值
        scene_bounds: 场景边界 [x_min, y_min, z_min, x_max, y_max, z_max]
        img_size: 图像尺寸
        enable_action_clipping: 是否启用动作截断（None则使用全局配置）
        max_position_change: 位置最大变化量（米）（None则使用全局配置）
        max_rotation_change: 旋转最大变化量（度）（None则使用全局配置）
    """
    # 初始化ROS节点（如果尚未初始化）
    # 注意：使用 disable_signals=True 以匹配 FrankaArm 的初始化参数
    if not rospy.core.is_initialized():
        rospy.init_node('robowan_real_robot', anonymous=False, disable_signals=True)

    # 使用全局配置作为默认值
    if enable_action_clipping is None:
        enable_action_clipping = ENABLE_ACTION_CLIPPING
    if max_position_change is None:
        max_position_change = MAX_POSITION_CHANGE
    if max_rotation_change is None:
        max_rotation_change = MAX_ROTATION_CHANGE

    # 检查机器人接口是否可用
    if not ROBOT_AVAILABLE:
        rospy.logerr("Robot interface not available! Please check imports.")
        return

    rospy.loginfo("="*60)
    rospy.loginfo("Starting Real Robot Control Loop")
    rospy.loginfo("="*60)
    if enable_action_clipping:
        rospy.loginfo(f"Action Clipping ENABLED:")
        rospy.loginfo(f"  - Max position change: {max_position_change*100:.1f} cm")
        rospy.loginfo(f"  - Max rotation change: {max_rotation_change:.1f}°")
    else:
        rospy.loginfo("Action Clipping DISABLED")

    # 创建保存目录
    if save_dir is None:
        save_root = Path("/media/casia/data4/lpy/RoboWan/logs")
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_root / f"run_{time_tag}"
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "actions").mkdir(exist_ok=True)
    rospy.loginfo(f"Saving data to: {save_dir}")

    # 初始化 RoboWan 客户端
    rospy.loginfo(f"Connecting to RoboWan server at {server_url}")
    if scene_bounds is None:
        scene_bounds = SCENE_BOUNDS
    if img_size is None:
        img_size = IMG_SIZE
    rospy.loginfo(f"Using scene bounds: {scene_bounds}")
    rospy.loginfo(f"Using image size: {img_size}")
    client = RoboWanClient(server_url=server_url, scene_bounds=scene_bounds, img_size=img_size)

    # 检查服务器健康状态
    if not client.check_health():
        rospy.logerr("Server is not healthy! Please start the server first.")
        return
    rospy.loginfo("✓ Server is healthy")

    # 初始化机器人控制器
    rospy.loginfo("Initializing robot controller...")
    robot = RobotController(frequency=10)
    rospy.loginfo("✓ Robot controller initialized")

    # 注册退出时的清理函数，确保程序意外退出时也能清理
    atexit.register(lambda: robot.cleanup())
    rospy.loginfo("✓ Exit cleanup handler registered")

    # 启动动态控制模式（持续跟踪发布的目标位姿）
    rospy.loginfo("Starting dynamic control mode with LOW impedance...")

    # 先停止任何正在运行的skill
    try:
        robot.franka.stop_skill()
        rospy.sleep(0.5)
        rospy.loginfo("✓ Previous skill stopped")
    except Exception as e:
        rospy.logwarn(f"Could not stop previous skill: {e}")

    # 获取当前位姿并启动动态控制
    current_pose = robot.get_pose()
    robot.franka.goto_pose(
        current_pose,
        duration=100000,  # 很长的时间，保持动态模式运行
        dynamic=True,
        buffer_time=0.1,  # 使用很短的buffer_time（0.1秒）以减少动作累积
        cartesian_impedances=[200.0, 200.0, 200.0, 20.0, 20.0, 20.0],  # 降低刚度，减少"弹簧"效应
    )
    rospy.loginfo("✓ Dynamic control mode started with LOW impedance (刚度降低约3倍, buffer_time=0.1s)")

    # 检查相机接口是否可用
    if not CAMERA_AVAILABLE:
        rospy.logerr("Camera interface not available! Please check imports.")
        return

    # 初始化相机控制器（只使用第三视角相机）
    rospy.loginfo("Initializing third-person camera...")
    camera = Camera(camera_type="3rd")
    rospy.loginfo("✓ Third-person camera initialized")

    # 获取初始状态
    current_pose = robot.get_pose()
    initial_rotation = current_pose.euler_angles  # [roll, pitch, yaw] - 保存初始旋转，全程不变
    current_gripper = 1 if robot.gripper_state else 0  # 0=闭合, 1=打开

    rospy.loginfo("\n" + "="*60)
    rospy.loginfo(f"Task: {task_prompt}")
    rospy.loginfo(f"Max steps: {max_steps}")
    rospy.loginfo(f"Initial rotation (FIXED): {np.rad2deg(initial_rotation)}")
    rospy.loginfo(f"Initial gripper: {current_gripper}")
    rospy.loginfo("="*60)
    rospy.loginfo("⚠️  DEBUG MODE:")
    rospy.loginfo("  - Rotation will remain FIXED at initial value")
    rospy.loginfo("  - Gripper will be controlled by USER INPUT")
    rospy.loginfo("="*60 + "\n")

    # 控制循环
    step = 0

    try:
        while step < max_steps:
            rospy.loginfo(f"{'='*60}")
            rospy.loginfo(f"Step {step}/{max_steps}")
            rospy.loginfo(f"{'='*60}")

            # ========== 0. 新的控制策略 ==========
            # 动态模式在初始化时启动，机器人会持续追踪发布的目标
            # 缓冲区会在执行新的action chunk之前清空（见步骤6）
            # 这样既保证了观测阶段的平滑运动，又避免了旧动作的干扰

            # ========== 1. 获取观测 ==========
            rospy.loginfo("Capturing observations...")
            try:
                obs = camera.capture()

                # 获取第三视角的RGB图像和点云
                rgb_3rd = obs['3rd']['rgb']
                pcd_3rd = obs['3rd']['pcd']

                # 将BGR格式转换为RGB格式
                rgb_3rd = cv2.cvtColor(rgb_3rd, cv2.COLOR_BGR2RGB)
                # vis_pcd(pcd_3rd, rgb_3rd) #TODO

                # 保存观测
                # save_observations(obs, save_dir, step)
                rospy.loginfo("✓ Observations captured")

                # 获取当前机器人状态
                current_pose_obj = robot.get_pose()
                # 使用固定的初始旋转，而不是当前旋转
                current_rotation_deg = np.rad2deg(initial_rotation).tolist()
                current_gripper = 1 if robot.gripper_state else 0

                # 将当前位姿转换为数组格式 [x, y, z, w, x, y, z] (wxyz格式)
                # 假设 current_pose_obj 有 translation 和 quaternion 属性
                current_position = current_pose_obj.translation  # [x, y, z]
                current_quat = current_pose_obj.quaternion  # 假设是 [w, x, y, z] 格式
                current_pose = np.concatenate([current_position, current_quat])  # [x, y, z, w, x, y, z]
                current_pose = current_pose.reshape(1, 7)  # [1, 7]

                rospy.loginfo(f"Current position: {current_position}")
                rospy.loginfo(f"Fixed rotation (used for inference): {current_rotation_deg}")
                rospy.loginfo(f"Current gripper: {current_gripper}")

            except Exception as e:
                rospy.logerr(f"Error capturing observations: {e}")
                break

            # ========== 2. 准备输入数据 ==========

            # 预处理数据
            processed_pcd_list, processed_rgb_list, processed_pos, processed_rot_xyzw, rev_trans = client.preprocess(
                [pcd_3rd], [rgb_3rd], current_pose
            )

            # 取第一个元素（因为只有一个观测）
            processed_pcd = processed_pcd_list[0]
            processed_rgb = processed_rgb_list[0]
            # 可视化处理后的点云 (可选)
            # RoboWanClient.visualize_pointcloud(processed_pcd, processed_rgb, title="Processed Point Cloud")

            # 准备RGB图像（只使用第三视角）
            rgb_images = client.get_rgb_input(processed_pcd, processed_rgb)
            # 可视化多视角RGB图像 (可选)
            # RoboWanClient.visualize_rgb_images(rgb_images, title="Multi-view RGB Images")

            # 准备热力图
            heatmap_images = client.get_heatmap_input(processed_pos)
            # 可视化热力图 (可选)
            # RoboWanClient.visualize_heatmap_images(heatmap_images, title="Multi-view Heatmaps")


            # ========== 3. 发送请求到服务器 ==========
            rospy.loginfo("Requesting action from server...")
            try:
                result = client.predict(
                    heatmap_images=heatmap_images,
                    rgb_images=rgb_images,
                    prompt=task_prompt,
                    initial_rotation=current_rotation_deg,
                    initial_gripper=current_gripper,
                    num_frames=num_frames
                )

                if not result['success']:
                    rospy.logerr(f"Server prediction failed: {result.get('error', 'Unknown error')}")
                    break

                rospy.loginfo("✓ Received action from server")

            except Exception as e:
                rospy.logerr(f"Error during server request: {e}")
                break

            # ========== 4. 解析动作 ==========
            # 提取位置、旋转和夹爪动作
            position_actions = result.get('position', [])
            rotation_actions = result.get('rotation', [])
            gripper_actions = result.get('gripper', [])

            # 调试信息
            rospy.loginfo(f"Received actions:")
            rospy.loginfo(f"  Position actions: {len(position_actions)} frames")
            rospy.loginfo(f"  Rotation actions: {len(rotation_actions)} frames")
            rospy.loginfo(f"  Gripper actions: {len(gripper_actions)} frames")

            # ========== DEBUG: 检查第一个预测位置和当前位置的差异 ==========
            # 理论上,如果空间位置->热力图->空间位置的变换是正确的,
            # 那么第一个预测位置应该和当前位置接近
            if len(position_actions) > 0:
                predicted_first_pos = np.array(position_actions[0])  # [x, y, z]
                position_diff = predicted_first_pos - current_position
                position_distance = np.linalg.norm(position_diff)

                rospy.loginfo(f"\n{'='*60}")
                rospy.loginfo(f"[DEBUG] 位置变换一致性检查:")
                rospy.loginfo(f"  当前位置 (current_position):     {current_position}")
                rospy.loginfo(f"  预测第一帧位置 (predicted[0]):    {predicted_first_pos}")
                rospy.loginfo(f"  位置差异向量:                      {position_diff}")
                rospy.loginfo(f"  位置差异距离 (L2范数):             {position_distance:.6f} 米")

                # 判断是否接近 (阈值可以调整,例如1cm = 0.01m)
                threshold = 0.01  # 1cm
                if position_distance < threshold:
                    rospy.loginfo(f"  ✓ 变换一致! (距离 < {threshold}m)")
                else:
                    rospy.logwarn(f"  ✗ 变换可能有误! (距离 >= {threshold}m)")
                    rospy.logwarn(f"    这可能意味着空间位置<->热力图的变换存在问题")
                rospy.loginfo(f"{'='*60}\n")

            # 确保所有动作数组长度一致
            num_actions = min(len(position_actions), len(rotation_actions), len(gripper_actions))
            if num_actions == 0:
                rospy.logerr("No actions received from server!")
                break

            rospy.loginfo(f"Using {num_actions} actions")

            # 构建动作矩阵
            action_matrix = np.zeros((num_actions, 7))
            action_matrix[:, :3] = np.array(position_actions[:num_actions])  # position
            # ⚠️ 使用固定的初始旋转，而不是模型预测的旋转
            action_matrix[:, 3:6] = initial_rotation  # 所有动作使用相同的初始旋转
            # action_matrix[:, 6] = np.array(gripper_actions[:num_actions])  # gripper - 不使用模型预测

            rospy.loginfo(f"✓ Using FIXED rotation for all actions: {np.rad2deg(initial_rotation)}")

            # 保存动作
            # np.save(str(save_dir / "actions" / f"step_{step:05d}.npy"), action_matrix)
            # rospy.loginfo(f"Action matrix shape: {action_matrix.shape}")

            # ========== 5. 获取用户输入的夹爪状态 ==========
            # 建议：如果仍然有干扰，可以减少执行的动作数量（比如改为1或2）
            # 这样可以更快地响应新的观测，减少旧动作的累积
            num_actions_to_execute = min(10, len(action_matrix)) # 可以修改为 min(1, ...) 或 min(2, ...)

            rospy.loginfo("\n" + "="*60)
            rospy.loginfo(f"准备执行 {num_actions_to_execute} 个动作")
            rospy.loginfo("="*60)

            # 获取用户输入的gripper状态
            user_gripper = None
            while user_gripper is None:
                try:
                    user_input = input("请输入此action chunk的夹爪状态 (0=闭合, 1=打开): ").strip()
                    if user_input in ['0', '1']:
                        user_gripper = int(user_input)
                        rospy.loginfo(f"✓ 用户设置夹爪状态为: {user_gripper} ({'闭合' if user_gripper == 0 else '打开'})")
                    else:
                        print("⚠️  无效输入，请输入0或1")
                except KeyboardInterrupt:
                    rospy.loginfo("\n用户中断")
                    return
                except Exception as e:
                    print(f"输入错误: {e}, 请重试")

            # 将用户输入的gripper状态应用于所有动作
            action_matrix[:, 6] = user_gripper
            rospy.loginfo(f"✓ 所有 {num_actions_to_execute} 个动作的夹爪状态设置为: {user_gripper}")
            rospy.loginfo("="*60 + "\n")

            # ========== 6. 执行动作 ==========
            # ⚠️ 在执行新的action chunk之前，先清空旧的动作缓冲区
            # 策略：停止旧skill，然后从当前位置重新启动一次dynamic模式
            # 之后在整个action chunk执行期间，只通过publish_pose()更新目标，不再重启
            try:
                rospy.loginfo("Clearing action buffer before executing new action chunk...")
                robot.franka.stop_skill()
                rospy.sleep(0.2)  # 等待skill停止

                # 重新启动动态控制模式（从当前位置开始）
                current_pose_for_restart = robot.get_pose()
                robot.franka.goto_pose(
                    current_pose_for_restart,
                    duration=100000,
                    dynamic=True,
                    buffer_time=0.5,  # 使用很短的buffer_time (0.1秒)，减少旧目标的影响
                    cartesian_impedances=[200.0, 200.0, 200.0, 20.0, 20.0, 20.0],  # 降低刚度，减少"弹簧"效应
                )
                rospy.sleep(0.1)
                rospy.loginfo("✓ Action buffer cleared, ready for new action chunk")
            except Exception as e:
                rospy.logwarn(f"Error clearing buffer: {e}")

            for action_idx in range(num_actions_to_execute):
                single_action = action_matrix[action_idx]
                rospy.loginfo(f"\nExecuting action {action_idx+1}/{num_actions_to_execute}")
                rospy.loginfo(f"  Position: {single_action[:3]}")
                rospy.loginfo(f"  Rotation (FIXED): {np.rad2deg(single_action[3:6])}")
                rospy.loginfo(f"  Gripper: {int(single_action[6])} ({'闭合' if single_action[6] == 0 else '打开'})")

                try:
                    # 获取当前位姿用于对比
                    current_pose_obj = robot.get_pose()
                    current_pos = current_pose_obj.translation
                    current_rot = current_pose_obj.euler_angles

                    rospy.loginfo(f"  Current position: {current_pos}")

                    # 计算位置差异
                    pos_diff = np.linalg.norm(single_action[:3] - current_pos)
                    rospy.loginfo(f"  Position difference: {pos_diff:.4f} m ({pos_diff*100:.2f} cm)")

                    # 应用动作截断（如果启用）
                    if enable_action_clipping:
                        # 只截断位置，旋转保持不变
                        target_position = single_action[:3]
                        position_delta = target_position - current_pos
                        position_distance = np.linalg.norm(position_delta)

                        if position_distance > max_position_change:
                            # 沿着原方向缩放到最大允许距离
                            scale_factor = max_position_change / position_distance
                            clipped_position = current_pos + position_delta * scale_factor
                            single_action[:3] = clipped_position
                            rospy.logwarn(f"  ⚠️  Position clipped: {position_distance:.4f}m -> {max_position_change:.4f}m")
                            rospy.loginfo(f"  Clipped position: {single_action[:3]}")

                    # 计算目标位姿（使用绝对动作）
                    target_pose = action_to_pose(
                        single_action[:6],
                        current_pose_obj,
                        is_relative=False
                    )

                    # ⚠️ 改进的执行策略：持续发布直到接近目标或超时
                    # 目标：确保机器人至少接近了目标位置，而不是还在路上就切换
                    publish_duration = action_duration  # 基础发布时间
                    max_wait_time = action_duration * 3  # 最大等待时间（3倍action_duration）
                    position_threshold = 0.01  # 位置接近阈值：1cm

                    t_start = time.time()
                    reached = False

                    # 第一阶段：持续发布action_duration时间
                    while (time.time() - t_start) < publish_duration:
                        robot.publish_pose(target_pose, time.time())

                    # 第二阶段：检查是否接近目标，如果没有则继续发布
                    while (time.time() - t_start) < max_wait_time:
                        current_pos_check = robot.get_pose().translation
                        distance_to_target = np.linalg.norm(single_action[:3] - current_pos_check)

                        if distance_to_target < position_threshold:
                            reached = True
                            rospy.loginfo(f"  ✓ Reached target (distance: {distance_to_target*100:.2f}cm)")
                            break

                        # 继续发布目标位姿
                        robot.publish_pose(target_pose, time.time())
                        rospy.sleep(0.05)  # 小延迟避免过于频繁

                    elapsed = time.time() - t_start
                    if not reached:
                        final_pos = robot.get_pose().translation
                        final_distance = np.linalg.norm(single_action[:3] - final_pos)
                        rospy.logwarn(f"  ⚠️  Action {action_idx+1} timeout after {elapsed:.2f}s (distance: {final_distance*100:.2f}cm)")
                    else:
                        rospy.loginfo(f"  ✓ Action {action_idx+1} completed in {elapsed:.2f}s")

                    # 控制夹爪
                    gripper_action = float(single_action[6])
                    robot.control_gripper(gripper_action, threshold=gripper_threshold)
                    rospy.loginfo(f"  ✓ Gripper controlled")

                except Exception as e:
                    rospy.logerr(f"Error executing action {action_idx}: {e}")
                    continue

            # 增加 action_id（因为使用了 publish_pose 而不是 execute_pose）
            robot.action_id += 1

            # 步数递增
            step += 1
            rospy.loginfo(f"Step {step} completed\n")

    except KeyboardInterrupt:
        rospy.loginfo("\nControl loop interrupted by user")
    except Exception as e:
        rospy.logerr(f"Fatal error in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理机器人资源，停止所有skill
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("Cleaning up robot resources...")
        try:
            robot.cleanup()
        except Exception as e:
            rospy.logwarn(f"Error during cleanup: {e}")

        rospy.loginfo("Control Loop Finished")
        rospy.loginfo(f"Total steps executed: {step}")
        rospy.loginfo(f"Data saved to: {save_dir}")
        rospy.loginfo("="*60)


def main():
    """
    RoboWan真机控制主程序入口

    使用方法：
        python RoboWan_client.py                    # 使用默认参数
        python RoboWan_client.py --help             # 显示帮助
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="RoboWan真机控制客户端 - 连接服务器并控制机器人执行任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  示例用法:
  # 基本用法（使用默认参数，默认启用动作截断）
  python RoboWan_client.py

  # 指定服务器地址和任务
  python RoboWan_client.py --server http://10.10.1.21:5555 --task "put the lion on the top shelf"

  # 指定最大步数、预测帧数和保存目录
  python RoboWan_client.py --max-steps 50 --num-frames 10 --save-dir /path/to/save

  # 自定义场景边界和图像尺寸
  python RoboWan_client.py --scene-bounds "0,-0.55,-0.05,0.8,0.45,0.6" --img-size 384

  # 自定义动作截断参数（位置最大变化5cm，旋转最大变化15度）
  python RoboWan_client.py --max-position-change 0.05 --max-rotation-change 15.0

  # 禁用动作截断
  python RoboWan_client.py --disable-action-clipping
        """
    )

    # 服务器配置
    parser.add_argument(
        '--server',
        type=str,
        default="http://localhost:5555",
        help='RoboWan服务器地址 (默认: http://localhost:5555，通过SSH隧道连接)'
    )

    # 任务配置
    parser.add_argument(
        '--task',
        type=str,
        default="put the lion on the top shelf",
        help='任务指令'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=100000,
        help='最大控制步数 (默认: 100)'
    )

    parser.add_argument(
        '--num-frames',
        type=int,
        default=13,
        help='每次预测的帧数 (默认: 5),包括了初始帧'
    )

    parser.add_argument(
        '--scene-bounds',
        type=str,
        default=None,
        help='场景边界，格式为逗号分隔的6个数字: x_min,y_min,z_min,x_max,y_max,z_max (默认: 使用SCENE_BOUNDS全局配置)'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help='图像尺寸 (默认: 使用IMG_SIZE全局配置)'
    )

    # 动作截断参数
    parser.add_argument(
        '--enable-action-clipping',
        action='store_true',
        default=None,
        help='启用动作截断（限制位置和旋转的最大变化）'
    )

    parser.add_argument(
        '--disable-action-clipping',
        action='store_true',
        default=False,
        help='禁用动作截断'
    )

    parser.add_argument(
        '--max-position-change',
        type=float,
        default=None,
        help='位置最大变化量（米） (默认: 0.04m = 4cm)'
    )

    parser.add_argument(
        '--max-rotation-change',
        type=float,
        default=None,
        help='旋转最大变化量（度） (默认: 10.0°)'
    )

    # 控制参数
    parser.add_argument(
        '--action-duration',
        type=float,
        default=0.5,
        help='每个动作执行时长（秒） (默认: 0.5)'
    )

    parser.add_argument(
        '--gripper-threshold',
        type=float,
        default=0.5,
        help='夹爪动作阈值 (默认: 0.5)' # 感觉这个没有什么用
    )

    # 数据保存
    parser.add_argument(
        '--save-dir',
        type=str,
        default="/media/casia/data4/lpy/RoboWan/logs",
        help='数据保存目录 (默认: 自动创建时间戳目录)'
    )

    args = parser.parse_args()

    # 解析scene_bounds参数
    scene_bounds = None
    if args.scene_bounds is not None:
        try:
            scene_bounds = [float(x.strip()) for x in args.scene_bounds.split(',')]
            if len(scene_bounds) != 6:
                print("错误: scene_bounds必须包含6个数字 (x_min,y_min,z_min,x_max,y_max,z_max)")
                return
        except ValueError:
            print("错误: scene_bounds格式错误，必须是逗号分隔的6个数字")
            return

    # 处理动作截断参数
    enable_action_clipping = None
    if args.disable_action_clipping:
        enable_action_clipping = False
    elif args.enable_action_clipping:
        enable_action_clipping = True
    # 如果两个都没指定，则为None，使用全局配置

    # 检查机器人接口是否可用
    if not ROBOT_AVAILABLE:
        print("="*60)
        print("错误：机器人接口不可用！")
        print("请确保已正确安装以下依赖：")
        print("  - frankapy")
        print("  - diffusion_policy")
        print("  - autolab_core")
        print("="*60)
        return

    # 初始化ROS - RobotController 内部会自动初始化ROS节点
    # 如果需要手动初始化，取消下面的注释
    # try:
    #     if not rospy.core.is_initialized():
    #         rospy.init_node('robowan_client', anonymous=True)
    # except Exception as e:
    #     print(f"错误: ROS初始化失败 - {e}")
    #     print("请确保已启动 roscore")
    #     return

    # 确定最终使用的配置
    final_enable_clipping = enable_action_clipping if enable_action_clipping is not None else ENABLE_ACTION_CLIPPING
    final_max_pos_change = args.max_position_change if args.max_position_change is not None else MAX_POSITION_CHANGE
    final_max_rot_change = args.max_rotation_change if args.max_rotation_change is not None else MAX_ROTATION_CHANGE

    # 显示配置信息
    print("\n" + "="*60)
    print("RoboWan真机控制客户端")
    print("="*60)
    print(f"服务器地址:        {args.server}")
    print(f"任务指令:          {args.task}")
    print(f"最大步数:          {args.max_steps}")
    print(f"预测帧数:          {args.num_frames}    (initial frame included)")
    print(f"场景边界:          {scene_bounds if scene_bounds else SCENE_BOUNDS} {'(自定义)' if scene_bounds else '(默认)'}")
    print(f"图像尺寸:          {args.img_size if args.img_size else IMG_SIZE} {'(自定义)' if args.img_size else '(默认)'}")
    print(f"动作执行时长:      {args.action_duration}秒")
    print(f"夹爪阈值:          {args.gripper_threshold}")
    print("")
    print("动作截断配置:")
    if final_enable_clipping:
        print("  状态:            启用 ✓")
        print(f"  最大位置变化:    {final_max_pos_change*100:.1f} cm")
        print(f"  最大旋转变化:    {final_max_rot_change:.1f}°")
    else:
        print("  状态:            禁用 ✗")
    print("")
    print(f"保存目录:          {args.save_dir if args.save_dir else '自动创建'}")
    print("="*60)

    # 确认启动
    try:
        user_input = input("\n是否开始控制机器人? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y']:
            print("已取消")
            return
    except KeyboardInterrupt:
        print("\n已取消")
        return

    print("\n开始真机控制...")
    print("按 Ctrl+C 可随时停止\n")

    # 启动真机控制循环
    try:
        real_robot_control_loop(
            server_url=args.server,
            task_prompt=args.task,
            max_steps=args.max_steps,
            num_frames=args.num_frames,
            save_dir=args.save_dir,
            action_duration=args.action_duration,
            gripper_threshold=args.gripper_threshold,
            scene_bounds=scene_bounds,
            img_size=args.img_size,
            enable_action_clipping=enable_action_clipping,
            max_position_change=args.max_position_change,
            max_rotation_change=args.max_rotation_change,
        )
    except KeyboardInterrupt:
        rospy.loginfo("\n用户中断控制")
    except Exception as e:
        rospy.logerr(f"\n控制过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
