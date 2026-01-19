"""
Robot Trajectory Dataset for Single View Heatmap Prediction
支持从机器人轨迹数据生成RGB图像和heatmap序列的数据集
"""

import os
import pickle
try:
    import pickle5
    USE_PICKLE5 = True
except ImportError:
    USE_PICKLE5 = False
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Callable
import glob
import sys
finetune_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../finetune"))
sys.path.append(finetune_path)
import bridgevla.utils.rvt_utils as rvt_utils
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con_shared

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


def convert_pcd_to_base(extrinsic_martix, pcd=[]):
    transform = extrinsic_martix

    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)  # 去掉A
    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd

def _norm_rgb(x):
    if isinstance(x, np.ndarray):
        # 处理负步长问题
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
    return (x.float() / 255.0) * 2.0 - 1.0

class ProjectionInterface:
    """
    点云投影接口 - 提供默认实现，用户可以继承并重写
    """
    def __init__(self,
                img_size=256,
                rend_three_views=True,
                add_depth=False,
                ):

        from point_renderer.rvt_renderer import RVTBoxRenderer
        import os
        # 使用LOCAL_RANK环境变量确定当前进程的GPU（分布式训练支持）
        # 这样每个进程的渲染器会使用自己对应的GPU，避免显存不均匀
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.renderer_device = f"cuda:{local_rank}"
            print(f"[ProjectionInterface] Using device: {self.renderer_device} (LOCAL_RANK={local_rank})")
        else:
            self.renderer_device = "cpu"
            print(f"[ProjectionInterface] Using device: cpu")
        self.renderer = RVTBoxRenderer(
            device=self.renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )
        self.img_size = (img_size, img_size)


    def project_pointcloud_to_rgb(self, pointcloud: np.ndarray, feat: np.ndarray, img_aug_before=0.1, img_aug_after=0.05) -> np.ndarray:
        """
        将点云投影到指定视角生成RGB图像

        Args:
            pointcloud: 点云数据 (N, 3) 
            feat: 颜色数据 (N, 3) 

        Returns:
            RGB图像 (N, H, W, 3) 范围[0, 1] N 表示有多少个视角
        """
        # aug before projection
        if img_aug_before !=0:
            
            stdv = img_aug_before * torch.rand(1, device=feat.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*feat.shape, device=feat.device)) - 1)
            feat = feat + noise #

        # 确保数据在正确的设备上
        renderer_device = self.renderer_device
        if hasattr(pointcloud, 'device') and str(pointcloud.device) != str(renderer_device):
            pointcloud = pointcloud.to(renderer_device)
        if hasattr(feat, 'device') and str(feat.device) != str(renderer_device):
            feat = feat.to(renderer_device)

        max_pc = 1.0 if len(pointcloud) == 0 else torch.max(torch.abs(pointcloud))

        img= self.renderer(
                pointcloud,
                torch.cat((pointcloud / max_pc, feat), dim=-1),
                fix_cam=True,
                dyn_cam_info=None
            ).unsqueeze(0)

        # aug after projection
        if img_aug_after != 0:
            stdv = img_aug_after * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)
        return img


    def project_pose_to_pixel(self, poses: np.ndarray) -> Tuple[int, int]:
        """
        将三维空间中的路径点坐标转换为图像坐标系下的坐标
        :param poses: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """

        pt_img = self.renderer.get_pt_loc_on_img(
            poses, fix_cam=True, dyn_cam_info=None
        )
        return pt_img

    def generate_heatmap_from_img_locations(self,
        img_locations,
        width=256, height=256, sigma=1.5):

        # (bs, np, num_img, 2)
        bs, np, num_img, _= img_locations.shape

        action_trans = mvt_utils.generate_hm_from_pt(
            img_locations.reshape(-1, 2),
            (self.img_size[0], self.img_size[1]),
            sigma=sigma,
            thres_sigma_times=3,
        )
        heatmap_sequence=action_trans.view(bs,np,num_img,height,width)
        return heatmap_sequence




class RobotTrajectoryDataset(Dataset):
    """
    机器人轨迹数据集类
    从包含多条轨迹的文件夹中加载数据，生成RGB-to-heatmap序列训练数据
    """

    def __init__(self,
                 data_root: str,
                 projection_interface: ProjectionInterface,
                 sequence_length: int = 10,
                 step_interval: int = 1,
                 min_trail_length: int = 15,
                 image_size: Tuple[int, int] = (256, 256),
                 sigma: float = 1.5,
                 augmentation: bool = True,
                 mode="train",
                 scene_bounds: List[float] = [0,-0.7,-0.05,0.8,0.7,0.65],
                 transform_augmentation_xyz=[0.1, 0.1, 0.1],
                 transform_augmentation_rpy=[0.0, 0.0, 20.0], # 确认一下到底应该是多少
                 debug=False,
                 ):
        """
        初始化数据集

        Args:
            data_root: 数据根目录路径
            projection_interface: 点云投影接口实现
            sequence_length: 预测的heatmap序列长度
            step_interval: step采样间隔
            min_trail_length: 最小轨迹长度要求
            image_size: 图像尺寸 (H, W)
            sigma: heatmap高斯分布标准差
        """
        self.data_root = data_root
        self.projection_interface = projection_interface
        self.sequence_length = sequence_length
        self.step_interval = step_interval
        self.min_trail_length = min_trail_length
        self.image_size = image_size
        self.sigma = sigma
        self.augmentation = augmentation
        self.mode="train"
        self.scene_bounds = scene_bounds
        self._transform_augmentation_xyz = torch.from_numpy( # 数据增强的xyz范围
            np.array(transform_augmentation_xyz)
        )
        self.debug=debug
        self._transform_augmentation_rpy = transform_augmentation_rpy # 数据增强的rpy范围

        # 扫描所有轨迹数据
        self.trail_data = self._scan_trails()
        self.valid_samples = self._generate_valid_samples()

        self.extrinsic_matrix = build_extrinsic_matrix(
            translation=np.array(
                [  1.0472367143501216,0.023761683274528322,0.8609737768789085]
            ),
            quaternion=np.array(
                [
                0.311290132566853,
                -0.6359435618886714,
                -0.64373193090706,
                0.29031610459898505,
                ]
            ),
        )

        print(f"Found {len(self.trail_data)} trails, {len(self.valid_samples)} valid samples")

    def _scan_trails(self) -> List[Dict]:
        """
        扫描数据目录，收集所有轨迹信息
        """
        trail_data = []

        # 找到所有trail_开头的文件夹
        trail_pattern = os.path.join(self.data_root, "trail_*")
        trail_dirs = sorted(glob.glob(trail_pattern))
        if self.debug:
            trail_dirs=trail_dirs[:2]
        for trail_dir in trail_dirs:
            if not os.path.isdir(trail_dir):
                continue

            trail_name = os.path.basename(trail_dir)

            # 检查必要的文件夹是否存在
            poses_dir = os.path.join(trail_dir, "poses")
            pcd_dir = os.path.join(trail_dir, "pcd")
            bgr_dir = os.path.join(trail_dir, "3rd_bgr")
            instruction_file = os.path.join(trail_dir, "instruction.txt")

            if not all(os.path.exists(p) for p in [poses_dir, pcd_dir, instruction_file]):
                print(f"Skipping {trail_name}: missing required directories/files")
                continue

            # 统计step数量
            pose_files = sorted(glob.glob(os.path.join(poses_dir, "*.pkl")))
            pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pkl")))
            bgr_files = sorted(glob.glob(os.path.join(bgr_dir, "*.pkl"))) 

            if len(pose_files) != len(pcd_files):
                print(f"Skipping {trail_name}: pose and pcd count mismatch")
                continue

            # 如果存在BGR文件夹，检查数量是否匹配
            if bgr_files and len(pose_files) != len(bgr_files):
                print(f"Warning {trail_name}: BGR file count mismatch with poses ({len(bgr_files)} vs {len(pose_files)})")
                # 可以选择继续或跳过，这里选择继续但清空bgr_files
                continue

            if len(pose_files) < self.min_trail_length:
                print(f"Skipping {trail_name}: too short ({len(pose_files)} steps)")
                continue

            # 读取instruction
            with open(instruction_file, 'r') as f:
                instruction = f.read().strip()

            trail_info = {
                'trail_name': trail_name,
                'trail_dir': trail_dir,
                'poses_dir': poses_dir,
                'pcd_dir': pcd_dir,
                'bgr_dir': bgr_dir,
                'instruction': instruction,
                'num_steps': len(pose_files),
                'pose_files': pose_files,
                'pcd_files': pcd_files,
                'bgr_files': bgr_files
            }

            trail_data.append(trail_info)

        return trail_data

    def _generate_valid_samples(self) -> List[Dict]:
        """
        生成所有有效的训练样本
        每个样本包含: 起始step的信息 + 后续sequence_length个step的信息
        支持末尾padding：超出轨迹的部分重复最后一个step
        """
        valid_samples = []

        for trail_info in self.trail_data:
            num_steps = trail_info['num_steps']

            # 扩大max_start_step的范围，允许更多起始点
            # 现在可以从轨迹的任何位置开始，包括接近末尾的位置
            max_start_step = num_steps - 1  # 可以从最后一步之前的任何位置开始

            for start_step in range(0, max_start_step, self.step_interval):
                # 计算后续step的索引，支持padding
                future_steps = []
                for i in range(1, self.sequence_length + 1):
                    future_step = start_step + i * self.step_interval
                    if future_step < num_steps:
                        # 正常情况：在轨迹范围内
                        future_steps.append(future_step)
                    else:
                        # padding情况：超出轨迹范围，使用最后一个step
                        future_steps.append(num_steps - 1)

                # 现在总是有完整的sequence_length个步骤
                sample = {
                    'trail_info': trail_info,
                    'start_step': start_step,
                    'future_steps': future_steps
                }
                valid_samples.append(sample)

        return valid_samples

    def _load_pickle_file(self, filepath: str):
        """
        加载pickle文件，处理兼容性问题
        """
        try:
            # 首先尝试使用标准pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data
        except (ModuleNotFoundError, AttributeError) as e:
            if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                # 处理numpy版本兼容性问题
                try:
                    import sys
                    # 创建兼容性映射
                    if hasattr(np, 'core'):
                        sys.modules['numpy._core.numeric'] = np.core.numeric if hasattr(np.core, 'numeric') else np
                        sys.modules['numpy._core.multiarray'] = np.core.multiarray if hasattr(np.core, 'multiarray') else np
                        sys.modules['numpy._core.fromnumeric'] = np.core.fromnumeric if hasattr(np.core, 'fromnumeric') else np
                        sys.modules['numpy._core'] = np.core

                    # 尝试使用pickle5如果可用
                    if USE_PICKLE5:
                        with open(filepath, 'rb') as f:
                            data = pickle5.load(f)
                    else:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                    return data
                except Exception as e2:
                    print(f"Error loading {filepath} after numpy fix: {e2}")
                    return None
            else:
                print(f"Module error loading {filepath}: {e}")
                return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def _load_step_data(self, trail_info: Dict, step_idx: int, pose_only: bool = False) -> Tuple:
        """
        加载指定step的pose、点云和BGR图像数据

        Args:
            trail_info: 轨迹信息
            step_idx: step索引
            pose_only: 是否只返回pose数据

        Returns:
            pose_only=True: (pose_data,)
            pose_only=False: (pose_data, pointcloud_data, rgb_image_data)
        """
        pose_file = trail_info['pose_files'][step_idx]
        pose_data = self._load_pickle_file(pose_file)

        if pose_only:
            return (pose_data,)

        pcd_file = trail_info['pcd_files'][step_idx]
        pcd_data = self._load_pickle_file(pcd_file)

        # 加载BGR图像数据并转换为RGB
        rgb_data = None
        if trail_info['bgr_files'] and step_idx < len(trail_info['bgr_files']):
            bgr_file = trail_info['bgr_files'][step_idx]
            bgr_data = self._load_pickle_file(bgr_file)
            if bgr_data is not None:
                # 快速BGR到RGB转换：交换第0和第2通道
                rgb_data = bgr_data[..., ::-1]

        return pose_data, pcd_data, rgb_data

    def __len__(self) -> int:
        return len(self.valid_samples)

    def preprocess(self, pcd_list, feat_list, all_poses: np.ndarray):
        """
        预处理点云序列、特征序列和姿态

        Args:
            pcd_list: 点云列表，每个元素为 np.ndarray
            feat_list: 特征列表，每个元素为 np.ndarray
            all_poses: 姿态数组 [num_poses, 7]

        Returns:
            pc_list: 处理后的点云列表
            img_feat_list: 处理后的特征列表
            wpt_local: 局部坐标系下的姿态 [num_poses, 3]
        """
        # 确保输入是列表
        if not isinstance(pcd_list, list):
            pcd_list = [pcd_list]
        if not isinstance(feat_list, list):
            feat_list = [feat_list]

        num_frames = len(pcd_list)

        # 归一化RGB特征
        feat_list = [_norm_rgb(feat) for feat in feat_list]

        # 使用外参矩阵对pcd进行变换
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

            # 数据增强 - 使用批处理版本
            if self.augmentation and self.mode == "train":
                from bridgevla.mvt.augmentation import apply_se3_aug_con_shared

                # 堆叠成batch [num_frames, num_points, 3]
                pc_batch = torch.stack(pc_list, dim=0)

                # 转换poses为tensor [num_frames, 7]
                all_poses_tensor = torch.from_numpy(np.array(all_poses)).float()

                # 应用共享增强
                perturbed_poses, pc_batch = apply_se3_aug_con_shared(
                    pcd=pc_batch,
                    action_gripper_pose=all_poses_tensor,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )

                # 分解回列表
                pc_list = [pc_batch[i] for i in range(num_frames)]
                action_trans_con = perturbed_poses[:, :3]
            else:
                # 没有数据增强时，直接使用原始poses
                action_trans_con = torch.from_numpy(np.array(all_poses)).float()[:, :3]

            # 对每个点云应用边界约束
            processed_pc_list = []
            processed_feat_list = []
            for pc, img_feat in zip(pc_list, img_feat_list):
                pc, img_feat = rvt_utils.move_pc_in_bound(
                    pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                )
                processed_pc_list.append(pc[0])
                processed_feat_list.append(img_feat[0])

            # 将点云和wpt放在一个cube里面 (使用第一个点云作为参考)
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

        return final_pc_list, processed_feat_list, wpt_local
        

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取训练样本

        Returns:
            {
                'rgb_image': torch.Tensor,      # 首帧投影后的RGB图像 (3, H, W)
                'rgb_sequence': torch.Tensor,   # RGB图像序列 (T, 3, H, W)
                'heatmap_start': torch.Tensor,  # 起始step的heatmap (1, H, W)
                'heatmap_sequence': torch.Tensor, # heatmap序列 (T, H, W)
                'instruction': str,             # 任务指令
                'trail_name': str,              # 轨迹名称
                'start_step': int,              # 起始step
                'metadata': dict                # 其他元数据
            }
        """
        sample = self.valid_samples[idx]
        trail_info = sample['trail_info']
        start_step = sample['start_step']
        future_steps = sample['future_steps']

        # 1. 加载起始step的数据
        start_pose, start_pcd, start_rgb = self._load_step_data(trail_info, start_step)

        if start_pose is None or start_pcd is None:
            # 如果数据加载失败，抛出错误而不是递归调用
            raise ValueError(f"Failed to load data for sample {idx}. Start pose: {start_pose is not None}, Start pcd: {start_pcd is not None}")

        # 2. 提取所有future_steps对应的future_poses和future_data
        future_poses = []
        future_pcds = []
        future_rgbs = []
        for future_step in future_steps:
            future_pose, future_pcd, future_rgb = self._load_step_data(trail_info, future_step)
            if future_pose is None:
                assert False
            future_poses.append(future_pose)
            future_pcds.append(future_pcd)
            future_rgbs.append(future_rgb)

        # 3. 拼接start和future数据，一起送入preprocess进行处理
        all_poses = [start_pose] + future_poses
        all_pcds = [start_pcd] + future_pcds
        all_rgbs = [start_rgb] + future_rgbs

        # 使用新的preprocess函数处理序列
        processed_pcd_list, processed_rgb_list, processed_poses = self.preprocess(
            all_pcds, all_rgbs, all_poses
        )

        # 分离处理后的数据
        processed_start_pcd = processed_pcd_list[0]
        processed_start_rgb = processed_rgb_list[0]
        processed_future_pcds = processed_pcd_list[1:]
        processed_future_rgbs = processed_rgb_list[1:]

        processed_start_pose = processed_poses[0]
        processed_future_poses = processed_poses[1:]

        # 4. 使用投影接口生成RGB图像序列
        # 4.1 生成起始RGB图像
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_start_pcd, processed_start_rgb
        )  # (1,3,256,256,6)
        rgb_image = rgb_image[0, 0, :, :, 3:]  # (256,256,3)

        # 确保是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()

        # 调整图像尺寸
        if rgb_image.shape[:2] != self.image_size:
            from PIL import Image
            rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
            rgb_pil = rgb_pil.resize((self.image_size[1], self.image_size[0]))
            rgb_image = np.array(rgb_pil) / 255.0

        # 4.2 生成未来RGB图像序列
        rgb_future_list = []
        for future_pcd, future_rgb in zip(processed_future_pcds, processed_future_rgbs):
            future_rgb_image = self.projection_interface.project_pointcloud_to_rgb(
                future_pcd, future_rgb
            )  # (1,3,256,256,6)
            future_rgb_image = future_rgb_image[0, 0, :, :, 3:]  # (256,256,3)

            # 确保是numpy数组
            if isinstance(future_rgb_image, torch.Tensor):
                future_rgb_image = future_rgb_image.cpu().numpy()

            # 调整图像尺寸
            if future_rgb_image.shape[:2] != self.image_size:
                from PIL import Image
                rgb_pil = Image.fromarray((future_rgb_image * 255).astype(np.uint8))
                rgb_pil = rgb_pil.resize((self.image_size[1], self.image_size[0]))
                future_rgb_image = np.array(rgb_pil) / 255.0

            # 转换为tensor并permute
            future_rgb_tensor = torch.from_numpy(future_rgb_image).float().permute(2, 0, 1)  # (3, H, W)
            rgb_future_list.append(future_rgb_tensor)

        # 堆叠成序列 (T, 3, H, W)
        rgb_future = torch.stack(rgb_future_list, dim=0)

        # 4. 使用处理后的poses生成heatmap序列
        img_locations= self.projection_interface.project_pose_to_pixel(
                processed_poses.unsqueeze(0).to(self.projection_interface.renderer_device)
        )

        # 使用用户提供的heatmap接口生成heatmap
        heatmap_sequence= self.projection_interface.generate_heatmap_from_img_locations(
            img_locations,
            self.image_size[0], self.image_size[1],
            self.sigma
        ) # (1,seq_len+1,3,256,256)
        heatmap_sequence = heatmap_sequence[0,:,0,:,:] # (seq_len+1,H,W) # 这里默认取了第一个投影视角

        # 5. 转换为torch tensors
        rgb_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1)  # (3, H, W)
        heatmap_tensor = heatmap_sequence.float()  # (seq_len+1, H, W)
        heatmap_future = heatmap_tensor[1:, :, :]  # (seq_len, H, W)
        heatmap_start = heatmap_tensor[0:1, :, :]  # (1, H, W)

        return {
            'heatmap_start': heatmap_start,
            'raw_rgb_image': start_rgb,
            'rgb_image': rgb_tensor,  # 起始帧RGB图像 (3, H, W)
            'rgb_sequence': rgb_future,  # 未来RGB图像序列 (T, 3, H, W)
            'heatmap_sequence': heatmap_future,  # 未来热力图序列 (T, H, W)
            'img_locations': img_locations,
            'future_poses': processed_future_poses,
            'instruction': trail_info['instruction'],
            'trail_name': trail_info['trail_name'],
            'start_step': start_step,
            'metadata': {
                'sequence_length': self.sequence_length,
                'future_steps': future_steps,
                'image_size': self.image_size
            }
        }

    def get_trail_info(self, idx: int) -> Dict:
        """
        获取指定样本的轨迹信息
        """
        return self.valid_samples[idx]['trail_info']

    def get_sample_info(self, idx: int) -> Dict:
        """
        获取指定样本的详细信息
        """
        return self.valid_samples[idx]



def create_dataloader(data_root: str,
                     projection_interface: ProjectionInterface,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     **dataset_kwargs) -> torch.utils.data.DataLoader:
    """
    创建数据加载器的便利函数

    Args:
        data_root: 数据根目录
        projection_interface: 投影接口实现
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        **dataset_kwargs: 数据集额外参数

    Returns:
        DataLoader实例
    """
    dataset = RobotTrajectoryDataset(
        data_root=data_root,
        projection_interface=projection_interface,
        **dataset_kwargs
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader


if __name__ == "__main__":
    # 使用示例
    data_root = "/share/project/lpy/test/FA_DATA/data/filtered_data/put_the_lion_on_the_top_shelf"

    # 创建示例接口（用户需要替换为实际实现）
    projection_interface = ProjectionInterface()

    # 创建数据集
    dataset = RobotTrajectoryDataset(
        data_root=data_root,
        projection_interface=projection_interface,
        sequence_length=5,
        min_trail_length=10,
        debug=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试加载样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"RGB shape: {sample['rgb_image'].shape}")
        print(f"Heatmap sequence shape: {sample['heatmap_sequence'].shape}")
        print(f"Instruction: {sample['instruction']}")
        print(f"Trail name: {sample['trail_name']}")