"""
Robot Trajectory Dataset for Multi-View Heatmap Prediction (3 Cameras Version)
支持从机器人轨迹数据生成RGB图像和heatmap序列的数据集
新版本：支持3个第三视角相机，点云拼接，动态外参加载

修改说明：
1. 支持3个第三视角相机（3rd_1, 3rd_2, 3rd_3）
2. 从extrinsics.pkl动态加载每条轨迹的外参矩阵
3. 多相机点云自动拼接
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
finetune_path = "/home/lpy/BridgeVLA_dev/finetune"

sys.path.append(finetune_path)
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con_shared

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
                device=None,  # 新增：允许外部指定设备
                ):

        from point_renderer.rvt_renderer import RVTBoxRenderer
        import os

        # 设备选择优先级：
        # 1. 外部传入的 device 参数
        # 2. LOCAL_RANK 环境变量（分布式训练）
        # 3. 默认 cuda:0 或 cpu
        if device is not None:
            self.renderer_device = device
            print(f"[ProjectionInterface] Using device: {self.renderer_device} (specified by user)")
        elif torch.cuda.is_available():
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
            feat = feat + noise
            # 裁剪到 [0, 1] 范围，确保增强后的特征仍然有效
            feat = torch.clamp(feat, 0, 1)

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

        # aug after projection  由于增强后范围可能不在0，1之间，所以去掉
        # if img_aug_after != 0:
        #     stdv = img_aug_after * torch.rand(1, device=img.device)
        #     # values in [-stdv, stdv]
        #     noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
        #     img = torch.clamp(img + noise, -1, 1)
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

        # 裁剪像素坐标到图像边界内，防止超出scene_bounds的pose导致无效坐标
        # pt_img shape: (bs, np, num_img, 2), 最后一维是 (x, y)
        h, w = self.img_size
        pt_img[..., 0] = torch.clamp(pt_img[..., 0], min=0, max=w-1)  # x 坐标裁剪到 [0, w-1]
        pt_img[..., 1] = torch.clamp(pt_img[..., 1], min=0, max=h-1)  # y 坐标裁剪到 [0, h-1]

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

    def visualize_hm(self, heatmaps, h, w, save_path=None):
        """
        可视化多视角heatmap序列并保存到指定路径

        Args:
            heatmaps: torch.Tensor (T, num_views, h*w) - heatmap张量
            h: int - heatmap高度
            w: int - heatmap宽度
            save_path: str - 保存图像的路径，如果为None则不保存

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 将heatmap reshape为 (T, num_views, h, w)
        T, num_views, hw = heatmaps.shape
        assert hw == h * w, f"Expected h*w={h*w}, got {hw}"

        # Reshape heatmaps
        heatmaps_reshaped = heatmaps.view(T, num_views, h, w)

        # 转换为numpy并归一化
        if torch.is_tensor(heatmaps_reshaped):
            heatmaps_np = heatmaps_reshaped.detach().cpu().numpy()
        else:
            heatmaps_np = heatmaps_reshaped

        # 对每个heatmap进行归一化到[0,1]
        heatmaps_normalized = []
        for t in range(T):
            frame_views = []
            for v in range(num_views):
                hm = heatmaps_np[t, v]
                # 归一化到[0, 1]
                hm_min = hm.min()
                hm_max = hm.max()
                if hm_max > hm_min:
                    hm_norm = (hm - hm_min) / (hm_max - hm_min)
                else:
                    hm_norm = hm
                frame_views.append(hm_norm)
            heatmaps_normalized.append(frame_views)

        # 创建可视化图形: 行=时间步，列=视角
        fig, axes = plt.subplots(T, num_views, figsize=(num_views * 3, T * 2.5))

        # 处理单行或单列的情况
        if T == 1 and num_views == 1:
            axes = np.array([[axes]])
        elif T == 1:
            axes = axes.reshape(1, -1)
        elif num_views == 1:
            axes = axes.reshape(-1, 1)

        # 绘制每个heatmap
        for t in range(T):
            for v in range(num_views):
                ax = axes[t, v]
                hm = heatmaps_normalized[t][v]

                # 使用jet colormap显示heatmap
                im = ax.imshow(hm, cmap='jet', interpolation='nearest')

                # 添加标题
                if t == 0:
                    ax.set_title(f'View {v}', fontsize=10, fontweight='bold')
                if v == 0:
                    ax.set_ylabel(f'T{t}', fontsize=9, fontweight='bold')

                # 找到最大值位置并标记
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                ax.plot(max_idx[1], max_idx[0], 'r+', markersize=8, markeredgewidth=2)

                # 移除坐标轴刻度
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.suptitle(f'Multi-View Heatmap Sequence (T={T}, Views={num_views})',
                    fontsize=12, fontweight='bold', y=0.995)

        # 保存图像
        if save_path is not None:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap visualization saved to: {save_path}")

        plt.close(fig)

    def get_position_from_heatmap(self, heatmaps,rev_trans,dyn_cam_info=None, y_q=None,visualize=False, use_softmax=True):
        """
        Estimate the q-values given output from mvt
        :param heatmap: heatmaps output from wan  (bs,view,h*w)
        :param rev_trans  逆变换函数
        :param use_softmax: 是否使用softmax归一化（默认True保持兼容性）
        """
        h ,w = self.img_size
        bs,nc,h_w=heatmaps.shape
        # if visualize:
        #     self.visualize_hm(heatmaps, h, w,save_path="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img/debug.png")

        if use_softmax:
            hm = torch.nn.functional.softmax(heatmaps, 2)
        else:
            # 简单归一化，保持原始分布
            hm = heatmaps / (heatmaps.sum(dim=2, keepdim=True) + 1e-8)
        hm = hm.view(bs, nc, h, w)
        hm=  hm.to(self.renderer_device)
        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        pred_wpt = pred_wpt.squeeze(1)
        pred_wpt = rev_trans(pred_wpt.to("cpu"))

        assert y_q is None

        return pred_wpt





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
                 min_trail_length: int = 5,
                 image_size: Tuple[int, int] = (256, 256),
                 sigma: float = 1.5,
                 augmentation: bool = True,
                 mode="train",
                 scene_bounds: List[float] = [0, -0.45, -0.05, 0.8, 0.55, 0.6],
                 transform_augmentation_xyz=[0.1, 0.1, 0.1],
                 transform_augmentation_rpy=[0.0, 0.0, 20.0], # 确认一下到底应该是多少
                 debug=False,
                 trail_start: int = None,
                 trail_end: int = None,
                 use_merged_pointcloud: bool = True,  # 是否使用拼接后的点云（True）或只使用相机1的点云（False）
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
        self.trail_start = trail_start  # 起始trail编号（如1表示trail_1）
        self.trail_end = trail_end  # 结束trail编号（如50表示trail_50）
        self.use_merged_pointcloud = use_merged_pointcloud  # 是否使用拼接后的点云

        # 扫描所有轨迹数据
        self.trail_data = self._scan_trails()
        self.valid_samples = self._generate_valid_samples()

        # 注意：外参矩阵现在从每个trail的extrinsics.pkl文件中动态加载
        # 不再使用硬编码的全局外参矩阵

        print(f"Found {len(self.trail_data)} trails, {len(self.valid_samples)} valid samples")

    def _scan_trails(self) -> List[Dict]:
        """
        扫描数据目录，收集所有轨迹信息
        """
        trail_data = []

        # 找到所有trail_开头的文件夹
        trail_pattern = os.path.join(self.data_root, "trail_*")
        trail_dirs = sorted(glob.glob(trail_pattern))

        # Debug模式：只使用前2个trails
        if self.debug:
            trail_dirs = trail_dirs[:2]

        # Trail范围过滤
        if self.trail_start is not None or self.trail_end is not None:
            filtered_dirs = []
            for trail_dir in trail_dirs:
                trail_name = os.path.basename(trail_dir)
                # 提取trail编号，例如 "trail_1" -> 1, "trail_50" -> 50
                if trail_name.startswith("trail_"):
                    try:
                        trail_num = int(trail_name.split("_")[1])
                        # 检查是否在指定范围内
                        if self.trail_start is not None and trail_num < self.trail_start:
                            continue
                        if self.trail_end is not None and trail_num > self.trail_end:
                            continue
                        filtered_dirs.append(trail_dir)
                    except (ValueError, IndexError):
                        # 如果无法解析trail编号，保留该trail
                        print(f"Warning: Cannot parse trail number from {trail_name}, including it anyway")
                        filtered_dirs.append(trail_dir)
            trail_dirs = filtered_dirs
            if self.trail_start is not None or self.trail_end is not None:
                print(f"Filtered trails to range [{self.trail_start}, {self.trail_end}]: {len(trail_dirs)} trails")

        for trail_dir in trail_dirs:
            if not os.path.isdir(trail_dir):
                continue

            trail_name = os.path.basename(trail_dir)

            # 检查必要的文件夹是否存在（3相机版本）
            poses_dir = os.path.join(trail_dir, "poses")
            pcd_1_dir = os.path.join(trail_dir, "3rd_1_pcd")
            pcd_2_dir = os.path.join(trail_dir, "3rd_2_pcd")
            pcd_3_dir = os.path.join(trail_dir, "3rd_3_pcd")
            bgr_1_dir = os.path.join(trail_dir, "3rd_1_bgr")
            bgr_2_dir = os.path.join(trail_dir, "3rd_2_bgr")
            bgr_3_dir = os.path.join(trail_dir, "3rd_3_bgr")
            gripper_states_dir = os.path.join(trail_dir, "gripper_states")
            instruction_file = os.path.join(trail_dir, "instruction.txt")
            extrinsics_file = os.path.join(trail_dir, "extrinsics.pkl")

            if not all(os.path.exists(p) for p in [poses_dir, pcd_1_dir, pcd_2_dir, pcd_3_dir, instruction_file, extrinsics_file]):
                print(f"Skipping {trail_name}: missing required directories/files")
                continue

            # 统计step数量
            pose_files = sorted(glob.glob(os.path.join(poses_dir, "*.pkl")))
            pcd_1_files = sorted(glob.glob(os.path.join(pcd_1_dir, "*.pkl")))
            pcd_2_files = sorted(glob.glob(os.path.join(pcd_2_dir, "*.pkl")))
            pcd_3_files = sorted(glob.glob(os.path.join(pcd_3_dir, "*.pkl")))
            bgr_1_files = sorted(glob.glob(os.path.join(bgr_1_dir, "*.pkl")))
            bgr_2_files = sorted(glob.glob(os.path.join(bgr_2_dir, "*.pkl")))
            bgr_3_files = sorted(glob.glob(os.path.join(bgr_3_dir, "*.pkl")))
            gripper_states_files = sorted(glob.glob(os.path.join(gripper_states_dir, "*.pkl"))) if os.path.exists(gripper_states_dir) else []

            # 检查三个相机的pcd数量是否一致
            if not (len(pose_files) == len(pcd_1_files) == len(pcd_2_files) == len(pcd_3_files)):
                print(f"Skipping {trail_name}: pose and pcd count mismatch (pose:{len(pose_files)}, cam1:{len(pcd_1_files)}, cam2:{len(pcd_2_files)}, cam3:{len(pcd_3_files)})")
                continue

            # 检查三个相机的BGR数量是否一致
            if bgr_1_files and bgr_2_files and bgr_3_files:
                if not (len(pose_files) == len(bgr_1_files) == len(bgr_2_files) == len(bgr_3_files)):
                    print(f"Warning {trail_name}: BGR file count mismatch with poses")
                    continue

            # 如果存在gripper_states文件夹，检查数量是否匹配
            if gripper_states_files and len(pose_files) != len(gripper_states_files):
                print(f"Warning {trail_name}: Gripper states file count mismatch with poses ({len(gripper_states_files)} vs {len(pose_files)})")
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
                'pcd_1_dir': pcd_1_dir,
                'pcd_2_dir': pcd_2_dir,
                'pcd_3_dir': pcd_3_dir,
                'bgr_1_dir': bgr_1_dir,
                'bgr_2_dir': bgr_2_dir,
                'bgr_3_dir': bgr_3_dir,
                'gripper_states_dir': gripper_states_dir,
                'extrinsics_file': extrinsics_file,
                'instruction': instruction,
                'num_steps': len(pose_files),
                'pose_files': pose_files,
                'pcd_1_files': pcd_1_files,
                'pcd_2_files': pcd_2_files,
                'pcd_3_files': pcd_3_files,
                'bgr_1_files': bgr_1_files,
                'bgr_2_files': bgr_2_files,
                'bgr_3_files': bgr_3_files,
                'gripper_states_files': gripper_states_files
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
            # 是必要的嘛？ 或者直接删除？
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
        加载指定step的pose、三个相机的点云和BGR图像数据（3相机版本）

        Args:
            trail_info: 轨迹信息
            step_idx: step索引
            pose_only: 是否只返回pose数据

        Returns:
            pose_only=True: (pose_data,)
            pose_only=False: (pose_data, [pcd_1, pcd_2, pcd_3], [rgb_1, rgb_2, rgb_3])
        """
        pose_file = trail_info['pose_files'][step_idx]
        pose_data = self._load_pickle_file(pose_file)

        if pose_only:
            return (pose_data,)

        # 加载三个相机的点云数据
        pcd_1_file = trail_info['pcd_1_files'][step_idx]
        pcd_2_file = trail_info['pcd_2_files'][step_idx]
        pcd_3_file = trail_info['pcd_3_files'][step_idx]
        pcd_1_data = self._load_pickle_file(pcd_1_file)
        pcd_2_data = self._load_pickle_file(pcd_2_file)
        pcd_3_data = self._load_pickle_file(pcd_3_file)

        # 加载三个相机的BGR图像数据并转换为RGB
        rgb_1_data = None
        rgb_2_data = None
        rgb_3_data = None

        if trail_info['bgr_1_files'] and step_idx < len(trail_info['bgr_1_files']):
            bgr_1_file = trail_info['bgr_1_files'][step_idx]
            bgr_1_data = self._load_pickle_file(bgr_1_file)
            if bgr_1_data is not None:
                rgb_1_data = bgr_1_data[..., ::-1]

        if trail_info['bgr_2_files'] and step_idx < len(trail_info['bgr_2_files']):
            bgr_2_file = trail_info['bgr_2_files'][step_idx]
            bgr_2_data = self._load_pickle_file(bgr_2_file)
            if bgr_2_data is not None:
                rgb_2_data = bgr_2_data[..., ::-1]

        if trail_info['bgr_3_files'] and step_idx < len(trail_info['bgr_3_files']):
            bgr_3_file = trail_info['bgr_3_files'][step_idx]
            bgr_3_data = self._load_pickle_file(bgr_3_file)
            if bgr_3_data is not None:
                rgb_3_data = bgr_3_data[..., ::-1]

        return pose_data, [pcd_1_data, pcd_2_data, pcd_3_data], [rgb_1_data, rgb_2_data, rgb_3_data]

    def __len__(self) -> int:
        return len(self.valid_samples)

    def preprocess(self, pcd_list, feat_list, all_poses: np.ndarray, trail_info: Dict):
        """
        预处理点云序列、特征序列和姿态（3相机版本）

        Args:
            pcd_list: 点云列表的列表，每个元素为 [pcd_cam1, pcd_cam2, pcd_cam3]
            feat_list: 特征列表的列表，每个元素为 [feat_cam1, feat_cam2, feat_cam3]
            all_poses: 姿态数组 [num_poses, 7]
            trail_info: 轨迹信息，包含外参文件路径

        Returns:
            pc_list: 处理后的点云列表（拼接后的）
            img_feat_list: 处理后的特征列表（拼接后的）
            wpt_local: 局部坐标系下的姿态 [num_poses, 3]
            rot_grip: rotaion and grip (num_poses,3 )
        """
        # 确保输入是列表
        if not isinstance(pcd_list, list):
            pcd_list = [pcd_list]
        if not isinstance(feat_list, list):
            feat_list = [feat_list]

        num_frames = len(pcd_list)

        # 加载外参矩阵（3个相机的外参）
        extrinsics_file = trail_info['extrinsics_file']
        with open(extrinsics_file, 'rb') as f:
            extrinsics_dict = pickle.load(f)

        extrinsic_matrix_1 = extrinsics_dict['3rd_1']
        extrinsic_matrix_2 = extrinsics_dict['3rd_2']
        extrinsic_matrix_3 = extrinsics_dict['3rd_3']

        # 处理每一帧的3个相机数据
        merged_pcd_list = []
        merged_feat_list = []

        for frame_idx in range(num_frames):
            # 获取这一帧的3个相机的点云和特征
            frame_pcds = pcd_list[frame_idx]  # [pcd_cam1, pcd_cam2, pcd_cam3]
            frame_feats = feat_list[frame_idx]  # [feat_cam1, feat_cam2, feat_cam3]

            # 归一化RGB特征
            frame_feats_norm = [_norm_rgb(feat) for feat in frame_feats]

            # 对3个相机的点云分别应用外参变换
            pcd_cam1_base = convert_pcd_to_base(extrinsic_martix=extrinsic_matrix_1, pcd=frame_pcds[0])
            pcd_cam2_base = convert_pcd_to_base(extrinsic_martix=extrinsic_matrix_2, pcd=frame_pcds[1])
            pcd_cam3_base = convert_pcd_to_base(extrinsic_martix=extrinsic_matrix_3, pcd=frame_pcds[2])

            # 转换为torch张量并展平
            pcd_cam1_flat = torch.from_numpy(np.ascontiguousarray(pcd_cam1_base)).float().view(-1, 3)
            pcd_cam2_flat = torch.from_numpy(np.ascontiguousarray(pcd_cam2_base)).float().view(-1, 3)
            pcd_cam3_flat = torch.from_numpy(np.ascontiguousarray(pcd_cam3_base)).float().view(-1, 3)

            # 展平RGB特征
            feat_cam1_flat = ((frame_feats_norm[0].view(-1, 3) + 1) / 2).float()
            feat_cam2_flat = ((frame_feats_norm[1].view(-1, 3) + 1) / 2).float()
            feat_cam3_flat = ((frame_feats_norm[2].view(-1, 3) + 1) / 2).float()

            # 根据配置决定是否拼接3个相机的点云和特征
            if self.use_merged_pointcloud:
                # 拼接3个相机的点云和特征
                merged_pcd = torch.cat([pcd_cam1_flat, pcd_cam2_flat, pcd_cam3_flat], dim=0)  # [num_points_total, 3]
                merged_feat = torch.cat([feat_cam1_flat, feat_cam2_flat, feat_cam3_flat], dim=0)  # [num_points_total, 3]
            else:
                # 只使用相机3（3rd_1）的点云和特征
                merged_pcd = pcd_cam3_flat  # [num_points_cam1, 3]
                merged_feat = feat_cam3_flat  # [num_points_cam1, 3]

            merged_pcd_list.append(merged_pcd)
            merged_feat_list.append(merged_feat)

        # 现在merged_pcd_list和merged_feat_list包含拼接后的点云和特征
        # 后续处理保持不变
        pc_list = merged_pcd_list  # 已经是展平的torch张量了
        img_feat_list = merged_feat_list  # 已经是归一化的torch张量了

        with torch.no_grad():

            # 数据增强 - 使用批处理版本
            # 同时返回posees
            if self.augmentation and self.mode == "train":
                from bridgevla.mvt.augmentation import apply_se3_aug_con_shared

                # 堆叠成batch [num_frames, num_points, 3]
                pc_batch = torch.stack(pc_list, dim=0)

                # 转换poses为tensor [num_frames, 7]
                all_poses_tensor = torch.from_numpy(np.array(all_poses)).float() # 对于我们的数据集而言 rot为 w xyz

                # FIX: augmentation.py的ver=2（默认）在第624行有四元数格式bug：
                # 它直接将action_gripper_pose[:, 3:7]传给Rotation.from_quat，
                # 但Rotation.from_quat期望[x,y,z,w]格式，而我们的数据是[w,x,y,z]格式
                # 解决方案：在调用前将wxyz转换为xyzw，让bug能"正确"处理
                position = all_poses_tensor[:, :3]  # [x, y, z]
                quat_wxyz = all_poses_tensor[:, 3:]  # [w, x, y, z]
                quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
                all_poses_tensor_fixed = torch.cat([position, quat_xyzw], dim=1)  # [x,y,z, x,y,z,w]

                # 应用共享增强
                perturbed_poses, pc_batch = apply_se3_aug_con_shared(
                    pcd=pc_batch,
                    action_gripper_pose=all_poses_tensor_fixed,  # 传入xyzw格式
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                ) # bs,7  (pos,xyzw)

                # 分解回列表
                pc_list = [pc_batch[i] for i in range(num_frames)]
                action_trans_con = perturbed_poses[:, :3]
                action_rot_xyzw=perturbed_poses[:, 3:]
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
                pc, img_feat = move_pc_in_bound(
                    pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                )
                processed_pc_list.append(pc[0])
                processed_feat_list.append(img_feat[0])

            # 将点云和wpt放在一个cube里面 (使用第一个点云作为参考)
            wpt_local, rev_trans = mvt_utils.place_pc_in_cube( # 不会影响到旋转
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

        return final_pc_list, processed_feat_list, wpt_local,action_rot_xyzw,rev_trans
        

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取训练样本

        Returns:
            {
                'rgb_image': torch.Tensor,      # 首帧投影后的多视角RGB图像 (num_views, 3, H, W) uint8 [0, 255]
                'rgb_sequence': torch.Tensor,   # 未来帧的多视角RGB图像序列 (T, num_views, 3, H, W) uint8 [0, 255]
                'heatmap_start': torch.Tensor,  # 起始step的多视角heatmap (1, num_views, H, W)
                'heatmap_sequence': torch.Tensor, # 未来帧的多视角heatmap序列 (T, num_views, H, W)
                'instruction': str,             # 任务指令
                'trail_name': str,              # 轨迹名称
                'start_step': int,              # 起始step
                'img_locations': torch.Tensor,  # 像素位置 (bs, num_poses, num_views, 2)
                'future_poses': torch.Tensor,   # 未来poses (T, 7) - 包括position和rotation (xyzw)
                'start_gripper_state': bool,    # 起始帧gripper state
                'future_gripper_states': torch.Tensor, # 未来gripper states序列 (T,)
                'raw_rgb_image': np.ndarray,    # 原始RGB图像
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
        future_gripper_states = []

        # 加载start step的gripper state
        if trail_info['gripper_states_files']:
            start_gripper_file = trail_info['gripper_states_files'][start_step]
            start_gripper_state = self._load_pickle_file(start_gripper_file)
        else:
            start_gripper_state = None

        for future_step in future_steps:
            future_pose, future_pcd, future_rgb = self._load_step_data(trail_info, future_step)
            if future_pose is None:
                assert False
            future_poses.append(future_pose)
            future_pcds.append(future_pcd)
            future_rgbs.append(future_rgb)

            # 加载gripper state
            if trail_info['gripper_states_files']:
                gripper_file = trail_info['gripper_states_files'][future_step]
                gripper_state = self._load_pickle_file(gripper_file)
                future_gripper_states.append(gripper_state)

        # 3. 拼接start和future数据，一起送入preprocess进行处理
        all_poses = [start_pose] + future_poses
        all_pcds = [start_pcd] + future_pcds
        all_rgbs = [start_rgb] + future_rgbs

        # 使用新的preprocess函数处理序列（添加trail_info参数以加载外参）
        processed_pcd_list, processed_rgb_list, processed_pos, processed_rot_xyzw,rev_trans = self.preprocess(
            all_pcds, all_rgbs, all_poses, trail_info
        )# 同时返回rotation和gripper  

        # 分离处理后的数据
        processed_start_pcd = processed_pcd_list[0]
        processed_start_rgb = processed_rgb_list[0]
        processed_future_pcds = processed_pcd_list[1:]
        processed_future_rgbs = processed_rgb_list[1:]

        processed_poses=torch.cat(( processed_pos, processed_rot_xyzw),dim=1) # num,7
        processed_start_pose = processed_poses[0]
        processed_future_poses = processed_poses[1:]

        # 4. 使用投影接口生成RGB图像序列
        # 4.1 生成起始RGB图像
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_start_pcd, processed_start_rgb
        )  # (1, num_views, H, W, 6)
        rgb_image = rgb_image[0, :, :, :, 3:]  # (num_views, H, W, 3)

        # 确保是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()

        # rgb_image shape: (num_views, H, W, 3)
        # projection_interface已经返回了正确尺寸，无需调整
        # 转换到 [0, 255] uint8 范围（图像已经在 [0, 1] 范围内）
        rgb_image = (rgb_image * 255).astype(np.uint8)  # (num_views, H, W, 3)
        num_views = rgb_image.shape[0]

        # 4.2 生成未来RGB图像序列 - 支持多视角
        rgb_future_list = []
        for future_pcd, future_rgb in zip(processed_future_pcds, processed_future_rgbs):
            future_rgb_image = self.projection_interface.project_pointcloud_to_rgb(
                future_pcd, future_rgb
            )  # (1, num_views, H, W, 6)
            future_rgb_image = future_rgb_image[0, :, :, :, 3:]  # (num_views, H, W, 3)

            # 确保是numpy数组
            if isinstance(future_rgb_image, torch.Tensor):
                future_rgb_image = future_rgb_image.cpu().numpy()

            # projection_interface已经返回了正确尺寸，无需调整
            # 转换到 [0, 255] uint8 范围（图像已经在 [0, 1] 范围内）
            future_rgb_image = (future_rgb_image * 255).astype(np.uint8)  # (num_views, H, W, 3)

            # 转换为tensor并permute: (num_views, 3, H, W)
            future_rgb_tensor = torch.from_numpy(future_rgb_image).permute(0, 3, 1, 2)
            rgb_future_list.append(future_rgb_tensor)

        # 堆叠成序列 (T, num_views, 3, H, W)
        rgb_future = torch.stack(rgb_future_list, dim=0)

        # 4. 使用处理后的poses生成heatmap序列 - 支持多视角
        img_locations= self.projection_interface.project_pose_to_pixel(
                processed_pos.unsqueeze(0).to(self.projection_interface.renderer_device) 
        )  # (bs, num_poses, num_views, 2)

        # 使用用户提供的heatmap接口生成heatmap
        heatmap_sequence= self.projection_interface.generate_heatmap_from_img_locations(
            img_locations,
            self.image_size[0], self.image_size[1],
            self.sigma
        ) # (bs, seq_len+1, num_views, H, W)
        heatmap_sequence = heatmap_sequence[0, :, :, :, :]  # (seq_len+1, num_views, H, W)

        # 5. 转换为torch tensors - 支持多视角
        # rgb_image shape: (num_views, H, W, 3) -> (num_views, 3, H, W)
        # 注意：rgb_image已经是uint8 [0, 255]格式
        rgb_tensor = torch.from_numpy(rgb_image).permute(0, 3, 1, 2)

        # heatmap_sequence shape: (seq_len+1, num_views, H, W)
        heatmap_tensor = heatmap_sequence.float()
        heatmap_future = heatmap_tensor[1:, :, :, :]  # (seq_len, num_views, H, W)
        heatmap_start = heatmap_tensor[0:1, :, :, :]  # (1, num_views, H, W)

        # 转换gripper states为tensor
        if future_gripper_states:
            future_gripper_states_tensor = torch.tensor(future_gripper_states, dtype=torch.bool)
        else:
            future_gripper_states_tensor = None

        result = {
            'heatmap_start': heatmap_start,
            'raw_rgb_image': start_rgb,
            'rgb_image': rgb_tensor,  # 起始帧RGB图像 (num_views, 3, H, W) uint8 [0, 255]
            'rgb_sequence': rgb_future,  # 未来RGB图像序列 (T, num_views, 3, H, W) uint8 [0, 255]
            'heatmap_sequence': heatmap_future,  # 未来热力图序列 (T, num_views, H, W)
            'img_locations': img_locations,
            'start_pose':processed_start_pose ,
            'future_poses': processed_future_poses, # 未来的动作序列，包括rotation  xyzw
            'start_gripper_state': start_gripper_state,  # 起始帧gripper state (bool)
            'future_gripper_states': future_gripper_states_tensor,  # 未来gripper states序列 (T,)
            'instruction': trail_info['instruction'],
            'trail_name': trail_info['trail_name'],
            'start_step': start_step,
            'rev_trans':rev_trans,
            'metadata': {
                'sequence_length': self.sequence_length,
                'future_steps': future_steps,
                'image_size': self.image_size
            }
        }

        return result


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
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端

    # 使用示例
    data_root = "/data/wxn/V2W_Real/put_the_lion_on_the_top_shelf"
    output_dir = "./debug_visualizations"
    os.makedirs(output_dir, exist_ok=True)

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

    # 测试加载样本并可视化
    if len(dataset) > 0:
        # 采样多个样本进行可视化
        num_samples_to_visualize = min(3, len(dataset))

        for sample_idx in range(num_samples_to_visualize):
            print(f"\n{'='*60}")
            print(f"Visualizing sample {sample_idx}...")
            print(f"{'='*60}")

            sample = dataset[sample_idx]

            # 打印基本信息
            print(f"Instruction: {sample['instruction']}")
            print(f"Trail name: {sample['trail_name']}")
            print(f"Start step: {sample['start_step']}")
            print("\nData shapes:")
            print(f"  RGB image (start frame): {sample['rgb_image'].shape}")
            print(f"  RGB sequence (future frames): {sample['rgb_sequence'].shape}")
            print(f"  Heatmap start: {sample['heatmap_start'].shape}")
            print(f"  Heatmap sequence: {sample['heatmap_sequence'].shape}")

            # 提取数据
            rgb_start = sample['rgb_image']  # (num_views, 3, H, W)
            rgb_sequence = sample['rgb_sequence']  # (T, num_views, 3, H, W)
            heatmap_start = sample['heatmap_start']  # (1, num_views, H, W)
            heatmap_sequence = sample['heatmap_sequence']  # (T, num_views, H, W)

            num_views = rgb_start.shape[0]
            num_frames = rgb_sequence.shape[0] + 1  # +1 包括起始帧

            print("\nVisualization info:")
            print(f"  Number of views: {num_views}")
            print(f"  Number of frames (including start): {num_frames}")

            # 创建大图: 每一行是一个视角，每一列是一个时间步
            # 上半部分显示RGB，下半部分显示heatmap
            fig = plt.figure(figsize=(4 * num_frames, 6 * num_views))

            for view_idx in range(num_views):
                # RGB序列可视化
                for frame_idx in range(num_frames):
                    # 计算子图位置: 每个视角占2行(RGB + Heatmap)
                    subplot_idx = view_idx * 2 * num_frames + frame_idx + 1
                    ax = plt.subplot(num_views * 2, num_frames, subplot_idx)

                    if frame_idx == 0:
                        # 显示起始帧
                        rgb_frame = rgb_start[view_idx]  # (3, H, W)
                    else:
                        # 显示未来帧
                        rgb_frame = rgb_sequence[frame_idx - 1, view_idx]  # (3, H, W)

                    # 转换为 (H, W, 3) 并归一化到 [0, 1]
                    rgb_img = rgb_frame.permute(1, 2, 0).cpu().numpy()
                    rgb_img = np.clip((rgb_img + 1) / 2, 0, 1)  # 从 [-1, 1] 转到 [0, 1]

                    ax.imshow(rgb_img)
                    ax.axis('off')
                    if frame_idx == 0:
                        ax.set_title(f'View {view_idx} - Start (RGB)', fontsize=10, fontweight='bold')
                    else:
                        ax.set_title(f'View {view_idx} - Frame {frame_idx} (RGB)', fontsize=10)

                # Heatmap序列可视化
                for frame_idx in range(num_frames):
                    # 计算子图位置: heatmap在RGB下方
                    subplot_idx = view_idx * 2 * num_frames + num_frames + frame_idx + 1
                    ax = plt.subplot(num_views * 2, num_frames, subplot_idx)

                    if frame_idx == 0:
                        # 显示起始帧heatmap
                        heatmap_frame = heatmap_start[0, view_idx]  # (H, W)
                    else:
                        # 显示未来帧heatmap
                        heatmap_frame = heatmap_sequence[frame_idx - 1, view_idx]  # (H, W)

                    # 显示heatmap
                    heatmap_img = heatmap_frame.cpu().numpy()
                    im = ax.imshow(heatmap_img, cmap='hot', vmin=0, vmax=1)

                    # 找到峰值位置并标记
                    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_img), heatmap_img.shape)
                    ax.plot(peak_x, peak_y, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)

                    ax.axis('off')
                    if frame_idx == 0:
                        ax.set_title(f'View {view_idx} - Start (Heatmap)\nPeak: ({peak_x}, {peak_y})',
                                   fontsize=10, fontweight='bold')
                    else:
                        ax.set_title(f'View {view_idx} - Frame {frame_idx} (Heatmap)\nPeak: ({peak_x}, {peak_y})',
                                   fontsize=10)

            # 添加整体标题
            fig.suptitle(f'Sample {sample_idx}: {sample["instruction"][:80]}\n' +
                        f'Trail: {sample["trail_name"]}, Start Step: {sample["start_step"]}',
                        fontsize=14, fontweight='bold', y=0.995)

            plt.tight_layout()

            # 保存图片
            output_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_visualization.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Visualization saved to: {output_path}")

            # 保存详细信息到文本文件
            info_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Sample {sample_idx} Information\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Instruction: {sample['instruction']}\n")
                f.write(f"Trail name: {sample['trail_name']}\n")
                f.write(f"Start step: {sample['start_step']}\n\n")
                f.write("Data Shapes:\n")
                f.write(f"  RGB image (start): {tuple(sample['rgb_image'].shape)}\n")
                f.write(f"  RGB sequence (future): {tuple(sample['rgb_sequence'].shape)}\n")
                f.write(f"  Heatmap start: {tuple(sample['heatmap_start'].shape)}\n")
                f.write(f"  Heatmap sequence: {tuple(sample['heatmap_sequence'].shape)}\n\n")
                f.write(f"Number of views: {num_views}\n")
                f.write(f"Number of frames: {num_frames}\n")
                f.write(f"\nMetadata: {sample['metadata']}\n")

            print(f"✓ Info saved to: {info_path}")

        print(f"\n{'='*60}")
        print(f"All visualizations saved to: {output_dir}")
        print(f"{'='*60}")