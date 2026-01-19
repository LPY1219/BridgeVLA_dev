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


    def project_pointcloud_to_rgb(self, pointclouds, feats, img_aug_before=0.1, img_aug_after=0.05) -> np.ndarray:
        """
        将3个点云分别投影到指定视角生成RGB图像，并从每个投影结果中选择对应的view

        Args:
            pointclouds: 3个点云数据的列表 [pcd_cam1, pcd_cam2, pcd_cam3]，每个为 (N, 3)
            feats: 3个颜色数据的列表 [feat_cam1, feat_cam2, feat_cam3]，每个为 (N, 3)

        Returns:
            RGB图像 (1, 3, H, W, 6) 范围[0, 1]
            映射关系：
            - 最终图像的view 0 来自相机1的投影view 0
            - 最终图像的view 1 来自相机3的投影view 1
            - 最终图像的view 2 来自相机2的投影view 2
        """
        # 检查输入
        if not isinstance(pointclouds, list):
            # 如果传入的是单个点云（兼容旧的调用方式），直接投影
            pointcloud = pointclouds
            feat = feats

            # aug before projection
            if img_aug_before !=0:
                stdv = img_aug_before * torch.rand(1, device=feat.device)
                noise = stdv * ((2 * torch.rand(*feat.shape, device=feat.device)) - 1)
                feat = feat + noise
                feat = torch.clamp(feat, 0, 1)

            # 确保数据在正确的设备上
            renderer_device = self.renderer_device
            if hasattr(pointcloud, 'device') and str(pointcloud.device) != str(renderer_device):
                pointcloud = pointcloud.to(renderer_device)
            if hasattr(feat, 'device') and str(feat.device) != str(renderer_device):
                feat = feat.to(renderer_device)

            max_pc = 1.0 if len(pointcloud) == 0 else torch.max(torch.abs(pointcloud))
            img = self.renderer(
                    pointcloud,
                    torch.cat((pointcloud / max_pc, feat), dim=-1),
                    fix_cam=True,
                    dyn_cam_info=None
                ).unsqueeze(0)
            return img

        # 新的逻辑：3个点云分别投影
        assert len(pointclouds) == 3, f"必须提供3个点云，当前提供了{len(pointclouds)}个"
        assert len(feats) == 3, f"必须提供3个特征，当前提供了{len(feats)}个"

        # 定义view选择映射：
        # 相机0 → 选择view 0 → 放在位置0
        # 相机1 → 选择view 2 → 放在位置2
        # 相机2 → 选择view 1 → 放在位置1
        view_index_mapping = [0, 2, 1]  # 相机i应该选择的view索引
        final_position_mapping = [0, 2, 1]  # 相机i的结果应该放在的最终位置

        # 创建一个列表来存储最终的views，按照最终位置顺序
        result_views = [None, None, None]

        for i, (pointcloud, feat) in enumerate(zip(pointclouds, feats)):
            # aug before projection
            if img_aug_before != 0:
                stdv = img_aug_before * torch.rand(1, device=feat.device)
                noise = stdv * ((2 * torch.rand(*feat.shape, device=feat.device)) - 1)
                feat = feat + noise
                feat = torch.clamp(feat, 0, 1)

            # 确保数据在正确的设备上
            renderer_device = self.renderer_device
            if hasattr(pointcloud, 'device') and str(pointcloud.device) != str(renderer_device):
                pointcloud = pointcloud.to(renderer_device)
            if hasattr(feat, 'device') and str(feat.device) != str(renderer_device):
                feat = feat.to(renderer_device)

            max_pc = 1.0 if len(pointcloud) == 0 else torch.max(torch.abs(pointcloud))

            # 投影得到 (num_views, H, W, 6)
            img = self.renderer(
                pointcloud,
                torch.cat((pointcloud / max_pc, feat), dim=-1),
                fix_cam=True,
                dyn_cam_info=None
            )

            # 根据映射选择对应的view
            # img shape: (num_views, H, W, 6)
            view_index = view_index_mapping[i]  # 相机i应该选择的view
            final_position = final_position_mapping[i]  # 相机i的结果应该放在的位置
            selected_view = img[view_index:view_index+1, :, :, :]  # (1, H, W, 6)
            result_views[final_position] = selected_view

        # 组合3个view成 (3, H, W, 6)
        result = torch.cat(result_views, dim=0)  # (3, H, W, 6)
        result = result.unsqueeze(0)  # (1, 3, H, W, 6)

        return result


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
        if visualize:
            self.visualize_hm(heatmaps, h, w,save_path="/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/wanvideo/model_inference/heatmap_inference_results/debug_img/debug.png")

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
                 scene_bounds: List[float] = [-0.1, -0.5, -0.1, 0.9, 0.5, 0.9],
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

            # 保存3个相机的点云和特征为独立列表，用于分别投影
            # merged_pcd_list 的每个元素是 [pcd_cam1, pcd_cam2, pcd_cam3]
            merged_pcd_list.append([pcd_cam1_flat, pcd_cam2_flat, pcd_cam3_flat])
            merged_feat_list.append([feat_cam1_flat, feat_cam2_flat, feat_cam3_flat])

        # 现在merged_pcd_list和merged_feat_list包含3个独立的点云列表
        # 每个元素是 [pcd_cam1, pcd_cam2, pcd_cam3]
        pc_list = merged_pcd_list
        img_feat_list = merged_feat_list

        with torch.no_grad():

            # 数据增强 - 使用批处理版本
            # 同时返回posees
            if self.augmentation and self.mode == "train":
                from bridgevla.mvt.augmentation import apply_se3_aug_con_shared

                # 先拼接3个相机的点云用于数据增强
                # 堆叠成batch [num_frames, num_points, 3]
                pc_batch_list = []
                for frame_pcs in pc_list:
                    # frame_pcs = [pcd_cam1, pcd_cam2, pcd_cam3]
                    merged_pc = torch.cat(frame_pcs, dim=0)  # 拼接3个相机的点云
                    pc_batch_list.append(merged_pc)
                pc_batch = torch.stack(pc_batch_list, dim=0)

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

                # 增强后的点云需要重新分解为3个独立的点云
                # 首先记录每个相机的点云数量
                num_points_per_cam = [len(frame_pcs[i]) for i, frame_pcs in enumerate(pc_list) for i in range(3)]
                num_points_cam1 = len(pc_list[0][0])
                num_points_cam2 = len(pc_list[0][1])
                num_points_cam3 = len(pc_list[0][2])

                # 将增强后的拼接点云分解回3个相机
                new_pc_list = []
                for i in range(num_frames):
                    augmented_pc = pc_batch[i]  # 拼接后的点云
                    # 分解为3个相机的点云
                    pcd_cam1 = augmented_pc[:num_points_cam1]
                    pcd_cam2 = augmented_pc[num_points_cam1:num_points_cam1+num_points_cam2]
                    pcd_cam3 = augmented_pc[num_points_cam1+num_points_cam2:]
                    new_pc_list.append([pcd_cam1, pcd_cam2, pcd_cam3])
                pc_list = new_pc_list

                action_trans_con = perturbed_poses[:, :3]
                action_rot_xyzw=perturbed_poses[:, 3:]
            else:
                # 没有数据增强时，直接使用原始poses
                action_trans_con = torch.from_numpy(np.array(all_poses)).float()[:, :3]
                # 将wxyz格式转换为xyzw格式
                quat_wxyz = torch.from_numpy(np.array(all_poses)).float()[:, 3:]
                action_rot_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]

            # 对每个帧的3个相机点云分别应用边界约束
            processed_pc_list = []
            processed_feat_list = []
            for frame_pcs, frame_feats in zip(pc_list, img_feat_list):
                # frame_pcs = [pcd_cam1, pcd_cam2, pcd_cam3]
                # frame_feats = [feat_cam1, feat_cam2, feat_cam3]
                processed_frame_pcs = []
                processed_frame_feats = []
                for pc, img_feat in zip(frame_pcs, frame_feats):
                    pc, img_feat = move_pc_in_bound(
                        pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                    )
                    processed_frame_pcs.append(pc[0])
                    processed_frame_feats.append(img_feat[0])
                processed_pc_list.append(processed_frame_pcs)
                processed_feat_list.append(processed_frame_feats)

            # 将点云和wpt放在一个cube里面 (使用第一帧拼接后的点云作为参考)
            # 先拼接第一帧的3个相机点云
            first_frame_merged = torch.cat(processed_pc_list[0], dim=0)
            wpt_local, rev_trans = mvt_utils.place_pc_in_cube( # 不会影响到旋转
                first_frame_merged,
                action_trans_con,
                with_mean_or_bounds=False,
                scene_bounds=self.scene_bounds,
            )

            # 对每个帧的3个相机点云分别应用place_pc_in_cube
            final_pc_list = []
            for frame_pcs in processed_pc_list:
                # frame_pcs = [pcd_cam1, pcd_cam2, pcd_cam3]
                final_frame_pcs = []
                for pc in frame_pcs:
                    pc_normalized = mvt_utils.place_pc_in_cube(
                        pc,
                        with_mean_or_bounds=False,
                        scene_bounds=self.scene_bounds,
                    )[0]
                    final_frame_pcs.append(pc_normalized)
                final_pc_list.append(final_frame_pcs)

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
        # processed_pcd_list[i] = [pcd_cam1, pcd_cam2, pcd_cam3]
        processed_start_pcd = processed_pcd_list[0]  # [pcd_cam1, pcd_cam2, pcd_cam3]
        processed_start_rgb = processed_rgb_list[0]  # [feat_cam1, feat_cam2, feat_cam3]
        processed_future_pcds = processed_pcd_list[1:]
        processed_future_rgbs = processed_rgb_list[1:]

        processed_poses=torch.cat(( processed_pos, processed_rot_xyzw),dim=1) # num,7
        processed_start_pose = processed_poses[0]
        processed_future_poses = processed_poses[1:]

        # 4. 使用投影接口生成RGB图像序列
        # 4.1 生成起始RGB图像
        # 传入3个独立的点云列表，project_pointcloud_to_rgb会分别投影并选择对应的view
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_start_pcd, processed_start_rgb
        )  # (1, 3, H, W, 6) - 第i个view来自第i个点云投影的第i个视角
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
        # future_pcd 和 future_rgb 都是包含3个相机点云/特征的列表
        rgb_future_list = []
        for future_pcd, future_rgb in zip(processed_future_pcds, processed_future_rgbs):
            # future_pcd = [pcd_cam1, pcd_cam2, pcd_cam3]
            # future_rgb = [feat_cam1, feat_cam2, feat_cam3]
            future_rgb_image = self.projection_interface.project_pointcloud_to_rgb(
                future_pcd, future_rgb
            )  # (1, 3, H, W, 6) - 第i个view来自第i个点云投影的第i个视角
            future_rgb_image = future_rgb_image[0, :, :, :, 3:]  # (3, H, W, 3)

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


def test_projection_reprojection_error(
    dataset: RobotTrajectoryDataset,
    projection_interface: ProjectionInterface,
    num_samples: int = 10,
    output_dir: str = "./projection_test_results",
    visualize: bool = True,
):
    """
    测试投影和反投影的误差

    大致思路：
    1. 从数据集获取样本，得到ground truth的3D位置和对应的heatmap
    2. 从heatmap中通过峰值检测反推3D位置
    3. 计算反推位置和ground truth位置的误差
    4. 可视化投影图像和误差统计

    Args:
        dataset: RobotTrajectoryDataset实例
        projection_interface: ProjectionInterface实例
        num_samples: 测试样本数量
        output_dir: 输出目录
        visualize: 是否可视化

    Returns:
        dict: 包含误差统计信息
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    os.makedirs(output_dir, exist_ok=True)

    # 存储误差统计
    all_errors = []  # 每个样本每个时间步的3D位置误差
    all_errors_per_axis = {'x': [], 'y': [], 'z': []}  # 每个轴的误差
    sample_results = []  # 每个样本的详细结果

    num_samples = min(num_samples, len(dataset))
    print(f"\n{'='*60}")
    print(f"Testing projection/reprojection error on {num_samples} samples")
    print(f"{'='*60}\n")

    for sample_idx in range(num_samples):
        print(f"Processing sample {sample_idx + 1}/{num_samples}...")

        try:
            sample = dataset[sample_idx]
        except Exception as e:
            print(f"  Error loading sample {sample_idx}: {e}")
            continue

        # 获取数据
        heatmap_start = sample['heatmap_start']  # (1, num_views, H, W)
        heatmap_sequence = sample['heatmap_sequence']  # (T, num_views, H, W)
        start_pose = sample['start_pose']  # (7,) - position + rotation (xyzw)
        future_poses = sample['future_poses']  # (T, 7)
        rev_trans = sample['rev_trans']  # 逆变换函数
        rgb_image = sample['rgb_image']  # (num_views, 3, H, W)
        rgb_sequence = sample['rgb_sequence']  # (T, num_views, 3, H, W)

        # 合并所有heatmap和pose用于统一处理
        # heatmap_all: (T+1, num_views, H, W)
        heatmap_all = torch.cat([heatmap_start, heatmap_sequence], dim=0)

        # ground truth位置 (T+1, 3)
        # 注意：sample中存储的pose是cube空间中的位置，需要用rev_trans转换回原始空间
        # 这样才能和get_position_from_heatmap的输出（原始空间）保持一致
        if isinstance(future_poses, torch.Tensor):
            gt_positions_cube = torch.cat([start_pose[:3].unsqueeze(0), future_poses[:, :3]], dim=0)
        else:
            gt_positions_cube = torch.cat([start_pose[:3].unsqueeze(0), torch.tensor(future_poses)[:, :3]], dim=0)

        # 将GT位置从cube空间转换到原始空间
        gt_positions = rev_trans(gt_positions_cube)

        num_frames = heatmap_all.shape[0]
        num_views = heatmap_all.shape[1]
        h, w = heatmap_all.shape[2], heatmap_all.shape[3]

        # 逐帧从heatmap反推3D位置
        sample_errors = []
        pred_positions_list = []

        for frame_idx in range(num_frames):
            # 获取该帧的heatmap (num_views, H, W) -> (1, num_views, H*W)
            hm_frame = heatmap_all[frame_idx]  # (num_views, H, W)
            hm_flat = hm_frame.view(1, num_views, h * w)  # (1, num_views, H*W)

            # 使用get_position_from_heatmap从heatmap反推3D位置
            # 注意：这个函数期望heatmap已经是softmax处理后的分布
            try:
                pred_position = projection_interface.get_position_from_heatmap(
                    hm_flat.to(projection_interface.renderer_device),
                    rev_trans,
                    dyn_cam_info=None,
                    y_q=None,
                    visualize=False,
                    use_softmax=True  # 对heatmap应用softmax
                )  # (1, 3)
                pred_position = pred_position.squeeze(0)  # (3,)
            except Exception as e:
                print(f"  Error in get_position_from_heatmap for frame {frame_idx}: {e}")
                pred_position = torch.zeros(3)

            pred_positions_list.append(pred_position)

            # 计算误差
            gt_pos = gt_positions[frame_idx].cpu()
            pred_pos = pred_position.cpu()

            error_vec = gt_pos - pred_pos  # (3,)
            error_dist = torch.norm(error_vec).item()  # 欧氏距离误差

            sample_errors.append({
                'frame_idx': frame_idx,
                'gt_position': gt_pos.numpy(),
                'pred_position': pred_pos.numpy(),
                'error_distance': error_dist,
                'error_x': error_vec[0].item(),
                'error_y': error_vec[1].item(),
                'error_z': error_vec[2].item(),
            })

            all_errors.append(error_dist)
            all_errors_per_axis['x'].append(abs(error_vec[0].item()))
            all_errors_per_axis['y'].append(abs(error_vec[1].item()))
            all_errors_per_axis['z'].append(abs(error_vec[2].item()))

        pred_positions = torch.stack(pred_positions_list, dim=0)  # (T+1, 3)

        # 保存该样本的结果
        sample_result = {
            'sample_idx': sample_idx,
            'trail_name': sample['trail_name'],
            'start_step': sample['start_step'],
            'instruction': sample['instruction'],
            'num_frames': num_frames,
            'frame_errors': sample_errors,
            'mean_error': np.mean([e['error_distance'] for e in sample_errors]),
            'max_error': np.max([e['error_distance'] for e in sample_errors]),
        }
        sample_results.append(sample_result)

        print(f"  Trail: {sample['trail_name']}, Mean Error: {sample_result['mean_error']*1000:.2f}mm, "
              f"Max Error: {sample_result['max_error']*1000:.2f}mm")

        # 可视化
        if visualize:
            # 为了在图像上标注位置，需要使用cube空间的坐标进行投影
            # 因为renderer是在cube空间中工作的
            _visualize_projection_test(
                sample_idx=sample_idx,
                rgb_image=rgb_image,
                rgb_sequence=rgb_sequence,
                heatmap_all=heatmap_all,
                gt_positions_cube=gt_positions_cube,  # cube空间，用于2D投影
                gt_positions_world=gt_positions,  # 原始空间，用于显示3D坐标
                pred_positions_world=pred_positions,  # 原始空间
                sample_errors=sample_errors,
                sample=sample,
                output_dir=output_dir,
                projection_interface=projection_interface,
            )

    # 计算总体统计
    if all_errors:
        all_errors_np = np.array(all_errors)
        stats = {
            'num_samples': num_samples,
            'total_frames': len(all_errors),
            'mean_error_m': np.mean(all_errors_np),
            'std_error_m': np.std(all_errors_np),
            'median_error_m': np.median(all_errors_np),
            'max_error_m': np.max(all_errors_np),
            'min_error_m': np.min(all_errors_np),
            'percentile_90_m': np.percentile(all_errors_np, 90),
            'percentile_95_m': np.percentile(all_errors_np, 95),
            'mean_error_mm': np.mean(all_errors_np) * 1000,
            'std_error_mm': np.std(all_errors_np) * 1000,
            'mean_error_x_mm': np.mean(all_errors_per_axis['x']) * 1000,
            'mean_error_y_mm': np.mean(all_errors_per_axis['y']) * 1000,
            'mean_error_z_mm': np.mean(all_errors_per_axis['z']) * 1000,
        }

        print(f"\n{'='*60}")
        print("Overall Statistics (Projection/Reprojection Error)")
        print(f"{'='*60}")
        print(f"  Total samples: {stats['num_samples']}")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Mean error: {stats['mean_error_mm']:.2f} mm (std: {stats['std_error_mm']:.2f} mm)")
        print(f"  Median error: {stats['median_error_m']*1000:.2f} mm")
        print(f"  Max error: {stats['max_error_m']*1000:.2f} mm")
        print(f"  90th percentile: {stats['percentile_90_m']*1000:.2f} mm")
        print(f"  95th percentile: {stats['percentile_95_m']*1000:.2f} mm")
        print(f"  Per-axis mean error: X={stats['mean_error_x_mm']:.2f}mm, "
              f"Y={stats['mean_error_y_mm']:.2f}mm, Z={stats['mean_error_z_mm']:.2f}mm")
        print(f"{'='*60}\n")

        # 绘制误差分布图
        if visualize:
            _plot_error_statistics(all_errors_np, all_errors_per_axis, stats, output_dir)

        # 保存统计结果到文件
        stats_file = os.path.join(output_dir, 'projection_error_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Projection/Reprojection Error Statistics\n")
            f.write("="*60 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\nPer-Sample Results:\n")
            f.write("-"*60 + "\n")
            for result in sample_results:
                f.write(f"\nSample {result['sample_idx']}: {result['trail_name']}\n")
                f.write(f"  Instruction: {result['instruction'][:100]}...\n")
                f.write(f"  Mean Error: {result['mean_error']*1000:.2f} mm\n")
                f.write(f"  Max Error: {result['max_error']*1000:.2f} mm\n")
        print(f"Statistics saved to: {stats_file}")

        return {
            'stats': stats,
            'sample_results': sample_results,
            'all_errors': all_errors,
        }
    else:
        print("No valid samples processed.")
        return None


def _visualize_projection_test(
    sample_idx: int,
    rgb_image: torch.Tensor,
    rgb_sequence: torch.Tensor,
    heatmap_all: torch.Tensor,
    gt_positions_cube: torch.Tensor,
    gt_positions_world: torch.Tensor,
    pred_positions_world: torch.Tensor,
    sample_errors: list,
    sample: dict,
    output_dir: str,
    projection_interface: ProjectionInterface,
):
    """
    可视化投影测试结果

    Args:
        sample_idx: 样本索引
        rgb_image: 起始帧RGB图像 (num_views, 3, H, W)
        rgb_sequence: 未来帧RGB序列 (T, num_views, 3, H, W)
        heatmap_all: 所有帧的heatmap (T+1, num_views, H, W)
        gt_positions_cube: ground truth位置 - cube空间 (T+1, 3)，用于2D投影
        gt_positions_world: ground truth位置 - 原始空间 (T+1, 3)，用于显示
        pred_positions_world: 预测位置 - 原始空间 (T+1, 3)
        sample_errors: 每帧的误差详情
        sample: 原始样本数据
        output_dir: 输出目录
        projection_interface: 投影接口
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    num_frames = heatmap_all.shape[0]
    num_views = heatmap_all.shape[1]

    # 合并RGB图像: (T+1, num_views, 3, H, W)
    rgb_all = torch.cat([rgb_image.unsqueeze(0), rgb_sequence], dim=0)

    # 计算GT位置在图像上的投影（使用cube空间坐标，因为renderer在cube空间工作）
    gt_img_locs = projection_interface.project_pose_to_pixel(
        gt_positions_cube.unsqueeze(0).to(projection_interface.renderer_device)
    )  # (1, T+1, num_views, 2)
    gt_img_locs = gt_img_locs[0]  # (T+1, num_views, 2)

    # 创建可视化图: 行=时间步, 列=视角*2 (RGB+Heatmap)
    fig, axes = plt.subplots(num_frames, num_views * 2, figsize=(5 * num_views * 2, 4 * num_frames))

    if num_frames == 1:
        axes = axes.reshape(1, -1)

    for frame_idx in range(num_frames):
        error_info = sample_errors[frame_idx]

        for view_idx in range(num_views):
            # RGB图像
            ax_rgb = axes[frame_idx, view_idx * 2]
            rgb_frame = rgb_all[frame_idx, view_idx]  # (3, H, W)

            # 转换为 (H, W, 3), 范围 [0, 255] -> [0, 1]
            if rgb_frame.max() > 1:
                rgb_img = rgb_frame.permute(1, 2, 0).cpu().numpy() / 255.0
            else:
                rgb_img = ((rgb_frame.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)

            ax_rgb.imshow(rgb_img)

            # 先获取heatmap峰值位置（作为预测的2D位置）
            hm_frame = heatmap_all[frame_idx, view_idx].cpu().numpy()  # (H, W)
            peak_y, peak_x = np.unravel_index(np.argmax(hm_frame), hm_frame.shape)

            # 标记GT位置 (绿色圆圈)
            gt_x, gt_y = gt_img_locs[frame_idx, view_idx].cpu().numpy()
            ax_rgb.plot(gt_x, gt_y, 'go', markersize=12, markeredgecolor='white',
                       markeredgewidth=2, label='GT')

            # 标记预测位置 - 使用heatmap峰值 (红色叉)
            ax_rgb.plot(peak_x, peak_y, 'rx', markersize=12, markeredgewidth=2, label='Peak')

            # 连接GT和预测位置
            ax_rgb.plot([gt_x, peak_x], [gt_y, peak_y], 'b--', linewidth=1.5, alpha=0.7)

            ax_rgb.axis('off')
            if frame_idx == 0:
                ax_rgb.set_title(f'View {view_idx} - RGB', fontsize=10, fontweight='bold')
            ax_rgb.legend(loc='upper right', fontsize=8)

            # Heatmap
            ax_hm = axes[frame_idx, view_idx * 2 + 1]

            # 归一化heatmap以便显示
            hm_min, hm_max = hm_frame.min(), hm_frame.max()
            if hm_max > hm_min:
                hm_norm = (hm_frame - hm_min) / (hm_max - hm_min)
            else:
                hm_norm = hm_frame

            ax_hm.imshow(hm_norm, cmap='hot')

            # 标记heatmap峰值 (红色叉，与RGB图一致)
            ax_hm.plot(peak_x, peak_y, 'rx', markersize=12, markeredgewidth=2, label='Peak')

            # 标记GT位置 (绿色圆圈)
            ax_hm.plot(gt_x, gt_y, 'go', markersize=10, markeredgecolor='white',
                      markeredgewidth=1.5, label='GT')

            ax_hm.axis('off')
            if frame_idx == 0:
                ax_hm.set_title(f'View {view_idx} - Heatmap', fontsize=10, fontweight='bold')
            ax_hm.legend(loc='upper right', fontsize=8)

        # 在行末添加误差信息（包含3D坐标）
        gt_pos = gt_positions_world[frame_idx].cpu().numpy()
        pred_pos = pred_positions_world[frame_idx].cpu().numpy()
        error_text = (
            f"Frame {frame_idx}: Error = {error_info['error_distance']*1000:.2f}mm\n"
            f"GT: ({gt_pos[0]*1000:.1f}, {gt_pos[1]*1000:.1f}, {gt_pos[2]*1000:.1f})mm\n"
            f"Pred: ({pred_pos[0]*1000:.1f}, {pred_pos[1]*1000:.1f}, {pred_pos[2]*1000:.1f})mm"
        )
        fig.text(0.99, 1 - (frame_idx + 0.5) / num_frames, error_text,
                ha='right', va='center', fontsize=9,
                transform=fig.transFigure,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 添加总标题
    mean_error = np.mean([e['error_distance'] for e in sample_errors]) * 1000
    fig.suptitle(
        f'Sample {sample_idx}: {sample["trail_name"]} (Step {sample["start_step"]})\n'
        f'Instruction: {sample["instruction"][:60]}...\n'
        f'Mean 3D Error: {mean_error:.2f}mm | Green=GT, Red=Heatmap Peak',
        fontsize=12, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 0.95, 0.92])

    # 保存图像
    output_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_projection_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to: {output_path}")


def _plot_error_statistics(all_errors: np.ndarray, errors_per_axis: dict, stats: dict, output_dir: str):
    """
    绘制误差统计图
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 误差分布直方图
    ax1 = axes[0, 0]
    errors_mm = all_errors * 1000
    ax1.hist(errors_mm, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(stats['mean_error_mm'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean_error_mm"]:.2f}mm')
    ax1.axvline(stats['median_error_m']*1000, color='green', linestyle='--', linewidth=2,
                label=f'Median: {stats["median_error_m"]*1000:.2f}mm')
    ax1.set_xlabel('Error (mm)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('3D Position Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 每个轴的误差分布
    ax2 = axes[0, 1]
    x_errors = np.array(errors_per_axis['x']) * 1000
    y_errors = np.array(errors_per_axis['y']) * 1000
    z_errors = np.array(errors_per_axis['z']) * 1000

    bp = ax2.boxplot([x_errors, y_errors, z_errors], labels=['X', 'Y', 'Z'], patch_artist=True)
    colors = ['lightcoral', 'lightgreen', 'lightskyblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Error (mm)', fontsize=12)
    ax2.set_title('Per-Axis Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加均值标签
    means = [stats['mean_error_x_mm'], stats['mean_error_y_mm'], stats['mean_error_z_mm']]
    for i, mean in enumerate(means):
        ax2.annotate(f'{mean:.2f}mm', xy=(i+1, mean), xytext=(i+1.3, mean),
                    fontsize=10, fontweight='bold')

    # 3. 累积分布函数 (CDF)
    ax3 = axes[1, 0]
    sorted_errors = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax3.plot(sorted_errors, cdf, linewidth=2, color='steelblue')
    ax3.axhline(0.9, color='orange', linestyle='--', linewidth=1.5,
                label=f'90%: {stats["percentile_90_m"]*1000:.2f}mm')
    ax3.axhline(0.95, color='red', linestyle='--', linewidth=1.5,
                label=f'95%: {stats["percentile_95_m"]*1000:.2f}mm')
    ax3.set_xlabel('Error (mm)', fontsize=12)
    ax3.set_ylabel('Cumulative Probability', fontsize=12)
    ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 统计信息文本
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = (
        f"Projection/Reprojection Error Statistics\n"
        f"{'='*40}\n\n"
        f"Total samples: {stats['num_samples']}\n"
        f"Total frames: {stats['total_frames']}\n\n"
        f"Mean error: {stats['mean_error_mm']:.2f} mm\n"
        f"Std error: {stats['std_error_mm']:.2f} mm\n"
        f"Median error: {stats['median_error_m']*1000:.2f} mm\n"
        f"Min error: {stats['min_error_m']*1000:.2f} mm\n"
        f"Max error: {stats['max_error_m']*1000:.2f} mm\n\n"
        f"90th percentile: {stats['percentile_90_m']*1000:.2f} mm\n"
        f"95th percentile: {stats['percentile_95_m']*1000:.2f} mm\n\n"
        f"Per-axis mean error:\n"
        f"  X: {stats['mean_error_x_mm']:.2f} mm\n"
        f"  Y: {stats['mean_error_y_mm']:.2f} mm\n"
        f"  Z: {stats['mean_error_z_mm']:.2f} mm"
    )
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'error_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error statistics plot saved to: {output_path}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    import argparse
    matplotlib.use('Agg')  # 使用非交互式后端

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Test projection/reprojection error')
    parser.add_argument('--data_root', type=str, default="/DATA/disk0/lpy/data/Franka_data_3zed_5/push_T_5",
                        help='Data root directory')
    parser.add_argument('--output_dir', type=str, default="./projection_test_results",
                        help='Output directory for visualization results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'visualize'],
                        help='Mode: "test" for projection error test, "visualize" for basic visualization')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (use only 2 trails)')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation (note: augmentation does not affect error accuracy since GT poses are transformed together with point clouds)')
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 创建示例接口（用户需要替换为实际实现）
    projection_interface = ProjectionInterface()

    # 创建数据集
    # 对于投影误差测试，关闭数据增强以获得准确的误差测量
    use_augmentation = not args.no_augmentation

    dataset = RobotTrajectoryDataset(
        data_root=data_root,
        projection_interface=projection_interface,
        sequence_length=5,
        min_trail_length=10,
        debug=args.debug,
        augmentation=use_augmentation,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")

    if args.mode == 'test':
        # ============================================
        # 模式1: 投影/反投影误差测试
        # ============================================
        print("\n" + "="*60)
        print("Running Projection/Reprojection Error Test")
        print("="*60)

        results = test_projection_reprojection_error(
            dataset=dataset,
            projection_interface=projection_interface,
            num_samples=args.num_samples,
            output_dir=output_dir,
            visualize=True,
        )

        if results:
            print("\nTest completed successfully!")
            print(f"Results saved to: {output_dir}")
        else:
            print("\nTest failed - no valid samples processed.")

    elif args.mode == 'visualize':
        # ============================================
        # 模式2: 基本可视化（原始功能）
        # ============================================
        if len(dataset) > 0:
            # 采样多个样本进行可视化
            num_samples_to_visualize = min(args.num_samples, len(dataset))

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
                        # 处理 [0, 255] 或 [-1, 1] 范围
                        if rgb_img.max() > 1:
                            rgb_img = rgb_img / 255.0
                        else:
                            rgb_img = np.clip((rgb_img + 1) / 2, 0, 1)

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

                print(f"Visualization saved to: {output_path}")

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

                print(f"Info saved to: {info_path}")

            print(f"\n{'='*60}")
            print(f"All visualizations saved to: {output_dir}")
            print(f"{'='*60}")