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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

finetune_path = "/mnt/data/workspace/BridgeVLA_dev"
sys.path.append(finetune_path)

import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con_shared
from peract_colab.rlbench.utils import get_stored_demo
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

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

def _norm_rgb(x):
    if isinstance(x, np.ndarray):
        # 处理负步长问题
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
    return (x.float() / 255.0) * 2.0 - 1.0


def _process_single_trail(args):
    data_root, trail_idx, step_interval, sequence_length, skip_step = args

    # 注意：不要在此函数内用 CUDA
    demo = get_stored_demo(data_root, index=trail_idx)
    num_steps = len(demo)

    var_desc_file = os.path.join(data_root, f"episode{trail_idx}", "variation_descriptions.pkl")
    with open(var_desc_file, "rb") as f:
        descs = pickle.load(f)
    instruction = descs[0]

    max_start_step = num_steps - 1
    samples = []

    for start_step in range(0, max_start_step, skip_step):
        future_steps = []
        for i in range(1, sequence_length + 1):
            fs = start_step + i * step_interval
            fs = fs if fs < num_steps else num_steps - 1
            future_steps.append(fs)

        samples.append({
            "trail_info": {
                "idx": trail_idx,
                "trail_name": f"episode{trail_idx}",
                "instruction": instruction,
                "num_steps": num_steps,
            },
            "start_step": start_step,
            "future_steps": future_steps
        })

    return samples


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

                # 使用jetormap显示heatmap
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
                 min_trail_length: int = 15,
                 image_size: Tuple[int, int] = (256, 256),
                 sigma: float = 1.5,
                 augmentation: bool = True,
                 mode="train",
                 scene_bounds: List[float] = [-0.3,-0.5,0.6,0.7,0.5,1.6],
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
        self.data_root = os.path.join(self.data_root, "all_variations/episodes")
        print("Data Root:", self.data_root)
        self.cameras = ["front", "left_shoulder", "right_shoulder", "wrist"]

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

        self.skip_step = 1

        # 扫描所有轨迹数据
        self.valid_samples = self._generate_valid_samples()

        # ============================================================
        # 预加载所有demos到内存（消除IO瓶颈）
        # ============================================================
        print("\n" + "="*70)
        print("PRELOADING ALL DEMOS INTO MEMORY")
        print("="*70)
        self._demo_cache = {}
        self._preload_all_demos()

    def _generate_valid_samples(self):

        idx = sorted(int(os.path.basename(p).replace("episode", ""))
                     for p in glob.glob(os.path.join(self.data_root, "episode*")))
        if not idx:
            return []

        start = idx[0] if self.trail_start is None else max(self.trail_start, idx[0])
        end   = idx[-1] if self.trail_end is None else min(self.trail_end, idx[-1])
        valid_idx = list(range(start, end + 1))

        args_list = [
            (self.data_root, trail_idx, self.step_interval, self.sequence_length, self.skip_step)
            for trail_idx in valid_idx
        ]

        valid_samples = []

        # --- 使用 spawn 方式的并行池 ---
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_process_single_trail, args): args for args in args_list}

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing episodes"):
                samples = fut.result()
                valid_samples.extend(samples)

        print(f"Found {len(valid_idx)} trails, {len(valid_samples)} valid samples")
        return valid_samples

    def _preload_all_demos(self):
        """
        在初始化时预加载所有demos到内存

        优势：
        - 第1个epoch就很快（不需要等待加载）
        - 完全消除训练时的IO瓶颈
        - 避免重复读取PNG文件和计算点云

        内存占用：
        - 每个demo约100-300MB（取决于steps数量和图像分辨率）
        - 100个episodes约10-30GB
        - 1000个episodes约100-300GB（适合多任务训练）
        """
        from peract_colab.rlbench.utils import get_stored_demo
        import time

        # 获取所有需要加载的trail_idx
        unique_trail_indices = set()
        for sample in self.valid_samples:
            unique_trail_indices.add(sample['trail_info']['idx'])

        unique_trail_indices = sorted(unique_trail_indices)
        total_demos = len(unique_trail_indices)

        print(f"Preloading {total_demos} demos...")
        print(f"Estimated memory usage: ~{total_demos * 0.2:.1f} - {total_demos * 0.3:.1f} GB")
        print("This may take a few minutes but will significantly speed up training...")

        start_time = time.time()

        # 使用tqdm显示进度
        for trail_idx in tqdm(unique_trail_indices, desc="Loading demos"):
            try:
                demo = get_stored_demo(self.data_root, index=trail_idx)
                self._demo_cache[trail_idx] = demo
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to load demo {trail_idx}: {e}")
                # 继续加载其他demos
                continue

        elapsed_time = time.time() - start_time

        print(f"\n✅ Preloading complete!")
        print(f"   Loaded: {len(self._demo_cache)}/{total_demos} demos")
        print(f"   Time taken: {elapsed_time:.1f} seconds")
        print(f"   Average: {elapsed_time/total_demos:.2f} seconds per demo")
        print("="*70 + "\n")

    def _extract_obs(self, obs: Observation, channels_last: bool = False):

        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        obs_dict = {
            k: v for k, v in obs_dict.items()
            if any(kw in k for kw in ["rgb", "point_cloud"])
        }
        
        for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
            obs_dict[k] = v.astype(np.float32)

        pcd = [obs_dict[f"{cam}_point_cloud"] for cam in self.cameras]
        rgb = [obs_dict[f"{cam}_rgb"] for cam in self.cameras]

        return pcd, rgb

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


        # 处理每一帧的3个相机数据
        merged_pcd_list = []
        merged_feat_list = []

        for frame_idx in range(num_frames):

            # 获取这一帧的所有相机的点云和特征
            frame_pcds = pcd_list[frame_idx]  # [pcd_cam1, pcd_cam2, pcd_cam3]
            frame_feats = feat_list[frame_idx]  # [feat_cam1, feat_cam2, feat_cam3]

            # 归一化RGB特征
            frame_feats_norm = [_norm_rgb(feat) for feat in frame_feats]

            all_pcds = []
            all_feats = []

            for pcd, feat in zip(frame_pcds, frame_feats_norm):
                # flatten point cloud
                pcd_flat = torch.as_tensor(
                    np.ascontiguousarray(pcd), dtype=torch.float32
                ).view(-1, 3)

                # flatten RGB features & 归一化到 [0,1]
                feat_flat = ((feat.view(-1, 3) + 1) / 2).float()

                all_pcds.append(pcd_flat)
                all_feats.append(feat_flat)

            # 根据配置决定是否合并
            if self.use_merged_pointcloud:
                merged_pcd = torch.cat(all_pcds, dim=0)
                merged_feat = torch.cat(all_feats, dim=0)
            else:
                # 只使用第一个相机的数据（front）
                merged_pcd = all_pcds[0]
                merged_feat = all_feats[0]

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
                all_poses_tensor = torch.from_numpy(np.array(all_poses)).float()

                # 应用共享增强
                perturbed_poses, pc_batch = apply_se3_aug_con_shared(
                    pcd=pc_batch,
                    action_gripper_pose=all_poses_tensor,  # 传入xyzw格式
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                ) # bs,7  (pos, xyzw)

                # 分解回列表
                pc_list = [pc_batch[i] for i in range(num_frames)]
                action_trans_con = perturbed_poses[:, :3]
                action_rot_xyzw=perturbed_poses[:, 3:]
            else:
                # 没有数据增强时，直接使用原始poses
                action_trans_con = torch.from_numpy(np.array(all_poses)).float()[:, :3]
                action_rot_xyzw = torch.from_numpy(np.array(all_poses)).float()[:, 3:]  # [x,y,z,w]

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

        return final_pc_list, processed_feat_list, wpt_local,action_rot_xyzw, rev_trans
        

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
        # print(sample['trail_info'], sample['start_step'], sample['future_steps'])

        # 使用预加载的demo（直接从内存读取，无IO开销）
        demo = self._demo_cache[trail_info["idx"]]

        # 1. 加载起始step的数据

        start_pcd, start_rgb = self._extract_obs(demo[start_step])
        # 下一时刻的动作
        start_pose = demo[start_step].gripper_pose
        start_gripper_state = demo[start_step].gripper_open

        if start_pose is None or start_pcd is None:
            # 如果数据加载失败，抛出错误而不是递归调用
            raise ValueError(f"Failed to load data for sample {idx}. Start pose: {start_pose is not None}, Start pcd: {start_pcd is not None}")

        # 2. 提取所有future_steps对应的future_poses和future_data
        future_poses = []
        future_pcds = []
        future_rgbs = []
        future_gripper_states = []

        for future_step in future_steps:

            future_pcd, future_rgb = self._extract_obs(demo[future_step])
            future_pose = demo[future_step].gripper_pose

            future_poses.append(future_pose)
            future_pcds.append(future_pcd)
            future_rgbs.append(future_rgb)

            gripper_state = demo[future_step].gripper_open
            future_gripper_states.append(gripper_state)

        # 3. 拼接start和future数据，一起送入preprocess进行处理
        all_poses = [start_pose] + future_poses
        all_pcds = [start_pcd] + future_pcds
        all_rgbs = [start_rgb] + future_rgbs

        # 使用新的preprocess函数处理序列（添加trail_info参数以加载外参）
        processed_pcd_list, processed_rgb_list, processed_pos, processed_rot_xyzw, rev_trans = self.preprocess(
            all_pcds, all_rgbs, all_poses, trail_info
        )# 同时返回rotation和gripper  

        # 分离处理后的数据
        processed_start_pcd = processed_pcd_list[0]
        processed_start_rgb = processed_rgb_list[0]
        processed_future_pcds = processed_pcd_list[1:]
        processed_future_rgbs = processed_rgb_list[1:]

        processed_poses=torch.cat((processed_pos, processed_rot_xyzw), dim=1) # num,7
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

    import debugpy
    debugpy.listen(("0.0.0.0", 5680)) 
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    # 使用示例
    data_root = "/mnt/workspace/cyx/datasets/RLBench/data/train/reach_and_drag"
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
        trail_end=2,
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