'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/models/rvt_agent.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''

import pprint
import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
import RLBench.utils.peract_utils_rlbench as rlbench_utils
import GemBench.utils.peract_utils_gembench as gembench_utils
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con, aug_utils
from yarr.agents.agent import ActResult
from PIL import Image, ImageDraw
import torch
import numpy as np
import os
import Real.utils.peract_utils as real_utils


def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    # Ensure data dimensions are correct
    assert points.shape[1] == 3 
    assert colors.shape[1] == 3
    
    # 将颜色信息统一到0-255
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)
    
    # Add keypoint (optional)
    if keypoint is not None:
        points = np.vstack([points, keypoint]) # 将目标点和点云拼接
        colors = np.vstack([colors, np.array([255, 0, 0])])  # 让目标点变成红色

    # Write to PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, clr in zip(points, colors): # 手动保存点云信息 {x, y, z, r, g, b}
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def visualize_images(
    color_tensor: torch.Tensor,  #  (3, 3, 224, 224)  num_img channel H, W
    gray_tensor: torch.Tensor,   #  (224, 224, 3)  H W num_img
    save_dir: str = "/opt/tiger/3D_OpenVLA/3d_policy/RVT/rvt_our/debug"
) -> None:
    """
    可视化三个视角的图像：如BridgeVLA中的三个视角图像
    1. original_0.png, original_1.png, original_2.png   (original image)
    2. gray_0.png, gray_1.png, gray_2.png              (gray image)
    3. overlay_0.png, overlay_1.png, overlay_2.png     (transparent image + annotation)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    color_imgs = color_tensor.cpu().numpy().transpose(0, 2, 3, 1) #  (num_img, H, W, channel)
    gray_imgs = gray_tensor.cpu().numpy().transpose(2, 0, 1)     #  (num_img, H, W)
    
    for i in range(3):
        # 原始图像
        original_img = np.clip(color_imgs[i], 0, 1) * 255
        original_img = original_img.astype(np.uint8)
        Image.fromarray(original_img).save(os.path.join(save_dir, f"original_{i}.png"))
        
        # 热力图
        gray_img = np.clip(gray_imgs[i], 0, 1) * 255
        gray_img = gray_img.astype(np.uint8)
        Image.fromarray(gray_img, mode="L").save(os.path.join(save_dir, f"gray_{i}.png"))
        
        # RGBA 透明图像    (H, W, 4) 4 -> RGBA
        rgba = np.zeros((*original_img.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = original_img  # 前三个维度都直接用original img
        rgba[..., 3] = 77             # 最后一个维度的透明度是77
        
        # 带热力图的半透明图
        overlay_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        
        # unravel_index 将一维索引转换为多维索引
        # argmax()返回热力图最大值的一维索引，转化成为根据gray_img形状的二维索引
        max_pos = np.unravel_index(gray_imgs[i].argmax(), gray_imgs[i].shape)
        x = max_pos[1]  # 最大值的x坐标
        y = max_pos[0]  # 最大值的y坐标
        
        # 画一个半径为5的圆，圆心是最大值的位置，颜色是红色
        point_radius = 5
        draw.ellipse( # 画红点
            [x-point_radius, y-point_radius, x+point_radius, y+point_radius],
            fill=(255, 0, 0, 255)  
        )
        
        overlay_img.save(os.path.join(save_dir, f"overlay_{i}.png"))


def apply_channel_wise_softmax(gray_tensor):
    """
    对于每一个通道的图像进行softmax，并将其恢复到原来的形状
    """
    # Convert to PyTorch tensor (if not already)
    if not isinstance(gray_tensor, torch.Tensor):
        gray_tensor = torch.tensor(gray_tensor, dtype=torch.float32)
    
    # Separate each channel (C, H, W)
    channels = gray_tensor.permute(2, 0, 1)
    
    # Apply softmax to each channel and flatten
    softmax_channels = []
    for c in range(channels.shape[0]):
        channel = channels[c].flatten() # 将某一个通道的灰度图展平
        softmax_channel = torch.softmax(channel, dim=0) # 对展平后的灰度图进行softmax
        softmax_channels.append(softmax_channel.view_as(channels[c])) # 将softmax后的结果恢复到原来的形状
    
    # Merge channels and restore original shape (H, W, C)
    return torch.stack(softmax_channels, dim=2)


def eval_con(gt, pred):
    '''
    评估连续值的误差
    '''
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    # gt shape: (bs, 3) x y z
    # pred shape: (bs, 3) x y z
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    评估连续分类的误差
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: 离散bin的数量
    :param res: 每个bin的分辨率
    :param symmetry: 对称性，2是180度对称，4是90度对称
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt # 判断gt的维度是否为0或1
    assert num_bin % symmetry == 0, (num_bin, symmetry) # 判断num_bin是否能被symmetry整除
    gt = torch.tensor(gt) # 将gt转换为tensor
    pred = torch.tensor(pred) # 将pred转换为tensor
    num_bin //= symmetry # 将num_bin除以symmetry
    pred %= num_bin # 将pred对num_bin取模
    gt %= num_bin # 将gt对num_bin取模
    dist = torch.abs(pred - gt) # 计算pred和gt的绝对差值
    dist = torch.min(dist, num_bin - dist) # 取得dist和num_bin-dist中的最小值
    dist_con = dist.float() * res # 将dist转换为float，并乘以res
    return {"avg err": dist_con.mean()} # 返回角度误差


def eval_cls(gt, pred):
    """
    评估分类能力，判断gt和pred是否相等
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    """
    评估模型预测的所有指标，包括位移误差，欧拉角误差，夹爪状态误差，碰撞误差
    wpt: waypoint 真实目标点
    pred_wpt: 模型预测的目标点
    action_rot: gt 旋转四元数
    pred_rot_quat: 模型预测的旋转四元数
    action_grip_one_hot: (bs, 2) [0,1] 或 [1,0] 表示是否抓取
    grip_q: 抓取 Q 值： 即当前观察下夹爪开闭的价值
    action_collision_one_hot: (bs, 2) [0,1] 或 [1,0] 表示是否忽略碰撞
    collision_q: 碰撞 Q 值： 即当前观察下碰撞的价值
    """
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append( # 位移误差
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True) # 将四元数转化成欧拉角
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)
        # 评估 x y z的角度误差
        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    # 如果agent没有loss_log属性，或者reset_log为True，则创建一个空的loss_log字典
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}
    
    # 将loss_log中的每个key对应的值添加到agent.loss_log中
    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    '''返回agent的log'''
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out) # 打印loss_log
    return out


class RVTAgent:
    def __init__(
        self,
        network: nn.Module, # 网络模块
        num_rotation_classes: int, # 旋转分类的类别，也就是bins
        stage_two: bool, # 是否使用两阶段训练
        move_pc_in_bound: bool, # 是否将点云移动到边界内
        lr: float = 0.0001, # 学习率
        image_resolution: list = None, # 图像分辨率
        lambda_weight_l2: float = 0.0, # L2正则化系数
        transform_augmentation: bool = True, # 是否使用数据增强
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1], # 数据增强的xyz范围
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0], # 数据增强的rpy范围
        place_with_mean: bool = True, # 是否使用均值放置
        transform_augmentation_rot_resolution: int = 5, # 数据增强的旋转分辨率
        optimizer_type: str = "lamb", # 优化器类型
        gt_hm_sigma: float = 1.5, # 高斯热力图的sigma
        img_aug: bool = False, # 是否使用图像增强
        add_rgc_loss: bool = False, # 是否添加RGC损失
        scene_bounds: list = rlbench_utils.SCENE_BOUNDS, # 场景边界
        cameras: list = rlbench_utils.CAMERAS, # 相机列表
        rot_ver: int = 0, # 旋转版本
        rot_x_y_aug: int = 2, # 旋转x和y的增强
        log_dir="", # 日志目录
    ):
        self._network = network # agent的主要网络模块
        self._num_rotation_classes = num_rotation_classes # 旋转分类的种类 72 种，每5度一种
        self._rotation_resolution = 360 / self._num_rotation_classes # 旋转分辨率 5度
        self._lr = lr # 学习率
        self._image_resolution = image_resolution # 图像分辨率
        self._lambda_weight_l2 = lambda_weight_l2 # L2正则化系数
        self._transform_augmentation = transform_augmentation # 是否使用数据增强
        self._place_with_mean = place_with_mean # 是否使用均值放置
        self._transform_augmentation_xyz = torch.from_numpy( # 数据增强的xyz范围
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy # 数据增强的rpy范围
        self._transform_augmentation_rot_resolution = ( # 数据增强的旋转分辨率
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type # 优化器类型
        self.gt_hm_sigma = gt_hm_sigma # 高斯热力图的sigma
        self.img_aug = img_aug # 是否使用图像增强
        self.add_rgc_loss = add_rgc_loss # 是否添加RGC损失
        self.stage_two = stage_two # 是否使用两阶段训练
        self.log_dir = log_dir # 日志目录
        self.scene_bounds = scene_bounds # 场景边界
        self.cameras = cameras # 相机列表

        print("Cameras:",self.cameras)
        self.move_pc_in_bound = move_pc_in_bound # 是否将点云放在bound内
        self.rot_ver = rot_ver # 旋转版本
        self.rot_x_y_aug = rot_x_y_aug # 旋转x和y的增强

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none") # 交叉熵损失
        if isinstance(self._network, DistributedDataParallel): # 如果网络是分布式训练的，则使用module
            self._net_mod = self._network.module
        else: # 否则使用网络本身
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3 # 72 * 3 = 216 旋转bins

    def build(self, training: bool, device: torch.device = None):
        '''创建一个optimizer在 self._optimizer'''
        self._training = training # 是否训练
        self._device = device
        params_to_optimize = filter(lambda p: p.requires_grad, self._network.parameters())

        self._optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self._lr,
            weight_decay=self._lambda_weight_l2,
        )


    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """返回 R P Y Gripper Collision 的one hot编码

        :param batch_size: int
        :param action_rot: np.array 形状为 (bs, 4) 是 x y z w 的四元数形式
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        # 先创建全0的tensor 形状为 (bs, num_rotation_classes)
        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # 用one hot填充全0tensor
        for b in range(bs):
            gt_rot = action_rot[b]
            # 将四元数化做欧拉角，并且根据欧拉角的角度分配对应bin的索引
            # gt_rot : [index_rx, index_ry, index_rz]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            # 填充one hot
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )


    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        从MVT的输出中获取夹爪位姿，夹爪状态，碰撞状态：返回 (trans_q, rot_q, gripper_q, collision_q)
        :param out: Multiview transformer (MVT) 的输出
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: 返回元组 (trans_q, rot_q, gripper_q, collision_q)
        """
        bs, nc, h, w = dims # nc 可以看作通道数量，但实际上是multiview的视角数量
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2) # 将trans的形状从(bs, nc, h, w) 转换为 (bs, h*w, nc)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            if self.stage_two:
                out = out["mvt2"] # 第二阶段MVT的输出
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2) # bs, h*w, nc
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2) # bs h*w 2*nc
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot] # feature 的前216个特征 Shape (bs, 216)
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2] # [216:218] Shape (bs, 2)
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[:, self.num_all_rot + 2 : self.num_all_rot + 4] # [218:220] Shape (bs, 2)
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1) # Shape (bs, 72*3)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2] # Shape (bs, 2)
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:] # Shape (bs, 2)
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts



    def update(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
    ) -> dict:
        """
        一次完整的训练过程
        :param replay_sample: 训练样本
        :param backprop: 是否进行反向传播
        :param reset_log: 是否重置日志
        :return: 返回损失字典
        1. 获取训练样本
        2. 对Observation进行预处理
        3. 进行数据增强
        4. 进行坐标系变换
        5. 进行动作编码
        6. 进行MVT的forward
        7. 计算损失 + 反向传播
        """
        # replay_sample 格式为 {
        #     "obs": (b, 3, 128, 128, 3),
        #     "pcd": (b, 3, 1024),
        #     "lang_goal": (b, 1024),
        #     "rot_grip_action_indicies": (b, 1, 4),
        #     "ignore_collisions": (b, 1, 1),
        #     "gripper_pose": (b, 1, 7),
        #     "tasks": (b, 1)
        # }
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4) # 离散的 r p y 在72个bins中的索引 + 夹爪状态
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        
        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7) xyz + quat
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4) 四元数 x y z w格式
        action_grip = action_rot_grip[:, -1]  # (b,) 夹爪状态
        tasks = replay_sample["tasks"]
        return_out = {}
        # 对观测数据进行预处理，获得obs和点云
        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd, ) # 获取点云和图像
            # 对点云 + 动作进行数据增强，返回增强后的点云和动作
            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            # 对夹爪旋转四元数进行归一化，并保证w为正
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)  
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con] # 夹爪位姿的xyz

            wpt_local = []
            rev_trans = []
            #! 将点云，路径点标准化在一个立方体空间中
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0)) # 标准化之后的waypoint
                rev_trans.append(b) # 逆变换函数列表

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            # 将点云标准化在一个立方体空间中
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None
        #! 对旋转，夹爪状态，碰撞进行one hot编码
        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )
        # 添加一定的数据增强，旋转x和y的增强
        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None, # 训练的时候采用归一化到立方体内的waypoint
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        # 获取模型预测的夹爪位姿，夹爪状态，碰撞状态
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )
        # 获取ground truth的夹爪位姿
        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )


        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()    # Soft-label cross-entropy loss. The target has the same shape as the input and is no longer one-hot encoded, but represented by class probabilities.
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                # 计算三个角度loss
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                # 计算夹爪状态loss
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                # 计算碰撞状态loss
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )


            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward() 
            self._optimizer.step()


            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)


        return return_out



    def update_gembench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
    ) -> dict:
        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"]  # (b, 8)  
        

        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3) 
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4) 

        action_grip = action_gripper_pose[:, -1].int()   # (b,)
        return_out = {}

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )
            import open3d as o3d
            def vis_pcd(pc, rgb,save_path):

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)  
                pcd.colors = o3d.utility.Vector3dVector(rgb) 
                o3d.io.write_point_cloud(save_path, pcd)
                # o3d.visualization.draw_geometries([pcd])
            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
            
            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )

        loss_log = {}
        if backprop:
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()  
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward()
            self._optimizer.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out
    
    def update_real(
        self,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
        cameras=["3rd","wrist"],
    ):
        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int() # (b, 1)
        action_gripper_pose = replay_sample["gripper_pose"] # (b, 8) quat在数据集里面是 xyzw
        
        action_trans_con = action_gripper_pose[:, 0:3] # (b, 3)
        action_rot = action_gripper_pose[:, 3:7] # (b, 4) xyzw
        action_grip = action_gripper_pose[:, -1].int() # (b,)
        return_out = {}
        # OBS : [rgb, pcd] PCD : [pcd]
        obs, pcd = real_utils._preprocess_inputs_real(replay_sample, cameras)
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs, pcd
            )
            
            pc = pc.float()
            img_feat = img_feat.float()
            
            # 数据增强
            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
            
            # 对四元数进行标准化
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot
                
            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            # 将点云和wpt放在一个cube里面
            wpt = [x[:3] for x in action_trans_con]
            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc, 
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)
            wpt_local = torch.cat(wpt_local, axis=0) # shape [b, 3]
            
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]
            
            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size
            
            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0
                
            dyn_cam_info = None
        
        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )
        
        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=img_aug,
            lang_emb = None,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["language_goal"]
        )
        mvt1_img = out['mvt1_ori_img'][0,:,3:6]
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )
        
        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )
        loss_log = {}
        if backprop:
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()
                
                total_loss = (
                    trans_loss
                    + rot_loss_x
                    + rot_loss_y
                    + rot_loss_z
                    + grip_loss
                    + collision_loss
                )
                self._optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self._optimizer.step()
                
                loss_log = {
                    "total_loss": total_loss.item(),
                    "trans_loss": trans_loss.item(),
                    "rot_loss_x": rot_loss_x.item(),
                    "rot_loss_y": rot_loss_y.item(),
                    "rot_loss_z": rot_loss_z.item(),
                    "grip_loss": grip_loss.item(),
                    "collision_loss": collision_loss.item(),
                    "lr": self._optimizer.param_groups[0]["lr"],
                }
                manage_loss_log(self, loss_log, reset_log=reset_log)
                return_out.update(loss_log)
        if not backprop and eval_log:
            with torch.no_grad():
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if self.add_rgc_loss:
                    
                    rot_loss_x = self._cross_entropy_loss(
                        rot_q[
                            :,
                            0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                        ],
                        action_rot_x_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_y = self._cross_entropy_loss(
                        rot_q[
                            :,
                            1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                        ],
                        action_rot_y_one_hot.argmax(-1),
                    ).mean()
                
                    rot_loss_z = self._cross_entropy_loss(
                        rot_q[
                            :,
                            2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                        ],
                        action_rot_z_one_hot.argmax(-1),
                    ).mean()
                    
                    grip_loss = self._cross_entropy_loss(
                        grip_q,
                        action_grip_one_hot.argmax(-1),
                    ).mean()

                    collision_loss = self._cross_entropy_loss(
                        collision_q, action_collision_one_hot.argmax(-1)
                    ).mean()

                total_loss = (
                    trans_loss
                    + rot_loss_x
                    + rot_loss_y
                    + rot_loss_z
                    + grip_loss
                    + collision_loss
                )
                
                eval_loss_log = {
                    "eval_total_loss": total_loss.item(),
                    "eval_trans_loss": trans_loss.item(),
                    "eval_rot_loss_x": rot_loss_x.item(),
                    "eval_rot_loss_y": rot_loss_y.item(),
                    "eval_rot_loss_z": rot_loss_z.item(),
                    "eval_grip_loss": grip_loss.item(),
                    "eval_collision_loss": collision_loss.item(),
                }
                
                return_out.update(eval_loss_log)

        return return_out
    
        

    @torch.no_grad()
    def act(
        self, step: int, observation: dict,visualize=False,visualize_save_dir="", return_gembench_action=False,
    ) -> ActResult:
        """推理函数，将Observation输入到MVT中，获得预测的夹爪位姿，夹爪状态，碰撞状态"""
        language_goal =observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_ori = pc[0].clone()
        img_feat_ori=img_feat[0].clone()
        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,  # no img augmentation while acting
            language_goal=language_goal,
        )
        # 获取模型预测的夹爪位姿，夹爪状态，碰撞状态
        if visualize:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )            
            
        # 动作decode
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )
        # 进行动作可视化
        if visualize:
            print("Visualizing")
            save_dir=visualize_save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir=os.path.join(save_dir,f"step{str(step)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            mvt1_img=out["mvt1_ori_img"][0,:,3:6]
            mvt2_img=out["mvt2_ori_img"][0,:,3:6]
            q_trans_1=q_trans[0,:,:3].clone().view(224,224,3)
            q_trans_2=q_trans[0,:,3:6].clone().view(224,224,3)
            q_trans_1=apply_channel_wise_softmax(q_trans_1)*100
            q_trans_2=apply_channel_wise_softmax(q_trans_2)*100
            visualize_images(mvt1_img,q_trans_1,save_dir=os.path.join(save_dir,"mvt1"))
            visualize_images(mvt2_img,q_trans_2,save_dir=os.path.join(save_dir,"mvt2"))
            save_point_cloud_with_color(os.path.join(save_dir,"point_cloud.ply"), pc_ori.cpu().numpy(), img_feat_ori.cpu().numpy(), pred_wpt[0].cpu().numpy())
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
                # [1.0],  # debug!!!!!!
            )
        )
        # 组装动作成为 [translation + quat + grip_state]
        if return_gembench_action:
            continuous_action = np.concatenate(
                    [
                        pred_wpt[0].cpu().numpy(),
                        pred_rot_quat[0],
                        pred_grip[0].cpu().numpy(),
                    ], -1
                )
            return continuous_action
        else:
            return ActResult(continuous_action)



    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        """
        进行动作解码：
        :param out: MVT的输出
        :param rot_q: 旋转四元数
        :param grip_q: 夹爪状态
        :param collision_q: 碰撞状态
        :param y_q: 
        :param rev_trans: 逆变换函数列表
        :param dyn_cam_info: 动态相机信息
        :return: 预测的夹爪位姿，夹爪状态，碰撞状态
        """
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        # 获取预测的夹爪位姿
        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )
        # 将预测的夹爪位姿逆变换回原始坐标系
        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        # 获取预测三个角度旋转的index，并转换为四元数
        pred_rot = torch.cat(
            (
                # 获取x轴旋转的index
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                # 获取y轴旋转的index
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                # 获取z轴旋转的index
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        # 将index转换为四元数
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll


    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        """
        获取ground truth的夹爪位置
        :param wpt_local: 归一化到立方体内的waypoint
        :param pts: 点云
        :param out: MVT的输出
        :param dyn_cam_info: 动态相机信息
        :param dims: 输入的维度
        :return: 预测的夹爪位姿
        """
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans



    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
