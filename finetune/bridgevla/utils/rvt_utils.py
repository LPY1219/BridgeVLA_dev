# Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/utils/rvt_utils.py
import pdb
import argparse
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from bridgevla.models.peract_official import PreprocessAgent2

def get_pc_img_feat(obs, pcd, bounds=None):
    """
    preprocess the data in the peract to our framework
    """
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )

    img_feat = (img_feat + 1) / 2

    # x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # inv_pnt = (
    #     (pc[:, :, 0] < x_min)
    #     | (pc[:, :, 0] > x_max)
    #     | (pc[:, :, 1] < y_min)
    #     | (pc[:, :, 1] > y_max)
    #     | (pc[:, :, 2] < z_min)
    #     | (pc[:, :, 2] > z_max)
    # )

    # # TODO: move from a list to a better batched version
    # pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    # img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]

    return pc, img_feat


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


class TensorboardManager:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            if "image" in k:
                for i, x in enumerate(v):
                    self.writer.add_image(f"{split}_{step}", x, i)
            elif "hist" in k:
                if isinstance(v, list):
                    self.writer.add_histogram(k, v, step)
                elif isinstance(v, dict):
                    hist_id = {}
                    for i, idx in enumerate(sorted(v.keys())):
                        self.writer.add_histogram(f"{split}_{k}_{step}", v[idx], i)
                        hist_id[i] = idx
                    self.writer.add_text(f"{split}_{k}_{step}_id", f"{hist_id}")
                else:
                    assert False
            else:
                self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
         "--tasks", type=str, nargs="+", default=["all"]
    )
    parser.add_argument("--model-folder", type=str,default="")
    parser.add_argument("--eval-datafolder", type=str,default="")
    parser.add_argument("--visualize_root_dir", type=str,default="")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--ground-truth", action="store_true", default=False)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--mvt_cfg_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-name", type=str, default="test/1")
    parser.add_argument("--model-name", type=str, default="model_80.pth")
    parser.add_argument("--use-input-place-with-mean", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualize", action="store_true",default=False)    
    return parser





def load_agent(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network

        if isinstance(model, DDP):
            model = model.module

        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in mvt1. "
                    "Be cautious if you are using a two stage network."
                )
                model.mvt1.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)
    return epoch



RLBENCH_TASKS = [
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_color_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]


COLOSSEUM_TASKS = [
    "basketball_in_hoop",
    "close_box",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "meat_on_grill",
    "move_hanger",
    "wipe_desk",
    "open_drawer",
    "slide_block_to_target",
    "reach_and_drag",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "insert_onto_square_peg",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula",
    "close_laptop_lid",
    "stack_cups",
]



# ---------- 基础工具 ----------

def quat_to_R_torch(q):
    """
    q: (B,4) quaternion (x,y,z,w)
    return: (B,3,3)
    """
    q = torch.as_tensor(q)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    x, y, z, w = q.unbind(-1)

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack([
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def R_to_quat_torch(R):
    """
    R: (B,3,3)
    return: (B,4) quaternion (x,y,z,w) 归一化
    """
    R = torch.as_tensor(R)
    w = torch.sqrt(torch.clamp(1 + R[:,0,0] + R[:,1,1] + R[:,2,2], min=0)) / 2
    x = torch.sqrt(torch.clamp(1 + R[:,0,0] - R[:,1,1] - R[:,2,2], min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - R[:,0,0] + R[:,1,1] - R[:,2,2], min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - R[:,0,0] - R[:,1,1] + R[:,2,2], min=0)) / 2
    x = x.copysign(R[:,2,1] - R[:,1,2])
    y = y.copysign(R[:,0,2] - R[:,2,0])
    z = z.copysign(R[:,1,0] - R[:,0,1])
    q = torch.stack([x, y, z, w], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    return q

def pose_apply_torch(points_local, poses_base):
    """
    points_local: (N,3) - 点在末端执行器局部坐标系中的坐标
    poses_base:   (B,7) - 末端执行器在基坐标系中的位姿 [tx,ty,tz, qx,qy,qz,qw]
    return:       (B,N,3) - 点在基坐标系中的坐标
    """
    P = torch.as_tensor(points_local)           # (N,3)
    poses = torch.as_tensor(poses_base)         # (B,7)
    t = poses[:, :3]                                                 # (B,3)
    q = poses[:, 3:]                                                 # (B,4)
    R = quat_to_R_torch(q)                                           # (B,3,3)


    # 确保 P 的形状是 (B, N, 3)
    P_ = P.unsqueeze(0).repeat(R.size(0), 1, 1)  # 复制 P 以匹配批量大小

    # 进行批量矩阵乘法
    pts_rot = torch.bmm(R, P_.permute(0, 2, 1))  # (B, 3, N)

    # 转换为 (B, N, 3) 以匹配目标输出形状
    pts_rot = pts_rot.permute(0, 2, 1)  # (B, N, 

    # pts = torch.einsum('bij,nj->bin', R, P) + t[:, None, :]          # (B,N,3)
    pts = pts_rot + t[:, None, :]          # (B,N,3)
    return pts

# ---------- Kabsch（无尺度）----------

def _kabsch_single(P, Q):
    """
    单批次 Kabsch：最小二乘 R,t 使 R P + t ~ Q
    P, Q: (M,3) 对应点（M>=3 且非退化）
    return: R(3,3), t(3,)
    """
    Pc = P.mean(dim=0)                   # (3,)
    Qc = Q.mean(dim=0)                   # (3,)
    P0 = P - Pc                          # (M,3)
    Q0 = Q - Qc                          # (M,3)
    H = P0.t() @ Q0                      # (3,3)
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t() @ U.t()
    if torch.linalg.det(R) < 0:
        Vh = Vh.clone()
        Vh[:, -1] *= -1
        R = Vh.t() @ U.t()
    t = Qc - R @ Pc
    return R, t

def kabsch_from_corresp_torch(points_local, points_base_pred):
    """
    纯 Kabsch（无 RANSAC）
    points_local:     (N,3)
    points_base_pred: (B,N,3)
    return: poses(B,7), R(B,3,3), t(B,3)
    """
    P = torch.as_tensor(points_local)     # (N,3)
    Q = torch.as_tensor(points_base_pred) # (B,N,3)
    B, N, _ = Q.shape
    assert P.shape == (N,3), "points_local must be (N,3) and match Q's N"

    Pc = P.mean(dim=0)                                         # (3,)
    Qc = Q.mean(dim=1)                                         # (B,3)
    P0 = P - Pc                                                # (N,3)
    Q0 = Q - Qc[:, None, :]                                    # (B,N,3)

    H = torch.einsum('ni,bnj->bij', P0, Q0)                    # (B,3,3)
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.transpose(-2,-1) @ U.transpose(-2,-1)
    detR = torch.linalg.det(R)
    mask = detR < 0
    if mask.any():
        Vh_adj = Vh.clone()
        Vh_adj[mask, :, -1] *= -1
        R = Vh_adj.transpose(-2,-1) @ U.transpose(-2,-1)
    t = Qc - torch.einsum('bij,j->bi', R, Pc)                  # (B,3)
    q = R_to_quat_torch(R)                                     # (B,4)
    poses = torch.cat([t, q], dim=-1)                          # (B,7)
    return poses, R, t

# ---------- RANSAC + Kabsch ----------

def pose_estimate_from_correspondences_torch(
    points_local, points_base_pred,
    use_ransac: bool = False,
    ransac_iters: int = 1000,
    ransac_thresh: float = 0.01,
    min_inliers: int = 3,
    random_seed: int | None = None
):
    """
    估计位姿（刚体，无尺度）。支持 RANSAC。
    points_local:     (N,3) 末端系下原始点
    points_base_pred: (B,N,3) 这些点在基坐标系下的对应/预测
    use_ransac:       是否启用 RANSAC
    ransac_iters:     迭代次数
    ransac_thresh:    内点阈值（以欧氏残差计，和数据单位一致）
    min_inliers:      至少内点数（>=3）
    random_seed:      随机种子
    return:
        poses:        (B,7)  [t, qx,qy,qz,qw]
        inlier_masks: (B,N)  bool，若 use_ransac=False 则全 True
    """
    P_all = torch.as_tensor(points_local)      # (N,3)
    Q_all = torch.as_tensor(points_base_pred)  # (B,N,3)
    B, N, _ = Q_all.shape
    assert P_all.shape == (N,3), "points_local must be (N,3) and match Q's N"

    if not use_ransac:
        poses, _, _ = kabsch_from_corresp_torch(P_all, Q_all)
        inliers = torch.ones((B, N), dtype=torch.bool)
        return poses, inliers

    if random_seed is not None:
        g = torch.Generator()
        g.manual_seed(random_seed)
    else:
        g = None

    poses_out = torch.zeros((B,7))
    inlier_masks = torch.zeros((B,N), dtype=torch.bool)

    for b in range(B):
        P = P_all
        Q = Q_all[b]                                              # (N,3)

        if N < 3:
            # 退化：直接全体 Kabsch
            Rb, tb = _kabsch_single(P, Q)
            qb = R_to_quat_torch(Rb[None])[0]
            poses_out[b, :3] = tb
            poses_out[b, 3:] = qb
            inlier_masks[b] = True
            continue

        best_inliers = None
        best_count = -1
        best_R, best_t = None, None

        for _ in range(ransac_iters):
            # 随机取 3 个索引（最小集）
            idx = torch.randperm(N, generator=g)[:3]
            Ps = P[idx]                                          # (3,3)
            Qs = Q[idx]                                          # (3,3)

            # 检测退化：三点是否近乎共线
            v1 = Ps[1] - Ps[0]
            v2 = Ps[2] - Ps[0]
            area = torch.linalg.norm(torch.cross(v1, v2)) / 2.0
            if area < 1e-9:
                continue

            # 用三点估计临时 R,t
            R_tmp, t_tmp = _kabsch_single(Ps, Qs)

            # 计算全体残差并判内点
            resid = torch.linalg.norm((P @ R_tmp.t()) + t_tmp - Q, dim=-1)  # (N,)
            inliers = resid <= ransac_thresh
            num_inliers = int(inliers.sum().item())

            if num_inliers >= min_inliers and num_inliers > best_count:
                best_count = num_inliers
                best_inliers = inliers
                # 用当前内点重新拟合
                R_refit, t_refit = _kabsch_single(P[best_inliers], Q[best_inliers])
                best_R, best_t = R_refit, t_refit

        # 若没有找到足够内点，退化为全体 Kabsch
        if best_inliers is None or best_count < 3:
            Rb, tb = _kabsch_single(P, Q)
            inliers = torch.ones(N, dtype=torch.bool)
        else:
            Rb, tb = best_R, best_t
            inliers = best_inliers

        qb = R_to_quat_torch(Rb[None])[0]
        poses_out[b, :3] = tb
        poses_out[b, 3:] = qb
        inlier_masks[b] = inliers

    return poses_out, inlier_masks

# ---------- 误差度量 ----------

def rotation_geodesic_error_deg(R_est, R_gt):
    """
    R_est, R_gt: (B,3,3)
    return: (B,) 角度（度）
    angle = arccos( (trace(R_gt^T R_est)-1)/2 )
    """
    M = R_gt.transpose(-2,-1) @ R_est                    # (B,3,3)
    cos_theta = (M[:,0,0] + M[:,1,1] + M[:,2,2] - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    ang = torch.arccos(cos_theta) * (180.0 / torch.pi)
    return ang





def compute_pose_errors_torch(poses_est, poses_gt):
    """
    计算位姿误差（刚体、无尺度）：
      - 'rot_deg': 旋转测地线角（度）
      - 'trans_l2': 平移 L2 范数

    参数
    ----
    poses_est: (B,7)  [tx,ty,tz, qx,qy,qz,qw]
    poses_gt:  (B,7)  [tx,ty,tz, qx,qy,qz,qw]

    返回
    ----
    dict:
      'rot_deg':  (B,)
      'trans_l2': (B,)
    """
    poses_est = torch.as_tensor(poses_est)
    poses_gt  = torch.as_tensor(poses_gt)
    assert poses_est.shape == poses_gt.shape and poses_est.shape[-1] == 7, \
        "poses_est/poses_gt must be (B,7)"

    t_est, q_est = poses_est[:, :3], poses_est[:, 3:]
    t_gt,  q_gt  = poses_gt[:,  :3], poses_gt[:,  3:]

    # 归一化一次以防输入未归一
    q_est = q_est / (q_est.norm(dim=-1, keepdim=True) + 1e-12)
    q_gt  = q_gt  / (q_gt.norm(dim=-1, keepdim=True)  + 1e-12)

    R_est = quat_to_R_torch(q_est)        # (B,3,3)
    R_gt  = quat_to_R_torch(q_gt)         # (B,3,3)

    rot_err = rotation_geodesic_error_deg(R_est, R_gt)          # (B,)
    trans_err = torch.linalg.norm(t_est - t_gt, dim=-1)         # (B,)

    return {'rot_deg': rot_err, 'trans_l2': trans_err}

if __name__ == "__main__":
    pass