from frankapy import FrankaArm
from autolab_core import RigidTransform
import numpy as np

import time
from real_camera_utils_lpy import Camera
import cv2
import os
import json
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

def quat2rotm(quat):
    """Quaternion to rotation matrix.
    quat:w,x,y,z
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    s = w * w + x * x + y * y + z * z
    rotm = np.array(
        [
            [
                1 - 2 * (y * y + z * z) / s,
                2 * (x * y - z * w) / s,
                2 * (x * z + y * w) / s,
            ],
            [
                2 * (x * y + z * w) / s,
                1 - 2 * (x * x + z * z) / s,
                2 * (y * z - x * w) / s,
            ],
            [
                2 * (x * z - y * w) / s,
                2 * (y * z + x * w) / s,
                1 - 2 * (x * x + y * y) / s,
            ],
        ]
    )
    return rotm


def extract_poses_from_file(file_path):
    """
    Reads the JSON content from the specified file and extracts
    all 'pose' arrays from the 'parameterized_poses' field.
    """
    # Read the file and parse the JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Navigate to "parameterized_poses"
    parameterized_poses = data.get("parameter", {}).get("parameterized_poses", [])
    
    # Collect the "pose" array from each item if present
    poses = []
    for item in parameterized_poses:
        pose_data = item.get("pose_with_joint_angles", {}).get("pose", [])
        poses.append(pose_data)
    
    return np.array(poses)

def gripper_open_flag(fa,gripper_thres):
    gripper_width=fa.get_gripper_width()
    return gripper_width > gripper_thres


def osc_move(fa,target_pose,gripper_thres,recording=False, recording_continous_frames=False, cameras=None):


    target_pos, target_quat, target_gripper = target_pose

    target_pose_fa = RigidTransform(from_frame="franka_tool")
    target_pose_fa.rotation =  quat2rotm(target_quat)  # 注意需要为（w,x,y,z)格式
    target_pose_fa.translation = target_pos


    current_gripper =gripper_open_flag(fa,gripper_thres)# 1 表示开  0 表示关
    last_capture_time = time.time()
    
    fa.goto_pose(target_pose_fa, use_impedance=False) 
    # time.sleep(1)
    if current_gripper != target_gripper[0]:
        
        if target_gripper[0]==1:
            fa.open_gripper()
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)

    pose = fa.get_pose()
    current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
    print("Current pos:", current_pos, "Target pos:", target_pos)
    print("Current Gripper State: ", current_gripper, "Target Gripper State: ", target_gripper[0])

    while np.max(np.abs(current_pos-target_pos)) > 0.02:
        print("The error is too big,retrying!!")
        fa.goto_pose(target_pose_fa, use_impedance=False) 
        pose = fa.get_pose()
        current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
        print("Current pos:", current_pos, "Target pos:", target_pos)     
    while current_gripper != target_gripper[0]:
        if target_gripper[0]==1:
            fa.open_gripper()   
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)
    if recording and not recording_continous_frames:
        result_dict = cameras.capture()
        result_dict['action'] = deepcopy(np.concatenate([target_pos, target_quat, \
                                                        target_gripper]))
        return result_dict



def osc_move_eval(fa,target_pose,gripper_thres,recording=False, recording_continous_frames=False, cameras=None):


    target_pos, target_quat, target_gripper = target_pose

    target_pose_fa = RigidTransform(from_frame="franka_tool")
    target_pose_fa.rotation =  quat2rotm(target_quat)  # 注意需要为（w,x,y,z)格式
    target_pose_fa.translation = target_pos


    current_gripper =gripper_open_flag(fa,gripper_thres)# 1 表示开  0 表示关
    last_capture_time = time.time()
    
    # fa.goto_pose(target_pose_fa, use_impedance=True,duration=6) 
    fa.goto_pose(target_pose_fa, use_impedance=False,duration=5) 
    time.sleep(1)
    if current_gripper != target_gripper[0]:
        
        if target_gripper[0]==1:
            fa.open_gripper()
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)

    pose = fa.get_pose()
    current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
    print("Current pos:", current_pos, "Target pos:", target_pos)
    print("Current Gripper State: ", current_gripper, "Target Gripper State: ", target_gripper[0])

    while np.max(np.abs(current_pos-target_pos)) > 0.02:
        print("The error is too big,retrying!!")
        fa.goto_pose(target_pose_fa, use_impedance=False) 
        pose = fa.get_pose()
        current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
        print("Current pos:", current_pos, "Target pos:", target_pos)     
    while current_gripper != target_gripper[0]:
        if target_gripper[0]==1:
            fa.open_gripper()   
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)
    if recording and not recording_continous_frames:
        result_dict = cameras.capture()
        result_dict['action'] = deepcopy(np.concatenate([target_pos, target_quat, \
                                                        target_gripper]))
        return result_dict




def get_cam_extrinsic(type):
    """获取相机外参矩阵"""
    if type == "3rd_1":
        trans = np.array([1.0500235453581752, 0.018840177419218063, 0.8526189719141685])
        quat = np.array([-0.650217823736857, -0.6493193457487534, 0.2805349490016194, 0.2773108568375482])  # x y z w
    elif type == "3rd_2":
        trans = np.array([0.41319647185014996, 0.671280823421573, 0.31847372597347334])
        quat = np.array([-0.004178189428432915, 0.8100601972017345, -0.5859203348840233, 0.0028462131169476428])  # x y z w
    elif type == "3rd_3":
        trans = np.array([1.0195456272363708, -0.09614207876725467, 0.579374291081325])
        quat = np.array([-0.6486533850105296, -0.6257763784047043, 0.299967106317545, 0.3125259116818196])  # x y z w
    else:
        raise ValueError("Invalid type")

    transform = np.eye(4)
    rot = R.from_quat(quat)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = trans.T

    return transform


if __name__ == "__main__":

    # # reset_robot
    cameras = Camera(camera_type="all")
    time.sleep(2)

    # ======================== 配置区域 ========================
    task_name = 'put_red_bull_in_pink_plate'    # Change 1: 任务名称
    instruction = "put the red bull in the pink plate"  # Change 2: 指令
    task_idx = 8   # Change 3: trail 索引
    gripper_open = np.array([True,True,False,False,False,True])  # Change 4: True 开 False 关

    # 保存目录
    data_result_dir = "/media/casia/data4/lpy/3zed_data/raw_data_keypoint"
    # ==========================================================

    # 使用 trail_{idx} 格式
    save_dir = os.path.join(data_result_dir, task_name, f"trail_{task_idx}")
    os.makedirs(save_dir, exist_ok=False)

    expert_action_file = f"/home/casia/Downloads/task (8c).task"  # Change 5: 对应到具体的下载文件 
    action_poses = extract_poses_from_file(expert_action_file)

    assert gripper_open.shape[0] == action_poses.shape[0]
    fa = FrankaArm()
    fa.reset_joints()

    # 创建目录结构（匹配目标格式）
    dir_keys = [
        "3rd_1_bgr", "3rd_1_bgr_images", "3rd_1_depth", "3rd_1_pcd",
        "3rd_2_bgr", "3rd_2_bgr_images", "3rd_2_depth", "3rd_2_pcd",
        "3rd_3_bgr", "3rd_3_bgr_images", "3rd_3_depth", "3rd_3_pcd",
        "poses", "joints", "gripper_states"
    ]

    for key in dir_keys:
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)

    # 保存外参
    extrinsics = {
        "3rd_1": get_cam_extrinsic("3rd_1"),
        "3rd_2": get_cam_extrinsic("3rd_2"),
        "3rd_3": get_cam_extrinsic("3rd_3")
    }
    with open(os.path.join(save_dir, "extrinsics.pkl"), 'wb') as f:
        pickle.dump(extrinsics, f)

    global_idx = 0  # for recording frames
    for idx, action_pose in enumerate(action_poses):

        action_pose = action_pose.reshape(4, 4).T  # 转置

        action_pose = deepcopy(RigidTransform(
            rotation=action_pose[:3, :3],
            translation=action_pose[:3, 3]
        ))
        target_pos, target_quat = action_pose.translation, action_pose.quaternion  # 使用frankapy的transform

        target_gripper = gripper_open[idx:idx+1]
        now_obs = osc_move(fa, (target_pos, target_quat, target_gripper), gripper_thres=0.07, recording=True, recording_continous_frames=False, cameras=cameras)

        # 获取机器人状态
        pose = fa.get_pose()
        joints = fa.get_joints()
        gripper_width = fa.get_gripper_width()
        gripper_state = gripper_width > 0.07  # 与 gripper_thres 一致

        # 格式化文件名（6位补零）
        file_idx = f"{global_idx:06d}"

        # 保存 pose: [x, y, z, qw, qx, qy, qz] (7维)
        current_pos = pose.translation
        current_quat = pose.quaternion  # frankapy quaternion: [w, x, y, z]
        # 保持 [x, y, z, qw, qx, qy, qz] 格式
        pose_data = np.array([
            current_pos[0], current_pos[1], current_pos[2],
            current_quat[0], current_quat[1], current_quat[2], current_quat[3]
        ], dtype=np.float32)
        with open(os.path.join(save_dir, f"poses/{file_idx}.pkl"), 'wb') as f:
            pickle.dump(pose_data, f)

        # 保存 joints (7维关节角度)
        joints_data = np.array(joints, dtype=np.float32)
        with open(os.path.join(save_dir, f"joints/{file_idx}.pkl"), 'wb') as f:
            pickle.dump(joints_data, f)

        # 保存 gripper_states (布尔值, False表示闭合)
        with open(os.path.join(save_dir, f"gripper_states/{file_idx}.pkl"), 'wb') as f:
            pickle.dump(gripper_state, f)

        # 移除 action 键（如果存在）
        now_obs.pop("action", None)

        # 保存相机数据
        for cam_type, cam_values in now_obs.items():
            # cam_type 是 "3rd_1", "3rd_2", "3rd_3"

            # 保存 bgr (rgb数据)
            with open(os.path.join(save_dir, f"{cam_type}_bgr/{file_idx}.pkl"), 'wb') as f:
                pickle.dump(cam_values["rgb"], f)

            # 保存 bgr_images (png格式)
            cv2.imwrite(os.path.join(save_dir, f"{cam_type}_bgr_images/{file_idx}.png"), cam_values["rgb"])

            # 保存 depth
            with open(os.path.join(save_dir, f"{cam_type}_depth/{file_idx}.pkl"), 'wb') as f:
                pickle.dump(cam_values["depth"], f)

            # 保存 pcd
            with open(os.path.join(save_dir, f"{cam_type}_pcd/{file_idx}.pkl"), 'wb') as f:
                pickle.dump(cam_values["pcd"], f)

        global_idx += 1

    # 保存 instruction.txt（文本格式）
    with open(os.path.join(save_dir, "instruction.txt"), 'w') as f:
        f.write(instruction + "\n")

    cameras.stop()
    print(f"Data collection completed! Saved to: {save_dir}")

    # 复位机械臂到初始位置
    print("Resetting robot to initial position...")
    fa.reset_joints()
    fa.open_gripper()
    print("Robot reset completed!")